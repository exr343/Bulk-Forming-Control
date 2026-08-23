"""Koopman Autoencoder for the Agility_Forge forging state.

Learns a latent space in which the control-driven forging dynamics are
exactly linear time-invariant (LTI):

    z_{k+1} = A z_k + B u_k

The full pipeline is state -> POD -> lift -> dynamics -> unlift -> POD^-1:
  encode: x (normalized, n_x-dim) -> POD-project onto a *fixed*, precomputed
          r-dim basis (pod_mean/pod_modes, fit on training data only --
          see koopman/dataset.py's compute_pod_basis) -> a (r-dim) ->
          learned nonlinear lift Psi(a) (the ``Lifting`` network) -> z
          (n_z-dim, n_z > r -- Koopman theory lifts to a *higher*-dimensional
          space where the dynamics become linear, so expanding here is
          intentional, not an oversight).
  decode: z -> learned linear map (n_z -> r) -> a_hat -> *fixed* POD
          reconstruction (pod_mean + a_hat @ pod_modes) -> x_hat (n_x-dim).

Unlike the LRAN_LD reference this is modeled on, x itself is never
concatenated into z, so there is no fixed identity/state block to lean on.
All nonlinearity lives in the lifting network; A, B, and the decoder's
learned half stay linear -- closer to a classical EDMD/Koopman-with-a-
learned-dictionary formulation than to a generic nonlinear autoencoder. The
POD projection/reconstruction is not learned at all -- it's fixed once at
dataset-build time from the training rollouts, the same way x_scale is.
"""

import torch
import torch.nn as nn


def _svd_init(n, scale):
    W = torch.randn(n, n)
    U, _, Vh = torch.linalg.svd(W)
    return (U @ Vh) * scale


class ResidualBlock(nn.Module):
    """`n_layers` stacked [LayerNorm -> Linear(width, width) -> activation]
    layers wrapped in a single residual connection: out = in + F(in).
    Operates at one fixed width throughout -- a block never changes
    dimension internally.

    Pre-norm (LayerNorm before each Linear, not after): 16 layers at
    width>=2500 with no normalization at all was numerically unstable in
    practice (loss grew ~10x/epoch within 3 epochs on this dataset) --
    LayerNorm keeps each layer's input to a fixed scale regardless of how
    the residual stream has grown, which is what stops that blow-up.
    """

    def __init__(self, width, n_layers, act_cls):
        super().__init__()
        layers = []
        for _ in range(n_layers):
            layers.extend([nn.LayerNorm(width), nn.Linear(width, width), act_cls()])
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.net(x)


class Lifting(nn.Module):
    """Learned observables Psi(a): R^n_in -> R^n_z (n_in is whatever feeds
    this network -- the POD-reduced coefficient vector, in this model, not
    the raw state).

    a -> [Linear(n_in, block_widths[0]) + act]
      -> ResidualBlock(block_widths[0], layers_per_block)
      -> for each subsequent width in block_widths:
           if width != prev_width: [Linear(prev_width, width) + act]
           (dimension-changing transition between blocks -- the residual
           connection only works *within* a block, where width is constant,
           so a plain projection bridges blocks of different widths; skipped
           entirely, not even an identity layer counted, when consecutive
           blocks share a width -- e.g. 4 blocks all at width 250 chain
           directly, giving exactly n_blocks * layers_per_block hidden
           layers, no extras)
           -> ResidualBlock(width, layers_per_block)
      -> [Linear(block_widths[-1], n_z)]
    """

    def __init__(self, n_in, n_z, block_widths, layers_per_block, act):
        super().__init__()
        block_widths = list(block_widths)

        activations = {
            "Tanh": nn.Tanh,
            "ReLU": nn.ReLU,
            "LeakyReLU": nn.LeakyReLU,
            "Sigmoid": nn.Sigmoid,
        }
        if act not in activations:
            raise ValueError(f"Unknown activation {act!r}; choose from {list(activations)}")
        act_cls = activations[act]

        self.in_proj = nn.Linear(n_in, block_widths[0])
        self.in_act = act_cls()

        self.blocks = nn.ModuleList([ResidualBlock(block_widths[0], layers_per_block, act_cls)])
        self.transitions = nn.ModuleList()
        for prev_width, width in zip(block_widths[:-1], block_widths[1:]):
            if width != prev_width:
                self.transitions.append(nn.Sequential(nn.Linear(prev_width, width), act_cls()))
            else:
                self.transitions.append(nn.Identity())
            self.blocks.append(ResidualBlock(width, layers_per_block, act_cls))

        self.out_proj = nn.Linear(block_widths[-1], n_z)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.in_act(self.in_proj(x))
        x = self.blocks[0](x)
        for transition, block in zip(self.transitions, self.blocks[1:]):
            x = block(transition(x))
        return self.out_proj(x)


class KoopmanAutoencoder(nn.Module):
    """
    Koopman autoencoder over a fixed POD-reduced representation.

    POD (fixed):   a = (x - pod_mean) @ pod_modes.T        (n_x -> r)
    Lift (learned): z = Psi(a)                              (r -> n_z, n_z > r)
    Dynamics:      z_{k+1} = A z_k + B u_k
    Unlift (learned): a_hat = W z + b                       (n_z -> r)
    POD^-1 (fixed): x_hat = pod_mean + a_hat @ pod_modes     (r -> n_x)

    pod_mean/pod_modes are registered buffers (not parameters -- fixed,
    precomputed from training data by koopman/dataset.py's
    compute_pod_basis, not learned by gradient descent). They move with
    .to(device) and are saved/restored via state_dict() like any buffer,
    but never receive gradients.
    """

    def __init__(self, n_x, n_u, n_z, pod_mean, pod_modes, block_widths, layers_per_block,
                 act, init_scale=0.99):
        super().__init__()
        self.n_x = n_x
        self.n_z = n_z
        self.r = pod_modes.shape[0]

        self.register_buffer("pod_mean", torch.as_tensor(pod_mean, dtype=torch.float32))
        self.register_buffer("pod_modes", torch.as_tensor(pod_modes, dtype=torch.float32))

        self.lifting = Lifting(self.r, n_z, block_widths, layers_per_block, act)
        self.decoder = nn.Linear(n_z, self.r)

        self.A = nn.Linear(n_z, n_z, bias=False)
        self.A.weight.data = _svd_init(n_z, init_scale)

        self.B = nn.Linear(n_u, n_z, bias=False)
        nn.init.xavier_uniform_(self.B.weight)

    def encode(self, x):
        """POD-project x onto the fixed basis, then lift: z = Psi(POD(x))."""
        a = (x - self.pod_mean) @ self.pod_modes.T
        return self.lifting(a)

    def decode(self, z):
        """Learned linear unlift to POD coefficients, then fixed POD
        reconstruction back to the full state."""
        a_hat = self.decoder(z)
        return self.pod_mean + a_hat @ self.pod_modes

    def rollout(self, z0, us):
        """
        Propagate latent state forward under control inputs.

        z0 : (batch, n_z)    initial latent state
        us : (batch, K, n_u) control sequence
        Returns list of K predicted latent states.
        """
        z = z0
        z_preds = []
        for k in range(us.shape[1]):
            z = self.A(z) + self.B(us[:, k])
            z_preds.append(z)
        return z_preds
