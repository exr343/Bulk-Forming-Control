"""Koopman Autoencoder for the Agility_Forge forging state.

Learns a latent space in which the control-driven forging dynamics are
exactly linear time-invariant (LTI):

    z_{k+1} = A z_k + B u_k

z is a *purely learned* observable, z = Psi(x) (the ``Lifting`` network) --
unlike the LRAN_LD reference this is modeled on, x itself is never
concatenated into z, so there is no fixed identity/state block to lean on.
Because of that, the decoder cannot be exact by construction: it is a single
learned linear map, x_hat = W z + b. All nonlinearity in the model lives in
the lifting network; A, B, and the decoder stay linear, which is closer to a
classical EDMD/Koopman-with-a-learned-dictionary formulation than to a
generic nonlinear autoencoder.
"""

import torch
import torch.nn as nn


def _svd_init(n, scale):
    W = torch.randn(n, n)
    U, _, Vh = torch.linalg.svd(W)
    return (U @ Vh) * scale


class Lifting(nn.Module):
    """Learned observables Psi(x): R^n_x -> R^n_z."""

    def __init__(self, n_x, n_z, n_h, act, alpha):
        super().__init__()
        width = 16 * alpha

        activations = {
            "Tanh": nn.Tanh,
            "ReLU": nn.ReLU,
            "LeakyReLU": nn.LeakyReLU,
            "Sigmoid": nn.Sigmoid,
        }
        if act not in activations:
            raise ValueError(f"Unknown activation {act!r}; choose from {list(activations)}")
        self.act = activations[act]()

        layers = [nn.Linear(n_x, width), self.act]
        for _ in range(n_h - 1):
            layers.extend([nn.Linear(width, width), self.act])
        layers.append(nn.Linear(width, n_z))
        self.net = nn.Sequential(*layers)

        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.net(x)


class KoopmanAutoencoder(nn.Module):
    """
    Koopman autoencoder with a purely-learned latent (no x-in-latent) and a
    learned linear decoder.

    Latent state:  z = Psi(x)          (dimension n_z, the full latent space)
    Decoder:       x_hat = W z + b     (learned linear map, not exact)
    Dynamics:      z_{k+1} = A z_k + B u_k
    """

    def __init__(self, n_x, n_u, n_z, n_h, act, alpha, init_scale=0.99):
        super().__init__()
        self.n_x = n_x
        self.n_z = n_z

        self.lifting = Lifting(n_x, n_z, n_h, act, alpha)
        self.decoder = nn.Linear(n_z, n_x)

        self.A = nn.Linear(n_z, n_z, bias=False)
        self.A.weight.data = _svd_init(n_z, init_scale)

        self.B = nn.Linear(n_u, n_z, bias=False)
        nn.init.xavier_uniform_(self.B.weight)

    def encode(self, x):
        """Lift x to z = Psi(x)."""
        return self.lifting(x)

    def decode(self, z):
        """Learned linear decoder."""
        return self.decoder(z)

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
