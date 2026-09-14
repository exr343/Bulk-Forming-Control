"""PyTorch port of the vendored reference's Encode-Process-Decode
MeshGraphNets architecture (core_model.py), specialized for the Agility
Forge forging surrogate.

See GNN/README.md for the full per-node/per-edge feature spec, the settled
architecture sizing (latent=128, num_layers=2, message_passing_steps=15,
kept at reference defaults -- see "Network shape (encoder -> processor ->
decoder)"), and the normalization scheme ("Normalization" section: online
accumulator, per-axis scaling, independent input/output normalizers, flags
and control left raw).

Unlike the reference (which batches disjoint graphs of varying size/topology
via segment-sum over a flattened node axis, since its tasks' meshes differ
across examples), every example here shares the exact same fixed surface
mesh (GNN/README.md's "Mesh topology" verification) -- so batching is just a
dense leading batch dimension (B, N, ...) / (B, E, ...), with `senders`/
`receivers` shared (unbatched) index tensors. This is a deliberate
simplification, not a partial port: it is strictly simpler and exactly
equivalent given fixed topology, so there is no reason to carry over the
reference's more general (and more complex) disjoint-graph batching here.
"""

import torch
import torch.nn as nn

from applications.Agility_Forge.GNN.data import SurfaceMeshInfo
from applications.Agility_Forge.GNN.normalizer import OnlineNormalizer

# Node feature column layout, fixed by GNN/README.md's per-node feature table
# and produced in this order by data.py's ForgeGNNDataset.
N_FLAG_DIMS = 3  # is_bottom_face, is_pin_node, is_lateral_wall
N_PHYSICAL_DIMS = 4  # signed distance to band (1) + x_k (3)
N_CONTROL_DIMS = 4  # d_j_frac, sin(R_j), cos(R_j), u_j_frac
NODE_IN_DIM = N_FLAG_DIMS + N_PHYSICAL_DIMS + N_CONTROL_DIMS  # 11
EDGE_IN_DIM = 8
OUTPUT_DIM = 3  # predicted displacement delta


class MLP(nn.Module):
    """`Linear -> activation` repeated, no activation after the last Linear
    (matches `snt.nets.MLP(widths, activate_final=False)`), optionally
    followed by LayerNorm on the final output (matches the reference's
    `_make_mlp`'s `layer_norm` flag)."""

    def __init__(self, in_dim: int, widths, layer_norm: bool = True):
        super().__init__()
        layers = []
        prev = in_dim
        for i, w in enumerate(widths):
            layers.append(nn.Linear(prev, w))
            if i < len(widths) - 1:
                layers.append(nn.ReLU())
            prev = w
        self.net = nn.Sequential(*layers)
        self.layer_norm = nn.LayerNorm(widths[-1]) if layer_norm else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        return x


class GraphNetBlock(nn.Module):
    """One message-passing step: edge update from [sender, receiver, edge]
    latents, then node update from [node, summed-incoming-edge] latents,
    both with a residual add -- matches core_model.py's `GraphNetBlock`.
    Single edge set (mesh edges only, per GNN/README.md's "Graph structure"
    decision), so unlike the reference's per-edge-set loop, this has exactly
    one edge_fn/node_fn pair."""

    def __init__(self, latent_size: int, num_layers: int):
        super().__init__()
        self.edge_fn = MLP(3 * latent_size, [latent_size] * num_layers + [latent_size])
        self.node_fn = MLP(2 * latent_size, [latent_size] * num_layers + [latent_size])

    def forward(self, node_latents, edge_latents, senders, receivers):
        # node_latents: (B, N, L), edge_latents: (B, E, L)
        sender_feats = node_latents[:, senders, :]
        receiver_feats = node_latents[:, receivers, :]
        edge_input = torch.cat([sender_feats, receiver_feats, edge_latents], dim=-1)
        new_edge_latents = self.edge_fn(edge_input)

        B, N, L = node_latents.shape
        agg = node_latents.new_zeros(B, N, L)
        agg.index_add_(1, receivers, new_edge_latents)

        node_input = torch.cat([node_latents, agg], dim=-1)
        new_node_latents = self.node_fn(node_input)

        return new_node_latents + node_latents, new_edge_latents + edge_latents


class EncodeProcessDecode(nn.Module):
    """Encoder (node + edge MLPs) -> `message_passing_steps` independently-
    weighted `GraphNetBlock`s -> decoder MLP (no LayerNorm). Matches
    core_model.py's `EncodeProcessDecode` exactly in structure."""

    def __init__(self, node_in_dim, edge_in_dim, output_dim,
                 latent_size=128, num_layers=2, message_passing_steps=15):
        super().__init__()
        self.node_encoder = MLP(node_in_dim, [latent_size] * num_layers + [latent_size])
        self.edge_encoder = MLP(edge_in_dim, [latent_size] * num_layers + [latent_size])
        self.blocks = nn.ModuleList([
            GraphNetBlock(latent_size, num_layers) for _ in range(message_passing_steps)
        ])
        self.decoder = MLP(latent_size, [latent_size] * num_layers + [output_dim], layer_norm=False)

    def forward(self, node_features, edge_features, senders, receivers):
        node_latents = self.node_encoder(node_features)
        edge_latents = self.edge_encoder(edge_features)
        for block in self.blocks:
            node_latents, edge_latents = block(node_latents, edge_latents, senders, receivers)
        return self.decoder(node_latents)


class ForgeGNN(nn.Module):
    """The Agility-Forge-specific wrapper (analogous to the reference's
    `cfd_model.Model`/`cloth_model.Model`): owns the fixed mesh buffers, the
    three online normalizers, edge-feature construction from `x_k`, and the
    integrate-forward step (`x_{k+1} = x_k + decoded delta`)."""

    def __init__(self, mesh_info: SurfaceMeshInfo, latent_size=128, num_layers=2,
                 message_passing_steps=15):
        super().__init__()
        self.register_buffer("rest_pos", mesh_info.rest_pos)
        self.register_buffer("senders", mesh_info.senders)
        self.register_buffer("receivers", mesh_info.receivers)

        self.node_physical_normalizer = OnlineNormalizer(size=N_PHYSICAL_DIMS)
        self.edge_normalizer = OnlineNormalizer(size=EDGE_IN_DIM)
        self.output_normalizer = OnlineNormalizer(size=OUTPUT_DIM)

        self.core = EncodeProcessDecode(
            NODE_IN_DIM, EDGE_IN_DIM, OUTPUT_DIM,
            latent_size=latent_size, num_layers=num_layers,
            message_passing_steps=message_passing_steps,
        )

    def _edge_features(self, x_k: torch.Tensor, accumulate: bool) -> torch.Tensor:
        """x_k: (B, N, 3) -> (B, E, 8) normalized edge features."""
        s, r = self.senders, self.receivers
        rel_mesh = (self.rest_pos[s] - self.rest_pos[r]).unsqueeze(0).expand(x_k.shape[0], -1, -1)
        world = self.rest_pos.unsqueeze(0) + x_k  # (B, N, 3)
        rel_world = world[:, s, :] - world[:, r, :]  # (B, E, 3)
        raw = torch.cat([
            rel_mesh, rel_mesh.norm(dim=-1, keepdim=True),
            rel_world, rel_world.norm(dim=-1, keepdim=True),
        ], dim=-1)
        return self.edge_normalizer(raw, accumulate=accumulate)

    def _node_features(self, node_features_raw: torch.Tensor, accumulate: bool) -> torch.Tensor:
        """node_features_raw: (B, N, 11) in data.py's fixed column order
        [flags(3), dist_to_band(1), x_k(3), control(4)] -> normalized-where-
        appropriate (B, N, 11) ready for the encoder."""
        flags = node_features_raw[..., 0:N_FLAG_DIMS]
        physical = node_features_raw[..., N_FLAG_DIMS:N_FLAG_DIMS + N_PHYSICAL_DIMS]
        control = node_features_raw[..., N_FLAG_DIMS + N_PHYSICAL_DIMS:]
        physical_n = self.node_physical_normalizer(physical, accumulate=accumulate)
        return torch.cat([flags, physical_n, control], dim=-1)

    def predict_delta(self, node_features_raw: torch.Tensor, x_k: torch.Tensor,
                       accumulate: bool = True) -> torch.Tensor:
        """Returns the predicted (denormalized, physical-units) displacement
        delta for this hit -- add onto `x_k` for the next state."""
        node_features = self._node_features(node_features_raw, accumulate)
        edge_features = self._edge_features(x_k, accumulate)
        pred_delta_n = self.core(node_features, edge_features, self.senders, self.receivers)
        return self.output_normalizer.inverse(pred_delta_n)

    def loss(self, node_features_raw: torch.Tensor, x_k: torch.Tensor,
             target_delta: torch.Tensor) -> torch.Tensor:
        """Per-node MSE in normalized-target space -- exactly
        `sum_c (pred_delta_norm - target_delta_norm)_c^2` averaged over every
        node and every example in the batch, where `pred_delta_norm` is the
        raw decoder output (no denormalization) and `target_delta_norm` is
        the true displacement delta passed through the *same* output
        normalizer. This is the actual quantity backpropagated through the
        network -- comparing in normalized space (not the denormalized
        prediction to the raw physical target) keeps the loss on the scale
        the network's LayerNorms already operate at, matching the reference
        MeshGraphNets convention. See GNN/README.md's "Loss function" note
        for why this is *not* the same quantity as the NRMSE train.py plots
        (a physical-units diagnostic, not the optimized objective).
        No node is masked out of this loss -- see GNN/README.md's resolved
        "Loss masking" decision."""
        loss, _ = self.loss_and_predict(node_features_raw, x_k, target_delta)
        return loss

    def loss_and_predict(self, node_features_raw: torch.Tensor, x_k: torch.Tensor,
                          target_delta: torch.Tensor):
        """Same computation as `loss()`, but also returns the denormalized
        (physical-units) prediction from the same forward pass -- avoids
        running the encoder/processor/decoder twice when a caller (train.py's
        NRMSE tracking) needs both the training loss and a physical-space
        prediction for the same inputs."""
        accumulate = self.training
        node_features = self._node_features(node_features_raw, accumulate)
        edge_features = self._edge_features(x_k, accumulate)
        pred_delta_n = self.core(node_features, edge_features, self.senders, self.receivers)
        target_n = self.output_normalizer(target_delta, accumulate=accumulate)
        loss = ((pred_delta_n - target_n) ** 2).sum(dim=-1).mean()
        pred_delta_phys = self.output_normalizer.inverse(pred_delta_n)
        return loss, pred_delta_phys
