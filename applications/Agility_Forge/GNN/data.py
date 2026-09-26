"""PyTorch data adapter for the Agility Forge MeshGraphNets surrogate.

Reuses `generate_dataset.py`'s manifest.json + per-hit .vtu files directly.
See GNN/README.md for the full design rationale (surface-only mesh, the
10-dim node / 8-dim edge feature definitions, why each feature is included,
and the by-rollout prefix train/test split) -- this module implements that
spec, it does not re-derive it.

Node/edge features here are RAW (unnormalized); the three online normalizers
(node-physical-subset, edge, output) live in model.py, matching the vendored
reference's own split between data loading and normalization.
"""

import json
import os
from dataclasses import dataclass

import meshio
import numpy as np
import torch


def _extract_surface_faces(points: np.ndarray, tets: np.ndarray):
    """Boundary triangles of a tet mesh: faces belonging to exactly one tet.

    Same convention as mesh_container.py's `MeshContainer.extract_faces()`
    (winding kept, though winding doesn't matter for our use -- we only need
    the node set and edges, not outward normals).
    """
    faces = np.vstack([
        tets[:, [1, 3, 2]], tets[:, [0, 2, 3]], tets[:, [0, 1, 2]], tets[:, [0, 3, 1]],
    ])
    sorted_faces = np.sort(faces, axis=1)
    keys = [tuple(f) for f in sorted_faces]
    counts = {}
    for k in keys:
        counts[k] = counts.get(k, 0) + 1
    mask = np.array([counts[k] == 1 for k in keys])
    return faces[mask]


def _triangles_to_edges(faces_local: np.ndarray):
    """Bidirectional, de-duplicated edges from a locally-indexed triangle array.

    NumPy port of common.py's `triangles_to_edges` -- the vendored reference
    version is written against TF ops (tf.unique/tf.bitcast) and operates on
    surface triangles, which matches our surface-only graph directly (see
    GNN/README.md's "Graph construction" section for why no tet-specific
    edge extractor is needed).
    """
    edges = np.vstack([faces_local[:, [0, 1]], faces_local[:, [1, 2]], faces_local[:, [2, 0]]])
    canon = np.sort(edges, axis=1)
    uniq = np.unique(canon, axis=0)
    senders = np.concatenate([uniq[:, 0], uniq[:, 1]])
    receivers = np.concatenate([uniq[:, 1], uniq[:, 0]])
    return senders.astype(np.int64), receivers.astype(np.int64)


@dataclass
class SurfaceMeshInfo:
    """Fixed graph structure + static per-node flags, computed once.

    Valid for every rollout/hit in the dataset -- mesh topology is identical
    across all of them (verified in GNN/README.md's "Mesh topology" section:
    same node/cell counts across 338 rollouts spanning 12 separate
    generate_dataset.py invocations).
    """

    rest_pos: torch.Tensor  # (n_surface, 3) float32, rest/undeformed positions (mm)
    is_bottom_face: torch.Tensor  # (n_surface, 1) float32, 0/1
    is_pin_node: torch.Tensor  # (n_surface, 1) float32, 0/1
    is_lateral_wall: torch.Tensor  # (n_surface, 1) float32, 0/1
    senders: torch.Tensor  # (n_edges,) int64, indices into the surface node set
    receivers: torch.Tensor  # (n_edges,) int64
    surf_faces_local: np.ndarray  # (n_surf_tris, 3) int, for writing eval .vtu files
    full_to_surface: np.ndarray  # (n_surface,) int64, indices into the full volumetric node array


def build_surface_mesh_info(reference_vtu_path: str) -> SurfaceMeshInfo:
    """Extracts the surface mesh and the static is_bottom_face/is_pin_node/
    is_lateral_wall flags from one rollout's undeformed.vtu (any rollout's
    works identically, since the mesh is shared)."""
    mesh = meshio.read(reference_vtu_path)
    pts = mesh.points.astype(np.float64)
    tets = mesh.cells_dict["tetra"]

    surf_faces_full = _extract_surface_faces(pts, tets)
    surf_node_ids = np.unique(surf_faces_full)

    remap = -np.ones(pts.shape[0], dtype=np.int64)
    remap[surf_node_ids] = np.arange(len(surf_node_ids))
    surf_faces_local = remap[surf_faces_full]

    senders, receivers = _triangles_to_edges(surf_faces_local)
    surf_pts = pts[surf_node_ids]

    # is_bottom_face: matches boundary_conditions.py's `bottom_face` exactly
    # (x == whole-billet min x, the axial end used as the symmetry-style BC).
    x_bottom = pts[:, 0].min()
    is_bottom_face = np.isclose(surf_pts[:, 0], x_bottom, atol=1e-5)

    # is_pin_node: matches boundary_conditions.py's `pin_node` selection
    # exactly (nearest mesh point to (x_bottom, y_min, 0)).
    y_min_initial = pts[:, 1].min()
    pin_id_full = int(np.argmin(
        (pts[:, 0] - x_bottom) ** 2 + (pts[:, 1] - y_min_initial) ** 2 + (pts[:, 2] - 0.0) ** 2
    ))
    pin_local = remap[pin_id_full]
    is_pin_node = np.zeros(len(surf_node_ids), dtype=bool)
    if pin_local >= 0:
        is_pin_node[pin_local] = True
    else:
        raise RuntimeError(
            "pin_node (per boundary_conditions.py's selection) is not on the "
            "extracted exterior surface -- this should never happen (it is "
            "chosen as the nearest point to a corner of the bounding box) "
            "and indicates the surface extraction disagrees with the FEM "
            "pipeline's own node selection."
        )

    # is_lateral_wall: matches boundary_conditions.py's `on_outer_surface_np`
    # exactly (radius within tol_r of the max observed radius R0).
    radii = np.sqrt(surf_pts[:, 1] ** 2 + surf_pts[:, 2] ** 2)
    R0 = radii.max()
    tol_r = 0.25 * (2 * R0 / 5)
    is_lateral_wall = np.abs(radii - R0) <= tol_r

    return SurfaceMeshInfo(
        rest_pos=torch.tensor(surf_pts, dtype=torch.float32),
        is_bottom_face=torch.tensor(is_bottom_face, dtype=torch.float32).unsqueeze(-1),
        is_pin_node=torch.tensor(is_pin_node, dtype=torch.float32).unsqueeze(-1),
        is_lateral_wall=torch.tensor(is_lateral_wall, dtype=torch.float32).unsqueeze(-1),
        senders=torch.tensor(senders, dtype=torch.long),
        receivers=torch.tensor(receivers, dtype=torch.long),
        surf_faces_local=surf_faces_local,
        full_to_surface=surf_node_ids,
    )


@dataclass
class HitExample:
    rollout: int
    hit: int
    x_k_vtu: str  # path (relative to dataset_dir) providing the pre-hit state
    x_next_vtu: str  # path providing the post-hit state
    d_j_frac: float
    d_j_mm: float
    x_max_band_mm: float
    R_j_deg: float
    u_j_frac: float
    u_j_mm: float


def _group_complete_rollouts(records, n_hits):
    """Groups manifest records by rollout, keeping only rollouts with an
    `undeformed` record and all `n_hits` `hit_final` records present --
    mirrors koopman/dataset.py's `_group_complete_rollouts` filtering, kept
    as an independent minimal implementation here (see GNN/README.md's Stack
    section: whether to import koopman's loader directly was left open;
    resolved here in favor of a standalone version to keep this module's
    only dependency on koopman being documentation, not code)."""
    by_rollout = {}
    for r in records:
        by_rollout.setdefault(r["rollout"], {})[r["hit"]] = r
    complete = {}
    for rollout_id, hits in by_rollout.items():
        if 0 in hits and all(h in hits for h in range(1, n_hits + 1)):
            complete[rollout_id] = hits
    return complete


def load_examples(dataset_dir: str, allowed_rollouts=None):
    """Returns (examples, manifest) -- one HitExample per hit, across every
    complete rollout in the manifest, in rollout order.

    `allowed_rollouts`, if given (an iterable of rollout ids), restricts to
    that subset -- e.g. reproducing an exact past training snapshot when the
    dataset directory has since grown (see GNN/mp_sweep/'s message-passing
    depth comparison, where every run must train on identical data so depth
    is the only varying factor)."""
    with open(os.path.join(dataset_dir, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    n_hits = manifest["n_hits_per_rollout"]
    complete = _group_complete_rollouts(manifest["records"], n_hits)
    if allowed_rollouts is not None:
        allowed_rollouts = set(allowed_rollouts)
        complete = {r: hits for r, hits in complete.items() if r in allowed_rollouts}

    # u_j_frac follows d_j_frac's existing convention exactly: a raw,
    # unnormalized [0,1] fraction over the dataset's configured range (see
    # GNN/README.md's "Normalization" section) rather than routing u_j
    # through the OnlineNormalizer -- settled by explicit interview.
    u_min = manifest["min_compression_displacement_mm"]
    u_max = manifest["max_compression_displacement_mm"]
    u_range = u_max - u_min

    examples = []
    for rollout_id in sorted(complete):
        hits = complete[rollout_id]
        for h in range(1, n_hits + 1):
            prev, cur = hits[h - 1], hits[h]
            u_j_mm = cur["u_j_mm"]
            examples.append(HitExample(
                rollout=rollout_id, hit=h,
                x_k_vtu=prev["vtu_path"], x_next_vtu=cur["vtu_path"],
                d_j_frac=cur["d_j_frac"], d_j_mm=cur["d_j_mm"],
                x_max_band_mm=cur["x_max_band_mm"], R_j_deg=cur["R_j_deg"],
                u_j_frac=(u_j_mm - u_min) / u_range, u_j_mm=u_j_mm,
            ))
    return examples, manifest


def split_examples_by_rollout(examples, train_frac: float = 0.8):
    """By-rollout prefix split (GNN/README.md's settled "Train/test split"
    decision) -- all of a rollout's hits go entirely to train or entirely to
    test, split on sorted rollout id, not shuffled."""
    rollout_ids = sorted({ex.rollout for ex in examples})
    n_train = max(1, int(round(len(rollout_ids) * train_frac)))
    train_rollouts = set(rollout_ids[:n_train])
    train = [ex for ex in examples if ex.rollout in train_rollouts]
    test = [ex for ex in examples if ex.rollout not in train_rollouts]
    return train, test


def build_node_features(mesh_info: SurfaceMeshInfo, x_k: torch.Tensor, d_j_frac: float,
                         d_j_mm: float, x_max_band_mm: float, R_j_deg: float,
                         u_j_frac: float) -> torch.Tensor:
    """Builds the raw (unnormalized) 11-dim node feature tensor for one hit,
    given an explicit `x_k` -- shared by `ForgeGNNDataset.__getitem__` (where
    `x_k` comes from the true .vtu) and train.py's autoregressive rollout
    eval (where `x_k` is the model's own previous prediction), so both paths
    build features identically."""
    band_center_mm = 0.5 * (d_j_mm + x_max_band_mm)
    # .to(x_k.device): d_j_mm/x_max_band_mm are plain floats on the training
    # path (band_center_mm stays a float, device-agnostic), but control/
    # mpc.py's SQP rollout passes grad-tracking tensors for these -- on GPU,
    # band_center_mm is then a CUDA tensor, and mesh_info.rest_pos (always
    # CPU, unlike ForgeGNN's own device-moved buffer copy of it) would
    # mismatch without this. Latent bug: every prior MPC run used
    # device="cpu", where the two device strings just happened to match.
    dist_to_band = mesh_info.rest_pos[:, 0:1].to(x_k.device) - band_center_mm

    # torch.as_tensor (not torch.tensor) so a grad-tracking tensor passed in
    # from control/mpc.py's SQP planner (differentiating the GNN w.r.t.
    # d_j/R_j/u_j) flows through unchanged; a plain float (the usual
    # training-data path) is just wrapped fresh, same as before.
    d_j_frac_t = torch.as_tensor(d_j_frac, dtype=torch.float32, device=x_k.device)
    R_j_rad_t = torch.deg2rad(torch.as_tensor(R_j_deg, dtype=torch.float32, device=x_k.device))
    u_j_frac_t = torch.as_tensor(u_j_frac, dtype=torch.float32, device=x_k.device)
    control = torch.stack([d_j_frac_t, torch.sin(R_j_rad_t), torch.cos(R_j_rad_t), u_j_frac_t])
    control_broadcast = control.unsqueeze(0).expand(x_k.shape[0], -1)

    return torch.cat([
        mesh_info.is_bottom_face.to(x_k.device),
        mesh_info.is_pin_node.to(x_k.device),
        mesh_info.is_lateral_wall.to(x_k.device),
        dist_to_band.to(x_k.device),
        x_k,
        control_broadcast,
    ], dim=-1)


class ForgeGNNDataset(torch.utils.data.Dataset):
    """One item = one hit transition: raw (unnormalized) node features,
    x_k, and the target displacement delta, all restricted to the surface
    node subset. Edge features are NOT built here -- they depend on x_k in
    the current (world) frame and are cheap to recompute, so model.py builds
    them from `mesh_info` + `x_k` inside the forward pass (matching the
    reference's own `_build_graph`, which builds edges from raw inputs, not
    a data-loader precomputation)."""

    def __init__(self, dataset_dir: str, examples, mesh_info: SurfaceMeshInfo):
        self.dataset_dir = dataset_dir
        self.examples = examples
        self.mesh_info = mesh_info
        self._cache = {}

    def __len__(self):
        return len(self.examples)

    def _load_surface_displacement(self, vtu_rel_path: str) -> torch.Tensor:
        if vtu_rel_path not in self._cache:
            mesh = meshio.read(os.path.join(self.dataset_dir, vtu_rel_path))
            disp_full = mesh.point_data["Displacement"].astype(np.float32)
            disp_surf = disp_full[self.mesh_info.full_to_surface]
            self._cache[vtu_rel_path] = torch.tensor(disp_surf, dtype=torch.float32)
        return self._cache[vtu_rel_path]

    def __getitem__(self, idx):
        ex = self.examples[idx]
        x_k = self._load_surface_displacement(ex.x_k_vtu)
        x_next = self._load_surface_displacement(ex.x_next_vtu)
        target_delta = x_next - x_k

        # Column order fixed by GNN/README.md's per-node feature table:
        # [is_bottom_face, is_pin_node, is_lateral_wall, dist_to_band, x_k(3), control(4)]
        node_features = build_node_features(
            self.mesh_info, x_k, ex.d_j_frac, ex.d_j_mm, ex.x_max_band_mm, ex.R_j_deg, ex.u_j_frac,
        )

        return {
            "node_features": node_features,  # (n_surface, 11), raw
            "x_k": x_k,  # (n_surface, 3)
            "target_delta": target_delta,  # (n_surface, 3)
            "rollout": ex.rollout,
            "hit": ex.hit,
        }


def save_surface_vtu(mesh_info: SurfaceMeshInfo, displacement: np.ndarray, out_path: str):
    """Writes a surface-only .vtu (points = surface rest positions, cells =
    the extracted surface triangulation, point_data = Displacement) for
    visual true-vs-predicted comparison in ParaView. Deliberately a separate
    surface mesh rather than embedding into a full-volumetric template
    (koopman/dataset.py's `save_state_vtu` pattern) since this model never
    predicts interior nodes -- see GNN/README.md's "Graph construction"
    section."""
    out_mesh = meshio.Mesh(
        points=mesh_info.rest_pos.numpy(),
        cells=[("triangle", mesh_info.surf_faces_local)],
        point_data={"Displacement": displacement.astype(np.float32)},
    )
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    out_mesh.write(out_path)
