"""Loads Agility_Forge rollouts (manifest.json + .vtu snapshots) into training tensors.

Data source: applications/Agility_Forge/generate_dataset.py writes, per rollout,
one manifest.json record per saved state:
    {rollout, hit, kind ("undeformed" | "hit_final"),
     d_j_frac, d_j_mm, x_max_band_frac, x_max_band_mm, R_j_deg, u_j_mm,
     total_time_s, step_count, wall_time_s, vtu_path}
and one .vtu per record holding per-node point fields "Displacement" (3-vec)
and "Temperature" (scalar) on the shared billet mesh -- only "Displacement"
is used here; Temperature is dropped from the state.

State vector: the flattened Displacement field over every node, n_x =
n_nodes * 3. Node ordering is fixed across every .vtu (one shared mesh,
reused for every rollout/hit by generate_dataset.py), so a single flatten
order is valid globally.

Control vector: [d_j_frac, sin(R_j), cos(R_j)] per hit, used *raw* -- in
Koopman model fitting the inputs are left untouched; only the state is
lifted/transformed (state gets a learned nonlinear lift + a data-driven
[-1, 1] rescaling, u does not).

State normalization is pooled per physical quantity, not per-feature: one
[-1, 1] min-max scale shared across every Displacement component (all nodes,
all 3 axes). Rescaling ux/uy/uz independently would distort displacement
*direction*, which is physically meaningful here.
"""

import json
import math
import os

import meshio
import numpy as np
import torch
from torch.utils.data import TensorDataset


def load_manifest(dataset_dir):
    """Read manifest.json and return (meta, records) — meta is the top-level
    generation config (n_hits_per_rollout, band_width_frac, ..., and
    generation_runs provenance if the dataset was built by more than one
    generate_dataset.py invocation), records is the list of per-state entries
    described in the module docstring.
    """
    manifest_path = os.path.join(dataset_dir, "manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    records = manifest["records"]
    meta = {k: v for k, v in manifest.items() if k != "records"}
    return meta, records


def load_state(vtu_path):
    """Read one .vtu snapshot and return a flattened (n_nodes*3,) state
    vector: Displacement.ravel(). Temperature is read from the same .vtu by
    generate_dataset.py but is not part of the model's state.

    Reads the "Displacement" point-data field explicitly (not the redundant
    "sol" alias jax_forge.utils.save_sol also writes).
    """
    mesh = meshio.read(vtu_path)
    displacement = np.asarray(mesh.point_data["Displacement"], dtype=np.float64)
    return displacement.reshape(-1)


def unflatten_state(x_flat, n_nodes):
    """Inverse of load_state's flatten: (n_nodes*3,) -> Displacement (n_nodes,3)."""
    x_flat = np.asarray(x_flat)
    return x_flat.reshape(n_nodes, 3)


def save_state_vtu(x_flat, template_vtu_path, out_path):
    """Write one state vector to a .vtu file.

    Reuses the mesh geometry (points, cells) from an existing dataset .vtu --
    every snapshot shares the same mesh, so there is no need to reconstruct
    the live jax_forge FiniteElement/Problem just to write output. Field name
    matches generate_dataset.py's convention (Displacement) so predicted/true
    files diff directly in ParaView. No Temperature field is written -- it
    isn't part of the model's state, so there is nothing to predict/compare.
    """
    template = meshio.read(template_vtu_path)
    n_nodes = template.points.shape[0]
    displacement = unflatten_state(x_flat, n_nodes)
    out_mesh = meshio.Mesh(
        points=template.points,
        cells=template.cells,
        point_data={"Displacement": displacement.astype(np.float32)},
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out_mesh.write(out_path)


def _control_vector(record):
    d_j_frac = float(record["d_j_frac"])
    r_rad = math.radians(float(record["R_j_deg"]))
    return np.array([d_j_frac, math.sin(r_rad), math.cos(r_rad)], dtype=np.float64)


def _group_complete_rollouts(records, n_hits):
    """Group records by rollout, keep only rollouts with a complete state
    sequence (one "undeformed" at hit=0, exactly n_hits "hit_final" records
    at hit=1..n_hits) -- robust to a rollout the data-generation job hasn't
    finished writing yet. Returns a list of (rollout_idx, [rec_0, ..., rec_K])
    sorted by rollout_idx ascending.
    """
    by_rollout = {}
    for rec in records:
        by_rollout.setdefault(rec["rollout"], {})[rec["hit"]] = rec

    complete = []
    for rollout_idx in sorted(by_rollout):
        hits = by_rollout[rollout_idx]
        if set(hits.keys()) != set(range(n_hits + 1)):
            continue
        if hits[0]["kind"] != "undeformed":
            continue
        if any(hits[k]["kind"] != "hit_final" for k in range(1, n_hits + 1)):
            continue
        complete.append((rollout_idx, [hits[k] for k in range(n_hits + 1)]))
    return complete


def _minmax(values):
    lo = float(values.min())
    rng = float(values.max() - values.min())
    if rng == 0.0:
        rng = 1.0
    return lo, rng


def compute_state_scale(state_vectors, n_nodes):
    """Pooled min-max scale over every Displacement component (all nodes,
    all 3 axes), from a list of raw (unnormalized) flat state vectors."""
    disp_vals = np.concatenate(state_vectors)
    disp_lo, disp_rng = _minmax(disp_vals)
    return {"disp_lo": disp_lo, "disp_rng": disp_rng}


def normalize_state(x_flat, x_scale, n_nodes):
    x_flat = np.asarray(x_flat, dtype=np.float64).copy()
    x_flat[:] = 2 * (x_flat - x_scale["disp_lo"]) / x_scale["disp_rng"] - 1
    return x_flat


def denormalize_state(x_flat, x_scale, n_nodes):
    x_flat = np.asarray(x_flat, dtype=np.float64).copy()
    x_flat[:] = (x_flat + 1) / 2 * x_scale["disp_rng"] + x_scale["disp_lo"]
    return x_flat


def compute_pod_basis(state_vectors, n_modes):
    """Fit a rank-n_modes POD basis on already-normalized ([-1,1]) state
    vectors: mean-center, then economy SVD of the centered matrix, keeping
    the first n_modes right singular vectors. Order matters -- these vectors
    must already be normalize_state-normalized, matching applications/
    Agility_Forge/SVD/svd_analysis.py's normalize-then-center convention, so
    the POD basis describes variance in exactly the space the model's
    encoder receives.

    Returns (mean_row, modes): mean_row is (n_features,), modes is
    (n_modes, n_features) with orthonormal rows. Projection onto the basis
    is (x - mean_row) @ modes.T; reconstruction is mean_row + a @ modes.
    """
    X = np.stack(state_vectors)
    mean_row = X.mean(axis=0)
    _, _, Vt = np.linalg.svd(X - mean_row, full_matrices=False)
    return mean_row, Vt[:n_modes]


def build_rollout_dataset(dataset_dir, train_frac=0.8, n_pod_modes=None):
    """Assemble train/test windowed datasets for Koopman Autoencoder training.

    Each complete rollout becomes exactly one window covering the whole
    rollout (K = n_hits_per_rollout): states [x_0, ..., x_K] and controls
    [u_0, ..., u_{K-1}], where u_{k-1} is the control that produced hit k's
    state x_k from x_{k-1}. Rollouts are split by index, prefix-style
    (rollouts are already IID -- independent seeded draws per rollout in
    generate_dataset.py -- so no shuffling is needed), and the state
    normalization scale is fit on the training rollouts only.

    If n_pod_modes is given, also fits a POD basis (see compute_pod_basis)
    on the normalized training states only -- same train-only-fit principle
    as x_scale, so the basis itself never sees test data -- and adds
    pod_mean/pod_modes to info for the model to consume.

    Returns (train_dataset, test_dataset, info), where train_dataset/
    test_dataset are torch.utils.data.TensorDataset instances yielding
    tuples (x_0, ..., x_K, u_0, ..., u_{K-1}), and info is a dict with
    x_scale, n_nodes, n_x, n_u, n_hits, rollout counts, a template .vtu
    path (for save_state_vtu) taken from the first complete rollout, and
    (if n_pod_modes given) pod_mean/pod_modes/n_pod_modes.
    """
    meta, records = load_manifest(dataset_dir)
    n_hits = meta["n_hits_per_rollout"]
    complete = _group_complete_rollouts(records, n_hits)
    if len(complete) < 2:
        raise RuntimeError(
            f"Need at least 2 complete rollouts (0..{n_hits} hits each) to form a "
            f"train/test split; found {len(complete)} under {dataset_dir}."
        )

    raw_states, raw_controls = [], []
    n_nodes = None
    template_vtu_path = None
    for rollout_idx, hit_records in complete:
        states = []
        for rec in hit_records:
            vtu_path = os.path.join(dataset_dir, rec["vtu_path"])
            if template_vtu_path is None:
                template_vtu_path = vtu_path
            x = load_state(vtu_path)
            if n_nodes is None:
                n_nodes = x.shape[0] // 3
            states.append(x)
        controls = [_control_vector(rec) for rec in hit_records[1:]]
        raw_states.append(states)
        raw_controls.append(controls)

    n_train = int(round(train_frac * len(raw_states)))
    n_train = min(max(n_train, 1), len(raw_states) - 1)
    train_states, test_states = raw_states[:n_train], raw_states[n_train:]
    train_controls, test_controls = raw_controls[:n_train], raw_controls[n_train:]

    x_scale = compute_state_scale([x for rollout in train_states for x in rollout], n_nodes)

    pod_mean = pod_modes = None
    if n_pod_modes is not None:
        train_states_norm = [normalize_state(x, x_scale, n_nodes)
                              for rollout in train_states for x in rollout]
        pod_mean, pod_modes = compute_pod_basis(train_states_norm, n_pod_modes)

    def _make_dataset(states_by_rollout, controls_by_rollout):
        xs = [[] for _ in range(n_hits + 1)]
        us = [[] for _ in range(n_hits)]
        for states, controls in zip(states_by_rollout, controls_by_rollout):
            for k, x in enumerate(states):
                xs[k].append(normalize_state(x, x_scale, n_nodes))
            for k, u in enumerate(controls):
                us[k].append(u)
        x_tensors = [torch.from_numpy(np.stack(a)).float() for a in xs]
        u_tensors = [torch.from_numpy(np.stack(a)).float() for a in us]
        return TensorDataset(*(x_tensors + u_tensors))

    train_dataset = _make_dataset(train_states, train_controls)
    test_dataset = _make_dataset(test_states, test_controls)

    rollout_indices = [idx for idx, _ in complete]
    info = {
        "x_scale": x_scale,
        "n_nodes": n_nodes,
        "n_x": n_nodes * 3,
        "n_u": 3,
        "n_hits": n_hits,
        "n_rollouts_train": len(train_states),
        "n_rollouts_test": len(test_states),
        "template_vtu_path": template_vtu_path,
        "train_rollout_indices": rollout_indices[:n_train],
        "test_rollout_indices": rollout_indices[n_train:],
    }
    if n_pod_modes is not None:
        info["pod_mean"] = pod_mean
        info["pod_modes"] = pod_modes
        info["n_pod_modes"] = n_pod_modes
    return train_dataset, test_dataset, info
