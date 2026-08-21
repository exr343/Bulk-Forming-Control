"""Loads Agility_Forge rollouts (manifest.json + .vtu snapshots) into training tensors.

Data source: applications/Agility_Forge/generate_dataset.py writes, per rollout,
one manifest.json record per saved state:
    {rollout, hit, kind ("undeformed" | "hit_final"),
     d_j_frac, d_j_mm, x_max_band_frac, x_max_band_mm, R_j_deg, u_j_mm,
     total_time_s, step_count, wall_time_s, vtu_path}
and one .vtu per record holding per-node point fields "Displacement" (3-vec)
and "Temperature" (scalar) on the shared billet mesh.

TODO: this module is a stub. Fill in once the state-vector representation is decided.
"""

import json
import os


def load_manifest(dataset_dir):
    """Read manifest.json and return (meta, records) — meta is the top-level
    generation config (seed, n_rollouts, n_hits_per_rollout, ...), records is
    the list of per-state entries described above.
    """
    manifest_path = os.path.join(dataset_dir, "manifest.json")
    raise NotImplementedError("TODO: parse manifest.json into (meta, records)")


def load_state(vtu_path):
    """Read one .vtu snapshot and return a fixed-size state vector.

    TODO: decide the state representation — e.g. flatten(Displacement) with or
    without Temperature, node ordering assumptions (mesh is shared across
    rollouts per generate_dataset.py, so a fixed flatten order should be valid),
    and whether to subsample/downsample nodes for a tractable Koopman latent dim.
    """
    raise NotImplementedError("TODO: meshio.read(vtu_path) -> flattened state vector")


def build_rollout_dataset(dataset_dir):
    """Assemble per-rollout (state_sequence, control_sequence) pairs suitable for
    Koopman Autoencoder training — i.e. for each rollout, the ordered sequence of
    hit_final states plus the control inputs (d_j, x_max_band, R_j, u_j) that
    produced each transition.

    TODO: implement once load_manifest/load_state are implemented. Should return
    a torch.utils.data.Dataset (or list of tensors) of (x_t, u_t, x_{t+1}) tuples,
    or full rollout sequences if the training loss needs multi-step consistency.
    """
    raise NotImplementedError
