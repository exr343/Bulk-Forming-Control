# SVD / POD analysis — design notes

## Data source

Reuses `applications/Agility_Forge/koopman/dataset.py`'s manifest + `.vtu` snapshot loading
directly (`load_manifest`, `load_state`, `compute_state_scale`, `normalize_state`,
`_group_complete_rollouts`) rather than duplicating it. `load_state` returns the flattened
**displacement-only** state (`n_x = n_nodes*3`, Temperature dropped) that the Koopman model
itself trains on — this analysis is meant to describe what that encoder actually has to
represent.

Default `--dataset-dir` is the frozen `data/dataset_snapshot_r25` snapshot (150 records, 25
complete rollouts), not the live `data/dataset/` — a separate `generate_dataset.py` job can be
actively (non-atomically) appending to that directory, and reading it concurrently risks a torn
`manifest.json` read (see the top-level `CLAUDE.md`). Point `--dataset-dir` at it explicitly
once that job has finished.

`_group_complete_rollouts` is single-underscore ("not public API"), but it's the same package,
not a foreign library. Reimplementing its rollout-completeness filtering here would risk two
copies of the same logic silently drifting apart -- importing it directly is a deliberate choice,
not an oversight.

## Design decisions (implemented)

- **Rows = snapshots.** The snapshot matrix `X` is `(n_snapshots, n_features)` — one row per
  saved state (undeformed + hit_final, every complete rollout), matching both the sklearn PCA
  convention and `koopman/dataset.py`'s own `[batch, n_x]` tensor layout. Because
  `n_snapshots (150) << n_features (22599)`, economy SVD (`np.linalg.svd(..., full_matrices=False)`)
  is bounded by `n_snapshots`, not `n_features` — sub-second on CPU, no GPU, no snapshot-method/
  Gram-matrix trick needed.
- **Normalize, then mean-center, in that order.** `load_snapshot_matrix` normalizes every row
  with `koopman/dataset.py`'s pooled `[-1,1]` min-max scale (fit here on the *full* snapshot set,
  not a train-only split — there's no train/test split in this analysis, it's exploratory).
  `compute_svd` then subtracts the mean of that normalized matrix before taking the SVD. Since
  `normalize_state` is an affine map, centering *after* normalizing means the centered matrix is
  exactly "what the Koopman encoder sees, minus its own mean" — the intended target of this
  analysis. Centering first and normalizing after would not give the same result and isn't done.
- **Cumulative energy**: `cumsum(S**2) / sum(S**2)` on the singular values of the centered
  matrix — the standard POD energy spectrum, i.e. how many modes/singular vectors are needed to
  capture a given fraction of the variance in how rollouts differ from the mean deformed shape.
- **Effective dimension**: first mode count `r` (1-indexed) whose cumulative energy crosses each
  of `--thresholds` (default `0.90, 0.95, 0.99, 0.999`), reported to stdout, saved to
  `effective_dimension.json`, and annotated on the plot.
- **Rank note**: after mean-centering, the matrix has rank `<= n_snapshots - 1` (one degree of
  freedom removed by centering) — with the default 150-snapshot dataset, expect the last
  singular value(s) to be ~0 even though economy SVD returns `min(150, 22599) = 150` of them.
  This is expected, not a bug.
- **Plot**: `cumulative_energy.png` — x = mode count (linear scale; the goal is *locating where
  the curve saturates*, not inspecting decay across orders of magnitude, so no log-x variant is
  produced), y = cumulative energy (0 to ~1.0), with dashed guide lines at each requested
  threshold.
- **Output directory**: `SVD/runs/` is gitignored (see `.gitignore`), same rationale as
  `koopman/runs/` — regenerable by re-running `svd_analysis.py`, not source. The right singular
  vectors (`Vt`, the actual spatial POD modes, ~27MB at this dataset size) are only saved when
  `--save-modes` is passed, to keep the default output small.

## Stack

NumPy (`np.linalg.svd`) + Matplotlib only — no PyTorch/JAX. Deliberately decoupled from both
`jax_forge` and the Koopman model's training stack: this is a plain offline analysis of the
`.vtu`-derived snapshot matrix, meant to run in seconds on a CPU, no SLURM job required.
