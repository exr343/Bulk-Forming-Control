# finetune_square — production square-rod finetune

The GNN the square-rod MPC uses (`control/eval_square_target.py`):
`checkpoint_mp_5_finetuned_square.pt`, produced by `GNN/finetune.py` via
`slurm_scripts/submit_gnn_finetune_square.sh` (job 3881754, 2026-09-25).

That script wipes everything in this folder except this README at the start
of every run, so the numbers below are the record of the current contents.

## Setup

- Start: `mp_sweep/checkpoint_mp_5.pt` (5 message-passing steps, pretrained
  on the uniform random-hit data, 383-rollout snapshot).
- Train: square + `square_jitter_seed{1,2,4..9}` (432 hits), plus 432
  pretraining-train hits freshly drawn each epoch (`--replay-ratio 1.0`).
- Test and early stopping: `square_jitter_seed3` (48 hits). Seed 3 both
  picks the stopping epoch and gives the score, so the score is slightly
  optimistic. The later depth sweep (`mp_sweep/`) uses a separate
  validation run instead.
- Same per-node MSE loss as pretraining, AdamW lr 1e-5, weight decay 1e-4,
  batch 4, normalizers frozen, patience 20, max 300 epochs.
- Best epoch 116 of 136; 20 min on an L40S.

## Results (seed 3, end of a 48-hit autoregressive rollout)

| Model | Chamfer, hit 48 (mm²) | Hausdorff, hit 48 (mm) | RMSE, hit 48 (mm) | Pretraining-test NRMSE |
|---|---|---|---|---|
| Pretrained (`baseline/`) | 23.79 | 17.46 | 4.39 | 0.298 |
| Finetuned (`finetuned/`) | 0.91 | 2.11 | 0.52 | 0.298 |

The last column is the forgetting check: one-hit-ahead NRMSE on the
77-rollout uniform test split. It's unchanged, so there's no forgetting.
The full comparison with the earlier no-replay finetune is in
`GNN/README.md` ("Finetuning on square-rod data").

## Contents

- `checkpoint_mp_5_finetuned_square.pt` — the finetuned model (git-ignored).
- `metrics.json` — before/after metrics for every split, per-hit seed-3
  rollout errors, training history, and the run's arguments.
- `baseline/eval/rollout_10/`, `finetuned/eval/rollout_10/` — true vs.
  predicted surface `.vtu` pairs for the 48-hit seed-3 rollout, plus a
  per-hit table (`eval_table.csv`). "rollout_10" is seed 3's label inside
  this run (training runs are numbered 1-9 first).
- `loss_curves.png`, `raw_loss_curve.png`, `geometry_error_curves.png` —
  training curves (the "test" curve is seed 3).
