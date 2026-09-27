# runs_3dim — first 3D-actuation GNN training run

Output of the GNN's first training run on the 3D-actuation uniform
random-hit data (station position, angle and stroke all randomized per
hit), 2026-09-11. The checkpoint it produced is `GNN/checkpoint_3dim.pt`.

## Setup

- `GNN/train.py` at its defaults: latent size 128, 2 layers per MLP,
  **15 message-passing steps** (2,333,699 parameters), batch 4, lr 1e-4,
  patience 20.
- Data: `data/dataset_pretraining`, 5-hit rollouts. Its rollouts were later
  pinned as the 383-rollout snapshot (306 train / 77 test), which every
  later model uses so dataset growth can't confound comparisons.
- Best epoch 37 of 57; 36 min.

## Results

- One-hit-ahead test NRMSE: 0.299.
- Autoregressive 5-hit rollout on test rollout 338:
  - RMSE per hit: 0.043, 0.098, 0.153, 0.178, 0.172 mm
  - Hausdorff per hit: 0.45, 0.30, 0.50, 0.53, 0.60 mm

## Status

Superseded for control by the 5-step models: the old depth sweep
(`GNN/old/mp_sweep/`) found 5 steps as accurate as 15 at a third of the MPC
gradient cost, and the MPC now uses `finetune_square/`'s square-rod
finetune of the 5-step model. `control/run_closed_loop.py` still defaults to
`checkpoint_3dim.pt`, and `control/validate_open_loop.py` reads
`eval/rollout_339/` from here.

## Contents

- `metrics.json` — training history, best epoch, per-hit rollout metrics.
- `eval/rollout_338/`, `eval/rollout_339/` — true vs. predicted surface
  `.vtu` pairs from autoregressive evaluation.
- `loss_curves.png`, `raw_loss_curve.png`, `geometry_error_curves.png` —
  training curves.
