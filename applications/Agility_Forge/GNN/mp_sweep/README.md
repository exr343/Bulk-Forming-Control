# mp_sweep — message-passing-steps (M) sweep, square-rod regime

Which number of message-passing steps M should the GNN processor use? The
first sweep (now `GNN/old/mp_sweep/`, M = 1/5/15/45/135) tested only the
uniform random-hit data, where every M reached the same one-hit-ahead
error (NRMSE 0.296-0.300). This sweep asks the same question in the regime
the MPC now works in: long square-rod rollouts with heavy deformation.

## Design (settled by interview, 2026-09-26)

- **Message-passing steps:** M = 1, 2, 3, 5, 8, 15 — fine resolution where the old sweep
  found saturation, plus 15 (the old default).
- **Two-stage training for each M**, the same pipeline as the production model:
  1. **Pretrain** on the uniform data, 383-rollout snapshot
     (`rollout_snapshot_383.json`, 306 train / 77 test), `GNN/train.py`
     defaults. M = 1, 5, 15 reuse the old sweep's checkpoints (copied here,
     identical files); M = 2, 3, 8 are new (`pretrain/mp_<M>/`).
  2. **Finetune** (`GNN/finetune.py`) on square + `square_jitter_seed{1,2,4..8}`
     (384 hits) plus an equal number of uniform-train hits drawn fresh each
     epoch (replay ratio 1.0). Normalizers frozen, lr 1e-5, patience 20,
     max 300 epochs.
- **Held-out runs:** early stopping on `square_jitter_seed9`; the test run
  `square_jitter_seed3` is used only for final scores, so they're unbiased.
- **Reported metrics** (`summarize.py`):
  1. **Gradient time of a 10-hit horizon:** one forward + backward pass
     through the MPC cost (`control/mpc.py`'s `_rollout_cost`), i.e. one SQP
     iteration's gradient. Undeformed billet, fixed control guess, 1 warm-up
     + 10 timed repeats, CUDA-synchronized, every M on the same GPU.
  2. **Hausdorff and Chamfer on the test run:** the finetuned model's full
     48-hit autoregressive rollout of seed 3, at every hit and as the mean
     over all 48 hits.
- **No selection rule** — results are reported and M chosen by hand.

## Running

From the repo root (L40S; jobs submitted 2026-09-26):

```bash
sbatch --array=2,3,8 applications/Agility_Forge/slurm_scripts/submit_mp_sweep_pretrain.sh      # 3885756
sbatch --array=1,5,15 applications/Agility_Forge/slurm_scripts/submit_mp_sweep_finetune.sh     # 3885757
sbatch --array=2,3,8 --dependency=afterok:<pretrain> .../submit_mp_sweep_finetune.sh            # 3885758
sbatch --dependency=afterok:<both finetunes> .../submit_mp_sweep_summary.sh                     # 3885759
```

## Layout

- `rollout_snapshot_383.json` — the pinned uniform split.
- `checkpoint_mp_<M>.pt` — uniform-pretrained checkpoints (git-ignored).
  `checkpoint_mp_5.pt` is also the default starting point for `finetune.py`
  and several `control/` evaluations.
- `pretrain/mp_<M>/` — pretraining metrics (M = 1, 5, 15 copied from the old
  sweep; 2, 3, 8 with loss curves and eval dumps).
- `finetune/mp_<M>/` — finetuned `checkpoint.pt`, `metrics.json`,
  baseline/finetuned seed-3 eval dumps, training curves (the "test" curve
  there is the seed-9 validation run).
- `summarize.py`, `summary.json`, `test_hausdorff_vs_hit.png`,
  `test_chamfer_vs_hit.png`, `mean_error_vs_mp_steps.png`, `gradient_time_vs_mp_steps.png` — the report.

## Results

Pending: the jobs above are queued or running.
