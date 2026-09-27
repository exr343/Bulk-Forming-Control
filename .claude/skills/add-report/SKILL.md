---
name: add-report
description: Write a progress report for a finished (or failed) experiment in this repo — an MPC/control run or a GNN training/sweep run — into applications/Agility_Forge/progress/{control_reports,GNN_reports}/, with a markdown report, the run's figures, and the key numbers, all verified against the run's own output files. Use when the user asks to "add a report", "write up" a run, or document results/conclusions for control or GNN work.
---

# add-report

Turn one experiment's output into a self-contained progress report the user can
read, share with collaborators, and later mine for the CIRP 2027 paper.

## Where reports go

```
applications/Agility_Forge/progress/
  README.md                       # index: one line per report, newest first
  control_reports/<YYYY-MM-DD>_<short_slug>/
  GNN_reports/<YYYY-MM-DD>_<short_slug>/
```

- `control_reports/`: anything in `applications/Agility_Forge/control/` (MPC runs, open/closed-loop evals).
- `GNN_reports/`: anything in `applications/Agility_Forge/GNN/` (training, finetuning, message-passing-steps sweeps).
- Use today's date and a short slug naming the experiment, e.g. `2026-09-26_cost_e0_original`, `2026-09-26_mp_sweep`.
- If the user didn't say which experiment, ask. Don't guess between runs.

Each report folder contains:

- `report.md`: the report.
- `figures/`: every image the report shows. Copy the run's own plots; add new ones only when they show something the existing ones don't.
- `make_<name>_figure.py`: the script behind any NEW figure, runnable from the repo root, reading only from the run's output folders, so the figure can be regenerated.
- A copy of the run's small summary file if there is one (`summary.json`, not checkpoints or `.vtu` dumps).

After writing, add one line to `progress/README.md` (create it if missing): date, link to `report.md`, and a one-sentence result.

## Where the facts come from

Read the run's own files. Never write a number from memory or from the chat.

| Experiment type | Read first |
|---|---|
| Control / MPC (`control/eval_*.py`) | the script's module docstring (the formulation and settled choices); `control/results/<run>/results.json` (`config`, per-step `steps`, `reference`); `failed_steps.jsonl` if present; the run's `.png` plots; the SLURM log in `slurm_logs/`; `control/README.md` for known bugs affecting older runs |
| GNN training / finetune (`GNN/train.py`, `GNN/finetune.py`) | the run's `metrics.json` (`args`, `best_epoch`, per-hit rollout metrics, `baseline` vs `finetuned`, `pretrain_test` forgetting check) and its plots |
| Message-passing-steps sweep (`GNN/mp_sweep/`) | `GNN/mp_sweep/summary.json`, `GNN/mp_sweep/finetune/mp_*/metrics.json`, `GNN/mp_sweep/README.md` |

A control report documents what the eval script actually ran: restate its setup
(target, surrogate checkpoint, horizon, cost, bounds, number of hits) from its
docstring and `results.json["config"]`, in plain words.

## Procedure

1. **Gather:** read the files above; confirm the job's final state
   (`sacct -j <id>`) and whether it finished, failed or is partial.
2. **Look at every figure** (Read the `.png`) before using it. If labels overlap,
   text is clipped, or the plot is misleading, fix the plotting code and redraw
   rather than shipping it.
3. **Explain, don't just list.** If the result is surprising (e.g. a controller
   doing something odd, a failure), dig into the saved state to find out why and
   put the evidence in the report: cost decomposition, where on the bar
   something happened, mesh distortion (min det F from the checkpoint's
   internal variables), GNN prediction error vs. the plant. Numbers over
   adjectives.
4. **Write `report.md`** with these sections, adapted to the experiment:
   - Title stating the outcome, and a line with date, script, SLURM job ids, raw output path.
   - **Summary:** 2-4 short paragraphs: what was asked, what happened, what it means.
   - **What was run:** the setup, restated plainly.
   - **Results:** tables of the key numbers and the figures, each figure followed by one or two sentences on what it shows.
   - **Why / analysis** (when something needs explaining), with evidence.
   - **Conclusions** (for GNN reports), numbered and each backed by the table.
   - **Caveats:** single seed, single test run, test set also used for early stopping, data from before a bug fix, anything else that limits the claim.
   - **Possible next steps.**
   - **Files:** what's in the folder and where each figure came from.
5. **Verify** every number in the report against the source files once more
   before finishing. Re-read the report for leftover edits or contradictions.
6. Tell the user where the report is and give its headline in two or three
   sentences. Don't commit unless asked.

## Required metrics

- **Control / MPC reports** always report, per hit and in a summary table:
  **Hausdorff error** and **Chamfer error** to the target, and **computation
  time**. That's MPC planning time per hit (mean, median, max), plus simulator
  time per hit for real-simulator runs. The main figure plots all three against
  hit (Hausdorff, Chamfer, planning time), like `eval_square_target.py`'s
  `error_vs_hit.png`. Extra diagnostics (stretch, cross-section, controls) are
  secondary figures.
- **GNN reports** always state the number of message-passing steps (M) of
  every model used.
- Every report states which GNN (checkpoint path and M) planned or was
  evaluated.

## Style

- Plain language for a reader who knows forging and ML but not this codebase.
  Define terms like Hausdorff, Chamfer, NRMSE, message-passing steps (M; never "depth"), and
  station/stroke on first use.
- No ASCII diagrams. Tables and the figures carry the structure.
- Don't overclaim: state what the evidence supports, and label judgment calls
  as such. Report failures and negative results as plainly as successes.
- New figures follow the repo's plot style: light surface `#fcfcfb`, recessive
  grid, top/right spines off; series colors in the fixed order
  `#2a78d6, #eb6834, #1baf7a, #eda100, #e87ba4, #008300`; a reference or
  baseline line in grey `#8a8984`, dotted; one axis per panel (never
  dual-axis); a legend for 2+ series, plus direct end labels when lines are
  close. See `control/eval_square_target.py` and `GNN/mp_sweep/summarize.py`
  for working examples.
