# Progress reports

One folder per experiment, each with a `report.md`, its figures, and any
script used to make new figures. Written with the `add-report` skill
(`.claude/skills/add-report/`). Newest first.

## Control (MPC)

- 2026-10-08 — [MPC stage briefing (for handing to another agent)](control_reports/2026-10-08_mpc_briefing/report.md) ([PDF](control_reports/2026-10-08_mpc_briefing/report.pdf)):
  self-contained summary of the coil MPC: problem, GNN interface, loop, cost, multi-start planner, run history, files, open issues.
- 2026-10-08 — [Multi-start planner (random shooting + top-K SLSQP)](control_reports/2026-10-08_multistart_planner/report.md) ([PDF](control_reports/2026-10-08_multistart_planner/report.pdf)):
  top-5 of 200 lowers the GNN-predicted cost 3.8% vs the single default start (5 of 6 problems), 5 random starts 2.5%;
  48 vs 7.6 s per plan; all checks pass (deterministic GPU mode needed). GNN-only; simulator test is MPC job 3990381.
- 2026-09-27 — [Cost experiment 12: cross-section-only cost](control_reports/2026-09-27_cost_e12_cross_section_only/report.md):
  squares the bar better (95% of hits on the flats) but stretches it less (free end 34.8 vs 40.7 mm); Chamfer 42.1 vs 22.0 mm²
  for the 25% share, so length does not follow on its own within 50 hits. 25% share + 0.5-2 mm stroke stays best.
- 2026-09-27 — [Cost experiments 10-11: why the gap control failed](control_reports/2026-09-27_cost_e10_e11_gap_diagnosis/report.md):
  not misses, bugs or mainly the optimizer; the gap-trained GNN over-predicts stretch on the MPC's hits (0.44 vs 0.20 mm/hit)
  and the MPC exploits it. Best surrogate design so far: stroke GNN with 0.5-2 mm stroke (Chamfer 22.0 mm²).
- 2026-09-27 — [Cost experiment 9: gap-control GNN on the real simulator](control_reports/2026-09-27_cost_e9_gap_control/report.md):
  stalls (12 skipped + 12 tiny hits; final Hausdorff 42.7 mm vs. 22.7 open loop) because the GNN predicts stretch
  for misses on thin sections; its no-change examples only covered thick bars.
- 2026-09-27 — [Cost experiment 8: 25% share on the real simulator](control_reports/2026-09-27_cost_e8_real_share25/report.md):
  beats the open loop and tracks the surrogate for 27 hits, then five small 0° hits at the clamp end flatten
  the bar to 6.8 × 24.5 mm (the GNN predicted no thinning) and the reheat fails before hit 33.
- 2026-09-26 — [Cost experiment 7: target-derived cross-section weight](control_reports/2026-09-26_cost_e7_cross_section_share/report.md):
  w set so the cross-section is a fixed share of the starting cost; 25% share, no stroke effort ranks first
  (surrogate, 3 seeds), but no design gets near the ideal part in 50 hits. Real-simulator run queued (e8).
- 2026-09-26 — [Cost experiment 6: ideal square target, lead design](control_reports/2026-09-26_cost_e6_ideal_target_w30/report.md):
  stopped at hit 13; w = 30 gives the cross-section only ~10% of the cost for this longer target, so the MPC
  kept pressing the clamp end. The weight has to be matched to the target (experiment 7).
- 2026-09-26 — [MPC cost function (PDF)](control_reports/2026-09-26_mpc_cost_function/cost_function.pdf):
  the full cost in mathematical notation, every symbol defined, and the weights used in E0-E4 and the real runs
  (source: `cost_function.tex`).
- 2026-09-26 — [Cost experiment 5: real-simulator check of the new costs](control_reports/2026-09-26_cost_e5_real_simulator/report.md):
  all four designs forge the bar close to the square target (lead design at hit 43: Hausdorff 2.86 mm,
  cross-section error 0.61 mm, free end 35.7 vs 35 mm) and no longer pin one spot, but none stops: they
  over-stretch past the target; stopping is the remaining problem.
- 2026-09-26 — [Cost experiment 4: robustness](control_reports/2026-09-26_cost_e4_robustness/report.md):
  5 noisy re-runs per design; w = 100 + effort 0.1's good result was luck, w = 30 + stroke effort 0.3 is the
  most robust (Hausdorff 3.0 ± 0.8 mm, cross-section error 1.37 ± 0.09 mm) and is now the lead candidate.
- 2026-09-26 — [Cost experiment 3: refining the weights](control_reports/2026-09-26_cost_e3_refine/report.md):
  results are sensitive to the weights; cross-section w = 30 with stroke effort 0.1-0.3 is the robust
  region (Hausdorff 2.7-3.2 mm) and w = 30 + effort 0.1 is recommended for the real simulator.
- 2026-09-26 — [Cost experiment 2: cross-section term](control_reports/2026-09-26_cost_e2_cross_section/report.md):
  adding a cross-section (y, z) error term makes the MPC alternate 0°/90° hits and square the bar; best
  surrogate result (w = 100 + stroke effort 0.1): Hausdorff 2.24 vs 8.12 mm; real-simulator runs queued.
- 2026-09-26 — [Cost experiment 1: control-change penalties](control_reports/2026-09-26_cost_e1_control_change/report.md):
  penalizing ||u_k+1 − u_k|| or station changes (weights 0.01-1) changes nothing that matters; the error
  term is about length, so the bar is stretched, never squared.
- 2026-09-26 — [Cost experiment 0: original cost, first MPC run](control_reports/2026-09-26_cost_e0_original/report.md):
  stopped at hit 11 of 50. The node-distance cost rewards stretching the bar,
  so the MPC pressed one clamp-side station every hit until the mesh distorted
  and the simulator failed.

## GNN

- 2026-10-07 — [Coil GNN with temperature in the state: nodes, edges and the MPC](GNN_reports/2026-10-07_coil_T_graph_and_mpc/report.md) ([PDF](GNN_reports/2026-10-07_coil_T_graph_and_mpc/report.pdf)):
  design report. 4355 surface nodes × 16 inputs (flags, die push, distance to die, displacement, temperature, controls),
  26,118 directed edges × 8 inputs, 4 outputs (change of x, y, z, temperature); how the MPC plans 6 hits per cycle through it.
  Temperature-state MPC changes decided, not yet implemented; training sweep in progress.
- 2026-09-27 — [Absolute gap control](GNN_reports/2026-09-27_gap_control/report.md): the half-gap input is as accurate
  as the stroke input on seed 3; adding 10% no-change examples cuts predicted stretch on missed hits from 0.30 to
  0.015 mm, and both gap models predict experiment 8's first failing hits better (experiment 9's model).
- 2026-09-26 — [Pretrain-then-finetune vs. training on everything at once](GNN_reports/2026-09-26_pretrain_vs_joint/report.md):
  pretraining on uniform data then finetuning on square data is more accurate over the
  48-hit test rollout at M = 3 and M = 5 (lower error on 35-48 of 48 hits); keep it.
- 2026-09-26 — [Message-passing-steps (M) sweep, square-rod regime](GNN_reports/2026-09-26_mp_sweep/report.md):
  M = 1 is clearly worse; accuracy saturates around M = 2-3, with M = 3 the
  most accurate on Hausdorff at two-thirds of M = 5's gradient cost (each M
  trained once, so the ordering of M ≥ 2 is tentative).
