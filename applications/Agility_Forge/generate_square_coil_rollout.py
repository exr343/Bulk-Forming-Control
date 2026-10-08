"""Square-rod rollout on the NEW simulator: the 12.7 mm die fixed in space
(hit_config.Hit(die_center_mm=...)) and the coil reheat every 6 hits
(README "The reheat"). Training data for the GNN with temperature, aimed at
the ideal square target (control/targets/ideal_square_10.6).

Schedule (settled 2026-10-04): OSU_materials/toolpaths/simple_square.jsonl
scaled to this billet.
  - Two passes over the bar, each walking from the clamp side to the free
    end: pass 1 squeezes to half-gap 7.9375 * 15.05 / 19.05 = 6.27 mm
    (simple_square's 79%), pass 2 to 5.3 mm (the target's 10.6 mm square).
  - Each station is a 0 deg hit then a 90 deg hit. Stations are spaced
    12.7 * 6 / 7.1 = 10.73 mm (simple_square steps 6 mm with Tool2's 7.1 mm
    flat), the first centred so the die's near edge sits where the target's
    square section starts (25.23 mm). No taper hit (the user's call; the die
    edge already leaves a ~4-5 mm taper).
  - A pass ends when the next station's die would have less than
    MIN_METAL_MM of bar under it. Volume-conservation estimate: 9 + 12
    stations = 42 hits.
  - Stroke per hit = current half-thickness under the die - half-gap,
    capped at --max-stroke (2 mm, the GNN's training range); hits needing
    less than MIN_STROKE_MM are skipped (logged, not run).

Cycle (README "The reheat"): scan -> plan the next 3 stations (6 hits) ->
coil reheat centred on the midpoint of their die centres -> the hits. The
bar is reheated before hit 1 too. If a planned station turns out to have no
bar under it when reached, the cycle ends early and the next cycle starts
the next pass (the scan before it shows the pass is done).

--jitter-seed N (9 jittered copies, same rules as the old square runs, with
amounts re-derived for the 12.7 mm die): each station centre +-1.0 mm (the
largest shift that keeps neighbouring dies overlapping: (12.7 - 10.73) / 2),
each hit's angle +-5 deg, each pass's half-gap +-0.2 mm. Draws are seeded
per (pass, station) so a resumed run redraws the same values.

Output: <dataset-dir>/manifest.json (generate_dataset.py's format; reheats
under "reheats", skipped hits under "skipped", outside "records" so the GNN
loader's per-hit grouping is unaffected) + rollout_01/{undeformed,
hit_XX_final, reheat_before_hit_XX}.vtu. undeformed.vtu already carries the
first reheat's temperature. Checkpoints after every hit and reheat; rerunning
with the same --dataset-dir resumes.

Usage:
    python -m applications.Agility_Forge.generate_square_coil_rollout \\
        --dataset-dir applications/Agility_Forge/data/dataset_die12_coil/square --extra-passes 2 [--jitter-seed N]

Reheat rule (changed 2026-10-04): a reheat keeps each node's hotter temperature,
current or coil (control/plant_interface.reheat_temperature); the first reheat,
on the fresh billet, is the coil profile. --overwrite-reheat restores the first
rule (set every node to the coil profile), which cooled far zones of long bars
by ~300 C and failed at pass starts; the --continue-from / --reverse-passes /
--reverse-failed-pass options below were workarounds for it (historical; that
data is in data/backup/die12_coil_overwrite_reheat/).

Continuing a finished run (settled 2026-10-04: the 2-pass runs ended ~2.5 mm
too thick and ~31 mm short of the target): --continue-from SRC --extra-passes
N copies SRC to --dataset-dir and adds N passes at the final half-gap (5.3 mm),
same stations/jitter rules; passes 1-2 keep their original values.

Reversed passes (settled 2026-10-04): the first reheat of a forward pass moves
the coil from the free end back to the clamp end, and the overwrite rule cools
the far end by ~300 C; the solver failed there in 4 of 10 runs. --reverse-passes
N runs pass N from the free end back toward the clamp (the coil then moves
~32 mm per cycle, as within a pass); --reverse-failed-pass does this
automatically on resume for a run that stopped on a pass-start reheat.
    python -m applications.Agility_Forge.generate_square_coil_rollout --extra-passes 2 \\
        --continue-from applications/Agility_Forge/data/dataset_die12_coil/square \\
        --dataset-dir applications/Agility_Forge/data/dataset_die12_coil_4pass/square
"""

import argparse
import json
import os
import shutil
import time
import traceback
from datetime import datetime, timezone

import numpy as onp
import jax.numpy as np
from scipy.spatial.transform import Rotation

from jax_forge.utils import save_sol

from applications.Agility_Forge.generate_dataset import _locked
from applications.Agility_Forge.generate_square_rollout import Manifest, log_failure, N_INT_VARS, TOTAL_TIME
from applications.Agility_Forge.hit_config import Hit, DIE_WIDTH_MM
from applications.Agility_Forge.control.plant_interface import (
    ForgingPlant, ForgingState, load_default_billet, coil_temperature, reheat_temperature,
    COIL_T_C, COIL_LENGTH_MM, COIL_SLOPE_C_PER_MM,
)

R0_MM = 7.9375                                  # billet radius
STATION_STEP_MM = DIE_WIDTH_MM * 6.0 / 7.1      # simple_square: 6 mm step, 7.1 mm flat
TARGET_SPEC = os.path.join(os.path.dirname(__file__), "control", "targets", "ideal_square_10.6", "target_spec.json")
PASS_HALF_GAPS_MM = [R0_MM * 15.05 / 19.05, 5.3]
HITS_PER_CYCLE = 6
STATIONS_PER_CYCLE = HITS_PER_CYCLE // 2
MIN_METAL_MM = 3.0
MIN_STROKE_MM = 0.05


def first_station_mm():
    with open(TARGET_SPEC, encoding="utf-8") as f:
        return json.load(f)["square_section_mm"][0] + 0.5 * DIE_WIDTH_MM


class Schedule:
    """Station positions, angles and half-gaps, with optional seeded jitter."""

    def __init__(self, jitter_seed, station_mm, angle_deg, gap_mm, extra_passes=0, last_pass_half_gap=None):
        self.seed, self.station_mm, self.angle_deg, self.gap_mm = jitter_seed, station_mm, angle_deg, gap_mm
        self.x0 = first_station_mm()
        # Extra passes repeat the final half-gap. Jitter draws are sequential
        # per pass, so passes 1-2 keep the values of the original runs.
        self.half_gaps = list(PASS_HALF_GAPS_MM) + [PASS_HALF_GAPS_MM[-1]] * extra_passes
        if jitter_seed is not None:
            rng = onp.random.default_rng([jitter_seed, 999])
            self.half_gaps = [g + rng.uniform(-gap_mm, gap_mm) for g in self.half_gaps]
        if last_pass_half_gap is not None:      # exact, no jitter (drawn above, so earlier passes keep theirs)
            self.half_gaps[-1] = float(last_pass_half_gap)

    def _rng(self, *key):
        return onp.random.default_rng([self.seed, *key])

    def center(self, p, k):
        c = self.x0 + k * STATION_STEP_MM
        if self.seed is not None:
            c += self._rng(p, k, 0).uniform(-self.station_mm, self.station_mm)
        return float(c)

    def angle(self, p, k, base):
        if self.seed is None:
            return float(base)
        return float(base + self._rng(p, k, 1, int(base)).uniform(-self.angle_deg, self.angle_deg))

    def hits(self, p, k):
        c = self.center(p, k)
        return [{"pass": p + 1, "station": k + 1, "die_center_mm": c, "R_j_deg": self.angle(p, k, base),
                 "half_gap_mm": float(self.half_gaps[p])} for base in (0.0, 90.0)]


def current_tip_mm(pts, sol_u):
    return float((pts[:, 0] + onp.asarray(sol_u)[:, 0]).max())


def die_half_thickness(pts, surf, sol_u, center, R_j_deg):
    """Half the distance between the extreme outer-surface nodes currently
    under the die, along the press direction -- the same node set and
    rotated frame build_cylinder_press_bcs uses to place the dies."""
    u = onp.asarray(sol_u)
    x = pts[:, 0] + u[:, 0]
    cand = surf & (x >= center - 0.5 * DIE_WIDTH_MM) & (x <= center + 0.5 * DIE_WIDTH_MM)
    rot = onp.array(Rotation.from_euler("x", R_j_deg, degrees=True).as_matrix())
    y = ((pts[cand] + u[cand]) @ rot.T)[:, 1]
    return 0.5 * float(y.max() - y.min())


def temperature_under_die(pts, sol_u, sol_dT, center):
    x = pts[:, 0] + onp.asarray(sol_u)[:, 0]
    m = (x >= center - 0.5 * DIE_WIDTH_MM) & (x <= center + 0.5 * DIE_WIDTH_MM)
    T = onp.asarray(sol_dT)[m, 0]
    return float(T.min()), float(T.mean()), float(T.max())


def save_checkpoint(path, state, sched):
    tmp = path + ".tmp.npz"
    onp.savez(tmp, sol_u=onp.asarray(state.sol_u), sol_dT=onp.asarray(state.sol_dT),
              sched=json.dumps(sched), **{f"iv_{k}": onp.asarray(v) for k, v in enumerate(state.int_vars)})
    os.replace(tmp, path)


def load_checkpoint(path):
    z = onp.load(path)
    state = ForgingState(np.array(z["sol_u"]), np.array(z["sol_dT"]),
                         [np.array(z[f"iv_{k}"]) for k in range(N_INT_VARS)])
    return state, json.loads(str(z["sched"]))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-dir", default="applications/Agility_Forge/data/dataset_die12_coil/square")
    p.add_argument("--max-stroke", type=float, default=2.0)
    p.add_argument("--jitter-seed", type=int, default=None)
    p.add_argument("--jitter-station-mm", type=float, default=1.0)
    p.add_argument("--jitter-angle-deg", type=float, default=5.0)
    p.add_argument("--jitter-gap-mm", type=float, default=0.2)
    p.add_argument("--extra-passes", type=int, default=0,
                   help="Passes after the two scaled simple_square passes, each at the final half-gap (5.3 mm).")
    p.add_argument("--continue-from", default=None,
                   help="Copy this finished run to --dataset-dir (if it isn't there yet) and continue it from its "
                        "saved final state; use with --extra-passes.")
    p.add_argument("--reverse-passes", default="",
                   help="Comma-separated pass numbers (e.g. 3) that run from the free end back toward the clamp, "
                        "so the coil doesn't jump from the free end back to the clamp end at the pass boundary. "
                        "On resume, a not-yet-started cycle planned for such a pass is replanned.")
    p.add_argument("--overwrite-reheat", action="store_true",
                   help="Old rule (data in data/backup/die12_coil_overwrite_reheat): a reheat sets every node to the coil "
                        "profile, cooling far zones. Default: keep the hotter of current and coil temperature.")
    p.add_argument("--reverse-failed-pass", action="store_true",
                   help="On resume: if the run stopped on the reheat that starts a pass (first station not yet "
                        "hit), rerun that pass in reverse.")
    p.add_argument("--last-pass-half-gap", type=float, default=None,
                   help="Half-gap (mm) of the final pass, exact (no jitter). Pass 5 of the 5-pass runs uses "
                        "5.0-5.3 mm across the 10 runs, so the data spans just above to slightly past the target.")
    p.add_argument("--linear-solver", choices=["jax", "scipy"], default="jax",
                   help="jax = unpreconditioned BiCGSTAB (default); scipy = direct sparse solve. Use scipy when "
                        "late-run reheats fail: BiCGSTAB stops converging there (output/reheat_failure_diag/).")
    args = p.parse_args()
    reverse_passes = {int(v) for v in args.reverse_passes.split(",") if v.strip()}

    if args.continue_from and not os.path.exists(args.dataset_dir):
        shutil.copytree(args.continue_from, args.dataset_dir)
        print(f"Copied {args.continue_from} -> {args.dataset_dir}")
    rollout_dir = os.path.join(args.dataset_dir, "rollout_01")
    os.makedirs(rollout_dir, exist_ok=True)
    ckpt_path = os.path.join(rollout_dir, "checkpoint.npz")

    # load_default_billet rebuilds the mesh through shared scratch files;
    # serialize it so parallel jobs don't read a half-written file.
    with _locked(os.path.join(os.path.dirname(__file__), "data", "msh", "jax_forge", "tmp")):
        mesh, R, H, T_linear_fn = load_default_billet()
    R, H = float(R), float(H)
    pts = onp.asarray(mesh.points)
    surf = onp.abs(onp.hypot(pts[:, 1], pts[:, 2]) - R) <= 0.1 * R   # the BC's outer-surface test
    plant = ForgingPlant(mesh, R, H, T_linear_fn, linear_solver=args.linear_solver)
    sch = Schedule(args.jitter_seed, args.jitter_station_mm, args.jitter_angle_deg, args.jitter_gap_mm,
                   args.extra_passes, args.last_pass_half_gap)

    manifest = Manifest(args.dataset_dir, {
        "part": "square_coil",
        "simulator": {"die": f"spatial flat die, {DIE_WIDTH_MM} mm wide (Hit.die_center_mm)",
                      "reheat": {"every_hits": HITS_PER_CYCLE, "coil_T_C": COIL_T_C, "coil_length_mm": COIL_LENGTH_MM,
                                 "slope_C_per_mm": COIL_SLOPE_C_PER_MM, "also_before_hit_1": True,
                                 "centre": "midpoint of the cycle's planned die centres, current x",
                                 "rule": "overwrite with the coil profile" if args.overwrite_reheat else
                                         "keep the hotter of current and coil temperature (first reheat: coil profile)"}},
        "schedule_config": {"source": "OSU_materials/toolpaths/simple_square.jsonl (scaled, no taper hit)",
                            "first_station_mm": sch.x0, "station_step_mm": STATION_STEP_MM,
                            "pass_half_gaps_mm": sch.half_gaps, "min_metal_mm": MIN_METAL_MM,
                            "min_stroke_mm": MIN_STROKE_MM, "max_stroke_mm": args.max_stroke,
                            "jitter": None if args.jitter_seed is None else {
                                "seed": args.jitter_seed, "station_mm": args.jitter_station_mm,
                                "angle_deg": args.jitter_angle_deg, "gap_mm": args.jitter_gap_mm}}})
    manifest.data.setdefault("reheats", [])
    manifest.data.setdefault("skipped", [])
    manifest.data["band_width_frac"] = None   # spatial die: no band
    manifest.data["generation_runs"][-1]["linear_solver"] = args.linear_solver
    if args.last_pass_half_gap is not None:
        manifest.data["schedule_config"]["last_pass_half_gap_mm"] = args.last_pass_half_gap
    manifest.data["schedule_config"]["pass_half_gaps_mm"] = sch.half_gaps
    if reverse_passes:
        manifest.data["schedule_config"]["reversed_passes"] = sorted(reverse_passes)
    if args.continue_from and not any(c["source"] == args.continue_from for c in manifest.data.get("continuations", [])):
        manifest.data.setdefault("continuations", []).append({
            "source": args.continue_from, "extra_passes": args.extra_passes,
            "start_hit": max(r["hit"] for r in manifest.data["records"])})
    manifest._write()

    if os.path.exists(ckpt_path):
        state, sched = load_checkpoint(ckpt_path)
        done_hits = sched["completed"]
        for key in ("records", "reheats", "skipped"):
            manifest.data[key] = [r for r in manifest.data[key] if r["hit"] <= done_hits
                                  or (key == "reheats" and r["hit"] == done_hits + 1 and sched["reheated"])]
        manifest.data["n_hits_per_rollout"] = done_hits
        manifest._write()
        print(f"Resuming: {done_hits} hits done, cycle {sched['cycle']}, pass {sched['pass_idx'] + 1}, "
              f"next station {sched['next_k'] + 1}")
        plan = sched["plan"]
        reverse_passes |= set(sched.get("reversed_passes", []))
        if (args.reverse_failed_pass and plan and sched["pos"] == 0 and not sched["reheated"]
                and plan[0]["station"] == 1 and plan[0]["pass"] not in reverse_passes):
            reverse_passes.add(plan[0]["pass"])
        sched["reversed_passes"] = sorted(reverse_passes)
        if reverse_passes:
            manifest.data["schedule_config"]["reversed_passes"] = sorted(reverse_passes)
            manifest._write()
        if (plan and sched["pos"] == 0 and not sched["reheated"] and plan[0]["pass"] in reverse_passes
                and plan[0]["station"] == 1):   # a forward plan for a pass that is now reversed
            # A cycle planned forward for a pass that is now reversed (its
            # reheat failed before any hit): drop it and replan the pass.
            sched.update(pass_idx=plan[0]["pass"] - 1, next_k=0, rev_k=None, plan=[], pos=0,
                         cycle=sched["cycle"] - 1)
            print(f"Replanning pass {plan[0]['pass']} in reverse (free end -> clamp)")
    else:
        state = plant.reset()
        sched = {"completed": 0, "pass_idx": 0, "next_k": 0, "cycle": 0, "plan": [], "pos": 0,
                 "reheated": False, "coil_center_mm": None, "pending_half_mm": None}
        save_checkpoint(ckpt_path, state, sched)

    def end_pass():
        sched["pass_idx"] += 1
        sched["next_k"] = 0
        sched["rev_k"] = None

    def plan_cycle():
        """The scan: skip passes whose next station already has no bar under
        it, then plan that pass's next STATIONS_PER_CYCLE stations."""
        tip = current_tip_mm(pts, state.sol_u)

        def has_bar(p, k):
            return sch.center(p, k) - 0.5 * DIE_WIDTH_MM <= tip - MIN_METAL_MM

        while sched["pass_idx"] < len(sch.half_gaps):
            p = sched["pass_idx"]
            if p + 1 in reverse_passes:
                if sched.get("rev_k") is None:   # pass starts at the outermost station with bar under it
                    k = 0
                    while has_bar(p, k + 1):
                        k += 1
                    sched["rev_k"] = k
                if sched["rev_k"] >= 0:
                    break
                print(f"[scan] pass {p + 1} done (reversed, reached station 1)")
                end_pass()
                continue
            if has_bar(p, sched["next_k"]):
                break
            print(f"[scan] pass {p + 1} done: station {sched['next_k'] + 1} at {sch.center(p, sched['next_k']):.2f} mm "
                  f"is past the free end ({tip:.2f} mm)")
            end_pass()
        if sched["pass_idx"] >= len(sch.half_gaps):
            return False
        p = sched["pass_idx"]
        if p + 1 in reverse_passes:
            ks = list(range(sched["rev_k"], max(sched["rev_k"] - STATIONS_PER_CYCLE, -1), -1))
            sched["rev_k"] -= len(ks)
        else:
            ks = list(range(sched["next_k"], sched["next_k"] + STATIONS_PER_CYCLE))
            sched["next_k"] += STATIONS_PER_CYCLE
        plan = []
        for k in ks:
            plan += sch.hits(p, k)
        centres = [h["die_center_mm"] for h in plan]
        sched.update(plan=plan, pos=0, reheated=False, cycle=sched["cycle"] + 1,
                     coil_center_mm=0.5 * (min(centres) + max(centres)))
        print(f"\n[scan] cycle {sched['cycle']}: pass {sched['pass_idx'] + 1}, stations "
              f"{sorted({h['station'] for h in plan})}, die centres {onp.round(sorted(set(centres)), 2)} mm, "
              f"coil centre {sched['coil_center_mm']:.2f} mm (free end {tip:.2f} mm)")
        return True

    def do_reheat():
        nonlocal state
        # First reheat (fresh billet): the coil profile. Later reheats keep
        # each node's hotter temperature, current or coil (rule settled
        # 2026-10-04); --overwrite-reheat restores the old overwrite.
        if sched["completed"] == 0 or args.overwrite_reheat:
            T_new = coil_temperature(pts, state.sol_u, sched["coil_center_mm"])
        else:
            T_new = reheat_temperature(pts, state.sol_u, state.sol_dT, sched["coil_center_mm"])
        nxt = sched["completed"] + 1
        t0 = time.time()
        if sched["completed"] == 0:
            # Undeformed, stress-free bar: set the temperature directly (the
            # first hit starts from it, as the old first hit did).
            state = ForgingState(state.sol_u, np.array(T_new), state.int_vars)
            path, info = os.path.join(rollout_dir, "undeformed.vtu"), {"max_jump_C": None, "n_ramp": 0}
        else:
            first = sched["plan"][sched["pos"]]
            hit = Hit(die_center_mm=first["die_center_mm"], compression_displacement=0.0,
                      rotation_euler_x=first["R_j_deg"], total_time=TOTAL_TIME)
            # Finer ramp on each retry; the last failure propagates (the run
            # resumes from the pre-reheat checkpoint).
            steps_C = (50.0, 25.0, 12.5)
            for i, step_C in enumerate(steps_C):
                try:
                    state = plant.reheat(state, T_new, hit, max_ramp_step_C=step_C)
                    break
                except RuntimeError as exc:
                    log_failure(args.dataset_dir, {"kind": "reheat", "before_hit": nxt, "attempt": f"{step_C:g} C/step",
                                                   "error": repr(exc), "traceback": traceback.format_exc(),
                                                   "timestamp": datetime.now(timezone.utc).isoformat()})
                    if i == len(steps_C) - 1:
                        raise
                    print(f"  [FAILED] reheat ({exc!r}); retrying with a {steps_C[i + 1]:g} C/step ramp")
            info = plant.last_reheat_info
            path = os.path.join(rollout_dir, f"reheat_before_hit_{nxt:02d}.vtu")
        save_sol(plant.problem.fes[0], state.sol_u, path,
                 point_infos=[("Displacement", state.sol_u), ("Temperature", state.sol_dT)])
        if sched["completed"] == 0:
            manifest.data["records"] = [r for r in manifest.data["records"] if r["hit"] != 0]
            manifest.add({"rollout": 1, "hit": 0, "kind": "undeformed",
                          "vtu_path": os.path.relpath(path, args.dataset_dir)})
        manifest.data["reheats"].append({
            "rollout": 1, "hit": nxt, "cycle": sched["cycle"], "coil_center_mm": sched["coil_center_mm"],
            "vtu_path": os.path.relpath(path, args.dataset_dir), "wall_time_s": time.time() - t0, **info})
        manifest._write()
        sched["reheated"] = True
        save_checkpoint(ckpt_path, state, sched)

    def run_hit(entry, u_mm, split_part):
        nonlocal state
        hit = Hit(die_center_mm=entry["die_center_mm"], compression_displacement=u_mm,
                  rotation_euler_x=entry["R_j_deg"], total_time=TOTAL_TIME)
        T_min, T_mean, T_max = temperature_under_die(pts, state.sol_u, state.sol_dT, entry["die_center_mm"])
        t0 = time.time()
        new_state = plant.step(state, hit, is_first_hit=(sched["completed"] == 0), reheat=False)
        sched["completed"] += 1
        n = sched["completed"]
        path = os.path.join(rollout_dir, f"hit_{n:02d}_final.vtu")
        save_sol(plant.problem.fes[0], new_state.sol_u, path,
                 point_infos=[("Displacement", new_state.sol_u), ("Temperature", new_state.sol_dT)])
        manifest.add({
            "rollout": 1, "hit": n, "kind": "hit_final", "cycle": sched["cycle"],
            "die_center_mm": entry["die_center_mm"], "die_width_mm": DIE_WIDTH_MM,
            "R_j_deg": entry["R_j_deg"], "u_j_mm": u_mm, "total_time_s": TOTAL_TIME,
            "pass": entry["pass"], "station": entry["station"], "half_gap_mm": entry["half_gap_mm"],
            "u_needed_mm": entry.get("u_needed_mm"), "capped": entry.get("capped", False),
            "split_part": split_part, "T_under_die_start_C": {"min": T_min, "mean": T_mean, "max": T_max},
            "free_end_mm": current_tip_mm(pts, new_state.sol_u),
            "wall_time_s": time.time() - t0, "vtu_path": os.path.relpath(path, args.dataset_dir)})
        print(f"  hit {n} done: u={u_mm:.3f} mm, T under die at start {T_min:.0f}-{T_max:.0f} C, "
              f"{time.time() - t0:.0f}s wall")
        state = new_state

    def run_with_retry(entry, u):
        """One hit; on failure, retry once as two half-strokes (logged)."""
        try:
            return run_hit(entry, u, split_part=0)
        except Exception as exc:
            log_failure(args.dataset_dir, {**entry, "u_attempted_mm": u, "attempt": "full",
                                           "after_hit": sched["completed"], "error": repr(exc),
                                           "traceback": traceback.format_exc(),
                                           "timestamp": datetime.now(timezone.utc).isoformat()})
            print(f"  [FAILED] full stroke {u:.3f} mm ({exc!r}); retrying as two half-strokes")
        half = 0.5 * u
        for part in (1, 2):
            try:
                run_hit(entry, half, split_part=part)
            except Exception as exc2:
                log_failure(args.dataset_dir, {**entry, "u_attempted_mm": half, "attempt": f"half_{part}",
                                               "after_hit": sched["completed"], "error": repr(exc2),
                                               "traceback": traceback.format_exc(),
                                               "timestamp": datetime.now(timezone.utc).isoformat()})
                print(f"  [FAILED] half-stroke {part}/2 also failed ({exc2!r}); stopping. "
                      f"Rollout truncated at {sched['completed']} hits (still loadable).")
                raise SystemExit(1)
            if part == 1:
                sched["pending_half_mm"] = half
                save_checkpoint(ckpt_path, state, sched)
        sched["pending_half_mm"] = None

    t_start = time.time()
    while True:
        if sched["pending_half_mm"] is not None:
            # Crashed between the two halves of a split retry: finish the second half.
            run_hit(sched["plan"][sched["pos"]], sched["pending_half_mm"], split_part=2)
            sched["pending_half_mm"] = None
            sched["pos"] += 1
            save_checkpoint(ckpt_path, state, sched)
            continue
        if sched["pos"] >= len(sched["plan"]):
            if not plan_cycle():
                break
            save_checkpoint(ckpt_path, state, sched)
        if not sched["reheated"]:
            do_reheat()

        entry = dict(sched["plan"][sched["pos"]])
        print("\n" + "=" * 80)
        print(f"HIT {sched['completed'] + 1} (cycle {sched['cycle']}): pass {entry['pass']} station "
              f"{entry['station']} at {entry['die_center_mm']:.2f} mm, R={entry['R_j_deg']:.1f} deg, "
              f"half-gap {entry['half_gap_mm']:.3f} mm")
        tip = current_tip_mm(pts, state.sol_u)
        x_now = pts[:, 0] + onp.asarray(state.sol_u)[:, 0]
        # The bulged end face can reach a few mm past the round side surface,
        # leaving no side-surface node under a die that passes the tip test.
        no_side = not onp.any(surf & (onp.abs(x_now - entry["die_center_mm"]) <= 0.5 * DIE_WIDTH_MM))
        if entry["die_center_mm"] - 0.5 * DIE_WIDTH_MM > tip - MIN_METAL_MM or no_side:
            print(f"  station is past the free end ({tip:.2f} mm" + (", no side surface under the die" if no_side else "")
                  + f"): pass {entry['pass']} done, cycle ends early")
            manifest.data["skipped"].append({"rollout": 1, "hit": sched["completed"], **entry,
                                             "reason": "no side surface under the die" if no_side else "past free end",
                                             "free_end_mm": tip})
            manifest._write()
            sched["pos"] = len(sched["plan"])
            if sched["pass_idx"] == entry["pass"] - 1:
                end_pass()
            save_checkpoint(ckpt_path, state, sched)
            continue

        ht = die_half_thickness(pts, surf, state.sol_u, entry["die_center_mm"], entry["R_j_deg"])
        u_needed = ht - entry["half_gap_mm"]
        u = float(onp.clip(u_needed, 0.0, args.max_stroke))
        entry.update(u_needed_mm=u_needed, capped=bool(u_needed > args.max_stroke))
        print(f"  half-thickness {ht:.3f} mm -> stroke needed {u_needed:.3f} mm, applying {u:.3f} mm"
              + ("  [CAPPED]" if entry["capped"] else ""))
        if u < MIN_STROKE_MM:
            print("  already at or below the half-gap: skipped")
            manifest.data["skipped"].append({"rollout": 1, "hit": sched["completed"], **entry,
                                             "reason": "stroke below minimum"})
            manifest._write()
        else:
            run_with_retry(entry, u)
        sched["pos"] += 1
        save_checkpoint(ckpt_path, state, sched)

    print("\n" + "#" * 80)
    print(f"SQUARE COIL ROLLOUT COMPLETE: {sched['completed']} hits, {sched['cycle']} cycles, "
          f"{(time.time() - t_start) / 3600:.2f} h this job, free end {current_tip_mm(pts, state.sol_u):.2f} mm")
    capped = [r["hit"] for r in manifest.data["records"] if r.get("capped")]
    print(f"Hits limited by the {args.max_stroke} mm stroke cap: {capped or 'none'}")
    print("#" * 80)


if __name__ == "__main__":
    main()
