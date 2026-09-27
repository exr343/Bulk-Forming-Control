"""Single 48-hit rollout forming a square cross-section, inspired by the
collaborator's `OSU_materials/toolpaths/simple_square.jsonl` -- fine-tuning
data for the GNN surrogate in a heavily-deformed regime the random-hit
dataset never reaches.

Schedule (settled by interview, see also OSU_materials/):
  - simple_square's pattern (0/90 deg pairs per axial station, repeated
    passes at a shrinking die gap) scaled by 7.9375/19.05 to this billet,
    with 4 passes instead of 2 (half-gap 7.1 -> 6.3 -> 5.7 -> 5.3 mm; final
    ~10.6 mm square = simple_square's 25.4 mm scaled).
  - Existing 19.3 mm (0.2*H) contact band, 6 stations with d_j spread over
    [0.17, 0.78] -- starts at 0.17 rather than 0.02 so every station sits at
    >= ~800 C on the initial temperature profile (collaborator's 800-1200 C
    working window).
  - "Half-gap" = distance from the billet's current centerline to each die
    face at the end of the hit. The stroke u_j is computed per hit from the
    CURRENT deformed surface in the band (half-thickness - half-gap), clamped
    to [0, --max-stroke] mm; hits where the clamp bit are flagged `capped`.
  - Full reheat before every hit (ForgingPlant.step's thermal ramp) -- same
    as the training data; the GNN has no temperature input.

Robustness:
  - Full solver state (sol_u, sol_dT, 8 internal-variable arrays) is
    checkpointed after every hit; rerunning with the same --dataset-dir
    resumes from the last completed hit.
  - A hit that fails to converge is retried once as two half-stroke hits.
    Every failed attempt is logged to failed_hits.jsonl (never to the
    manifest); if a retry also fails, the run stops with the rollout
    truncated but still loadable.

Output (same layout GNN/data.py reads): <dataset-dir>/manifest.json +
rollout_01/{undeformed,hit_XX_final}.vtu. Each executed hit (including each
half of a split retry) is its own consecutive hit record, and
n_hits_per_rollout always equals the number of completed hits so a partial
run still loads. min/max_compression_displacement_mm stay at the original
dataset's 0.5/2.0 so u_j_frac keeps the same meaning.

--part bulge_head (T2.4-inspired, settled by a later interview): four
butted bands over the usable length, from the clamp: thin (d_j 0.18) /
square bulge (0.38) / thin (0.58) / square head (0.78). Phase 1 squares all
four to 10.6 mm with the passes above (32 hits); phase 2 thins zones 1 and 3
to 8.5 mm over three more passes, 9.9 -> 9.2 -> 8.5 mm (12 hits). 44 hits.

Finishing phase (--finish-from, settled 2026-09-27): the 48-hit schedule
leaves the bar short of 10.6 mm (each pass ends on the 90 deg hit, which
bulges the 0 deg direction back out; the 2 mm cap also bites). With
--finish-from SRC, SRC (a completed square run) is copied to --dataset-dir
and continued from its saved state: sweep the run's own stations clamp to
tip, and at each station press whichever of 0 / 90 deg is still thicker than
10.6 + --finish-tol-mm (measured along 0 / 90 deg), to the nominal 10.6 mm
(half-gap --finish-half-gap-mm, stroke from the current surface, 2 mm cap),
with the run's per-hit angle jitter (+/- --jitter-angle-deg; none for the
unjittered run). Stops when a full sweep needs no hit, or after
--finish-max-hits extra hits. Finishing records have pass = "finish".

Usage:
    python -m applications.Agility_Forge.generate_square_rollout \\
        --dataset-dir applications/Agility_Forge/data/dataset_finetuning/square
    python -m applications.Agility_Forge.generate_square_rollout --jitter-seed 1 \\
        --finish-from applications/Agility_Forge/data/dataset_finetuning/square_jitter_seed1 \\
        --dataset-dir applications/Agility_Forge/data/dataset_finetuning/finished/square_jitter_seed1
    python -m applications.Agility_Forge.generate_square_rollout --part bulge_head \\
        --dataset-dir applications/Agility_Forge/data/dataset_finetuning/bulge_head
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
from applications.Agility_Forge.hit_config import Hit
from applications.Agility_Forge.control.plant_interface import ForgingPlant, ForgingState, load_default_billet

BAND_WIDTH_FRAC = 0.2
TOTAL_TIME = 0.8
U_MIN_MM, U_MAX_MM = 0.5, 2.0  # original dataset's range -> keeps u_j_frac consistent
N_INT_VARS = 8


def build_schedule(stations, half_gaps, rotations=(0.0, 90.0), pass_offset=0, station_ids=None):
    """Pass-major: for each half-gap, sweep every station, 0 then 90 deg.
    `station_ids` labels the stations (default 1..n); `pass_offset` shifts
    pass numbering so phases can be concatenated."""
    station_ids = station_ids or list(range(1, len(stations) + 1))
    return [
        {"pass": pass_offset + p + 1, "station": sid, "d_j_frac": float(d), "R_j_deg": float(r),
         "half_gap_mm": float(g)}
        for p, g in enumerate(half_gaps)
        for sid, d in zip(station_ids, stations)
        for r in rotations
    ]


# Bulge-and-head part (T2.4-inspired; settled by interview): four butted,
# non-overlapping bands over the usable length -- thin / bulge / thin / head
# from the clamp. Phase 1 squares all four zones to 10.6 mm with the square
# run's passes; phase 2 thins only zones 1 and 3 to 8.5 mm.
BULGE_HEAD_STATIONS = [0.18, 0.38, 0.58, 0.78]
BULGE_HEAD_THIN_IDS = [1, 3]
BULGE_HEAD_PHASE1_HALF_GAPS = [7.1, 6.3, 5.7, 5.3]
BULGE_HEAD_PHASE2_HALF_GAPS = [4.95, 4.6, 4.25]


# Reference x where the initial temperature profile (676 C at the clamp face
# x=-5 rising linearly to 1096 C at x=72) crosses 800 C -- the collaborator's
# stated working-window floor.
X_800C_MM = -5.0 + 77.0 * (800.0 - 676.0) / (1096.0 - 676.0)
D_J_MAX = 0.78  # generate_dataset.py's tip margin: 1 - band_width_frac - edge_margin_frac


def jitter_square_schedule(stations, half_gaps, H, seed, station_mm, angle_deg, gap_mm):
    """Jittered-square variant (settled by interview). Amounts and their rationale:
      - station_mm (default 3.7): each station shifted independently; the
        largest shift that keeps neighbouring 19.3 mm bands overlapping for the
        11.8 mm reference spacing ((19.3 - 11.8) / 2), so no strip goes
        unpressed. Clipped to [800 C point, D_J_MAX].
      - angle_deg (default 5, user's choice): per-hit offset from 0/90 deg.
      - gap_mm (default 0.2): per-pass half-gap offset; half the smallest
        step between consecutive passes (5.7 -> 5.3), so every pass still
        ends deeper than the previous one.
    Returns (schedule, draws) -- draws recorded for reproducibility."""
    rng = onp.random.default_rng(seed)
    d_lo = X_800C_MM / H
    st = onp.clip(onp.asarray(stations) + rng.uniform(-station_mm, station_mm, len(stations)) / H, d_lo, D_J_MAX)
    gaps = onp.asarray(half_gaps) + rng.uniform(-gap_mm, gap_mm, len(half_gaps))
    schedule = build_schedule(st, gaps)
    for e in schedule:
        e["R_j_deg"] = float(e["R_j_deg"] + rng.uniform(-angle_deg, angle_deg))
    draws = {"seed": seed, "station_mm": station_mm, "angle_deg": angle_deg, "gap_mm": gap_mm,
             "stations_d_j_frac": st.tolist(), "half_gaps_mm": gaps.tolist()}
    return schedule, draws


def build_bulge_head_schedule():
    phase1 = build_schedule(BULGE_HEAD_STATIONS, BULGE_HEAD_PHASE1_HALF_GAPS)
    thin = [BULGE_HEAD_STATIONS[i - 1] for i in BULGE_HEAD_THIN_IDS]
    phase2 = build_schedule(thin, BULGE_HEAD_PHASE2_HALF_GAPS,
                            pass_offset=len(BULGE_HEAD_PHASE1_HALF_GAPS), station_ids=BULGE_HEAD_THIN_IDS)
    return phase1 + phase2


def band_half_thickness(mesh, R, H, sol_u, d_j_frac, R_j_deg, x_min_mm=None):
    """Half the current distance between the two extreme outer-surface nodes
    in the band, measured along the press direction -- the same candidate set
    and rotated frame build_cylinder_press_bcs uses to place the platens.
    `x_min_mm` (optional) drops band nodes whose original x is below it."""
    pts = onp.asarray(mesh.points)
    r_ref = onp.hypot(pts[:, 1], pts[:, 2])
    x_lo, x_hi = d_j_frac * H, (d_j_frac + BAND_WIDTH_FRAC) * H
    if x_min_mm is not None:
        x_lo = max(x_lo, x_min_mm)
    cand = (onp.abs(r_ref - R) <= 0.1 * R) & (pts[:, 0] >= x_lo) & (pts[:, 0] <= x_hi)
    rot = onp.array(Rotation.from_euler("x", R_j_deg, degrees=True).as_matrix())
    y = ((pts[cand] + onp.asarray(sol_u)[cand]) @ rot.T)[:, 1]
    return 0.5 * float(y.max() - y.min())


# Stop rule (settled 2026-09-27; the finishing phase's "converged" test, and
# the GNN-MPC runs' stop rule in control/eval_square_target.py): the part is
# done when, in every check window (the 19.3 mm die band at each of the six
# square-run stations), the bar is at most 2 * half-gap + tol across at both
# 0 and 90 deg. One-sided: thinner than the target is not flagged (forging
# can't add metal back). `x_min_mm` limits the windows to original x >= it
# (the MPC uses the target's square-section start, so the taper and the
# unpressable stretch below the 800 C point are not required to be 10.6 mm).
SQUARE_STATIONS_D_J = onp.linspace(0.17, 0.78, 6)  # the unjittered square run's stations


def across_by_station(mesh, R, H, sol_u, stations_d_j, x_min_mm=None):
    """{station number: [across at 0 deg, across at 90 deg]} in mm."""
    return {i + 1: [2.0 * band_half_thickness(mesh, R, H, sol_u, float(d), a, x_min_mm) for a in (0.0, 90.0)]
            for i, d in enumerate(stations_d_j)}


def within_tolerance(across, half_gap_mm, tol_mm):
    return all(max(v) <= 2.0 * half_gap_mm + tol_mm for v in across.values())


def save_checkpoint(path, state, completed_hits, next_sched_idx, pending_half_mm):
    tmp = path + ".tmp.npz"
    onp.savez(tmp, sol_u=onp.asarray(state.sol_u), sol_dT=onp.asarray(state.sol_dT),
              completed_hits=completed_hits, next_sched_idx=next_sched_idx,
              pending_half_mm=-1.0 if pending_half_mm is None else pending_half_mm,
              **{f"iv_{k}": onp.asarray(v) for k, v in enumerate(state.int_vars)})
    os.replace(tmp, path)


def load_checkpoint(path):
    z = onp.load(path)
    state = ForgingState(np.array(z["sol_u"]), np.array(z["sol_dT"]),
                         [np.array(z[f"iv_{k}"]) for k in range(N_INT_VARS)])
    pending = float(z["pending_half_mm"])
    return state, int(z["completed_hits"]), int(z["next_sched_idx"]), (None if pending < 0 else pending)


class Manifest:
    """Single-writer manifest in generate_dataset.py's format, rewritten
    atomically after every record."""

    def __init__(self, dataset_dir, meta):
        self.path = os.path.join(dataset_dir, "manifest.json")
        if os.path.exists(self.path):
            with open(self.path, encoding="utf-8") as f:
                self.data = json.load(f)
        else:
            self.data = {"n_hits_per_rollout": 0, "band_width_frac": BAND_WIDTH_FRAC,
                         "min_compression_displacement_mm": U_MIN_MM,
                         "max_compression_displacement_mm": U_MAX_MM,
                         "total_time_s": TOTAL_TIME, "generation_runs": [], "records": [], **meta}
        self.data["generation_runs"].append({"timestamp": datetime.now(timezone.utc).isoformat(),
                                             "slurm_job_id": os.environ.get("SLURM_JOB_ID")})
        self._write()

    def add(self, record):
        self.data["records"].append(record)
        if record["kind"] == "hit_final":
            self.data["n_hits_per_rollout"] = record["hit"]
        self._write()

    def _write(self):
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2)
        os.replace(tmp, self.path)


def log_failure(dataset_dir, entry):
    with open(os.path.join(dataset_dir, "failed_hits.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-dir", default="applications/Agility_Forge/data/dataset_finetuning/square")
    p.add_argument("--half-gaps-mm", default="7.1,6.3,5.7,5.3")
    p.add_argument("--d-j-min", type=float, default=0.17)
    p.add_argument("--d-j-max", type=float, default=0.78)
    p.add_argument("--n-stations", type=int, default=6)
    p.add_argument("--max-stroke", type=float, default=2.0)
    p.add_argument("--jitter-seed", type=int, default=None,
                   help="square only: if set, randomly perturb the schedule (see jitter_square_schedule).")
    p.add_argument("--jitter-station-mm", type=float, default=3.7)
    p.add_argument("--jitter-angle-deg", type=float, default=5.0)
    p.add_argument("--jitter-gap-mm", type=float, default=0.2)
    p.add_argument("--finish-from", default=None,
                   help="square only: copy this completed run to --dataset-dir and add the finishing phase.")
    p.add_argument("--finish-half-gap-mm", type=float, default=5.3)
    p.add_argument("--finish-tol-mm", type=float, default=0.2,
                   help="Done when every station is at most 2 * half-gap + tol across at 0 and 90 deg.")
    p.add_argument("--finish-max-hits", type=int, default=52)
    p.add_argument("--part", choices=["square", "bulge_head"], default="square",
                   help="square: uniform square along the bar (the --half-gaps/--d-j/--n-stations args). "
                        "bulge_head: fixed T2.4-inspired thin/bulge/thin/head schedule (ignores those args).")
    args = p.parse_args()

    if args.part == "square":
        stations = onp.linspace(args.d_j_min, args.d_j_max, args.n_stations)
        half_gaps = [float(g) for g in args.half_gaps_mm.split(",")]
        schedule = build_schedule(stations, half_gaps)
        schedule_meta = {"stations_d_j_frac": stations.tolist(), "half_gaps_mm": half_gaps,
                         "source": "OSU_materials/toolpaths/simple_square.jsonl (scaled)"}
    else:
        schedule = build_bulge_head_schedule()
        schedule_meta = {"stations_d_j_frac": BULGE_HEAD_STATIONS, "thin_station_ids": BULGE_HEAD_THIN_IDS,
                         "phase1_half_gaps_mm": BULGE_HEAD_PHASE1_HALF_GAPS,
                         "phase2_half_gaps_mm": BULGE_HEAD_PHASE2_HALF_GAPS,
                         "source": "OSU_materials/target_geometries/T2.4.stl (inspiration only)"}

    if args.finish_from:
        assert args.part == "square", "--finish-from is for square runs"
        if not os.path.exists(args.dataset_dir):
            shutil.copytree(args.finish_from, args.dataset_dir)
            print(f"Copied {args.finish_from} -> {args.dataset_dir}")
    rollout_dir = os.path.join(args.dataset_dir, "rollout_01")
    os.makedirs(rollout_dir, exist_ok=True)
    ckpt_path = os.path.join(rollout_dir, "checkpoint.npz")

    # load_default_billet rebuilds the mesh through shared scratch files
    # (assets/*.json, msh/tmp/tmp.vtk); serialize it so parallel jobs don't
    # read a half-written file (what crashed square_jitter_seed3's first try).
    with _locked(os.path.join(os.path.dirname(__file__), "data", "msh", "jax_forge", "tmp")):
        mesh, R, H, T_linear_fn = load_default_billet()
    R, H = float(R), float(H)
    if args.part == "square" and args.jitter_seed is not None:
        schedule, draws = jitter_square_schedule(stations, half_gaps, H, args.jitter_seed, args.jitter_station_mm,
                                                 args.jitter_angle_deg, args.jitter_gap_mm)
        schedule_meta["jitter"] = draws
        print(f"Jittered schedule (seed {args.jitter_seed}): stations {onp.round(draws['stations_d_j_frac'], 4)}, "
              f"half-gaps {onp.round(draws['half_gaps_mm'], 3)} mm")
    plant = ForgingPlant(mesh, R, H, T_linear_fn)
    manifest = Manifest(args.dataset_dir, {"part": args.part,
                                           "schedule_config": {**schedule_meta, "max_stroke_mm": args.max_stroke}})

    # Planned schedule in the collaborator's toolpath style (rho = half-gap,
    # phi = rotation, z = band center in mm from the clamped end's x=0).
    with open(os.path.join(args.dataset_dir, "planned_schedule.jsonl"), "w", encoding="utf-8") as f:
        for s in schedule:
            f.write(json.dumps({"action": {"rho": s["half_gap_mm"], "phi": s["R_j_deg"],
                                           "z": (s["d_j_frac"] + BAND_WIDTH_FRAC / 2) * H,
                                           "band_width_mm": BAND_WIDTH_FRAC * H},
                                "pass": s["pass"], "station": s["station"]}) + "\n")

    if os.path.exists(ckpt_path):
        state, completed, sched_idx, pending_half = load_checkpoint(ckpt_path)
        # Drop records for any hit that finished after the last checkpoint
        # (it will be re-run from the checkpointed state).
        manifest.data["records"] = [r for r in manifest.data["records"] if r["hit"] <= completed]
        manifest.data["n_hits_per_rollout"] = completed
        manifest._write()
        print(f"Resuming from checkpoint: {completed} hits done, schedule entry {sched_idx + 1}/{len(schedule)}"
              + (f", second half-stroke ({pending_half:.3f} mm) pending" if pending_half else ""))
    else:
        state, completed, sched_idx, pending_half = plant.reset(), 0, 0, None
        und_path = os.path.join(rollout_dir, "undeformed.vtu")
        save_sol(plant.problem.fes[0], state.sol_u, und_path,
                 point_infos=[("Displacement", state.sol_u), ("Temperature", state.sol_dT)])
        manifest.add({"rollout": 1, "hit": 0, "kind": "undeformed",
                      "vtu_path": os.path.relpath(und_path, args.dataset_dir)})
        save_checkpoint(ckpt_path, state, 0, 0, None)

    def run_hit(state, entry, u_mm, split_part):
        """Applies one hit; on success saves .vtu + manifest record and returns the new state."""
        nonlocal completed
        hit = Hit(x_min_band=entry["d_j_frac"], x_max_band=entry["d_j_frac"] + BAND_WIDTH_FRAC,
                  compression_displacement=u_mm, rotation_euler_x=entry["R_j_deg"], total_time=TOTAL_TIME)
        t0 = time.time()
        new_state = plant.step(state, hit, is_first_hit=(completed == 0))
        completed += 1
        path = os.path.join(rollout_dir, f"hit_{completed:02d}_final.vtu")
        save_sol(plant.problem.fes[0], new_state.sol_u, path,
                 point_infos=[("Displacement", new_state.sol_u), ("Temperature", new_state.sol_dT)])
        d_j = entry["d_j_frac"]
        manifest.add({
            "rollout": 1, "hit": completed, "kind": "hit_final",
            "d_j_frac": d_j, "d_j_mm": d_j * H,
            "x_max_band_frac": d_j + BAND_WIDTH_FRAC, "x_max_band_mm": (d_j + BAND_WIDTH_FRAC) * H,
            "R_j_deg": entry["R_j_deg"], "u_j_mm": u_mm, "total_time_s": TOTAL_TIME,
            "wall_time_s": time.time() - t0, "vtu_path": os.path.relpath(path, args.dataset_dir),
            "pass": entry["pass"], "station": entry["station"], "half_gap_mm": entry["half_gap_mm"],
            "schedule_idx": entry["idx"], "u_needed_mm": entry["u_needed_mm"],
            "capped": entry["capped"], "split_part": split_part,
        })
        print(f"  hit {completed} done: u={u_mm:.3f} mm, {time.time() - t0:.0f}s wall")
        return new_state

    def run_with_retry(state, entry, u, save_pending):
        """One hit; on failure, retry once as two half-strokes (logged to
        failed_hits.jsonl). `save_pending`: checkpoint the pending second
        half so a crash in between resumes it (schedule loop only)."""
        try:
            return run_hit(state, entry, u, split_part=0)
        except Exception as exc:
            log_failure(args.dataset_dir, {**entry, "u_attempted_mm": u, "attempt": "full",
                                           "after_hit": completed, "error": repr(exc),
                                           "traceback": traceback.format_exc(),
                                           "timestamp": datetime.now(timezone.utc).isoformat()})
            print(f"  [FAILED] full stroke {u:.3f} mm ({exc!r}); retrying as two half-strokes")
            half = 0.5 * u
            for part in (1, 2):
                try:
                    state = run_hit(state, entry, half, split_part=part)
                except Exception as exc2:
                    log_failure(args.dataset_dir, {**entry, "u_attempted_mm": half, "attempt": f"half_{part}",
                                                   "after_hit": completed, "error": repr(exc2),
                                                   "traceback": traceback.format_exc(),
                                                   "timestamp": datetime.now(timezone.utc).isoformat()})
                    print(f"  [FAILED] half-stroke {part}/2 also failed ({exc2!r}); stopping. "
                          f"Rollout truncated at {completed} hits (still loadable).")
                    raise SystemExit(1)
                if part == 1 and save_pending:
                    save_checkpoint(ckpt_path, state, completed, sched_idx, half)
            return state

    t_start = time.time()
    while sched_idx < len(schedule):
        entry = dict(schedule[sched_idx], idx=sched_idx)
        print("\n" + "=" * 80)
        print(f"SCHEDULE {sched_idx + 1}/{len(schedule)}: pass {entry['pass']} station {entry['station']} "
              f"R={entry['R_j_deg']:.0f} half-gap={entry['half_gap_mm']} mm")

        if pending_half is not None:
            # Crashed between the two halves of a split retry: finish the second half.
            entry.update(u_needed_mm=None, capped=False)
            state = run_hit(state, entry, pending_half, split_part=2)
            pending_half, sched_idx = None, sched_idx + 1
            save_checkpoint(ckpt_path, state, completed, sched_idx, None)
            continue

        ht = band_half_thickness(mesh, R, H, state.sol_u, entry["d_j_frac"], entry["R_j_deg"])
        u_needed = ht - entry["half_gap_mm"]
        u = float(onp.clip(u_needed, 0.0, args.max_stroke))
        entry.update(u_needed_mm=u_needed, capped=bool(u_needed > args.max_stroke))
        print(f"  current half-thickness {ht:.3f} mm -> stroke needed {u_needed:.3f} mm, applying {u:.3f} mm"
              + ("  [CAPPED]" if entry["capped"] else ""))

        state = run_with_retry(state, entry, u, save_pending=True)

        sched_idx += 1
        save_checkpoint(ckpt_path, state, completed, sched_idx, None)

    print("\n" + "#" * 80)
    print(f"SQUARE ROLLOUT COMPLETE: {completed} hits, {(time.time() - t_start) / 3600:.2f} h this job")
    capped = [r["hit"] for r in manifest.data["records"] if r.get("capped")]
    print(f"Hits limited by the {args.max_stroke} mm stroke cap: {capped or 'none'}")
    print("#" * 80)

    if not args.finish_from:
        return
    # ---- Finishing phase ----
    fin = manifest.data.setdefault("finish_config", {
        "source": args.finish_from, "half_gap_mm": args.finish_half_gap_mm, "tol_mm": args.finish_tol_mm,
        "max_hits": args.finish_max_hits, "start_hit": completed,
        "angle_jitter_deg": args.jitter_angle_deg if args.jitter_seed is not None else 0.0})
    manifest._write()
    target_mm = 2.0 * fin["half_gap_mm"] + fin["tol_mm"]
    stations_run = []
    for e in schedule:  # the run's own (possibly jittered) stations, clamp to tip
        if e["station"] not in [sid for sid, _ in stations_run]:
            stations_run.append((e["station"], e["d_j_frac"]))
    sweep, done = 0, False
    while not done:
        sweep += 1
        applied = 0
        for sid, d_j in stations_run:
            for base in (0.0, 90.0):
                if completed - fin["start_hit"] >= fin["max_hits"]:
                    print(f"Finishing stopped at the {fin['max_hits']}-hit limit.")
                    done = True
                    break
                across = 2.0 * band_half_thickness(mesh, R, H, state.sol_u, d_j, base)
                if across <= target_mm:
                    continue
                # Deterministic per-hit angle offset (reproducible across resumes).
                rng = onp.random.default_rng([args.jitter_seed or 0, completed])
                R_j = base + (rng.uniform(-fin["angle_jitter_deg"], fin["angle_jitter_deg"])
                              if fin["angle_jitter_deg"] else 0.0)
                ht = band_half_thickness(mesh, R, H, state.sol_u, d_j, R_j)
                u_needed = ht - fin["half_gap_mm"]
                u = float(onp.clip(u_needed, 0.0, args.max_stroke))
                if u < 0.05:
                    continue
                entry = {"pass": "finish", "station": sid, "d_j_frac": d_j, "R_j_deg": float(R_j),
                         "half_gap_mm": fin["half_gap_mm"], "idx": -1, "u_needed_mm": u_needed,
                         "capped": bool(u_needed > args.max_stroke)}
                print("\n" + "=" * 80)
                print(f"FINISH sweep {sweep}, extra hit {completed - fin['start_hit'] + 1}/{fin['max_hits']}: "
                      f"station {sid} {base:.0f} deg ({across:.2f} mm across > {target_mm:.2f}) R={R_j:.1f} "
                      f"stroke {u:.3f} mm" + ("  [CAPPED]" if entry["capped"] else ""))
                state = run_with_retry(state, entry, u, save_pending=False)
                save_checkpoint(ckpt_path, state, completed, sched_idx, None)
                applied += 1
            if done:
                break
        if applied == 0:
            print(f"Finished: every station within {target_mm:.2f} mm at 0 and 90 deg after sweep {sweep}.")
            done = True
    widths = {sid: [round(2.0 * band_half_thickness(mesh, R, H, state.sol_u, d_j, b), 3) for b in (0.0, 90.0)]
              for sid, d_j in stations_run}
    fin.update(end_hit=completed, n_extra_hits=completed - fin["start_hit"],
               final_across_mm_by_station={str(k): v for k, v in widths.items()},
               converged=all(max(v) <= target_mm for v in widths.values()))
    manifest._write()
    print(f"FINISHING DONE: {fin['n_extra_hits']} extra hits, converged={fin['converged']}, "
          f"across (0/90 deg) by station: {widths}")


if __name__ == "__main__":
    main()
