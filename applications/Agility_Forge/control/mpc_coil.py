"""MPC for the NEW simulator (12.7 mm die fixed in space, coil reheat every 6
hits, a reheat keeps the hotter of current and coil temperature), using the
coil GNN with TEMPERATURE IN THE NODAL STATE (../GNN/coil_T.py). Settled with
the user on 2026-10-04, temperature state and the items below on 2026-10-07/08.

One cycle = scan -> plan 6 hits -> reheat -> apply ALL 6 hits to the
simulator -> scan again. Planning (18 variables: die centre, angle, stroke for
each of the 6 hits), since 2026-10-08 by random shooting + top-K refinement
(RoboCraft, Shi et al. RSS 2022, Sec. III-E.2; see Planner): N plans drawn
uniformly in the bounds are scored through the GNN in batches without
gradients, and the K best are each refined by the single-shooting SQP
(scipy SLSQP, unchanged settings) independently; the best refined plan wins.
  - the scan reads the bar's surface shape AND its real surface temperature
    (an IR camera in practice);
  - the coil sits at the AVERAGE of the 6 planned die centres (smooth, so the
    optimizer gets a gradient; moving a hit moves the coil);
  - forecast: the reheat rule (hotter of the measured temperature and the coil
    profile; cycle 1, a fresh billet: the coil profile) applied at the start,
    the GNN's reheat step for its shape change (not for cycle 1), then 6 GNN
    hit steps, each predicting shape AND temperature, the predicted temperature
    carried from hit to hit (smooth max and smooth plateau edge while
    planning, exact for the stored forecast);
  - cost: cross-section (y, z) distance of every surface node to the ideal
    square target, summed over the 6 forecast shapes (as in e14; length is
    left to volume conservation); temperature is not in the cost;
  - bounds: die centre from 31.58 mm (the data runs' first station, die edge
    at the target's square start) to the scanned free end minus half the die
    and 3 mm; angle 0-180 deg; stroke 0.5-2 mm. (The +-11 mm coil-zone
    constraint of the first run was removed on 2026-10-08.)
Stops when every width from the first station (31.58 mm) to 4 mm before the
free end is within 10.6 + 0.2 mm (same width measure as the progress
figures), or after 20 cycles. The x ~ 27 mm spot next to the taper is not
checked: no data run ever got it below 11-13 mm. The simulator uses the
direct linear solver (late reheats fail with the iterative one).

The run is saved in generate_square_coil_rollout.py's manifest format
(records/reheats + one .vtu per hit and reheat), so MPC-visited hits can be
added to GNN training later. Checkpoints after every hit and reheat; rerun
the same command to resume.

    python -m applications.Agility_Forge.control.mpc_coil --checkpoint <coil_T GNN .pt> --out-dir <dir>
"""

import argparse
import json
import os
import time
import traceback
from datetime import datetime, timezone

import numpy as onp
import torch

# Bit-for-bit repeatable GPU sums (the same seed gives the same plan; without it
# SLSQP amplifies run-to-run float differences into different plans; ~20% slower).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
torch.use_deterministic_algorithms(True)
import torch.nn.functional as Fn
from scipy.optimize import minimize

from applications.Agility_Forge.GNN.coil_T import (Hit as GHit, Mesh, load as load_gnn, node_features, DIE_W, SOFT_MM,
                                                   U_MIN, U_RANGE, DT_SCALE)
from applications.Agility_Forge.hit_config import Hit, DIE_WIDTH_MM
from applications.Agility_Forge.control.plant_interface import (
    COIL_LENGTH_MM, COIL_SLOPE_C_PER_MM, COIL_T_C, ForgingPlant, ForgingState, coil_temperature,
    load_default_billet, reheat_temperature)

AF = "applications/Agility_Forge"
TARGET_VTU = f"{AF}/control/targets/ideal_square_10.6/target_on_billet.vtu"
TARGET_SPEC = f"{AF}/control/targets/ideal_square_10.6/target_spec.json"
MESH_VTU = f"{AF}/data/dataset_pretraining/rollout_01/undeformed.vtu"
HITS_PER_CYCLE = 6
STATION_STEP_MM = DIE_WIDTH_MM * 6.0 / 7.1     # only for the initial guess (the data runs' station spacing)
U_MIN_MM, U_MAX_MM = 0.5, 2.0
ANGLE_RANGE = (0.0, 180.0)
MIN_METAL_MM = 3.0
TOTAL_TIME_S = 0.8
N_INT_VARS = 8


# ----------------------------------------------------------------------------
# Reheat temperature (surface nodes), the simulator's rule
# ----------------------------------------------------------------------------
def coil_profile(x_cur, c, smooth):
    """Coil temperature at current x; `smooth` softens the plateau edge (~0.5 mm)."""
    over = torch.abs(x_cur - c) - 0.5 * COIL_LENGTH_MM
    d = Fn.softplus(over, beta=2.0) if smooth else torch.relu(over)
    return COIL_T_C - COIL_SLOPE_C_PER_MM * d


def hotter(a, b, smooth):
    return 0.5 * (a + b + torch.sqrt((a - b) ** 2 + 25.0)) if smooth else torch.maximum(a, b)


def reheat_T(mesh, x_scan, T_scan, c, first, smooth):
    """Temperature after the reheat: the coil profile on a fresh billet (cycle 1),
    else the hotter of the measured temperature and the coil profile."""
    coil = coil_profile(mesh.rest[:, 0] + x_scan[:, 0], c, smooth)
    return coil if first else hotter(T_scan, coil, smooth)


# ----------------------------------------------------------------------------
# Forecast and cost
# ----------------------------------------------------------------------------
def forecast(model, mesh, x_scan, T_scan, ctrl, first, smooth=True):
    """ctrl: (6, 3) tensor [die centre mm, angle deg, stroke mm]; x_scan, T_scan:
    the scanned surface displacement and measured surface temperature. Returns
    the 6 forecast displacements and the 6 forecast temperatures (after each hit)."""
    c = ctrl[:, 0].mean()
    T = reheat_T(mesh, x_scan, T_scan, c, first, smooth)
    x = x_scan
    if not first:
        x = model.step_shape(node_features(mesh, x, T, None, dT=T - T_scan)[None], x[None])[0]
    shapes, temps = [], []
    for k in range(ctrl.shape[0]):
        h = GHit(ctrl[k, 1], ctrl[k, 2], center=ctrl[k, 0])
        xn, Tn = model.step(node_features(mesh, x, T, h)[None], x[None], T[None])
        x, T = xn[0], Tn[0]
        shapes.append(x); temps.append(T)
    return shapes, temps


def cross_section_cost(shapes, x_tgt):
    return sum(((x - x_tgt)[:, 1:] ** 2).sum() for x in shapes)


# ----------------------------------------------------------------------------
# Batched forecast for screening (many plans at once, no per-sample loop)
# ----------------------------------------------------------------------------
def node_features_batch(mesh, x, T, c, a_deg, u):
    """Hit-step inputs for a batch: x (B, N, 3), T (B, N), c/a_deg/u (B,) ->
    (B, N, 16). The same numbers as GNN.coil_T.node_features computes for each
    sample (checked in control/ablation_multistart.py)."""
    B, n = x.shape[0], mesh.n
    pos = mesh.rest[None] + x
    xs = pos[..., 0]
    lo, hi = (c - 0.5 * DIE_W)[:, None], (c + 0.5 * DIE_W)[:, None]
    a = torch.deg2rad(a_deg)[:, None]
    y = torch.cos(a) * pos[..., 1] - torch.sin(a) * pos[..., 2]
    wall = mesh.wall[None].expand(B, n)
    hard = wall & (xs >= lo) & (xs <= hi)
    d = (xs - 0.5 * (lo + hi)).abs().detach()              # fallback: nearest side-wall nodes (as node_features)
    dmin = torch.where(wall, d, torch.full_like(d, float("inf"))).min(dim=1, keepdim=True).values
    hard = torch.where(hard.any(dim=1, keepdim=True), hard, wall & (d <= dmin + 1.0))
    soft = wall.float() * torch.sigmoid((xs - lo) / SOFT_MM) * torch.sigmoid((hi - xs) / SOFT_MM)
    inf = torch.full_like(y, float("inf"))
    y_lo = torch.where(hard, y, inf).min(dim=1, keepdim=True).values
    y_hi = torch.where(hard, y, -inf).max(dim=1, keepdim=True).values
    uu = u[:, None]
    mag = (torch.relu(y_lo + uu - y) - torch.relu(y - (y_hi - uu))) * soft
    e = torch.stack([torch.zeros_like(a), torch.cos(a), -torch.sin(a)], dim=-1)          # (B, 1, 3)
    push = mag[..., None] * e
    dist = (xs - 0.5 * (lo + hi))[..., None]
    tail = torch.cat([torch.zeros_like(a), torch.sin(a), torch.cos(a), (uu - U_MIN) / U_RANGE], dim=-1)  # (B, 4)
    return torch.cat([mesh.flags[None].expand(B, n, 3), torch.zeros(B, n, 1, device=x.device), push, dist, x,
                      T[..., None], tail[:, None, :].expand(B, n, 4)], dim=-1)


def forecast_cost_batch(model, mesh, x_scan, T_scan, ctrl, first, x_tgt):
    """ctrl (B, 6, 3) -> the existing cost (cross-section error summed over the 6
    forecast shapes) for each plan, (B,). Same rules as forecast(..., smooth=True)."""
    B = ctrl.shape[0]
    c = ctrl[:, :, 0].mean(dim=1)
    x = x_scan[None].expand(B, -1, -1)
    coil = coil_profile(mesh.rest[None, :, 0] + x[..., 0], c[:, None], True)
    T = coil if first else hotter(T_scan[None], coil, True)
    if not first:
        f = torch.cat([mesh.flags[None].expand(B, -1, -1), torch.ones(B, mesh.n, 1, device=x.device),
                       torch.zeros(B, mesh.n, 4, device=x.device), x, T[..., None],
                       ((T - T_scan[None]) / DT_SCALE)[..., None], torch.zeros(B, mesh.n, 3, device=x.device)], dim=-1)
        x = model.step_shape(f, x)
    cost = torch.zeros(B, device=x.device)
    for k in range(ctrl.shape[1]):
        x, T = model.step(node_features_batch(mesh, x, T, ctrl[:, k, 0], ctrl[:, k, 1], ctrl[:, k, 2]), x, T)
        cost = cost + ((x - x_tgt[None])[..., 1:] ** 2).sum(dim=(1, 2))
    return cost


def training_action_ranges(root=f"{AF}/data/dataset_die12_coil_5pass"):
    """(min, max) of die centre, angle and stroke over the new-simulator training
    runs (the plain square, held out, is excluded)."""
    vals = []
    for d in sorted(os.listdir(root)):
        if d == "square":
            continue
        for r in json.load(open(f"{root}/{d}/manifest.json"))["records"]:
            if r["kind"] == "hit_final":
                vals.append((r["die_center_mm"], r["R_j_deg"], r["u_j_mm"]))
    v = onp.array(vals)
    return {"die_center_mm": [float(v[:, 0].min()), float(v[:, 0].max())],
            "angle_deg": [float(v[:, 1].min()), float(v[:, 1].max())],
            "stroke_mm": [float(v[:, 2].min()), float(v[:, 2].max())]}


class Planner:
    """Random shooting + top-K gradient refinement (RoboCraft, Shi et al. RSS 2022,
    Sec. III-E.2), with this planner's own refinement (SLSQP) and cost:

      A  screening: N plans drawn uniformly in the bounds (seeded), rolled out
         through the GNN in batches of B with gradients off, ranked by the cost;
      B  selection: the K lowest-cost plans, sorted;
      C  refinement: the existing SLSQP run, unchanged, from each of the K starts,
         independently (one optimizer per start, run one after another);
      D  output: the refined plan with the lowest model cost + diagnostics.

    mode "default" (or n_samples <= 1) is the previous single-start planner (the
    data runs' pattern as the start); mode "random" refines K uniform random
    starts without screening (ablation)."""

    def __init__(self, model, mesh, x_tgt, device, maxiter=100, n_samples=100, top_k=5, sample_batch=50, seed=0,
                 sample_train_range=False, include_warm_start=False):
        self.model, self.mesh, self.x_tgt, self.device, self.maxiter = model, mesh, x_tgt, device, maxiter
        self.n_samples, self.top_k, self.sample_batch, self.seed = n_samples, top_k, sample_batch, seed
        self.sample_train_range, self.include_warm_start = sample_train_range, include_warm_start
        self.train_ranges = training_action_ranges()

    def bounds(self, x_scan):
        tip = float((self.mesh.rest[:, 0] + x_scan[:, 0]).max())
        x0 = json.load(open(TARGET_SPEC))["square_section_mm"][0] + 0.5 * DIE_WIDTH_MM
        hi = max(x0 + 1.0, tip - 0.5 * DIE_WIDTH_MM - MIN_METAL_MM)
        return [(x0, hi), ANGLE_RANGE, (U_MIN_MM, U_MAX_MM)] * HITS_PER_CYCLE, x0, hi

    def initial_guess(self, pointer, x0, hi):
        """The data runs' pattern: the next 3 adjacent stations of a sweep (a new
        sweep from the first station when fewer than 3 remain), 0 then 90 deg, 1.5 mm."""
        n_st = int((hi - x0) // STATION_STEP_MM) + 1
        start = pointer % n_st
        if start + 2 >= n_st:
            start = 0
        u = []
        for j in range(3):
            c = min(x0 + (start + j) * STATION_STEP_MM, hi)
            u += [c, 0.0, 1.5, c, 90.0, 1.5]
        return onp.array(u, dtype=onp.float64)

    def _sample(self, n, lo, span, rng):
        """n plans uniform in the bounds (or in their overlap with the training data's ranges)."""
        lo_s, hi_s = lo.copy(), lo + span
        if self.sample_train_range:
            for j, key in enumerate(("die_center_mm", "angle_deg", "stroke_mm")):
                a, b = self.train_ranges[key]
                lo_s[j::3] = onp.maximum(lo[j::3], a); hi_s[j::3] = onp.minimum(lo[j::3] + span[j::3], b)
                bad = hi_s[j::3] <= lo_s[j::3]                     # no overlap: fall back to the full bounds
                lo_s[j::3][bad], hi_s[j::3][bad] = lo[j::3][bad], (lo + span)[j::3][bad]
        return lo_s + rng.random((n, len(lo))) * (hi_s - lo_s)

    @torch.no_grad()
    def _screen(self, U, x_scan, T_scan, first):
        assert not torch.is_grad_enabled()                       # screening stores no gradients
        costs = []
        for i in range(0, len(U), self.sample_batch):
            ctrl = torch.tensor(U[i:i + self.sample_batch], dtype=torch.float32, device=self.device)
            costs.append(forecast_cost_batch(self.model, self.mesh, x_scan, T_scan, ctrl.view(-1, HITS_PER_CYCLE, 3),
                                             first, self.x_tgt))
        return torch.cat(costs).cpu().numpy().astype(onp.float64)

    def _refine(self, u0, lo, span, x_scan, T_scan, first):
        """The existing single-start SLSQP solve (unchanged settings) from u0."""
        nfev = [0]

        def cost_grad(z):
            nfev[0] += 1
            u = torch.tensor(lo + z * span, dtype=torch.float32, device=self.device, requires_grad=True)
            shapes, _ = forecast(self.model, self.mesh, x_scan, T_scan, u.view(HITS_PER_CYCLE, 3), first)
            cost = cross_section_cost(shapes, self.x_tgt)
            cost.backward()
            return cost.item(), u.grad.cpu().numpy().astype(onp.float64) * span

        z0 = onp.clip((u0 - lo) / span, 0.0, 1.0)
        c0, g0 = cost_grad(z0)
        c0 = max(c0, 1e-12)                        # SLSQP on a rescaled problem (see mpc.py's 2026-09-26 fix)
        t0 = time.perf_counter()
        res = minimize(lambda z: tuple(v / c0 for v in cost_grad(z)), z0, jac=True, method="SLSQP",
                       bounds=[(0.0, 1.0)] * len(z0), options={"maxiter": self.maxiter})
        z = onp.clip(res.x, 0.0, 1.0)
        return lo + z * span, {"initial_cost": c0, "final_cost": float(res.fun) * c0, "nit": int(res.nit),
                               "nfev": nfev[0], "message": str(res.message), "status": int(res.status),
                               "time_s": time.perf_counter() - t0,
                               "start_grad_norm": float(onp.linalg.norm(g0)), "start_grad_finite": bool(onp.isfinite(g0).all()),
                               "n_at_bound": int(((z <= 1e-9) | (z >= 1 - 1e-9)).sum())}

    def plan(self, x_scan, T_scan, first, pointer, cycle=0, warm=None, mode="topk"):
        b, x0, hi = self.bounds(x_scan)
        lo = onp.array([v[0] for v in b]); span = onp.array([v[1] - v[0] for v in b])
        self.model.eval()
        rng = onp.random.default_rng([self.seed, cycle])
        if mode == "topk" and self.n_samples <= 1:
            mode = "default"
        t0 = time.perf_counter()
        diag = {"mode": mode, "n_samples": self.n_samples, "top_k": self.top_k, "sample_batch": self.sample_batch,
                "seed": self.seed, "cycle_seed": [self.seed, cycle], "sample_train_range": self.sample_train_range,
                "training_action_ranges": self.train_ranges}
        if mode == "default":
            starts, idx, screen = [onp.clip(self.initial_guess(pointer, x0, hi), lo, lo + span)], [None], None
            tA = tB = 0.0
        elif mode == "random":
            starts, idx, screen = list(self._sample(self.top_k, lo, span, rng)), list(range(self.top_k)), None
            tA = tB = 0.0
        else:
            U = self._sample(self.n_samples, lo, span, rng)
            screen = self._screen(U, x_scan, T_scan, first)
            tA = time.perf_counter() - t0
            t1 = time.perf_counter()
            order = onp.argsort(screen, kind="stable")[:self.top_k]
            idx = [int(i) for i in order]
            starts = [U[i] for i in idx]
            if self.include_warm_start and warm is not None:   # replaces the worst of the K
                starts[-1], idx[-1] = onp.clip(onp.asarray(warm, dtype=onp.float64).reshape(-1), lo, lo + span), "warm"
            tB = time.perf_counter() - t1
        t2 = time.perf_counter()
        runs = []
        for k, u0 in enumerate(starts):
            u, rec = self._refine(u0, lo, span, x_scan, T_scan, first)
            runs.append((u, {"start": idx[k], "screen_cost": None if screen is None or idx[k] == "warm" else float(screen[idx[k]]),
                             **rec}))
        tC = time.perf_counter() - t2
        best = int(onp.argmin([r["final_cost"] for _, r in runs]))
        u, rb = runs[best]
        diag.update({"screening_costs": None if screen is None else [float(v) for v in screen],
                     "selected": idx, "starts": [r for _, r in runs], "best_start": best,
                     "time_s": {"A_screen": tA, "B_select": tB, "C_refine": tC},
                     "screen_time_per_rollout_s": tA / self.n_samples if screen is not None else None})
        info = {"initial_cost": rb["initial_cost"], "final_cost": rb["final_cost"], "nit": rb["nit"],
                "message": rb["message"], "plan_time_s": time.perf_counter() - t0, "bounds_die_mm": [x0, hi],
                "multistart": diag}
        return u.reshape(HITS_PER_CYCLE, 3), info


# ----------------------------------------------------------------------------
# Run against the simulator
# ----------------------------------------------------------------------------
def widths_ok(mesh, x, tol_mm=0.2, side=10.6):
    """Every 2 mm slice from the first station (current x = square start + half
    the die, 31.58 mm) to 4 mm before the free end within side + tol at 0 and
    90 deg. The slices next to the taper are left out: no data run got them
    below 11-13 mm (decided 2026-10-08)."""
    p = (mesh.rest + x)[mesh.wall].cpu().numpy()
    sq0 = json.load(open(TARGET_SPEC))["square_section_mm"][0] + 0.5 * DIE_WIDTH_MM
    tip = p[:, 0].max()
    worst = 0.0
    for a in onp.arange(sq0, tip - 4.0, 2.0):
        q = p[(p[:, 0] >= a) & (p[:, 0] < a + 2.0)]
        if len(q) >= 8:
            worst = max(worst, onp.ptp(q[:, 1]), onp.ptp(q[:, 2]))
    return bool(worst <= side + tol_mm), float(worst)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="coil_T GNN checkpoint (GNN/coil_T_sweep_test_square/mp_M/stage3.pt).")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--max-cycles", type=int, default=20)
    p.add_argument("--stop-tol-mm", type=float, default=0.2)
    p.add_argument("--maxiter", type=int, default=100)
    p.add_argument("--n-samples", type=int, default=100, help="Screening samples N (<= 1: single default start).")
    p.add_argument("--top-k", type=int, default=5, help="Starts refined K.")
    p.add_argument("--sample-batch", type=int, default=50, help="Screening rollouts per batch B.")
    p.add_argument("--seed", type=int, default=0, help="Sampling seed (combined with the cycle number).")
    p.add_argument("--sample-train-range", action="store_true",
                   help="Sample only within the training data's action ranges (overlap with the bounds).")
    p.add_argument("--include-warm-start", action="store_true",
                   help="The previous cycle's plan replaces the worst of the K starts.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    from jax_forge.utils import save_sol
    import jax.numpy as jnp
    from applications.Agility_Forge.generate_dataset import _locked
    from applications.Agility_Forge.generate_square_rollout import Manifest, log_failure

    dev = args.device
    mesh = Mesh(MESH_VTU, dev)
    model = load_gnn(mesh, args.checkpoint, dev)
    model.freeze_normalizers()
    import meshio
    x_tgt = torch.tensor(meshio.read(TARGET_VTU).point_data["Displacement"], dtype=torch.float32, device=dev)
    planner = Planner(model, mesh, x_tgt, dev, args.maxiter, args.n_samples, args.top_k, args.sample_batch, args.seed,
                      args.sample_train_range, args.include_warm_start)
    print("training-data action ranges (for comparison with the bounds):", json.dumps(planner.train_ranges), flush=True)

    os.makedirs(f"{args.out_dir}/rollout_01", exist_ok=True)
    ckpt = f"{args.out_dir}/rollout_01/checkpoint.npz"
    with _locked(os.path.join(AF, "data", "msh", "jax_forge", "tmp")):
        fmesh, R, H, T_fn = load_default_billet()
    plant = ForgingPlant(fmesh, float(R), float(H), T_fn, linear_solver="scipy")   # direct solve: late reheats need it
    pts = onp.asarray(fmesh.points)
    surf = mesh.full_to_surface
    manifest = Manifest(args.out_dir, {"part": "square_coil_mpc",
                                       "mpc": {"checkpoint": args.checkpoint, "cost": "cross-section, sum over 6",
                                               "gnn_state": "displacement + temperature (coil_T.py)",
                                               "temperature_at_scan": "measured (simulator surface temperature, an IR camera in practice)",
                                               "stroke_mm": [U_MIN_MM, U_MAX_MM], "angle_deg": list(ANGLE_RANGE),
                                               "coil": "average of the 6 planned die centres",
                                               "planner": {"n_samples": args.n_samples, "top_k": args.top_k,
                                                           "sample_batch": args.sample_batch, "seed": args.seed,
                                                           "sample_train_range": args.sample_train_range,
                                                           "include_warm_start": args.include_warm_start,
                                                           "refinement": f"SLSQP, maxiter {args.maxiter}, one per start"},
                                               "reheat_rule": "keep the hotter of current and coil temperature",
                                               "stop_check_from_mm": json.load(open(TARGET_SPEC))["square_section_mm"][0] + 0.5 * DIE_WIDTH_MM,
                                               "linear_solver": "scipy (direct)",
                                               "max_cycles": args.max_cycles, "stop_tol_mm": args.stop_tol_mm}})
    manifest.data.setdefault("reheats", []); manifest.data.setdefault("cycles", [])
    manifest.data["band_width_frac"] = None
    manifest._write()

    def save_ckpt(state, sched):
        tmp = ckpt + ".tmp.npz"
        onp.savez(tmp, sol_u=onp.asarray(state.sol_u), sol_dT=onp.asarray(state.sol_dT), sched=json.dumps(sched),
                  **{f"iv_{k}": onp.asarray(v) for k, v in enumerate(state.int_vars)})
        os.replace(tmp, ckpt)

    def save_vtu(state, rel):
        save_sol(plant.problem.fes[0], state.sol_u, f"{args.out_dir}/{rel}",
                 point_infos=[("Displacement", state.sol_u), ("Temperature", state.sol_dT)])

    surf_x = lambda state: torch.tensor(onp.asarray(state.sol_u)[surf], dtype=torch.float32, device=dev)
    surf_T = lambda state: torch.tensor(onp.asarray(state.sol_dT).reshape(-1)[surf], dtype=torch.float32, device=dev)

    if os.path.exists(ckpt):
        z = onp.load(ckpt)
        state = ForgingState(jnp.array(z["sol_u"]), jnp.array(z["sol_dT"]), [jnp.array(z[f"iv_{k}"]) for k in range(N_INT_VARS)])
        sched = json.loads(str(z["sched"]))
        manifest.data["records"] = [r for r in manifest.data["records"] if r["hit"] <= sched["completed"]]
        manifest.data["reheats"] = [r for r in manifest.data["reheats"] if r["hit"] <= sched["completed"]
                                    or (r["hit"] == sched["completed"] + 1 and sched["reheated"])]
        manifest.data["n_hits_per_rollout"] = sched["completed"]
        manifest._write()
        print(f"Resuming at hit {sched['completed']}, cycle {sched['cycle']}")
    else:
        state = plant.reset()
        sched = {"completed": 0, "cycle": 0, "plan": [], "pos": 0, "reheated": False, "pointer": 0, "done": False}
        save_ckpt(state, sched)

    while not sched["done"]:
        if sched["pos"] >= len(sched["plan"]):
            x_scan = surf_x(state)
            ok, worst = widths_ok(mesh, x_scan, args.stop_tol_mm)
            if (ok and sched["completed"] > 0) or sched["cycle"] >= args.max_cycles:
                sched["done"] = True
                manifest.data["stop"] = {"reason": "target reached" if ok else "max cycles", "worst_width_mm": worst,
                                         "hits": sched["completed"], "cycles": sched["cycle"]}
                manifest._write(); save_ckpt(state, sched)
                break
            first = sched["completed"] == 0
            T_scan = surf_T(state)                     # measured surface temperature (IR camera in practice)
            warm = [[e_["die_center_mm"], e_["R_j_deg"], e_["u_j_mm"]] for e_ in sched["plan"]] or None
            u, info = planner.plan(x_scan, T_scan, first, sched["pointer"], cycle=sched["cycle"] + 1, warm=warm)
            with torch.no_grad():
                pred, predT = forecast(model, mesh, x_scan, T_scan, torch.tensor(u, dtype=torch.float32, device=dev),
                                       first, smooth=False)
            c = float(u[:, 0].mean())
            sched.update(plan=[{"die_center_mm": float(r[0]), "R_j_deg": float(r[1]), "u_j_mm": float(r[2])} for r in u],
                         pos=0, reheated=False, cycle=sched["cycle"] + 1, coil_center_mm=c,
                         pointer=sched["pointer"] + 3)
            onp.save(f"{args.out_dir}/rollout_01/cycle_{sched['cycle']:02d}_forecast.npy",
                     onp.stack([p_.cpu().numpy() for p_ in pred]))
            onp.save(f"{args.out_dir}/rollout_01/cycle_{sched['cycle']:02d}_forecast_T.npy",
                     onp.stack([p_.cpu().numpy() for p_ in predT]))
            manifest.data["cycles"].append({"cycle": sched["cycle"], "start_hit": sched["completed"] + 1,
                                            "coil_center_mm": c, "plan": sched["plan"], "worst_width_at_scan_mm": worst,
                                            **info})
            manifest._write()
            save_ckpt(state, sched)
            print(f"[cycle {sched['cycle']}] coil {c:.1f} mm, plan {onp.round(u, 2).tolist()}, cost "
                  f"{info['initial_cost']:.3e} -> {info['final_cost']:.3e} ({info['nit']} it, {info['plan_time_s']:.0f} s)",
                  flush=True)

        if not sched["reheated"]:
            nxt = sched["completed"] + 1
            c = sched["coil_center_mm"]
            t0 = time.time()
            if sched["completed"] == 0:
                state = ForgingState(state.sol_u, jnp.array(coil_temperature(pts, state.sol_u, c)), state.int_vars)
                rel, rinfo = "rollout_01/undeformed.vtu", {"max_jump_C": None, "n_ramp": 0}
            else:
                first_hit = sched["plan"][0]
                hit = Hit(die_center_mm=first_hit["die_center_mm"], compression_displacement=0.0,
                          rotation_euler_x=first_hit["R_j_deg"], total_time=TOTAL_TIME_S)
                T_new = reheat_temperature(pts, state.sol_u, state.sol_dT, c)
                for i, step_C in enumerate((50.0, 25.0, 12.5)):
                    try:
                        state = plant.reheat(state, T_new, hit, max_ramp_step_C=step_C)
                        break
                    except RuntimeError as exc:
                        log_failure(args.out_dir, {"kind": "reheat", "before_hit": nxt, "attempt": f"{step_C:g} C/step",
                                                   "error": repr(exc), "traceback": traceback.format_exc(),
                                                   "timestamp": datetime.now(timezone.utc).isoformat()})
                        if i == 2:
                            raise
                rel, rinfo = f"rollout_01/reheat_before_hit_{nxt:02d}.vtu", plant.last_reheat_info
            save_vtu(state, rel)
            if sched["completed"] == 0:
                manifest.add({"rollout": 1, "hit": 0, "kind": "undeformed", "vtu_path": rel})
            manifest.data["reheats"].append({"rollout": 1, "hit": nxt, "cycle": sched["cycle"], "coil_center_mm": c,
                                             "vtu_path": rel, "wall_time_s": time.time() - t0, **rinfo})
            manifest._write()
            sched["reheated"] = True
            save_ckpt(state, sched)

        e = sched["plan"][sched["pos"]]
        pred = onp.load(f"{args.out_dir}/rollout_01/cycle_{sched['cycle']:02d}_forecast.npy")[sched["pos"]]
        predT = onp.load(f"{args.out_dir}/rollout_01/cycle_{sched['cycle']:02d}_forecast_T.npy")[sched["pos"]]

        def apply(u_mm, split_part):
            nonlocal state
            n = sched["completed"] + 1
            hit = Hit(die_center_mm=e["die_center_mm"], compression_displacement=u_mm,
                      rotation_euler_x=e["R_j_deg"], total_time=TOTAL_TIME_S)
            t0 = time.time()
            state = plant.step(state, hit, is_first_hit=(sched["completed"] == 0), reheat=False)
            sched["completed"] = n
            rel = f"rollout_01/hit_{n:02d}_final.vtu"
            save_vtu(state, rel)
            x_now = onp.asarray(state.sol_u)[surf]
            manifest.add({"rollout": 1, "hit": n, "kind": "hit_final", "cycle": sched["cycle"],
                          "die_center_mm": e["die_center_mm"], "die_width_mm": DIE_WIDTH_MM, "R_j_deg": e["R_j_deg"],
                          "u_j_mm": u_mm, "total_time_s": TOTAL_TIME_S, "pass": None, "station": None,
                          "split_part": split_part,
                          "forecast_rmse_mm": None if split_part == 1 else float(onp.sqrt(((pred - x_now) ** 2).mean())),
                          "forecast_T_abs_err_C": None if split_part == 1 else
                          float(onp.abs(predT - onp.asarray(state.sol_dT).reshape(-1)[surf]).mean()),
                          "free_end_mm": float((pts[:, 0] + onp.asarray(state.sol_u)[:, 0]).max()),
                          "wall_time_s": time.time() - t0, "vtu_path": rel})
            print(f"  hit {n} done ({time.time() - t0:.0f} s)"
                  + ("" if split_part == 1 else f", forecast RMSE {manifest.data['records'][-1]['forecast_rmse_mm']:.3f} mm"),
                  flush=True)

        try:
            apply(e["u_j_mm"], 0)
        except Exception as exc:          # retry once as two half-strokes, as the data runs do
            log_failure(args.out_dir, {**e, "attempt": "full", "after_hit": sched["completed"], "error": repr(exc),
                                       "traceback": traceback.format_exc(),
                                       "timestamp": datetime.now(timezone.utc).isoformat()})
            print(f"  [FAILED] full stroke ({exc!r}); retrying as two half-strokes", flush=True)
            apply(0.5 * e["u_j_mm"], 1)
            apply(0.5 * e["u_j_mm"], 2)
        sched["pos"] += 1
        save_ckpt(state, sched)
    print(f"MPC done: {json.dumps(manifest.data.get('stop'))}")


if __name__ == "__main__":
    main()
