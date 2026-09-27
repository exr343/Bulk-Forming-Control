"""MPC planner: GNN surrogate (ForgeGNN, ../GNN/) for prediction, ForgingPlant
for ground truth (see ./README.md's "Open design questions" for what's
settled vs. still open -- this implements the settled subset only).

Single-shooting SQP: the only decision variables are the controls
u_0..u_{N-1} (3 per step -- d_j_frac, R_j_deg, u_j_frac), N = horizon. x_k is
NOT a separate variable -- it's produced by rolling the GNN forward from x_0,
so the dynamics constraint x_{k+1}=f(x_k,u_k) is satisfied by construction
and never appears as an explicit constraint (unlike multi-shooting, which
would need ~13,065-dim x_k as a free variable per step -- intractable here).
Solved with scipy's SLSQP, which does exactly the requested BFGS-approximated
SQP internally (sequential QP subproblems, quasi-Newton Hessian of the
Lagrangian) -- no hand-rolled Hessian update needed.

`run()` closes the loop against the real plant (control/plant_interface.py):
plan over the remaining horizon from the plant's TRUE current state, apply
only the first control for real, re-plan from the resulting TRUE state (not
the GNN's own prediction) -- horizon shrinks each step (5,4,3,2,1). This is
the direct measurement of GNN surrogate quality ./README.md's "Validation
loop" section describes.
"""

import time

import numpy as onp
import torch
from scipy.optimize import minimize

from applications.Agility_Forge.GNN.data import band_half_thickness, build_node_features, gap_frac
from applications.Agility_Forge.GNN.model import ForgeGNN
from applications.Agility_Forge.hit_config import Hit
from applications.Agility_Forge.control.plant_interface import ForgingPlant, extract_surface_state

N_CONTROLS = 3  # (d_j_frac, R_j_deg, u_j_frac) per step

# Bounds as actually seen in the training data (generate_dataset.py):
# d_j_frac in [edge_margin_frac, 1 - band_width_frac - edge_margin_frac] with
# the dataset's band_width_frac=0.2, edge_margin_frac=0.02 (default, used for
# every generation run to date -- not itself stored in manifest.json, unlike
# band_width_frac). R_j is periodic (0deg==360deg); optimizing a plain bounded
# scalar rather than (sin,cos)+unit-circle-constraint is a deliberate
# simplification -- fine unless the true optimum sits at the wraparound.
# u_j_frac in [0,1] by construction (data.py's own normalization).
CONTROL_BOUNDS = [(0.02, 0.78), (0.0, 360.0), (0.0, 1.0)]
# U_J_MM_MIN settled at 0.0, NOT the training data's actual sampled floor
# (0.5mm) -- an extended-horizon pilot (control/results/before_2026-09-26_fixes/rollout_415_10hits/)
# showed that with the 0.5mm floor, receding-horizon MPC has no way to choose
# "no hit": once already near target, every remaining required hit is forced
# to apply >=0.5mm of real plastic deformation somewhere, which plateaued
# Chamfer and made Hausdorff (worst-case error) creep upward hit-over-hit
# rather than staying flat. Lowering the floor to 0.0 let the optimizer
# choose a genuine near-zero stroke instead (observed: u_j_mm=0.0 on
# multiple hits once close to target), and Hausdorff stayed flat rather than
# climbing. Accepted tradeoff: u_j in [0, 0.5)mm is outside the range
# generate_dataset.py ever sampled, so the GNN is extrapolating there --
# judged worth it given the alternative (forced continued disturbance) was
# actively making results worse, not just untested.
U_J_MM_MIN, U_J_MM_MAX = 0.0, 2.0
TOTAL_TIME_S = 0.8  # per-hit duration; matches generate_dataset.py's default (never varied)
# The stroke range the GNN was TRAINED on (dataset manifest min/max_compression_
# displacement_mm). The GNN reads its u_j_frac input on this scale
# ((u_mm - 0.5) / 1.5), which differs from the MPC's own u_j_frac -> mm map
# above (0.0-2.0mm) since the floor was lowered. Every GNN call converts the
# MPC's frac -> mm -> GNN frac via gnn_u_frac(), so the GNN predicts exactly
# the stroke the plant applies. Before this fix (2026-09-26) the GNN was fed
# the MPC's frac directly: e.g. frac 0.5 = 1.0mm to the plant but 1.25mm to
# the GNN. Runs made with U_J_MM_MIN=0 before then (results/before_2026-09-26_fixes/prism_target*,
# results/before_2026-09-26_fixes/rollout_415_chamfer_hausdorff, results/before_2026-09-26_fixes/closed_loop_test_set) carry
# that mismatch. Strokes below 0.5mm map to negative GNN fracs: extrapolation,
# as before, but now consistent.
GNN_U_MM_MIN, GNN_U_MM_MAX = 0.5, 2.0


# Absolute gap control (control="gap", GNN checkpoints trained with
# `--control gap`; see GNN/data.py, settled 2026-09-27). The 3rd control is
# then the half-gap the dies close to (centred on the bar in the band), as a
# [0,1] fraction over GAP_BOUNDS_MM. The per-hit travel, band half-thickness
# minus half-gap, is capped at TRAVEL_CAP_MM: an SLSQP inequality constraint
# on each planned hit, with the half-thickness taken from the GNN's predicted
# state (the true state for the first hit). A half-gap at or above the
# half-thickness means the dies miss the bar: no change.
# Lower bound = the training data's smallest half-gap (4.96 mm in the pretraining-train split, 5.11 in the
# square finetune runs). Experiment 9 ran with 4.25, taken by mistake from bulge_head, which is not in
# the training set (2 of its hits, 21 and 29, used 4.77 and 4.89 mm). Upper bound clears the widest bulge.
GAP_BOUNDS_MM = (4.96, 9.0)
TRAVEL_CAP_MM = 2.0


def gnn_u_frac(u_j_mm):
    """Physical stroke (mm, float or tensor) -> the GNN's u_j_frac input."""
    return (u_j_mm - GNN_U_MM_MIN) / (GNN_U_MM_MAX - GNN_U_MM_MIN)


def _rollout_cost(u_flat, model, mesh_info, x0, H_mm, band_width_frac, x_ref, horizon,
                  u_j_mm_min=U_J_MM_MIN, u_j_mm_max=U_J_MM_MAX, transverse_weight=0.0, parts=None,
                  control="stroke", travel=None, lengthwise_weight=1.0):
    """u_flat: (horizon*3,) tensor, requires_grad=True. Rolls the GNN forward
    from x0 under each step's control and accumulates
    sum_k ||x_ref - x_k||^2 (Q = identity -- see ./README.md's cost-function
    note; every step is compared against the SAME final-geometry target
    since no intermediate reference trajectory exists, which pulls early
    states toward the final shape too -- a deliberate reading of the
    literal 'sum over horizon' spec, easy to change to a terminal-only cost
    if that pull looks wrong in practice)."""
    u = u_flat.view(horizon, N_CONTROLS)
    x_k = x0.unsqueeze(0)
    cost = x_k.new_zeros(())
    trans = x_k.new_zeros(())
    for k in range(horizon):
        d_j_frac, R_j_deg, u_j_frac = u[k, 0], u[k, 1], u[k, 2]
        d_j_mm = d_j_frac * H_mm
        x_max_band_mm = (d_j_frac + band_width_frac) * H_mm
        if control == "gap":
            # u_j_frac is the half-gap fraction here (see GAP_BOUNDS_MM).
            g_mm = GAP_BOUNDS_MM[0] + u_j_frac * (GAP_BOUNDS_MM[1] - GAP_BOUNDS_MM[0])
            if travel is not None:
                travel.append(band_half_thickness(mesh_info, x_k[0], d_j_mm, x_max_band_mm, R_j_deg) - g_mm)
            gnn_ctrl = gap_frac(g_mm)
        elif control == "gap_travel":
            # Gap-trained GNN, but the optimizer's variable is the travel
            # (stroke) as in stroke control; converted to the half-gap from
            # the (predicted) band half-thickness before the GNN call
            # (experiment 11: separates the GNN's input from the planner's
            # variable).
            u_j_mm = u_j_mm_min + u_j_frac * (u_j_mm_max - u_j_mm_min)
            ht = band_half_thickness(mesh_info, x_k[0], d_j_mm, x_max_band_mm, R_j_deg)
            gnn_ctrl = gap_frac(ht - u_j_mm)
        else:
            u_j_mm = u_j_mm_min + u_j_frac * (u_j_mm_max - u_j_mm_min)
            gnn_ctrl = gnn_u_frac(u_j_mm)
        node_features = build_node_features(
            mesh_info, x_k[0], d_j_frac, d_j_mm, x_max_band_mm, R_j_deg, gnn_ctrl,
        ).unsqueeze(0)
        delta = model.predict_delta(node_features, x_k, accumulate=False)
        x_k = x_k + delta
        cost = cost + ((x_k[0] - x_ref) ** 2).sum()
        if transverse_weight or lengthwise_weight != 1.0:
            # Extra cross-section term (cost experiment 2): the same distance,
            # counting only the cross-section directions (y, z).
            trans = trans + ((x_k[0, :, 1:] - x_ref[:, 1:]) ** 2).sum()
    if parts is not None:
        parts["error"] = cost.item()
        parts["transverse"] = trans.item() if transverse_weight else 0.0
    if lengthwise_weight != 1.0:
        # Scale the lengthwise (x) part of the node distance; 0 = cross-
        # section-only cost (cost experiment 12): length is left to volume
        # conservation.
        cost = lengthwise_weight * (cost - trans) + trans
    if transverse_weight:
        cost = cost + transverse_weight * trans
    return cost


class MPCController:
    def __init__(self, model: ForgeGNN, mesh_info, horizon: int, target_state: torch.Tensor,
                 band_width_frac: float, device: str = "cpu",
                 u_j_mm_min: float = U_J_MM_MIN, u_j_mm_max: float = U_J_MM_MAX,
                 d_j_bounds=None, penalty=None, control="stroke", min_travel_mm=0.0):
        """u_j_mm_min/max: physical range u_j_frac in [0,1] maps to, default
        the module-level U_J_MM_MIN/MAX -- settled at 0.0/2.0 (NOT the
        training data's literal 0.5-2.0mm sampled range; see U_J_MM_MIN's
        own comment for why the floor was deliberately lowered). Overridable
        per-instance for experiments that want the literal training-range
        floor back (e.g. u_j_mm_min=0.5) or something else entirely. Does
        not change CONTROL_BOUNDS's u_j_frac bound itself (still [0,1]) --
        only what physical stroke frac=0 maps
        to."""
        self.control = control  # "stroke", "gap" (see GAP_BOUNDS_MM) or "gap_travel" (see _rollout_cost)
        # Gap control only: > 0 adds travel_k >= min_travel_mm for every planned
        # hit, i.e. no misses or near-misses (experiment 10).
        self.min_travel_mm = min_travel_mm
        self.model = model.to(device).eval()
        self.mesh_info = mesh_info
        self.horizon = horizon
        self.target_state = target_state.to(device)  # (N_surface, 3) target displacement field
        self.band_width_frac = band_width_frac
        self.u_j_mm_min = u_j_mm_min
        self.u_j_mm_max = u_j_mm_max
        # H = max(x), NOT (max-min) -- must match generate_dataset.py's own
        # convention exactly (H = np.max(mesh.points[:, 0])), since d_j_frac/
        # u_j_frac were fit against THAT H. The billet's rest mesh does not
        # start at x=0 (starts around x=-5mm), so max-min would silently be
        # ~5mm too large and shift every dist_to_band feature computed here.
        self.H_mm = float(mesh_info.rest_pos[:, 0].max().item())
        self.device = device
        # d_j_bounds overrides CONTROL_BOUNDS' d_j_frac range, e.g. a floor at
        # the initial temperature profile's 800 C point.
        self.control_bounds = [tuple(d_j_bounds) if d_j_bounds else CONTROL_BOUNDS[0]] + CONTROL_BOUNDS[1:]
        self.bounds = self.control_bounds * horizon
        # Optional control penalties added to the node-distance error (cost-
        # design experiments, 2026-09-26). Keys, all default 0 (= original cost):
        #   delta_d / delta_R / delta_s: weight on the squared change between
        #     consecutive hits of station (normalized to [0,1] over its bounds),
        #     angle (sin^2 of the difference -- the dies press both sides, so
        #     0 and 180 deg are the same hit) and stroke (u_j_frac). The first
        #     planned hit is compared with `u_prev`, the last applied hit.
        #   effort_s: weight on the squared stroke (u_j_frac^2).
        #   transverse: weight on an extra error term counting only the
        #     cross-section (y, z) distance to the target, NOT scaled by E0 --
        #     it's in the error's own units (w = 100 makes cross-section shape
        #     comparable to the lengthwise error at the start of the square run).
        # Penalties are scaled so a unit penalty on one hit costs one hit's
        # share of the plan's starting error (E0 / horizon): weights are
        # relative, independent of the error's ~1e6 magnitude.
        #   lengthwise: weight on the lengthwise (x) part of the node distance
        #     (default 1 = the plain distance; 0 = cross-section only).
        self.penalty = {"delta_d": 0.0, "delta_R": 0.0, "delta_s": 0.0, "effort_s": 0.0, "transverse": 0.0,
                        "lengthwise": 1.0, **(penalty or {})}

    def _penalty(self, u, u_prev):
        """Control penalty (torch scalar, differentiable) for a (horizon, 3)
        physical control tensor; see self.penalty."""
        w = self.penalty
        d_lo, d_hi = self.control_bounds[0]
        d = (u[:, 0] - d_lo) / (d_hi - d_lo)
        R = u[:, 1] * (torch.pi / 180.0)
        s = u[:, 2]
        if u_prev is not None:
            p = torch.as_tensor(onp.asarray(u_prev, dtype=onp.float64), dtype=u.dtype, device=u.device)
            d = torch.cat([((p[0] - d_lo) / (d_hi - d_lo)).view(1), d])
            R = torch.cat([(p[1] * (torch.pi / 180.0)).view(1), R])
            s = torch.cat([p[2].view(1), s])
        pen = u.new_zeros(())
        if len(d) > 1:
            pen = pen + w["delta_d"] * ((d[1:] - d[:-1]) ** 2).sum()
            pen = pen + w["delta_R"] * (torch.sin(R[1:] - R[:-1]) ** 2).sum()
            pen = pen + w["delta_s"] * ((s[1:] - s[:-1]) ** 2).sum()
        return pen + w["effort_s"] * (u[:, 2] ** 2).sum()  # "transverse" is applied in _rollout_cost

    def plan(self, x0: torch.Tensor, horizon: int = None, u_init=None, u_prev=None):
        """Optimizes a `horizon`-length control sequence (default
        self.horizon; `run()` passes a shrinking value as hits are applied)
        from state `x0` toward `self.target_state`. Returns (u_sequence
        (horizon, 3) numpy array, scipy OptimizeResult). `u_prev` (d_j_frac,
        R_j_deg, u_j_frac) of the last applied hit, if any, enters the
        change penalties. result.error_cost / result.penalty_cost give the
        solution's two parts (physical units)."""
        horizon = horizon or self.horizon
        use_penalty = any(v != 0.0 for k, v in self.penalty.items() if k not in ("transverse", "lengthwise"))
        w_trans = self.penalty["transverse"]
        x0 = x0.to(self.device)
        u0 = onp.asarray(u_init if u_init is not None else
                          onp.tile([min(max(0.4, self.control_bounds[0][0]), self.control_bounds[0][1]), 90.0, 0.5],
                                   horizon), dtype=onp.float64)

        pen_scale = [0.0]  # E0 / horizon, set from the initial guess below
        parts = {}
        gap = self.control == "gap"
        cache = {}  # gap mode: travel constraints (+ Jacobian) from the same rollout as the cost

        def cost_and_grad(u_flat_np):
            u_t = torch.tensor(u_flat_np, dtype=torch.float32, device=self.device, requires_grad=True)
            rparts = {}
            travel = [] if gap else None
            error = _rollout_cost(u_t, self.model, self.mesh_info, x0, self.H_mm,
                                  self.band_width_frac, self.target_state, horizon,
                                  self.u_j_mm_min, self.u_j_mm_max, w_trans, rparts,
                                  control=self.control, travel=travel,
                                  lengthwise_weight=self.penalty["lengthwise"])
            cost = error
            if use_penalty:
                pen = pen_scale[0] * self._penalty(u_t.view(horizon, N_CONTROLS), u_prev)
                cost = error + pen
                parts["penalty"] = pen.item()
            parts["error"] = rparts["error"]
            parts["transverse"] = w_trans * rparts["transverse"]
            if gap:
                # c_k = cap - travel_k >= 0, one per planned hit; Jacobian row by row.
                jac = onp.stack([torch.autograd.grad(-t, u_t, retain_graph=True)[0].cpu().numpy()
                                 for t in travel]).astype(onp.float64)
                cache["key"] = u_flat_np.tobytes()
                c = [TRAVEL_CAP_MM - t.item() for t in travel]
                if self.min_travel_mm > 0:  # travel_k - min >= 0: Jacobian rows are -jac
                    c += [t.item() - self.min_travel_mm for t in travel]
                    jac = onp.concatenate([jac, -jac])
                cache["c"] = onp.array(c, dtype=onp.float64)
                cache["jac"] = jac
                parts["travel_mm"] = [t.item() for t in travel]
            cost.backward()
            return cost.item(), u_t.grad.cpu().numpy().astype(onp.float64)

        # SLSQP runs on a rescaled problem: every control mapped to [0, 1]
        # within its bounds, and the cost divided by its value at the initial
        # guess. Unscaled, the cost is ~1e6 (squared mm summed over ~4.4k
        # nodes) with gradients ~1e5, and R_j spans 0-360 against 0-1 for the
        # others; SLSQP then returned the initial guess unchanged while
        # reporting success (found 2026-09-26 -- likely behind earlier runs'
        # "do nothing" plans, e.g. results/before_2026-09-26_fixes/prism_target*). Returned controls
        # and result.fun are in physical units.
        lo = onp.array([b[0] for b in self.control_bounds] * horizon, dtype=onp.float64)
        span = onp.array([b[1] - b[0] for b in self.control_bounds] * horizon, dtype=onp.float64)
        z0 = onp.clip((u0 - lo) / span, 0.0, 1.0)
        if use_penalty:
            with torch.no_grad():
                e0 = _rollout_cost(torch.tensor(lo + z0 * span, dtype=torch.float32, device=self.device),
                                   self.model, self.mesh_info, x0, self.H_mm, self.band_width_frac,
                                   self.target_state, horizon, self.u_j_mm_min, self.u_j_mm_max).item()
            pen_scale[0] = e0 / horizon
        cost_scale = max(cost_and_grad(lo + z0 * span)[0], 1e-12)

        def scaled_cost_and_grad(z):
            cost, grad = cost_and_grad(lo + z * span)
            return cost / cost_scale, grad * span / cost_scale

        constraints = ()
        if gap:
            def _travel(z):
                u_np = lo + z * span
                if cache.get("key") != u_np.tobytes():
                    cost_and_grad(u_np)
                return cache

            constraints = ({"type": "ineq", "fun": lambda z: _travel(z)["c"],
                            "jac": lambda z: _travel(z)["jac"] * span},)
        result = minimize(scaled_cost_and_grad, z0, jac=True, method="SLSQP",
                           bounds=[(0.0, 1.0)] * len(z0), constraints=constraints)
        result.x = lo + result.x * span
        result.fun = result.fun * cost_scale
        cost_and_grad(result.x)  # record the solution's error / penalty split
        result.error_cost = parts["error"]
        result.penalty_cost = parts.get("penalty", 0.0)
        result.transverse_cost = parts["transverse"]
        result.planned_travel_mm = parts.get("travel_mm")
        return result.x.reshape(horizon, N_CONTROLS), result

    def run(self, plant: ForgingPlant, n_hits: int = None, step_callback=None):
        """Receding-horizon loop against the REAL plant: re-plan from the
        plant's true current state each step, apply only the first control
        for real, shrink the horizon by one and repeat. Returns
        (applied_controls, plan_vs_actual_rmse_mm, final_state) -- the
        latter is exactly the surrogate-quality measurement
        ./README.md's "Validation loop" section describes.

        `step_callback(step_idx, state, hit)`, if given, is called after
        every successful real step (state = the resulting ForgingState) --
        lets the caller save each step's .vtu as it happens rather than only
        at the end, so a multi-hour run's progress survives a crash/timeout
        partway through.

        Raises whatever plant.step raises on solver non-convergence (no
        retry -- matches generate_dataset.py's own fail-fast convention).

        Returns a 4th element, `plan_time_s` (list, one wall-clock seconds
        entry per real step, timing just the `self.plan(...)` call -- not
        the real-plant FEM solve), alongside the pre-existing three."""
        assert self.control == "stroke", "run() supports stroke control only; see eval_square_target.py for gap"
        n_hits = n_hits or self.horizon
        state = plant.reset()
        applied_controls, plan_vs_actual_rmse_mm, plan_time_s = [], [], []

        u_init = None  # warm-start: each plan's un-applied tail seeds the next (one step shorter)
        for step_idx in range(n_hits):
            remaining = n_hits - step_idx
            x0 = extract_surface_state(state.sol_u, self.mesh_info).to(self.device)
            t0 = time.perf_counter()
            u_seq, _ = self.plan(x0, horizon=remaining, u_init=u_init)
            plan_time_s.append(time.perf_counter() - t0)
            u_init = u_seq[1:].flatten() if remaining > 1 else None
            d_j_frac, R_j_deg, u_j_frac = [float(v) for v in u_seq[0]]
            u_j_mm = self.u_j_mm_min + u_j_frac * (self.u_j_mm_max - self.u_j_mm_min)

            # GNN's own prediction for this one step, for the plan-vs-actual
            # comparison below -- NOT what gets applied to the plant.
            with torch.no_grad():
                d_j_mm = d_j_frac * self.H_mm
                x_max_band_mm = (d_j_frac + self.band_width_frac) * self.H_mm
                nf = build_node_features(self.mesh_info, x0, d_j_frac, d_j_mm,
                                          x_max_band_mm, R_j_deg, gnn_u_frac(u_j_mm)).unsqueeze(0)
                predicted_next = x0 + self.model.predict_delta(nf, x0.unsqueeze(0), accumulate=False)[0]

            hit = Hit(x_min_band=d_j_frac, x_max_band=d_j_frac + self.band_width_frac,
                      compression_displacement=u_j_mm, rotation_euler_x=R_j_deg,
                      total_time=TOTAL_TIME_S)
            state = plant.step(state, hit, is_first_hit=(step_idx == 0))
            actual_next = extract_surface_state(state.sol_u, self.mesh_info).to(self.device)
            if step_callback is not None:
                step_callback(step_idx, state, hit)

            applied_controls.append((d_j_frac, R_j_deg, u_j_mm))
            plan_vs_actual_rmse_mm.append(((predicted_next - actual_next) ** 2).mean().sqrt().item())

        return applied_controls, plan_vs_actual_rmse_mm, plan_time_s, state
