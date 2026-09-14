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

import numpy as onp
import torch
from scipy.optimize import minimize

from applications.Agility_Forge.GNN.data import build_node_features
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
U_J_MM_MIN, U_J_MM_MAX = 0.5, 2.0  # must match the dataset's actual generation range
TOTAL_TIME_S = 0.8  # per-hit duration; matches generate_dataset.py's default (never varied)


def _rollout_cost(u_flat, model, mesh_info, x0, H_mm, band_width_frac, x_ref, horizon):
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
    for k in range(horizon):
        d_j_frac, R_j_deg, u_j_frac = u[k, 0], u[k, 1], u[k, 2]
        d_j_mm = d_j_frac * H_mm
        x_max_band_mm = (d_j_frac + band_width_frac) * H_mm
        node_features = build_node_features(
            mesh_info, x_k[0], d_j_frac, d_j_mm, x_max_band_mm, R_j_deg, u_j_frac,
        ).unsqueeze(0)
        delta = model.predict_delta(node_features, x_k, accumulate=False)
        x_k = x_k + delta
        cost = cost + ((x_k[0] - x_ref) ** 2).sum()
    return cost


class MPCController:
    def __init__(self, model: ForgeGNN, mesh_info, horizon: int, target_state: torch.Tensor,
                 band_width_frac: float, device: str = "cpu"):
        self.model = model.to(device).eval()
        self.mesh_info = mesh_info
        self.horizon = horizon
        self.target_state = target_state.to(device)  # (N_surface, 3) target displacement field
        self.band_width_frac = band_width_frac
        # H = max(x), NOT (max-min) -- must match generate_dataset.py's own
        # convention exactly (H = np.max(mesh.points[:, 0])), since d_j_frac/
        # u_j_frac were fit against THAT H. The billet's rest mesh does not
        # start at x=0 (starts around x=-5mm), so max-min would silently be
        # ~5mm too large and shift every dist_to_band feature computed here.
        self.H_mm = float(mesh_info.rest_pos[:, 0].max().item())
        self.device = device
        self.bounds = CONTROL_BOUNDS * horizon

    def plan(self, x0: torch.Tensor, horizon: int = None, u_init=None):
        """Optimizes a `horizon`-length control sequence (default
        self.horizon; `run()` passes a shrinking value as hits are applied)
        from state `x0` toward `self.target_state`. Returns (u_sequence
        (horizon, 3) numpy array, scipy OptimizeResult)."""
        horizon = horizon or self.horizon
        x0 = x0.to(self.device)
        u0 = onp.asarray(u_init if u_init is not None else
                          onp.tile([0.4, 90.0, 0.5], horizon), dtype=onp.float64)

        def cost_and_grad(u_flat_np):
            u_t = torch.tensor(u_flat_np, dtype=torch.float32, device=self.device, requires_grad=True)
            cost = _rollout_cost(u_t, self.model, self.mesh_info, x0, self.H_mm,
                                  self.band_width_frac, self.target_state, horizon)
            cost.backward()
            return cost.item(), u_t.grad.cpu().numpy().astype(onp.float64)

        result = minimize(cost_and_grad, u0, jac=True, method="SLSQP",
                           bounds=CONTROL_BOUNDS * horizon)
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
        retry -- matches generate_dataset.py's own fail-fast convention)."""
        n_hits = n_hits or self.horizon
        state = plant.reset()
        applied_controls, plan_vs_actual_rmse_mm = [], []

        u_init = None  # warm-start: each plan's un-applied tail seeds the next (one step shorter)
        for step_idx in range(n_hits):
            remaining = n_hits - step_idx
            x0 = extract_surface_state(state.sol_u, self.mesh_info).to(self.device)
            u_seq, _ = self.plan(x0, horizon=remaining, u_init=u_init)
            u_init = u_seq[1:].flatten() if remaining > 1 else None
            d_j_frac, R_j_deg, u_j_frac = [float(v) for v in u_seq[0]]
            u_j_mm = U_J_MM_MIN + u_j_frac * (U_J_MM_MAX - U_J_MM_MIN)

            # GNN's own prediction for this one step, for the plan-vs-actual
            # comparison below -- NOT what gets applied to the plant.
            with torch.no_grad():
                d_j_mm = d_j_frac * self.H_mm
                x_max_band_mm = (d_j_frac + self.band_width_frac) * self.H_mm
                nf = build_node_features(self.mesh_info, x0, d_j_frac, d_j_mm,
                                          x_max_band_mm, R_j_deg, u_j_frac).unsqueeze(0)
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

        return applied_controls, plan_vs_actual_rmse_mm, state
