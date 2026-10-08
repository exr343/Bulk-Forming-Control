"""Thin wrapper exposing the Agility_Forge forging simulation as an MPC "plant".

Wraps applications.Agility_Forge.lib.constitutive.ThermalMechanical +
applications.Agility_Forge.lib.time_stepper.AutomaticTimeStepperTM behind a
step(state, hit) -> next_state interface. Mirrors generate_dataset.py's
per-hit loop body exactly (same mesh, same BC rebuild, same thermal
relaxation ramp) -- settled by explicit interview: the GNN surrogate was
trained exclusively on data generated WITH the ramp, so skipping it here
would make every real state the MPC re-plans from out-of-distribution.
"""

import os

import numpy as onp
import jax.numpy as np
from jax import config

config.update("jax_enable_x64", True)

try:
    import open3d as o3d
except ImportError:
    o3d = None
    print("[Warning] open3d not available, ForgingPlant mesh loading is disabled")

from applications.Agility_Forge.hit_config import Hit
from applications.Agility_Forge.mesh_container import MeshContainer
from applications.Agility_Forge.lib.constitutive import ThermalMechanical
from applications.Agility_Forge.lib.time_stepper import AutomaticTimeStepperTM
from applications.Agility_Forge.lib.boundary_conditions import (
    build_cylinder_press_bcs,
    refresh_problem_surface_integrals,
)
from applications.Agility_Forge.generate_dataset import open3d_to_json, convert_meshio_to_jaxfem


def load_default_billet(ele_type="TET4"):
    """Loads the exact same billet mesh + initial temperature gradient
    generate_dataset.py uses -- required so the GNN's mesh_info (built from a
    .vtu this same mesh produced) has node ordering consistent with this
    plant's sol_u array. Returns (mesh, R, H, T_linear_fn)."""
    agility_forge_dir = os.path.dirname(os.path.dirname(__file__))
    assets_dir = os.path.join(agility_forge_dir, "data", "msh", "jax_forge", "assets")
    stock_mesh_path = os.path.join(assets_dir, "0104-00_jaxforge_stock.obj")
    json_mesh_path = os.path.join(assets_dir, "0104-00_jaxforge_stock.json")

    mesh_0 = o3d.io.read_triangle_mesh(str(stock_mesh_path))
    open3d_to_json(mesh_0, json_mesh_path)
    stock_mesh = MeshContainer.from_json(json_mesh_path)
    mesh = convert_meshio_to_jaxfem(stock_mesh.vtk, ele_type=ele_type)

    H = np.max(mesh.points[:, 0])
    R = np.max(mesh.points[:, 1])

    # Same constants as generate_dataset.py's "Initial temperature profile"
    # section -- see that module for the physical rationale.
    T_top, T_bot, x_top = 1096.0, 676.0, 72.0
    x_min = np.min(mesh.points[:, 0])

    def T_linear_fn(point):
        x = point[0]
        if x > x_top:
            return T_top
        xc = np.clip(x, x_min, x_top)
        return T_bot + (T_top - T_bot) * ((xc - x_min) / (x_top - x_min))

    return mesh, R, H, T_linear_fn


# Coil reheat (new simulator, agreed 2026-10-03 -- README "The reheat"): an
# induction coil heats a COIL_LENGTH_MM stretch of bar to COIL_T_C; outside it
# the temperature drops by COIL_SLOPE_C_PER_MM on both sides. Both come from
# the starting profile above: flat 1096 C from x=72 mm to the free end
# (24.5 mm), dropping (1096 - 676) / (72 - (-5)) C/mm toward the clamp.
COIL_T_C = 1096.0
COIL_LENGTH_MM = 24.5
COIL_SLOPE_C_PER_MM = (1096.0 - 676.0) / (72.0 - (-5.0))


def coil_temperature(points, sol_u, coil_center_mm):
    """(n_nodes, 1) coil temperature profile centred at `coil_center_mm`,
    from each node's CURRENT x (points + sol_u). A reheat applies it with
    reheat_temperature (keep the hotter); the first reheat of a run, on the
    fresh billet, uses it as is."""
    x = onp.asarray(points)[:, 0] + onp.asarray(sol_u)[:, 0]
    dist = onp.maximum(onp.abs(x - coil_center_mm) - 0.5 * COIL_LENGTH_MM, 0.0)
    return (COIL_T_C - COIL_SLOPE_C_PER_MM * dist)[:, None]


def reheat_temperature(points, sol_u, sol_dT, coil_center_mm):
    """Temperature after a coil reheat (rule settled 2026-10-04): each node
    keeps the hotter of its current temperature and the coil profile, so a
    reheat only adds heat. (Overwriting with the coil profile cooled the far
    end of long bars by ~300 C at once, which the solver could not handle.)"""
    return onp.maximum(onp.asarray(sol_dT), coil_temperature(points, sol_u, coil_center_mm))


def extract_surface_state(sol_u, mesh_info):
    """Full-volumetric sol_u (jax array) -> surface-only displacement
    (torch tensor), matching GNN/data.py's ForgeGNNDataset._load_surface_
    displacement (which does the same subset extraction from a saved .vtu
    instead of a live plant state)."""
    import torch
    return torch.tensor(onp.array(sol_u)[mesh_info.full_to_surface], dtype=torch.float32)


class ForgingState:
    """Plain-NumPy snapshot of the plant state: (sol_u, sol_dT, int_vars)."""

    def __init__(self, sol_u, sol_dT, int_vars):
        self.sol_u = sol_u
        self.sol_dT = sol_dT
        self.int_vars = int_vars


class ForgingPlant:
    """MPC-facing wrapper around one shared, reused ThermalMechanical problem
    instance -- mirrors generate_dataset.py's "Build problem once" section."""

    def __init__(self, mesh, R, H, T_linear_fn, ele_type="TET4", linear_solver="jax"):
        self.mesh, self.R, self.H, self.T_linear_fn = mesh, R, H, T_linear_fn
        self.linear_solver = linear_solver   # "jax" (BiCGSTAB) or "scipy" (direct, slower but robust)

        def _no_location(p):
            return False

        self.problem = ThermalMechanical(
            mesh=[mesh, mesh], vec=[3, 1], dim=3,
            ele_type=[ele_type, ele_type], gauss_order=[2, 2],
            dirichlet_bc_info=[None, None], location_fns=[_no_location, _no_location],
        )
        self._pristine_int_vars = self.problem.internal_vars

        pts = onp.array(mesh.points)
        sol_dT0 = onp.zeros((len(pts), 1), dtype=onp.float64)
        for i in range(len(pts)):
            sol_dT0[i, 0] = float(T_linear_fn(np.array(pts[i])))
        self._sol_dT_initial = np.array(sol_dT0)
        self._sol_u0 = np.zeros((len(pts), 3))

    def reset(self) -> ForgingState:
        """Undeformed billet, initial temperature gradient, pristine (never-
        mutated) internal plasticity variables."""
        return ForgingState(self._sol_u0, self._sol_dT_initial, self._pristine_int_vars)

    def _apply_bcs(self, hit, sol_u_for_bc):
        """Builds the hit's contact/thermal BCs on `sol_u_for_bc` (None = the
        reference config) and installs them on the shared problem."""
        dirichlet_bc_info_u, dirichlet_bc_info_T, location_fns_convect, bc_update_fn, surface_inds = \
            build_cylinder_press_bcs(self.mesh, R=self.R, H=self.H, hit=hit,
                                      current_sol_u=sol_u_for_bc, T_linear_fn=self.T_linear_fn)
        self.problem.dirichlet_bc_info = [dirichlet_bc_info_u, dirichlet_bc_info_T]
        self.problem.location_fns = location_fns_convect
        refresh_problem_surface_integrals(self.problem)
        return bc_update_fn, surface_inds

    def _ramp_temperature(self, sol_u, int_vars, T_from, T_to, n_ramp, bc_update_fn, surface_inds):
        """Thermal relaxation ramp: blends T_old from T_from to T_to in n_ramp
        tiny (1e-4 s, no cooling) steps, re-equilibrating the mechanics after
        each, since changing temperature in one shot stalls Newton. Returns
        the relaxed (sol_u, int_vars); the ramp's own temperatures are
        discarded (the caller sets the final temperature)."""
        T_cold, T_hot = onp.array(T_from), onp.array(T_to)
        relax_dt = 1e-4
        for k in range(1, n_ramp + 1):
            frac = k / n_ramp
            T_blend = np.array((1.0 - frac) * T_cold + frac * T_hot)
            relax_stepper = AutomaticTimeStepperTM(
                self.problem, total_time=relax_dt, initial_dt=relax_dt,
                min_dt=1e-8, max_dt=relax_dt, max_retries=10,
                increase_factor=1.0, decrease_factor=0.5, line_search_after=0,
                cool_factor=0.0, surface_inds=surface_inds, linear_solver=self.linear_solver,
            )
            relax_stepper.seed_state(sol_u, T_blend, int_vars)
            ok_relax = relax_stepper.run(
                bc_update_fn=bc_update_fn, bc_params_fn=lambda step, scale: (step, 0.0),
                rho_ini=np.array([1.0, 1.0, 1.0, 1.0]), vtk_dir=None, save_every=1,
            )
            if not ok_relax:
                raise RuntimeError(f"Thermal relaxation failed at ramp step {k}/{n_ramp}: "
                                    f"{relax_stepper.step_report[-5:]}")
            sol_u, int_vars = relax_stepper.sol_u, relax_stepper.int_vars
        return sol_u, int_vars

    def reheat(self, state: ForgingState, T_new, hit: Hit, max_ramp_step_C: float = 50.0) -> ForgingState:
        """Coil reheat (new simulator): sets the temperature to `T_new` (see
        coil_temperature) through the same relaxation ramp step() uses. The
        ramp has 10 steps, or more if some node would change by more than
        `max_ramp_step_C` per step -- the old per-hit ramps already moved
        nodes by up to ~530 C in 10 steps (~53 C/step), so 50 keeps every
        step within what has converged before. `hit` (the first hit after the
        reheat) only places the die during the ramp, touching but not moving,
        as step()'s ramp does. Follow with step(..., reheat=False)."""
        bc_update_fn, surface_inds = self._apply_bcs(hit, state.sol_u)
        jump = float(onp.max(onp.abs(onp.asarray(T_new) - onp.asarray(state.sol_dT))))
        n_ramp = max(10, int(onp.ceil(jump / max_ramp_step_C)))
        self.last_reheat_info = {"max_jump_C": jump, "n_ramp": n_ramp}
        print(f"[reheat] max temperature change {jump:.1f} C -> {n_ramp} ramp steps")
        sol_u, int_vars = self._ramp_temperature(state.sol_u, state.int_vars, state.sol_dT, T_new,
                                                 n_ramp, bc_update_fn, surface_inds)
        return ForgingState(sol_u, np.array(T_new), int_vars)

    def step(self, state: ForgingState, hit: Hit, is_first_hit: bool = False, reheat: bool = True) -> ForgingState:
        """Applies one forging hit, returning the resulting state. Raises
        RuntimeError on solver non-convergence (matches generate_dataset.py's
        own fail-fast convention -- no retry) rather than silently
        propagating a bad state.

        `is_first_hit`: True only for the very first hit off a fresh
        `reset()` -- skips the thermal relaxation ramp (nothing to relax
        from) and builds BCs on the reference config, matching
        generate_dataset.py's `j == 1` special case exactly.

        `reheat`: True (old simulator, all data before 2026-10-04) ramps the
        temperature back to the starting profile before the hit. False (new
        simulator) carries the temperature over from the previous hit; coil
        reheats are applied separately with reheat()."""
        current_sol_u, current_sol_dT, current_int_vars = state.sol_u, state.sol_dT, state.int_vars

        bc_update_fn, surface_inds = self._apply_bcs(hit, None if is_first_hit else current_sol_u)

        if not is_first_hit and reheat:
            # 10-step thermal relaxation ramp -- identical to
            # generate_dataset.py's; settled (not skipped) so the plant
            # stays in-distribution with what the GNN was trained on.
            current_sol_u, current_int_vars = self._ramp_temperature(
                current_sol_u, current_int_vars, current_sol_dT, self._sol_dT_initial, 10,
                bc_update_fn, surface_inds)
            current_sol_dT = self._sol_dT_initial
            bc_update_fn, surface_inds = self._apply_bcs(hit, current_sol_u)

        stepper = AutomaticTimeStepperTM(
            self.problem, total_time=hit.total_time, initial_dt=1e-3,
            min_dt=1e-6, max_dt=0.05, max_retries=10,
            increase_factor=1.3, decrease_factor=0.5, line_search_after=0,
            cool_factor=0.8, surface_inds=surface_inds, linear_solver=self.linear_solver,
        )
        stepper.seed_state(current_sol_u, current_sol_dT, current_int_vars)
        ok = stepper.run(
            bc_update_fn=bc_update_fn, bc_params_fn=lambda step, scale: (step, scale),
            rho_ini=np.array([1.0, 1.0, 1.0, 1.0]), vtk_dir=None, save_every=1,
        )
        if not ok:
            raise RuntimeError(f"Hit failed to converge: {stepper.step_report[-5:]}")

        return ForgingState(stepper.sol_u, stepper.sol_dT, stepper.int_vars)
