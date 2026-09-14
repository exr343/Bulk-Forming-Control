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

    def __init__(self, mesh, R, H, T_linear_fn, ele_type="TET4"):
        self.mesh, self.R, self.H, self.T_linear_fn = mesh, R, H, T_linear_fn

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

    def step(self, state: ForgingState, hit: Hit, is_first_hit: bool = False) -> ForgingState:
        """Applies one forging hit, returning the resulting state. Raises
        RuntimeError on solver non-convergence (matches generate_dataset.py's
        own fail-fast convention -- no retry) rather than silently
        propagating a bad state.

        `is_first_hit`: True only for the very first hit off a fresh
        `reset()` -- skips the thermal relaxation ramp (nothing to relax
        from) and builds BCs on the reference config, matching
        generate_dataset.py's `j == 1` special case exactly."""
        current_sol_u, current_sol_dT, current_int_vars = state.sol_u, state.sol_dT, state.int_vars

        current_sol_u_for_bc = None if is_first_hit else current_sol_u
        dirichlet_bc_info_u, dirichlet_bc_info_T, location_fns_convect, bc_update_fn, surface_inds = \
            build_cylinder_press_bcs(self.mesh, R=self.R, H=self.H, hit=hit,
                                      current_sol_u=current_sol_u_for_bc, T_linear_fn=self.T_linear_fn)
        self.problem.dirichlet_bc_info = [dirichlet_bc_info_u, dirichlet_bc_info_T]
        self.problem.location_fns = location_fns_convect
        refresh_problem_surface_integrals(self.problem)

        if not is_first_hit:
            # 10-step thermal relaxation ramp -- identical to
            # generate_dataset.py's; settled (not skipped) so the plant
            # stays in-distribution with what the GNN was trained on.
            T_cold, T_hot = onp.array(current_sol_dT), onp.array(self._sol_dT_initial)
            n_ramp, relax_dt = 10, 1e-4
            for k in range(1, n_ramp + 1):
                frac = k / n_ramp
                T_blend = np.array((1.0 - frac) * T_cold + frac * T_hot)
                relax_stepper = AutomaticTimeStepperTM(
                    self.problem, total_time=relax_dt, initial_dt=relax_dt,
                    min_dt=1e-8, max_dt=relax_dt, max_retries=10,
                    increase_factor=1.0, decrease_factor=0.5, line_search_after=0,
                    cool_factor=0.0, surface_inds=surface_inds,
                )
                relax_stepper.seed_state(current_sol_u, T_blend, current_int_vars)
                ok_relax = relax_stepper.run(
                    bc_update_fn=bc_update_fn, bc_params_fn=lambda step, scale: (step, 0.0),
                    rho_ini=np.array([1.0, 1.0, 1.0, 1.0]), vtk_dir=None, save_every=1,
                )
                if not ok_relax:
                    raise RuntimeError(f"Thermal relaxation failed at ramp step {k}/{n_ramp}: "
                                        f"{relax_stepper.step_report[-5:]}")
                current_sol_u, current_int_vars = relax_stepper.sol_u, relax_stepper.int_vars
            current_sol_dT = self._sol_dT_initial

            dirichlet_bc_info_u, dirichlet_bc_info_T, location_fns_convect, bc_update_fn, surface_inds = \
                build_cylinder_press_bcs(self.mesh, R=self.R, H=self.H, hit=hit,
                                          current_sol_u=current_sol_u, T_linear_fn=self.T_linear_fn)
            self.problem.dirichlet_bc_info = [dirichlet_bc_info_u, dirichlet_bc_info_T]
            self.problem.location_fns = location_fns_convect
            refresh_problem_surface_integrals(self.problem)

        stepper = AutomaticTimeStepperTM(
            self.problem, total_time=hit.total_time, initial_dt=1e-3,
            min_dt=1e-6, max_dt=0.05, max_retries=10,
            increase_factor=1.3, decrease_factor=0.5, line_search_after=0,
            cool_factor=0.8, surface_inds=surface_inds,
        )
        stepper.seed_state(current_sol_u, current_sol_dT, current_int_vars)
        ok = stepper.run(
            bc_update_fn=bc_update_fn, bc_params_fn=lambda step, scale: (step, scale),
            rho_ini=np.array([1.0, 1.0, 1.0, 1.0]), vtk_dir=None, save_every=1,
        )
        if not ok:
            raise RuntimeError(f"Hit failed to converge: {stepper.step_report[-5:]}")

        return ForgingState(stepper.sol_u, stepper.sol_dT, stepper.int_vars)
