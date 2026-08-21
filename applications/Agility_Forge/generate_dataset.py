"""Dataset generation for the hot-forging control pipeline.

Implements the scheme in "Data Generation for Hot Forging Control Pipeline":

    for i = 1, ..., 10
        Initialize x_i as an undeformed billet
        Save undeformed .vtu file associated with undeformed billet
        for j = 1, ..., 10
            Uniformly sample d_j (axial distance), R_j (axial orientation)
            u_j (depth of input) is fixed -- same value for every hit,
                             identical to main.py / main_random_hits.py
            Perform sequence of steps to apply d_j, R_j, u_j to x_i
            Save final step (.vtu file) of x_i associated with
                application of d_j, R_j, u_j
        end
    end

Mapping onto this codebase's `Hit` object ([hit_config.py]):
  - d_j (axial distance)    -> x_min_band (fraction of H); band WIDTH is
                                fixed (--band-width-frac, default 0.2) so d_j
                                alone determines the band's position.
  - R_j (axial orientation) -> rotation_euler_x, drawn from Uniform(0, 360) deg.
  - u_j (depth of input)    -> compression_displacement (mm). NOT sampled --
                                fixed at --compression-displacement (default
                                2.5 mm) for every hit, every rollout, exactly
                                matching main_random_hits.py's fixed stroke.
  - Hit duration (total_time) is NOT sampled by the pseudocode; it is fixed
    (--total-time, default 0.8 s) for every hit, same as main_random_hits.py.

Differences from main_random_hits.py (which this file is based on):
  - 10 independent ROLLOUTS (outer loop i): each starts from a fresh
    undeformed billet -- mechanical/thermal state is NOT carried across
    rollouts, only within a rollout's 10 hits (j loop).
  - Only the FINAL accepted step of each hit is written to disk (plus one
    undeformed snapshot per rollout) -- not every intermediate adaptive
    step. This matches "Save final step (.vtu file)" in the pseudocode and
    keeps the dataset's storage footprint to ~110 files total instead of
    ~2,800 (110 x ~28 adaptive steps/hit).
  - d_j and R_j are randomized per hit, same as main_random_hits.py; u_j
    (stroke depth) is fixed, also same as main_random_hits.py.

Note: no springback / die-retraction step is modeled, matching main.py and
main_random_hits.py exactly -- each hit's saved "final step" is the state
with the die still fully advanced (scale=1), not a released/unloaded state.

Output layout, under applications/Agility_Forge/data/dataset/:
  manifest.json                       -- one record per saved .vtu (see below)
  rollout_01/undeformed.vtu           -- x_1 before any hit
  rollout_01/hit_01_final.vtu         -- x_1 after hit 1 (d_1, R_1, u_1)
  rollout_01/hit_02_final.vtu         -- x_1 after hit 2
  ...
  rollout_01/hit_10_final.vtu
  rollout_02/undeformed.vtu           -- x_2, independent of rollout 1
  ...
  rollout_10/hit_10_final.vtu

manifest.json is rewritten after every saved file (not just at the end), so
a run that is interrupted partway still leaves a valid, complete record of
everything finished so far.

Usage:
    python -m applications.Agility_Forge.generate_dataset
    python -m applications.Agility_Forge.generate_dataset --seed 7 --n-rollouts 10 --n-hits 10
"""

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

try:
    import open3d as o3d
except ImportError:
    o3d = None
    print("[Warning] open3d not available, mesh conversion functions will be disabled")

import argparse
import json
import random
import shutil
import glob
import time
import numpy as onp
import jax
import jax.numpy as np

from jax import config
config.update("jax_enable_x64", True)

_ = jax.devices()
print(f"JAX devices: {jax.devices()}")

from jax_forge.generate_mesh import get_meshio_cell_type, Mesh
from jax_forge.utils import save_sol

from applications.Agility_Forge.mesh_container import MeshContainer

from applications.Agility_Forge.hit_config import Hit
from applications.Agility_Forge.lib.constitutive import ThermalMechanical
from applications.Agility_Forge.lib.time_stepper import AutomaticTimeStepperTM
from applications.Agility_Forge.lib.boundary_conditions import build_cylinder_press_bcs, refresh_problem_surface_integrals


# =============================================================================
# Mesh utilities (unchanged from main.py / main_random_hits.py)
# =============================================================================
def open3d_to_json(o3d_mesh, json_path):
    vertices = np.asarray(o3d_mesh.vertices)
    faces = np.asarray(o3d_mesh.triangles)
    data = {
        "Vertices": vertices.flatten().tolist(),
        "Triangles": faces.flatten().tolist(),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def convert_meshio_to_jaxfem(meshio_mesh, ele_type="TET4"):
    print("Converting to jax-fem mesh...")
    cell_type = get_meshio_cell_type(ele_type)
    return Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])


class ManifestWriter:
    """Accumulates dataset records and rewrites manifest.json after each one,
    so an interrupted run still leaves a complete record of finished work."""

    def __init__(self, path, seed, n_rollouts, n_hits, band_width_frac,
                 compression_displacement, total_time):
        self.path = path
        self.meta = {
            "seed": seed,
            "n_rollouts": n_rollouts,
            "n_hits_per_rollout": n_hits,
            "band_width_frac": band_width_frac,
            "compression_displacement_mm": compression_displacement,
            "total_time_s": total_time,
        }
        self.records = []
        self._flush()

    def add(self, record):
        self.records.append(record)
        self._flush()

    def _flush(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump({**self.meta, "records": self.records}, f, indent=2)


def run_dataset_generation(n_rollouts=10, n_hits=10, seed=42, band_width_frac=0.2,
                            compression_displacement=2.5, total_time=0.8):
    t_start = time.time()

    crt_file_path = os.path.dirname(__file__)
    data_dir = os.path.join(crt_file_path, "data")
    msh_dir = os.path.join(data_dir, "msh/jax_forge")
    dataset_dir = os.path.join(data_dir, "dataset")
    assets_dir = os.path.join(msh_dir, "assets")
    stock_mesh_path = os.path.join(assets_dir, "0104-00_jaxforge_stock.obj")
    json_mesh_path = os.path.join(assets_dir, "0104-00_jaxforge_stock.json")

    # Clean this script's own output directory only.
    if os.path.isdir(dataset_dir):
        shutil.rmtree(dataset_dir)
    os.makedirs(dataset_dir, exist_ok=True)

    rng = random.Random(seed)

    manifest = ManifestWriter(
        os.path.join(dataset_dir, "manifest.json"),
        seed=seed, n_rollouts=n_rollouts, n_hits=n_hits,
        band_width_frac=band_width_frac, compression_displacement=compression_displacement,
        total_time=total_time,
    )

    # === Load mesh (once, shared across all rollouts) ===
    mesh_0 = o3d.io.read_triangle_mesh(str(stock_mesh_path))
    open3d_to_json(mesh_0, json_mesh_path)
    stock_mesh = MeshContainer.from_json(json_mesh_path)
    ele_type = "TET4"
    mesh = convert_meshio_to_jaxfem(stock_mesh.vtk, ele_type=ele_type)

    H = np.max(mesh.points[:, 0])
    R_y = np.max(mesh.points[:, 1])
    R_z = np.max(mesh.points[:, 2])
    R = R_y
    print(f"H={H}, R_y={R_y}, R_z={R_z}")

    max_start_frac = 1.0 - band_width_frac
    if max_start_frac < 0.0:
        raise ValueError(f"band_width_frac={band_width_frac} exceeds 1.0")

    # === Initial temperature profile (same for every rollout) ===
    T_top = 1096.0
    T_bot = 676.0
    x_min = np.min(mesh.points[:, 0])
    x_top = 72.0

    def T_linear_x(point):
        x = point[0]
        if x > x_top:
            return T_top
        xc = np.clip(x, x_min, x_top)
        return T_bot + (T_top - T_bot) * ((xc - x_min) / (x_top - x_min))

    pts = onp.array(mesh.points)
    sol_dT_initial_np = onp.zeros((len(pts), 1), dtype=onp.float64)
    for i in range(len(pts)):
        sol_dT_initial_np[i, 0] = float(T_linear_x(np.array(pts[i])))
    sol_dT_initial = np.array(sol_dT_initial_np)
    sol_u0 = np.zeros((len(mesh.points), 3))

    # === Build problem once (reused/reset across all rollouts) ===
    print("\n" + "=" * 80)
    print("Building coupled thermo-mechanical problem...")
    print("=" * 80)

    def _no_location(p):
        return False

    problem = ThermalMechanical(
        mesh=[mesh, mesh],
        vec=[3, 1],
        dim=3,
        ele_type=[ele_type, ele_type],
        gauss_order=[2, 2],
        dirichlet_bc_info=[None, None],
        location_fns=[_no_location, _no_location]
    )
    pristine_int_vars = problem.internal_vars  # never mutated below; reset point for every rollout

    # =========================================================================
    # Outer loop: i = 1..n_rollouts -- each starts from a fresh undeformed billet
    # =========================================================================
    for i in range(1, n_rollouts + 1):
        rollout_dir = os.path.join(dataset_dir, f"rollout_{i:02d}")
        os.makedirs(rollout_dir, exist_ok=True)

        print("\n" + "#" * 80)
        print(f"ROLLOUT {i}/{n_rollouts}")
        print("#" * 80)

        current_sol_u = sol_u0
        current_sol_dT = sol_dT_initial
        current_int_vars = pristine_int_vars

        # --- Save undeformed billet ---
        undeformed_path = os.path.join(rollout_dir, "undeformed.vtu")
        save_sol(
            problem.fes[0],
            current_sol_u,
            undeformed_path,
            point_infos=[("Displacement", current_sol_u), ("Temperature", current_sol_dT)],
        )
        print(f"  Saved undeformed billet: {undeformed_path}")
        manifest.add({
            "rollout": i, "hit": 0, "kind": "undeformed",
            "vtu_path": os.path.relpath(undeformed_path, dataset_dir),
        })

        # =====================================================================
        # Inner loop: j = 1..n_hits -- applied sequentially to x_i
        # =====================================================================
        for j in range(1, n_hits + 1):
            t_hit_start = time.time()
            print("\n" + "=" * 80)
            print(f"ROLLOUT {i}/{n_rollouts}  HIT {j}/{n_hits}")
            print("=" * 80)

            # --- Uniformly sample d_j, R_j; u_j (depth of input) is fixed,
            #     identical to main.py / main_random_hits.py, not sampled ---
            d_j = rng.uniform(0.0, max_start_frac)        # axial distance (fraction of H)
            R_j = rng.uniform(0.0, 360.0)                 # axial orientation (deg)
            u_j = compression_displacement                 # depth of input (mm), fixed

            hit = Hit(
                x_min_band=d_j,
                x_max_band=d_j + band_width_frac,
                compression_displacement=u_j,
                rotation_euler_x=R_j,
                total_time=total_time,
            )
            print(f"  d_j={d_j:.4f} (x_band=[{d_j:.4f},{d_j + band_width_frac:.4f}]*H = "
                  f"[{d_j * float(H):.2f},{(d_j + band_width_frac) * float(H):.2f}] mm), "
                  f"R_j={R_j:.2f} deg, u_j={u_j:.3f} mm")

            current_sol_u_for_bc = current_sol_u if j > 1 else None
            dirichlet_bc_info_u, dirichlet_bc_info_T, location_fns_convect, bc_update_fn, surface_inds = \
                build_cylinder_press_bcs(mesh, R=R, H=H, hit=hit,
                                         current_sol_u=current_sol_u_for_bc, T_linear_fn=T_linear_x)

            problem.dirichlet_bc_info = [dirichlet_bc_info_u, dirichlet_bc_info_T]
            problem.location_fns = location_fns_convect
            refresh_problem_surface_integrals(problem)

            # --- Thermal-state relaxation before every hit after the first
            #     (within this rollout) -- identical to main_random_hits.py ---
            if j > 1:
                print("\nResetting temperature to initial profile (cool-down between hits)...")
                T_cold = onp.array(current_sol_dT)
                T_hot = onp.array(sol_dT_initial)
                n_ramp = 10
                relax_dt = 1e-4

                for k in range(1, n_ramp + 1):
                    frac = k / n_ramp
                    T_blend = np.array((1.0 - frac) * T_cold + frac * T_hot)
                    print(f"  [relax {k}/{n_ramp}] T_old ramp frac={frac:.2f}, "
                          f"T min/max = {float(np.min(T_blend)):.1f}/{float(np.max(T_blend)):.1f} C")

                    relax_stepper = AutomaticTimeStepperTM(
                        problem,
                        total_time=relax_dt,
                        initial_dt=relax_dt,
                        min_dt=1e-8,
                        max_dt=relax_dt,
                        max_retries=10,
                        increase_factor=1.0,
                        decrease_factor=0.5,
                        line_search_after=0,
                        cool_factor=0.0,
                        surface_inds=surface_inds,
                    )
                    relax_stepper.seed_state(current_sol_u, T_blend, current_int_vars)

                    ok_relax = relax_stepper.run(
                        bc_update_fn=bc_update_fn,
                        bc_params_fn=lambda step, scale: (step, 0.0),
                        rho_ini=np.array([1.0, 1.0, 1.0, 1.0]),
                        vtk_dir=None,
                        save_every=1,
                    )
                    if not ok_relax:
                        print(f"\n[ERROR] Rollout {i} hit {j} thermal-state relaxation "
                              f"failed at ramp step {k}/{n_ramp}!")
                        print(f"Report: {relax_stepper.step_report[-5:]}")
                        return False

                    current_sol_u = relax_stepper.sol_u
                    current_int_vars = relax_stepper.int_vars

                current_sol_dT = sol_dT_initial
                print("Relaxation complete: mechanical state re-equilibrated under reheated T_old.")

                dirichlet_bc_info_u, dirichlet_bc_info_T, location_fns_convect, bc_update_fn, surface_inds = \
                    build_cylinder_press_bcs(mesh, R=R, H=H, hit=hit,
                                             current_sol_u=current_sol_u, T_linear_fn=T_linear_x)
                problem.dirichlet_bc_info = [dirichlet_bc_info_u, dirichlet_bc_info_T]
                problem.location_fns = location_fns_convect
                refresh_problem_surface_integrals(problem)

            # --- Apply the hit; vtk_dir=None so no intermediate steps are written ---
            stepper = AutomaticTimeStepperTM(
                problem,
                total_time=hit.total_time,
                initial_dt=1e-3,
                min_dt=1e-6,
                max_dt=0.05,
                max_retries=10,
                increase_factor=1.3,
                decrease_factor=0.5,
                line_search_after=0,
                cool_factor=0.8,
                surface_inds=surface_inds,
            )
            stepper.seed_state(current_sol_u, current_sol_dT, current_int_vars)

            ok = stepper.run(
                bc_update_fn=bc_update_fn,
                bc_params_fn=lambda step, scale: (step, scale),
                rho_ini=np.array([1.0, 1.0, 1.0, 1.0]),
                vtk_dir=None,
                save_every=1,
            )

            if not ok:
                print(f"\n[ERROR] Rollout {i} hit {j} solver failed to converge!")
                print(f"Report: {stepper.step_report[-5:]}")
                return False

            current_sol_u = stepper.sol_u
            current_sol_dT = stepper.sol_dT
            current_int_vars = stepper.int_vars

            # --- Save only the final step of this hit ---
            hit_path = os.path.join(rollout_dir, f"hit_{j:02d}_final.vtu")
            save_sol(
                problem.fes[0],
                current_sol_u,
                hit_path,
                point_infos=[("Displacement", current_sol_u), ("Temperature", current_sol_dT)],
            )
            hit_wall = time.time() - t_hit_start
            print(f"  Rollout {i} hit {j} done in {stepper.step_count} steps, "
                  f"{hit_wall:.1f}s wall. Saved: {hit_path}")

            manifest.add({
                "rollout": i, "hit": j, "kind": "hit_final",
                "d_j_frac": d_j, "d_j_mm": d_j * float(H),
                "x_max_band_frac": d_j + band_width_frac,
                "x_max_band_mm": (d_j + band_width_frac) * float(H),
                "R_j_deg": R_j,
                "u_j_mm": u_j,
                "total_time_s": total_time,
                "step_count": stepper.step_count,
                "wall_time_s": hit_wall,
                "vtu_path": os.path.relpath(hit_path, dataset_dir),
            })

    total_wall = time.time() - t_start
    print("\n" + "#" * 80)
    print(f"DATASET GENERATION COMPLETE: {n_rollouts} rollouts x {n_hits} hits "
          f"= {n_rollouts * n_hits} hit samples + {n_rollouts} undeformed samples "
          f"= {n_rollouts * (n_hits + 1)} .vtu files")
    print(f"Total wall time: {total_wall / 3600:.2f} hours")
    print(f"Manifest: {os.path.join(dataset_dir, 'manifest.json')}")
    print("#" * 80)

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-rollouts", type=int, default=10, help="Number of independent rollouts (outer loop i).")
    parser.add_argument("--n-hits", type=int, default=10, help="Hits per rollout (inner loop j).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for d_j, R_j draws.")
    parser.add_argument("--band-width-frac", type=float, default=0.2,
                         help="Axial contact-band width as a fraction of H (fixed for every hit).")
    parser.add_argument("--compression-displacement", type=float, default=2.5,
                         help="u_j, depth of input (mm). NOT sampled -- fixed for every hit, "
                              "every rollout (default matches main_random_hits.py).")
    parser.add_argument("--total-time", type=float, default=0.8, help="Hit duration, s (fixed for every hit).")
    args = parser.parse_args()

    ok = run_dataset_generation(
        n_rollouts=args.n_rollouts,
        n_hits=args.n_hits,
        seed=args.seed,
        band_width_frac=args.band_width_frac,
        compression_displacement=args.compression_displacement,
        total_time=args.total_time,
    )
    if not ok:
        raise SystemExit(1)
