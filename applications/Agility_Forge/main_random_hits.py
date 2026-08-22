"""Driver for a 10-hit thermo-mechanical forging simulation with a randomized
hit schedule.

Based on ``applications/Agility_Forge/main.py`` (Version 8, multi-hit
rotational forging with Updated-Lagrangian dynamic contact / "Route B").
Reuses that file's mesh loading, temperature profile, thermal-relaxation
handoff, and adaptive time-stepping unchanged; the only thing this file
changes is how the hit schedule itself is built.

Randomized schedule
--------------------
  - Number of hits: 10 (configurable via --n-hits).
  - Axial contact-band WIDTH is the same for every hit (0.2 * H, matching
    the fixed-width bands used in main.py) — "equal applied bands per hit".
  - Axial contact-band POSITION (x_min_band, as a fraction of H) is drawn
    independently per hit from Uniform(edge_margin_frac, 1 - band_width_frac -
    edge_margin_frac), so the band stays within the bar's forged length
    [0, H] for every hit, with a margin (default 2% of H, --edge-margin-frac)
    kept from both ends to avoid degenerate contact right at the tip.
  - Rotation about the bar's x-axis (rotation_euler_x) is drawn independently
    per hit from Uniform(0, 360) degrees.
  - Die stroke (compression_displacement) and hit duration (total_time) are
    NOT randomized: every hit uses the same fixed values, so the only
    randomized quantities are axial position and rotation, as requested.
  - The draw is seeded (default 42, override with --seed) so a given seed
    reproduces the exact same schedule; the realized schedule (fractional
    and mm bands, rotation, stroke, duration) is printed and written to
    ``data/random_hits/hit_schedule.json`` before the run starts.

Usage:
    python -m applications.Agility_Forge.main_random_hits
    python -m applications.Agility_Forge.main_random_hits --seed 7 --n-hits 10
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
import numpy as onp
import jax
import jax.numpy as np

from jax import config
config.update("jax_enable_x64", True)

_ = jax.devices()

from jax_forge.generate_mesh import get_meshio_cell_type, Mesh

from applications.Agility_Forge.mesh_container import MeshContainer

from applications.Agility_Forge.hit_config import Hit
from applications.Agility_Forge.lib.constitutive import ThermalMechanical
from applications.Agility_Forge.lib.time_stepper import AutomaticTimeStepperTM
from applications.Agility_Forge.lib.boundary_conditions import build_cylinder_press_bcs, refresh_problem_surface_integrals


# =============================================================================
# Mesh utilities (unchanged from main.py)
# =============================================================================
def open3d_to_json(o3d_mesh, json_path):
    """Convert ForgeDataset mesh (Open3D) to the JSON format expected by jax-forge."""
    vertices = np.asarray(o3d_mesh.vertices)
    faces = np.asarray(o3d_mesh.triangles)
    data = {
        "Vertices": vertices.flatten().tolist(),
        "Triangles": faces.flatten().tolist(),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def convert_meshio_to_jaxfem(meshio_mesh, ele_type="TET4"):
    """Convert a meshio mesh to a jax-fem Mesh object."""
    print("Converting to jax-fem mesh...")
    cell_type = get_meshio_cell_type(ele_type)
    return Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])


# =============================================================================
# Randomized hit-schedule generation
# =============================================================================
def generate_random_hits(n_hits=10, band_width_frac=0.2, compression_displacement=2.5,
                          total_time=0.8, rotation_range=(0.0, 360.0), seed=42,
                          edge_margin_frac=0.02):
    """Build a list of ``Hit`` objects with a fixed band width, fixed stroke,
    and fixed duration, but a randomly drawn axial band position and rotation
    for every hit.

    Parameters
    ----------
    n_hits : int
        Number of hits to generate.
    band_width_frac : float
        Axial contact-band width as a fraction of H, identical for every hit.
    compression_displacement : float
        Die stroke per side (mm), identical for every hit.
    total_time : float
        Hit duration (s), identical for every hit.
    rotation_range : (float, float)
        Inclusive range (degrees) that rotation_euler_x is drawn from.
    seed : int
        Seed for the random draw; the same seed reproduces the same schedule.
    edge_margin_frac : float
        Minimum distance (fraction of H) kept between the band and either
        stock end. A band sampled right at x=0 or x=H sits on a mesh tip
        already distorted by prior hits, which can produce a Newton
        divergence that no dt-halving recovers from.

    Returns
    -------
    hits : list[Hit]
    schedule : list[dict]
        Plain-dict record of each hit's parameters (for logging/reporting).
    """
    rng = random.Random(seed)
    max_start_frac = 1.0 - band_width_frac
    if max_start_frac < 0.0:
        raise ValueError(f"band_width_frac={band_width_frac} exceeds 1.0")

    x_min_frac_lo = edge_margin_frac
    x_min_frac_hi = max_start_frac - edge_margin_frac
    if x_min_frac_hi < x_min_frac_lo:
        raise ValueError(
            f"edge_margin_frac={edge_margin_frac} leaves no valid sampling range "
            f"for band_width_frac={band_width_frac} (need 2*edge_margin_frac < {max_start_frac})"
        )

    hits = []
    schedule = []
    for i in range(n_hits):
        x_min_frac = rng.uniform(x_min_frac_lo, x_min_frac_hi)
        x_max_frac = x_min_frac + band_width_frac
        rotation = rng.uniform(*rotation_range)

        hit = Hit(
            x_min_band=x_min_frac,
            x_max_band=x_max_frac,
            compression_displacement=compression_displacement,
            rotation_euler_x=rotation,
            total_time=total_time,
        )
        hits.append(hit)
        schedule.append({
            "hit": i + 1,
            "x_min_band_frac": x_min_frac,
            "x_max_band_frac": x_max_frac,
            "band_width_frac": band_width_frac,
            "compression_displacement_mm": compression_displacement,
            "rotation_euler_x_deg": rotation,
            "total_time_s": total_time,
        })

    return hits, schedule


# =============================================================================
# Main thermo-mechanical driver for randomized multi-hit forging simulation
# =============================================================================
def run_thermo_mech_cylinder_press_random_hits(n_hits=10, seed=42, band_width_frac=0.2,
                                                compression_displacement=2.5, total_time=0.8,
                                                edge_margin_frac=0.02):
    """
    Run a randomized multi-hit forging simulation:
      - n_hits hits, applied sequentially.
      - Every hit uses the same axial band WIDTH, die stroke, and duration.
      - Every hit's axial band POSITION and rotation about the bar's x-axis
        are drawn independently at random (seeded for reproducibility).

    Key: F and alpha (plasticity history) carried forward; T reset;
         displacement continuity via ramp formula. Identical mechanics to
         main.py — only the hit schedule differs.
    """
    # Directories
    crt_file_path = os.path.dirname(__file__)
    data_dir = os.path.join(crt_file_path, "data")
    msh_dir = os.path.join(data_dir, "msh/jax_forge")
    vtk_dir = os.path.join(data_dir, "vtk_tm_press_T_convect_tet_random_hits")
    random_out_dir = os.path.join(data_dir, "random_hits")
    assets_dir = os.path.join(msh_dir, "assets")
    stock_mesh_path = os.path.join(assets_dir, "0104-00_jaxforge_stock.obj")
    json_mesh_path = os.path.join(assets_dir, "0104-00_jaxforge_stock.json")

    os.makedirs(vtk_dir, exist_ok=True)
    os.makedirs(random_out_dir, exist_ok=True)

    # clean VTK directory (this run's dedicated subfolder only)
    for item in glob.glob(os.path.join(vtk_dir, "*")):
        try:
            if os.path.isdir(item):
                shutil.rmtree(item)
            else:
                os.remove(item)
        except (OSError, Exception) as e:
            print(f"Warning: Could not remove {item}: {e}")

    # clean per-hit step/scale plots and schedule log from previous runs
    # (this run's dedicated subfolder only — does not touch main.py's output)
    for pattern in ("hit_*_step_data.png", "hit_*_step_data_scales.txt", "hit_schedule.json"):
        for item in glob.glob(os.path.join(random_out_dir, pattern)):
            try:
                os.remove(item)
            except OSError as e:
                print(f"Warning: Could not remove {item}: {e}")

    # === Build the randomized hit schedule ===
    hits, schedule = generate_random_hits(
        n_hits=n_hits,
        band_width_frac=band_width_frac,
        compression_displacement=compression_displacement,
        total_time=total_time,
        seed=seed,
        edge_margin_frac=edge_margin_frac,
    )

    # Load initial mesh
    mesh_0 = o3d.io.read_triangle_mesh(str(stock_mesh_path))
    open3d_to_json(mesh_0, json_mesh_path)
    stock_mesh = MeshContainer.from_json(json_mesh_path)
    ele_type = "TET4"
    mesh = convert_meshio_to_jaxfem(stock_mesh.vtk, ele_type=ele_type)

    # Mesh dimensions
    H = np.max(mesh.points[:, 0])
    R_y = np.max(mesh.points[:, 1])
    R_z = np.max(mesh.points[:, 2])
    R = R_y

    print("H={0}, R_y={1}, R_z={2}".format(H, R_y, R_z))

    # Now that H is known, record the schedule in mm and print/save it.
    for rec, hit in zip(schedule, hits):
        rec["x_min_band_mm"] = float(hit.x_min_band * float(H))
        rec["x_max_band_mm"] = float(hit.x_max_band * float(H))

    print("\n" + "=" * 80)
    print(f"RANDOMIZED HIT SCHEDULE (seed={seed}, n_hits={n_hits})")
    print("=" * 80)
    for rec in schedule:
        print(f"  Hit {rec['hit']:2d}: axial band = [{rec['x_min_band_mm']:.2f}, "
              f"{rec['x_max_band_mm']:.2f}] mm, rotation = {rec['rotation_euler_x_deg']:.2f} deg, "
              f"stroke = {rec['compression_displacement_mm']} mm, "
              f"duration = {rec['total_time_s']} s")

    schedule_path = os.path.join(random_out_dir, "hit_schedule.json")
    with open(schedule_path, "w", encoding="utf-8") as f:
        json.dump({"seed": seed, "n_hits": n_hits, "hits": schedule}, f, indent=2)
    print(f"\nSchedule written to: {schedule_path}")

    # === Initial temperature profile ===
    T_top = 1096.0  # °C
    T_bot = 676.0   # °C
    x_min = np.min(mesh.points[:, 0])
    x_top = 72.0

    def T_linear_x(point):
        """Piecewise temperature profile along x."""
        x = point[0]
        if x > x_top:
            return T_top
        xc = np.clip(x, x_min, x_top)
        return T_bot + (T_top - T_bot) * ((xc - x_min) / (x_top - x_min))

    # === Build initial temperature field ===
    pts = onp.array(mesh.points)
    sol_dT_initial = onp.zeros((len(pts), 1), dtype=onp.float64)
    for i in range(len(pts)):
        sol_dT_initial[i, 0] = float(T_linear_x(np.array(pts[i])))

    sol_dT0 = np.array(sol_dT_initial)
    sol_u0 = np.zeros((len(mesh.points), 3))

    # === Build problem once ===
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

    # === Multi-hit loop (identical mechanics to main.py; hits list is now
    #     the randomized schedule, and the loop already generalizes to any
    #     number of hits) ===
    current_sol_u = sol_u0
    current_sol_dT = sol_dT0
    current_int_vars = problem.internal_vars

    for hit_idx, hit in enumerate(hits):
        print("\n" + "=" * 80)
        print(f"HIT {hit_idx + 1}/{len(hits)}")
        print("=" * 80)
        print(f"Hit params: x_band=[{hit.x_min_band:.4f}, {hit.x_max_band:.4f}]*H, "
              f"disp={hit.compression_displacement}, rot_x={hit.rotation_euler_x:.2f}")

        current_sol_u_for_bc = current_sol_u if hit_idx > 0 else None
        dirichlet_bc_info_u, dirichlet_bc_info_T, location_fns_convect, bc_update_fn, surface_inds = \
            build_cylinder_press_bcs(mesh, R=R, H=H, hit=hit,
                                     current_sol_u=current_sol_u_for_bc, T_linear_fn=T_linear_x)

        problem.dirichlet_bc_info = [dirichlet_bc_info_u, dirichlet_bc_info_T]
        problem.location_fns = location_fns_convect
        refresh_problem_surface_integrals(problem)

        # Reset temperature between hits
        if hit_idx > 0:
            print("\nResetting temperature to initial profile (cool-down between hits)...")

            # === Thermal-state relaxation by gradual temperature ramp ===
            # See main.py for the full rationale: carried mechanical history
            # was equilibrated under the previous hit's cooled temperature,
            # so T_old is ramped gradually rather than reset in one shot.
            T_cold = onp.array(current_sol_dT)        # previous hit's cooled end-state
            T_hot = onp.array(sol_dT_initial)         # reheated target profile
            n_ramp = 10
            relax_dt = 1e-4
            print(f"\nRunning thermal-state relaxation: {n_ramp}-step temperature "
                  f"ramp (cold -> reheated), scale=0, no contact motion...")

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
                    cool_factor=0.0,            # no extra cooling during ramp
                    surface_inds=surface_inds,
                )
                relax_stepper.seed_state(current_sol_u, T_blend, current_int_vars)

                ok_relax = relax_stepper.run(
                    bc_update_fn=bc_update_fn,
                    bc_params_fn=lambda step, scale: (step, 0.0),  # force scale=0
                    rho_ini=np.array([1.0, 1.0, 1.0, 1.0]),
                    vtk_dir=None,
                    save_every=1,
                )
                if not ok_relax:
                    print(f"\n[ERROR] Hit {hit_idx + 1} thermal-state relaxation "
                          f"failed at ramp step {k}/{n_ramp}!")
                    print(f"Report: {relax_stepper.step_report[-5:]}")
                    return False

                current_sol_u = relax_stepper.sol_u
                current_int_vars = relax_stepper.int_vars

            current_sol_dT = sol_dT_initial
            print("Relaxation complete: mechanical state re-equilibrated under reheated T_old.")

            # Rebuild the contact BCs on the POST-relaxation surface.
            dirichlet_bc_info_u, dirichlet_bc_info_T, location_fns_convect, bc_update_fn, surface_inds = \
                build_cylinder_press_bcs(mesh, R=R, H=H, hit=hit,
                                         current_sol_u=current_sol_u, T_linear_fn=T_linear_x)
            problem.dirichlet_bc_info = [dirichlet_bc_info_u, dirichlet_bc_info_T]
            problem.location_fns = location_fns_convect
            refresh_problem_surface_integrals(problem)

        # Create stepper
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

        # Run
        print(f"\nStarting hit {hit_idx + 1} solver...")
        rho_ini = np.array([1.0, 1.0, 1.0, 1.0])
        ok = stepper.run(
            bc_update_fn=bc_update_fn,
            bc_params_fn=lambda step, scale: (step, scale),
            rho_ini=rho_ini,
            vtk_dir=os.path.join(vtk_dir, f"hit_{hit_idx + 1}"),
            save_every=1,
        )

        if not ok:
            print(f"\n[ERROR] Hit {hit_idx + 1} solver failed to converge!")
            print(f"Report: {stepper.step_report[-5:]}")
            return False

        # Extract state at end of hit
        current_sol_u = stepper.sol_u
        current_sol_dT = stepper.sol_dT
        current_int_vars = stepper.int_vars

        print(f"\nHit {hit_idx + 1} completed successfully")
        print(f"  Converged in {stepper.step_count} steps")
        print(f"  Wall time: {stepper.total_wall:.2f}s")
        print(f"  Internal vars shape: F={current_int_vars[0].shape}, alpha={current_int_vars[2].shape}")

        plot_path = os.path.join(random_out_dir, f"hit_{hit_idx + 1}_step_data.png")
        stepper.plot_step_data(save_path=plot_path)

    # === Summary ===
    print("\n" + "=" * 80)
    print("ALL HITS COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"Total VTK output dir: {vtk_dir}")
    print(f"Final displacement norm: {np.linalg.norm(current_sol_u):.6e}")
    print(f"Final temp min/max: {np.min(current_sol_dT):.2f} / {np.max(current_sol_dT):.2f} °C")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-hits", type=int, default=10, help="Number of hits to apply sequentially.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the hit schedule.")
    parser.add_argument("--band-width-frac", type=float, default=0.2,
                         help="Axial contact-band width as a fraction of H (same for every hit).")
    parser.add_argument("--compression-displacement", type=float, default=2.5,
                         help="Die stroke per side, mm (same for every hit).")
    parser.add_argument("--total-time", type=float, default=0.8,
                         help="Hit duration, s (same for every hit).")
    parser.add_argument("--edge-margin-frac", type=float, default=0.02,
                         help="Minimum distance (fraction of H) kept between the band and either "
                              "stock end when sampling the band position, to avoid degenerate "
                              "contact at the tip.")
    args = parser.parse_args()

    run_thermo_mech_cylinder_press_random_hits(
        n_hits=args.n_hits,
        seed=args.seed,
        band_width_frac=args.band_width_frac,
        compression_displacement=args.compression_displacement,
        total_time=args.total_time,
        edge_margin_frac=args.edge_margin_frac,
    )
