"""First real-plant validation: replay the already-computed 5-step MPC plan
(from mpc.MPCController.plan(), run against the GNN surrogate targeting
rollout_339's true hit-5 geometry) through the REAL JAX-FORGE plant,
open-loop (no re-planning between hits -- cheaper than run()'s full
receding-horizon loop, while still exercising ForgingPlant across all 5 real
hits including the thermal relaxation ramp).

Answers the question flagged when the plan was first computed: the GNN
predicted 0.165mm RMSE against the target from this exact sequence, well
BELOW the 0.627mm the GNN itself gets replaying rollout_339's real controls
-- i.e. the plan may be exploiting the surrogate's own error structure
rather than describing an achievable real result. This script checks the
real number.

Usage: python -m applications.Agility_Forge.control.validate_open_loop
"""

import meshio
import numpy as onp
import torch

from applications.Agility_Forge.hit_config import Hit
from applications.Agility_Forge.GNN.data import build_surface_mesh_info
from applications.Agility_Forge.control.plant_interface import (
    ForgingPlant, load_default_billet, extract_surface_state,
)
from applications.Agility_Forge.control.mpc import U_J_MM_MIN, U_J_MM_MAX, TOTAL_TIME_S

# Re-planned after fixing mpc.py's H_mm bug (was using max-min of rest-x
# instead of generate_dataset.py's max(x) convention -- see chat history).
# (d_j_frac, R_j_deg, u_j_frac) per hit.
PLANNED_U_SEQ = [
    (0.2756858870876226, 190.05180450282566, 0.6823507673544683),
    (0.539420882768297, 142.7133738156345, 0.9999999129441168),
    (0.5728066016918203, 144.57585120265054, 0.5391884187101159),
    (0.2994256964133446, 111.4183460886947, 0.20959428997207905),
    (0.40813727382665455, 128.84469394925637, 3.470035560476882e-08),
]


def main():
    mesh, R, H, T_linear_fn = load_default_billet()
    plant = ForgingPlant(mesh, R, H, T_linear_fn)
    mesh_info = build_surface_mesh_info(
        "applications/Agility_Forge/data/dataset_pretraining/rollout_01/undeformed.vtu")

    state = plant.reset()
    for step_idx, (d_j_frac, R_j_deg, u_j_frac) in enumerate(PLANNED_U_SEQ):
        u_j_mm = U_J_MM_MIN + u_j_frac * (U_J_MM_MAX - U_J_MM_MIN)
        hit = Hit(x_min_band=d_j_frac, x_max_band=d_j_frac + 0.2,
                  compression_displacement=u_j_mm, rotation_euler_x=R_j_deg,
                  total_time=TOTAL_TIME_S)
        print(f"--- applying real hit {step_idx + 1}/5: d_j_frac={d_j_frac:.4f} "
              f"R_j_deg={R_j_deg:.2f} u_j_mm={u_j_mm:.3f} ---", flush=True)
        state = plant.step(state, hit, is_first_hit=(step_idx == 0))

    actual_final = extract_surface_state(state.sol_u, mesh_info)
    true_m = meshio.read(
        "applications/Agility_Forge/GNN/runs_3dim/eval/rollout_339/hit_05_true.vtu")
    target = torch.tensor(true_m.point_data["Displacement"], dtype=torch.float32)

    rmse = ((actual_final - target) ** 2).mean().sqrt().item()
    print(f"\nREAL plant final state vs. rollout_339 hit5 target -- RMSE: {rmse:.4f} mm")
    print("(GNN-internal prediction for this same plan was 0.152mm; "
          "GNN replaying rollout_339's real controls was 0.627mm)")

    out_mesh = meshio.Mesh(
        points=mesh_info.rest_pos.numpy(),
        cells=[("triangle", mesh_info.surf_faces_local)],
        point_data={"Displacement": onp.array(actual_final)},
    )
    out_mesh.write("applications/Agility_Forge/control/open_loop_validation_result.vtu")
    print("Saved: applications/Agility_Forge/control/open_loop_validation_result.vtu")


if __name__ == "__main__":
    main()
