"""One-hit predictions on the real transitions of experiments 8 and 9: each
GNN predicts each applied hit from the simulator's TRUE pre-hit state (the
gap GNN gets the true applied half-gap), compared with the true post-hit
state. Skipped hits excluded. Writes one_hit_real.json.

Usage (repo root):
    PYTHONPATH=. JAX_PLATFORMS=cpu python applications/Agility_Forge/progress/control_reports/2026-09-27_cost_e10_e11_gap_diagnosis/one_hit_real.py
"""

import json
import os

import meshio
import numpy as np
import torch

from applications.Agility_Forge.GNN.data import band_half_thickness, build_node_features, build_surface_mesh_info, gap_frac
from applications.Agility_Forge.GNN.model import ForgeGNN
from applications.Agility_Forge.control.mpc import gnn_u_frac

HERE = os.path.dirname(os.path.abspath(__file__))
RS = "applications/Agility_Forge/control/results/real_simulator"
RUNS = {"Experiment 8": f"{RS}/e8_ideal_share25", "Experiment 9": f"{RS}/e9_ideal_share25_gap"}
MODELS = {"Stroke GNN": ("applications/Agility_Forge/GNN/mp_sweep/finetune/mp_3/checkpoint.pt", "stroke"),
          "Gap + no-change GNN": ("applications/Agility_Forge/GNN/gap_control/gap_nochange10/finetune/checkpoint.pt",
                                  "gap")}


def main():
    mi = build_surface_mesh_info("applications/Agility_Forge/data/dataset_finetuning/square/rollout_01/undeformed.vtu")
    H = float(mi.rest_pos[:, 0].max())
    out = {}
    for run, d in RUNS.items():
        with open(os.path.join(d, "results.json")) as f:
            steps = json.load(f)["steps"]
        disp = lambda h: torch.tensor(meshio.read(os.path.join(d, f"step_{h:02d}.vtu")).point_data["Displacement"])
        out[run] = {}
        for name, (path, control) in MODELS.items():
            ck = torch.load(path, map_location="cpu", weights_only=False)
            a = ck["args"]
            m = ForgeGNN(mi, latent_size=a["latent_size"], num_layers=a["num_layers"],
                         message_passing_steps=a["message_passing_steps"])
            m.load_state_dict(ck["state_dict"])
            m.eval()
            rows = []
            with torch.no_grad():
                for s in steps:
                    if s.get("skipped", False):
                        continue
                    x0, x1 = disp(s["step"] - 1), disp(s["step"])
                    d_mm, R, u = s["d_j_mm"], s["R_j_deg"], s["u_j_mm"]
                    ctrl = (gap_frac(float(band_half_thickness(mi, x0, d_mm, d_mm + 0.2 * H, R)) - u)
                            if control == "gap" else gnn_u_frac(u))
                    nf = build_node_features(mi, x0, d_mm / H, d_mm, d_mm + 0.2 * H, R, ctrl).unsqueeze(0)
                    xp = x0 + m.predict_delta(nf, x0.unsqueeze(0), accumulate=False)[0]
                    true_d, pred_d = x1 - x0, xp - x0
                    rows.append({"hit": s["step"], "rmse_mm": float(((xp - x1) ** 2).mean().sqrt()),
                                 "rel_err": float((pred_d - true_d).norm() / true_d.norm().clamp_min(1e-9)),
                                 "stretch_pred_mm": float(xp[:, 0].max() - x0[:, 0].max()),
                                 "stretch_true_mm": float(x1[:, 0].max() - x0[:, 0].max())})
            out[run][name] = {"hits": rows,
                              "mean_rmse_mm": float(np.mean([r["rmse_mm"] for r in rows])),
                              "mean_rel_err": float(np.mean([r["rel_err"] for r in rows])),
                              "mean_abs_stretch_err_mm": float(np.mean([abs(r["stretch_pred_mm"] - r["stretch_true_mm"])
                                                                        for r in rows])),
                              "mean_stretch_bias_mm": float(np.mean([r["stretch_pred_mm"] - r["stretch_true_mm"]
                                                                     for r in rows]))}
            o = out[run][name]
            print(f"{run} | {name} | {len(rows)} hits | one-hit RMSE {o['mean_rmse_mm']:.3f} mm | relative error "
                  f"{o['mean_rel_err']:.2f} | stretch error {o['mean_abs_stretch_err_mm']:.3f} mm "
                  f"(bias {o['mean_stretch_bias_mm']:+.3f})")
    with open(os.path.join(HERE, "one_hit_real.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
