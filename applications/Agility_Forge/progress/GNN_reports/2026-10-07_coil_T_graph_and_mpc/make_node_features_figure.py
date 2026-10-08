"""What the coil_T GNN's nodes see for one real hit: hit 42 of the plain square
5-pass run (90 deg, die centre 63.8 mm, stroke 1.5 mm; the second hit at its
station, so it starts on metal the first hit's dies just chilled).

Three views of the same bar (the shape at the start of the hit):
  1. die push magnitude: the displacement the die would impose on each node
     (the node input, exactly as GNN/coil_T.node_features computes it);
  2. temperature input: the real surface temperature at the start of the hit;
  3. temperature change over the hit: the simulator's result, i.e. the 4th
     output the GNN learns to predict.

Run from the repo root:
    python applications/Agility_Forge/progress/GNN_reports/2026-10-07_coil_T_graph_and_mpc/make_node_features_figure.py
"""
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm, LinearSegmentedColormap
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

sys.path.insert(0, "/home/exr343/CIRP_2027")
os.chdir("/home/exr343/CIRP_2027")
from applications.Agility_Forge.GNN.coil_T import (Mesh, VtuCache, PRE_DIR, NEW_SQ, new_cycles, run_dir,
                                                   node_features)
sys.path.insert(0, "applications/Agility_Forge/output/die12_coil_animation")
from progress_plot import TCMAP, LIGHT, ELEV, AZIM, INK, INK2, MUTED

OUT = os.path.dirname(os.path.abspath(__file__))
HIT = 42
PUSH_CMAP = LinearSegmentedColormap.from_list("push", ["#eef3fb", "#8fb3e6", "#2a78d6", "#0d2a5c"])
DT_CMAP = LinearSegmentedColormap.from_list("dT", ["#0d2a5c", "#2a78d6", "#e8e6e1", "#eb6834", "#7d1716"])

mesh = Mesh(f"{PRE_DIR}/rollout_01/undeformed.vtu", "cpu")
cache = VtuCache(mesh)
step = next(s for c in new_cycles(run_dir(NEW_SQ, None), cache) for s in c["steps"] if s.label == f"hit {HIT}")
x0, T0 = cache.get(step.x_in)
x1, T1 = cache.get(step.x_out)
f = node_features(mesh, x0, T0, step.hit)                       # (N, 16): exactly the GNN's raw node input
push = f[:, 4:7].norm(dim=-1).numpy()                   # columns 4-6 = die push
assert torch.allclose(f[:, 11], T0)                             # column 11 = temperature input
dT = (T1 - T0).numpy()
tris = mesh.info.surf_faces_local
rest = mesh.rest.numpy().astype(np.float64)
u = x0.numpy().astype(np.float64)
print(f"hit {HIT}: centre {float(step.hit.center):.1f} mm, angle {float(step.hit.angle_deg):.0f} deg, "
      f"stroke {float(step.hit.stroke_mm):.2f} mm | nodes pushed (>0.01 mm): {(push > 0.01).sum()}, max push {push.max():.2f} mm "
      f"| T input {T0.min():.0f}-{T0.max():.0f} C | dT {dT.min():.0f} to {dT.max():.0f} C")

fig = plt.figure(figsize=(11, 9.6))
fig.patch.set_facecolor("#fcfcfb")
fig.text(0.03, 0.975, f"What Each Node Sees: Hit {HIT}, Plain Square Run", fontsize=15, fontweight="bold", color=INK, va="top")
fig.text(0.03, 0.945, f"90° hit at {float(step.hit.center):.1f} mm, stroke {float(step.hit.stroke_mm):.1f} mm; "
         "second hit at this station. Shape at the start of the hit; colour = per-node value.",
         fontsize=10, color=INK2, va="top")
panels = [(push, Normalize(0, max(push.max(), 1e-3)), PUSH_CMAP, "1. Die Push (Node Input): Displacement the Die Would Impose", "mm"),
          (T0.numpy(), TwoSlopeNorm(vmin=650, vcenter=800, vmax=1100), TCMAP, "2. Temperature (Node Input): Real Surface Temperature at the Start", "°C"),
          (dT, TwoSlopeNorm(vmin=min(dT.min(), -1), vcenter=0, vmax=max(dT.max(), 1)), DT_CMAP,
           "3. Temperature Change over the Hit (Learned Output, from the Simulator)", "°C")]
P = (rest + u)[:, [0, 2, 1]]
tri = P[tris]
n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
shade = (0.80 + 0.20 * np.abs(n @ LIGHT))[:, None]
xmax = (rest[:, 0] + u[:, 0]).max() + 6
for k, (val, norm, cmap, title, unit) in enumerate(panels):
    y0 = 0.64 - k * 0.3
    ax = fig.add_axes([0.0, y0, 0.86, 0.26], projection="3d")
    cols = cmap(norm(val[tris].mean(axis=1))); cols[:, :3] *= shade
    ax.add_collection3d(Poly3DCollection(tri, facecolors=cols, edgecolors=(0.15, 0.15, 0.18, 0.25),
                                         linewidths=0.15)).set_clip_on(False)
    ax.set_xlim(-8, xmax); ax.set_ylim(-12, 12); ax.set_zlim(-12, 12)
    ax.set_box_aspect((xmax + 8, 24, 24), zoom=2.9)
    ax.view_init(elev=ELEV, azim=AZIM); ax.set_axis_off()
    fig.text(0.05, y0 + 0.255, title, fontsize=11, color=INK, va="top")
    cax = fig.add_axes([0.89, y0 + 0.05, 0.012, 0.16])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_label(unit, color=INK2); cb.outline.set_edgecolor(MUTED)
    if k == 2:
        cb.set_ticks([t for t in range(-200, 101, 50) if norm.vmin <= t <= norm.vmax])
fig.savefig(f"{OUT}/figures/node_features_hit{HIT}.png", dpi=110, facecolor="#fcfcfb")
print("wrote", f"{OUT}/figures/node_features_hit{HIT}.png")
