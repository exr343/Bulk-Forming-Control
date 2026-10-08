"""Two figures on the old reheat: temperature set from each node's UNDEFORMED x,
so the profile moves and stretches with the metal. Old 48-hit square run,
undeformed vs the reheat applied before hit 48 (state after hit 47)."""
import json, os, sys
import meshio, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

RUN = "/home/exr343/CIRP_2027/applications/Agility_Forge/data/dataset_finetuning/square"
OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)

T_TOP, T_BOT, X_TOP, X_MIN = 1096.0, 676.0, 72.0, -5.0
def T_old(x_ref):
    return T_BOT + (T_TOP - T_BOT) * (np.clip(x_ref, X_MIN, X_TOP) - X_MIN) / (X_TOP - X_MIN)

man = json.load(open(f"{RUN}/manifest.json"))
rec = {r["hit"]: r for r in man["records"]}
def load(h):
    v = meshio.read(f"{RUN}/{rec[h]['vtu_path']}")
    return v.points, v.points + v.point_data["Displacement"]

X, x0 = load(0)
_, x47 = load(47)
T = T_old(X[:, 0])
end0 = X[:, 0].max()

def stats(x):
    hot = X[:, 0] >= X_TOP
    near800 = np.abs(T - 800.0) < 3
    return x[hot, 0].min(), x[hot, 0].max(), x[near800, 0].mean(), x[:, 0].max()
s0, s47 = stats(x0), stats(x47)
print("undeformed: hot zone %.1f-%.1f (%.1f mm), 800C at %.1f, end %.1f" % (s0[0], s0[1], s0[1]-s0[0], s0[2], s0[3]))
print("before hit 48: hot zone %.1f-%.1f (%.1f mm), 800C at %.1f, end %.1f" % (s47[0], s47[1], s47[1]-s47[0], s47[2], s47[3]))

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
XLIM = (-8, 142)

# ---------------- Figure 1: temperature vs current axial position ----------------
fig, ax = plt.subplots(figsize=(10, 5.2), dpi=200)
for x, c, lab in ((x0, BLUE, "Undeformed billet"), (x47, ORANGE, "Reheat before hit 48")):
    o = np.argsort(x[:, 0])
    ax.scatter(x[o, 0], T[o], s=4, color=c, lw=0, label=lab, rasterized=True)
ax.axvline(end0, color=MUTED, ls="--", lw=1.5)
ax.text(end0 + 1.5, 700, "original\nfree end", color=INK2, fontsize=11, va="bottom")
ax.text(0.5 * (s0[0] + s0[1]), 1108, "%.1f mm" % (s0[1] - s0[0]), color=BLUE, ha="center", fontsize=12)
ax.text(0.5 * (s47[0] + s47[1]), 1108, "%.1f mm" % (s47[1] - s47[0]), color=ORANGE, ha="center", fontsize=12)
ax.text(s0[0], 1135, "1096 °C zone:", color=INK2, ha="left", fontsize=12)
k0 = (T_TOP - T_BOT) / (s0[0] - X_MIN); k47 = (T_TOP - T_BOT) / (s47[0] - X_MIN)
ax.text(36, 950, "%.2f °C/mm" % k0, color=BLUE, ha="right", fontsize=12)
ax.text(52, 905, "%.2f °C/mm" % k47, color=ORANGE, ha="left", fontsize=12)
print("slopes", k0, k47)
ax.set_xlim(*XLIM); ax.set_ylim(660, 1160)
ax.set_xlabel("Current axial position x (mm)")
ax.set_ylabel("Temperature after reheat (°C)")
ax.set_title("Old reheat: the same profile, stretched with the metal", loc="left", color=INK, fontsize=15)
ax.grid(color=GRID, lw=0.8); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.legend(loc="upper left", frameon=False, markerscale=4, labelcolor=INK2)
fig.tight_layout(); fig.savefig(f"{OUT}/old_reheat_profile.png"); plt.close(fig)

# ---------------- Figure 2: side view coloured by reheat temperature ----------------
cmap = LinearSegmentedColormap.from_list("heat", ["#fde3d3", "#f4a37c", "#eb6834", "#b8401a", "#6e2208"])
fig, axs = plt.subplots(2, 1, figsize=(10, 4.6), dpi=200, sharex=True)
for a, x, title in ((axs[0], x0, "Undeformed billet"), (axs[1], x47, "Reheat before hit 48 (after 47 hits)")):
    o = np.argsort(x[:, 2])
    sc = a.scatter(x[o, 0], x[o, 1], c=T[o], cmap=cmap, vmin=T_BOT, vmax=T_TOP, s=3, lw=0, rasterized=True)
    a.axvline(end0, color=MUTED, ls="--", lw=1.5)
    hot = X[:, 0] >= X_TOP
    lo, hi = x[hot, 0].min(), x[hot, 0].max()
    a.plot([lo, hi], [11.5, 11.5], color=INK2, lw=2, solid_capstyle="butt")
    a.plot([lo, lo], [10.5, 12.5], color=INK2, lw=2); a.plot([hi, hi], [10.5, 12.5], color=INK2, lw=2)
    a.text(lo - 1.5, 11.5, "1096 °C zone: %.1f mm" % (hi - lo), color=INK2, fontsize=10, ha="right", va="center")
    a.set_title(title, loc="left", color=INK, fontsize=13)
    a.set_aspect("equal"); a.set_ylim(-11, 14); a.set_yticks([])
    for s in ("top", "right", "left"): a.spines[s].set_visible(False)
axs[0].text(end0 + 1.5, -9, "original free end", color=INK2, fontsize=10)
axs[1].set_xlim(*XLIM); axs[1].set_xlabel("Current axial position x (mm)")
cb = fig.colorbar(sc, ax=axs, fraction=0.025, pad=0.02); cb.set_label("Temperature after reheat (°C)", color=INK2)
cb.outline.set_visible(False)
fig.savefig(f"{OUT}/old_reheat_side_view.png", bbox_inches="tight"); plt.close(fig)
