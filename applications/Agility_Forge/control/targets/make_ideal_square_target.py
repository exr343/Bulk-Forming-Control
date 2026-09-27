"""Idealized square-rod target on the billet's surface mesh (2026-09-26).

Geometry (from the clamp end; volume conserved exactly, no thermal expansion):
  1. round, the billet's own diameter (15.875 mm), from the clamp face to the
     800 C point of the initial temperature profile (x = 17.73 mm): unchanged;
  2. taper: the four flats close linearly from the round bar to the square over
     TAPER_MM of length (~20 deg flank angle, the opening taper of OSU's
     simple_square toolpath scaled by 0.417);
  3. square, SIDE_MM x SIDE_MM, to the free end; its length follows from
     volume conservation.
Flats face +/-y and +/-z, i.e. the 0 / 90 deg hit directions.

Node mapping (needed because the MPC cost compares each surface node with the
same node's target position):
  - lengthwise: a node at rest position x maps to x' such that the target's
    volume between the 800 C point and x' equals the round bar's volume
    between the 800 C point and x (nodes before the 800 C point stay put);
  - across: each node is moved radially, keeping its angle, onto the target
    outline at x' (circle clipped by the square half-width h(x')); nodes
    inside the outline (end faces) are scaled by the same factor.

Writes into control/targets/ideal_square_<SIDE_MM>/:
  target_on_billet.vtu -- billet surface points + "Displacement" to the target
    (same format as eval_square_target.py's target.vtu; use Warp By Vector)
  target_geometry.vtu  -- the same surface with points already at the target
    positions (opens directly as the target shape)
  target_spec.json     -- dimensions, lengths, volume check

Usage (repo root):
    python -m applications.Agility_Forge.control.targets.make_ideal_square_target
"""

import json
import os

import meshio
import numpy as np

from applications.Agility_Forge.GNN.data import build_surface_mesh_info
from applications.Agility_Forge.generate_square_rollout import X_800C_MM

SIDE_MM = 10.6
TAPER_MM = 7.5
BILLET_VTU = "applications/Agility_Forge/data/dataset_finetuning/square/rollout_01/undeformed.vtu"
OUT_DIR = f"applications/Agility_Forge/control/targets/ideal_square_{SIDE_MM:g}"


def clipped_area(R, h, n=4001):
    """Area of the circle of radius R clipped to the square |y|, |z| <= h."""
    if h >= R:
        return np.pi * R ** 2
    y = np.linspace(-min(h, R), min(h, R), n)
    half_chord = np.minimum(np.sqrt(np.maximum(R ** 2 - y ** 2, 0.0)), h)
    return np.trapz(2.0 * half_chord, y)


def outline_radius(theta, R, h):
    """Distance from the axis to the target outline in direction theta."""
    to_square = h / np.maximum(np.abs(np.cos(theta)), np.abs(np.sin(theta)))
    return np.minimum(R, to_square)


def main():
    mesh_info = build_surface_mesh_info(BILLET_VTU)
    rest = mesh_info.rest_pos.numpy().astype(np.float64)
    x, y, z = rest[:, 0], rest[:, 1], rest[:, 2]
    R = float(np.hypot(y, z).max())
    A0 = np.pi * R ** 2
    h_sq = SIDE_MM / 2.0
    x0 = X_800C_MM

    # Target half-width along the target's own axis (x' measured like x).
    def h_of(xp):
        t = np.clip((xp - x0) / TAPER_MM, 0.0, 1.0)
        return np.where(xp <= x0, R, R + t * (h_sq - R))

    # Cumulative target volume beyond the 800 C point, on a fine x' grid long
    # enough to hold all the metal.
    grid = np.linspace(x0, x0 + 3.0 * (x.max() - x0), 20001)
    area = np.array([clipped_area(R, h) for h in h_of(grid)])
    vol = np.concatenate([[0.0], np.cumsum(0.5 * (area[1:] + area[:-1]) * np.diff(grid))])

    # Lengthwise map: equal volume from the 800 C point.
    xp = np.where(x <= x0, x, np.interp(A0 * (x - x0), vol, grid))
    # Cross-section map: radial projection onto the outline at x'.
    theta = np.arctan2(z, y)
    r = np.hypot(y, z)
    scale = outline_radius(theta, R, h_of(xp)) / R
    target = np.stack([xp, y * scale, z * scale], axis=1)
    disp = target - rest

    os.makedirs(OUT_DIR, exist_ok=True)
    faces = [("triangle", mesh_info.surf_faces_local)]
    meshio.Mesh(points=rest.astype(np.float32), cells=faces,
                point_data={"Displacement": disp.astype(np.float32)}).write(
        os.path.join(OUT_DIR, "target_on_billet.vtu"))
    meshio.Mesh(points=target.astype(np.float32), cells=faces).write(
        os.path.join(OUT_DIR, "target_geometry.vtu"))

    # Checks: enclosed volume of both surfaces (divergence theorem).
    def enclosed_volume(P):
        a, b, c = (P[mesh_info.surf_faces_local[:, i]] for i in range(3))
        return abs(np.einsum("ij,ij->i", a, np.cross(b, c)).sum()) / 6.0

    L_total = float(target[:, 0].max() - target[:, 0].min())
    spec = {
        "square_side_mm": SIDE_MM, "taper_length_mm": TAPER_MM,
        "taper_flank_angle_deg": float(np.degrees(np.arctan((R - h_sq) / TAPER_MM))),
        "billet_diameter_mm": 2 * R, "billet_length_mm": float(x.max() - x.min()),
        "round_section_mm": [float(x.min()), x0],
        "taper_mm": [x0, x0 + TAPER_MM],
        "square_section_mm": [x0 + TAPER_MM, float(target[:, 0].max())],
        "square_section_length_mm": float(target[:, 0].max() - (x0 + TAPER_MM)),
        "total_length_mm": L_total,
        "free_end_displacement_mm": float(disp[:, 0].max()),
        "billet_volume_mm3": enclosed_volume(rest), "target_volume_mm3": enclosed_volume(target),
    }
    with open(os.path.join(OUT_DIR, "target_spec.json"), "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2)
    print(json.dumps(spec, indent=2))


if __name__ == "__main__":
    main()
