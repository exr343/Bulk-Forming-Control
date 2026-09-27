# MPC target geometries

A target tells the MPC where every point on the billet's surface should end up.
The MPC's cost compares each surface node of the current bar with **the same
node's** target position, so a target is a displacement for every surface
node of the billet mesh, not just a final shape.

Each target folder contains:

| File | What it is |
|---|---|
| `target_on_billet.vtu` | The billet's surface mesh (rest positions) with a `Displacement` field to each node's target position. This is what the MPC reads (`--target-vtu`). In ParaView, use *Warp By Vector* to see the target shape. |
| `target_geometry.vtu` | The same surface with the points already moved to the target, so it opens directly as the target shape. |
| `target_spec.json` | Dimensions, section lengths and a volume check. |

## `ideal_square_10.6/`: the idealized square rod

Built by `make_ideal_square_target.py`:

```
python -m applications.Agility_Forge.control.targets.make_ideal_square_target
```

### The shape

The billet is 15.875 mm across (radius R = 7.9375 mm). It runs from the clamp
face at x = −5 mm to the free end at x = 96.52 mm (101.52 mm long). The target
has three sections along the bar:

| Section | From | To | Shape |
|---|---|---|---|
| Round | clamp face, x = −5 mm | the 800 °C point, x = 17.73 mm | Unchanged billet, 15.875 mm diameter |
| Taper | x = 17.73 mm | x = 25.23 mm (7.5 mm long) | The four flats close in linearly from the round bar to the square (flank angle 19.4°) |
| Square | x = 25.23 mm | x = 153.15 mm (127.9 mm long) | 10.6 × 10.6 mm, flats facing ±y and ±z (the 0° and 90° press directions) |

Total length: 158.15 mm.

Why each dimension:
- **800 °C point:** where the initial temperature profile reaches 800 °C.
  Closer to the clamp the metal is too cold to forge (the MPC's lowest station
  is also there), so that part stays round.
- **10.6 mm square:** the square run's final pass thickness (the OSU
  simple_square toolpath scaled to this billet).
- **7.5 mm taper:** the opening taper of that toolpath, scaled the same way.
- **158.15 mm length:** not chosen, but forced by volume conservation (below).
- **Nominal (cold) dimensions:** the hot bar in the simulator is about 1.2%
  larger. That difference is neglected by decision.

The cross-section at any position x′ along the target is the billet's circle
(radius R) clipped by a square of half-width h(x′):
- **Round section:** h = R, so it's the full circle.
- **Taper:** h falls linearly from R to 5.3 mm. While h is above R/√2 ≈ 5.61 mm,
  the corners are still arcs of the original circle, so the section is a
  square with rounded corners.
- **Square section:** h = 5.3 mm. Its corners are at 7.5 mm from the axis,
  inside the circle, so it's a sharp square.

### How each surface node is moved to its target

Two steps: first where along the bar the node goes, then where around the
cross-section it goes.

**1. Along the bar (volume conservation).** The metal is assumed to be
incompressible, and every cross-sectional slice of the billet is assumed to
stay a slice (plane sections stay plane).

A node at rest position x then goes to the position x′ at which the target
holds exactly as much metal between the 800 °C point (x₀ = 17.73 mm) and x′
as the billet holds between x₀ and x:

    volume of target between x₀ and x′  =  π R² · (x − x₀)

- **Nodes nearer the clamp than x₀:** unchanged (x′ = x).
- **How the target volume is computed:** the target's cross-sectional area
  (circle clipped by the square, integrated numerically) is summed along a
  fine grid of x′ values. Each node's x′ is then read off by interpolation.
- **Where the stretch comes from:** the square has only 112.4 mm² of area
  against the round bar's 197.9 mm². So each millimetre of billet becomes
  about 1.76 mm of square bar.
- **Result:** the free end, at x = 96.52 mm, maps to x′ = 153.15 mm, 56.6 mm
  further along. That's where the total length comes from.

**2. Around the cross-section (radial projection, angle kept).** Each node
keeps its angle around the bar's axis, θ = atan2(z, y). It is moved straight
in toward the axis until it sits on the target outline at its new position x′.

The outline's distance from the axis in direction θ is:

    r_outline(θ, x′) = min( R,  h(x′) / max(|cos θ|, |sin θ|) )

and the node's y and z are both multiplied by r_outline / R.

- **Lateral surface nodes** (at radius R) land exactly on the outline.
- **Nodes on the two end faces** (radius less than R) are scaled by the same
  factor, so the end faces shrink in proportion.
- **The move depends on angle:** in the square section, a node facing a flat
  (θ = 0°, 90°, …) moves in from 7.94 to 5.3 mm. A node facing a corner
  (θ = 45°) moves only from 7.94 to 7.5 mm.

The displacement stored for each node is (target position − rest position).

### Check

The enclosed volume of the two surfaces is 19,866.6 mm³ for the billet and
19,833.8 mm³ for the target, a 0.16% difference from the surface
discretization.

### Limitations

- **The node-to-node matching is an assumption.** Real forging does not keep
  plane sections exactly plane or keep each surface point at its angle: the
  surface and core flow differently, and the corners fold in. So even a
  perfectly formed bar would not reach zero node-by-node error.
  - Hausdorff and Chamfer distance compare shapes only, with no matching, so
    they are unaffected. That's why every report scores the target with them.
- **The cost's lengthwise error depends on this matching.** The node-by-node
  cost the MPC uses counts, for every node, how far it is from where step 1
  says it should be.
