"""Hit configuration dataclass for multi-hit forging simulations."""

from dataclasses import dataclass
from typing import Optional

# Flat die width along the bar (mm): the JAX-FORGE paper's tool (Sec. 3.3,
# "12.7 mm width flat tools", forged on the same 15.9 mm stock as our billet).
DIE_WIDTH_MM = 12.7


@dataclass(kw_only=True)
class Hit:
    """
    Parameters for a single compression hit.

    The die's position along the bar is given in exactly one of two ways:

    * ``die_center_mm`` (spatial die): a flat die ``die_width_mm`` wide, centred
      at this x in space (mm, the mesh's x coordinate). It presses whichever
      outer-surface points are CURRENTLY under it, so its footprint stays
      ``die_width_mm`` however far the bar has stretched.
    * ``x_min_band`` / ``x_max_band`` (original band): fractions of H marking a
      band on the UNDEFORMED bar. The die presses the points that started in
      that band, so the pressed stretch grows as the bar lengthens. All data
      generated before the spatial die was added uses this.

    Parameters
    ----------
    compression_displacement : float
        Total platen displacement in this hit (mm).
    die_center_mm : float or None
        Spatial die centre along x (mm).
    die_width_mm : float
        Spatial die width along x (mm).
    x_min_band : float or None
        Lower bound of contact band along tool axis (relative to H).
    x_max_band : float or None
        Upper bound of contact band along tool axis (relative to H).
    rotation_euler_x : float
        Rotation angle around X-axis in degrees.
    total_time : float
        Duration (seconds) allocated for this hit's time-stepping.
    """
    compression_displacement: float
    die_center_mm: Optional[float] = None
    die_width_mm: float = DIE_WIDTH_MM
    x_min_band: Optional[float] = None
    x_max_band: Optional[float] = None
    rotation_euler_x: float = 0.0
    total_time: float = 1.0

    def __post_init__(self):
        band = self.x_min_band is not None or self.x_max_band is not None
        if (self.die_center_mm is None) == (not band):
            raise ValueError("Hit needs exactly one of die_center_mm or x_min_band/x_max_band")
        if band and (self.x_min_band is None or self.x_max_band is None):
            raise ValueError("Hit needs both x_min_band and x_max_band")

    @property
    def spatial_die(self) -> bool:
        return self.die_center_mm is not None
