"""
trajectories/lawnmower.py
-------------------------
Rectangular lawnmower as sparse mission waypoints.

Every pass crosses the cable at `angle_deg`. All pass ends sit on two
"rails" perpendicular to the pass direction, so the transit legs between
passes are perpendicular to the passes and every corner is exactly 90 deg:

      S1 ────────┐   E2 ────────┐   S5 ...
      │          │   │          │
   ═══X══════════X═══X══════════X═══  cable
      │          │   │          │
      E1         S2 ─┘          E3 ...

(drawn for angle_deg = 90; for other angles the passes tilt but the
 pattern stays a chain of rectangles).

Waypoints per pass: start corner, cable crossing, end corner.
"""

import math
import numpy as np
from .base import SparseWaypointTrajectory


class LawnmowerTrajectory(SparseWaypointTrajectory):
    """
    Parameters
    ----------
    n_crossings : number of passes (= cable crossings), spread P1 → P2
    angle_deg   : crossing angle w.r.t. the cable direction (deg)
    pass_width  : how far each pass extends beyond the cable, as a fraction
                  of cable length (measured from the farthest crossing, so
                  every pass clears the cable by at least this much)
    """

    def __init__(
        self,
        cable_p1,
        cable_p2,
        sampling_freq,
        drone_speed,
        n_crossings: int = 5,
        angle_deg: float = 90.0,
        pass_width: float = 0.55,
        **kwargs,
    ):
        super().__init__(cable_p1, cable_p2, sampling_freq, drone_speed,
                         reverse=kwargs.get("reverse", False))
        self.n_crossings = max(1, int(round(n_crossings)))
        self.angle_deg = max(5.0, min(175.0, float(angle_deg)))
        self.pass_width = max(0.01, float(pass_width))

    # ── spacing (metres) ──────────────────────────────────────────────────
    @property
    def cable_length_m(self) -> float:
        return float(np.linalg.norm(self.cable_p2 - self.cable_p1))

    @property
    def crossing_spacing_m(self) -> float:
        """Distance between consecutive crossings, measured ALONG the cable."""
        n = self.n_crossings
        return 0.0 if n < 2 else self.cable_length_m / (n - 1)

    @property
    def pass_spacing_m(self) -> float:
        """Perpendicular distance between adjacent parallel passes
        (= length of the short transit legs)."""
        return self.crossing_spacing_m * abs(math.sin(math.radians(self.angle_deg)))

    def _waypoints_local(self) -> np.ndarray:
        d = self.cable_p2 - self.cable_p1
        cable_len = float(np.linalg.norm(d))
        if cable_len < 1e-9:
            return np.empty((0, 2))
        u_along = d / cable_len
        u_perp = np.array([-u_along[1], u_along[0]])
        centre = (self.cable_p1 + self.cable_p2) / 2.0

        alpha = math.radians(self.angle_deg)
        v = math.sin(alpha) * u_perp + math.cos(alpha) * u_along
        v /= np.linalg.norm(v)

        n = self.n_crossings
        offsets = [cable_len / 2.0] if n == 1 else np.linspace(0.0, cable_len, n)

        # Rail half-distance along v, measured from the cable centre.
        rail = cable_len * self.pass_width + 0.5 * cable_len * abs(math.cos(alpha))

        wps = []
        for i, off in enumerate(offsets):
            anchor = self.cable_p1 + off * u_along
            t = float(np.dot(anchor - centre, v))       # anchor's position on v
            sgn = 1.0 if i % 2 == 0 else -1.0           # pass direction = sgn * v
            start = anchor + (-sgn * rail - t) * v
            end = anchor + (sgn * rail - t) * v
            wps.extend([start, anchor, end])

        pts = np.array(wps)
        return pts
