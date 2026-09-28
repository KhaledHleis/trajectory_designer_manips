"""
trajectories/revisit.py
-----------------------
"Revisit" / C-pattern: fly parallel to the cable on one side at a fixed
distance, wrap around the far end (without crossing the cable), and come
back along the other side.

      A ─────────────────────── B
                                │   <- overshoot beyond P2
   P1 ═══════ cable ═══════ P2  │   <- `offset_m` either side
                                │
      D ─────────────────────── C

Waypoints: A (start) → B → C → D (end). Headings = course of each leg.

`start_side` picks which cable end the C wraps around (i.e. flips the
shape left/right, "]" <-> "["):
    +1 : wrap around P2  ->  "]"   (drawn above)
    -1 : wrap around P1  ->  "["
The start is always on the left of P1→P2; use the global "swap start/end"
(`reverse`) to fly the same shape the other way.
"""

import numpy as np
from .base import SparseWaypointTrajectory


class RevisitTrajectory(SparseWaypointTrajectory):
    """
    Parameters
    ----------
    offset_m    : lateral distance from the cable to each parallel leg [m]
    overshoot_m : how far the legs extend beyond each cable end [m]
    start_side  : +1 = wrap around the P2 end ("]"), -1 = around P1 ("[")
    """

    def __init__(
        self,
        cable_p1,
        cable_p2,
        sampling_freq,
        drone_speed,
        offset_m: float = 5.0,
        overshoot_m: float = 5.0,
        start_side: float = 1.0,
        **kwargs,
    ):
        super().__init__(cable_p1, cable_p2, sampling_freq, drone_speed,
                         reverse=kwargs.get("reverse", False))
        self.offset_m = max(0.01, float(offset_m))
        self.overshoot_m = max(0.0, float(overshoot_m))
        self.start_side = 1.0 if float(start_side) >= 0 else -1.0

    def _waypoints_local(self) -> np.ndarray:
        d = self.cable_p2 - self.cable_p1
        cable_len = float(np.linalg.norm(d))
        if cable_len < 1e-9:
            return np.empty((0, 2))
        u_along = d / cable_len
        u_left = np.array([u_along[1], -u_along[0]])  # left of P1→P2 in (N, E)

        e, o = self.overshoot_m, self.offset_m
        back = self.cable_p1 - e * u_along
        front = self.cable_p2 + e * u_along
        open_end, wrap_end = (back, front) if self.start_side > 0 else (front, back)

        pts = np.array([
            open_end + o * u_left,   # A  start
            wrap_end + o * u_left,   # B
            wrap_end - o * u_left,   # C
            open_end - o * u_left,   # D  end
        ])
        return pts
