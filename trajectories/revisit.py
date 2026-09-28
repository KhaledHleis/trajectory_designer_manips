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
"""

import numpy as np
from .base import SparseWaypointTrajectory


class RevisitTrajectory(SparseWaypointTrajectory):
    """
    Parameters
    ----------
    offset_m    : lateral distance from the cable to each parallel leg [m]
    overshoot_m : how far the legs extend beyond each cable end [m]
    start_side  : +1 = start on the left of P1→P2, -1 = start on the right
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
        # (north, east) frame: left of the P1→P2 direction
        u_perp = np.array([u_along[1], -u_along[0]]) * self.start_side

        e, o = self.overshoot_m, self.offset_m
        back = self.cable_p1 - e * u_along
        front = self.cable_p2 + e * u_along

        pts = np.array([
            back + o * u_perp,    # A  start
            front + o * u_perp,   # B
            front - o * u_perp,   # C
            back - o * u_perp,    # D  end
        ])
        return pts
