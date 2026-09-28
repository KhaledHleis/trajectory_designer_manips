"""
trajectories/zigzag.py
----------------------
Zigzag: same hard crossing constraints as lawnmower, but V-shaped
straight-line transits between passes instead of Hermite curves.

The cable-crossing angle is guaranteed exact (same anchor+tangent
constraint). Between passes the drone flies a straight diagonal —
no smooth turn, so this gives the most compact coverage footprint
at the cost of a sharp direction change at each turn apex.
"""

import math
import numpy as np
from .base import BaseTrajectory, direction_to_heading


class ZigzagTrajectory(BaseTrajectory):

    def __init__(
        self,
        cable_p1,
        cable_p2,
        sampling_freq,
        drone_speed,
        n_crossings: int = 5,
        angle_deg: float = 45.0,
        turn_sharpness: float = 1.0,
        pass_width: float = 0.55,
        **kwargs,
    ):
        super().__init__(cable_p1, cable_p2, sampling_freq, drone_speed)
        self.n_crossings = max(1, int(n_crossings))
        self.angle_deg   = max(5.0, min(175.0, float(angle_deg)))
        self.pass_width  = max(0.05, float(pass_width))

    def generate_trajectory_local(self) -> tuple[np.ndarray, np.ndarray]:
        d = self.cable_p2 - self.cable_p1
        cable_len = np.linalg.norm(d)
        if cable_len < 1e-15:
            return np.empty((0, 2)), np.empty((0,))
        u_along = d / cable_len
        u_perp  = np.array([-u_along[1], u_along[0]])

        alpha = math.radians(self.angle_deg)
        cross_fwd = math.sin(alpha) * u_perp + math.cos(alpha) * u_along
        cross_fwd /= np.linalg.norm(cross_fwd)

        n = self.n_crossings
        offsets = [0.0] if n == 1 else list(np.linspace(0.0, cable_len, n))
        pass_half = cable_len * self.pass_width

        all_pts: list[np.ndarray] = []
        all_hdg: list[np.ndarray] = []
        prev_end = None

        for i, off in enumerate(offsets):
            anchor    = self.cable_p1 + off * u_along
            direction = cross_fwd if (i % 2 == 0) else -cross_fwd
            h = direction_to_heading(direction[0], direction[1])

            pass_start = anchor - pass_half * direction
            pass_end   = anchor + pass_half * direction

            if prev_end is not None:
                # Straight diagonal transit
                tr_pts, tr_hdg = self._sample_segment(prev_end, pass_start)
                all_pts.append(tr_pts)
                all_hdg.append(tr_hdg)

            seg_pts, _ = self._sample_segment(pass_start, pass_end)
            all_pts.append(seg_pts)
            all_hdg.append(np.full(len(seg_pts), h))

            prev_end = pass_end.copy()

        return np.vstack(all_pts), np.concatenate(all_hdg)
