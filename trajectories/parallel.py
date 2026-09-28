"""
trajectories/parallel.py
------------------------
Parallel-pass trajectory: straight passes parallel to the cable,
offset laterally by a user-defined spacing, connected by straight
transit legs at the ends.

This is simpler than the lawnmower — no U-turn arcs — and produces
very few waypoints, making it suitable for "drone command" mode.
"""

import math
import numpy as np
from .base import BaseTrajectory, direction_to_heading


class ParallelTrajectory(BaseTrajectory):

    def __init__(
        self,
        cable_p1,
        cable_p2,
        sampling_freq,
        drone_speed,
        n_crossings: int = 3,     # number of parallel passes
        offset_m: float = 2.0,    # lateral offset between passes [m]
        angle_deg: float = 90.0,  # kept for API compat (not used here)
        turn_sharpness: float = 0.9,
        pass_width: float = 0.55,
        **kwargs,
    ):
        super().__init__(cable_p1, cable_p2, sampling_freq, drone_speed)
        self.n_passes = max(1, int(n_crossings))
        self.offset_m = max(0.1, float(offset_m))
        self.pass_width = max(0.01, float(pass_width))

    def generate_trajectory_local(self) -> tuple[np.ndarray, np.ndarray]:
        d = self.cable_p2 - self.cable_p1
        cable_len = np.linalg.norm(d)
        if cable_len < 1e-15:
            return np.empty((0, 2)), np.empty((0,))
        u_along = d / cable_len
        u_perp = np.array([-u_along[1], u_along[0]])

        offset_deg = self.metres_to_deg(self.offset_m)
        extension = cable_len * self.pass_width

        # Centre the passes symmetrically around the cable
        half = (self.n_passes - 1) / 2.0
        lateral_offsets = [(i - half) * offset_deg for i in range(self.n_passes)]

        all_pts: list[np.ndarray] = []
        all_hdg: list[np.ndarray] = []

        prev_end = None

        for i, lat_off in enumerate(lateral_offsets):
            centre = (self.cable_p1 + self.cable_p2) / 2.0 + lat_off * u_perp
            direction = u_along if (i % 2 == 0) else -u_along
            p_start = centre - (cable_len / 2.0 + extension) * u_along * (1 if i % 2 == 0 else -1)
            p_end   = centre + (cable_len / 2.0 + extension) * u_along * (1 if i % 2 == 0 else -1)

            if prev_end is not None:
                # Straight transit to start of next pass
                tr_pts, tr_hdg = self._sample_segment(prev_end, p_start)
                all_pts.append(tr_pts)
                all_hdg.append(tr_hdg)

            seg_pts, seg_hdg = self._sample_segment(p_start, p_end)
            all_pts.append(seg_pts)
            all_hdg.append(seg_hdg)
            prev_end = seg_pts[-1].copy()

        return np.vstack(all_pts), np.concatenate(all_hdg)
