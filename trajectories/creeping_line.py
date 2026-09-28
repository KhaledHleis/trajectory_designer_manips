"""
Creeping Line Search
--------------------
Classic naval search pattern: parallel legs offset progressively
to one side of the cable, then sweeping back.
Each leg crosses the cable exactly once at a fixed angle.

Parameters
----------
n_legs        : number of parallel legs
leg_spacing_m : lateral spacing between legs [m]
angle_deg     : crossing angle [deg]
pass_width    : extension beyond cable ends [fraction of cable length]
start_side    : +1 or -1, which side the first offset goes to
"""
import math
import numpy as np
from .base import BaseTrajectory, direction_to_heading


class CreepingLineTrajectory(BaseTrajectory):

    def __init__(self, cable_p1, cable_p2, sampling_freq, drone_speed,
                 n_crossings=6, leg_spacing_m=2.0, angle_deg=90.0,
                 pass_width=0.6, start_side=1, **kwargs):
        super().__init__(cable_p1, cable_p2, sampling_freq, drone_speed)
        self.n_legs        = max(2, int(n_crossings))
        self.leg_spacing_m = max(0.1, float(leg_spacing_m))
        self.angle_deg     = max(5.0, min(175.0, float(angle_deg)))
        self.pass_width    = max(0.05, float(pass_width))
        self.start_side    = 1 if start_side >= 0 else -1

    def generate_trajectory_local(self):
        d = self.cable_p2 - self.cable_p1
        cable_len = np.linalg.norm(d)
        u_along = d / cable_len
        u_perp  = np.array([-u_along[1], u_along[0]])

        alpha    = math.radians(self.angle_deg)
        cross_fwd = (math.sin(alpha) * u_perp + math.cos(alpha) * u_along)
        cross_fwd /= np.linalg.norm(cross_fwd)

        spacing_deg = self.metres_to_deg(self.leg_spacing_m)
        pass_half   = cable_len * self.pass_width

        all_pts, all_hdg = [], []
        prev_end = None
        prev_dir = None

        for i in range(self.n_legs):
            # Creeping offset: 0, +1, -1, +2, -2, +3, -3, ...
            if i == 0:
                offset_idx = 0
            elif i % 2 == 1:
                offset_idx =  self.start_side * ((i + 1) // 2)
            else:
                offset_idx = -self.start_side * (i // 2)

            lateral    = offset_idx * spacing_deg * u_perp
            anchor     = (self.cable_p1 + self.cable_p2) / 2.0 + lateral
            direction  = cross_fwd if (i % 2 == 0) else -cross_fwd
            h          = direction_to_heading(direction[0], direction[1])

            p_start = anchor - pass_half * direction
            p_end   = anchor + pass_half * direction

            if prev_end is not None:
                tr_pts, tr_hdg = self._sample_segment(prev_end, p_start)
                all_pts.append(tr_pts); all_hdg.append(tr_hdg)

            seg_pts, _ = self._sample_segment(p_start, p_end)
            all_pts.append(seg_pts)
            all_hdg.append(np.full(len(seg_pts), h))
            prev_end = p_end.copy()
            prev_dir = direction.copy()

        return np.vstack(all_pts), np.concatenate(all_hdg)
