"""
Expanding Square (Williamson Turn)
-----------------------------------
Classic search pattern: squares expanding outward from a centre point.
Each loop crosses the cable at a different point and angle, giving good
coverage at varied approach angles.

Parameters
----------
n_loops       : number of expanding square loops
step_m        : initial side length [m]
growth_m      : how much the square grows per side [m]
centre_offset_m: offset of pattern centre from cable midpoint [m]
angle_deg     : rotation of the square w.r.t. cable axis [deg]
"""
import math
import numpy as np
from .base import BaseTrajectory, direction_to_heading


class ExpandingSquareTrajectory(BaseTrajectory):

    def __init__(self, cable_p1, cable_p2, sampling_freq, drone_speed,
                 n_crossings=4, step_m=2.0, growth_m=2.0,
                 centre_offset_m=0.0, angle_deg=0.0, **kwargs):
        super().__init__(cable_p1, cable_p2, sampling_freq, drone_speed)
        self.n_loops         = max(1, int(n_crossings))
        self.step_m          = max(0.5, float(step_m))
        self.growth_m        = max(0.0, float(growth_m))
        self.centre_offset_m = float(centre_offset_m)
        self.angle_deg       = float(angle_deg)

    def generate_trajectory_local(self):
        d = self.cable_p2 - self.cable_p1
        cable_len = np.linalg.norm(d)
        u_along = d / cable_len
        u_perp  = np.array([-u_along[1], u_along[0]])
        mid     = (self.cable_p1 + self.cable_p2) / 2.0
        centre  = mid + self.metres_to_deg(self.centre_offset_m) * u_perp

        # Rotate the two axes by angle_deg
        rot = math.radians(self.angle_deg)
        ax1 =  math.cos(rot) * u_along + math.sin(rot) * u_perp
        ax2 = -math.sin(rot) * u_along + math.cos(rot) * u_perp

        all_pts, all_hdg = [], []
        pos = centre.copy()

        step_deg = self.metres_to_deg(self.step_m)
        grow_deg = self.metres_to_deg(self.growth_m)
        side_len = step_deg

        # Expanding square: 1 right, 1 up, 2 left, 2 down, 3 right, 3 up, ...
        dirs = [ax1, ax2, -ax1, -ax2]  # E N W S (in rotated frame)
        rep_counts = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8]

        seg_idx = 0
        for rep in rep_counts[:self.n_loops * 4]:
            direction = dirs[seg_idx % 4]
            for _ in range(rep):
                p_end = pos + side_len * direction
                seg_pts, seg_hdg = self._sample_segment(pos, p_end)
                all_pts.append(seg_pts)
                all_hdg.append(seg_hdg)
                pos = p_end.copy()
            side_len += grow_deg
            seg_idx  += 1
            if seg_idx >= self.n_loops * 4:
                break

        if not all_pts:
            return np.empty((0,2)), np.empty((0,))
        return np.vstack(all_pts), np.concatenate(all_hdg)
