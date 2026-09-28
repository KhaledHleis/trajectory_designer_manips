"""
Random Crossings
----------------
Generates a set of random straight-line passes, each guaranteed to
cross the cable at a randomly chosen angle within a configurable range.
Between passes the drone takes a direct transit.

This is the most useful trajectory for ML training: each instance
gives the detector a different (angle, speed, lateral_position) triple.

Parameters
----------
n_crossings    : number of crossing passes
angle_min_deg  : minimum crossing angle [deg]
angle_max_deg  : maximum crossing angle [deg]
pass_width     : pass extension beyond cable [fraction of cable length]
lateral_jitter : fraction of cable length by which the crossing point
                 is randomly displaced along the cable
seed           : RNG seed
"""
import math
import numpy as np
from .base import BaseTrajectory, direction_to_heading


class RandomCrossingsTrajectory(BaseTrajectory):

    def __init__(self, cable_p1, cable_p2, sampling_freq, drone_speed,
                 n_crossings=8, angle_min_deg=20.0, angle_max_deg=160.0,
                 pass_width=0.6, lateral_jitter=0.8, seed=None, **kwargs):
        super().__init__(cable_p1, cable_p2, sampling_freq, drone_speed)
        self.n_crossings   = max(1, int(n_crossings))
        self.angle_min_deg = max(5.0, float(angle_min_deg))
        self.angle_max_deg = min(175.0, float(angle_max_deg))
        self.pass_width    = max(0.05, float(pass_width))
        self.lateral_jitter= max(0.0, min(1.0, float(lateral_jitter)))
        self.rng           = np.random.default_rng(seed)

    def generate_trajectory_local(self):
        d = self.cable_p2 - self.cable_p1
        cable_len = np.linalg.norm(d)
        u_along = d / cable_len
        u_perp  = np.array([-u_along[1], u_along[0]])

        pass_half = cable_len * self.pass_width
        all_pts, all_hdg = [], []
        prev_end = None

        for _ in range(self.n_crossings):
            # Random crossing angle
            angle_deg = self.rng.uniform(self.angle_min_deg, self.angle_max_deg)
            alpha = math.radians(angle_deg)
            cross_dir = math.sin(alpha) * u_perp + math.cos(alpha) * u_along
            cross_dir /= np.linalg.norm(cross_dir)

            # Random crossing point along cable
            jitter = self.rng.uniform(
                0.5 - self.lateral_jitter / 2,
                0.5 + self.lateral_jitter / 2
            )
            anchor = self.cable_p1 + jitter * cable_len * u_along

            # Randomly flip crossing direction
            if self.rng.random() < 0.5:
                cross_dir = -cross_dir

            p_start = anchor - pass_half * cross_dir
            p_end   = anchor + pass_half * cross_dir
            h       = direction_to_heading(cross_dir[0], cross_dir[1])

            if prev_end is not None:
                tr_pts, tr_hdg = self._sample_segment(prev_end, p_start)
                all_pts.append(tr_pts); all_hdg.append(tr_hdg)

            seg_pts, _ = self._sample_segment(p_start, p_end)
            all_pts.append(seg_pts)
            all_hdg.append(np.full(len(seg_pts), h))
            prev_end = p_end.copy()

        if not all_pts:
            return np.empty((0,2)), np.empty((0,))
        return np.vstack(all_pts), np.concatenate(all_hdg)
