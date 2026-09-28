"""
Random Walk trajectory
----------------------
Constrained random walk that is guaranteed to cross the cable multiple times.
The drone wanders randomly but is attracted back to the cable so it keeps
crossing it — like a search pattern by a disoriented observer.

This is the most realistic "unplanned" trajectory for training a detector
to be robust against arbitrary approach angles and speeds.

Parameters
----------
n_crossings   : minimum number of cable crossings to guarantee
step_m        : length of each random step [m]
lateral_bound_m: max lateral distance from cable before correction kicks in [m]
seed          : RNG seed (None = random)
attraction    : cable-attraction strength [0=pure random, 1=always toward cable]
"""
import math
import numpy as np
from .base import BaseTrajectory, direction_to_heading


class RandomWalkTrajectory(BaseTrajectory):

    def __init__(self, cable_p1, cable_p2, sampling_freq, drone_speed,
                 n_crossings=10, step_m=1.0, lateral_bound_m=5.0,
                 seed=None, attraction=0.3, **kwargs):
        super().__init__(cable_p1, cable_p2, sampling_freq, drone_speed)
        self.n_crossings    = max(3, int(n_crossings))
        self.step_m         = max(0.1, float(step_m))
        self.lateral_bound_m= max(0.5, float(lateral_bound_m))
        self.attraction     = max(0.0, min(1.0, float(attraction)))
        self.rng            = np.random.default_rng(seed)

    def generate_trajectory_local(self):
        d = self.cable_p2 - self.cable_p1
        cable_len = np.linalg.norm(d)
        u_along = d / cable_len
        u_perp  = np.array([-u_along[1], u_along[0]])

        step_deg   = self.metres_to_deg(self.step_m)
        bound_deg  = self.metres_to_deg(self.lateral_bound_m)

        # Start at a random point near one end of the cable
        pos = self.cable_p1 + self.rng.uniform(-0.3, 0.3) * cable_len * u_along \
            + self.rng.uniform(-1, 1) * bound_deg * u_perp

        pts    = [pos.copy()]
        headings = []
        crossings_made = 0
        prev_lateral = np.dot(pos - self.cable_p1, u_perp)
        max_steps = self.n_crossings * 200  # safety cap

        heading_bias = self.rng.uniform(0, 2 * math.pi)  # current walk direction

        for _ in range(max_steps):
            if crossings_made >= self.n_crossings:
                break

            # Current lateral distance from cable line
            along_t = np.dot(pos - self.cable_p1, u_along)
            lateral = np.dot(pos - self.cable_p1, u_perp)

            # Attraction force toward cable (in perp direction)
            if abs(lateral) > bound_deg:
                # Hard correction: point mostly toward cable
                attract_angle = math.atan2(-lateral * u_perp[1], -lateral * u_perp[0])
                heading_bias  = attract_angle + self.rng.normal(0, 0.3)
            else:
                # Soft drift: random walk with mild cable attraction
                attract_angle = math.atan2(-lateral * u_perp[1], -lateral * u_perp[0])
                random_turn   = self.rng.normal(0, 0.6)  # radians
                heading_bias += random_turn
                heading_bias  = (1 - self.attraction) * heading_bias \
                               + self.attraction * attract_angle

            move = step_deg * np.array([math.cos(heading_bias), math.sin(heading_bias)])

            # Also keep drone from going too far along-cable
            if along_t < -cable_len * 0.2:
                move += step_deg * 0.5 * u_along
            elif along_t > cable_len * 1.2:
                move -= step_deg * 0.5 * u_along

            pos = pos + move
            pts.append(pos.copy())

            # Count crossings (sign change of lateral component)
            new_lateral = np.dot(pos - self.cable_p1, u_perp)
            along_frac  = np.dot(pos - self.cable_p1, u_along) / cable_len
            if (prev_lateral * new_lateral < 0) and (0.0 <= along_frac <= 1.0):
                crossings_made += 1
            prev_lateral = new_lateral

        pts = np.array(pts)

        # Resample at correct spacing
        # Compute cumulative arc length
        diffs = np.diff(pts, axis=0)
        seg_lens = np.linalg.norm(diffs, axis=1)
        cum_len  = np.concatenate([[0], np.cumsum(seg_lens)])
        total_m  = self.deg_to_metres(cum_len[-1])
        n_out    = max(4, int(total_m / self.drone_speed * self.sampling_freq))
        t_uniform = np.linspace(0, cum_len[-1], n_out)
        resampled = np.column_stack([
            np.interp(t_uniform, cum_len, pts[:, 0]),
            np.interp(t_uniform, cum_len, pts[:, 1]),
        ])

        dp = np.gradient(resampled, axis=0)
        hdgs = np.array([direction_to_heading(dp[i,0], dp[i,1]) for i in range(n_out)])
        return resampled, hdgs
