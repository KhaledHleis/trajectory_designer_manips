"""
Spiral trajectory
-----------------
The drone spirals inward (or outward) toward the cable midpoint,
making multiple crossings at varying angles and distances.

Parameters
----------
n_loops       : number of spiral loops
start_radius_m: starting radius from cable midpoint [m]
end_radius_m  : ending radius (smaller = closer to cable) [m]
angle_offset  : angular offset of spiral start [deg]
tilt_deg      : tilt of spiral plane relative to cable normal [deg]
               (0 = orbit perpendicular to cable, >0 = tilted ellipse)
"""
import math
import numpy as np
from .base import BaseTrajectory, direction_to_heading


class SpiralTrajectory(BaseTrajectory):

    def __init__(self, cable_p1, cable_p2, sampling_freq, drone_speed,
                 n_loops=3, start_radius_m=8.0, end_radius_m=1.0,
                 angle_offset=0.0, tilt_deg=30.0, **kwargs):
        super().__init__(cable_p1, cable_p2, sampling_freq, drone_speed)
        self.n_loops        = max(1, float(n_loops))
        self.start_radius_m = max(0.5, float(start_radius_m))
        self.end_radius_m   = max(0.1, float(end_radius_m))
        self.angle_offset   = float(angle_offset)
        self.tilt_deg       = max(0.0, min(80.0, float(tilt_deg)))

    def generate_trajectory_local(self):
        d = self.cable_p2 - self.cable_p1
        cable_len = np.linalg.norm(d)
        u_along = d / cable_len
        u_perp  = np.array([-u_along[1], u_along[0]])
        centre  = (self.cable_p1 + self.cable_p2) / 2.0

        r_start = self.metres_to_deg(self.start_radius_m)
        r_end   = self.metres_to_deg(self.end_radius_m)

        total_angle = self.n_loops * 2 * math.pi
        # avg circumference
        avg_r_m = (self.start_radius_m + self.end_radius_m) / 2.0
        arc_m   = avg_r_m * total_angle
        n_pts   = max(12, int(arc_m / self.drone_speed * self.sampling_freq))

        angles  = np.linspace(
            math.radians(self.angle_offset),
            math.radians(self.angle_offset) + total_angle,
            n_pts
        )
        radii = np.linspace(r_start, r_end, n_pts)

        # tilt: scale the perp component by cos(tilt), along by sin(tilt)*fraction
        tilt = math.radians(self.tilt_deg)
        perp_scale  = math.cos(tilt)
        along_scale = math.sin(tilt)

        pts = np.array([
            centre
            + radii[i] * math.cos(angles[i]) * u_perp * perp_scale
            + radii[i] * math.sin(angles[i]) * u_along * along_scale
            + radii[i] * math.sin(angles[i]) * u_perp * (1 - perp_scale)
            for i in range(n_pts)
        ])
        # simpler: elliptical orbit in (u_perp, u_along) plane
        pts = np.array([
            centre
            + radii[i] * math.cos(angles[i]) * u_perp
            + radii[i] * math.sin(angles[i]) * along_scale * u_along
            for i in range(n_pts)
        ])

        dp = np.gradient(pts, axis=0)
        headings = np.array([direction_to_heading(dp[i,0], dp[i,1]) for i in range(n_pts)])
        return pts, headings
