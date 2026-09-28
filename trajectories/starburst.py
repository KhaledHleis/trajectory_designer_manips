"""
Starburst trajectory
--------------------
Radial passes from a central point (on or near the cable), each
crossing the cable at a different angle. Good for testing
detection at many angle combinations in one flight.

Parameters
----------
n_rays        : number of radial passes (= number of crossings)
ray_length_m  : half-length of each ray from centre [m]
centre_offset_m: lateral offset of centre from cable midpoint [m]
angle_spread  : angular spread between rays [deg] (360 = full circle)
angle_start   : starting angle of first ray [deg, nav]
"""
import math
import numpy as np
from .base import BaseTrajectory, direction_to_heading


class StarburstTrajectory(BaseTrajectory):

    def __init__(self, cable_p1, cable_p2, sampling_freq, drone_speed,
                 n_crossings=8, ray_length_m=6.0, centre_offset_m=0.0,
                 angle_spread=180.0, angle_start=0.0, **kwargs):
        super().__init__(cable_p1, cable_p2, sampling_freq, drone_speed)
        self.n_rays          = max(2, int(n_crossings))
        self.ray_length_m    = max(0.5, float(ray_length_m))
        self.centre_offset_m = float(centre_offset_m)
        self.angle_spread    = max(10.0, min(360.0, float(angle_spread)))
        self.angle_start     = float(angle_start)

    def generate_trajectory_local(self):
        d = self.cable_p2 - self.cable_p1
        cable_len = np.linalg.norm(d)
        u_along = d / cable_len
        u_perp  = np.array([-u_along[1], u_along[0]])
        mid     = (self.cable_p1 + self.cable_p2) / 2.0
        centre  = mid + self.metres_to_deg(self.centre_offset_m) * u_perp

        ray_deg = self.metres_to_deg(self.ray_length_m)
        n = self.n_rays
        angles_nav = [
            self.angle_start + i * self.angle_spread / (n - 1 if n > 1 else 1)
            for i in range(n)
        ]

        all_pts, all_hdg = [], []
        prev_tip = None

        for ang_nav in angles_nav:
            # nav→math: math_angle = 90 - nav_angle
            math_ang = math.radians(90.0 - ang_nav)
            direction = np.array([math.sin(math.radians(ang_nav)),   # dlat: sin from nav
                                   math.cos(math.radians(ang_nav - 90))])  # dlon
            # Cleaner: build from nav convention
            dlat = math.cos(math.radians(ang_nav))  # north component
            dlon = math.sin(math.radians(ang_nav))  # east component
            direction = np.array([dlat, dlon])

            tip = centre + ray_deg * direction

            if prev_tip is not None:
                # fly back to centre then out to new tip
                pts_in,  hdg_in  = self._sample_segment(prev_tip, centre)
                pts_out, hdg_out = self._sample_segment(centre, tip)
                all_pts.extend([pts_in, pts_out])
                all_hdg.extend([hdg_in, hdg_out])
            else:
                pts_out, hdg_out = self._sample_segment(centre, tip)
                all_pts.append(pts_out)
                all_hdg.append(hdg_out)

            prev_tip = tip

        return np.vstack(all_pts), np.concatenate(all_hdg)
