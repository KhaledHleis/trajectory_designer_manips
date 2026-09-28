"""
Sinusoidal trajectory
---------------------
The drone follows a sine wave path along the cable axis.
The wave crosses the cable repeatedly at a controlled angle.

Parameters
----------
n_crossings   : number of cable crossings
angle_deg     : incidence angle at each crossing (controls amplitude/wavelength ratio)
amplitude_m   : peak lateral deviation from cable [m]
pass_width    : longitudinal extension beyond cable ends (fraction of cable length)
phase_offset  : phase shift [0,1] — shifts the wave start
"""
import math
import numpy as np
from .base import BaseTrajectory, direction_to_heading


class SinusoidalTrajectory(BaseTrajectory):

    def __init__(self, cable_p1, cable_p2, sampling_freq, drone_speed,
                 n_crossings=8, angle_deg=60.0, amplitude_m=3.0,
                 pass_width=0.1, phase_offset=0.0, **kwargs):
        super().__init__(cable_p1, cable_p2, sampling_freq, drone_speed)
        self.n_crossings = max(2, int(n_crossings))
        self.angle_deg   = max(5.0, min(85.0, float(angle_deg)))
        self.amplitude_m = max(0.1, float(amplitude_m))
        self.pass_width  = max(0.0, float(pass_width))
        self.phase_offset = float(phase_offset)

    def generate_trajectory_local(self):
        d = self.cable_p2 - self.cable_p1
        cable_len = np.linalg.norm(d)
        u_along = d / cable_len
        u_perp  = np.array([-u_along[1], u_along[0]])

        amp_deg = self.metres_to_deg(self.amplitude_m)
        ext     = cable_len * self.pass_width

        # n full half-cycles = n_crossings zero-crossings
        half_cycles = self.n_crossings
        total_along = cable_len + 2 * ext
        t_start = -ext / cable_len  # normalised along-cable coord

        # Total samples
        total_m = self.deg_to_metres(total_along)
        # approximate arc length of sinusoid
        arc_factor = math.sqrt(1 + (amp_deg * math.pi * half_cycles / total_along)**2)
        n_pts = max(8, int(total_m * arc_factor / self.drone_speed * self.sampling_freq))

        t_vals = np.linspace(t_start, t_start + total_along / cable_len, n_pts)
        along  = self.cable_p1 + np.outer(t_vals * cable_len, u_along)

        phase  = self.phase_offset * 2 * math.pi
        wave   = amp_deg * np.sin(math.pi * half_cycles * t_vals + phase)
        pts    = along + np.outer(wave, u_perp)

        # Headings from finite differences
        dp = np.gradient(pts, axis=0)
        headings = np.array([direction_to_heading(dp[i,0], dp[i,1]) for i in range(n_pts)])
        return pts, headings
