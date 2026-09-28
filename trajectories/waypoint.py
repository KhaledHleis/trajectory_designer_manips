"""
trajectories/waypoint.py
------------------------
"Waypoint mode" — generates a minimal list of key waypoints
(not densely sampled) that describe a cable-survey mission as
simple drone commands.

The output is a sparse list: start → WP1 → WP2 → … → end.
Each waypoint has a (lat, lon, heading) and represents a point where
the drone should either start a new leg or turn.

This mode is NOT intended for high-frequency sampled output;
it produces only as many points as there are direction changes.

Pattern
-------
The drone flies a lawnmower-style mission but the output only contains
the corner points of the path (start, cable-crossing, turn, next start, …)
not every sample along the way.  This maps directly to MAVLink/ArduPilot
MISSION_ITEM commands.
"""

import math
import numpy as np
from .base import BaseTrajectory, direction_to_heading


class WaypointTrajectory(BaseTrajectory):
    """
    Minimal waypoints for actual drone mission planning.

    Parameters
    ----------
    n_crossings   : number of cable crossings
    angle_deg     : incidence angle with cable (°)
    pass_width    : fraction of cable length to extend each pass beyond the cable
    side_offset_m : lateral distance between adjacent passes [m]
    """

    def __init__(
        self,
        cable_p1,
        cable_p2,
        sampling_freq,      # kept for API compat, not used for WP spacing
        drone_speed,
        n_crossings: int = 5,
        angle_deg: float = 90.0,
        turn_sharpness: float = 0.9,
        pass_width: float = 0.55,
        **kwargs,
    ):
        super().__init__(cable_p1, cable_p2, sampling_freq, drone_speed)
        self.n_crossings = max(1, int(n_crossings))
        self.angle_deg = max(5.0, min(175.0, float(angle_deg)))
        self.pass_width = max(0.01, float(pass_width))

    def generate_trajectory_local(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns sparse waypoints + headings.
        Each point is a mission waypoint (corner of the path).
        """
        d = self.cable_p2 - self.cable_p1
        cable_len = np.linalg.norm(d)
        if cable_len < 1e-15:
            return np.empty((0, 2)), np.empty((0,))
        u_along = d / cable_len
        u_perp = np.array([-u_along[1], u_along[0]])

        alpha = math.radians(self.angle_deg)
        cross_dir = math.sin(alpha) * u_perp + math.cos(alpha) * u_along
        cross_dir /= np.linalg.norm(cross_dir)

        n = self.n_crossings
        offsets = [0.0] if n == 1 else list(np.linspace(0.0, cable_len, n))
        pass_half = cable_len * self.pass_width

        wps: list[np.ndarray] = []
        hdgs: list[float] = []

        for i, off in enumerate(offsets):
            anchor = self.cable_p1 + off * u_along
            direction = cross_dir if (i % 2 == 0) else -cross_dir
            h = direction_to_heading(direction[0], direction[1])

            p_start = anchor - pass_half * direction
            p_end   = anchor + pass_half * direction

            # For the very first pass we also emit the start
            if i == 0:
                wps.append(p_start)
                hdgs.append(h)

            # Cable-crossing midpoint
            wps.append(anchor)
            hdgs.append(h)

            # End of pass
            wps.append(p_end)
            hdgs.append(h)

            # If not the last pass, add a "turn" waypoint
            if i < n - 1:
                next_off = offsets[i + 1]
                next_anchor = self.cable_p1 + next_off * u_along
                next_dir = -direction  # alternating
                next_h = direction_to_heading(next_dir[0], next_dir[1])
                next_start = next_anchor - pass_half * next_dir
                # Transit waypoint = midpoint of p_end → next_start
                transit = (p_end + next_start) / 2.0
                # Heading for transit leg
                tr = next_start - p_end
                tr_norm = np.linalg.norm(tr)
                if tr_norm > 1e-12:
                    tr /= tr_norm
                    tr_h = direction_to_heading(tr[0], tr[1])
                else:
                    tr_h = h
                wps.append(transit)
                hdgs.append(tr_h)

        pts = np.array(wps)
        return pts, np.array(hdgs)
