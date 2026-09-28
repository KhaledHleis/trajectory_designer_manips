"""
trajectories/lawnmower.py
-------------------------
Lawnmower trajectory with guaranteed cable-crossing angles.

Core idea
---------
The crossing point and its tangent direction are **hard constraints**.
Rather than constructing arcs that may overshoot, each pass is built as:

  pass_start  ──straight──►  crossing_point  ──straight──►  pass_end
                                    ↑
                              HARD CONSTRAINT:
                              direction == cross_dir (or -cross_dir)

The turn between pass i end and pass i+1 start is a **cubic Hermite spline**:
  - starts at pass_end_i       with tangent = pass_direction_i
  - ends   at pass_start_{i+1} with tangent = pass_direction_{i+1}

This guarantees:
  ✓ Every cable crossing is exactly at the right angle (by construction)
  ✓ C1 continuity everywhere (no heading jumps)
  ✓ Turn shape is free (Hermite handles it gracefully)
  ✓ No arc radius / overshoot problems

Hermite spline
--------------
P(t) = h00(t)*p0 + h10(t)*m0 + h01(t)*p1 + h11(t)*m1,  t ∈ [0,1]
where m0, m1 are the tangent *vectors* (not unit), scaled so the turn
looks natural (scale = chord_length * tension).
"""

import math
import numpy as np
from .base import BaseTrajectory, direction_to_heading


class LawnmowerTrajectory(BaseTrajectory):

    def __init__(
        self,
        cable_p1,
        cable_p2,
        sampling_freq,
        drone_speed,
        n_crossings: int = 5,
        angle_deg: float = 90.0,
        turn_sharpness: float = 1.0,   # tension: higher → tighter, closer to straight
        pass_width: float = 0.55,
        **kwargs,
    ):
        super().__init__(cable_p1, cable_p2, sampling_freq, drone_speed)
        self.n_crossings   = max(1, int(n_crossings))
        self.angle_deg     = max(5.0, min(175.0, float(angle_deg)))
        self.turn_sharpness = max(0.1, min(10.0, float(turn_sharpness)))
        self.pass_width    = max(0.05, float(pass_width))

    # ── cable frame ───────────────────────────────────────────────────────────

    def _cable_frame(self):
        d = self.cable_p2 - self.cable_p1
        cable_len = np.linalg.norm(d)
        if cable_len < 1e-15:
            cable_len = 1e-15
        u_along = d / cable_len
        u_perp  = np.array([-u_along[1], u_along[0]])
        return u_along, u_perp, cable_len

    def _crossing_dir(self, u_along, u_perp):
        """Unit vector for even passes: angle_deg w.r.t. cable."""
        alpha = math.radians(self.angle_deg)
        v = math.sin(alpha) * u_perp + math.cos(alpha) * u_along
        return v / np.linalg.norm(v)

    # ── Hermite spline turn ───────────────────────────────────────────────────

    def _hermite_turn(
        self,
        p0: np.ndarray, d0: np.ndarray,   # start point + unit tangent
        p1: np.ndarray, d1: np.ndarray,   # end   point + unit tangent
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Cubic Hermite arc from p0 (tangent d0) to p1 (tangent d1).
        Returns (pts, headings).
        """
        chord = np.linalg.norm(p1 - p0)
        if chord < 1e-12:
            h = direction_to_heading(d1[0], d1[1])
            return p1.reshape(1, 2), np.array([h])

        # Scale tangents by chord * tension so the spline looks natural
        tension = self.turn_sharpness
        m0 = d0 * chord * tension
        m1 = d1 * chord * tension

        # Arc length estimate (chord is a lower bound; Hermite is longer)
        arc_len_est = chord * (1.0 + tension * 0.5)
        arc_len_m = self.deg_to_metres(arc_len_est)
        n = max(4, int(arc_len_m / self.drone_speed * self.sampling_freq))

        t = np.linspace(0.0, 1.0, n)

        # Hermite basis functions
        h00 =  2*t**3 - 3*t**2 + 1
        h10 =    t**3 - 2*t**2 + t
        h01 = -2*t**3 + 3*t**2
        h11 =    t**3 -   t**2

        pts = (np.outer(h00, p0) + np.outer(h10, m0) +
               np.outer(h01, p1) + np.outer(h11, m1))

        # Derivative for heading
        dh00 =  6*t**2 - 6*t
        dh10 =  3*t**2 - 4*t + 1
        dh01 = -6*t**2 + 6*t
        dh11 =  3*t**2 - 2*t

        dpts = (np.outer(dh00, p0) + np.outer(dh10, m0) +
                np.outer(dh01, p1) + np.outer(dh11, m1))

        headings = np.array([
            direction_to_heading(dpts[i, 0], dpts[i, 1])
            for i in range(n)
        ])
        return pts, headings

    # ── straight segment (with heading) ──────────────────────────────────────

    def _straight(
        self, p0: np.ndarray, p1: np.ndarray, direction: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Straight segment forced to use `direction` as heading."""
        seg_pts, _ = self._sample_segment(p0, p1)
        h = direction_to_heading(direction[0], direction[1])
        return seg_pts, np.full(len(seg_pts), h)

    # ── main generator ────────────────────────────────────────────────────────

    def generate_trajectory_local(self) -> tuple[np.ndarray, np.ndarray]:
        u_along, u_perp, cable_len = self._cable_frame()
        cross_fwd = self._crossing_dir(u_along, u_perp)

        n = self.n_crossings
        offsets = [0.0] if n == 1 else list(np.linspace(0.0, cable_len, n))
        pass_half = cable_len * self.pass_width

        all_pts:  list[np.ndarray] = []
        all_hdg:  list[np.ndarray] = []

        prev_end     : np.ndarray | None = None
        prev_dir     : np.ndarray | None = None

        for i, off in enumerate(offsets):
            anchor    = self.cable_p1 + off * u_along
            direction = cross_fwd if (i % 2 == 0) else -cross_fwd

            # ── pass: [pass_start] → [anchor] → [pass_end] ──────────────────
            pass_start = anchor - pass_half * direction
            pass_end   = anchor + pass_half * direction

            if prev_end is not None:
                # Hermite turn: prev_end → pass_start
                turn_pts, turn_hdg = self._hermite_turn(
                    prev_end, prev_dir, pass_start, direction
                )
                all_pts.append(turn_pts)
                all_hdg.append(turn_hdg)

            # Straight: pass_start → anchor → pass_end  (enforces crossing angle)
            seg_pts, seg_hdg = self._straight(pass_start, pass_end, direction)
            all_pts.append(seg_pts)
            all_hdg.append(seg_hdg)

            prev_end = pass_end.copy()
            prev_dir = direction.copy()

        if not all_pts:
            return np.empty((0, 2)), np.empty((0,))

        return np.vstack(all_pts), np.concatenate(all_hdg)
