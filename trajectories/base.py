"""
trajectories/base.py
--------------------
Abstract base class shared by all trajectory algorithms.

Coordinate system
-----------------
Cable endpoints are given in (lat, lon) decimal degrees, and trajectories are
returned in (lat, lon) decimal degrees — but ALL geometry in between is done
in a local tangent-plane frame in **metres**, with axes (north, east).

This matters, and it is the fix for a real defect. This module used to do its
geometry directly on (lat, lon) degree pairs, with a single ISOTROPIC scale
factor:

    self._m_per_deg = (m_per_deg_lat + m_per_deg_lon) / 2      # WRONG

i.e. one degree of longitude was treated as the same distance as one degree of
latitude. At the Brest test site (lat 48.49) a degree of longitude is only
~0.66 of a degree of latitude, so that plane is stretched east-west by ~1.5x,
and three things came out wrong at once:

  1. HEADING. `direction_to_heading` reads the direction off the working
     plane, so the exported heading was the course through the *distorted*
     plane, not the real ground track. Measured: +9.81 deg for a 60 deg
     lawnmower, +11.59 deg for a 45 deg zigzag, on every straight leg. The
     simulator (mSIMU) projects the same lon/lat correctly with pyproj and
     then obeys the commanded heading verbatim — as it should, a real
     platform can crab — so every dataset generated from those files has the
     drone's nose permanently off its own track.

  2. THE CROSSING ANGLE ITSELF. `angle_deg` is measured against the cable
     direction in the working plane, so the commanded angle is not the angle
     actually flown. Measured: commanding 60 deg produced a true crossing
     angle of 61.891 deg. That is where the 61.89 deg in the MARTOC handoff
     comes from, and why a file named "S60" is not a 60 deg survey.

  3. GROUND SPEED AND DISTANCES. Every metre-valued parameter (offsets,
     radii, step sizes, and the speed used to pick the sample count) was
     converted with the same averaged scale factor. Measured: 1.947 m/s
     actually flown against 1.78 m/s commanded, +9.4%.

Working in metres removes all three by construction: the plane is isotropic,
so a heading read off it IS the ground course, an angle measured in it IS the
angle flown, and a metre IS a metre. Nothing but this file and the projection
boundary needs to know — subclasses build their shapes in the working plane
exactly as before, and `metres_to_local` (formerly `metres_to_deg`) is now
simply the identity.

Subclasses implement `generate_trajectory_local()`, returning points in the
local metre frame. The base class's `generate_trajectory()` projects those
back to (lat, lon) for callers, so the public API is unchanged.

Heading convention
------------------
All headings are in **navigation convention**:
  0° = North, 90° = East, 180° = South, 270° = West  (clockwise from North)
Range: [0, 360)
"""

import math
from abc import ABC, abstractmethod

import numpy as np

try:  # same projection the simulator uses, so the two agree exactly
    from pyproj import Transformer

    _HAVE_PYPROJ = True
except ImportError:  # pragma: no cover - fallback path
    _HAVE_PYPROJ = False


def direction_to_heading(d_north: float, d_east: float) -> float:
    """
    Convert a (north, east) direction vector to navigation heading [0, 360).
    North = 0°, East = 90°.

    The arguments are METRES in the local frame, not degrees of lat/lon —
    passing raw degree differences here is what produced the heading error
    described in the module docstring.
    """
    h = math.degrees(math.atan2(d_east, d_north))
    return h % 360.0


class LocalFrame:
    """
    Local tangent plane in metres about (lat0, lon0), axes (north, east).

    Uses pyproj transverse Mercator when available — identical to
    mSIMU's backend/utilities/utilities_converter.py, so a trajectory
    generated here and re-projected by the simulator round-trips. Falls
    back to a WGS84 series expansion for the degree scale factors, which is
    accurate to about a centimetre per degree and keeps the two axes
    correctly ANISOTROPIC (the fallback must never average them — that is
    the original bug).
    """

    def __init__(self, lat0: float, lon0: float):
        self.lat0 = float(lat0)
        self.lon0 = float(lon0)
        if _HAVE_PYPROJ:
            proj = (
                f"+proj=tmerc +lat_0={self.lat0} +lon_0={self.lon0} "
                f"+k=1 +x_0=0 +y_0=0 +ellps=WGS84"
            )
            self._fwd = Transformer.from_crs("epsg:4326", proj, always_xy=True)
            self._inv = Transformer.from_crs(proj, "epsg:4326", always_xy=True)
        else:
            phi = math.radians(self.lat0)
            self._m_per_deg_lat = (
                111132.92
                - 559.82 * math.cos(2 * phi)
                + 1.175 * math.cos(4 * phi)
                - 0.0023 * math.cos(6 * phi)
            )
            self._m_per_deg_lon = (
                111412.84 * math.cos(phi)
                - 93.5 * math.cos(3 * phi)
                + 0.118 * math.cos(5 * phi)
            )

    def to_local(self, latlon) -> np.ndarray:
        """(..., 2) of (lat, lon) degrees -> (..., 2) of (north, east) metres."""
        arr = np.asarray(latlon, dtype=float).reshape(-1, 2)
        if arr.size == 0:
            return arr.copy()
        if _HAVE_PYPROJ:
            east, north = self._fwd.transform(arr[:, 1], arr[:, 0])
            out = np.column_stack([np.atleast_1d(north), np.atleast_1d(east)])
        else:
            out = np.column_stack(
                [
                    (arr[:, 0] - self.lat0) * self._m_per_deg_lat,
                    (arr[:, 1] - self.lon0) * self._m_per_deg_lon,
                ]
            )
        return out.reshape(np.shape(latlon))

    def to_latlon(self, north_east) -> np.ndarray:
        """(..., 2) of (north, east) metres -> (..., 2) of (lat, lon) degrees."""
        arr = np.asarray(north_east, dtype=float).reshape(-1, 2)
        if arr.size == 0:
            return arr.copy()
        if _HAVE_PYPROJ:
            lon, lat = self._inv.transform(arr[:, 1], arr[:, 0])
            out = np.column_stack([np.atleast_1d(lat), np.atleast_1d(lon)])
        else:
            out = np.column_stack(
                [
                    self.lat0 + arr[:, 0] / self._m_per_deg_lat,
                    self.lon0 + arr[:, 1] / self._m_per_deg_lon,
                ]
            )
        return out.reshape(np.shape(north_east))


class BaseTrajectory(ABC):
    """
    Parameters
    ----------
    cable_p1, cable_p2 : (lat, lon) tuples defining the cable endpoints
    sampling_freq      : sample rate [Hz]
    drone_speed        : linear speed [m/s]

    Subclasses see `self.cable_p1` / `self.cable_p2` in the LOCAL METRE
    frame (north, east) and must implement `generate_trajectory_local`.
    The original (lat, lon) endpoints stay available as
    `self.cable_p1_latlon` / `self.cable_p2_latlon`.
    """

    def __init__(self, cable_p1, cable_p2, sampling_freq, drone_speed):
        self.cable_p1_latlon = np.array(cable_p1, dtype=float)
        self.cable_p2_latlon = np.array(cable_p2, dtype=float)
        self.sampling_freq = float(sampling_freq)
        self.drone_speed = float(drone_speed)

        self.frame = LocalFrame(
            (self.cable_p1_latlon[0] + self.cable_p2_latlon[0]) / 2.0,
            (self.cable_p1_latlon[1] + self.cable_p2_latlon[1]) / 2.0,
        )
        self.cable_p1 = self.frame.to_local(self.cable_p1_latlon)
        self.cable_p2 = self.frame.to_local(self.cable_p2_latlon)

        # The working plane IS metres now, so this is 1.0 by construction.
        # It is kept only so the metres_to_local / local_to_metres helpers
        # below read the same as they always did in subclasses.
        self._m_per_deg = 1.0

    # ── unit helpers ──────────────────────────────────────────────────────────

    def metres_to_local(self, metres: float) -> float:
        """Metres -> working-plane units. Identity: the plane is metres."""
        return metres

    def local_to_metres(self, units: float) -> float:
        """Working-plane units -> metres. Identity: the plane is metres."""
        return units

    # Backwards-compatible names. They used to convert metres <-> degrees
    # through an averaged, isotropic scale factor; the working plane is now
    # metric, so both are the identity. Prefer the *_local names in new code.
    metres_to_deg = metres_to_local
    deg_to_metres = local_to_metres

    # ── public API ────────────────────────────────────────────────────────────

    def generate_trajectory(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Return ((N,2) array of (lat, lon), (N,) array of headings in deg).

        Headings come straight from the subclass: they were measured in the
        local metric plane, so they are already true ground courses and need
        no correction here. Only the POSITIONS are projected back.
        """
        pts_local, headings = self.generate_trajectory_local()
        pts_local = np.asarray(pts_local, dtype=float).reshape(-1, 2)
        headings = np.asarray(headings, dtype=float).reshape(-1)
        if pts_local.size == 0:
            return np.empty((0, 2)), np.empty((0,))
        return self.frame.to_latlon(pts_local), headings

    @abstractmethod
    def generate_trajectory_local(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Return ((N,2) array of (north, east) METRES, (N,) headings in deg).

        Build geometry here exactly as you would on graph paper: the frame is
        isotropic and metric, so lengths are metres and angles are real.
        """

    # ── sampling helpers ──────────────────────────────────────────────────────

    def _sample_segment(self, p_start: np.ndarray, p_end: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Uniformly sample a straight segment. Returns (pts, headings)."""
        d = p_end - p_start
        dist_m = np.linalg.norm(d)
        if dist_m < 1e-9:
            h = direction_to_heading(0.0, 0.0)
            return p_start.reshape(1, 2), np.array([h])
        n = max(2, int(dist_m / self.drone_speed * self.sampling_freq))
        t = np.linspace(0, 1, n)
        pts = p_start + np.outer(t, d)
        heading = direction_to_heading(d[0], d[1])
        headings = np.full(n, heading)
        return pts, headings

    def _sample_arc(
        self, centre: np.ndarray, radius_m: float, a_start: float, a_end: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Sample a circular arc. a_start/a_end are math angles (CCW from +north axis).
        Returns (pts, headings) where headings follow navigation convention.
        """
        arc_len_m = abs(a_end - a_start) * radius_m
        if arc_len_m < 1e-9:
            pt = centre + radius_m * np.array([math.cos(a_start), math.sin(a_start)])
            heading = direction_to_heading(
                -math.sin(a_start) * (1 if a_end >= a_start else -1),
                math.cos(a_start) * (1 if a_end >= a_start else -1),
            )
            return pt.reshape(1, 2), np.array([heading])
        n = max(4, int(arc_len_m / self.drone_speed * self.sampling_freq))
        angles = np.linspace(a_start, a_end, n)
        pts = centre + radius_m * np.column_stack([np.cos(angles), np.sin(angles)])
        # tangent direction: CCW orbit => tangent = (-sin a, cos a), CW => (sin a, -cos a)
        ccw = (a_end > a_start)
        sign = 1.0 if ccw else -1.0
        d_north = -np.sin(angles) * sign
        d_east = np.cos(angles) * sign
        headings = np.array([direction_to_heading(dn, de) for dn, de in zip(d_north, d_east)])
        return pts, headings
