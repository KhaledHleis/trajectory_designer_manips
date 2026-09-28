"""
generator/verify.py
-------------------
One check, run on every trajectory this project emits: does the exported
`heading` column agree with the course its own consecutive (lat, lon) points
imply?

It has to, and for a long time it did not. Geometry was built on raw
lat/lon degree pairs with one averaged scale factor for both axes, so the
heading written out was the course through a plane stretched ~1.5x east-west
— off by +9.8 deg on a 60 deg lawnmower, +11.6 deg on a 45 deg zigzag. The
simulator projects lon/lat correctly and then obeys the commanded heading
verbatim, so nothing downstream can detect the disagreement; it just flies
every leg crabbed and quietly corrupts any bearing or crossing-angle
estimate derived from the result.

The fix lives in trajectories/base.py. This function exists so a regression
cannot get out of the door silently.
"""

import numpy as np

from trajectories.base import LocalFrame


def heading_consistency(pts_latlon, headings, turn_rate_threshold: float = 0.05) -> dict:
    """
    Compare `headings` against the course implied by `pts_latlon`.

    Args:
        pts_latlon: (N, 2) array of (lat, lon) degrees.
        headings: (N,) navigation headings, degrees clockwise from North.
        turn_rate_threshold: deg/sample below which a sample counts as being
            on a straight leg. Turns have a real heading rate and a noisy
            finite-difference course, so they are excluded.

    Returns:
        {"median_deg": float, "max_deg": float, "n_straight": int,
         "consistent": bool} — consistent means |median| < 0.1 deg.
    """
    pts = np.asarray(pts_latlon, dtype=float).reshape(-1, 2)
    hdg = np.asarray(headings, dtype=float).reshape(-1)
    if len(pts) < 4:
        return {"median_deg": 0.0, "max_deg": 0.0, "n_straight": 0, "consistent": True}

    frame = LocalFrame(float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1])))
    xy = frame.to_local(pts)
    course = np.degrees(np.arctan2(np.gradient(xy[:, 1]), np.gradient(xy[:, 0]))) % 360.0

    unwrapped = np.degrees(np.unwrap(np.radians(hdg)))
    straight = np.abs(np.gradient(unwrapped)) < turn_rate_threshold
    if straight.sum() < 4:
        straight = np.ones(len(hdg), dtype=bool)

    diff = (course[straight] - hdg[straight] + 180.0) % 360.0 - 180.0
    median = float(np.median(diff))
    return {
        "median_deg": median,
        "max_deg": float(np.max(np.abs(diff))),
        "n_straight": int(straight.sum()),
        "consistent": bool(abs(median) < 0.1),
    }
