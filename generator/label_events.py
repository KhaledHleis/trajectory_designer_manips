"""
generator/label_events.py
--------------------------
Extracts cable-crossing events from a sampled trajectory.
These are the ground-truth labels for the ML detection task.

Each crossing event records:
  - sample index of the crossing
  - lat/lon at crossing
  - incidence angle (deg) — actual measured angle at crossing
  - approach heading (deg, nav) — drone heading just before crossing
  - crossing_speed_ms — drone speed component normal to cable
  - along_frac — fractional position along cable [0, 1]

Also computes per-sample features useful for training:
  - dist_to_cable_m  — signed perpendicular distance to cable
  - along_frac       — normalised position projected onto cable axis
  - heading_deg      — drone heading
  - speed_mps        — estimated speed (from sampling freq and step size)
"""
import math
import numpy as np

from trajectories.base import LocalFrame


def _cable_frame(pts, cable_p1, cable_p2):
    """
    Project the trajectory and the cable into a local metric frame
    (north, east metres) and return everything the feature/label code needs.

    Both functions below used to do this arithmetic straight on (lat, lon)
    degree pairs and then multiply by a single `m_per_deg = 111_320.0`. At
    this latitude a degree of longitude is only ~0.66 of a degree of
    latitude, so that plane is stretched ~1.5x east-west: `dist_cable_m` was
    wrong by a factor that depended on the cable's orientation, and
    `incidence_deg` — the ML ground-truth label — was the angle in the
    distorted plane, not the angle actually flown. Same defect as the one
    fixed in trajectories/base.py, and it has to be fixed in both places:
    fixing the generator alone would leave the labels disagreeing with the
    trajectories they describe.
    """
    p1_ll = np.asarray(cable_p1, dtype=float)
    p2_ll = np.asarray(cable_p2, dtype=float)
    frame = LocalFrame((p1_ll[0] + p2_ll[0]) / 2.0, (p1_ll[1] + p2_ll[1]) / 2.0)

    p1 = frame.to_local(p1_ll)
    p2 = frame.to_local(p2_ll)
    xy = frame.to_local(np.asarray(pts, dtype=float).reshape(-1, 2))

    d = p2 - p1
    cable_len = float(np.linalg.norm(d))
    u_along = d / cable_len if cable_len > 1e-9 else np.array([1.0, 0.0])
    u_perp = np.array([-u_along[1], u_along[0]])
    return frame, xy, p1, u_along, u_perp, cable_len


def extract_crossings(pts: np.ndarray, headings: np.ndarray,
                      cable_p1, cable_p2,
                      sampling_freq: float, drone_speed: float) -> list[dict]:
    """
    Returns list of crossing event dicts.
    """
    if np.linalg.norm(np.asarray(cable_p2, float) - np.asarray(cable_p1, float)) < 1e-15:
        return []
    frame, xy, p1, u_along, u_perp, cable_len = _cable_frame(pts, cable_p1, cable_p2)
    if cable_len < 1e-9:
        return []

    # Perpendicular distance (signed) for each sample, in METRES
    v = xy - p1
    along_vals  = v @ u_along    # projection onto cable axis
    lateral_vals = v @ u_perp   # perpendicular (signed)

    crossings = []
    for i in range(1, len(pts)):
        # Sign change AND within cable extent (with small margin)
        if lateral_vals[i - 1] * lateral_vals[i] < 0:
            # Linear interpolation for exact crossing index
            frac = abs(lateral_vals[i - 1]) / (abs(lateral_vals[i - 1]) + abs(lateral_vals[i]) + 1e-15)
            cross_pt   = xy[i-1] + frac * (xy[i] - xy[i-1])
            along_frac = float(np.dot(cross_pt - p1, u_along) / cable_len)

            # Only count if crossing is within cable bounds (±10% margin)
            if not (-0.1 <= along_frac <= 1.1):
                continue

            # Tangent at crossing
            i0 = max(0, i - 3)
            i1 = min(len(pts) - 1, i + 3)
            tangent = xy[i1] - xy[i0]
            t_norm  = np.linalg.norm(tangent)
            if t_norm < 1e-15:
                continue
            tangent /= t_norm

            # Angle between the flight tangent and the cable axis, folded to
            # [0, 90]: 0 = flying along the cable, 90 = perpendicular
            # crossing. This is the SAME convention cable_pipeline's angle
            # estimators report, so a label can be compared against an
            # estimate without a mental flip.
            #
            # NB this used to be `90 - angle_from_cable`, which is the
            # complement — it reported 90 for a pass flown parallel to the
            # cable and 0 for a perpendicular one, the exact opposite of the
            # comment beside it and of the pipeline.
            dot = abs(float(np.dot(tangent, u_along)))
            dot = min(1.0, dot)
            incidence_deg = math.degrees(math.acos(dot))

            approach_heading = float(headings[i])

            cross_ll = frame.to_latlon(cross_pt)
            crossings.append({
                "sample_idx":     int(i),
                "lat":            float(cross_ll[0]),
                "lon":            float(cross_ll[1]),
                "incidence_deg":  round(incidence_deg, 3),
                "heading_deg":    round(approach_heading, 3),
                "along_frac":     round(along_frac, 4),
                "lateral_sign":   int(np.sign(lateral_vals[i - 1])),
            })

    return crossings


def compute_sample_features(pts: np.ndarray, headings: np.ndarray,
                              cable_p1, cable_p2,
                              sampling_freq: float, drone_speed: float) -> np.ndarray:
    """
    Returns (N, 5) array of per-sample features:
      col 0: dist_to_cable_m    (signed, + = left of cable direction)
      col 1: along_frac         (0=P1 end, 1=P2 end, can exceed [0,1])
      col 2: heading_deg        (nav)
      col 3: heading_sin        (sin of heading rad — circular feature)
      col 4: heading_cos        (cos of heading rad — circular feature)
    """
    frame, xy, p1, u_along, u_perp, cable_len = _cable_frame(pts, cable_p1, cable_p2)
    cable_len = cable_len + 1e-15

    v = xy - p1
    lateral  = v @ u_perp                 # metres, signed — the frame IS metric
    along_f  = (v @ u_along) / cable_len

    hdg_rad  = np.radians(headings)
    features = np.column_stack([
        lateral,
        along_f,
        headings,
        np.sin(hdg_rad),
        np.cos(hdg_rad),
    ])
    return features
