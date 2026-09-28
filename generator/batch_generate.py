"""
generator/batch_generate.py
----------------------------
Main batch generation script.

Usage:
    python batch_generate.py [--config config/default_config.json]
                             [--n N]          # override n_trajectories
                             [--out OUTPUT_DIR]
                             [--seed SEED]

Outputs per trajectory (in output/<name>/traj_NNNN/):
    trajectory.csv     — lat, lon, heading, dist_to_cable_m, along_frac, ...
    metadata.json      — all parameters used + crossing events
    features.npy       — (N, 5) float32 feature array (for fast ML loading)
    labels.npy         — (N,)   int8 array: 1 at crossing samples, 0 elsewhere

Dataset-level outputs (in output/<name>/):
    manifest.csv       — one row per trajectory: id, type, n_pts, n_crossings, ...
    dataset_info.json  — full config + stats
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trajectories import TRAJECTORY_REGISTRY
from generator.sampler import sample_params, sample_global
from generator.label_events import extract_crossings, compute_sample_features
from generator.verify import heading_consistency


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _resolve_output_dir(cfg: dict, override: str | None) -> str:
    base = override or cfg["dataset"].get("output_dir", "output")
    name = cfg["dataset"].get("name", "dataset")
    return os.path.join(base, name)


def _make_labels_array(n_pts: int, crossings: list[dict]) -> np.ndarray:
    labels = np.zeros(n_pts, dtype=np.int8)
    for ev in crossings:
        idx = ev["sample_idx"]
        # Mark a small window around each crossing (±3 samples)
        lo = max(0, idx - 3)
        hi = min(n_pts - 1, idx + 3)
        labels[lo:hi+1] = 1
    return labels


def _write_trajectory(traj_dir: str, pts, headings, features, labels,
                       crossings, params_used, traj_id, traj_type,
                       cable_p1, cable_p2, formats):
    os.makedirs(traj_dir, exist_ok=True)

    # ── CSV ──────────────────────────────────────────────────────────────────
    if "csv" in formats:
        df = pd.DataFrame({
            "sample_idx":      np.arange(len(pts)),
            "latitude":        pts[:, 0],
            "longitude":       pts[:, 1],
            "heading_deg":     headings,
            "dist_cable_m":    features[:, 0],
            "along_frac":      features[:, 1],
            "heading_sin":     features[:, 3],
            "heading_cos":     features[:, 4],
            "is_crossing":     labels.astype(int),
        })
        # 10 decimal places, not 8. At 50 Hz and 1-3 m/s the along-track step
        # is 2-6 cm, while 8 dp of latitude quantises position to ~1.1 mm —
        # enough that a course recomputed from the CSV by finite differences
        # picks up a staircase of up to ~2 deg. 10 dp puts that below a
        # hundredth of a degree, and matches what the waypoint exporter and
        # the existing waypoints_*.csv files already use.
        df.to_csv(os.path.join(traj_dir, "trajectory.csv"), index=False, float_format="%.10f")

    # ── NumPy arrays ─────────────────────────────────────────────────────────
    np.save(os.path.join(traj_dir, "features.npy"), features.astype(np.float32))
    np.save(os.path.join(traj_dir, "labels.npy"),   labels)
    np.save(os.path.join(traj_dir, "positions.npy"), pts.astype(np.float64))

    # ── metadata JSON ─────────────────────────────────────────────────────────
    if "json" in formats:
        meta = {
            "id":            traj_id,
            "type":          traj_type,
            "cable_p1":      cable_p1,
            "cable_p2":      cable_p2,
            "params":        {k: (int(v) if isinstance(v, np.integer) else
                                  float(v) if isinstance(v, (np.floating, float)) else v)
                              for k, v in params_used.items()},
            "n_samples":     int(len(pts)),
            "n_crossings":   int(len(crossings)),
            "crossings":     crossings,
            "feature_names": ["dist_cable_m", "along_frac", "heading_deg",
                               "heading_sin", "heading_cos"],
        }
        with open(os.path.join(traj_dir, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)


# ── main generation loop ──────────────────────────────────────────────────────

def generate(config_path: str, n_override=None, out_override=None, seed_override=None):
    cfg    = _load_config(config_path)
    ds_cfg = cfg["dataset"]
    n_traj = n_override or ds_cfg["n_trajectories"]
    seed   = seed_override if seed_override is not None else ds_cfg.get("seed", 42)
    rng    = np.random.default_rng(seed)

    out_dir  = _resolve_output_dir(cfg, out_override)
    os.makedirs(out_dir, exist_ok=True)
    formats  = ds_cfg.get("formats", ["csv", "json"])

    cable_p1 = tuple(cfg["cable"]["p1"])
    cable_p2 = tuple(cfg["cable"]["p2"])
    types    = ds_cfg["trajectory_types"]
    weights  = ds_cfg.get("trajectory_weights")
    if weights is not None:
        weights = np.array(weights, dtype=float)
        weights /= weights.sum()

    manifest_rows = []
    worst_heading_err = 0.0
    t0 = time.time()

    for i in range(n_traj):
        # Pick trajectory type
        traj_type = rng.choice(types, p=weights)
        klass     = TRAJECTORY_REGISTRY[traj_type]

        # Sample global params
        g_params  = sample_global(cfg["global"], rng)
        # Sample type-specific params
        t_cfg     = cfg["per_type"].get(traj_type, {})
        t_params  = sample_params(t_cfg, rng)

        params = {
            "cable_p1":     cable_p1,
            "cable_p2":     cable_p2,
            "sampling_freq": g_params.get("sampling_freq", 50.0),
            "drone_speed":   g_params.get("drone_speed", 1.5),
            **t_params,
        }

        try:
            traj   = klass(**params)
            pts, headings = traj.generate_trajectory()
        except Exception as e:
            print(f"  [WARN] traj {i:04d} ({traj_type}) failed: {e}")
            continue

        if len(pts) < 4:
            print(f"  [WARN] traj {i:04d} ({traj_type}) too short, skipping")
            continue

        # The heading column must be the course these very points imply.
        # See generator/verify.py — this used to be off by ~10 deg on every
        # straight leg, and nothing downstream could tell.
        hc = heading_consistency(pts, headings)
        if not hc["consistent"]:
            print(f"  [WARN] traj {i:04d} ({traj_type}) heading disagrees with its own "
                  f"ground track by {hc['median_deg']:+.3f} deg — NOT written")
            continue
        worst_heading_err = max(worst_heading_err, abs(hc["median_deg"]))

        crossings = extract_crossings(pts, headings, cable_p1, cable_p2,
                                       params["sampling_freq"], params["drone_speed"])
        features  = compute_sample_features(pts, headings, cable_p1, cable_p2,
                                             params["sampling_freq"], params["drone_speed"])
        labels    = _make_labels_array(len(pts), crossings)

        traj_id  = f"traj_{i:04d}"
        traj_dir = os.path.join(out_dir, traj_id)
        _write_trajectory(traj_dir, pts, headings, features, labels,
                           crossings, params, traj_id, traj_type,
                           list(cable_p1), list(cable_p2), formats)

        manifest_rows.append({
            "id":            traj_id,
            "type":          traj_type,
            "n_samples":     len(pts),
            "n_crossings":   len(crossings),
            "sampling_freq": params["sampling_freq"],
            "drone_speed":   params["drone_speed"],
            "duration_s":    round(len(pts) / params["sampling_freq"], 2),
            "dir":           traj_dir,
        })

        if (i + 1) % 10 == 0 or i == n_traj - 1:
            elapsed = time.time() - t0
            print(f"  {i+1:4d}/{n_traj}  ({elapsed:.1f}s)  last: {traj_type}")

    # ── manifest ──────────────────────────────────────────────────────────────
    if manifest_rows and ds_cfg.get("write_manifest", True):
        mdf = pd.DataFrame(manifest_rows)
        mdf.to_csv(os.path.join(out_dir, "manifest.csv"), index=False)

        # Dataset info JSON
        type_counts = mdf["type"].value_counts().to_dict()
        info = {
            "name":          ds_cfg.get("name"),
            "n_generated":   len(manifest_rows),
            "seed":          seed,
            "type_counts":   type_counts,
            "total_samples": int(mdf["n_samples"].sum()),
            "total_crossings": int(mdf["n_crossings"].sum()),
            "config_used":   cfg,
            "max_heading_vs_course_deg": round(worst_heading_err, 6),
        }
        with open(os.path.join(out_dir, "dataset_info.json"), "w") as f:
            json.dump(info, f, indent=2)

        print(f"\n✓ Generated {len(manifest_rows)} trajectories → {out_dir}")
        print(f"  Total samples  : {info['total_samples']:,}")
        print(f"  Total crossings: {info['total_crossings']:,}")
        print(f"  Type breakdown : {type_counts}")
        print(f"  Worst heading-vs-course disagreement: {worst_heading_err:.4f} deg")

    return out_dir, manifest_rows


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch trajectory generator")
    parser.add_argument("--config", default="config/default_config.json")
    parser.add_argument("--n",    type=int,   default=None)
    parser.add_argument("--out",  type=str,   default=None)
    parser.add_argument("--seed", type=int,   default=None)
    args = parser.parse_args()

    generate(args.config, n_override=args.n,
             out_override=args.out, seed_override=args.seed)
