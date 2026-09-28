# Trajectory Maker v3 — ML Dataset Generator

## Project Structure

```
trajectory_maker_v3/
├── generate.py          ← Batch dataset generation (CLI)
├── view.py              ← Dataset viewer (GUI)
├── main.py              ← Interactive single-trajectory designer (GUI)
│
├── config/
│   └── default_config.json   ← All parameters + randomisation ranges
│
├── trajectories/        ← 10 trajectory types
│   ├── base.py
│   ├── lawnmower.py     Hermite-spline turns, exact crossing angles
│   ├── zigzag.py        V-turns, compact, exact crossing angles
│   ├── parallel.py      Parallel passes along cable axis
│   ├── sinusoidal.py    Sine wave crossing repeatedly
│   ├── spiral.py        Inward/outward spiral
│   ├── starburst.py     Radial passes from centre
│   ├── random_walk.py   Cable-attracted random walk
│   ├── creeping_line.py Naval creeping line search
│   ├── expanding_square.py  Williamson expanding square
│   ├── random_crossings.py  Random angles + positions (best for ML variety)
│   └── waypoint.py      Sparse drone-command waypoints
│
├── generator/
│   ├── batch_generate.py   Core generation loop
│   ├── sampler.py          Parameter randomisation from config
│   └── label_events.py     Crossing detection + feature extraction
│
├── viewer/
│   └── viewer_app.py    Dataset browser (tkinter + matplotlib)
│
└── output/              Generated datasets land here
```

## Quick Start

### 1. Generate a dataset
```bash
# Default: 50 trajectories using settings in config/default_config.json
python generate.py

# Override count and output
python generate.py --n 500 --out output --seed 123

# Use a custom config
python generate.py --config config/my_config.json
```

### 2. View the dataset
```bash
python view.py --dataset output/cable_survey_dataset
# or just:
python view.py      # then click "LOAD DATASET"
```

### 3. Interactive designer
```bash
python main.py      # live preview of single trajectory
```

## Output Format (per trajectory)

Each `output/<dataset>/traj_NNNN/` contains:

| File | Description |
|------|-------------|
| `trajectory.csv` | lat, lon, heading, dist_cable_m, along_frac, heading_sin/cos, is_crossing |
| `features.npy` | (N, 5) float32 — [dist_cable_m, along_frac, heading_deg, sin, cos] |
| `labels.npy` | (N,) int8 — 1 at ±3 samples around each crossing, else 0 |
| `positions.npy` | (N, 2) float64 — raw lat/lon |
| `metadata.json` | params used, crossing events with incidence angle, along_frac, etc. |

Dataset-level:
| File | Description |
|------|-------------|
| `manifest.csv` | One row per trajectory: id, type, n_samples, n_crossings, ... |
| `dataset_info.json` | Full config used, type counts, total stats |

## Loading for ML (PyTorch example)

```python
import numpy as np, pandas as pd, json, os

dataset_dir = "output/cable_survey_dataset"
manifest    = pd.read_csv(f"{dataset_dir}/manifest.csv")

all_features, all_labels = [], []
for _, row in manifest.iterrows():
    d = f"{dataset_dir}/{row['id']}"
    X = np.load(f"{d}/features.npy")   # (N, 5) float32
    y = np.load(f"{d}/labels.npy")     # (N,)   int8
    all_features.append(X)
    all_labels.append(y)

# Stack into one big array
X = np.concatenate(all_features)  # (total_N, 5)
y = np.concatenate(all_labels)    # (total_N,)
print(f"Dataset: {len(X):,} samples, {y.mean()*100:.2f}% positives")
```

## Config Reference

The `config/default_config.json` file controls everything:

```json
{
  "dataset": {
    "n_trajectories": 50,
    "trajectory_types": ["Lawnmower", "Random Crossings", ...],
    "trajectory_weights": null    // null = uniform, or list of weights
  },
  "global": {
    "sampling_freq": { "fixed": 50.0 },
    "drone_speed":   { "range": [1.0, 3.0] }   // randomised per trajectory
  },
  "per_type": {
    "Lawnmower": {
      "n_crossings": { "range": [3, 12], "dtype": "int" },
      "angle_deg":   { "range": [20, 160] },
      ...
    }
  }
}
```

Parameter spec options:
- `{"fixed": value}` — always this value
- `{"range": [min, max]}` — uniform random, add `"dtype": "int"` to round
- `{"random_int": [min, max]}` — random integer
- `{"choice": [a, b, c]}` — random pick from list

## Heading Convention

All headings: **navigation convention** — 0° = North, 90° = East, clockwise, range [0, 360).
