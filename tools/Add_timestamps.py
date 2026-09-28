#!/usr/bin/env python3
"""
Add a Unix timestamp column (unit: 0.1 ns) to trajectory CSV files.

Usage:
    python add_timestamps.py --freq 10 file1.csv file2.csv ...
    python add_timestamps.py --freq 10 --start 13693304080000000000 file.csv
    python add_timestamps.py --freq 10 --overwrite file.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# 1 second expressed in units of 0.1 ns  (1 s = 10^10 * 0.1 ns)
_UNITS_PER_SECOND = 1_000_000


def add_timestamps(
    csv_path: Path,
    freq_hz: float,
    start: int,
    overwrite: bool,
) -> None:
    df = pd.read_csv(csv_path)

    if "timestamp" in df.columns:
        if not overwrite:
            print(f"  {csv_path.name}: skipped (timestamp column already exists, use --overwrite)")
            return
        df = df.drop(columns=["timestamp"])

    n = len(df)
    step = int(round(_UNITS_PER_SECOND / freq_hz))   # step in 0.1 ns units
    timestamps = [start + i * step for i in range(n)]
    df.insert(0, "timestamp", timestamps)

    df.to_csv(csv_path, index=False)
    print(f"  {csv_path.name}: {n} rows, {timestamps[0]} → {timestamps[-1]}")


def main():
    parser = argparse.ArgumentParser(
        description="Add a Unix timestamp column (unit: 0.1 ns) to trajectory CSV files."
    )
    parser.add_argument(
        "--freq", "-f",
        type=float,
        required=True,
        help="Sampling frequency in Hz (e.g. 10 → step of 10_000 units)",
    )
    parser.add_argument(
        "--start", "-s",
        type=int,
        default=0,
        help="Start timestamp in 0.1 ns units (default: 0)",
    )
    parser.add_argument(
        "--overwrite", "-o",
        action="store_true",
        help="Overwrite the timestamp column if it already exists",
    )
    parser.add_argument(
        "files",
        nargs="+",
        type=Path,
        help="One or more CSV files to process (modified in-place)",
    )
    args = parser.parse_args()

    if args.freq <= 0:
        print("Error: --freq must be a positive number.", file=sys.stderr)
        sys.exit(1)

    step = int(round(_UNITS_PER_SECOND / args.freq))
    print(f"Sampling frequency : {args.freq} Hz  ({step} × 0.1 ns / row)")
    print(f"Start timestamp    : {args.start}")
    print(f"Overwrite          : {args.overwrite}\n")

    for path in args.files:
        if not path.exists():
            print(f"  WARNING: {path} not found, skipping.")
            continue
        add_timestamps(path, args.freq, args.start, args.overwrite)

    print("\nDone.")


if __name__ == "__main__":
    main()