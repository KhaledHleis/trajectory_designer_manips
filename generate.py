#!/usr/bin/env python3
"""
Quick entry point: python generate.py [--config ...] [--n N] [--out DIR] [--seed S]
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generator.batch_generate import generate
import argparse

parser = argparse.ArgumentParser(description="Batch trajectory dataset generator")
parser.add_argument("--config", default="config/default_config.json", help="Path to JSON config")
parser.add_argument("--n",    type=int, default=None,  help="Override n_trajectories")
parser.add_argument("--out",  type=str, default=None,  help="Override output directory")
parser.add_argument("--seed", type=int, default=None,  help="Override random seed")
args = parser.parse_args()

generate(args.config, n_override=args.n, out_override=args.out, seed_override=args.seed)
