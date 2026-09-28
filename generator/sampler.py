"""
generator/sampler.py
--------------------
Samples random parameters from a config dict for a single trajectory.
"""
import numpy as np


def sample_params(type_cfg: dict, rng: np.random.Generator) -> dict:
    """
    For each key in type_cfg, sample a value:
      {"fixed": v}           → v
      {"range": [a, b]}      → uniform(a, b), int if dtype=="int"
      {"random_int": [a, b]} → randint(a, b+1)
      {"choice": [a, b, ...]}→ random choice
    """
    out = {}
    for key, spec in type_cfg.items():
        if "fixed" in spec:
            out[key] = spec["fixed"]
        elif "range" in spec:
            a, b = spec["range"]
            val = rng.uniform(a, b)
            if spec.get("dtype") == "int":
                val = int(round(val))
            out[key] = val
        elif "random_int" in spec:
            a, b = spec["random_int"]
            out[key] = int(rng.integers(a, b + 1))
        elif "choice" in spec:
            out[key] = rng.choice(spec["choice"])
    return out


def sample_global(global_cfg: dict, rng: np.random.Generator) -> dict:
    """Sample global params (sampling_freq, drone_speed)."""
    return sample_params(global_cfg, rng)
