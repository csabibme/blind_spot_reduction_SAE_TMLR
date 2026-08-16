#!/usr/bin/env python3
"""Family-size ablation for actual E1/E1R Standard–V-reg effects."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REVISION_ROOT = Path(__file__).resolve().parents[1]
E5_ROOT = Path(__file__).resolve().parent
for p in (REVISION_ROOT, E5_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

METRICS = {
    "s": "per_pair_s",
    "abs_dz": "per_pair_abs_dz",
    "decode_resp": "per_pair_decode_resp",
}


def lower_fraction_mean(values: np.ndarray, fraction: float = 0.20) -> float:
    arr = np.asarray(values, dtype=np.float64)
    k = max(1, math.ceil(fraction * len(arr)))
    return float(np.sort(arr)[:k].mean())


def delta_l20(std: np.ndarray, vreg: np.ndarray, idx: np.ndarray) -> float:
    return lower_fraction_mean(vreg[idx]) - lower_fraction_mean(std[idx])


def extract_profiles(paths: list[Path]) -> dict:
    profiles = {}
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        for profile, block in data["profiles"].items():
            if profile in profiles:
                raise ValueError(f"Duplicate profile in inputs: {profile}")
            profiles[profile] = block
    return profiles


def metric_arrays(family_block: dict, metric: str) -> tuple[np.ndarray, np.ndarray]:
    field = METRICS[metric]
    std = np.asarray(family_block["standard"][field], dtype=np.float64)
    vreg = np.asarray(family_block["vreg"][field], dtype=np.float64)
    if len(std) != len(vreg):
        raise ValueError(f"Length mismatch for {metric}")
    return std, vreg


def ablate_family(family_block: dict, sizes: list[int], n_repeats: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n = family_block["n_pairs"]
    out = {}
    for metric in METRICS:
        std, vreg = metric_arrays(family_block, metric)
        out[metric] = {}
        for size in sizes:
            if size > n:
                continue
            vals = []
            for _ in range(n_repeats):
                idx = rng.choice(n, size=size, replace=False)
                vals.append(delta_l20(std, vreg, idx))
            vals_arr = np.asarray(vals, dtype=np.float64)
            out[metric][str(size)] = {
                "n_repeats": n_repeats,
                "delta_L20_mean": float(vals_arr.mean()),
                "delta_L20_std": float(vals_arr.std()),
                "delta_L20_q05": float(np.quantile(vals_arr, 0.05)),
                "delta_L20_q50": float(np.quantile(vals_arr, 0.50)),
                "delta_L20_q95": float(np.quantile(vals_arr, 0.95)),
                "positive_fraction": float(np.mean(vals_arr > 0)),
            }
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="E5 E1-effect family-size ablation")
    p.add_argument("--input-json", type=Path, nargs="+", required=True)
    p.add_argument("--subset-sizes", type=int, nargs="*", default=[10, 20, 30, 50])
    p.add_argument("--n-repeats", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--output-json",
        type=Path,
        default=E5_ROOT / "results" / "e1_effect_family_size_ablation.json",
    )
    p.add_argument(
        "--output-md",
        type=Path,
        default=E5_ROOT / "results" / "e1_effect_family_size_ablation.md",
    )
    args = p.parse_args()

    profiles = extract_profiles(args.input_json)
    result = {
        "input_json": [str(p) for p in args.input_json],
        "metrics": METRICS,
        "subset_sizes": args.subset_sizes,
        "n_repeats": args.n_repeats,
        "seed": args.seed,
        "profiles": {},
    }
    md = [
        "# E5 E1-Effect Family-Size Ablation",
        "",
        "| Profile | Family | Metric | size | mean ΔL20 | q05 | q95 | positive frac |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for profile, block in sorted(profiles.items()):
        result["profiles"][profile] = {"families": {}}
        for family, family_block in sorted(block["families"].items()):
            fam_res = ablate_family(family_block, args.subset_sizes, args.n_repeats, args.seed)
            result["profiles"][profile]["families"][family] = fam_res
            for metric, metric_block in fam_res.items():
                for size, vals in metric_block.items():
                    md.append(
                        f"| {profile} | {family} | {metric} | {size} | "
                        f"{vals['delta_L20_mean']:+.6f} | {vals['delta_L20_q05']:+.6f} | "
                        f"{vals['delta_L20_q95']:+.6f} | {vals['positive_fraction']:.3f} |"
                    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    args.output_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Saved -> {args.output_json}")
    print(f"Saved -> {args.output_md}")


if __name__ == "__main__":
    main()
