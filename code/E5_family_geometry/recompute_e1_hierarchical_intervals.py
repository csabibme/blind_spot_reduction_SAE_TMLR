#!/usr/bin/env python3
"""Recompute E1/E1R uncertainty with E5 template-cluster hierarchy."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

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
    "g": "per_pair_g",
}


def json_safe(obj):
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    return obj


def lower_fraction_mean(values: np.ndarray, fraction: float = 0.20) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return float("nan")
    k = max(1, math.ceil(fraction * len(arr)))
    return float(np.sort(arr)[:k].mean())


def delta_l20(std_values: np.ndarray, vreg_values: np.ndarray, idx: np.ndarray) -> float:
    return lower_fraction_mean(vreg_values[idx]) - lower_fraction_mean(std_values[idx])


def load_clusters(path: Path) -> tuple[dict[str, dict[str, list[int]]], dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for family, block in data["families"].items():
        mapping: dict[str, list[int]] = {}
        for rec in block["pairs"]:
            mapping.setdefault(rec["template_id"], []).append(int(rec["pair_index"]))
        out[family] = mapping
    return out, {"pairs_file": data.get("pairs_file"), "pairs_sha256": data.get("pairs_sha256")}


def extract_profile_blocks(paths: list[Path]) -> dict[str, dict]:
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
        raise ValueError(f"Metric length mismatch for {metric}: {len(std)} != {len(vreg)}")
    return std, vreg


def selected_cluster_positions(
    family_block: dict,
    family_clusters: dict[str, list[int]],
) -> dict[str, list[int]]:
    selected = family_block.get("selected_pair_indices")
    if selected is None:
        selected = list(range(family_block["n_pairs"]))
    selected = [int(x) for x in selected]
    original_to_position = {pair_idx: pos for pos, pair_idx in enumerate(selected)}
    out: dict[str, list[int]] = {}
    for template_id, original_indices in family_clusters.items():
        positions = [
            original_to_position[i]
            for i in original_indices
            if i in original_to_position
        ]
        if positions:
            out[template_id] = positions

    covered = [pos for positions in out.values() for pos in positions]
    expected = list(range(len(selected)))
    if sorted(covered) != expected:
        raise ValueError(
            f"Cluster coverage mismatch: covered={len(covered)}, expected={len(expected)}"
        )
    if len(covered) != len(set(covered)):
        raise ValueError("A pair was assigned to multiple template clusters")
    return out


def profile_family_points(profile_block: dict, metric: str) -> dict[str, float]:
    out = {}
    for family, family_block in profile_block["families"].items():
        std, vreg = metric_arrays(family_block, metric)
        out[family] = delta_l20(std, vreg, np.arange(len(std)))
    return out


def resample_family_metric(
    family_block: dict,
    family_clusters: dict[str, list[int]],
    metric: str,
    rng: np.random.Generator,
) -> float:
    std, vreg = metric_arrays(family_block, metric)
    mapped = selected_cluster_positions(family_block, family_clusters)
    template_ids = list(mapped)
    chosen_templates = rng.choice(template_ids, size=len(template_ids), replace=True)
    idx = np.concatenate([np.asarray(mapped[t], dtype=np.int64) for t in chosen_templates])
    return delta_l20(std, vreg, idx)


def fixed_family_cluster_bootstrap(
    profile_block: dict,
    clusters: dict[str, dict[str, list[int]]],
    metric: str,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    families = sorted(profile_block["families"])
    points = profile_family_points(profile_block, metric)
    boots = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        stats = [
            resample_family_metric(profile_block["families"][family], clusters[family], metric, rng)
            for family in families
        ]
        boots[b] = float(np.mean(stats))
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {
        "estimand": "fixed_16_family_template_cluster",
        "point": float(np.mean(list(points.values()))),
        "lo": float(lo),
        "hi": float(hi),
        "n_boot": n_boot,
        "n_families": len(families),
        "family_points": {k: float(v) for k, v in sorted(points.items())},
    }


def family_resampled_cluster_bootstrap(
    profile_block: dict,
    clusters: dict[str, dict[str, list[int]]],
    metric: str,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    families = sorted(profile_block["families"])
    points = profile_family_points(profile_block, metric)
    boots = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        sampled_families = rng.choice(families, size=len(families), replace=True)
        stats = [
            resample_family_metric(profile_block["families"][family], clusters[family], metric, rng)
            for family in sampled_families
        ]
        boots[b] = float(np.mean(stats))
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {
        "estimand": "family_population_template_cluster",
        "point": float(np.mean(list(points.values()))),
        "lo": float(lo),
        "hi": float(hi),
        "n_boot": n_boot,
        "n_families": len(families),
        "family_points": {k: float(v) for k, v in sorted(points.items())},
    }


def leave_one_template_out(
    profile_block: dict,
    clusters: dict[str, dict[str, list[int]]],
    metric: str,
) -> dict:
    out = {}
    for family, family_block in profile_block["families"].items():
        std, vreg = metric_arrays(family_block, metric)
        mapped = selected_cluster_positions(family_block, clusters[family])
        fam_out = {}
        for template_id, omit_idx in mapped.items():
            keep = np.setdiff1d(np.arange(len(std)), np.asarray(omit_idx, dtype=np.int64))
            fam_out[template_id] = {
                "n_omitted_pairs": int(len(omit_idx)),
                "delta_L20": float(delta_l20(std, vreg, keep)) if len(keep) else None,
            }
        out[family] = fam_out
    return out


def leave_one_family_out(profile_block: dict, metric: str) -> dict:
    points = profile_family_points(profile_block, metric)
    return {
        family: {
            "macro_delta_L20": float(np.mean([v for f, v in points.items() if f != family])),
            "omitted_family_delta_L20": float(points[family]),
        }
        for family in sorted(points)
    }


def validate_profile(profile: str, profile_block: dict, clusters: dict) -> None:
    if set(profile_block["families"]) != set(clusters):
        raise ValueError(
            f"Family set mismatch for {profile}: "
            f"profile={sorted(profile_block['families'])} clusters={sorted(clusters)}"
        )
    for family, block in profile_block["families"].items():
        for metric in METRICS:
            metric_arrays(block, metric)
        selected_cluster_positions(block, clusters[family])


def main() -> None:
    p = argparse.ArgumentParser(description="E5 hierarchical bootstrap for E1/E1R")
    p.add_argument("--template-clusters", type=Path, default=E5_ROOT / "results" / "template_clusters.json")
    p.add_argument("--input-json", type=Path, nargs="+", required=True)
    p.add_argument("--n-boot", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-json", type=Path, default=E5_ROOT / "results" / "hierarchical_intervals.json")
    p.add_argument("--output-md", type=Path, default=E5_ROOT / "results" / "hierarchical_intervals.md")
    args = p.parse_args()

    clusters, cluster_meta = load_clusters(args.template_clusters)
    profiles = extract_profile_blocks(args.input_json)
    result = {
        "template_clusters": str(args.template_clusters),
        "cluster_meta": cluster_meta,
        "input_json": [str(p) for p in args.input_json],
        "n_boot": args.n_boot,
        "seed": args.seed,
        "metrics": METRICS,
        "profiles": {},
    }
    md = [
        "# E5 Hierarchical Intervals",
        "",
        "Primary estimand: fixed 16-family template-cluster bootstrap.",
        "Secondary estimand: family-resampled + template-cluster bootstrap.",
        "",
        "| Profile | Metric | fixed point | fixed lo | fixed hi | family-resampled lo | family-resampled hi |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]

    for profile, block in sorted(profiles.items()):
        validate_profile(profile, block, clusters)
        result["profiles"][profile] = {}
        for metric in METRICS:
            fixed = fixed_family_cluster_bootstrap(block, clusters, metric, args.n_boot, args.seed)
            family_resampled = family_resampled_cluster_bootstrap(
                block, clusters, metric, args.n_boot, args.seed + 1000,
            )
            result["profiles"][profile][metric] = {
                "primary_fixed_family_cluster_bootstrap": fixed,
                "secondary_family_resampled_cluster_bootstrap": family_resampled,
                "leave_one_template_out": leave_one_template_out(block, clusters, metric),
                "leave_one_family_out": leave_one_family_out(block, metric),
            }
            md.append(
                f"| {profile} | {metric} | {fixed['point']:+.6f} | {fixed['lo']:+.6f} | "
                f"{fixed['hi']:+.6f} | {family_resampled['lo']:+.6f} | "
                f"{family_resampled['hi']:+.6f} |"
            )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(json_safe(result), indent=2, allow_nan=False), encoding="utf-8")
    args.output_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Saved -> {args.output_json}")
    print(f"Saved -> {args.output_md}")


if __name__ == "__main__":
    main()
