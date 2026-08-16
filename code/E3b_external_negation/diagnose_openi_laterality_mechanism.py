#!/usr/bin/env python3
"""Post-hoc mechanism diagnostics for E3b OpenI laterality probes.

This script intentionally does not change the frozen primary endpoint. It diagnoses whether
V-reg lifts pairs that were weak under the Standard SAE-code readout, and whether code-space
left/right distances change independently of the logistic probe.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REVISION_ROOT = Path(__file__).resolve().parents[1]
E3_ROOT = REVISION_ROOT / "E3_heldout_probes"
if str(E3_ROOT) not in sys.path:
    sys.path.insert(0, str(E3_ROOT))

from run_e3_negation_probes import (  # noqa: E402
    deduplicate_examples,
    feature_cache_path,
    read_json,
    sha256_file,
)

PROFILES = ("gpt2", "gemma-2-2b", "qwen-2.5-3b")
PRIMARY_SCALING = "standard"
STANDARD_REP = "sae_standard_code"
VREG_REP = "sae_vreg_code"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def ci(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "bootstrap_mean": float(np.mean(arr)),
        "ci_low": float(np.quantile(arr, 0.025)),
        "ci_high": float(np.quantile(arr, 0.975)),
        "positive_fraction": float(np.mean(arr > 0)),
    }


def lower_tail_mean(values: list[float], frac: float = 0.20) -> float:
    arr = np.sort(np.asarray(values, dtype=np.float64))
    k = max(1, int(np.ceil(frac * len(arr))))
    return float(np.mean(arr[:k]))


def ranks(values: list[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    order = np.argsort(arr)
    out = np.empty(len(arr), dtype=np.float64)
    i = 0
    while i < len(arr):
        j = i
        while j + 1 < len(arr) and arr[order[j + 1]] == arr[order[i]]:
            j += 1
        rank = (i + j) / 2.0 + 1.0
        out[order[i : j + 1]] = rank
        i = j + 1
    return out


def spearman(x: list[float], y: list[float]) -> float:
    if len(x) < 3:
        return float("nan")
    rx = ranks(x)
    ry = ranks(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def slope(x: list[float], y: list[float]) -> float:
    if len(x) < 2 or np.std(x) == 0:
        return float("nan")
    return float(np.polyfit(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64), deg=1)[0])


def rep_probe_block(profile_block: dict[str, Any], rep: str) -> dict[str, Any]:
    return profile_block["representations"][rep]["scaling_modes"][PRIMARY_SCALING]["individual_example_probe"]


def paired_records(profile_block: dict[str, Any]) -> list[dict[str, Any]]:
    std = rep_probe_block(profile_block, STANDARD_REP)["test"]["paired_margins"]
    vreg = rep_probe_block(profile_block, VREG_REP)["test"]["paired_margins"]
    std_by_id = {record["global_pair_id"]: record for record in std}
    vreg_by_id = {record["global_pair_id"]: record for record in vreg}
    if set(std_by_id) != set(vreg_by_id):
        raise ValueError("Standard/V-reg pair IDs differ")
    records = []
    for pair_id in sorted(std_by_id):
        s = std_by_id[pair_id]
        v = vreg_by_id[pair_id]
        if s.get("report_id") != v.get("report_id"):
            raise ValueError(f"Report ID mismatch for {pair_id}")
        records.append(
            {
                "global_pair_id": pair_id,
                "report_id": s["report_id"],
                "template_id": s["template_id"],
                "standard": {
                    "probability_margin": s["paired_margin"],
                    "logit_margin": s["logit_paired_margin"],
                    "geometric_margin": s["geometric_paired_margin"],
                },
                "vreg": {
                    "probability_margin": v["paired_margin"],
                    "logit_margin": v["logit_paired_margin"],
                    "geometric_margin": v["geometric_paired_margin"],
                },
                "delta": {
                    "probability_margin": v["paired_margin"] - s["paired_margin"],
                    "logit_margin": v["logit_paired_margin"] - s["logit_paired_margin"],
                    "geometric_margin": v["geometric_paired_margin"] - s["geometric_paired_margin"],
                },
            }
        )
    return records


def standard_tail(records: list[dict[str, Any]], frac: float = 0.20) -> dict[str, Any]:
    ordered = sorted(records, key=lambda row: row["standard"]["probability_margin"])
    k = max(1, int(np.ceil(frac * len(ordered))))
    tail = ordered[:k]
    vreg_tail_ids = {
        row["global_pair_id"]
        for row in sorted(records, key=lambda row: row["vreg"]["probability_margin"])[:k]
    }
    std_tail_ids = {row["global_pair_id"] for row in tail}
    return {
        "tail_size": k,
        "standard_tail_pair_ids": [row["global_pair_id"] for row in tail],
        "vreg_own_tail_pair_ids": sorted(vreg_tail_ids),
        "tail_overlap_count": len(std_tail_ids & vreg_tail_ids),
        "tail_jaccard": len(std_tail_ids & vreg_tail_ids) / len(std_tail_ids | vreg_tail_ids),
        "standard_tail_l20_standard": float(np.mean([row["standard"]["probability_margin"] for row in tail])),
        "standard_tail_l20_vreg": float(np.mean([row["vreg"]["probability_margin"] for row in tail])),
        "standard_tail_delta_probability": float(np.mean([row["delta"]["probability_margin"] for row in tail])),
        "standard_tail_delta_logit": float(np.mean([row["delta"]["logit_margin"] for row in tail])),
        "standard_tail_delta_geometric": float(np.mean([row["delta"]["geometric_margin"] for row in tail])),
    }


def delta_summary(records: list[dict[str, Any]]) -> dict[str, float]:
    deltas = [row["delta"]["probability_margin"] for row in records]
    return {
        "mean_delta_probability": float(np.mean(deltas)),
        "median_delta_probability": float(np.median(deltas)),
        "fraction_positive_delta_probability": float(np.mean(np.asarray(deltas) > 0)),
    }


def quintile_deltas(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(records, key=lambda row: row["standard"]["probability_margin"])
    chunks = np.array_split(np.asarray(ordered, dtype=object), 5)
    out = []
    for index, chunk in enumerate(chunks, start=1):
        rows = list(chunk)
        out.append(
            {
                "quintile": index,
                "n_pairs": len(rows),
                "standard_probability_margin_min": float(min(row["standard"]["probability_margin"] for row in rows)),
                "standard_probability_margin_max": float(max(row["standard"]["probability_margin"] for row in rows)),
                "mean_delta_probability": float(np.mean([row["delta"]["probability_margin"] for row in rows])),
                "mean_delta_logit": float(np.mean([row["delta"]["logit_margin"] for row in rows])),
                "mean_delta_geometric": float(np.mean([row["delta"]["geometric_margin"] for row in rows])),
            }
        )
    return out


def margin_delta_relationship(records: list[dict[str, Any]]) -> dict[str, Any]:
    x = [row["standard"]["probability_margin"] for row in records]
    y = [row["delta"]["probability_margin"] for row in records]
    return {
        "spearman_standard_margin_vs_delta_probability": spearman(x, y),
        "linear_slope_delta_probability_on_standard_margin": slope(x, y),
        "quintiles_by_standard_probability_margin": quintile_deltas(records),
    }


def by_report(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        out[row["report_id"]].append(row)
    return dict(out)


def sample_reports(report_records: dict[str, list[dict[str, Any]]], rng: random.Random) -> list[dict[str, Any]]:
    report_ids = sorted(report_records)
    sampled = [rng.choice(report_ids) for _ in report_ids]
    return [row for report_id in sampled for row in report_records[report_id]]


def bootstrap_margin_diagnostics(records: list[dict[str, Any]], n_boot: int, seed: int) -> dict[str, Any]:
    report_records = by_report(records)
    rng = random.Random(seed)
    draws: dict[str, list[float]] = defaultdict(list)
    for _ in range(n_boot):
        sampled = sample_reports(report_records, rng)
        tail = standard_tail(sampled)
        relationship = margin_delta_relationship(sampled)
        summary = delta_summary(sampled)
        draws["standard_tail_delta_probability"].append(tail["standard_tail_delta_probability"])
        draws["standard_tail_delta_logit"].append(tail["standard_tail_delta_logit"])
        draws["standard_tail_delta_geometric"].append(tail["standard_tail_delta_geometric"])
        draws["spearman_standard_margin_vs_delta_probability"].append(
            relationship["spearman_standard_margin_vs_delta_probability"]
        )
        draws["linear_slope_delta_probability_on_standard_margin"].append(
            relationship["linear_slope_delta_probability_on_standard_margin"]
        )
        draws["mean_delta_probability"].append(summary["mean_delta_probability"])
    return {key: ci([value for value in values if np.isfinite(value)]) for key, values in draws.items()}


def code_distance_records(
    split_payload: dict[str, Any],
    split_sha: str,
    feature_cache_dir: Path,
    profile: str,
) -> list[dict[str, Any]]:
    examples = deduplicate_examples(split_payload["examples"])
    texts = [example["text"] for example in examples]
    cache_path = feature_cache_path(feature_cache_dir, profile, split_sha, texts)
    if not cache_path.is_file():
        raise FileNotFoundError(f"Missing feature cache: {cache_path}")
    with np.load(cache_path, allow_pickle=True) as data:
        std = data[STANDARD_REP]
        vreg = data[VREG_REP]
    by_pair: dict[str, dict[str, Any]] = defaultdict(dict)
    for index, example in enumerate(examples):
        if example["split"] != "test":
            continue
        entry = by_pair[example["global_pair_id"]]
        entry["report_id"] = example["report_id"]
        entry["template_id"] = example["template_id"]
        entry[example["label"]] = index
    records = []
    for pair_id, entry in sorted(by_pair.items()):
        if "left" not in entry or "right" not in entry:
            continue
        left_idx = entry["left"]
        right_idx = entry["right"]
        std_distance = float(np.linalg.norm(std[right_idx] - std[left_idx]))
        vreg_distance = float(np.linalg.norm(vreg[right_idx] - vreg[left_idx]))
        records.append(
            {
                "global_pair_id": pair_id,
                "report_id": entry["report_id"],
                "template_id": entry["template_id"],
                "standard_code_distance": std_distance,
                "vreg_code_distance": vreg_distance,
                "delta_code_distance": vreg_distance - std_distance,
            }
        )
    return records


def code_distance_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    std = [row["standard_code_distance"] for row in records]
    vreg = [row["vreg_code_distance"] for row in records]
    delta = [row["delta_code_distance"] for row in records]
    return {
        "n_pairs": len(records),
        "standard_mean": float(np.mean(std)),
        "vreg_mean": float(np.mean(vreg)),
        "delta_mean": float(np.mean(delta)),
        "standard_median": float(np.median(std)),
        "vreg_median": float(np.median(vreg)),
        "delta_median": float(np.median(delta)),
        "standard_l20": lower_tail_mean(std),
        "vreg_l20": lower_tail_mean(vreg),
        "delta_l20_own_tail": lower_tail_mean(vreg) - lower_tail_mean(std),
        "fraction_positive_delta": float(np.mean(np.asarray(delta) > 0)),
    }


def bootstrap_code_distance(records: list[dict[str, Any]], n_boot: int, seed: int) -> dict[str, Any]:
    report_records = by_report(records)
    rng = random.Random(seed)
    draws: dict[str, list[float]] = defaultdict(list)
    for _ in range(n_boot):
        sampled = sample_reports(report_records, rng)
        summary = code_distance_summary(sampled)
        draws["delta_mean"].append(summary["delta_mean"])
        draws["delta_median"].append(summary["delta_median"])
        draws["delta_l20_own_tail"].append(summary["delta_l20_own_tail"])
    return {key: ci(values) for key, values in draws.items()}


def analyze_profile(
    analysis: dict[str, Any],
    split_payload: dict[str, Any],
    split_sha: str,
    feature_cache_dir: Path,
    profile: str,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    records = paired_records(analysis["profiles"][profile])
    distances = code_distance_records(split_payload, split_sha, feature_cache_dir, profile)
    return {
        "n_test_pairs": len(records),
        "n_test_reports": len({row["report_id"] for row in records}),
        "standard_anchored_tail": standard_tail(records),
        "all_pair_delta_summary": delta_summary(records),
        "margin_delta_relationship": margin_delta_relationship(records),
        "report_cluster_bootstrap": bootstrap_margin_diagnostics(records, n_boot, seed),
        "code_distance": {
            "summary": code_distance_summary(distances),
            "report_cluster_bootstrap": bootstrap_code_distance(distances, n_boot, seed),
        },
    }


def write_md(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# E3b Laterality Mechanism Diagnostics",
        "",
        f"Source analysis: `{payload['source_analysis_json']}`",
        f"Bootstrap reps: {payload['n_boot']}",
        "",
        "## Standard-Anchored Tail",
        "",
        "| Model | Std-tail Std L20 | Std-tail V-reg L20 | Delta prob | Delta logit | Delta geom | Tail overlap |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for profile in PROFILES:
        tail = payload["profiles"][profile]["standard_anchored_tail"]
        lines.append(
            f"| {profile} | {tail['standard_tail_l20_standard']:.4f} | "
            f"{tail['standard_tail_l20_vreg']:.4f} | "
            f"{tail['standard_tail_delta_probability']:+.4f} | "
            f"{tail['standard_tail_delta_logit']:+.4f} | "
            f"{tail['standard_tail_delta_geometric']:+.4f} | "
            f"{tail['tail_overlap_count']}/{tail['tail_size']} |"
        )
    lines.extend(
        [
            "",
            "## Margin-Delta Relationship",
            "",
            "| Model | Mean delta prob | Median delta prob | Fraction positive | Spearman(std margin, delta) | Slope |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for profile in PROFILES:
        block = payload["profiles"][profile]
        summary = block["all_pair_delta_summary"]
        rel = block["margin_delta_relationship"]
        lines.append(
            f"| {profile} | {summary['mean_delta_probability']:+.4f} | "
            f"{summary['median_delta_probability']:+.4f} | "
            f"{summary['fraction_positive_delta_probability']:.3f} | "
            f"{rel['spearman_standard_margin_vs_delta_probability']:+.3f} | "
            f"{rel['linear_slope_delta_probability_on_standard_margin']:+.3f} |"
        )
    lines.extend(
        [
            "",
            "## Code Distances",
            "",
            "| Model | Std mean | V-reg mean | Delta mean | Std L20 | V-reg L20 | Delta L20 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for profile in PROFILES:
        summary = payload["profiles"][profile]["code_distance"]["summary"]
        lines.append(
            f"| {profile} | {summary['standard_mean']:.4f} | {summary['vreg_mean']:.4f} | "
            f"{summary['delta_mean']:+.4f} | {summary['standard_l20']:.4f} | "
            f"{summary['vreg_l20']:.4f} | {summary['delta_l20_own_tail']:+.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose E3b laterality weak-direction behavior")
    parser.add_argument("--analysis-json", type=Path, required=True)
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--feature-cache-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    analysis = read_json(args.analysis_json)
    split_payload = read_json(args.split_json)
    split_sha = sha256_file(args.split_json)
    payload = {
        "experiment": "E3b_external_laterality",
        "analysis": "post_hoc_mechanistic_diagnostics",
        "status": "complete",
        "source_analysis_json": str(args.analysis_json),
        "source_split_json": str(args.split_json),
        "task_split_sha256": split_sha,
        "feature_cache_dir": str(args.feature_cache_dir),
        "estimand_note": (
            "Post-hoc diagnostics only: Standard-anchored weak-tail lifting, margin-delta "
            "relationship, and direct code distances. These do not replace the frozen primary "
            "own-tail L20 endpoint."
        ),
        "n_boot": args.n_boot,
        "seed": args.seed,
        "profiles": {},
    }
    for profile in PROFILES:
        payload["profiles"][profile] = analyze_profile(
            analysis,
            split_payload,
            split_sha,
            args.feature_cache_dir,
            profile,
            args.n_boot,
            args.seed,
        )
    write_json(args.output_json, payload)
    write_md(args.output_md, payload)
    print(f"Saved JSON -> {args.output_json}")
    print(f"Saved Markdown -> {args.output_md}")


if __name__ == "__main__":
    main()
