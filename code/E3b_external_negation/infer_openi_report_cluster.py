#!/usr/bin/env python3
"""Report-cluster inference for E3b OpenI cached analyses."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

PROFILES = ("gpt2", "gemma-2-2b", "qwen-2.5-3b")
STANDARD_REP = "sae_standard_code"
VREG_REP = "sae_vreg_code"
PRIMARY_SCALING = "standard"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def lower_tail_mean(values: list[float], frac: float = 0.20) -> float:
    arr = np.sort(np.asarray(values, dtype=np.float64))
    k = max(1, int(np.ceil(frac * len(arr))))
    return float(np.mean(arr[:k]))


def ci(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "bootstrap_mean": float(np.mean(arr)),
        "ci_low": float(np.quantile(arr, 0.025)),
        "ci_high": float(np.quantile(arr, 0.975)),
        "positive_fraction": float(np.mean(arr > 0)),
    }


def rep_block(profile_block: dict[str, Any], rep: str) -> dict[str, Any]:
    return profile_block["representations"][rep]["scaling_modes"][PRIMARY_SCALING]["individual_example_probe"]


def paired_by_report(block: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in block["test"]["paired_margins"]:
        report_id = record.get("report_id")
        if report_id is None:
            raise ValueError("paired_margins missing report_id; rerun cached analysis with updated exporter")
        out[report_id].append(record)
    return dict(out)


def predictions_by_report(block: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in block["test"]["predictions"]:
        report_id = record.get("report_id")
        if report_id is None:
            raise ValueError("predictions missing report_id; rerun cached analysis with updated exporter")
        out[report_id].append(record)
    return dict(out)


def metrics_from_predictions(records: list[dict[str, Any]]) -> dict[str, float]:
    y_true = np.asarray([record["y_true"] for record in records], dtype=np.int64)
    y_pred = np.asarray([record["y_pred"] for record in records], dtype=np.int64)
    prob = np.asarray(
        [record.get("prob_positive", record["prob_negated"]) for record in records],
        dtype=np.float64,
    )
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "auroc": float(roc_auc_score(y_true, prob)) if len(set(y_true.tolist())) == 2 else float("nan"),
    }


def paired_metrics(report_pairs: dict[str, list[dict[str, Any]]], report_ids: list[str]) -> dict[str, float]:
    records = [record for report_id in report_ids for record in report_pairs[report_id]]
    return {
        "probability_l20": lower_tail_mean([record["paired_margin"] for record in records]),
        "logit_l20": lower_tail_mean([record["logit_paired_margin"] for record in records]),
        "geometric_l20": lower_tail_mean([record["geometric_paired_margin"] for record in records]),
        "pairwise_accuracy": float(np.mean([record["paired_margin"] > 0 for record in records])),
    }


def conventional_metrics(report_predictions: dict[str, list[dict[str, Any]]], report_ids: list[str]) -> dict[str, float]:
    return metrics_from_predictions([record for report_id in report_ids for record in report_predictions[report_id]])


def analyze_profile(profile_block: dict[str, Any], n_boot: int, seed: int) -> dict[str, Any]:
    std = rep_block(profile_block, STANDARD_REP)
    vreg = rep_block(profile_block, VREG_REP)
    std_pairs = paired_by_report(std)
    vreg_pairs = paired_by_report(vreg)
    std_preds = predictions_by_report(std)
    vreg_preds = predictions_by_report(vreg)
    report_ids = sorted(set(std_pairs) & set(vreg_pairs))
    if set(std_pairs) != set(vreg_pairs):
        raise ValueError("Standard/V-reg paired report IDs differ")

    point_std = {**paired_metrics(std_pairs, report_ids), **conventional_metrics(std_preds, report_ids)}
    point_vreg = {**paired_metrics(vreg_pairs, report_ids), **conventional_metrics(vreg_preds, report_ids)}
    point_delta = {key: point_vreg[key] - point_std[key] for key in point_std}

    rng = random.Random(seed)
    draws: dict[str, list[float]] = defaultdict(list)
    for _ in range(n_boot):
        sampled = [rng.choice(report_ids) for _ in range(len(report_ids))]
        b_std = {**paired_metrics(std_pairs, sampled), **conventional_metrics(std_preds, sampled)}
        b_vreg = {**paired_metrics(vreg_pairs, sampled), **conventional_metrics(vreg_preds, sampled)}
        for key in b_std:
            if np.isfinite(b_std[key]) and np.isfinite(b_vreg[key]):
                draws[key].append(b_vreg[key] - b_std[key])

    return {
        "n_test_reports": len(report_ids),
        "n_test_pairs": sum(len(std_pairs[report_id]) for report_id in report_ids),
        "point": {
            "standard": point_std,
            "vreg": point_vreg,
            "delta": point_delta,
        },
        "bootstrap": {key: ci(values) for key, values in draws.items()},
    }


def write_md(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# E3b OpenI Report-Cluster Inference",
        "",
        f"Dataset: `{payload['dataset_kind']}`",
        f"Bootstrap reps: {payload['n_boot']}",
        "",
        "| Model | Std L20 | V-reg L20 | Delta L20 | 95% CI | Pos. frac |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for profile in PROFILES:
        block = payload["profiles"][profile]
        boot = block["bootstrap"]["probability_l20"]
        lines.append(
            f"| {profile} | {block['point']['standard']['probability_l20']:.4f} | "
            f"{block['point']['vreg']['probability_l20']:.4f} | "
            f"{block['point']['delta']['probability_l20']:.4f} | "
            f"[{boot['ci_low']:.4f}, {boot['ci_high']:.4f}] | {boot['positive_fraction']:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run E3b report-cluster inference")
    parser.add_argument("--analysis-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    analysis = read_json(args.analysis_json)
    if analysis.get("status") != "complete":
        raise ValueError("Expected complete all-profile cached analysis")
    payload = {
        "experiment": "E3b_external_negation",
        "analysis": "openi_report_cluster_inference",
        "status": "complete",
        "source_analysis_json": str(args.analysis_json),
        "dataset_kind": analysis.get("split_variant") or "openi_external",
        "primary_cluster_unit": "report_id",
        "n_boot": args.n_boot,
        "seed": args.seed,
        "profiles": {},
    }
    for profile in PROFILES:
        payload["profiles"][profile] = analyze_profile(analysis["profiles"][profile], args.n_boot, args.seed)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    write_md(args.output_md, payload)
    print(f"Saved JSON -> {args.output_json}")
    print(f"Saved Markdown -> {args.output_md}")


if __name__ == "__main__":
    main()
