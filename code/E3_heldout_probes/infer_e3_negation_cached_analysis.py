#!/usr/bin/env python3
"""Infer E3 negation effects from cached-feature analysis outputs.

Offline only: no LM/SAE loading. The primary estimand is the paired Standard vs V-reg
difference in test-template lower-tail pair margin L20 for SAE codes under standard scaling.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
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
MARGIN_KEYS = {
    "probability_l20": "paired_margin",
    "logit_l20": "logit_paired_margin",
    "geometric_l20": "geometric_paired_margin",
}
TEST_MARGIN_KEYS = {
    "probability_l20": "pair_margin_l20",
    "logit_l20": "logit_pair_margin_l20",
    "geometric_l20": "geometric_pair_margin_l20",
}
SECONDARY_METRICS = ("balanced_accuracy", "auroc", "macro_f1", "pairwise_accuracy")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def lower_tail_mean(values: list[float] | np.ndarray, frac: float = 0.20) -> float:
    arr = np.sort(np.asarray(values, dtype=np.float64))
    if arr.size == 0:
        raise ValueError("Cannot compute lower-tail mean for empty array")
    k = max(1, int(np.ceil(frac * arr.size)))
    return float(np.mean(arr[:k]))


def ci(values: list[float], alpha: float = 0.05) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "ci_low": float(np.quantile(arr, alpha / 2)),
        "ci_high": float(np.quantile(arr, 1 - alpha / 2)),
        "positive_fraction": float(np.mean(arr > 0)),
    }


def rep_block(profile_block: dict[str, Any], representation: str, scaling: str) -> dict[str, Any]:
    return profile_block["representations"][representation]["scaling_modes"][scaling][
        "individual_example_probe"
    ]


def paired_records(block: dict[str, Any]) -> dict[str, dict[str, Any]]:
    records = {}
    for record in block["test"]["paired_margins"]:
        template_id = record["template_id"]
        if template_id in records:
            raise ValueError(f"Duplicate test template in paired records: {template_id}")
        records[template_id] = record
    return records


def prediction_records(block: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    by_template: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in block["test"]["predictions"]:
        by_template[record["template_id"]].append(record)
    return dict(by_template)


def class_metrics(records: list[dict[str, Any]]) -> dict[str, float]:
    y_true = np.asarray([record["y_true"] for record in records], dtype=np.int64)
    y_pred = np.asarray([record["y_pred"] for record in records], dtype=np.int64)
    prob = np.asarray([record["prob_negated"] for record in records], dtype=np.float64)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "auroc": float(roc_auc_score(y_true, prob)),
    }


def metrics_for_templates(
    pair_map: dict[str, dict[str, Any]],
    pred_map: dict[str, list[dict[str, Any]]],
    template_ids: list[str],
) -> dict[str, float]:
    pair_records = [pair_map[template_id] for template_id in template_ids]
    predictions = [
        prediction
        for template_id in template_ids
        for prediction in pred_map[template_id]
    ]
    output = class_metrics(predictions)
    output["pairwise_accuracy"] = float(
        np.mean([record["paired_margin"] > 0 for record in pair_records])
    )
    for out_key, record_key in MARGIN_KEYS.items():
        output[out_key] = lower_tail_mean([record[record_key] for record in pair_records])
    return output


def method_maps(block: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    return paired_records(block), prediction_records(block)


def collect_template_ids(
    std_pairs: dict[str, dict[str, Any]],
    vreg_pairs: dict[str, dict[str, Any]],
) -> list[str]:
    std_ids = set(std_pairs)
    vreg_ids = set(vreg_pairs)
    if std_ids != vreg_ids:
        raise ValueError(f"Standard/V-reg template mismatch: {std_ids ^ vreg_ids}")
    return sorted(std_ids)


def family_groups(pair_map: dict[str, dict[str, Any]], template_ids: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for template_id in template_ids:
        groups[pair_map[template_id]["family"]].append(template_id)
    return {family: sorted(ids) for family, ids in groups.items()}


def paired_delta(
    std_pairs: dict[str, dict[str, Any]],
    std_preds: dict[str, list[dict[str, Any]]],
    vreg_pairs: dict[str, dict[str, Any]],
    vreg_preds: dict[str, list[dict[str, Any]]],
    template_ids: list[str],
) -> dict[str, float]:
    std = metrics_for_templates(std_pairs, std_preds, template_ids)
    vreg = metrics_for_templates(vreg_pairs, vreg_preds, template_ids)
    return {metric: vreg[metric] - std[metric] for metric in (*MARGIN_KEYS, *SECONDARY_METRICS)}


def bootstrap_deltas(
    std_pairs: dict[str, dict[str, Any]],
    std_preds: dict[str, list[dict[str, Any]]],
    vreg_pairs: dict[str, dict[str, Any]],
    vreg_preds: dict[str, list[dict[str, Any]]],
    template_ids: list[str],
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    groups = family_groups(std_pairs, template_ids)
    draws = {metric: [] for metric in (*MARGIN_KEYS, *SECONDARY_METRICS)}
    for _ in range(n_boot):
        sampled: list[str] = []
        for ids in groups.values():
            sampled.extend(rng.choice(ids) for _ in range(len(ids)))
        delta = paired_delta(std_pairs, std_preds, vreg_pairs, vreg_preds, sampled)
        for metric, value in delta.items():
            draws[metric].append(value)
    return {
        "n_boot": n_boot,
        "seed": seed,
        "family_group_sizes": {family: len(ids) for family, ids in groups.items()},
        "metrics": {metric: ci(values) for metric, values in draws.items()},
    }


def exact_permutation(
    std_pairs: dict[str, dict[str, Any]],
    vreg_pairs: dict[str, dict[str, Any]],
    template_ids: list[str],
) -> dict[str, Any]:
    observed = {}
    output = {}
    for metric, record_key in MARGIN_KEYS.items():
        std_values = np.asarray([std_pairs[tid][record_key] for tid in template_ids], dtype=np.float64)
        vreg_values = np.asarray([vreg_pairs[tid][record_key] for tid in template_ids], dtype=np.float64)
        observed_delta = lower_tail_mean(vreg_values) - lower_tail_mean(std_values)
        observed[metric] = observed_delta
        null_deltas = []
        for mask in itertools.product((0, 1), repeat=len(template_ids)):
            mask_arr = np.asarray(mask, dtype=bool)
            perm_std = np.where(mask_arr, vreg_values, std_values)
            perm_vreg = np.where(mask_arr, std_values, vreg_values)
            null_deltas.append(lower_tail_mean(perm_vreg) - lower_tail_mean(perm_std))
        null = np.asarray(null_deltas, dtype=np.float64)
        output[metric] = {
            "observed_delta": float(observed_delta),
            "n_permutations": int(null.size),
            "one_sided_p_ge_observed": float(np.mean(null >= observed_delta - 1e-15)),
            "two_sided_p_abs_ge_observed": float(np.mean(np.abs(null) >= abs(observed_delta) - 1e-15)),
            "null_min": float(np.min(null)),
            "null_max": float(np.max(null)),
        }
    return output


def leave_one_template_out(
    std_pairs: dict[str, dict[str, Any]],
    vreg_pairs: dict[str, dict[str, Any]],
    template_ids: list[str],
) -> dict[str, Any]:
    rows = []
    for held_out in template_ids:
        kept = [template_id for template_id in template_ids if template_id != held_out]
        row = {
            "held_out_template_id": held_out,
            "family": std_pairs[held_out]["family"],
        }
        for metric, record_key in MARGIN_KEYS.items():
            std_values = [std_pairs[template_id][record_key] for template_id in kept]
            vreg_values = [vreg_pairs[template_id][record_key] for template_id in kept]
            row[f"delta_{metric}"] = lower_tail_mean(vreg_values) - lower_tail_mean(std_values)
        rows.append(row)
    summary = {}
    for metric in MARGIN_KEYS:
        values = [row[f"delta_{metric}"] for row in rows]
        summary[metric] = {
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "all_positive": bool(np.all(np.asarray(values) > 0)),
        }
    return {"summary": summary, "rows": rows}


def random_label_summary(block: dict[str, Any]) -> dict[str, Any]:
    controls = block["random_label_controls"]
    output = {}
    for metric in MARGIN_KEYS:
        test_key = TEST_MARGIN_KEYS[metric]
        values = [controls[seed]["test"][test_key] for seed in sorted(controls)]
        output[metric] = {
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "mean": float(np.mean(values)),
            "values_by_seed": {seed: controls[seed]["test"][test_key] for seed in sorted(controls)},
        }
    return output


def scaling_signs(profile_block: dict[str, Any]) -> dict[str, Any]:
    output = {}
    std_rep = profile_block["representations"][STANDARD_REP]["scaling_modes"]
    vreg_rep = profile_block["representations"][VREG_REP]["scaling_modes"]
    for scaling in sorted(std_rep):
        std_test = std_rep[scaling]["individual_example_probe"]["test"]
        vreg_test = vreg_rep[scaling]["individual_example_probe"]["test"]
        output[scaling] = {
            metric: {
                "delta": vreg_test[TEST_MARGIN_KEYS[metric]] - std_test[TEST_MARGIN_KEYS[metric]],
                "positive": bool(
                    vreg_test[TEST_MARGIN_KEYS[metric]] - std_test[TEST_MARGIN_KEYS[metric]] > 0
                ),
            }
            for metric in MARGIN_KEYS
        }
    return output


def per_template_table(
    std_pairs: dict[str, dict[str, Any]],
    vreg_pairs: dict[str, dict[str, Any]],
    template_ids: list[str],
) -> list[dict[str, Any]]:
    rows = []
    for template_id in template_ids:
        row = {
            "template_id": template_id,
            "family": std_pairs[template_id]["family"],
            "global_pair_id": std_pairs[template_id]["global_pair_id"],
        }
        for metric, record_key in MARGIN_KEYS.items():
            row[f"standard_{metric}"] = std_pairs[template_id][record_key]
            row[f"vreg_{metric}"] = vreg_pairs[template_id][record_key]
            row[f"delta_{metric}"] = vreg_pairs[template_id][record_key] - std_pairs[template_id][record_key]
        rows.append(row)
    return rows


def analyze_profile(profile: str, profile_block: dict[str, Any], n_boot: int, seed: int) -> dict[str, Any]:
    std_block = rep_block(profile_block, STANDARD_REP, PRIMARY_SCALING)
    vreg_block = rep_block(profile_block, VREG_REP, PRIMARY_SCALING)
    std_pairs, std_preds = method_maps(std_block)
    vreg_pairs, vreg_preds = method_maps(vreg_block)
    template_ids = collect_template_ids(std_pairs, vreg_pairs)
    std_point = metrics_for_templates(std_pairs, std_preds, template_ids)
    vreg_point = metrics_for_templates(vreg_pairs, vreg_preds, template_ids)
    point_delta = paired_delta(std_pairs, std_preds, vreg_pairs, vreg_preds, template_ids)
    readout = {
        "selected_C": {
            "standard": std_block["selected_C"],
            "vreg": vreg_block["selected_C"],
        },
        "standard": {metric: std_point[metric] for metric in (*MARGIN_KEYS, *SECONDARY_METRICS)},
        "vreg": {metric: vreg_point[metric] for metric in (*MARGIN_KEYS, *SECONDARY_METRICS)},
        "delta": point_delta,
        "scaling_signs": scaling_signs(profile_block),
        "random_label_l20_summary": {
            "standard": random_label_summary(std_block),
            "vreg": random_label_summary(vreg_block),
        },
        "per_template_margins": per_template_table(std_pairs, vreg_pairs, template_ids),
    }
    return {
        "profile": profile,
        "n_test_templates": len(template_ids),
        "readout": readout,
        "bootstrap": bootstrap_deltas(
            std_pairs,
            std_preds,
            vreg_pairs,
            vreg_preds,
            template_ids,
            n_boot,
            seed,
        ),
        "exact_permutation": exact_permutation(std_pairs, vreg_pairs, template_ids),
        "leave_one_template_out": leave_one_template_out(std_pairs, vreg_pairs, template_ids),
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# E3 Negation Cached Inference",
        "",
        f"Primary scaling: `{payload['primary_scaling']}`",
        f"Primary representation comparison: `{STANDARD_REP}` vs `{VREG_REP}`",
        f"Bootstrap reps: {payload['n_boot']}",
        "",
        "## Primary Readout",
        "",
        "| Model | Std L20 | V-reg L20 | Delta L20 | 95% CI | Pos. frac | Perm p(one-sided) | LOTO all + |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for profile in PROFILES:
        block = payload["profiles"][profile]
        readout = block["readout"]
        boot = block["bootstrap"]["metrics"]["probability_l20"]
        perm = block["exact_permutation"]["probability_l20"]
        loto = block["leave_one_template_out"]["summary"]["probability_l20"]
        lines.append(
            f"| {profile} | {readout['standard']['probability_l20']:.4f} | "
            f"{readout['vreg']['probability_l20']:.4f} | {readout['delta']['probability_l20']:.4f} | "
            f"[{boot['ci_low']:.4f}, {boot['ci_high']:.4f}] | {boot['positive_fraction']:.3f} | "
            f"{perm['one_sided_p_ge_observed']:.4f} | {loto['all_positive']} |"
        )
    lines.extend(["", "## Secondary Margins", ""])
    lines.append("| Model | Delta logit L20 | Delta geom L20 | BA delta | AUROC delta |")
    lines.append("|---|---:|---:|---:|---:|")
    for profile in PROFILES:
        delta = payload["profiles"][profile]["readout"]["delta"]
        lines.append(
            f"| {profile} | {delta['logit_l20']:.4f} | {delta['geometric_l20']:.4f} | "
            f"{delta['balanced_accuracy']:.4f} | {delta['auroc']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Infer E3 negation effects from cached analysis")
    parser.add_argument("--analysis-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    analysis = read_json(args.analysis_json)
    if analysis.get("status") != "complete":
        raise ValueError(f"Expected complete all-profile cached analysis, got {analysis.get('status')!r}")

    payload: dict[str, Any] = {
        "experiment": "E3_heldout_probes",
        "analysis": "cached_negation_template_inference",
        "status": "complete",
        "source_analysis_json": str(args.analysis_json),
        "source_analysis_sha256": sha256_file(args.analysis_json),
        "inference_script_sha256": sha256_file(Path(__file__)),
        "task_split_sha256": analysis["task_split_sha256"],
        "primary_scaling": PRIMARY_SCALING,
        "primary_representations": {
            "standard": STANDARD_REP,
            "vreg": VREG_REP,
        },
        "primary_endpoint": "delta_probability_l20",
        "n_boot": args.n_boot,
        "seed": args.seed,
        "profiles": {},
    }
    for profile in PROFILES:
        payload["profiles"][profile] = analyze_profile(
            profile,
            analysis["profiles"][profile],
            args.n_boot,
            args.seed,
        )

    write_json(args.output_json, payload)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    write_markdown(args.output_md, payload)
    print(f"Saved JSON -> {args.output_json}")
    print(f"Saved Markdown -> {args.output_md}")


if __name__ == "__main__":
    main()
