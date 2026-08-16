#!/usr/bin/env python3
"""Evaluate targeted GPT-2 laterality repair checkpoint on frozen OpenI reports."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

REVISION_ROOT = Path(__file__).resolve().parents[1]
E3_ROOT = REVISION_ROOT / "E3_heldout_probes"
E3B_ROOT = REVISION_ROOT / "E3b_external_negation"
SAE_ROOT = REVISION_ROOT.parent.parent / "SAE_scaling"
for path in (E3_ROOT, E3B_ROOT, SAE_ROOT, REVISION_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from analyze_e3_negation_cached_features import analyze_representation  # noqa: E402
from infer_openi_report_cluster import ci  # noqa: E402
from run_e3_negation_probes import deduplicate_examples, read_json, sha256_file  # noqa: E402
from sae_model_v2 import StandardSAE, load_any_sae  # noqa: E402
from shared.path_registry import checkpoint_dir, load_manifest  # noqa: E402

PRIMARY_SCALING = "standard"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def validate_feature_cache_metadata(
    cache_meta: dict[str, Any],
    split_sha256: str,
    profile: str,
    model_cfg: dict[str, Any],
) -> None:
    required = {
        "profile": profile,
        "task_split_sha256": split_sha256,
        "model_id": model_cfg["model_id"],
        "layer": int(model_cfg["hf_hidden_state_index"]),
        "extraction_protocol": "true_last",
    }
    for key, expected in required.items():
        observed = cache_meta.get(key)
        if observed != expected:
            raise ValueError(f"Feature cache metadata mismatch for {key}: {observed!r} != {expected!r}")


def paired_by_report(block: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in block["paired_margins"]:
        out[row["report_id"]].append(row)
    return dict(out)


def predictions_by_report(block: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in block["predictions"]:
        out[row["report_id"]].append(row)
    return dict(out)


def lower_tail_mean(values: list[float]) -> float:
    arr = np.sort(np.asarray(values, dtype=np.float64))
    k = max(1, int(np.ceil(0.2 * len(arr))))
    return float(np.mean(arr[:k]))


def paired_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "probability_l20": lower_tail_mean([row["paired_margin"] for row in rows]),
        "logit_l20": lower_tail_mean([row["logit_paired_margin"] for row in rows]),
        "geometric_l20": lower_tail_mean([row["geometric_paired_margin"] for row in rows]),
        "pairwise_accuracy": float(np.mean([row["paired_margin"] > 0 for row in rows])),
    }


def prediction_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

    y_true = np.asarray([row["y_true"] for row in rows], dtype=np.int64)
    y_pred = np.asarray([row["y_pred"] for row in rows], dtype=np.int64)
    prob = np.asarray([row["prob_positive"] for row in rows], dtype=np.float64)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "auroc": float(roc_auc_score(y_true, prob)),
    }


def report_cluster_delta(
    standard_block: dict[str, Any],
    candidate_block: dict[str, Any],
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    std_pairs = paired_by_report(standard_block)
    cand_pairs = paired_by_report(candidate_block)
    std_preds = predictions_by_report(standard_block)
    cand_preds = predictions_by_report(candidate_block)
    report_ids = sorted(set(std_pairs) & set(cand_pairs))
    if set(std_pairs) != set(cand_pairs):
        raise ValueError("Report IDs differ across compared representations")

    def metrics(ids: list[str], source_pairs: dict[str, list[dict[str, Any]]], source_preds: dict[str, list[dict[str, Any]]]):
        pair_rows = [row for report_id in ids for row in source_pairs[report_id]]
        pred_rows = [row for report_id in ids for row in source_preds[report_id]]
        return {**paired_metrics(pair_rows), **prediction_metrics(pred_rows)}

    point_std = metrics(report_ids, std_pairs, std_preds)
    point_candidate = metrics(report_ids, cand_pairs, cand_preds)
    point_delta = {key: point_candidate[key] - point_std[key] for key in point_std}

    rng = random.Random(seed)
    draws: dict[str, list[float]] = defaultdict(list)
    for _ in range(n_boot):
        sampled = [rng.choice(report_ids) for _ in report_ids]
        b_std = metrics(sampled, std_pairs, std_preds)
        b_cand = metrics(sampled, cand_pairs, cand_preds)
        for key in b_std:
            draws[key].append(b_cand[key] - b_std[key])
    return {
        "n_test_reports": len(report_ids),
        "n_test_pairs": sum(len(std_pairs[report_id]) for report_id in report_ids),
        "point": {
            "standard": point_std,
            "candidate": point_candidate,
            "delta": point_delta,
        },
        "bootstrap": {key: ci(values) for key, values in draws.items()},
    }


@torch.no_grad()
def encode_code(sae, hidden: np.ndarray, device: str, batch_size: int) -> np.ndarray:
    chunks = []
    for start in range(0, len(hidden), batch_size):
        h = torch.as_tensor(hidden[start : start + batch_size], dtype=torch.float32, device=device)
        z = sae.encode(h).float().cpu().numpy()
        chunks.append(z)
    return np.concatenate(chunks, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate E3c targeted laterality repair")
    parser.add_argument("--profile", choices=("gpt2", "gemma-2-2b", "qwen-2.5-3b"), default="gpt2")
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--feature-cache-npz", type=Path, required=True)
    parser.add_argument("--target-checkpoint", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--encode-batch-size", type=int, default=128)
    parser.add_argument("--selection-metric", default="pair_margin_l20")
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    manifest = load_manifest()
    model_cfg = manifest["models"][args.profile]
    standard_checkpoint_id = f"{args.profile}_standard_joint16_owt"
    split = read_json(args.split_json)
    split_sha = sha256_file(args.split_json)
    examples = deduplicate_examples(split["examples"])
    with np.load(args.feature_cache_npz, allow_pickle=True) as data:
        hidden = data["hidden"]
        zero_shot_vreg = data["sae_vreg_code"]
        cache_meta = data["metadata"].item()
    validate_feature_cache_metadata(cache_meta, split_sha, args.profile, model_cfg)
    if len(examples) != hidden.shape[0] or len(examples) != zero_shot_vreg.shape[0]:
        raise ValueError(
            "Example/cache row mismatch: "
            f"{len(examples)} examples, {hidden.shape[0]} hidden rows, {zero_shot_vreg.shape[0]} vreg rows"
        )

    standard_sae = load_any_sae(checkpoint_dir(standard_checkpoint_id, manifest), device=args.device)
    target_sae = StandardSAE.load_checkpoint(args.target_checkpoint, device=args.device)
    standard_code = encode_code(standard_sae, hidden, args.device, args.encode_batch_size)
    target_code = encode_code(target_sae, hidden, args.device, args.encode_batch_size)

    reps = {
        "standard_code": standard_code,
        "zero_shot_vreg_code": zero_shot_vreg,
        "targeted_repair_code": target_code,
    }
    payload: dict[str, Any] = {
        "experiment": "E3c_targeted_laterality_repair",
        "analysis": "targeted_repair_probe_evaluation",
        "status": "complete",
        "split_json": str(args.split_json),
        "split_sha256": split_sha,
        "feature_cache_npz": str(args.feature_cache_npz),
        "feature_cache_metadata": cache_meta,
        "profile": args.profile,
        "standard_checkpoint_id": standard_checkpoint_id,
        "target_checkpoint": str(args.target_checkpoint),
        "selection_metric": args.selection_metric,
        "representations": {},
        "comparisons": {},
    }
    for name, matrix in reps.items():
        payload["representations"][name] = analyze_representation(
            matrix,
            examples,
            name,
            args.selection_metric,
            split["random_label_control"]["seeds"],
        )
    std_block = payload["representations"]["standard_code"]["scaling_modes"][PRIMARY_SCALING]["individual_example_probe"]["test"]
    for candidate in ("zero_shot_vreg_code", "targeted_repair_code"):
        cand_block = payload["representations"][candidate]["scaling_modes"][PRIMARY_SCALING]["individual_example_probe"]["test"]
        payload["comparisons"][f"{candidate}_minus_standard"] = report_cluster_delta(
            std_block,
            cand_block,
            args.n_boot,
            args.seed,
        )
    write_json(args.output_json, payload)

    lines = [
        "# E3c Targeted Laterality Repair Evaluation",
        "",
        f"Target checkpoint: `{args.target_checkpoint}`",
        "",
        "| Representation | BA | AUROC | F1 | Pair acc | L20 | Logit L20 | Geom L20 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in reps:
        test = payload["representations"][name]["scaling_modes"][PRIMARY_SCALING]["individual_example_probe"]["test"]
        lines.append(
            f"| {name} | {test['balanced_accuracy']:.4f} | {test['auroc']:.4f} | "
            f"{test['macro_f1']:.4f} | {test['pairwise_accuracy']:.4f} | "
            f"{test['pair_margin_l20']:.4f} | {test['logit_pair_margin_l20']:.4f} | "
            f"{test['geometric_pair_margin_l20']:.4f} |"
        )
    lines.extend(["", "## Report-Cluster Deltas Vs Standard", ""])
    lines.append("| Candidate | Delta L20 | 95% CI | Delta BA | Delta F1 |")
    lines.append("|---|---:|---:|---:|---:|")
    for key, block in payload["comparisons"].items():
        boot = block["bootstrap"]["probability_l20"]
        lines.append(
            f"| {key} | {block['point']['delta']['probability_l20']:+.4f} | "
            f"[{boot['ci_low']:.4f}, {boot['ci_high']:.4f}] | "
            f"{block['point']['delta']['balanced_accuracy']:+.4f} | "
            f"{block['point']['delta']['macro_f1']:+.4f} |"
        )
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved JSON -> {args.output_json}")
    print(f"Saved Markdown -> {args.output_md}")


if __name__ == "__main__":
    main()
