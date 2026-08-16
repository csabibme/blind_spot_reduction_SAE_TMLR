#!/usr/bin/env python3
"""Analyze frozen E3 negation probes from cached features only.

This script does not load LMs or SAEs. It consumes feature caches written by
``run_e3_negation_probes.py`` and evaluates the probe endpoints that are aligned with the
V-reg perturbation mechanism: pairwise direction, paired margins, and lower-tail margins.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
REVISION_ROOT = SCRIPT_DIR.parents[0]
E1_ROOT = REVISION_ROOT / "E1_absolute_sensitivity"
for path in (SCRIPT_DIR, REVISION_ROOT, E1_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from shared.path_registry import checkpoint_dir, load_manifest  # noqa: E402
from run_e3_negation_probes import (  # noqa: E402
    C_GRID,
    DEFAULT_GEMMA_E1R_VREG,
    PROFILE_RUNS,
    REPRESENTATIONS,
    checkpoint_metadata,
    deduplicate_examples,
    deduplication_summary,
    extraction_protocol_for_profile,
    feature_cache_path,
    label_to_int,
    load_feature_cache,
    read_json,
    sha256_file,
    shuffled_train_labels,
)

SCALING_MODES = ("standard", "standard_no_mean", "train_rms", "raw")
SELECTION_METRICS = (
    "balanced_accuracy",
    "auroc",
    "macro_f1",
    "pairwise_accuracy",
    "pair_margin_q10",
    "pair_margin_l20",
    "pair_margin_min",
    "logit_pair_margin_q10",
    "logit_pair_margin_l20",
    "geometric_pair_margin_q10",
    "geometric_pair_margin_l20",
)
DEFAULT_SELECTION_METRIC = "pair_margin_l20"


def split_indices(examples: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    by_split: dict[str, list[int]] = {"train": [], "dev": [], "test": []}
    for index, example in enumerate(examples):
        by_split[example["split"]].append(index)
    return {split: np.asarray(indices, dtype=np.int64) for split, indices in by_split.items()}


def labels_for_examples(examples: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([label_to_int(example["label"]) for example in examples], dtype=np.int64)


def fit_transform_scaling(
    x_train: np.ndarray,
    x_dev: np.ndarray,
    x_test: np.ndarray,
    mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    if mode == "standard":
        scaler = StandardScaler()
        return (
            scaler.fit_transform(x_train),
            scaler.transform(x_dev),
            scaler.transform(x_test),
            {"mode": mode, "with_mean": True, "with_std": True},
        )
    if mode == "standard_no_mean":
        scaler = StandardScaler(with_mean=False)
        return (
            scaler.fit_transform(x_train),
            scaler.transform(x_dev),
            scaler.transform(x_test),
            {"mode": mode, "with_mean": False, "with_std": True},
        )
    if mode == "train_rms":
        rms = np.sqrt(np.mean(np.square(x_train), axis=0))
        scale = np.where(rms > 0, rms, 1.0)
        return (
            x_train / scale,
            x_dev / scale,
            x_test / scale,
            {"mode": mode, "center": False, "scale": "per_coordinate_train_rms"},
        )
    if mode == "raw":
        return x_train, x_dev, x_test, {"mode": mode, "center": False, "scale": "none"}
    raise ValueError(f"Unknown scaling mode: {mode}")


def train_logreg(x_train: np.ndarray, y_train: np.ndarray, c_value: float) -> LogisticRegression:
    model = LogisticRegression(
        C=c_value,
        solver="lbfgs",
        max_iter=20000,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(x_train, y_train)
    return model


def logreg_fit_metadata(model: LogisticRegression) -> dict[str, Any]:
    n_iter = [int(value) for value in np.ravel(model.n_iter_).tolist()]
    max_iter = int(model.max_iter)
    return {
        "solver": model.solver,
        "max_iter": max_iter,
        "n_iter": n_iter,
        "converged": all(value < max_iter for value in n_iter),
    }


def lower_tail_mean(values: np.ndarray, frac: float = 0.20) -> float:
    ordered = np.sort(values.astype(np.float64))
    k = max(1, int(np.ceil(frac * len(ordered))))
    return float(np.mean(ordered[:k]))


def margin_summary(values: np.ndarray, prefix: str) -> dict[str, float]:
    values = values.astype(np.float64)
    return {
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_q10": float(np.quantile(values, 0.10)),
        f"{prefix}_q20": float(np.quantile(values, 0.20)),
        f"{prefix}_l20": lower_tail_mean(values),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_max": float(np.max(values)),
    }


def paired_label_names(examples: list[dict[str, Any]]) -> tuple[str, str]:
    labels = {example["label"] for example in examples}
    if labels <= {"affirmed", "negated"}:
        return "affirmed", "negated"
    if labels <= {"left", "right"}:
        return "left", "right"
    raise ValueError(f"Unsupported paired label set: {sorted(labels)}")


def pair_margins(
    examples: list[dict[str, Any]],
    prob_positive: np.ndarray,
    decision_score: np.ndarray,
    geometric_score: np.ndarray,
) -> dict[str, Any]:
    baseline_label, positive_label = paired_label_names(examples)
    by_pair: dict[str, dict[str, Any]] = {}
    for index, example in enumerate(examples):
        entry = by_pair.setdefault(
            example["global_pair_id"],
            {
                "global_pair_id": example["global_pair_id"],
                "template_id": example["template_id"],
                "report_id": example.get("report_id"),
                "family": example["family"],
            },
        )
        entry[example["label"]] = {
            "example_id": example["example_id"],
            "prob_positive": float(prob_positive[index]),
            "decision_score": float(decision_score[index]),
            "geometric_score": float(geometric_score[index]),
        }

    records = []
    prob_margins = []
    logit_margins = []
    geometric_margins = []
    for pair_id, entry in sorted(by_pair.items()):
        if positive_label not in entry or baseline_label not in entry:
            continue
        prob_margin = entry[positive_label]["prob_positive"] - entry[baseline_label]["prob_positive"]
        logit_margin = entry[positive_label]["decision_score"] - entry[baseline_label]["decision_score"]
        geometric_margin = entry[positive_label]["geometric_score"] - entry[baseline_label]["geometric_score"]
        prob_margins.append(prob_margin)
        logit_margins.append(logit_margin)
        geometric_margins.append(geometric_margin)
        records.append(
            {
                "global_pair_id": pair_id,
                "template_id": entry["template_id"],
                "report_id": entry.get("report_id"),
                "family": entry["family"],
                "baseline_label": baseline_label,
                "positive_label": positive_label,
                "baseline_example_id": entry[baseline_label]["example_id"],
                "positive_example_id": entry[positive_label]["example_id"],
                "prob_baseline_sentence_positive": entry[baseline_label]["prob_positive"],
                "prob_positive_sentence_positive": entry[positive_label]["prob_positive"],
                "decision_baseline_sentence": entry[baseline_label]["decision_score"],
                "decision_positive_sentence": entry[positive_label]["decision_score"],
                "geometric_baseline_sentence": entry[baseline_label]["geometric_score"],
                "geometric_positive_sentence": entry[positive_label]["geometric_score"],
                "paired_margin": float(prob_margin),
                "logit_paired_margin": float(logit_margin),
                "geometric_paired_margin": float(geometric_margin),
                "direction_correct": bool(prob_margin > 0),
                "logit_direction_correct": bool(logit_margin > 0),
                "geometric_direction_correct": bool(geometric_margin > 0),
            }
        )

    if not prob_margins:
        empty = {
            "paired_endpoint_available": False,
            "pairwise_accuracy": None,
            "logit_pairwise_accuracy": None,
            "geometric_pairwise_accuracy": None,
            "paired_margins": [],
        }
        for prefix in ("pair_margin", "logit_pair_margin", "geometric_pair_margin"):
            for suffix in ("min", "q10", "q20", "l20", "median", "mean", "max"):
                empty[f"{prefix}_{suffix}"] = None
        return empty
    prob_array = np.asarray(prob_margins, dtype=np.float64)
    logit_array = np.asarray(logit_margins, dtype=np.float64)
    geometric_array = np.asarray(geometric_margins, dtype=np.float64)
    return {
        "pairwise_accuracy": float(np.mean(prob_array > 0)),
        "logit_pairwise_accuracy": float(np.mean(logit_array > 0)),
        "geometric_pairwise_accuracy": float(np.mean(geometric_array > 0)),
        **margin_summary(prob_array, "pair_margin"),
        **margin_summary(logit_array, "logit_pair_margin"),
        **margin_summary(geometric_array, "geometric_pair_margin"),
        "paired_margins": records,
    }


def example_metrics(
    model: LogisticRegression,
    x: np.ndarray,
    y: np.ndarray,
    examples: list[dict[str, Any]],
) -> dict[str, Any]:
    preds = model.predict(x)
    prob = model.predict_proba(x)[:, 1]
    decision = model.decision_function(x)
    coef_norm = float(np.linalg.norm(model.coef_))
    geometric = decision / coef_norm if coef_norm > 0 else decision
    output = {
        "balanced_accuracy": float(balanced_accuracy_score(y, preds)),
        "macro_f1": float(f1_score(y, preds, average="macro")),
        "auroc": float(roc_auc_score(y, prob)),
        "predictions": [
            {
                "example_id": example["example_id"],
                "global_pair_id": example["global_pair_id"],
                "template_id": example["template_id"],
                "report_id": example.get("report_id"),
                "family": example["family"],
                "y_true": int(y[index]),
                "y_pred": int(preds[index]),
                "prob_negated": float(prob[index]),
                "prob_positive": float(prob[index]),
                "decision_score": float(decision[index]),
                "geometric_score": float(geometric[index]),
            }
            for index, example in enumerate(examples)
        ],
    }
    output.update(pair_margins(examples, prob, decision, geometric))
    return output


def select_c_from_curve(curve: list[dict[str, Any]], selection_metric: str) -> float:
    if selection_metric not in SELECTION_METRICS:
        raise ValueError(f"Unknown selection metric: {selection_metric}")
    # Ties keep the first grid value, i.e. the strongest regularization in C_GRID order.
    best = max(curve, key=lambda row: row[selection_metric])
    return float(best["C"])


def evaluate_individual_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    train_examples: list[dict[str, Any]],
    x_dev: np.ndarray,
    y_dev: np.ndarray,
    dev_examples: list[dict[str, Any]],
    x_test: np.ndarray,
    y_test: np.ndarray,
    test_examples: list[dict[str, Any]],
    selection_metric: str,
    random_label_seeds: list[int],
) -> dict[str, Any]:
    _ = train_examples
    dev_curve = []
    for c_value in C_GRID:
        model = train_logreg(x_train, y_train, c_value)
        dev_metrics = example_metrics(model, x_dev, y_dev, dev_examples)
        dev_curve.append(
            {
                "C": c_value,
                "fit": logreg_fit_metadata(model),
                **{metric: dev_metrics[metric] for metric in SELECTION_METRICS},
            }
        )
    selected_c = select_c_from_curve(dev_curve, selection_metric)
    model = train_logreg(x_train, y_train, selected_c)
    output = {
        "selection_metric": selection_metric,
        "selected_C": selected_c,
        "selected_fit": logreg_fit_metadata(model),
        "dev_C_curve": dev_curve,
        "dev": example_metrics(model, x_dev, y_dev, dev_examples),
        "test": example_metrics(model, x_test, y_test, test_examples),
        "random_label_controls": {},
    }
    for shuffle_seed in random_label_seeds:
        shuffled_y_train = shuffled_train_labels(y_train, shuffle_seed)
        shuffle_curve = []
        for c_value in C_GRID:
            shuffle_model = train_logreg(x_train, shuffled_y_train, c_value)
            shuffle_dev_metrics = example_metrics(shuffle_model, x_dev, y_dev, dev_examples)
            shuffle_curve.append(
                {
                    "C": c_value,
                    "fit": logreg_fit_metadata(shuffle_model),
                    **{metric: shuffle_dev_metrics[metric] for metric in SELECTION_METRICS},
                }
            )
        shuffle_c = select_c_from_curve(shuffle_curve, selection_metric)
        shuffle_model = train_logreg(x_train, shuffled_y_train, shuffle_c)
        output["random_label_controls"][str(shuffle_seed)] = {
            "selection_metric": selection_metric,
            "selected_C": shuffle_c,
            "selected_fit": logreg_fit_metadata(shuffle_model),
            "dev_C_curve": shuffle_curve,
            "dev": example_metrics(shuffle_model, x_dev, y_dev, dev_examples),
            "test": example_metrics(shuffle_model, x_test, y_test, test_examples),
        }
    return output


def pair_difference_dataset(
    examples: list[dict[str, Any]],
    x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    baseline_label, positive_label = paired_label_names(examples)
    by_pair: dict[str, dict[str, Any]] = {}
    for index, example in enumerate(examples):
        entry = by_pair.setdefault(
            example["global_pair_id"],
            {
                "template_id": example["template_id"],
                "report_id": example.get("report_id"),
                "family": example["family"],
            },
        )
        entry[example["label"]] = {"index": index, "example_id": example["example_id"]}

    rows = []
    labels = []
    meta = []
    for pair_id, entry in sorted(by_pair.items()):
        if positive_label not in entry or baseline_label not in entry:
            continue
        pos_idx = entry[positive_label]["index"]
        base_idx = entry[baseline_label]["index"]
        rows.append(x[pos_idx] - x[base_idx])
        labels.append(1)
        meta.append(
            {
                "global_pair_id": pair_id,
                "template_id": entry["template_id"],
                "report_id": entry.get("report_id"),
                "family": entry["family"],
                "baseline_label": baseline_label,
                "positive_label": positive_label,
                "direction": f"{positive_label}_minus_{baseline_label}",
            }
        )
        rows.append(x[base_idx] - x[pos_idx])
        labels.append(0)
        meta.append(
            {
                "global_pair_id": pair_id,
                "template_id": entry["template_id"],
                "report_id": entry.get("report_id"),
                "family": entry["family"],
                "baseline_label": baseline_label,
                "positive_label": positive_label,
                "direction": f"{baseline_label}_minus_{positive_label}",
            }
        )
    return np.asarray(rows), np.asarray(labels, dtype=np.int64), meta


def difference_metrics(
    model: LogisticRegression,
    x: np.ndarray,
    y: np.ndarray,
    meta: list[dict[str, Any]],
) -> dict[str, Any]:
    preds = model.predict(x)
    prob = model.predict_proba(x)[:, 1]
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y, preds)),
        "macro_f1": float(f1_score(y, preds, average="macro")),
        "auroc": float(roc_auc_score(y, prob)),
        "predictions": [
            {
                **meta[index],
                "y_true": int(y[index]),
                "y_pred": int(preds[index]),
                "prob_negated_minus_affirmed": float(prob[index]),
            }
            for index in range(len(meta))
        ],
    }


def evaluate_pair_difference_probe(
    x_train: np.ndarray,
    train_examples: list[dict[str, Any]],
    x_dev: np.ndarray,
    dev_examples: list[dict[str, Any]],
    x_test: np.ndarray,
    test_examples: list[dict[str, Any]],
    random_label_seeds: list[int],
) -> dict[str, Any]:
    x_pair_train, y_pair_train, train_meta = pair_difference_dataset(train_examples, x_train)
    x_pair_dev, y_pair_dev, dev_meta = pair_difference_dataset(dev_examples, x_dev)
    x_pair_test, y_pair_test, test_meta = pair_difference_dataset(test_examples, x_test)
    _ = train_meta
    if len(y_pair_train) == 0 or len(y_pair_dev) == 0 or len(y_pair_test) == 0:
        return {
            "paired_endpoint_available": False,
            "selection_metric": "balanced_accuracy",
            "selected_C": None,
            "dev_C_curve": [],
            "dev": None,
            "test": None,
            "random_label_controls": {},
        }

    dev_curve = []
    for c_value in C_GRID:
        model = train_logreg(x_pair_train, y_pair_train, c_value)
        dev_metrics = difference_metrics(model, x_pair_dev, y_pair_dev, dev_meta)
        dev_curve.append(
            {
                "C": c_value,
                "fit": logreg_fit_metadata(model),
                "balanced_accuracy": dev_metrics["balanced_accuracy"],
                "auroc": dev_metrics["auroc"],
                "macro_f1": dev_metrics["macro_f1"],
            }
        )
    selected_c = select_c_from_curve(dev_curve, "balanced_accuracy")
    model = train_logreg(x_pair_train, y_pair_train, selected_c)
    output = {
        "selection_metric": "balanced_accuracy",
        "selected_C": selected_c,
        "selected_fit": logreg_fit_metadata(model),
        "dev_C_curve": dev_curve,
        "dev": difference_metrics(model, x_pair_dev, y_pair_dev, dev_meta),
        "test": difference_metrics(model, x_pair_test, y_pair_test, test_meta),
        "random_label_controls": {},
    }
    for shuffle_seed in random_label_seeds:
        shuffled_y_pair_train = shuffled_train_labels(y_pair_train, shuffle_seed)
        shuffle_curve = []
        for c_value in C_GRID:
            shuffle_model = train_logreg(x_pair_train, shuffled_y_pair_train, c_value)
            shuffle_dev_metrics = difference_metrics(shuffle_model, x_pair_dev, y_pair_dev, dev_meta)
            shuffle_curve.append(
                {
                    "C": c_value,
                    "fit": logreg_fit_metadata(shuffle_model),
                    "balanced_accuracy": shuffle_dev_metrics["balanced_accuracy"],
                    "auroc": shuffle_dev_metrics["auroc"],
                    "macro_f1": shuffle_dev_metrics["macro_f1"],
                }
            )
        shuffle_c = select_c_from_curve(shuffle_curve, "balanced_accuracy")
        shuffle_model = train_logreg(x_pair_train, shuffled_y_pair_train, shuffle_c)
        output["random_label_controls"][str(shuffle_seed)] = {
            "selection_metric": "balanced_accuracy",
            "selected_C": shuffle_c,
            "selected_fit": logreg_fit_metadata(shuffle_model),
            "dev_C_curve": shuffle_curve,
            "dev": difference_metrics(shuffle_model, x_pair_dev, y_pair_dev, dev_meta),
            "test": difference_metrics(shuffle_model, x_pair_test, y_pair_test, test_meta),
        }
    return output


def analyze_representation(
    x: np.ndarray,
    examples: list[dict[str, Any]],
    representation: str,
    selection_metric: str,
    random_label_seeds: list[int],
) -> dict[str, Any]:
    indices = split_indices(examples)
    y = labels_for_examples(examples)
    train_examples = [examples[i] for i in indices["train"].tolist()]
    dev_examples = [examples[i] for i in indices["dev"].tolist()]
    test_examples = [examples[i] for i in indices["test"].tolist()]

    output = {"representation": representation, "scaling_modes": {}}
    for mode in SCALING_MODES:
        x_train, x_dev, x_test, scaling_meta = fit_transform_scaling(
            x[indices["train"]],
            x[indices["dev"]],
            x[indices["test"]],
            mode,
        )
        y_train = y[indices["train"]]
        y_dev = y[indices["dev"]]
        y_test = y[indices["test"]]
        output["scaling_modes"][mode] = {
            "scaling": scaling_meta,
            "individual_example_probe": evaluate_individual_probe(
                x_train,
                y_train,
                train_examples,
                x_dev,
                y_dev,
                dev_examples,
                x_test,
                y_test,
                test_examples,
                selection_metric,
                random_label_seeds,
            ),
            "pair_difference_probe": evaluate_pair_difference_probe(
                x_train,
                train_examples,
                x_dev,
                dev_examples,
                x_test,
                test_examples,
                random_label_seeds,
            ),
        }
    return output


def profile_cache_metadata(
    profile: str,
    examples: list[dict[str, Any]],
    example_summary: dict[str, Any],
    manifest: dict[str, Any],
    task_split_sha256: str,
    lm_dtype: str,
    hidden_batch_size: int,
    max_length: int,
    gemma_vreg_checkpoint: Path,
    feature_cache_dir: Path,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    texts = [example["text"] for example in examples]
    model_cfg = manifest["models"][profile]
    layer = int(model_cfg["hf_hidden_state_index"])
    std_id, vreg_id = PROFILE_RUNS[profile]
    std_ckpt = checkpoint_dir(std_id, manifest)
    vreg_ckpt = gemma_vreg_checkpoint if profile == "gemma-2-2b" else checkpoint_dir(vreg_id, manifest)
    vreg_override = profile == "gemma-2-2b"
    std_meta = checkpoint_metadata(std_id, std_ckpt, manifest, override_used=False)
    vreg_meta = checkpoint_metadata(
        vreg_id if not vreg_override else "gemma_e1r_vreg_override",
        vreg_ckpt,
        manifest,
        override_used=vreg_override,
    )
    cache_meta = {
        "profile": profile,
        "task_split_sha256": task_split_sha256,
        "model_id": model_cfg["model_id"],
        "layer": layer,
        "dtype": lm_dtype,
        "hidden_batch_size": hidden_batch_size,
        "max_length": max_length,
        "extraction_protocol": extraction_protocol_for_profile(profile),
        "standard_checkpoint": str(std_ckpt),
        "vreg_checkpoint": str(vreg_ckpt),
        "standard_checkpoint_sha256": std_meta["sha256_sae_pt"],
        "vreg_checkpoint_sha256": vreg_meta["sha256_sae_pt"],
        "deduplication_unit": example_summary["deduplication_unit"],
        "deduplicated_examples": example_summary["deduplicated_examples"],
    }
    run_meta = {
        "model_id": model_cfg["model_id"],
        "layer": layer,
        "dtype": lm_dtype,
        "hidden_batch_size": hidden_batch_size,
        "max_length": max_length,
        "extraction_protocol": extraction_protocol_for_profile(profile),
        "standard_checkpoint": std_meta,
        "vreg_checkpoint": vreg_meta,
    }
    cache_path = feature_cache_path(
        feature_cache_dir,
        profile,
        task_split_sha256,
        texts,
    )
    return cache_path, cache_meta, run_meta


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# E3 Negation Cached-Feature Analysis",
        "",
        f"Status: `{payload['status']}`",
        f"Selection metric: `{payload['selection_metric']}`",
        f"Split variant: `{payload['split_variant']}`",
        "",
    ]
    for profile, profile_block in payload["profiles"].items():
        lines.extend([f"## {profile}", ""])
        for representation, rep_block in profile_block["representations"].items():
            lines.extend([f"### {representation}", ""])
            lines.append(
                "| Scaling | BA | AUROC | Macro F1 | Pair acc | Prob L20 | "
                "Logit L20 | Geom L20 | Prob median |"
            )
            lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
            for scaling, scaling_block in rep_block["scaling_modes"].items():
                test = scaling_block["individual_example_probe"]["test"]
                lines.append(
                    f"| {scaling} | {test['balanced_accuracy']:.3f} | {test['auroc']:.3f} | "
                    f"{test['macro_f1']:.3f} | {test['pairwise_accuracy']:.3f} | "
                    f"{test['pair_margin_l20']:.4f} | {test['logit_pair_margin_l20']:.4f} | "
                    f"{test['geometric_pair_margin_l20']:.4f} | "
                    f"{test['pair_margin_median']:.4f} |"
                )
            lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze E3 negation probes from cached features")
    parser.add_argument("--task-split-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--profile", choices=["gpt2", "gemma-2-2b", "qwen-2.5-3b", "all"], default="all")
    parser.add_argument("--lm-dtype", default="float16")
    parser.add_argument("--hidden-batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--gemma-vreg-checkpoint", type=Path, default=DEFAULT_GEMMA_E1R_VREG)
    parser.add_argument(
        "--feature-cache-dir",
        type=Path,
        default=REVISION_ROOT / "E3_heldout_probes" / "results" / "feature_cache",
    )
    parser.add_argument("--selection-metric", choices=SELECTION_METRICS, default=DEFAULT_SELECTION_METRIC)
    args = parser.parse_args()

    split_payload = read_json(args.task_split_json)
    raw_examples = split_payload["examples"]
    examples = deduplicate_examples(raw_examples)
    example_summary = deduplication_summary(raw_examples, examples)
    task_split_sha256 = sha256_file(args.task_split_json)
    manifest = load_manifest()
    profiles = ["gpt2", "gemma-2-2b", "qwen-2.5-3b"] if args.profile == "all" else [args.profile]
    requested_profiles = profiles
    random_label_seeds = split_payload["random_label_control"]["seeds"]

    payload: dict[str, Any] = {
        "experiment": "E3_heldout_probes",
        "analysis": "cached_feature_negation_probe_endpoints",
        "status": "complete" if args.profile == "all" else "complete_for_requested_profiles",
        "requested_profiles": requested_profiles,
        "task_split_json": str(args.task_split_json),
        "task_split_sha256": task_split_sha256,
        "split_variant": split_payload.get("split_variant", split_payload.get("dataset_kind")),
        "selection_metric": args.selection_metric,
        "primary_mechanistic_endpoint": "pair_margin_l20",
        "scaling_modes": list(SCALING_MODES),
        "random_label_control": {
            "shuffle_scope": "train_only",
            "dev_test_labels": "real",
            "seeds": random_label_seeds,
        },
        "protocol_stage": (
            "gpt2_protocol_development_pilot"
            if requested_profiles == ["gpt2"]
            else "confirmatory_application_of_frozen_cached_endpoint_protocol"
        ),
        "example_summary": example_summary,
        "profiles": {},
    }

    for profile in profiles:
        cache_path, cache_meta, run_meta = profile_cache_metadata(
            profile,
            examples,
            example_summary,
            manifest,
            task_split_sha256,
            args.lm_dtype,
            args.hidden_batch_size,
            args.max_length,
            args.gemma_vreg_checkpoint,
            args.feature_cache_dir,
        )
        features = load_feature_cache(cache_path, cache_meta)
        if features is None:
            raise FileNotFoundError(
                f"Missing or stale feature cache for {profile}: {cache_path}. "
                "Run run_e3_negation_probes.py for this profile first."
            )
        profile_block = {
            "profile": profile,
            "feature_cache_path": str(cache_path),
            "run_metadata": run_meta,
            "representations": {},
        }
        for representation in REPRESENTATIONS:
            profile_block["representations"][representation] = analyze_representation(
                features[representation],
                examples,
                representation,
                args.selection_metric,
                random_label_seeds,
            )
        payload["profiles"][profile] = profile_block

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    write_markdown(args.output_md, payload)
    print(f"Saved JSON -> {args.output_json}")
    print(f"Saved Markdown -> {args.output_md}")


if __name__ == "__main__":
    main()
