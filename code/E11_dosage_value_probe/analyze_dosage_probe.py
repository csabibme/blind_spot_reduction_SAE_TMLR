#!/usr/bin/env python3
"""Analyze held-out dosage numeric probe from cached pair features only."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dosage_probe_common import (  # noqa: E402
    C_GRID,
    LABEL_CRITICAL,
    PROBE_SEED,
    REPRESENTATIONS,
    STATUS_FROZEN,
    absolute_code_distance,
    labels_array,
    lower_tail_mean,
    pair_feature_abs_diff,
    pair_indices_by_split,
    read_json,
    relative_code_distance,
    sha256_file,
    write_json,
)
from revision_paths import REVISION_1_ROOT, ensure_import_paths  # noqa: E402

ensure_import_paths()
if str(REVISION_1_ROOT) not in sys.path:
    sys.path.insert(0, str(REVISION_1_ROOT))

from run_dosage_probe_features import (  # noqa: E402
    PAIR_REPRESENTATIONS,
    PROFILE_RUNS,
)

BOOTSTRAP_N = 1000
BOOTSTRAP_SEED = 42


def representation_side_keys(representation: str) -> tuple[str, str]:
    return f"{representation}_left", f"{representation}_right"


def validate_features_finite(features: dict[str, np.ndarray], profile: str) -> None:
    for name, arr in features.items():
        if not np.isfinite(arr).all():
            bad = int((~np.isfinite(arr)).sum())
            raise ValueError(
                f"{profile}:{name} contains {bad} non-finite values; "
                "re-run feature extraction (gemma/qwen need lm_dtype=float32)."
            )


def fit_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_dev: np.ndarray,
    y_dev: np.ndarray,
    probe_seed: int,
) -> tuple[StandardScaler, LogisticRegression, float]:
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_dev_scaled = scaler.transform(x_dev)
    best_c = C_GRID[0]
    best_score = -1.0
    for c in C_GRID:
        model = LogisticRegression(
            C=c,
            solver="lbfgs",
            max_iter=20000,
            class_weight="balanced",
            random_state=probe_seed,
        )
        model.fit(x_train_scaled, y_train)
        preds = model.predict(x_dev_scaled)
        score = balanced_accuracy_score(y_dev, preds)
        if score > best_score:
            best_score = score
            best_c = c
    final_model = LogisticRegression(
        C=best_c,
        solver="lbfgs",
        max_iter=20000,
        class_weight="balanced",
        random_state=probe_seed,
    )
    final_model.fit(x_train_scaled, y_train)
    return scaler, final_model, best_c


def evaluate_split(
    scaler: StandardScaler,
    model: LogisticRegression,
    x: np.ndarray,
    y: np.ndarray,
) -> dict[str, Any]:
    x_scaled = scaler.transform(x)
    preds = model.predict(x_scaled)
    prob = model.predict_proba(x_scaled)[:, 1]
    signed_margin = np.where(y == 1, prob, 1.0 - prob)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y, preds)),
        "auroc": float(roc_auc_score(y, prob)),
        "prob_margin_l20": lower_tail_mean(signed_margin),
        "prob_margin_mean": float(np.mean(signed_margin)),
        "n_pairs": int(len(y)),
        "prob_positive": prob.astype(np.float64).tolist(),
        "y_true": y.astype(np.int64).tolist(),
    }


def response_magnitude_metrics(
    left: np.ndarray,
    right: np.ndarray,
    y: np.ndarray,
) -> dict[str, Any]:
    abs_dist = absolute_code_distance(left, right)
    rel_dist = relative_code_distance(left, right)
    critical = y == 1
    if not np.any(critical):
        raise ValueError("No critical numeric-change pairs in split.")
    return {
        "critical_n_pairs": int(np.sum(critical)),
        "critical_distance_abs_l20": lower_tail_mean(abs_dist[critical]),
        "critical_distance_abs_mean": float(np.mean(abs_dist[critical])),
        "critical_distance_rel_l20": lower_tail_mean(rel_dist[critical]),
        "critical_distance_rel_mean": float(np.mean(rel_dist[critical])),
        "critical_distance_abs": abs_dist[critical].astype(np.float64).tolist(),
        "critical_distance_rel": rel_dist[critical].astype(np.float64).tolist(),
    }


def distance_metrics(
    left: np.ndarray,
    right: np.ndarray,
    y: np.ndarray,
) -> dict[str, Any]:
    abs_dist = absolute_code_distance(left, right)
    rel_dist = relative_code_distance(left, right)
    return {
        "distance_abs_auroc": float(roc_auc_score(y, abs_dist)),
        "distance_rel_auroc": float(roc_auc_score(y, rel_dist)),
        "distance_abs_mean_critical": float(np.mean(abs_dist[y == 1])),
        "distance_abs_mean_nuisance": float(np.mean(abs_dist[y == 0])),
        "distance_abs": abs_dist.astype(np.float64).tolist(),
        "distance_rel": rel_dist.astype(np.float64).tolist(),
        "y_true": y.astype(np.int64).tolist(),
    }


def cluster_bootstrap_metric(
    values: np.ndarray,
    y: np.ndarray,
    cluster_ids: np.ndarray,
    metric_fn,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    clusters = np.asarray(cluster_ids)
    unique = np.unique(clusters)
    cluster_to_idx = {cluster: np.where(clusters == cluster)[0] for cluster in unique}
    rng = np.random.default_rng(seed)
    point = float(metric_fn(values, y))
    boots: list[float] = []
    for _ in range(n_boot):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([cluster_to_idx[cluster] for cluster in chosen])
        if len(np.unique(y[idx])) < 2:
            continue
        boots.append(float(metric_fn(values[idx], y[idx])))
    if not boots:
        return {"point": point, "lo": float("nan"), "hi": float("nan"), "n_boot": n_boot, "n_clusters": len(unique)}
    alpha = 0.025
    lo, hi = np.quantile(boots, [alpha, 1.0 - alpha])
    return {
        "point": point,
        "lo": float(lo),
        "hi": float(hi),
        "n_boot": n_boot,
        "n_clusters": int(len(unique)),
        "n_successful": len(boots),
    }


def cluster_bootstrap_delta(
    std_values: np.ndarray,
    vreg_values: np.ndarray,
    y: np.ndarray,
    cluster_ids: np.ndarray,
    metric_fn,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    std_values = np.asarray(std_values, dtype=np.float64)
    vreg_values = np.asarray(vreg_values, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    clusters = np.asarray(cluster_ids)
    unique = np.unique(clusters)
    cluster_to_idx = {cluster: np.where(clusters == cluster)[0] for cluster in unique}
    rng = np.random.default_rng(seed)
    point = float(metric_fn(vreg_values, y) - metric_fn(std_values, y))
    boots: list[float] = []
    for _ in range(n_boot):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([cluster_to_idx[cluster] for cluster in chosen])
        if len(np.unique(y[idx])) < 2:
            continue
        boots.append(float(metric_fn(vreg_values[idx], y[idx]) - metric_fn(std_values[idx], y[idx])))
    if not boots:
        return {"point": point, "lo": float("nan"), "hi": float("nan"), "n_boot": n_boot, "n_clusters": len(unique)}
    alpha = 0.025
    lo, hi = np.quantile(boots, [alpha, 1.0 - alpha])
    return {
        "point": point,
        "lo": float(lo),
        "hi": float(hi),
        "n_boot": n_boot,
        "n_clusters": int(len(unique)),
        "n_successful": len(boots),
    }


def cluster_bootstrap_value_delta(
    std_values: np.ndarray,
    vreg_values: np.ndarray,
    cluster_ids: np.ndarray,
    metric_fn,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    std_values = np.asarray(std_values, dtype=np.float64)
    vreg_values = np.asarray(vreg_values, dtype=np.float64)
    clusters = np.asarray(cluster_ids)
    unique = np.unique(clusters)
    cluster_to_idx = {cluster: np.where(clusters == cluster)[0] for cluster in unique}
    rng = np.random.default_rng(seed)
    point = float(metric_fn(vreg_values) - metric_fn(std_values))
    boots = np.empty(n_boot, dtype=np.float64)
    for index in range(n_boot):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([cluster_to_idx[cluster] for cluster in chosen])
        boots[index] = float(metric_fn(vreg_values[idx]) - metric_fn(std_values[idx]))
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {
        "point": point,
        "lo": float(lo),
        "hi": float(hi),
        "n_boot": n_boot,
        "n_clusters": int(len(unique)),
        "n_successful": n_boot,
    }


def auroc_metric(values: np.ndarray, y: np.ndarray) -> float:
    return float(roc_auc_score(y, values))


def balanced_accuracy_metric(values: np.ndarray, y: np.ndarray) -> float:
    preds = (values >= 0.5).astype(np.int64)
    return float(balanced_accuracy_score(y, preds))


def mean_metric(values: np.ndarray) -> float:
    return float(np.mean(values))


def l20_metric(values: np.ndarray) -> float:
    return lower_tail_mean(values)


def signed_direction_features(
    left: np.ndarray,
    right: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a hard numeric-direction task from critical pairs only.

    Class 1 is the observed left->right increase; class 0 is the reversed
    right->left decrease. Nuisance pairs are excluded from this endpoint.
    """
    critical = y == 1
    signed = right[critical] - left[critical]
    x = np.concatenate([signed, -signed], axis=0)
    labels = np.concatenate(
        [
            np.ones(signed.shape[0], dtype=np.int64),
            np.zeros(signed.shape[0], dtype=np.int64),
        ]
    )
    return x, labels


def hard_direction_probe(
    left: np.ndarray,
    right: np.ndarray,
    y: np.ndarray,
    split_idx: dict[str, np.ndarray],
) -> dict[str, Any]:
    train_x, train_y = signed_direction_features(left[split_idx["train"]], right[split_idx["train"]], y[split_idx["train"]])
    dev_x, dev_y = signed_direction_features(left[split_idx["dev"]], right[split_idx["dev"]], y[split_idx["dev"]])
    test_x, test_y = signed_direction_features(left[split_idx["test"]], right[split_idx["test"]], y[split_idx["test"]])
    scaler, model, best_c = fit_probe(train_x, train_y, dev_x, dev_y, PROBE_SEED)
    return {
        "task": "critical_numeric_direction_increase_vs_decrease",
        "probe_feature": "signed_diff",
        "selected_c": best_c,
        "train": evaluate_split(scaler, model, train_x, train_y),
        "dev": evaluate_split(scaler, model, dev_x, dev_y),
        "test": evaluate_split(scaler, model, test_x, test_y),
    }


def analyze_representation(
    representation: str,
    features: dict[str, np.ndarray],
    pairs: list[dict[str, Any]],
    split_idx: dict[str, np.ndarray],
    y: np.ndarray,
) -> dict[str, Any]:
    left_key, right_key = representation_side_keys(representation)
    left = features[left_key]
    right = features[right_key]
    phi = pair_feature_abs_diff(left, right)

    train_idx = split_idx["train"]
    dev_idx = split_idx["dev"]
    test_idx = split_idx["test"]

    scaler, model, best_c = fit_probe(
        phi[train_idx],
        y[train_idx],
        phi[dev_idx],
        y[dev_idx],
        PROBE_SEED,
    )

    probe_train = evaluate_split(scaler, model, phi[train_idx], y[train_idx])
    probe_dev = evaluate_split(scaler, model, phi[dev_idx], y[dev_idx])
    probe_test = evaluate_split(scaler, model, phi[test_idx], y[test_idx])

    dist_train = distance_metrics(left[train_idx], right[train_idx], y[train_idx])
    dist_dev = distance_metrics(left[dev_idx], right[dev_idx], y[dev_idx])
    dist_test = distance_metrics(left[test_idx], right[test_idx], y[test_idx])
    response_train = response_magnitude_metrics(left[train_idx], right[train_idx], y[train_idx])
    response_dev = response_magnitude_metrics(left[dev_idx], right[dev_idx], y[dev_idx])
    response_test = response_magnitude_metrics(left[test_idx], right[test_idx], y[test_idx])
    direction_probe = hard_direction_probe(left, right, y, split_idx)

    return {
        "representation": representation,
        "probe_feature": "abs_diff",
        "selected_c": best_c,
        "distance_metrics": {
            "train": dist_train,
            "dev": dist_dev,
            "test": dist_test,
        },
        "critical_response_metrics": {
            "train": response_train,
            "dev": response_dev,
            "test": response_test,
        },
        "probe_metrics": {
            "train": probe_train,
            "dev": probe_dev,
            "test": probe_test,
        },
        "hard_direction_probe_metrics": direction_probe,
    }


def delta_block(
    std_block: dict[str, Any],
    vreg_block: dict[str, Any],
    test_idx: np.ndarray,
    cluster_ids: np.ndarray,
) -> dict[str, Any]:
    std_test = std_block["probe_metrics"]["test"]
    vreg_test = vreg_block["probe_metrics"]["test"]
    std_dist = std_block["distance_metrics"]["test"]
    vreg_dist = vreg_block["distance_metrics"]["test"]
    std_response = std_block["critical_response_metrics"]["test"]
    vreg_response = vreg_block["critical_response_metrics"]["test"]
    std_direction = std_block["hard_direction_probe_metrics"]["test"]
    vreg_direction = vreg_block["hard_direction_probe_metrics"]["test"]

    y_test = np.asarray(std_test["y_true"], dtype=np.int64)
    test_cluster_ids = cluster_ids[test_idx]
    critical_cluster_ids = test_cluster_ids[y_test == 1]
    std_probe_scores = np.asarray(std_test["prob_positive"], dtype=np.float64)
    vreg_probe_scores = np.asarray(vreg_test["prob_positive"], dtype=np.float64)
    std_dist_scores = np.asarray(std_dist["distance_abs"], dtype=np.float64)
    vreg_dist_scores = np.asarray(vreg_dist["distance_abs"], dtype=np.float64)
    std_critical_abs = np.asarray(std_response["critical_distance_abs"], dtype=np.float64)
    vreg_critical_abs = np.asarray(vreg_response["critical_distance_abs"], dtype=np.float64)
    std_critical_rel = np.asarray(std_response["critical_distance_rel"], dtype=np.float64)
    vreg_critical_rel = np.asarray(vreg_response["critical_distance_rel"], dtype=np.float64)

    return {
        "delta_probe_auroc": vreg_test["auroc"] - std_test["auroc"],
        "delta_probe_balanced_accuracy": vreg_test["balanced_accuracy"] - std_test["balanced_accuracy"],
        "delta_probe_prob_margin_l20": vreg_test["prob_margin_l20"] - std_test["prob_margin_l20"],
        "delta_distance_abs_auroc": vreg_dist["distance_abs_auroc"] - std_dist["distance_abs_auroc"],
        "delta_distance_rel_auroc": vreg_dist["distance_rel_auroc"] - std_dist["distance_rel_auroc"],
        "delta_critical_distance_abs_l20": (
            vreg_response["critical_distance_abs_l20"] - std_response["critical_distance_abs_l20"]
        ),
        "delta_critical_distance_abs_mean": (
            vreg_response["critical_distance_abs_mean"] - std_response["critical_distance_abs_mean"]
        ),
        "delta_critical_distance_rel_l20": (
            vreg_response["critical_distance_rel_l20"] - std_response["critical_distance_rel_l20"]
        ),
        "delta_critical_distance_rel_mean": (
            vreg_response["critical_distance_rel_mean"] - std_response["critical_distance_rel_mean"]
        ),
        "delta_hard_direction_probe_auroc": vreg_direction["auroc"] - std_direction["auroc"],
        "delta_hard_direction_balanced_accuracy": (
            vreg_direction["balanced_accuracy"] - std_direction["balanced_accuracy"]
        ),
        "bootstrap": {
            "probe_auroc": cluster_bootstrap_delta(
                std_probe_scores,
                vreg_probe_scores,
                y_test,
                test_cluster_ids,
                auroc_metric,
                BOOTSTRAP_N,
                BOOTSTRAP_SEED,
            ),
            "probe_balanced_accuracy": cluster_bootstrap_delta(
                std_probe_scores,
                vreg_probe_scores,
                y_test,
                test_cluster_ids,
                balanced_accuracy_metric,
                BOOTSTRAP_N,
                BOOTSTRAP_SEED + 1,
            ),
            "distance_abs_auroc": cluster_bootstrap_delta(
                std_dist_scores,
                vreg_dist_scores,
                y_test,
                test_cluster_ids,
                auroc_metric,
                BOOTSTRAP_N,
                BOOTSTRAP_SEED + 2,
            ),
            "critical_distance_abs_l20": cluster_bootstrap_value_delta(
                std_critical_abs,
                vreg_critical_abs,
                critical_cluster_ids,
                l20_metric,
                BOOTSTRAP_N,
                BOOTSTRAP_SEED + 3,
            ),
            "critical_distance_abs_mean": cluster_bootstrap_value_delta(
                std_critical_abs,
                vreg_critical_abs,
                critical_cluster_ids,
                mean_metric,
                BOOTSTRAP_N,
                BOOTSTRAP_SEED + 4,
            ),
            "critical_distance_rel_l20": cluster_bootstrap_value_delta(
                std_critical_rel,
                vreg_critical_rel,
                critical_cluster_ids,
                l20_metric,
                BOOTSTRAP_N,
                BOOTSTRAP_SEED + 5,
            ),
            "critical_distance_rel_mean": cluster_bootstrap_value_delta(
                std_critical_rel,
                vreg_critical_rel,
                critical_cluster_ids,
                mean_metric,
                BOOTSTRAP_N,
                BOOTSTRAP_SEED + 6,
            ),
        },
        "standard_test": {
            "probe_auroc": std_test["auroc"],
            "probe_balanced_accuracy": std_test["balanced_accuracy"],
            "distance_abs_auroc": std_dist["distance_abs_auroc"],
            "critical_distance_abs_l20": std_response["critical_distance_abs_l20"],
            "critical_distance_abs_mean": std_response["critical_distance_abs_mean"],
            "critical_distance_rel_l20": std_response["critical_distance_rel_l20"],
            "critical_distance_rel_mean": std_response["critical_distance_rel_mean"],
            "hard_direction_probe_auroc": std_direction["auroc"],
            "hard_direction_balanced_accuracy": std_direction["balanced_accuracy"],
        },
        "vreg_test": {
            "probe_auroc": vreg_test["auroc"],
            "probe_balanced_accuracy": vreg_test["balanced_accuracy"],
            "distance_abs_auroc": vreg_dist["distance_abs_auroc"],
            "critical_distance_abs_l20": vreg_response["critical_distance_abs_l20"],
            "critical_distance_abs_mean": vreg_response["critical_distance_abs_mean"],
            "critical_distance_rel_l20": vreg_response["critical_distance_rel_l20"],
            "critical_distance_rel_mean": vreg_response["critical_distance_rel_mean"],
            "hard_direction_probe_auroc": vreg_direction["auroc"],
            "hard_direction_balanced_accuracy": vreg_direction["balanced_accuracy"],
        },
    }


def analyze_profile(
    profile: str,
    pairs: list[dict[str, Any]],
    dataset_json: Path,
    feature_cache_dir: Path,
    features_manifest: dict[str, Any] | None,
) -> dict[str, Any]:
    dataset_sha = sha256_file(dataset_json)
    if features_manifest is None:
        raise ValueError("Missing features manifest.")
    if features_manifest.get("dataset_sha256") != dataset_sha:
        raise ValueError(
            "Feature manifest dataset SHA does not match dataset JSON: "
            f"{features_manifest.get('dataset_sha256')} != {dataset_sha}"
        )

    split_idx = pair_indices_by_split(pairs)
    y = labels_array(pairs)
    cluster_ids = np.asarray([pair["template_cluster_id"] for pair in pairs])

    if profile in features_manifest.get("profiles", {}):
        profile_manifest = features_manifest["profiles"][profile]
        cache_path = Path(profile_manifest["feature_cache_path"])
        cached = np.load(cache_path, allow_pickle=True)
        cache_meta = cached["metadata"].item()
        features = {name: cached[name] for name in PAIR_REPRESENTATIONS}
        validate_features_finite(features, profile)
        if cache_meta.get("dataset_sha256") != dataset_sha:
            raise ValueError(
                "Feature cache dataset SHA does not match dataset JSON: "
                f"{cache_meta.get('dataset_sha256')} != {dataset_sha}"
            )
        if cache_meta.get("n_pairs") != len(pairs):
            raise ValueError(
                f"Feature cache n_pairs mismatch: {cache_meta.get('n_pairs')} != {len(pairs)}"
            )
        for name, values in features.items():
            if values.shape[0] != len(pairs):
                raise ValueError(
                    f"Feature length mismatch for {name}: {values.shape[0]} != {len(pairs)}"
                )
        cache_info = {"feature_cache_path": str(cache_path), "cache_meta": cache_meta}
    else:
        raise ValueError(f"Missing features manifest entry for profile {profile!r}")

    rep_results = {
        rep: analyze_representation(rep, features, pairs, split_idx, y) for rep in REPRESENTATIONS
    }

    test_idx = split_idx["test"]
    std_rep = rep_results["sae_standard_code"]
    vreg_rep = rep_results["sae_vreg_code"]
    deltas = delta_block(std_rep, vreg_rep, test_idx, cluster_ids)

    return {
        "profile": profile,
        "feature_cache": cache_info,
        "representations": rep_results,
        "standard_vs_vreg_test": deltas,
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Held-out dosage numeric probe analysis",
        "",
        f"Dataset: `{payload['dataset_json']}`",
        "",
        "## Test metrics by profile",
        "",
        "| Profile | Representation | Distance AUROC | Probe AUROC | Balanced acc. | Probe L20 margin | Critical L20 `||dz||` | Critical mean `||dz||` | Hard-dir AUROC |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for profile, block in payload["profiles"].items():
        for rep_name, rep_block in block["representations"].items():
            dist = rep_block["distance_metrics"]["test"]
            probe = rep_block["probe_metrics"]["test"]
            response = rep_block["critical_response_metrics"]["test"]
            hard_direction = rep_block["hard_direction_probe_metrics"]["test"]
            lines.append(
                f"| {profile} | {rep_name} | {dist['distance_abs_auroc']:.4f} | "
                f"{probe['auroc']:.4f} | {probe['balanced_accuracy']:.4f} | "
                f"{probe['prob_margin_l20']:.4f} | "
                f"{response['critical_distance_abs_l20']:.4f} | "
                f"{response['critical_distance_abs_mean']:.4f} | "
                f"{hard_direction['auroc']:.4f} |"
            )

    lines.extend(["", "## V-reg minus Standard (test)", ""])
    for profile, block in payload["profiles"].items():
        delta = block["standard_vs_vreg_test"]
        lines.append(f"### {profile}")
        lines.append("")
        lines.append(
            f"- Δ probe AUROC: {delta['delta_probe_auroc']:+.4f}"
        )
        lines.append(
            f"- Δ balanced accuracy: {delta['delta_probe_balanced_accuracy']:+.4f}"
        )
        lines.append(
            f"- Δ distance AUROC: {delta['delta_distance_abs_auroc']:+.4f}"
        )
        lines.append(
            f"- Δ critical L20 `||dz||`: {delta['delta_critical_distance_abs_l20']:+.4f}"
        )
        lines.append(
            f"- Δ critical mean `||dz||`: {delta['delta_critical_distance_abs_mean']:+.4f}"
        )
        lines.append(
            f"- Δ critical L20 relative response `D`: {delta['delta_critical_distance_rel_l20']:+.4f}"
        )
        lines.append(
            f"- Δ hard-direction AUROC: {delta['delta_hard_direction_probe_auroc']:+.4f}"
        )
        boot = delta["bootstrap"]["probe_auroc"]
        if boot.get("n_successful", 0) > 0:
            lines.append(
                f"- 95% CI Δ probe AUROC (template cluster): [{boot['lo']:+.4f}, {boot['hi']:+.4f}]"
            )
        critical_boot = delta["bootstrap"]["critical_distance_abs_l20"]
        lines.append(
            "- 95% CI Δ critical L20 `||dz||` (template cluster): "
            f"[{critical_boot['lo']:+.4f}, {critical_boot['hi']:+.4f}]"
        )
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze dosage probe cached features")
    parser.add_argument("--dataset-json", type=Path, required=True)
    parser.add_argument("--features-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--profile", default="all", choices=["gpt2", "gemma-2-2b", "qwen-2.5-3b", "all"])
    parser.add_argument(
        "--feature-cache-dir",
        type=Path,
        default=SCRIPT_DIR.parent / "results" / "feature_cache",
    )
    args = parser.parse_args()

    dataset = read_json(args.dataset_json)
    if dataset.get("status") != STATUS_FROZEN:
        raise ValueError(f"Unexpected dataset status: {dataset.get('status')!r}")
    pairs = dataset["pairs"]
    features_manifest = read_json(args.features_json)

    profiles = list(PROFILE_RUNS) if args.profile == "all" else [args.profile]
    profile_results = {
        profile: analyze_profile(profile, pairs, args.dataset_json, args.feature_cache_dir, features_manifest)
        for profile in profiles
    }

    payload = {
        "experiment": dataset.get("experiment"),
        "dataset_json": str(args.dataset_json),
        "features_json": str(args.features_json),
        "profiles": profile_results,
        "label_positive": LABEL_CRITICAL,
        "bootstrap_n": BOOTSTRAP_N,
    }
    write_json(args.output_json, payload)
    if args.output_md:
        write_markdown(args.output_md, payload)
    print(f"Wrote analysis -> {args.output_json}")


if __name__ == "__main__":
    main()
