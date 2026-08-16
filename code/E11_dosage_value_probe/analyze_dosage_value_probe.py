#!/usr/bin/env python3
"""Confound-free single-side numeric-value accessibility probe for dosage codes.

Motivation
----------
The pair-difference probe in ``analyze_dosage_probe.py`` is confounded: critical
numeric-change pairs (e.g. "3"->"8") are the *smallest* representation edits, while
the nuisance controls (paraphrase / digit->word) are much *larger* edits. A probe on
``|z_right - z_left|`` therefore separates classes by edit magnitude (distance AUROC < 0.5)
rather than by numeric meaning, and saturates at AUROC=1.0 for every representation,
leaving no Standard-vs-V-reg signal.

This script instead asks the direct question, mirroring the toy primary endpoint:
**is the dose magnitude linearly accessible in a single SAE code, on held-out templates?**
Each text side becomes one sample ``(z, value)``; the label is high-dose (value >= 6)
vs low-dose (value <= 5). The probe is trained on train-template sides, tuned on dev,
and evaluated on disjoint test templates. No SAE training and no feature re-extraction:
the cached left/right representations are reused.
"""

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
    PROBE_SEED,
    REPRESENTATIONS,
    read_json,
    write_json,
)

PAIR_REPRESENTATIONS = tuple(f"{rep}_{side}" for rep in REPRESENTATIONS for side in ("left", "right"))
HIGH_VALUE_THRESHOLD = 6  # value pairs are (3,8),(2,7),(4,9),(5,10),(1,6): low={1..5}, high={6..10}
BOOTSTRAP_N = 1000
BOOTSTRAP_SEED = 42


def value_label(value: int) -> int:
    return 1 if int(value) >= HIGH_VALUE_THRESHOLD else 0


def build_side_samples(
    pairs: list[dict[str, Any]],
    features: dict[str, np.ndarray],
    representation: str,
) -> dict[str, np.ndarray]:
    """Deduplicate text sides into (vector, value-label, cluster, split) samples."""
    left_key = f"{representation}_left"
    right_key = f"{representation}_right"
    left = features[left_key]
    right = features[right_key]

    by_text: dict[str, dict[str, Any]] = {}
    for index, pair in enumerate(pairs):
        for text_key, num_key, vec in (
            ("text_left", "numeric_left", left[index]),
            ("text_right", "numeric_right", right[index]),
        ):
            text = pair[text_key]
            if text in by_text:
                continue
            by_text[text] = {
                "vector": np.asarray(vec, dtype=np.float32),
                "label": value_label(pair[num_key]),
                "cluster": pair["template_cluster_id"],
                "split": pair["split"],
                "value": int(pair[num_key]),
            }

    vectors = np.stack([s["vector"] for s in by_text.values()]).astype(np.float32)
    labels = np.asarray([s["label"] for s in by_text.values()], dtype=np.int64)
    clusters = np.asarray([s["cluster"] for s in by_text.values()])
    splits = np.asarray([s["split"] for s in by_text.values()])
    return {"x": vectors, "y": labels, "cluster": clusters, "split": splits}


def fit_probe(x_train, y_train, x_dev, y_dev, probe_seed):
    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_dev_s = scaler.transform(x_dev)
    best_c, best_score = C_GRID[0], -1.0
    for c in C_GRID:
        model = LogisticRegression(
            C=c, solver="lbfgs", max_iter=20000, class_weight="balanced", random_state=probe_seed
        )
        model.fit(x_train_s, y_train)
        score = balanced_accuracy_score(y_dev, model.predict(x_dev_s))
        if score > best_score:
            best_score, best_c = score, c
    final = LogisticRegression(
        C=best_c, solver="lbfgs", max_iter=20000, class_weight="balanced", random_state=probe_seed
    )
    final.fit(x_train_s, y_train)
    return scaler, final, best_c


def eval_probe(scaler, model, x, y):
    x_s = scaler.transform(x)
    prob = model.predict_proba(x_s)[:, 1]
    preds = (prob >= 0.5).astype(np.int64)
    return {
        "auroc": float(roc_auc_score(y, prob)),
        "balanced_accuracy": float(balanced_accuracy_score(y, preds)),
        "n": int(len(y)),
        "n_positive": int(np.sum(y == 1)),
        "prob_positive": prob.astype(np.float64),
    }


def probe_representation(samples: dict[str, np.ndarray]) -> dict[str, Any]:
    split = samples["split"]
    train = split == "train"
    dev = split == "dev"
    test = split == "test"
    scaler, model, best_c = fit_probe(
        samples["x"][train], samples["y"][train], samples["x"][dev], samples["y"][dev], PROBE_SEED
    )
    test_eval = eval_probe(scaler, model, samples["x"][test], samples["y"][test])
    return {
        "selected_c": float(best_c),
        "n_train": int(np.sum(train)),
        "n_dev": int(np.sum(dev)),
        "n_test": int(np.sum(test)),
        "test": test_eval,
        "_test_cluster": samples["cluster"][test],
        "_test_y": samples["y"][test],
    }


def cluster_bootstrap_delta_auroc(
    std_prob: np.ndarray,
    vreg_prob: np.ndarray,
    y: np.ndarray,
    clusters: np.ndarray,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    unique = np.unique(clusters)
    cluster_to_idx = {c: np.where(clusters == c)[0] for c in unique}
    rng = np.random.default_rng(seed)
    point = float(roc_auc_score(y, vreg_prob) - roc_auc_score(y, std_prob))
    boots: list[float] = []
    for _ in range(n_boot):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([cluster_to_idx[c] for c in chosen])
        if len(np.unique(y[idx])) < 2:
            continue
        boots.append(float(roc_auc_score(y[idx], vreg_prob[idx]) - roc_auc_score(y[idx], std_prob[idx])))
    if not boots:
        return {"point": point, "lo": float("nan"), "hi": float("nan"), "n_clusters": int(len(unique))}
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {
        "point": point,
        "lo": float(lo),
        "hi": float(hi),
        "n_clusters": int(len(unique)),
        "n_successful": len(boots),
    }


def analyze_profile(profile: str, pairs: list[dict[str, Any]], cache_path: Path) -> dict[str, Any]:
    cached = np.load(cache_path, allow_pickle=True)
    features = {name: cached[name] for name in PAIR_REPRESENTATIONS}
    for name, arr in features.items():
        if not np.isfinite(arr).all():
            raise ValueError(f"{profile}:{name} contains non-finite values.")
        if arr.shape[0] != len(pairs):
            raise ValueError(f"{profile}:{name} length {arr.shape[0]} != n_pairs {len(pairs)}")

    rep_results: dict[str, Any] = {}
    for rep in REPRESENTATIONS:
        samples = build_side_samples(pairs, features, rep)
        rep_results[rep] = probe_representation(samples)

    std = rep_results["sae_standard_code"]
    vreg = rep_results["sae_vreg_code"]
    boot = cluster_bootstrap_delta_auroc(
        std["test"]["prob_positive"],
        vreg["test"]["prob_positive"],
        std["_test_y"],
        std["_test_cluster"],
        BOOTSTRAP_N,
        BOOTSTRAP_SEED,
    )
    delta = {
        "delta_auroc": vreg["test"]["auroc"] - std["test"]["auroc"],
        "delta_balanced_accuracy": vreg["test"]["balanced_accuracy"] - std["test"]["balanced_accuracy"],
        "bootstrap_delta_auroc": boot,
    }

    clean = {}
    for rep, block in rep_results.items():
        clean[rep] = {k: v for k, v in block.items() if not k.startswith("_")}
        clean[rep]["test"] = {k: v for k, v in clean[rep]["test"].items() if k != "prob_positive"}
    return {"profile": profile, "representations": clean, "standard_vs_vreg_test": delta}


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Held-out dosage numeric VALUE-accessibility probe (confound-free)",
        "",
        "Single-side probe: classify high-dose (value >= 6) vs low-dose (value <= 5) from one "
        "representation, trained on train templates, evaluated on disjoint test templates. "
        "Avoids the edit-magnitude confound of the pair-difference probe.",
        "",
        "## Test AUROC by profile",
        "",
        "| Profile | hidden | Standard | V-reg | Δ AUROC (V-reg−Std) | 95% CI (template cluster) |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for profile, block in payload["profiles"].items():
        reps = block["representations"]
        d = block["standard_vs_vreg_test"]
        boot = d["bootstrap_delta_auroc"]
        ci = f"[{boot['lo']:+.4f}, {boot['hi']:+.4f}]" if boot.get("n_successful") else "n/a"
        lines.append(
            f"| {profile} | {reps['hidden']['test']['auroc']:.4f} | "
            f"{reps['sae_standard_code']['test']['auroc']:.4f} | "
            f"{reps['sae_vreg_code']['test']['auroc']:.4f} | "
            f"{d['delta_auroc']:+.4f} | {ci} |"
        )
    lines += [
        "",
        "## Balanced accuracy (test)",
        "",
        "| Profile | hidden | Standard | V-reg | Δ BA |",
        "|---|---:|---:|---:|---:|",
    ]
    for profile, block in payload["profiles"].items():
        reps = block["representations"]
        d = block["standard_vs_vreg_test"]
        lines.append(
            f"| {profile} | {reps['hidden']['test']['balanced_accuracy']:.4f} | "
            f"{reps['sae_standard_code']['test']['balanced_accuracy']:.4f} | "
            f"{reps['sae_vreg_code']['test']['balanced_accuracy']:.4f} | "
            f"{d['delta_balanced_accuracy']:+.4f} |"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Confound-free single-side dosage value probe")
    parser.add_argument("--dataset-json", type=Path, required=True)
    parser.add_argument("--features-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--profile", default="all", choices=["gpt2", "gemma-2-2b", "qwen-2.5-3b", "all"])
    args = parser.parse_args()

    dataset = read_json(args.dataset_json)
    pairs = dataset["pairs"]
    manifest = read_json(args.features_json)
    available = list(manifest.get("profiles", {}))
    profiles = available if args.profile == "all" else [args.profile]

    profile_results: dict[str, Any] = {}
    for profile in profiles:
        if profile not in manifest.get("profiles", {}):
            print(f"[value-probe] skip {profile}: no feature manifest entry", flush=True)
            continue
        cache_path = Path(manifest["profiles"][profile]["feature_cache_path"])
        if not cache_path.is_file():
            print(f"[value-probe] skip {profile}: cache missing {cache_path}", flush=True)
            continue
        print(f"[value-probe] {profile}: analyzing {cache_path.name}", flush=True)
        profile_results[profile] = analyze_profile(profile, pairs, cache_path)

    payload = {
        "experiment": dataset.get("experiment"),
        "endpoint": "single_side_value_accessibility",
        "high_value_threshold": HIGH_VALUE_THRESHOLD,
        "dataset_json": str(args.dataset_json),
        "features_json": str(args.features_json),
        "bootstrap_n": BOOTSTRAP_N,
        "profiles": profile_results,
    }
    write_json(args.output_json, payload)
    if args.output_md:
        write_markdown(args.output_md, payload)
    print(f"Wrote value-probe analysis -> {args.output_json}")


if __name__ == "__main__":
    main()
