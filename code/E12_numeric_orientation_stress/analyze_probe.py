#!/usr/bin/env python3
"""Analyze hidden/SAE-code probes on the numeric-orientation stress test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

C_GRID = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]


def lower_tail_mean(values: np.ndarray, frac: float = 0.2) -> float:
    v = np.sort(np.asarray(values, dtype=np.float64).reshape(-1))
    k = max(1, int(np.ceil(frac * len(v))))
    return float(np.mean(v[:k]))


def fit_probe(x_train: np.ndarray, y_train: np.ndarray, x_dev: np.ndarray, y_dev: np.ndarray, seed: int, c_grid: Iterable[float]):
    best_c, best_score = None, -1.0
    for c in c_grid:
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=float(c), solver="lbfgs", max_iter=10000, class_weight="balanced", random_state=seed),
        )
        clf.fit(x_train, y_train)
        pred = clf.predict(x_dev)
        score = balanced_accuracy_score(y_dev, pred)
        if score > best_score:
            best_score, best_c = score, float(c)
    final = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=float(best_c), solver="lbfgs", max_iter=10000, class_weight="balanced", random_state=seed),
    )
    final.fit(x_train, y_train)
    return final, best_c


def eval_subset(clf, x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    if len(y) == 0:
        return {"n": 0}
    pred = clf.predict(x)
    prob = clf.predict_proba(x)[:, 1]
    signed_margin = np.where(y == 1, prob, 1.0 - prob)
    out = {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "prob_margin_mean": float(np.mean(signed_margin)),
        "prob_margin_l20": lower_tail_mean(signed_margin),
    }
    if len(np.unique(y)) == 2:
        out["auroc"] = float(roc_auc_score(y, prob))
    else:
        out["auroc"] = None
    return out


def cluster_bootstrap_delta(prob_std: np.ndarray, prob_vreg: np.ndarray, y: np.ndarray, clusters: np.ndarray, metric: str, seed: int, n_boot: int) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    unique = np.unique(clusters)
    by_cluster = {c: np.where(clusters == c)[0] for c in unique}

    def metric_fn(prob: np.ndarray, yy: np.ndarray) -> float:
        pred = (prob >= 0.5).astype(int)
        if metric == "auroc":
            if len(np.unique(yy)) < 2:
                return np.nan
            return float(roc_auc_score(yy, prob))
        if metric == "balanced_accuracy":
            return float(balanced_accuracy_score(yy, pred))
        raise ValueError(metric)

    point = metric_fn(prob_vreg, y) - metric_fn(prob_std, y)
    boots = []
    for _ in range(n_boot):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([by_cluster[c] for c in chosen])
        val_std = metric_fn(prob_std[idx], y[idx])
        val_vreg = metric_fn(prob_vreg[idx], y[idx])
        if np.isfinite(val_std) and np.isfinite(val_vreg):
            boots.append(val_vreg - val_std)
    if not boots:
        return {"point": float(point), "lo": None, "hi": None, "n_clusters": int(len(unique)), "n_successful": 0}
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {"point": float(point), "lo": float(lo), "hi": float(hi), "n_clusters": int(len(unique)), "n_successful": len(boots)}


def analyze_one(name: str, x: np.ndarray, items: List[Dict[str, Any]], seed: int) -> Dict[str, Any]:
    y = np.asarray([int(it["correct_index"]) for it in items], dtype=np.int64)
    split = np.asarray([it["split"] for it in items])
    train, dev, test = split == "train", split == "dev", split == "test"
    clf, best_c = fit_probe(x[train], y[train], x[dev], y[dev], seed, C_GRID)
    test_prob = clf.predict_proba(x[test])[:, 1]
    out: Dict[str, Any] = {
        "representation": name,
        "selected_c": best_c,
        "overall_test": eval_subset(clf, x[test], y[test]),
        "by_family_test": {},
        "by_regime_test": {},
        "test_prob_positive": test_prob.astype(float).tolist(),
        "test_y_true": y[test].astype(int).tolist(),
        "test_template_id": [items[i]["template_id"] for i in np.where(test)[0]],
    }
    families = sorted({it["family"] for it in items})
    regimes = sorted({it["regime"] for it in items})
    for fam in families:
        mask = np.asarray([it["family"] == fam and it["split"] == "test" for it in items])
        out["by_family_test"][fam] = eval_subset(clf, x[mask], y[mask])
    for reg in regimes:
        mask = np.asarray([it["regime"] == reg and it["split"] == "test" for it in items])
        out["by_regime_test"][reg] = eval_subset(clf, x[mask], y[mask])
    return out


def write_md(result: Dict[str, Any], out_path: Path) -> None:
    lines = ["# Numeric-orientation stress probe analysis", ""]
    lines.append("## Overall test metrics")
    lines.append("")
    lines.append("| Representation | AUROC | Balanced acc. | Accuracy | L20 prob margin |")
    lines.append("|---|---:|---:|---:|---:|")
    for name, m in result["representations"].items():
        o = m["overall_test"]
        au = o.get("auroc")
        lines.append(f"| {name} | {au if au is not None else float('nan'):.4f} | {o['balanced_accuracy']:.4f} | {o['accuracy']:.4f} | {o['prob_margin_l20']:.4f} |")
    if result.get("delta_vreg_minus_standard"):
        lines += ["", "## V-reg minus Standard", ""]
        for k, v in result["delta_vreg_minus_standard"].items():
            if isinstance(v, dict):
                lines.append(f"- {k}: {v['point']:+.4f} 95% CI [{v.get('lo')}, {v.get('hi')}]")
            else:
                lines.append(f"- {k}: {v:+.4f}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--features", required=True)
    p.add_argument("--out", default="results/probe_analysis.json")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bootstrap-n", type=int, default=1000)
    args = p.parse_args()

    dataset = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    items = dataset["items"]
    feat = np.load(args.features, allow_pickle=True)
    arrays = {k: feat[k] for k in feat.files if k != "metadata"}
    metadata = feat["metadata"].item() if "metadata" in feat.files else {}
    for name, arr in arrays.items():
        if arr.shape[0] != len(items):
            raise ValueError(f"Feature length mismatch for {name}: {arr.shape[0]} vs {len(items)}")
        if not np.isfinite(arr).all():
            raise ValueError(f"Non-finite values in {name}")

    reps = {name: analyze_one(name, arr, items, args.seed) for name, arr in arrays.items()}
    result: Dict[str, Any] = {
        "experiment": "numeric_orientation_stress_probe_analysis",
        "dataset": args.dataset,
        "features": args.features,
        "feature_metadata": metadata,
        "representations": reps,
    }

    if "standard_code" in reps and "vreg_code" in reps:
        std = reps["standard_code"]["overall_test"]
        vr = reps["vreg_code"]["overall_test"]
        result["delta_vreg_minus_standard"] = {
            "auroc": (vr["auroc"] - std["auroc"]) if (vr.get("auroc") is not None and std.get("auroc") is not None) else None,
            "balanced_accuracy": vr["balanced_accuracy"] - std["balanced_accuracy"],
            "accuracy": vr["accuracy"] - std["accuracy"],
            "prob_margin_l20": vr["prob_margin_l20"] - std["prob_margin_l20"],
        }
        test_mask = np.asarray([it["split"] == "test" for it in items])
        y_test = np.asarray([int(it["correct_index"]) for it in items if it["split"] == "test"], dtype=np.int64)
        clusters = np.asarray([it["template_id"] for it in items if it["split"] == "test"])
        prob_std = np.asarray(reps["standard_code"]["test_prob_positive"], dtype=np.float64)
        prob_vr = np.asarray(reps["vreg_code"]["test_prob_positive"], dtype=np.float64)
        result["delta_vreg_minus_standard"]["bootstrap_delta_auroc"] = cluster_bootstrap_delta(
            prob_std, prob_vr, y_test, clusters, "auroc", args.seed + 1, args.bootstrap_n
        )
        result["delta_vreg_minus_standard"]["bootstrap_delta_balanced_accuracy"] = cluster_bootstrap_delta(
            prob_std, prob_vr, y_test, clusters, "balanced_accuracy", args.seed + 2, args.bootstrap_n
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_md(result, out.with_suffix(".md"))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
