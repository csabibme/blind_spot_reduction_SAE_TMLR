#!/usr/bin/env python3
"""Held-out-template negation probe on cached features.

Primary result: on TEST templates the V-regulariser never saw, is the
affirmed/negated distinction more accessible from the V-reg SAE code than from
the Standard SAE code? Reports AUROC + balanced accuracy for hidden vs
sae_standard_code vs sae_vreg_code (and reconstructions).

Protocol (matches E3): StandardScaler fit on train only; LogisticRegression
with a C grid selected on dev AUROC; test evaluated once.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
C_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]
REPS = ["hidden", "std_code", "vreg_code", "std_recon", "vreg_recon"]
REP_LABEL = {
    "hidden": "hidden",
    "std_code": "sae_standard_code",
    "vreg_code": "sae_vreg_code",
    "std_recon": "sae_standard_reconstruction",
    "vreg_recon": "sae_vreg_reconstruction",
}


def fit_eval(x, y, splits):
    tr, dv, te = splits == "train", splits == "dev", splits == "test"
    scaler = StandardScaler().fit(x[tr])
    xtr, xdv, xte = scaler.transform(x[tr]), scaler.transform(x[dv]), scaler.transform(x[te])
    best_c, best_auc, best_model = None, -1.0, None
    for c in C_GRID:
        m = LogisticRegression(C=c, solver="lbfgs", max_iter=20000,
                               class_weight="balanced", random_state=42)
        m.fit(xtr, y[tr])
        dev_auc = roc_auc_score(y[dv], m.predict_proba(xdv)[:, 1]) if len(set(y[dv])) > 1 else 0.5
        if dev_auc > best_auc:
            best_c, best_auc, best_model = c, dev_auc, m
    p_te = best_model.predict_proba(xte)[:, 1]
    yhat_te = best_model.predict(xte)
    return {
        "C": best_c,
        "dev_auroc": round(float(best_auc), 4),
        "test_auroc": round(float(roc_auc_score(y[te], p_te)), 4) if len(set(y[te])) > 1 else None,
        "test_ba": round(float(balanced_accuracy_score(y[te], yhat_te)), 4),
        "n_test": int(te.sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, default=HERE / "results/features_negation.npz")
    ap.add_argument("--out-dir", type=Path, default=HERE / "results")
    args = ap.parse_args()

    d = np.load(args.features, allow_pickle=True)
    y = d["labels"]
    splits = d["splits"]

    results = {}
    for rep in REPS:
        results[rep] = fit_eval(d[rep], y, splits)

    (args.out_dir / "probe_negation.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    lines = ["# Held-out-template negation probe (TEST templates unseen by V-loss)", "",
             "| Representation | test AUROC | test BA | C |", "|---|---:|---:|---:|"]
    for rep in REPS:
        r = results[rep]
        lines.append(f"| {REP_LABEL[rep]} | {r['test_auroc']} | {r['test_ba']} | {r['C']} |")
    lines += ["",
              f"Standard code AUROC -> V-reg code AUROC: "
              f"{results['std_code']['test_auroc']} -> {results['vreg_code']['test_auroc']} "
              f"(Δ {round((results['vreg_code']['test_auroc'] or 0) - (results['std_code']['test_auroc'] or 0), 4)})",
              "",
              f"Standard code BA -> V-reg code BA: "
              f"{results['std_code']['test_ba']} -> {results['vreg_code']['test_ba']} "
              f"(Δ {round(results['vreg_code']['test_ba'] - results['std_code']['test_ba'], 4)})"]
    (args.out_dir / "probe_negation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
