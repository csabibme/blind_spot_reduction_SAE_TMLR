#!/usr/bin/env python3
"""Behavioural case study: does routing the hidden state through the SAE flip a
clinical affirmed/negated decision?

A fixed decision readout is trained ONLY on true hidden states (train split).
At test time (held-out templates) we feed the readout three inputs:
  - the true hidden state (upper bound),
  - the Standard-SAE reconstruction of that hidden state,
  - the V-reg-SAE reconstruction of that hidden state.

If the Standard reconstruction causes the readout to mis-decide a sentence that
it decides correctly from the true hidden state, the Standard SAE has silently
dropped the task-relevant distinction in the reconstruction pathway. We report
per-pair 'flips' where Standard collapses the affirmed/negated decision but
V-reg preserves it.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
C_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, default=HERE / "results/features_negation.npz")
    ap.add_argument("--out-dir", type=Path, default=HERE / "results")
    args = ap.parse_args()

    d = np.load(args.features, allow_pickle=True)
    y = d["labels"]
    splits = d["splits"]
    texts = d["texts"]
    pair_ids = d["pair_ids"]
    sides = d["sides"]

    tr, dv, te = splits == "train", splits == "dev", splits == "test"

    # Readout trained on TRUE hidden states only.
    scaler = StandardScaler().fit(d["hidden"][tr])
    Xtr = scaler.transform(d["hidden"][tr])
    Xdv = scaler.transform(d["hidden"][dv])

    best_c, best_acc, readout = None, -1.0, None
    for c in C_GRID:
        m = LogisticRegression(C=c, solver="lbfgs", max_iter=20000,
                               class_weight="balanced", random_state=42)
        m.fit(Xtr, y[tr])
        acc = accuracy_score(y[dv], m.predict(Xdv))
        if acc > best_acc:
            best_c, best_acc, readout = c, acc, m

    def predict(feat_key: str, mask):
        X = scaler.transform(d[feat_key][mask])
        return readout.predict(X)

    yte = y[te]
    pred_hidden = predict("hidden", te)
    pred_std = predict("std_recon", te)
    pred_vreg = predict("vreg_recon", te)

    acc = {
        "hidden": round(float(accuracy_score(yte, pred_hidden)), 4),
        "std_recon": round(float(accuracy_score(yte, pred_std)), 4),
        "vreg_recon": round(float(accuracy_score(yte, pred_vreg)), 4),
    }
    ba = {
        "hidden": round(float(balanced_accuracy_score(yte, pred_hidden)), 4),
        "std_recon": round(float(balanced_accuracy_score(yte, pred_std)), 4),
        "vreg_recon": round(float(balanced_accuracy_score(yte, pred_vreg)), 4),
    }

    # Per-example: correct from hidden, wrong via Standard recon, correct via V-reg recon.
    te_idx = np.where(te)[0]
    saves = []  # V-reg preserves what Standard drops
    for j, gi in enumerate(te_idx):
        hid_ok = pred_hidden[j] == yte[j]
        std_ok = pred_std[j] == yte[j]
        vreg_ok = pred_vreg[j] == yte[j]
        if hid_ok and (not std_ok) and vreg_ok:
            saves.append({
                "text": str(texts[gi]),
                "true_label": "negated" if yte[j] == 1 else "affirmed",
                "hidden_pred": "negated" if pred_hidden[j] == 1 else "affirmed",
                "std_recon_pred": "negated" if pred_std[j] == 1 else "affirmed",
                "vreg_recon_pred": "negated" if pred_vreg[j] == 1 else "affirmed",
            })

    # Per-pair collapse: does the reconstruction pathway still separate the two sides?
    pair_map = defaultdict(dict)
    for j, gi in enumerate(te_idx):
        pair_map[str(pair_ids[gi])][str(sides[gi])] = j
    std_collapsed = vreg_collapsed = both_ok = n_pairs = 0
    for pid, sd in pair_map.items():
        if "aff" not in sd or "neg" not in sd:
            continue
        n_pairs += 1
        std_sep = pred_std[sd["aff"]] != pred_std[sd["neg"]]
        vreg_sep = pred_vreg[sd["aff"]] != pred_vreg[sd["neg"]]
        if not std_sep:
            std_collapsed += 1
        if not vreg_sep:
            vreg_collapsed += 1
        if std_sep and vreg_sep:
            both_ok += 1

    result = {
        "readout": {"trained_on": "true hidden (train split)", "C": best_c,
                    "dev_acc": round(float(best_acc), 4)},
        "test_accuracy": acc,
        "test_balanced_accuracy": ba,
        "n_test_pairs": n_pairs,
        "pairs_collapsed_by_standard_recon": std_collapsed,
        "pairs_collapsed_by_vreg_recon": vreg_collapsed,
        "n_vreg_saves_over_standard": len(saves),
        "vreg_save_examples": saves[:12],
    }
    (args.out_dir / "behavioral_negation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    lines = ["# Behavioural case study: SAE-in-the-loop decision flips (held-out templates)", "",
             "Readout trained on true hidden states, then applied to reconstructions.", "",
             "| Input to readout | test accuracy | test BA |", "|---|---:|---:|",
             f"| true hidden (upper bound) | {acc['hidden']} | {ba['hidden']} |",
             f"| Standard SAE reconstruction | {acc['std_recon']} | {ba['std_recon']} |",
             f"| V-reg SAE reconstruction | {acc['vreg_recon']} | {ba['vreg_recon']} |",
             "",
             f"Test minimal pairs: {n_pairs}",
             f"- collapsed (aff/neg decided the same) by Standard recon: **{std_collapsed}**",
             f"- collapsed by V-reg recon: **{vreg_collapsed}**",
             f"- V-reg saves (hidden correct, Standard wrong, V-reg correct): **{len(saves)}**",
             ""]
    if saves:
        lines.append("## Example V-reg saves")
        lines.append("")
        for s in saves[:8]:
            lines.append(f"- \"{s['text']}\" (true={s['true_label']}; "
                         f"Standard->{s['std_recon_pred']}, V-reg->{s['vreg_recon_pred']})")
    (args.out_dir / "behavioral_negation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
