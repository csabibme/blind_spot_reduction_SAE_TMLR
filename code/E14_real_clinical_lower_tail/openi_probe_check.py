#!/usr/bin/env python3
"""Cheap headroom check on a real OpenI clinical probe split, using EXISTING
experiment_101 Standard vs V-reg checkpoints (no training).

Reads a split JSON with examples [{text, label, split}], extracts last-token
hidden states with the given LM, encodes them with the two frozen SAEs, and
fits a linear probe (hidden / std_code / vreg_code). Reports test AUROC + BA.
Purpose: does this real clinical task have Standard-code headroom AND a genuine
V-reg gain? Factual go/no-go, no cherry-picking.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import common  # noqa: F401
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from activations import last_token_hidden
from lm_loader import load_model_and_tokenizer, resolve_device
from sae_model_v2 import load_any_sae

HERE = Path(__file__).resolve().parent
EXP101 = HERE.parents[1] / "tmlr_revision/prepare/experiment_101_hybrid_owt/checkpoints"
C_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]


@torch.no_grad()
def encode(sae, h, device, bs=256):
    out = []
    for i in range(0, h.shape[0], bs):
        out.append(sae.encode(h[i:i + bs].to(device=device, dtype=torch.float32)).cpu())
    return torch.cat(out).numpy()


def fit_eval(x, y, sp, return_test=False):
    tr, dv, te = sp == "train", sp == "dev", sp == "test"
    sc = StandardScaler().fit(x[tr])
    xtr, xdv, xte = sc.transform(x[tr]), sc.transform(x[dv]), sc.transform(x[te])
    best = (-1, None)
    for c in C_GRID:
        m = LogisticRegression(C=c, solver="lbfgs", max_iter=20000,
                               class_weight="balanced", random_state=42).fit(xtr, y[tr])
        a = roc_auc_score(y[dv], m.predict_proba(xdv)[:, 1]) if len(set(y[dv])) > 1 else 0.5
        if a > best[0]:
            best = (a, m)
    m = best[1]
    p_te = m.predict_proba(xte)[:, 1]
    yhat_te = m.predict(xte)
    out = {"test_auroc": round(float(roc_auc_score(y[te], p_te)), 4),
           "test_ba": round(float(balanced_accuracy_score(y[te], yhat_te)), 4),
           "n_test": int(te.sum())}
    if return_test:
        out["_p"] = p_te
        out["_yhat"] = yhat_te
    return out


def grouped_cv_predict(x, y, groups, n_folds, C=1.0):
    """Grouped K-fold by report; return pooled out-of-fold probs + hard preds."""
    gkf = GroupKFold(n_splits=n_folds)
    p = np.zeros(len(y)); yhat = np.zeros(len(y), dtype=int)
    for tr, te in gkf.split(x, y, groups):
        sc = StandardScaler().fit(x[tr])
        m = LogisticRegression(C=C, solver="lbfgs", max_iter=20000,
                               class_weight="balanced", random_state=42).fit(sc.transform(x[tr]), y[tr])
        p[te] = m.predict_proba(sc.transform(x[te]))[:, 1]
        yhat[te] = m.predict(sc.transform(x[te]))
    return p, yhat


def clustered_bootstrap_delta(y_te, p_std, yh_std, p_vreg, yh_vreg, reports_te,
                              n_boot=5000, seed=42):
    """95% CI for Δ(V-reg − Standard) on AUROC and BA, resampling test reports."""
    rng = np.random.default_rng(seed)
    uniq = np.array(sorted(set(reports_te)))
    idx_by_report = {r: np.where(reports_te == r)[0] for r in uniq}
    d_auc, d_ba = [], []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_report[r] for r in pick])
        yy = y_te[idx]
        if len(set(yy)) < 2:
            continue
        d_auc.append(roc_auc_score(yy, p_vreg[idx]) - roc_auc_score(yy, p_std[idx]))
        d_ba.append(balanced_accuracy_score(yy, yh_vreg[idx]) - balanced_accuracy_score(yy, yh_std[idx]))
    def ci(a):
        a = np.array(a)
        return [round(float(np.percentile(a, 2.5)), 4),
                round(float(np.percentile(a, 97.5)), 4),
                round(float(a.mean()), 4)]
    return {"n_reports": int(len(uniq)), "n_boot_valid": len(d_auc),
            "delta_auroc_ci_lo_hi_mean": ci(d_auc), "delta_ba_ci_lo_hi_mean": ci(d_ba)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-json", required=True, type=Path)
    ap.add_argument("--profile", default="gpt2", choices=["gpt2", "qwen-2.5-3b"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--cv-folds", type=int, default=0,
                    help="if >0, also run grouped-by-report K-fold CV (pooled OOF) for higher power")
    ap.add_argument("--out", type=Path, default=HERE / "results/openi_check.json")
    args = ap.parse_args()

    cfg = {
        "gpt2": ("gpt2", 12, False, "gpt2"),
        "qwen-2.5-3b": ("Qwen/Qwen2.5-3B", 18, True, "qwen-2.5-3b"),
    }[args.profile]
    model_id, layer, trc, ckdir = cfg
    std_ck = EXP101 / ckdir / "joint/standard_joint16_owt"
    vreg_ck = EXP101 / ckdir / "joint/vreg_joint16_owt"

    device = resolve_device(args.device)
    payload = json.loads(args.split_json.read_text(encoding="utf-8"))
    exs = payload["examples"]
    texts = [e["text"] for e in exs]
    y = np.array([1 if e["label"] == "negated" else 0 for e in exs])
    sp = np.array([e["split"] for e in exs])
    reports = np.array([str(e.get("report_id", e.get("example_id", i))) for i, e in enumerate(exs)])

    std = load_any_sae(std_ck, device=device)
    vreg = load_any_sae(vreg_ck, device=device)
    lm, tok = load_model_and_tokenizer(model_id, device, "auto", trust_remote_code=trc)
    hidden = last_token_hidden(lm, tok, texts, layer, device, batch_size=16).numpy()
    del lm, tok

    reps = {"hidden": hidden,
            "std_code": encode(std, torch.from_numpy(hidden), device),
            "vreg_code": encode(vreg, torch.from_numpy(hidden), device)}
    res = {"hidden": fit_eval(reps["hidden"], y, sp),
           "std_code": fit_eval(reps["std_code"], y, sp, return_test=True),
           "vreg_code": fit_eval(reps["vreg_code"], y, sp, return_test=True)}

    te = sp == "test"
    boot = clustered_bootstrap_delta(
        y[te], res["std_code"]["_p"], res["std_code"]["_yhat"],
        res["vreg_code"]["_p"], res["vreg_code"]["_yhat"], reports[te])
    for r in ("std_code", "vreg_code"):
        res[r].pop("_p", None); res[r].pop("_yhat", None)

    out = {"profile": args.profile, "split_json": str(args.split_json),
           "n": len(exs), "label_counts": {"negated": int(y.sum()), "affirmed": int((1 - y).sum())},
           "results": res, "vreg_vs_std_code_bootstrap": boot}

    cv = None
    if args.cv_folds and args.cv_folds > 1:
        p_std, yh_std = grouped_cv_predict(reps["std_code"], y, reports, args.cv_folds)
        p_vreg, yh_vreg = grouped_cv_predict(reps["vreg_code"], y, reports, args.cv_folds)
        p_hid, _ = grouped_cv_predict(reps["hidden"], y, reports, args.cv_folds)
        cv = {"n_folds": args.cv_folds,
              "hidden": {"auroc": round(float(roc_auc_score(y, p_hid)), 4)},
              "std_code": {"auroc": round(float(roc_auc_score(y, p_std)), 4),
                           "ba": round(float(balanced_accuracy_score(y, yh_std)), 4)},
              "vreg_code": {"auroc": round(float(roc_auc_score(y, p_vreg)), 4),
                            "ba": round(float(balanced_accuracy_score(y, yh_vreg)), 4)},
              "vreg_vs_std_bootstrap": clustered_bootstrap_delta(
                  y, p_std, yh_std, p_vreg, yh_vreg, reports)}
        out["grouped_cv"] = cv
    args.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    print(f"[{args.profile}] {args.split_json.name}  n={len(exs)}")
    print(f"  hidden    AUROC={res['hidden']['test_auroc']}  BA={res['hidden']['test_ba']}")
    print(f"  std_code  AUROC={res['std_code']['test_auroc']}  BA={res['std_code']['test_ba']}")
    print(f"  vreg_code AUROC={res['vreg_code']['test_auroc']}  BA={res['vreg_code']['test_ba']}")
    d_auc = res['vreg_code']['test_auroc'] - res['std_code']['test_auroc']
    d_ba = res['vreg_code']['test_ba'] - res['std_code']['test_ba']
    print(f"  Δ (V-reg - Std): AUROC {d_auc:+.4f}  BA {d_ba:+.4f}")
    print(f"  [fixed split] report-clustered bootstrap ({boot['n_reports']} reports): "
          f"ΔAUROC 95% CI {boot['delta_auroc_ci_lo_hi_mean'][:2]} "
          f"ΔBA 95% CI {boot['delta_ba_ci_lo_hi_mean'][:2]}")
    if cv is not None:
        cb = cv["vreg_vs_std_bootstrap"]
        print(f"  [grouped {cv['n_folds']}-fold CV, pooled OOF]  "
              f"hidden AUROC={cv['hidden']['auroc']}  "
              f"std AUROC={cv['std_code']['auroc']}/BA={cv['std_code']['ba']}  "
              f"vreg AUROC={cv['vreg_code']['auroc']}/BA={cv['vreg_code']['ba']}")
        print(f"  [CV] report-clustered bootstrap ({cb['n_reports']} reports): "
              f"ΔAUROC 95% CI {cb['delta_auroc_ci_lo_hi_mean'][:2]} "
              f"ΔBA 95% CI {cb['delta_ba_ci_lo_hi_mean'][:2]}")


if __name__ == "__main__":
    main()
