#!/usr/bin/env python3
"""Rev3 pre-flight headroom screen for non-dosage perturbation families.

Purpose
-------
Before promising any main-text extrinsic claim, cheaply check whether a candidate
perturbation family gives a *sub-ceiling* held-out probe on real models — i.e. the hidden
state is informative (signal present) but the Standard SAE code loses some of it, leaving
headroom for V-reg to recover. This is the same "headroom gate" logic used for the rev2
stress test, applied per (family, model).

It reuses the frozen rev1 infrastructure (true-last extraction, SAE loaders, manifest
checkpoints) WITHOUT retraining or modifying any rev1 artifact.

Endpoint per (family, model, representation): held-out template probe test AUROC / balanced
accuracy, averaged over several template-held-out splits (features are extracted once and
re-split per seed).

Label definitions
-----------------
- severity_change: each pair is (milder, more-severe); side label = low(0) / high(1).
  A genuine ordinal-severity axis (analogous to the E3 affirmed/negated single-side probe).
- anatomical_direction: mixed axes (anterior/posterior, proximal/distal, ...). No single
  coherent binary pole, so we treat side 0 vs side 1 as an EXPLORATORY within-family
  direction contrast only; interpret with caution.

The gate is decided primarily on severity_change.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

REVISION3_DIR = Path(__file__).resolve().parent
REVISION1_ROOT = REVISION3_DIR.parents[1] / "REVISION_1"
E1_ROOT = REVISION1_ROOT / "E1_absolute_sensitivity"
for _path in (REVISION1_ROOT, E1_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from shared.path_registry import checkpoint_dir, load_manifest, pairs_path, sae_scaling_root  # noqa: E402
from eval_core import clear_device_cache, load_sae, setup_sae_scaling_imports  # noqa: E402

PROFILES = ("gpt2", "gemma-2-2b", "qwen-2.5-3b")
REPRESENTATIONS = ("hidden", "sae_standard_code", "sae_vreg_code")
C_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]
PROBE_SEED = 42
SPLIT_SEEDS = (0, 1, 2, 3, 4)
TRAIN_FRAC = 0.60
DEV_FRAC = 0.20

# Headroom gate thresholds (fixed before inspecting results).
HIDDEN_INFORMATIVE_MIN = 0.70   # signal must be present in the hidden state
STANDARD_SUBCEILING_MAX = 0.95  # Standard SAE code must leave room to improve


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_side_examples(pairs: list[list[str]]) -> list[dict[str, Any]]:
    """One sample per side. label = side index (0 = original pole, 1 = perturbed pole).

    For severity_change this is a genuine low(0)/high(1) severity axis. For
    anatomical_direction it is an exploratory side-0-vs-side-1 direction contrast.
    Pair index is the template group used for held-out splitting (both sides share it).
    """
    examples: list[dict[str, Any]] = []
    for pair_idx, pair in enumerate(pairs):
        if len(pair) != 2:
            continue
        for side, text in enumerate(pair):
            examples.append({"pair_idx": pair_idx, "side": side, "text": text, "label": side})
    return examples


@torch.no_grad()
def extract_features(
    profile: str,
    texts: list[str],
    manifest: dict[str, Any],
    device: str,
    lm_dtype: str,
    max_length: int,
    batch_size: int,
) -> dict[str, np.ndarray]:
    setup_sae_scaling_imports(sae_scaling_root(manifest))
    from lm_loader import load_model_and_tokenizer
    from activations import last_token_hidden_true_last

    model_cfg = manifest["models"][profile]
    layer = int(model_cfg["hf_hidden_state_index"])
    std_ckpt = checkpoint_dir(f"{profile}_standard_joint16_owt", manifest)
    vreg_ckpt = checkpoint_dir(f"{profile}_vreg_joint16_owt", manifest)

    lm, tok = load_model_and_tokenizer(
        model_cfg["model_id"],
        device=device,
        dtype=lm_dtype,
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
    )
    hidden = last_token_hidden_true_last(
        lm, tok, texts, layer, device, max_length=max_length, batch_size=batch_size
    ).float().cpu()
    if not torch.isfinite(hidden).all():
        raise ValueError(f"[{profile}] Non-finite hidden states; try --lm-dtype float32.")

    sae_std = load_sae(std_ckpt, device)
    sae_vreg = load_sae(vreg_ckpt, device)
    h = hidden.to(device)
    z_std = sae_std.encode(h).float().cpu().numpy()
    z_vreg = sae_vreg.encode(h).float().cpu().numpy()
    del lm, sae_std, sae_vreg
    clear_device_cache(device)

    return {
        "hidden": hidden.numpy(),
        "sae_standard_code": z_std,
        "sae_vreg_code": z_vreg,
    }


def pair_grouped_split(pair_indices: list[int], seed: int) -> dict[int, str]:
    rng = np.random.default_rng(seed)
    order = list(pair_indices)
    rng.shuffle(order)
    n = len(order)
    n_train = max(1, round(n * TRAIN_FRAC))
    n_dev = max(1, round(n * DEV_FRAC))
    assign: dict[int, str] = {}
    for pidx in order[:n_train]:
        assign[pidx] = "train"
    for pidx in order[n_train : n_train + n_dev]:
        assign[pidx] = "dev"
    for pidx in order[n_train + n_dev :]:
        assign[pidx] = "test"
    if "test" not in assign.values():
        # Guarantee a non-empty test split for tiny families.
        assign[order[-1]] = "test"
    return assign


def fit_and_eval(
    x: np.ndarray,
    y: np.ndarray,
    split: np.ndarray,
) -> dict[str, float] | None:
    train, dev, test = split == "train", split == "dev", split == "test"
    if test.sum() == 0 or len(np.unique(y[test])) < 2 or len(np.unique(y[train])) < 2:
        return None
    scaler = StandardScaler().fit(x[train])
    xt, xd, xs = scaler.transform(x[train]), scaler.transform(x[dev]), scaler.transform(x[test])
    best_c, best_score = C_GRID[0], -1.0
    for c in C_GRID:
        clf = LogisticRegression(C=c, solver="lbfgs", max_iter=20000,
                                 class_weight="balanced", random_state=PROBE_SEED)
        clf.fit(xt, y[train])
        if len(np.unique(y[dev])) < 2:
            score = balanced_accuracy_score(y[dev], clf.predict(xd))
        else:
            score = roc_auc_score(y[dev], clf.predict_proba(xd)[:, 1])
        if score > best_score:
            best_score, best_c = score, c
    clf = LogisticRegression(C=best_c, solver="lbfgs", max_iter=20000,
                             class_weight="balanced", random_state=PROBE_SEED)
    clf.fit(xt, y[train])
    prob = clf.predict_proba(xs)[:, 1]
    pred = clf.predict(xs)
    return {
        "auroc": float(roc_auc_score(y[test], prob)),
        "balanced_accuracy": float(balanced_accuracy_score(y[test], pred)),
        "selected_c": float(best_c),
        "n_test": int(test.sum()),
    }


def analyze_family(
    family: str,
    examples: list[dict[str, Any]],
    features_by_profile: dict[str, dict[str, np.ndarray]],
) -> dict[str, Any]:
    y = np.array([e["label"] for e in examples], dtype=np.int64)
    pair_of = np.array([e["pair_idx"] for e in examples], dtype=np.int64)
    unique_pairs = sorted(set(pair_of.tolist()))

    out: dict[str, Any] = {"family": family, "n_examples": len(examples),
                           "n_pairs": len(unique_pairs), "profiles": {}}
    for profile, feats in features_by_profile.items():
        rep_out: dict[str, Any] = {}
        for rep in REPRESENTATIONS:
            x = feats[rep]
            per_seed = []
            for seed in SPLIT_SEEDS:
                assign = pair_grouped_split(unique_pairs, seed)
                split = np.array([assign[p] for p in pair_of.tolist()])
                res = fit_and_eval(x, y, split)
                if res is not None:
                    per_seed.append(res)
            if per_seed:
                aurocs = [r["auroc"] for r in per_seed]
                bas = [r["balanced_accuracy"] for r in per_seed]
                rep_out[rep] = {
                    "auroc_mean": float(np.mean(aurocs)),
                    "auroc_sd": float(np.std(aurocs)),
                    "ba_mean": float(np.mean(bas)),
                    "n_splits": len(per_seed),
                    "n_test_sides": per_seed[0]["n_test"],
                }
            else:
                rep_out[rep] = None
        # Headroom gate (per profile): hidden informative AND standard sub-ceiling.
        h = rep_out.get("hidden")
        s = rep_out.get("sae_standard_code")
        v = rep_out.get("sae_vreg_code")
        gate = None
        if h and s and v:
            hidden_informative = h["auroc_mean"] >= HIDDEN_INFORMATIVE_MIN
            standard_subceiling = s["auroc_mean"] <= STANDARD_SUBCEILING_MAX
            delta_auroc = v["auroc_mean"] - s["auroc_mean"]
            gate = {
                "hidden_informative": bool(hidden_informative),
                "standard_subceiling": bool(standard_subceiling),
                "headroom": bool(hidden_informative and standard_subceiling),
                "delta_auroc_vreg_minus_standard": float(delta_auroc),
            }
        rep_out["gate"] = gate
        out["profiles"][profile] = rep_out
    return out


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = ["# Rev3 pre-flight headroom screen", ""]
    lines.append(f"Gate: hidden AUROC ≥ {HIDDEN_INFORMATIVE_MIN} (signal present) AND "
                 f"Standard SAE-code AUROC ≤ {STANDARD_SUBCEILING_MAX} (sub-ceiling headroom).")
    lines.append(f"Splits per point: {len(SPLIT_SEEDS)} template-held-out (pair-grouped) splits, mean reported.")
    lines.append("")
    for fam, fam_res in payload["families"].items():
        lines.append(f"## {fam} (n_pairs={fam_res['n_pairs']}, n_sides={fam_res['n_examples']})")
        lines.append("")
        lines.append("| Model | hidden AUROC | Standard code AUROC | V-reg code AUROC | Δ AUROC (V−S) | headroom? |")
        lines.append("|---|---:|---:|---:|---:|:--:|")
        for profile, rep_out in fam_res["profiles"].items():
            def cell(rep: str) -> str:
                r = rep_out.get(rep)
                return f"{r['auroc_mean']:.3f}" if r else "—"
            gate = rep_out.get("gate")
            if gate:
                delta = f"{gate['delta_auroc_vreg_minus_standard']:+.3f}"
                hr = "yes" if gate["headroom"] else "no"
            else:
                delta, hr = "—", "—"
            lines.append(f"| {profile} | {cell('hidden')} | {cell('sae_standard_code')} | "
                         f"{cell('sae_vreg_code')} | {delta} | {hr} |")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Rev3 pre-flight headroom screen")
    parser.add_argument("--families", nargs="+", default=["severity_change", "anatomical_direction"])
    parser.add_argument("--profiles", nargs="+", default=list(PROFILES))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--lm-dtype", default="float16")
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--out-json", type=Path, default=REVISION3_DIR / "results" / "preflight_headroom.json")
    parser.add_argument("--cache-dir", type=Path, default=REVISION3_DIR / "results" / "feature_cache")
    args = parser.parse_args()

    manifest = load_manifest()
    device = resolve_device(args.device)
    all_pairs = json.loads(pairs_path(manifest).read_text(encoding="utf-8"))["families"]

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)

    families_out: dict[str, Any] = {}
    for family in args.families:
        if family not in all_pairs:
            print(f"[skip] family not found: {family}", flush=True)
            continue
        pairs = all_pairs[family]["pairs"]
        examples = build_side_examples(pairs)
        texts = [e["text"] for e in examples]

        features_by_profile: dict[str, dict[str, np.ndarray]] = {}
        for profile in args.profiles:
            cache_path = args.cache_dir / f"{profile}_{family}.npz"
            if cache_path.is_file():
                cached = np.load(cache_path)
                features_by_profile[profile] = {r: cached[r] for r in REPRESENTATIONS}
                print(f"[{profile}/{family}] loaded cache", flush=True)
                continue
            print(f"[{profile}/{family}] extracting features ({len(texts)} texts)...", flush=True)
            feats = extract_features(profile, texts, manifest, device, args.lm_dtype,
                                     args.max_length, args.batch_size)
            np.savez_compressed(cache_path, **feats)
            features_by_profile[profile] = feats
            print(f"[{profile}/{family}] saved cache -> {cache_path}", flush=True)

        families_out[family] = analyze_family(family, examples, features_by_profile)

    payload = {
        "experiment": "rev3_preflight_headroom",
        "gate": {
            "hidden_informative_min": HIDDEN_INFORMATIVE_MIN,
            "standard_subceiling_max": STANDARD_SUBCEILING_MAX,
        },
        "split_seeds": list(SPLIT_SEEDS),
        "families": families_out,
    }
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown(args.out_json.with_suffix(".md"), payload)
    print(f"Wrote {args.out_json}")


if __name__ == "__main__":
    main()
