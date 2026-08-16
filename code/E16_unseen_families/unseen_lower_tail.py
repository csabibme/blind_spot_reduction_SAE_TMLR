#!/usr/bin/env python3
"""E16 — held-out family generalization audit (frozen joint-16, zero-shot).

Question (eFS4 §4): how sensitive are the conclusions to the choice of
perturbation family, and does the effect extend to broader classes of
representational blind spots?

Data: `experiment_100/data/pairs_100.json` — 100 families prepared BEFORE
reviewer feedback; the 84 template families were never used in any training
run submitted with the paper (held-out family kinds by construction).

Design: identical to the RC-LT audit (`dedicated_gpt2/openi_lower_tail.py`).
The frozen general joint-16 Standard and V-reg checkpoints (experiment_101,
zero-shot, not tuned on any of these families) are evaluated per family;
endpoint is ΔL20(s) with bootstrap CI.

Pre-registered tier rule (fixed before running, applied uniformly):
  - EXCLUDED: the 16 trained families themselves.
  - Tier A (unseen instances / lexicon of trained kinds): shard variants of
    trained kinds and content-word substitution families (semantic_*,
    medical_auto swaps, homophone/spelling/typo-like surface edits).
  - Tier B (kinds unseen in nature): operator-, inflection-, structure- and
    discourse-level transformations that are not content-word substitutions.
All families are reported regardless of outcome; a family is called positive
when its 95% CI excludes zero.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_external_root = os.environ.get("SAE_REPO_ROOT")
SAE_ROOT = (
    Path(_external_root).resolve()
    if _external_root
    else HERE.parents[1]
)
REV1 = SAE_ROOT / "FINAL/REVISION_1"
SAE_SCALING = SAE_ROOT / "SAE_scaling"
for _p in (REV1, REV1 / "E1_absolute_sensitivity", SAE_SCALING):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from eval_core import (  # noqa: E402
    collect_hidden_pairs, evaluate_sae_on_hidden, load_sae,
    stable_family_seed, subsample_pairs,
)
from shared.bootstrap import paired_delta_bootstrap_ci  # noqa: E402
from shared.metrics import gini_coefficient, lower_fraction_mean  # noqa: E402
from lm_loader import load_model_and_tokenizer, resolve_device  # noqa: E402

EXP100_PAIRS = SAE_ROOT / "FINAL/tmlr_revision/prepare/experiment_100/data/pairs_100.json"
EXP101 = SAE_ROOT / "FINAL/tmlr_revision/prepare/experiment_101_hybrid_owt/checkpoints"
GEMMA_VREG_OVERRIDE = (SAE_ROOT / "FINAL/REVISION_1/E1R_gemma_protocol_repair/checkpoints"
                       / "gemma-2-2b/joint/vreg_joint16_owt_true_last")

PROFILES = {
    "gpt2": ("gpt2", 12, False),
    "qwen-2.5-3b": ("Qwen/Qwen2.5-3B", 18, True),
    "gemma-2-2b": ("google/gemma-2-2b", 13, True),
}

TRAINED_16 = {
    "semantic_substitution", "typo", "negation", "synonym", "number_swap",
    "word_order", "unit_of_measure", "gender_swap", "drug_name_swap",
    "date_time_change", "body_part_swap", "severity_change",
    "frequency_change", "causal_reversal", "condition_negation",
    "anatomical_direction",
}

# Tier B: kinds unseen in nature (operator / inflection / structure / discourse)
TIER_B = {
    "modal_swap", "hedging_swap",            # epistemic / modality operators
    "tense_shift",                            # inflectional grammar
    "voice_swap", "clause_order",             # syntactic structure
    "coreference_swap",                       # discourse reference
    "conjunction_swap",                       # discourse connective
    "determiner_quantifier",                  # logical quantification
    "comparative_swap", "degree_intensifier", # graded operators
    "pluralization", "article_swap",          # grammatical morphology
    "preposition_swap", "adverb_swap",        # function-word grammar
    "contraction_expand", "punctuation_change",
    "capitalization", "polarity_flip",
}
# Everything else (not trained, not Tier B) is Tier A: shard variants and
# content-word substitution families on new lexicons/domains.

# Duplicate shard families in pairs_100.json, found by a SET-LEVEL (order-
# insensitive) content audit: each listed family contains exactly the same
# pair set as its retained representative, so it is the same measurement and
# is excluded to avoid double-counting in the macro. In particular all eight
# negation shards are one 8-pair set (shards differ only in pair order);
# negation_shard_00 is kept as the single negation measurement.
DUPLICATE_FAMILIES = {
    "negation_shard_01",      # set-equal to negation_shard_00
    "negation_shard_02",      # set-equal to negation_shard_00
    "negation_shard_03",      # set-equal to negation_shard_00 (also order-equal)
    "negation_shard_04",      # set-equal to negation_shard_00
    "negation_shard_05",      # set-equal to negation_shard_00
    "negation_shard_06",      # set-equal to negation_shard_00
    "negation_shard_07",      # set-equal to negation_shard_00
    "semantic_sub_shard_04",  # set-equal to semantic_sub_shard_00
    "synonym_shard_04",       # set-equal to synonym_shard_00
    "synonym_shard_05",       # set-equal to synonym_shard_01
    "synonym_shard_06",       # set-equal to synonym_shard_02
    "synonym_shard_07",       # set-equal to synonym_shard_03
}


def tier_of(name: str) -> str | None:
    if name in TRAINED_16 or name in DUPLICATE_FAMILIES:
        return None
    if name in TIER_B:
        return "B"
    return "A"


def d_l20(a, b):
    return float(lower_fraction_mean(b, 0.20) - lower_fraction_mean(a, 0.20))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True, choices=list(PROFILES))
    ap.add_argument("--tiers", nargs="*", default=["A", "B"])
    ap.add_argument("--max-pairs", type=int, default=30)
    ap.add_argument("--max-families", type=int, default=0,
                    help="0 = all held-out families")
    ap.add_argument("--max-length", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=16,
                    help="use 1 for gemma on MPS float16 (padding-sensitive)")
    ap.add_argument("--lm-dtype", default="float16")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    model_id, layer, trc = PROFILES[args.profile]
    std_ck = EXP101 / args.profile / "joint/standard_joint16_owt"
    vreg_ck = EXP101 / args.profile / "joint/vreg_joint16_owt"
    if args.profile == "gemma-2-2b" and GEMMA_VREG_OVERRIDE.is_dir():
        vreg_ck = GEMMA_VREG_OVERRIDE

    fams_raw = json.loads(EXP100_PAIRS.read_text(encoding="utf-8"))["families"]
    families = []
    for name in sorted(fams_raw):
        t = tier_of(name)
        if t in args.tiers:
            families.append((name, t))
    if args.max_families:
        families = families[: args.max_families]
    print(f"[{args.profile}] {len(families)} held-out families "
          f"(A={sum(1 for _, t in families if t == 'A')}, "
          f"B={sum(1 for _, t in families if t == 'B')})", flush=True)

    device = resolve_device(args.device)
    lm, tok = load_model_and_tokenizer(model_id, device, dtype=args.lm_dtype,
                                       trust_remote_code=trc)
    lm.eval()
    hidden = {}
    for i, (fam, _t) in enumerate(families, 1):
        pairs_all = [tuple(p) for p in fams_raw[fam]["pairs"]]
        fam_seed = stable_family_seed(args.seed, fam)
        pairs, _ = subsample_pairs(pairs_all, args.max_pairs, seed=fam_seed)
        print(f"  hidden {i}/{len(families)}: {fam} ({len(pairs)})", flush=True)
        hidden[fam] = collect_hidden_pairs(lm, tok, pairs, layer, device,
                                           batch_size=args.batch_size,
                                           extraction_protocol="true_last",
                                           max_length=args.max_length)
    del lm, tok

    std_sae = load_sae(std_ck, device)
    std_arr = {f: evaluate_sae_on_hidden(std_sae, hidden[f]) for f, _ in families}
    del std_sae
    vreg_sae = load_sae(vreg_ck, device)
    vreg_arr = {f: evaluate_sae_on_hidden(vreg_sae, hidden[f]) for f, _ in families}
    del vreg_sae

    fam_res = {}
    for fam, t in families:
        s_std, s_vr = std_arr[fam]["s"], vreg_arr[fam]["s"]
        clusters = np.arange(len(s_std))  # independent template pairs
        boot = paired_delta_bootstrap_ci(s_std, s_vr, d_l20, clusters,
                                         n_boot=5000, seed=100)
        ci = boot.get("cluster", boot["pair"])
        fam_res[fam] = {
            "tier": t,
            "n_pairs": int(len(s_std)),
            "L20_std": round(lower_fraction_mean(s_std, 0.20), 6),
            "L20_vreg": round(lower_fraction_mean(s_vr, 0.20), 6),
            "delta_L20_s": round(d_l20(s_std, s_vr), 6),
            "delta_L20_ci": [round(ci["lo"], 6), round(ci["hi"], 6)],
            "ci_excludes_zero": bool(ci["lo"] > 0 or ci["hi"] < 0),
            "V_gini_std": round(gini_coefficient(s_std), 4),
            "V_gini_vreg": round(gini_coefficient(s_vr), 4),
            "nmse_std": round(float(std_arr[fam]["nmse"]), 6),
            "nmse_vreg": round(float(vreg_arr[fam]["nmse"]), 6),
        }

    def macro(tier):
        rs = [r for r in fam_res.values() if r["tier"] == tier]
        if not rs:
            return None
        d = [r["delta_L20_s"] for r in rs]
        return {
            "n_families": len(rs),
            "mean_delta_L20_s": round(float(np.mean(d)), 6),
            "frac_positive": round(float(np.mean([x > 0 for x in d])), 4),
            "frac_ci_positive": round(float(np.mean(
                [r["delta_L20_s"] > 0 and r["ci_excludes_zero"] for r in rs])), 4),
        }

    out_path = args.out or HERE / f"results/unseen_lowertail_{args.profile}.json"
    out = {"profile": args.profile, "model_id": model_id, "layer": layer,
           "pairs_file": str(EXP100_PAIRS),
           "std_ckpt": str(std_ck), "vreg_ckpt": str(vreg_ck),
           "max_pairs": args.max_pairs, "seed": args.seed,
           "tier_rule": "pre-registered fixed lists in this script",
           "families": fam_res,
           "macro": {"tier_A": macro("A"), "tier_B": macro("B")}}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    print(f"\n=== {args.profile}: held-out family lower tail (ΔL20 of s) ===", flush=True)
    for tier in ("A", "B"):
        rows = [(f, r) for f, r in sorted(fam_res.items()) if r["tier"] == tier]
        if not rows:
            continue
        print(f"\n--- Tier {tier} ---", flush=True)
        for f, r in rows:
            star = "*" if r["ci_excludes_zero"] and r["delta_L20_s"] > 0 else " "
            print(f"{f:28} n={r['n_pairs']:>3} ΔL20={r['delta_L20_s']:+.5f}{star} "
                  f"[{r['delta_L20_ci'][0]:+.4f},{r['delta_L20_ci'][1]:+.4f}]",
                  flush=True)
        m = macro(tier)
        print(f"Tier {tier} macro: mean ΔL20={m['mean_delta_L20_s']:+.5f}  "
              f">0: {m['frac_positive']:.0%}  CI>0: {m['frac_ci_positive']:.0%}",
              flush=True)
    print(f"\nwrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
