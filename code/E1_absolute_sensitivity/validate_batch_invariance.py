#!/usr/bin/env python3
"""
Batch vs singleton hidden-state comparison at the E1-metric level.

This script goes beyond the norm-difference check in validate_mse.py:
it computes the full E1 metrics under both batch and singleton hidden-state
collection and reports whether the Standard–V-reg contrasts are stable.

Blocking criterion: if |ΔL20_batch - ΔL20_singleton| / |ΔL20_batch| > 0.05,
the batch pipeline cannot be trusted for this profile.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REVISION_ROOT = Path(__file__).resolve().parents[1]
E1_ROOT = Path(__file__).resolve().parent
for p in (REVISION_ROOT, E1_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from eval_core import (  # noqa: E402
    clear_device_cache,
    collect_hidden_pairs,
    evaluate_sae_on_hidden,
    load_sae,
    module_param_info,
    setup_sae_scaling_imports,
    stable_family_seed,
    subsample_pairs,
    summarize_pair_arrays,
)
from shared.metrics import lower_fraction_mean, paired_delta_summary  # noqa: E402
from shared.path_registry import (  # noqa: E402
    checkpoint_dir,
    load_manifest,
    pairs_path,
    sae_scaling_root,
)

PROFILE_RUNS = {
    "gpt2": ("gpt2_standard_joint16_owt", "gpt2_vreg_joint16_owt"),
    "gemma-2-2b": ("gemma-2-2b_standard_joint16_owt", "gemma-2-2b_vreg_joint16_owt"),
    "qwen-2.5-3b": ("qwen-2.5-3b_standard_joint16_owt", "qwen-2.5-3b_vreg_joint16_owt"),
}


def _key_metrics(std_sum: dict, vreg_sum: dict) -> dict[str, float]:
    delta = paired_delta_summary(std_sum, vreg_sum)
    return {
        "delta_L20_s": delta["delta_L20_s"],
        "delta_mean_s": delta["delta_mean_s"],
        "delta_L20_abs_dz": delta["delta_L20_abs_dz"],
        "delta_L20_decode_resp": delta["delta_L20_decode_resp"],
        "delta_L20_g": delta["delta_L20_g"],
        "std_nmse": std_sum["nmse_mean"],
        "vreg_nmse": vreg_sum["nmse_mean"],
        "std_L20_s": std_sum["s_L20"],
        "vreg_L20_s": vreg_sum["s_L20"],
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description="Batch vs singleton E1-metric invariance check"
    )
    p.add_argument("--profile", default="qwen-2.5-3b")
    p.add_argument("--family", default="number_swap")
    p.add_argument("--max-pairs", type=int, default=50)
    p.add_argument("--device", default="mps")
    p.add_argument("--lm-dtype", default="float16")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-json", default=None)
    args = p.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    manifest = load_manifest()
    setup_sae_scaling_imports(sae_scaling_root(manifest))

    from lm_loader import load_model_and_tokenizer
    from perturbation_data import load_perturbation_families

    model_cfg = manifest["models"][args.profile]
    std_id, vreg_id = PROFILE_RUNS[args.profile]
    pf = load_perturbation_families(pairs_path(manifest))
    fam_seed = stable_family_seed(args.seed, args.family)
    all_pairs = [tuple(x) for x in pf[args.family]["pairs"]]
    pairs, indices = subsample_pairs(all_pairs, args.max_pairs, fam_seed)
    layer = model_cfg["hf_hidden_state_index"]

    print(f"=== Batch vs Singleton E1-metric check ===")
    print(f"  Profile: {args.profile}, Family: {args.family}, Pairs: {len(pairs)}")

    # --- Load LM ---
    print(f"\n[1] Loading LM {model_cfg['model_id']} ...")
    lm, tok = load_model_and_tokenizer(
        model_cfg["model_id"], args.device, dtype=args.lm_dtype,
        trust_remote_code=model_cfg.get("trust_remote_code", False),
    )
    lm.eval()

    # --- Collect: full batch ---
    print("\n[2] Collecting hidden states — FULL BATCH ...")
    hidden_batch = collect_hidden_pairs(lm, tok, pairs, layer, args.device, batch_size=0)

    # --- Collect: singleton ---
    print("[3] Collecting hidden states — SINGLETON ...")
    hidden_single = collect_hidden_pairs(lm, tok, pairs, layer, args.device, batch_size=1)

    # --- Also test perturbed sentences ---
    from activations import last_token_hidden  # noqa: E402

    texts_p = [p[1] for p in pairs]
    h_p_batch = last_token_hidden(lm, tok, texts_p, layer, args.device).float().cpu()
    h_p_singles = []
    for t in texts_p:
        h_p_singles.append(last_token_hidden(lm, tok, [t], layer, args.device).float().cpu())
    h_p_single = torch.cat(h_p_singles, dim=0)
    pert_diff = (h_p_batch - h_p_single).abs()
    pert_rel = float(pert_diff.norm().item() / (h_p_single.norm().item() + 1e-8))

    del lm, tok
    clear_device_cache(args.device)

    # --- Hidden-state norm comparison ---
    orig_diff = (hidden_batch["h_orig"] - hidden_single["h_orig"]).abs()
    orig_rel = float(orig_diff.norm().item() / (hidden_single["h_orig"].norm().item() + 1e-8))
    print(f"\n  Hidden diff (original): relative_norm = {orig_rel:.2e}")
    print(f"  Hidden diff (perturbed): relative_norm = {pert_rel:.2e}")

    # --- Load SAEs and compute metrics for both hidden sets ---
    print(f"\n[4] Loading Standard SAE ...")
    std_sae = load_sae(checkpoint_dir(std_id, manifest), args.device)
    std_sae.eval()

    std_batch = evaluate_sae_on_hidden(std_sae, hidden_batch)
    std_single = evaluate_sae_on_hidden(std_sae, hidden_single)
    del std_sae
    clear_device_cache(args.device)

    print(f"[5] Loading V-reg SAE ...")
    vreg_sae = load_sae(checkpoint_dir(vreg_id, manifest), args.device)
    vreg_sae.eval()

    vreg_batch = evaluate_sae_on_hidden(vreg_sae, hidden_batch)
    vreg_single = evaluate_sae_on_hidden(vreg_sae, hidden_single)
    del vreg_sae
    clear_device_cache(args.device)

    # --- Compare E1 metrics ---
    std_sum_b = summarize_pair_arrays(std_batch)
    vreg_sum_b = summarize_pair_arrays(vreg_batch)
    std_sum_s = summarize_pair_arrays(std_single)
    vreg_sum_s = summarize_pair_arrays(vreg_single)

    metrics_batch = _key_metrics(std_sum_b, vreg_sum_b)
    metrics_single = _key_metrics(std_sum_s, vreg_sum_s)

    print(f"\n{'Metric':<25} {'Batch':>12} {'Singleton':>12} {'AbsDiff':>12} {'RelDiff%':>10}")
    print("-" * 73)
    diffs = {}
    for key in metrics_batch:
        vb = metrics_batch[key]
        vs = metrics_single[key]
        abs_d = abs(vb - vs)
        rel_d = abs_d / (abs(vb) + 1e-12) * 100
        diffs[key] = {"batch": vb, "singleton": vs, "abs_diff": abs_d, "rel_pct": rel_d}
        print(f"  {key:<23} {vb:>12.6f} {vs:>12.6f} {abs_d:>12.2e} {rel_d:>9.2f}%")

    # --- Blocking criterion ---
    primary = "delta_L20_s"
    rel_primary = diffs[primary]["rel_pct"]
    passed = rel_primary < 5.0
    print(f"\n  Primary metric ({primary}) batch-singleton drift: {rel_primary:.2f}%")
    print(f"  PASS: {'YES' if passed else 'NO — consider singleton or microbatch hidden collection'}")

    # --- Save JSON ---
    report = {
        "profile": args.profile,
        "family": args.family,
        "n_pairs": len(pairs),
        "selected_pair_indices": indices,
        "hidden_norm_diff": {
            "original_relative": orig_rel,
            "perturbed_relative": pert_rel,
        },
        "metrics_batch": metrics_batch,
        "metrics_singleton": metrics_single,
        "diffs": diffs,
        "primary_metric": primary,
        "primary_rel_drift_pct": rel_primary,
        "pass": passed,
    }

    if args.output_json:
        out = Path(args.output_json)
    else:
        out = E1_ROOT / "results" / f"batch_invariance_{args.profile}_{args.family}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n  Saved -> {out}")


if __name__ == "__main__":
    main()
