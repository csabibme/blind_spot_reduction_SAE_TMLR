#!/usr/bin/env python3
"""
MSE / forward-path / batch-invariance validation before full E1 runs.

Run this BEFORE the full audit to confirm:
  1. forward() == encode() + decode()
  2. batch vs singleton hidden states match (padding invariance)
  3. MSE, NMSE, explained variance, cosine similarity are sensible
  4. The denominator effect (z_orig_norm) is documented
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
    batch_invariance_test,
    clear_device_cache,
    collect_hidden_pairs,
    evaluate_sae_on_hidden,
    load_sae,
    module_param_info,
    setup_sae_scaling_imports,
    stable_family_seed,
    subsample_pairs,
    verify_forward_matches_encode_decode,
)
from shared.metrics import lower_fraction_mean, nmse  # noqa: E402
from shared.path_registry import (  # noqa: E402
    checkpoint_dir,
    load_manifest,
    pairs_path,
    sae_scaling_root,
)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Pre-flight validation: MSE + batch invariance"
    )
    p.add_argument("--profile", default="gpt2")
    p.add_argument("--checkpoint-id", default="gpt2_standard_joint16_owt")
    p.add_argument("--family", default="number_swap")
    p.add_argument("--max-pairs", type=int, default=10)
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
    pf = load_perturbation_families(pairs_path(manifest))
    fam_seed = stable_family_seed(args.seed, args.family)
    all_pairs = [tuple(x) for x in pf[args.family]["pairs"]]
    pairs, indices = subsample_pairs(all_pairs, args.max_pairs, fam_seed)
    layer = model_cfg["hf_hidden_state_index"]

    print(f"=== Validation: {args.profile} / {args.checkpoint_id} / {args.family} ===")
    print(f"  Pairs: {len(pairs)} (indices: {indices[:5]}{'...' if len(indices)>5 else ''})")

    # --- Load LM ---
    print(f"\n[1] Loading LM {model_cfg['model_id']} dtype={args.lm_dtype} ...")
    lm, tok = load_model_and_tokenizer(
        model_cfg["model_id"], args.device, dtype=args.lm_dtype,
        trust_remote_code=model_cfg.get("trust_remote_code", False),
    )
    lm.eval()
    lm_info = module_param_info(lm)
    print(f"    LM: {lm_info}")

    # --- Batch invariance test ---
    print("\n[2] Batch vs singleton invariance test ...")
    texts_mixed = [p[0] for p in pairs[:10]]
    bi = batch_invariance_test(lm, tok, texts_mixed, layer, args.device)
    print(f"    max_abs_diff:  {bi['max_abs_diff']:.2e}")
    print(f"    mean_abs_diff: {bi['mean_abs_diff']:.2e}")
    print(f"    relative_norm: {bi['relative_norm_diff']:.2e}")
    if bi["relative_norm_diff"] > 1e-3:
        print("    *** WARNING: batch/singleton mismatch > 1e-3 ***")

    # --- Collect hidden ---
    print("\n[3] Collecting hidden states ...")
    hidden = collect_hidden_pairs(lm, tok, pairs, layer, args.device)
    h_norm = hidden["h_orig"].float().norm(dim=-1)
    print(f"    h_orig shape: {tuple(hidden['h_orig'].shape)}")
    print(f"    ||h_orig|| mean={h_norm.mean():.2f}  min={h_norm.min():.2f}  max={h_norm.max():.2f}")
    del lm, tok
    clear_device_cache(args.device)

    # --- Load SAE ---
    ckpt = checkpoint_dir(args.checkpoint_id, manifest)
    print(f"\n[4] Loading SAE {args.checkpoint_id} ...")
    sae = load_sae(ckpt, args.device)
    sae.eval()
    sae_info = module_param_info(sae)
    print(f"    SAE: {sae_info}")

    # --- Forward == encode+decode ---
    max_diff = verify_forward_matches_encode_decode(sae, hidden["h_orig"][:1])
    print(f"\n[5] forward vs encode/decode max diff: {max_diff:.2e}")
    if max_diff > 1e-5:
        print("    *** WARNING: forward path differs from encode+decode ***")

    # --- Full evaluation ---
    print("\n[6] Evaluating SAE on hidden cache ...")
    result = evaluate_sae_on_hidden(sae, hidden)

    s = result["s"]
    g = result["g"]
    abs_dz = result["abs_dz"]
    z_norm = result["z_orig_norm"]
    decode_resp = result["decode_resp"]

    print(f"\n    --- Sensitivity ---")
    print(f"    s:  mean={s.mean():.6f}  L20={lower_fraction_mean(s):.6f}")
    print(f"    g:  mean={g.mean():.6f}  L20={lower_fraction_mean(g):.6f}")
    print(f"    |Δz| (absolute): mean={abs_dz.mean():.4f}  L20={lower_fraction_mean(abs_dz):.4f}")
    print(f"    ||z_orig||:      mean={z_norm.mean():.4f}")
    print(f"    decode_resp:     mean={decode_resp.mean():.6f}  L20={lower_fraction_mean(decode_resp):.6f}")

    print(f"\n    --- Reconstruction ---")
    print(f"    MSE:  orig={result['mse_orig_mean']:.6f}  pert={result['mse_pert_mean']:.6f}  mean={result['mse_pair_mean']:.6f}")
    print(f"    NMSE: {result['nmse']:.6f}")
    print(f"    Explained variance: {result['explained_variance']:.6f}")
    print(f"    Cosine sim: mean={result['cosine_sim_mean']:.6f}  min={result['cosine_sim_min']:.6f}")

    print(f"\n    --- Sparsity ---")
    sp = result["sparsity_all"]
    print(f"    L0={sp['L0_mean']:.1f}  density={sp['density_mean']:.4f}  "
          f"code_norm={sp['code_norm_mean']:.4f}  inactive_frac={sp['inactive_frac_mean']:.4f}")

    # --- Output JSON ---
    report = {
        "profile": args.profile,
        "checkpoint_id": args.checkpoint_id,
        "family": args.family,
        "n_pairs": len(pairs),
        "selected_pair_indices": indices,
        "lm_info": lm_info,
        "sae_info": sae_info,
        "batch_invariance": bi,
        "forward_encode_decode_max_diff": max_diff,
        "h_orig_norm_stats": {
            "mean": float(h_norm.mean()),
            "min": float(h_norm.min()),
            "max": float(h_norm.max()),
        },
        "sensitivity": {
            "s_mean": float(s.mean()),
            "s_L20": float(lower_fraction_mean(s)),
            "g_mean": float(g.mean()),
            "g_L20": float(lower_fraction_mean(g)),
            "abs_dz_mean": float(abs_dz.mean()),
            "abs_dz_L20": float(lower_fraction_mean(abs_dz)),
            "z_orig_norm_mean": float(z_norm.mean()),
            "decode_resp_mean": float(decode_resp.mean()),
            "decode_resp_L20": float(lower_fraction_mean(decode_resp)),
        },
        "reconstruction": {
            "mse_pair_mean": result["mse_pair_mean"],
            "nmse": result["nmse"],
            "explained_variance": result["explained_variance"],
            "cosine_sim_mean": result["cosine_sim_mean"],
            "cosine_sim_min": result["cosine_sim_min"],
        },
        "sparsity": sp,
    }

    if args.output_json:
        out = Path(args.output_json)
    else:
        out = E1_ROOT / "results" / f"validate_{args.profile}_{args.checkpoint_id.split('_', 1)[-1]}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n  Saved -> {out}")
    print("  DONE.")


if __name__ == "__main__":
    main()
