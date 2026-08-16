#!/usr/bin/env python3
"""
OWT-cache reconstruction control.

Evaluates reconstruction quality (MSE, NMSE, EV, cosine) of Standard and V-reg
SAEs on OWT hidden-state samples — the same distribution they were trained on.

Purpose: determine whether high perturbation-sentence MSE is
(a) a domain/family-specific trade-off, or
(b) a general property of the V-reg checkpoint.

Uses the pre-cached OWT activations from experiment_101_hybrid_owt/data/.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REVISION_ROOT = Path(__file__).resolve().parents[1]
E1_ROOT = Path(__file__).resolve().parent
for p in (REVISION_ROOT, E1_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from eval_core import (  # noqa: E402
    clear_device_cache,
    load_sae,
    module_param_info,
    setup_sae_scaling_imports,
)
from shared.metrics import (  # noqa: E402
    code_sparsity_stats,
    cosine_similarity_batch,
    explained_variance,
    nmse,
)
from shared.path_registry import (  # noqa: E402
    checkpoint_dir,
    load_manifest,
    repo_root,
    sae_scaling_root,
)

OWT_CACHE_MAP = {
    "gpt2": "owt_cache_gpt2_l12_25k.pt",
    "gemma-2-2b": "owt_cache_gemma_l13_25k.pt",
    "qwen-2.5-3b": "owt_cache_qwen_l18_25k.pt",
}

PROFILE_RUNS = {
    "gpt2": ("gpt2_standard_joint16_owt", "gpt2_vreg_joint16_owt"),
    "gemma-2-2b": ("gemma-2-2b_standard_joint16_owt", "gemma-2-2b_vreg_joint16_owt"),
    "qwen-2.5-3b": ("qwen-2.5-3b_standard_joint16_owt", "qwen-2.5-3b_vreg_joint16_owt"),
}


@torch.no_grad()
def evaluate_reconstruction(sae, h: torch.Tensor) -> dict[str, float]:
    """Reconstruction metrics on a batch of hidden states."""
    sae_dev = next(sae.parameters()).device
    h_dev = h.to(sae_dev)
    x_hat, z = sae(h_dev)

    x_hat_f = x_hat.float()
    h_f = h_dev.float()

    mse_val = float(F.mse_loss(x_hat_f, h_f).item())

    x_hat_np = x_hat_f.cpu().numpy()
    h_np = h.float().numpy()
    z_np = z.float().cpu().numpy()

    nmse_val = nmse(h_np, x_hat_np)
    ev_val = explained_variance(h_np, x_hat_np)
    cos_arr = cosine_similarity_batch(h_np, x_hat_np)
    sp = code_sparsity_stats(z_np)

    return {
        "mse": mse_val,
        "nmse": nmse_val,
        "explained_variance": ev_val,
        "cosine_sim_mean": float(cos_arr.mean()),
        "cosine_sim_min": float(cos_arr.min()),
        "cosine_sim_q05": float(np.quantile(cos_arr, 0.05)),
        "L0_mean": sp["L0_mean"],
        "code_norm_mean": sp["code_norm_mean"],
        "density_mean": sp["density_mean"],
        "inactive_frac_mean": sp["inactive_frac_mean"],
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description="OWT reconstruction control for Standard vs V-reg SAEs"
    )
    p.add_argument("--profile", default="qwen-2.5-3b",
                   choices=list(OWT_CACHE_MAP.keys()))
    p.add_argument("--n-samples", type=int, default=1000,
                   help="Number of OWT activation vectors to evaluate")
    p.add_argument("--device", default="mps")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-json", default=None)
    p.add_argument("--std-checkpoint-override", type=Path, default=None)
    p.add_argument("--vreg-checkpoint-override", type=Path, default=None)
    args = p.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    manifest = load_manifest()
    setup_sae_scaling_imports(sae_scaling_root(manifest))

    # --- Load OWT cache ---
    owt_dir = (
        repo_root()
        / "FINAL" / "tmlr_revision" / "prepare" / "experiment_101_hybrid_owt" / "data"
    )
    cache_file = owt_dir / OWT_CACHE_MAP[args.profile]
    print(f"=== OWT Reconstruction Control: {args.profile} ===")
    print(f"  Cache: {cache_file}")

    payload = torch.load(cache_file, map_location="cpu", weights_only=False)
    all_acts = payload["activations"].float()
    print(f"  Total cached: {all_acts.shape[0]} vectors, d={all_acts.shape[1]}")

    rng = np.random.default_rng(args.seed)
    n = min(args.n_samples, all_acts.shape[0])
    idx = rng.choice(all_acts.shape[0], size=n, replace=False)
    h_sample = all_acts[idx]
    print(f"  Sampled: {n} vectors (seed={args.seed})")
    print(f"  ||h|| mean={h_sample.norm(dim=-1).mean():.2f}  "
          f"std={h_sample.norm(dim=-1).std():.2f}")

    std_id, vreg_id = PROFILE_RUNS[args.profile]
    std_ckpt = args.std_checkpoint_override or checkpoint_dir(std_id, manifest)
    vreg_ckpt = args.vreg_checkpoint_override or checkpoint_dir(vreg_id, manifest)

    # --- Standard SAE ---
    print(f"\n[1] Standard SAE: {std_id}")
    print(f"    path={std_ckpt}")
    std_sae = load_sae(std_ckpt, args.device)
    std_sae.eval()
    print(f"    {module_param_info(std_sae)}")
    std_metrics = evaluate_reconstruction(std_sae, h_sample)
    del std_sae
    clear_device_cache(args.device)
    print(f"    MSE={std_metrics['mse']:.6f}  NMSE={std_metrics['nmse']:.6f}  "
          f"EV={std_metrics['explained_variance']:.6f}  "
          f"cos={std_metrics['cosine_sim_mean']:.6f}")
    print(f"    L0={std_metrics['L0_mean']:.1f}  density={std_metrics['density_mean']:.4f}  "
          f"code_norm={std_metrics['code_norm_mean']:.2f}")

    # --- V-reg SAE ---
    vreg_label = args.vreg_checkpoint_override.name if args.vreg_checkpoint_override else vreg_id
    print(f"\n[2] V-reg SAE: {vreg_label}")
    print(f"    path={vreg_ckpt}")
    vreg_sae = load_sae(vreg_ckpt, args.device)
    vreg_sae.eval()
    print(f"    {module_param_info(vreg_sae)}")
    vreg_metrics = evaluate_reconstruction(vreg_sae, h_sample)
    del vreg_sae
    clear_device_cache(args.device)
    print(f"    MSE={vreg_metrics['mse']:.6f}  NMSE={vreg_metrics['nmse']:.6f}  "
          f"EV={vreg_metrics['explained_variance']:.6f}  "
          f"cos={vreg_metrics['cosine_sim_mean']:.6f}")
    print(f"    L0={vreg_metrics['L0_mean']:.1f}  density={vreg_metrics['density_mean']:.4f}  "
          f"code_norm={vreg_metrics['code_norm_mean']:.2f}")

    # --- Comparison ---
    mse_ratio = vreg_metrics["mse"] / (std_metrics["mse"] + 1e-12)
    nmse_ratio = vreg_metrics["nmse"] / (std_metrics["nmse"] + 1e-12)

    print(f"\n  --- Comparison ---")
    print(f"  MSE ratio (V-reg/Std):  {mse_ratio:.2f}x")
    print(f"  NMSE ratio:             {nmse_ratio:.2f}x")
    print(f"  EV drop:                {std_metrics['explained_variance'] - vreg_metrics['explained_variance']:.6f}")
    print(f"  Cos drop:               {std_metrics['cosine_sim_mean'] - vreg_metrics['cosine_sim_mean']:.6f}")

    # --- Save ---
    report = {
        "profile": args.profile,
        "owt_cache": str(cache_file),
        "n_samples": n,
        "seed": args.seed,
        "h_norm_mean": float(h_sample.norm(dim=-1).mean()),
        "checkpoint_standard": (
            args.std_checkpoint_override.name if args.std_checkpoint_override else std_id
        ),
        "checkpoint_vreg": vreg_label,
        "checkpoint_standard_manifest_id": std_id,
        "checkpoint_vreg_manifest_id": vreg_id,
        "checkpoint_standard_override_used": args.std_checkpoint_override is not None,
        "checkpoint_vreg_override_used": args.vreg_checkpoint_override is not None,
        "checkpoint_standard_path": str(std_ckpt),
        "checkpoint_vreg_path": str(vreg_ckpt),
        "standard": std_metrics,
        "vreg": vreg_metrics,
        "comparison": {
            "mse_ratio": mse_ratio,
            "nmse_ratio": nmse_ratio,
            "ev_drop": std_metrics["explained_variance"] - vreg_metrics["explained_variance"],
            "cos_drop": std_metrics["cosine_sim_mean"] - vreg_metrics["cosine_sim_mean"],
        },
    }

    if args.output_json:
        out = Path(args.output_json)
    else:
        out = E1_ROOT / "results" / f"owt_recon_{args.profile}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n  Saved -> {out}")


if __name__ == "__main__":
    main()
