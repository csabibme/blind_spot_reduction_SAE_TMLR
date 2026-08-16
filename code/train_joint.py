#!/usr/bin/env python3
"""
Joint V-regularised SAE training.

Differences from ``train_sae.py``:
  * uses ``PerturbationHiddenCache`` (LM forward only once per pair, not per step)
  * forces ALL families to participate in the V-loss every step (or N of K),
    instead of sampling a small subset (the default ``--v-families-per-step 3``
    that prevented Tab. 2 reproduction on GPT-2)
  * supports the merged 16-family pairs file produced by
    ``combine_perturbation_pairs.py`` for the auto-blindspot generalisation
    test

Usage examples
--------------

GPT-2 joint, 6 paper families, every family every step:
  python train_joint.py --profile gpt2 \
      --run-name vreg_joint6 \
      --pairs-file ../large_model_exp/data/perturbation_pairs.json \
      --lambda-v 0.1 --steps 5000 --device mps

GPT-2 joint on the 16-family superset (paper + automatic):
  python train_joint.py --profile gpt2 \
      --run-name vreg_joint16 \
      --pairs-file data/joint16_pairs.json \
      --lambda-v 0.2 --steps 6000 --device mps
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from activations import (
    build_text_corpus,
    cache_activations,
    load_cached_activations,
)
from lm_loader import load_model_and_tokenizer, resolve_device
from model_profiles import PROFILE_CHOICES, apply_profile_args, get_profile
from perturbation_data import load_perturbation_families
from perturbation_hidden_cache import PerturbationHiddenCache, get_family_names
from sae_model_v2 import StandardSAE
from v_gini_loss_v2 import v_gini_loss_multi_family


def train(args):
    device = resolve_device(args.device)
    out_dir = Path(args.output_dir) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"JOINT V-REG TRAINING: {args.run_name}")
    print(f"  profile={args.profile}  model={args.model}  layer={args.layer}")
    print(f"  pairs_file={args.pairs_file}")
    print(f"  lambda_v={args.lambda_v}  steps={args.steps}  d_sae={args.d_sae}")
    print("=" * 72)

    # 1. Activation cache for reconstruction
    cache_path = Path(args.activation_cache)
    trust_rc = getattr(args, "trust_remote_code", False)
    if not cache_path.is_file() or args.refresh_cache:
        print(f"\nBuilding activation cache -> {cache_path}")
        lm0, tok0 = load_model_and_tokenizer(
            args.model, device, args.dtype, trust_remote_code=trust_rc
        )
        texts = build_text_corpus(args.pairs_file, n_generic=args.n_corpus_sentences)
        cache_activations(lm0, tok0, texts, args.layer, device, cache_path, args.batch_size)
        del lm0
    acts = load_cached_activations(cache_path)
    d_in = acts.shape[1]
    print(f"  d_in (LM hidden)  = {d_in}")
    print(f"  activations cache = {acts.shape}")
    norms = torch.norm(acts, dim=-1)
    print(
        f"  activation L2: mean={norms.mean().item():.2f}  "
        f"std={norms.std().item():.2f}  max={norms.max().item():.2f}"
    )
    if args.act_clip > 0:
        scale = (args.act_clip / norms.clamp(min=args.act_clip)).unsqueeze(-1)
        n_clipped = (norms > args.act_clip).sum().item()
        acts = acts * scale
        print(
            f"  --act-clip {args.act_clip} active: {n_clipped}/{len(acts)} samples clipped"
        )

    # 2. Perturbation families + hidden cache (one LM pass per pair, total)
    pf = load_perturbation_families(args.pairs_file)
    family_names = get_family_names(pf)
    n_total_families = len(family_names)
    print(f"  families ({n_total_families}): " + ", ".join(family_names))

    lm, tok = load_model_and_tokenizer(
        args.model, device, args.dtype, trust_remote_code=trust_rc
    )
    perturb_cache = PerturbationHiddenCache(
        lm, tok, pf, args.layer, device, max_pairs=args.max_pairs
    )

    # 3. SAE + optimiser
    sae = StandardSAE(d_in, args.d_sae, device=device)
    opt = torch.optim.Adam(sae.parameters(), lr=args.lr)

    loader = DataLoader(
        TensorDataset(acts), batch_size=args.batch_size, shuffle=True, drop_last=True
    )

    # 4. V-loss family size: default = ALL families every step (joint)
    families_per_step = (
        n_total_families if args.v_families_per_step is None
        else min(args.v_families_per_step, n_total_families)
    )
    print(
        f"  V-loss: {families_per_step}/{n_total_families} families per step, "
        f"{args.v_pairs_per_family} pairs/family, normalize_by_input={args.normalize_by_input}"
    )

    rng = random.Random(args.seed)
    history = []
    step = 0
    data_iter = iter(loader)

    while step < args.steps:
        try:
            (batch_h,) = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            (batch_h,) = next(data_iter)

        batch_h = batch_h.to(device)
        opt.zero_grad(set_to_none=True)
        loss_recon_tensor, m = sae.loss(batch_h, args.l1_coeff)
        loss = loss_recon_tensor

        v_metric = 0.0
        if args.lambda_v > 0 and (step % args.v_every == 0 or step == 0):
            h_orig_list, h_pert_list = perturb_cache.sample_for_v_loss(
                families_per_step, args.v_pairs_per_family, rng, device
            )
            if args.act_clip > 0:
                def _clip(h):
                    n = torch.norm(h, dim=-1, keepdim=True)
                    return h * (args.act_clip / n.clamp(min=args.act_clip))
                h_orig_list = [_clip(h) for h in h_orig_list]
                h_pert_list = [_clip(h) for h in h_pert_list]
            loss_v = v_gini_loss_multi_family(
                sae, h_orig_list, h_pert_list,
                normalize_by_input=args.normalize_by_input,
            )
            loss = loss + args.lambda_v * loss_v
            v_metric = float(loss_v.item())

        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(sae.parameters(), args.grad_clip)
        opt.step()
        step += 1

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            rec = {
                "step": step,
                "loss": float(loss.item()),
                "recon": float(m["recon"]),
                "l1": float(m["l1"]),
                "v_gini_loss": v_metric,
            }
            history.append(rec)
            print(
                f"  step {step:5d}  loss={rec['loss']:.4f}  "
                f"recon={rec['recon']:.4f}  l1={rec['l1']:.4f}  "
                f"v={v_metric:.4f}"
            )

    meta = {
        "run_name": args.run_name,
        "sae_type": "standard",
        "lambda_v": args.lambda_v,
        "l1_coeff": args.l1_coeff,
        "d_sae": args.d_sae,
        "layer": args.layer,
        "steps": args.steps,
        "model": args.model,
        "pairs_file": str(args.pairs_file) if args.pairs_file else None,
        "families": family_names,
        "v_families_per_step": families_per_step,
        "v_pairs_per_family": args.v_pairs_per_family,
        "normalize_by_input": args.normalize_by_input,
        "history": history,
    }
    sae.save_checkpoint(out_dir, meta)
    print(f"\n  Saved checkpoint -> {out_dir}")
    return out_dir


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", required=True)
    p.add_argument("--lambda-v", type=float, default=0.1)
    p.add_argument("--l1-coeff", type=float, default=1e-3)
    p.add_argument("--d-sae", type=int, default=4096)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--profile", choices=PROFILE_CHOICES, default="gpt2")
    p.add_argument("--model", default=None)
    p.add_argument("--layer", type=int, default=None)
    p.add_argument("--dtype", default="auto")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default=None,
                   help="default: checkpoints/{profile}/joint")
    p.add_argument("--activation-cache", default=None)
    p.add_argument("--pairs-file", default=None,
                   help="Perturbation pairs JSON. Default: paper 6-family set.")
    p.add_argument("--n-corpus-sentences", type=int, default=2000)
    p.add_argument("--refresh-cache", action="store_true")
    p.add_argument("--max-pairs", type=int, default=50)
    p.add_argument("--v-families-per-step", type=int, default=None,
                   help="None=use ALL families every step (joint training)")
    p.add_argument("--v-pairs-per-family", type=int, default=8)
    p.add_argument("--v-every", type=int, default=1)
    p.add_argument("--normalize-by-input", action="store_true",
                   help="Use V-reg_norm (input-normalised gain) instead of raw V")
    p.add_argument("--grad-clip", type=float, default=0.0,
                   help="Max grad norm (0 = no clipping). Recommended 1.0 for Gemma "
                        "due to outlier activation dimensions.")
    p.add_argument("--act-clip", type=float, default=0.0,
                   help="Clip activations to L2 norm threshold per-sample (0 = off). "
                        "Optional activation clipping for unusually large hidden-state norms; suggest --act-clip 300.")
    p.add_argument("--log-every", type=int, default=200)
    args = p.parse_args()

    apply_profile_args(args, smoke=False)
    profile = get_profile(args.profile)
    if args.activation_cache is None:
        args.activation_cache = profile.activation_cache
    if args.output_dir is None:
        args.output_dir = f"checkpoints/{args.profile}/joint"

    train(args)


if __name__ == "__main__":
    main()
