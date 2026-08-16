#!/usr/bin/env python3
"""Dedicated paired GPT-2 SAE training (Standard vs V-reg) for REVISION_3.

Isolated: writes ONLY under this folder's checkpoints/ (explicit output-dir,
unique run-names, refuses a non-empty output dir). Reuses the experiment_101
OWT activation cache READ-ONLY for reconstruction, and the frozen phase2 helper
modules for the SAE / V-Gini loss / perturbation cache. Nothing here touches
manifest.yaml, experiment_101, or any phase checkpoint.

Paired common-randomness (the fix that made exp_toy causally clean):
  - both arms share the SAME weight initialisation (global torch seed reset
    identically before each SAE is built);
  - both arms share the SAME reconstruction minibatch stream (DataLoader driven
    by a generator seeded identically per arm);
  - only the V-regularised arm draws perturbation-pair samples, from an
    independent random.Random, so the difference between arms is the V term,
    not optimiser noise.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import common  # noqa: F401  (sets up sys.path to phase2 scripts)

import torch
from torch.utils.data import DataLoader, TensorDataset

from lm_loader import load_model_and_tokenizer, resolve_device
from perturbation_data import load_perturbation_families
from perturbation_hidden_cache import PerturbationHiddenCache, get_family_names
from sae_model_v2 import StandardSAE
from v_gini_loss_v2 import v_gini_loss_multi_family

HERE = Path(__file__).resolve().parent


def load_owt_cache(path: Path) -> tuple[torch.Tensor, dict]:
    data = torch.load(path.expanduser().resolve(), map_location="cpu", weights_only=False)
    acts = data["activations"]
    meta = {k: data[k] for k in data if k != "activations"}
    return acts, meta


def train_one_arm(
    arm: str,
    lambda_v: float,
    acts: torch.Tensor,
    perturb_cache: PerturbationHiddenCache,
    family_names: list[str],
    out_dir: Path,
    args: argparse.Namespace,
    device: str,
) -> dict:
    d_in = acts.shape[1]

    # Identical initialisation across arms.
    torch.manual_seed(args.seed)
    sae = StandardSAE(d_in, args.d_sae, device=device)
    opt = torch.optim.Adam(sae.parameters(), lr=args.lr)

    # Identical minibatch stream across arms (generator-driven shuffle).
    gen = torch.Generator()
    gen.manual_seed(args.seed)
    loader = DataLoader(
        TensorDataset(acts), batch_size=args.batch_size,
        shuffle=True, drop_last=True, generator=gen,
    )

    # Independent stream for perturbation-pair sampling (only used by V-reg).
    rng = random.Random(args.seed)

    n_fams = len(family_names)
    fams_per_step = n_fams if args.v_families_per_step is None else min(args.v_families_per_step, n_fams)

    print(f"\n[{arm}] lambda_v={lambda_v}  d_sae={args.d_sae}  steps={args.steps}  "
          f"V-loss {fams_per_step}/{n_fams} fam x {args.v_pairs_per_family} pairs")

    history = []
    step = 0
    data_iter = iter(loader)
    while step < args.steps:
        try:
            (batch_h,) = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            (batch_h,) = next(data_iter)
        batch_h = batch_h.to(device=device, dtype=torch.float32)

        opt.zero_grad(set_to_none=True)
        loss, m = sae.loss(batch_h, args.l1_coeff)

        v_metric = 0.0
        if lambda_v > 0 and (step % args.v_every == 0 or step == 0):
            h_o_list, h_p_list = perturb_cache.sample_for_v_loss(
                fams_per_step, args.v_pairs_per_family, rng, device
            )
            loss_v = v_gini_loss_multi_family(sae, h_o_list, h_p_list, normalize_by_input=False)
            loss = loss + lambda_v * loss_v
            v_metric = float(loss_v.item())

        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(sae.parameters(), args.grad_clip)
        opt.step()
        step += 1

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            rec = {"step": step, "loss": float(loss.item()),
                   "recon": float(m["recon"]), "l1": float(m["l1"]),
                   "v_gini_loss": v_metric}
            history.append(rec)
            print(f"  step {step:5d}  loss={rec['loss']:.4f}  recon={rec['recon']:.4f}  "
                  f"l1={rec['l1']:.4f}  v={v_metric:.4f}")

    meta = {
        "run_name": out_dir.name,
        "arm": arm,
        "experiment": "REVISION_3/dedicated_gpt2",
        "training_protocol": "paired_owt_recon_negation_v_loss",
        "paired_common_randomness": True,
        "seed": args.seed,
        "lambda_v": lambda_v,
        "l1_coeff": args.l1_coeff,
        "d_sae": args.d_sae,
        "layer": args.layer,
        "steps": args.steps,
        "model": args.model,
        "owt_cache": str(Path(args.owt_cache).resolve()),
        "pairs_file": str(Path(args.pairs_file).resolve()),
        "families": family_names,
        "v_pairs_per_family": args.v_pairs_per_family,
        "history": history,
    }
    sae.save_checkpoint(out_dir, meta)
    print(f"  saved -> {out_dir}")
    return {"arm": arm, "final": history[-1] if history else None, "out_dir": str(out_dir)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--owt-cache", type=Path,
                   default=HERE.parents[1] / "tmlr_revision/prepare/experiment_101_hybrid_owt/data/owt_cache_gpt2_l12_25k.pt")
    p.add_argument("--pairs-file", type=Path, default=HERE / "data/negation_vtrain_pairs.json")
    p.add_argument("--out-root", type=Path, default=HERE / "checkpoints")
    p.add_argument("--run-tag", default="negation")
    p.add_argument("--model", default="gpt2")
    p.add_argument("--layer", type=int, default=12)
    p.add_argument("--dtype", default="auto")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lambda-v", type=float, default=0.2)
    p.add_argument("--l1-coeff", type=float, default=1e-3)
    p.add_argument("--d-sae", type=int, default=4096)
    p.add_argument("--steps", type=int, default=15000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-pairs", type=int, default=200)
    p.add_argument("--v-families-per-step", type=int, default=None)
    p.add_argument("--v-pairs-per-family", type=int, default=32)
    p.add_argument("--v-every", type=int, default=1)
    p.add_argument("--grad-clip", type=float, default=0.0)
    p.add_argument("--log-every", type=int, default=1000)
    p.add_argument("--allow-existing", action="store_true",
                   help="overwrite (default: refuse non-empty arm dirs)")
    args = p.parse_args()

    device = resolve_device(args.device)
    out_std = args.out_root / f"{args.run_tag}_standard"
    out_vreg = args.out_root / f"{args.run_tag}_vreg"
    for d in (out_std, out_vreg):
        if d.exists() and any(d.iterdir()) and not args.allow_existing:
            raise FileExistsError(f"Refusing to overwrite non-empty dir: {d} (use --allow-existing)")
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"DEDICATED PAIRED TRAINING (device={device})")
    print(f"  owt_cache = {args.owt_cache}")
    print(f"  pairs     = {args.pairs_file}")
    print(f"  out       = {out_std}  |  {out_vreg}")
    print("=" * 72)

    acts, owt_meta = load_owt_cache(Path(args.owt_cache))
    print(f"[owt] shape={tuple(acts.shape)} meta_keys={list(owt_meta)[:6]}")

    pf = load_perturbation_families(str(args.pairs_file))
    family_names = get_family_names(pf)
    lm, tok = load_model_and_tokenizer(args.model, device, args.dtype,
                                       trust_remote_code=args.trust_remote_code)
    perturb_cache = PerturbationHiddenCache(lm, tok, pf, args.layer, device, max_pairs=args.max_pairs)
    del lm, tok

    results = []
    results.append(train_one_arm("standard", 0.0, acts, perturb_cache, family_names, out_std, args, device))
    results.append(train_one_arm("vreg", args.lambda_v, acts, perturb_cache, family_names, out_vreg, args, device))

    summary_path = args.out_root / f"{args.run_tag}_train_summary.json"
    summary_path.write_text(json.dumps({
        "device": device, "seed": args.seed, "steps": args.steps,
        "lambda_v": args.lambda_v, "d_sae": args.d_sae,
        "owt_cache": str(Path(args.owt_cache).resolve()),
        "arms": results,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {summary_path}")


if __name__ == "__main__":
    main()
