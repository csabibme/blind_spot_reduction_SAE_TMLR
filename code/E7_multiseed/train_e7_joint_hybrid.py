#!/usr/bin/env python3
"""E7 freeze_v1 multi-seed SAE training.

This is an E7-local copy of the OWT-hybrid training path. It reads frozen OWT and
true-last perturbation caches, then writes new checkpoints only under E7 output roots.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

REVISION_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = REVISION_ROOT.parent.parent
SAE_ROOT = REPO_ROOT / "SAE_scaling"
for path in (REVISION_ROOT, SAE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from lm_loader import resolve_device  # noqa: E402
from model_profiles import PROFILE_CHOICES, apply_profile_args  # noqa: E402
from sae_model_v2 import StandardSAE  # noqa: E402
from shared.path_registry import load_manifest  # noqa: E402
from v_gini_loss_v2 import v_gini_loss_multi_family  # noqa: E402

FREEZE_ID = "E7_FREEZE_V1"


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_owt_cache(path: Path) -> tuple[torch.Tensor, dict[str, Any]]:
    payload = torch.load(path.expanduser().resolve(), map_location="cpu", weights_only=False)
    acts = payload["activations"].float()
    meta = {key: value for key, value in payload.items() if key != "activations"}
    return acts, meta


def seed_everything(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def load_perturbation_cache(path: Path) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], dict[str, Any]]:
    resolved = path.expanduser().resolve()
    payload = torch.load(resolved, map_location="cpu", weights_only=False)
    required = {
        "cache",
        "profile",
        "model_id",
        "pairs_sha256",
        "extraction_protocol",
        "hidden_batch_size",
        "layer",
        "padding_side",
        "attention_mask_passed",
    }
    missing = required - payload.keys()
    if missing:
        raise KeyError(f"Missing perturbation cache fields: {sorted(missing)}")

    cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for family, pair in payload["cache"].items():
        h_orig, h_pert = pair
        cache[family] = (
            torch.as_tensor(h_orig, dtype=torch.float32, device="cpu"),
            torch.as_tensor(h_pert, dtype=torch.float32, device="cpu"),
        )
    return cache, payload


class FrozenPerturbationCache:
    def __init__(self, cache: dict[str, tuple[torch.Tensor, torch.Tensor]]):
        self._cache = cache
        self.family_names = list(cache)

    def sample_for_v_loss(
        self,
        n_families: int,
        n_pairs: int,
        rng: random.Random,
        device: str,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        selected = rng.sample(self.family_names, k=min(n_families, len(self.family_names)))
        h_orig_list, h_pert_list = [], []
        for family in selected:
            h_orig, h_pert = self._cache[family]
            count = min(n_pairs, len(h_orig))
            indices = rng.sample(range(len(h_orig)), k=count)
            h_orig_list.append(h_orig[indices].to(device))
            h_pert_list.append(h_pert[indices].to(device))
        return h_orig_list, h_pert_list


def validate_protocol(
    args: argparse.Namespace,
    cache: dict[str, tuple[torch.Tensor, torch.Tensor]],
    cache_meta: dict[str, Any],
    pairs_sha: str,
    d_in: int,
) -> None:
    manifest = load_manifest()
    model_cfg = manifest["models"][args.profile]
    expected_lambda_v = 0.2 if args.objective == "vreg" else 0.0
    n_pairs_total = sum(len(value[0]) for value in cache.values())
    checks = {
        "freeze_id": args.freeze_id == FREEZE_ID,
        "profile": cache_meta["profile"] == args.profile,
        "model_id": cache_meta["model_id"] == model_cfg["model_id"],
        "layer": cache_meta["layer"] == int(model_cfg["hf_hidden_state_index"]) == args.layer,
        "lambda_v": args.lambda_v == expected_lambda_v,
        "lambda_l1": args.l1_coeff == 0.001,
        "d_sae": args.d_sae == int(model_cfg["d_sae"]) == 4096,
        "steps": args.steps == 15000,
        "batch_size": args.batch_size == 64,
        "max_pairs": args.max_pairs == 50,
        "v_families_per_step": args.v_families_per_step == 16,
        "v_pairs_per_family": args.v_pairs_per_family == 8,
        "extraction_protocol": cache_meta["extraction_protocol"] == "true_last",
        "attention_mask_passed": cache_meta["attention_mask_passed"] is True,
        "hidden_batch_size": cache_meta["hidden_batch_size"] == 16,
        "pairs_sha256": cache_meta["pairs_sha256"] == pairs_sha,
        "n_families": len(cache) == 16,
        "n_pairs_total": n_pairs_total == 600,
        "finite_cache": all(torch.isfinite(v[0]).all() and torch.isfinite(v[1]).all() for v in cache.values()),
        "d_in": all(v[0].shape[1] == d_in and v[1].shape[1] == d_in for v in cache.values()),
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise ValueError(f"E7 protocol validation failed: {failed}")


def train(args: argparse.Namespace) -> Path:
    device = resolve_device(args.device)
    out_dir = Path(args.output_dir) / args.run_name
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    loader_generator = seed_everything(args.seed)
    rng = random.Random(args.seed)

    owt_path = Path(args.owt_cache).expanduser().resolve()
    acts, owt_meta = load_owt_cache(owt_path)
    d_in = acts.shape[1]
    pairs_path = Path(args.pairs_file).expanduser().resolve()
    pairs_sha = sha256_file(pairs_path)
    perturb_cache_path = Path(args.perturb_cache).expanduser().resolve()
    perturb_cache_sha = sha256_file(perturb_cache_path)
    cache_data, cache_meta = load_perturbation_cache(perturb_cache_path)
    validate_protocol(args, cache_data, cache_meta, pairs_sha, d_in)

    print("=" * 72)
    print(f"E7 FREEZE_V1 TRAINING: {args.run_name}")
    print(f"  profile={args.profile} objective={args.objective} seed={args.seed}")
    print(f"  model={args.model} layer={args.layer} lambda_v={args.lambda_v}")
    print(f"  owt_cache={owt_path}")
    print(f"  perturb_cache={perturb_cache_path}")
    print(f"  output={out_dir}")
    print("=" * 72)

    perturb_cache = FrozenPerturbationCache(cache_data)
    family_names = perturb_cache.family_names
    families_per_step = min(args.v_families_per_step, len(family_names))

    sae = StandardSAE(d_in, args.d_sae, device=device)
    opt = torch.optim.Adam(sae.parameters(), lr=args.lr)
    loader = DataLoader(
        TensorDataset(acts),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        generator=loader_generator,
    )

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
        loss_recon, meta = sae.loss(batch_h, args.l1_coeff)
        loss = loss_recon
        v_metric = 0.0
        if args.lambda_v > 0 and (step % args.v_every == 0 or step == 0):
            h_orig_list, h_pert_list = perturb_cache.sample_for_v_loss(
                families_per_step, args.v_pairs_per_family, rng, device,
            )
            loss_v = v_gini_loss_multi_family(
                sae,
                h_orig_list,
                h_pert_list,
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
                "recon": float(meta["recon"]),
                "l1": float(meta["l1"]),
                "v_gini_loss": v_metric,
            }
            history.append(rec)
            print(
                f"  step {step:5d} loss={rec['loss']:.4f} recon={rec['recon']:.4f} "
                f"l1={rec['l1']:.4f} v={v_metric:.4f}",
                flush=True,
            )

    checkpoint_meta = {
        "experiment": "E7_multiseed",
        "freeze_id": args.freeze_id,
        "run_name": args.run_name,
        "profile": args.profile,
        "model_id": load_manifest()["models"][args.profile]["model_id"],
        "objective": args.objective,
        "sae_architecture": "StandardSAE",
        "training_protocol": "e7_freeze_v1_true_last_frozen_perturb_cache_hybrid_owt",
        "owt_cache": str(owt_path),
        "owt_cache_sha256": sha256_file(owt_path),
        "owt_cache_meta": owt_meta,
        "perturb_cache": str(perturb_cache_path),
        "perturb_cache_sha256": perturb_cache_sha,
        "perturb_cache_meta": {key: value for key, value in cache_meta.items() if key != "cache"},
        "pairs_file": str(pairs_path),
        "pairs_sha256": pairs_sha,
        "lambda_v": args.lambda_v,
        "l1_coeff": args.l1_coeff,
        "d_sae": args.d_sae,
        "layer": args.layer,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "families": family_names,
        "v_families_per_step": families_per_step,
        "v_pairs_per_family": args.v_pairs_per_family,
        "normalize_by_input": args.normalize_by_input,
        "git_commit": git_commit(),
        "training_seed": args.seed,
        "seed_label": args.seed_label,
        "seed_controls": {
            "python": args.seed,
            "numpy": args.seed,
            "torch": args.seed,
            "dataloader": args.seed,
            "perturbation_sampler": args.seed,
        },
        "isolation_note": "E7 freeze_v1 local output; historical checkpoints are read-only references",
        "history": history,
    }
    sae.save_checkpoint(out_dir, checkpoint_meta)
    print(f"\nSaved checkpoint -> {out_dir}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Train E7 freeze_v1 Standard/V-reg SAE")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--profile", choices=PROFILE_CHOICES, required=True)
    parser.add_argument("--objective", choices=("standard", "vreg"), required=True)
    parser.add_argument("--seed-label", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--owt-cache", type=Path, required=True)
    parser.add_argument("--perturb-cache", type=Path, required=True)
    parser.add_argument("--pairs-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--freeze-id", default=FREEZE_ID)
    parser.add_argument("--lambda-v", type=float, required=True)
    parser.add_argument("--l1-coeff", type=float, default=1e-3)
    parser.add_argument("--d-sae", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=15000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--model", default=None)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-pairs", type=int, default=50)
    parser.add_argument("--v-families-per-step", type=int, default=16)
    parser.add_argument("--v-pairs-per-family", type=int, default=8)
    parser.add_argument("--v-every", type=int, default=1)
    parser.add_argument("--normalize-by-input", action="store_true")
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument("--log-every", type=int, default=500)
    args = parser.parse_args()
    apply_profile_args(args, smoke=False)
    train(args)


if __name__ == "__main__":
    main()
