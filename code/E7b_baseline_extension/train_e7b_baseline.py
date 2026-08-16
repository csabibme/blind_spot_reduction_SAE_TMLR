#!/usr/bin/env python3
"""E7b freeze_v1 JumpReLU/MDL baseline training."""

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
from sae_model_v2 import JumpReLUSAE, MDLSAE  # noqa: E402
from shared.path_registry import load_manifest  # noqa: E402

FREEZE_ID = "E7B_BASELINE_FREEZE_V1"


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


def seed_everything(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def load_owt_cache(path: Path) -> tuple[torch.Tensor, dict[str, Any]]:
    payload = torch.load(path.expanduser().resolve(), map_location="cpu", weights_only=False)
    acts = payload["activations"].float()
    meta = {key: value for key, value in payload.items() if key != "activations"}
    return acts, meta


def load_perturb_cache_meta(path: Path) -> dict[str, Any]:
    payload = torch.load(path.expanduser().resolve(), map_location="cpu", weights_only=False)
    return {key: value for key, value in payload.items() if key != "cache"}


def validate_protocol(args: argparse.Namespace, owt_acts: torch.Tensor, perturb_meta: dict[str, Any], pairs_sha: str) -> None:
    manifest = load_manifest()
    model_cfg = manifest["models"][args.profile]
    checks = {
        "freeze_id": args.freeze_id == FREEZE_ID,
        "objective": args.objective in {"jumprelu", "mdl"},
        "profile": perturb_meta["profile"] == args.profile,
        "model_id": perturb_meta["model_id"] == model_cfg["model_id"],
        "layer": perturb_meta["layer"] == int(model_cfg["hf_hidden_state_index"]) == args.layer,
        "d_in": int(model_cfg["d_in"]) == int(owt_acts.shape[1]),
        "d_sae": args.d_sae == int(model_cfg["d_sae"]) == 4096,
        "steps": args.steps == 15000,
        "batch_size": args.batch_size == 64,
        "l1_coeff": args.l1_coeff == 0.001,
        "extraction_protocol": perturb_meta["extraction_protocol"] == "true_last",
        "attention_mask_passed": perturb_meta["attention_mask_passed"] is True,
        "hidden_batch_size": perturb_meta["hidden_batch_size"] == 16,
        "pairs_sha256": perturb_meta["pairs_sha256"] == pairs_sha,
        "n_families": perturb_meta["n_families"] == 16,
        "max_pairs": perturb_meta["max_pairs"] == 50,
    }
    failed = [key for key, ok in checks.items() if not ok]
    if failed:
        raise ValueError(f"E7b protocol validation failed: {failed}")


def build_sae(objective: str, d_in: int, d_sae: int, device: str):
    if objective == "jumprelu":
        return JumpReLUSAE(d_in, d_sae, device=device)
    if objective == "mdl":
        return MDLSAE(d_in, d_sae, device=device)
    raise ValueError(f"Unknown objective: {objective}")


def train(args: argparse.Namespace) -> Path:
    device = resolve_device(args.device)
    out_dir = Path(args.output_dir) / args.run_name
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    loader_generator = seed_everything(args.seed)
    pairs_path = Path(args.pairs_file).expanduser().resolve()
    pairs_sha = sha256_file(pairs_path)
    owt_path = Path(args.owt_cache).expanduser().resolve()
    acts, owt_meta = load_owt_cache(owt_path)
    perturb_path = Path(args.perturb_cache).expanduser().resolve()
    perturb_meta = load_perturb_cache_meta(perturb_path)
    validate_protocol(args, acts, perturb_meta, pairs_sha)

    print("=" * 72)
    print(f"E7B BASELINE TRAINING: {args.run_name}")
    print(f"  profile={args.profile} objective={args.objective} seed={args.seed}")
    print(f"  model={args.model} layer={args.layer}")
    print(f"  owt_cache={owt_path}")
    print(f"  perturb_cache={perturb_path}")
    print(f"  output={out_dir}")
    print("=" * 72)

    sae = build_sae(args.objective, acts.shape[1], args.d_sae, device)
    opt = torch.optim.Adam(sae.parameters(), lr=args.lr)
    loader = DataLoader(
        TensorDataset(acts),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        generator=loader_generator,
    )

    history = []
    data_iter = iter(loader)
    for step in range(1, args.steps + 1):
        try:
            (batch_h,) = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            (batch_h,) = next(data_iter)
        batch_h = batch_h.to(device=device, dtype=torch.float32)
        opt.zero_grad(set_to_none=True)
        loss, meta = sae.loss(batch_h, args.l1_coeff)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(sae.parameters(), args.grad_clip)
        opt.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            rec = {
                "step": step,
                "loss": float(loss.item()),
                "recon": float(meta["recon"]),
                "sparsity_term": float(meta["l1"]),
            }
            history.append(rec)
            print(
                f"  step {step:5d} loss={rec['loss']:.4f} recon={rec['recon']:.4f} "
                f"sparsity={rec['sparsity_term']:.4f}",
                flush=True,
            )

    checkpoint_meta = {
        "experiment": "E7b_baseline_extension",
        "freeze_id": args.freeze_id,
        "run_name": args.run_name,
        "profile": args.profile,
        "model_id": load_manifest()["models"][args.profile]["model_id"],
        "objective": args.objective,
        "sae_architecture": "JumpReLUSAE" if args.objective == "jumprelu" else "MDLSAE",
        "training_protocol": "e7b_freeze_v1_owt_hybrid_non_perturbation_baseline",
        "owt_cache": str(owt_path),
        "owt_cache_sha256": sha256_file(owt_path),
        "owt_cache_meta": owt_meta,
        "perturb_cache": str(perturb_path),
        "perturb_cache_sha256": sha256_file(perturb_path),
        "perturb_cache_meta": perturb_meta,
        "pairs_file": str(pairs_path),
        "pairs_sha256": pairs_sha,
        "l1_coeff": args.l1_coeff,
        "d_sae": args.d_sae,
        "layer": args.layer,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "git_commit": git_commit(),
        "training_seed": args.seed,
        "seed_label": args.seed_label,
        "seed_controls": {
            "python": args.seed,
            "numpy": args.seed,
            "torch": args.seed,
            "dataloader": args.seed,
        },
        "isolation_note": "E7b freeze_v1 local output; E7a Standard/V-reg are read-only comparators",
        "history": history,
    }
    sae.save_checkpoint(out_dir, checkpoint_meta)
    print(f"\nSaved checkpoint -> {out_dir}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Train E7b JumpReLU/MDL baseline SAE")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--profile", choices=PROFILE_CHOICES, required=True)
    parser.add_argument("--objective", choices=("jumprelu", "mdl"), required=True)
    parser.add_argument("--seed-label", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--owt-cache", type=Path, required=True)
    parser.add_argument("--perturb-cache", type=Path, required=True)
    parser.add_argument("--pairs-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--freeze-id", default=FREEZE_ID)
    parser.add_argument("--l1-coeff", type=float, default=1e-3)
    parser.add_argument("--d-sae", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=15000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--model", default=None)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument("--log-every", type=int, default=500)
    args = parser.parse_args()
    apply_profile_args(args, smoke=False)
    train(args)


if __name__ == "__main__":
    main()
