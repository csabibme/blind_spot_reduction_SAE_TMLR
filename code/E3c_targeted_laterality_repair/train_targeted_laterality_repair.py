#!/usr/bin/env python3
"""Targeted laterality repair pilot.

This is the symmetry-matched follow-up to E3b laterality: use only OpenI train-report
left/right pairs as the repair family, keep OWT reconstruction pressure, and write a new SAE
checkpoint without touching the legacy checkpoints.
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
E3_ROOT = REVISION_ROOT / "E3_heldout_probes"
for path in (SAE_ROOT, REVISION_ROOT, E3_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from sae_model_v2 import StandardSAE  # noqa: E402
from run_e3_negation_probes import feature_cache_path  # noqa: E402
from shared.path_registry import checkpoint_dir, load_manifest  # noqa: E402
from v_gini_loss_v2 import v_gini_loss_from_codes  # noqa: E402


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_owt_cache(path: Path) -> tuple[torch.Tensor, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
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


def load_laterality_pairs_by_split(
    split_json: Path,
    feature_cache_npz: Path,
    profile: str,
    model_id: str,
    layer: int,
) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], dict[str, Any]]:
    split = read_json(split_json)
    examples = split["examples"]
    split_sha = sha256_file(split_json)
    texts = [example["text"] for example in examples]
    expected_cache_path = feature_cache_path(feature_cache_npz.parent, profile, split_sha, texts)
    if expected_cache_path.name != feature_cache_npz.name:
        raise ValueError(
            "Feature cache filename does not match split SHA/text order digest: "
            f"expected {expected_cache_path.name}, got {feature_cache_npz.name}"
        )
    with np.load(feature_cache_npz, allow_pickle=True) as data:
        hidden = torch.as_tensor(data["hidden"], dtype=torch.float32)
        cache_meta = data["metadata"].item()
    required_meta = {
        "profile": profile,
        "task_split_sha256": split_sha,
        "model_id": model_id,
        "layer": layer,
        "extraction_protocol": "true_last",
    }
    for key, expected in required_meta.items():
        if cache_meta.get(key) != expected:
            raise ValueError(f"Feature cache metadata mismatch for {key}: {cache_meta.get(key)!r} != {expected!r}")
    if len(examples) != hidden.shape[0]:
        raise ValueError(f"Example/cache mismatch: {len(examples)} examples vs {hidden.shape[0]} hidden rows")
    example_id_sha256 = sha256_text("\n".join(example["example_id"] for example in examples))
    text_sha256 = sha256_text("\n".join(texts))
    by_split_pair: dict[str, dict[str, dict[str, int]]] = {
        "train": {},
        "dev": {},
        "test": {},
    }
    for index, example in enumerate(examples):
        entry = by_split_pair[example["split"]].setdefault(example["global_pair_id"], {})
        entry[example["label"]] = index

    tensors: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    pair_id_summary: dict[str, list[str]] = {}
    for split_name, by_pair in by_split_pair.items():
        left_rows = []
        right_rows = []
        pair_ids = []
        for pair_id, entry in sorted(by_pair.items()):
            if set(entry) != {"left", "right"}:
                continue
            left_rows.append(hidden[entry["left"]])
            right_rows.append(hidden[entry["right"]])
            pair_ids.append(pair_id)
        if left_rows:
            tensors[split_name] = (torch.stack(left_rows), torch.stack(right_rows))
        else:
            tensors[split_name] = (torch.empty((0, hidden.shape[1])), torch.empty((0, hidden.shape[1])))
        pair_id_summary[split_name] = pair_ids[:10]
    if len(tensors["train"][0]) == 0:
        raise ValueError("No complete train left/right pairs found")
    meta = {
        "split_sha256": split_sha,
        "feature_cache": str(feature_cache_npz),
        "feature_cache_metadata": cache_meta,
        "example_id_sha256": example_id_sha256,
        "text_sha256": text_sha256,
        "expected_feature_cache_name": expected_cache_path.name,
        "n_pairs_by_split": {split_name: int(len(pair[0])) for split_name, pair in tensors.items()},
        "pair_ids_sample_by_split": pair_id_summary,
    }
    return tensors, meta


def pair_distances(
    z_left: torch.Tensor,
    z_right: torch.Tensor,
    relative: bool,
    eps: float = 1e-8,
) -> torch.Tensor:
    distance = torch.norm(z_right - z_left, dim=-1)
    if relative:
        denom = 0.5 * (torch.norm(z_left, dim=-1) + torch.norm(z_right, dim=-1)) + eps
        distance = distance / denom
    return distance


def minimum_gain_loss(
    z_left: torch.Tensor,
    z_right: torch.Tensor,
    gamma: float,
    relative: bool,
) -> torch.Tensor:
    distance = pair_distances(z_left, z_right, relative=relative)
    return torch.square(torch.relu(distance.new_tensor(gamma) - distance)).mean()


def trust_region_loss(sae: StandardSAE, base_state: dict[str, torch.Tensor]) -> torch.Tensor:
    total = None
    count = 0
    for name, param in sae.named_parameters():
        diff = torch.square(param - base_state[name].to(param.device)).mean()
        total = diff if total is None else total + diff
        count += 1
    if total is None:
        return torch.tensor(0.0, device=next(sae.parameters()).device)
    return total / count


@torch.no_grad()
def pair_metrics(sae: StandardSAE, h_left: torch.Tensor, h_right: torch.Tensor, device: str) -> dict[str, float]:
    z_left = sae.encode(h_left.to(device))
    z_right = sae.encode(h_right.to(device))
    distance = pair_distances(z_left, z_right, relative=False)
    rel = pair_distances(z_left, z_right, relative=True)
    return {
        "code_distance_mean": float(distance.mean().item()),
        "code_distance_l20": float(torch.sort(distance).values[: max(1, int(np.ceil(0.2 * len(distance))))].mean().item()),
        "relative_distance_mean": float(rel.mean().item()),
        "relative_distance_l20": float(torch.sort(rel).values[: max(1, int(np.ceil(0.2 * len(rel))))].mean().item()),
        "relative_distance_min": float(rel.min().item()),
        "distance_min": float(distance.min().item()),
    }


def certificate_status(
    sae: StandardSAE,
    h_left: torch.Tensor,
    h_right: torch.Tensor,
    device: str,
    gamma: float,
    relative: bool,
    pass_fraction: float,
) -> dict[str, Any]:
    with torch.no_grad():
        z_left = sae.encode(h_left.to(device))
        z_right = sae.encode(h_right.to(device))
        distances = pair_distances(z_left, z_right, relative=relative).float().cpu().numpy()
    return {
        "gamma": gamma,
        "relative": relative,
        "pass_fraction_required": pass_fraction,
        "n_pairs": int(len(distances)),
        "min": float(np.min(distances)),
        "q05": float(np.quantile(distances, 0.05)),
        "l20": float(np.mean(np.sort(distances)[: max(1, int(np.ceil(0.2 * len(distances))))])),
        "mean": float(np.mean(distances)),
        "pass_fraction": float(np.mean(distances >= gamma)),
        "passed": bool(np.mean(distances >= gamma) >= pass_fraction),
    }


def train(args: argparse.Namespace) -> Path:
    device = args.device
    manifest = load_manifest()
    model_cfg = manifest["models"][args.profile]
    standard_checkpoint_id = f"{args.profile}_standard_joint16_owt"
    standard_checkpoint = checkpoint_dir(standard_checkpoint_id, manifest)
    out_dir = args.output_dir / args.profile / args.run_name
    if out_dir.exists() and any(out_dir.iterdir()) and not args.force:
        raise FileExistsError(f"Output directory is not empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    generator = seed_everything(args.seed)
    rng = random.Random(args.seed)
    owt_acts, owt_meta = load_owt_cache(args.owt_cache)
    pair_tensors, laterality_meta = load_laterality_pairs_by_split(
        args.split_json,
        args.feature_cache_npz,
        args.profile,
        model_cfg["model_id"],
        int(model_cfg["hf_hidden_state_index"]),
    )
    h_left, h_right = pair_tensors["train"]
    h_left_dev, h_right_dev = pair_tensors["dev"]

    sae = StandardSAE.load_checkpoint(standard_checkpoint, device=device)
    base_state = {name: param.detach().clone().cpu() for name, param in sae.named_parameters()}
    sae.train()
    opt = torch.optim.Adam(sae.parameters(), lr=args.lr)
    loader = DataLoader(
        TensorDataset(owt_acts),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        generator=generator,
    )
    data_iter = iter(loader)
    history = []
    baseline_metrics = pair_metrics(sae, h_left, h_right, device)

    for step in range(1, args.steps + 1):
        try:
            (batch_h,) = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            (batch_h,) = next(data_iter)
        batch_h = batch_h.to(device)
        indices = rng.sample(range(len(h_left)), k=min(args.pairs_per_step, len(h_left)))
        left = h_left[indices].to(device)
        right = h_right[indices].to(device)

        opt.zero_grad(set_to_none=True)
        loss_recon, recon_meta = sae.loss(batch_h, args.l1_coeff)
        z_left = sae.encode(left)
        z_right = sae.encode(right)
        loss_v = v_gini_loss_from_codes(
            z_left,
            z_right,
            left,
            right,
            normalize_by_input=args.normalize_by_input,
        )
        loss_gamma = minimum_gain_loss(
            z_left,
            z_right,
            gamma=args.gamma,
            relative=args.gamma_relative,
        )
        loss_trust = trust_region_loss(sae, base_state)
        loss = (
            loss_recon
            + args.lambda_v * loss_v
            + args.lambda_gamma * loss_gamma
            + args.lambda_trust * loss_trust
        )
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(sae.parameters(), args.grad_clip)
        opt.step()

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            dev_certificate = certificate_status(
                sae,
                h_left_dev,
                h_right_dev,
                device,
                args.certificate_gamma,
                args.certificate_relative,
                args.certificate_pass_fraction,
            )
            rec = {
                "step": step,
                "loss": float(loss.item()),
                "recon": float(recon_meta["recon"]),
                "l1": float(recon_meta["l1"]),
                "v_gini_loss": float(loss_v.item()),
                "minimum_gain_loss": float(loss_gamma.item()),
                "trust_region_loss": float(loss_trust.item()),
                "dev_certificate": dev_certificate,
            }
            history.append(rec)
            print(
                f"step {step:5d} loss={rec['loss']:.4f} recon={rec['recon']:.4f} "
                f"v={rec['v_gini_loss']:.4f} gamma={rec['minimum_gain_loss']:.4f} "
                f"trust={rec['trust_region_loss']:.6f} "
                f"dev_pass={dev_certificate['pass_fraction']:.3f}"
            )
            if args.certificate_stop and step >= args.min_steps and dev_certificate["passed"]:
                print(f"Certificate stop at step {step}")
                break

    sae.eval()
    final_metrics = pair_metrics(sae, h_left, h_right, device)
    final_dev_certificate = certificate_status(
        sae,
        h_left_dev,
        h_right_dev,
        device,
        args.certificate_gamma,
        args.certificate_relative,
        args.certificate_pass_fraction,
    )
    meta = {
        "experiment": "E3c_targeted_laterality_repair",
        "run_name": args.run_name,
        "profile": args.profile,
        "model_id": model_cfg["model_id"],
        "layer": int(model_cfg["hf_hidden_state_index"]),
        "base_checkpoint_id": standard_checkpoint_id,
        "base_checkpoint_path": str(standard_checkpoint),
        "base_checkpoint_sha256": sha256_file(standard_checkpoint / "sae.pt"),
        "training_variant": "targeted_laterality_v_gini_repair",
        "lambda_v": args.lambda_v,
        "lambda_gamma": args.lambda_gamma,
        "lambda_trust": args.lambda_trust,
        "gamma": args.gamma,
        "gamma_relative": args.gamma_relative,
        "certificate_stop": args.certificate_stop,
        "certificate_gamma": args.certificate_gamma,
        "certificate_relative": args.certificate_relative,
        "certificate_pass_fraction": args.certificate_pass_fraction,
        "certificate_final": final_dev_certificate,
        "normalize_by_input": args.normalize_by_input,
        "steps_requested": args.steps,
        "steps_completed": history[-1]["step"] if history else 0,
        "batch_size": args.batch_size,
        "pairs_per_step": args.pairs_per_step,
        "lr": args.lr,
        "l1_coeff": args.l1_coeff,
        "seed": args.seed,
        "owt_cache": str(args.owt_cache),
        "owt_cache_sha256": sha256_file(args.owt_cache),
        "owt_cache_meta": owt_meta,
        "laterality_train_cache": laterality_meta,
        "train_pair_metrics_before": baseline_metrics,
        "train_pair_metrics_after": final_metrics,
        "git_commit": git_commit(),
        "history": history,
        "isolation_note": "E3c-local targeted repair; legacy checkpoints are read-only baselines",
    }
    sae.save_checkpoint(out_dir, meta)
    print(f"Saved checkpoint -> {out_dir}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Train targeted laterality repair SAE")
    parser.add_argument("--run-name", default="gpt2_laterality_vgini_seed42")
    parser.add_argument("--profile", choices=("gpt2", "gemma-2-2b", "qwen-2.5-3b"), default="gpt2")
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--feature-cache-npz", type=Path, required=True)
    parser.add_argument("--owt-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=REVISION_ROOT / "E3c_targeted_laterality_repair" / "checkpoints")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--pairs-per-step", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--l1-coeff", type=float, default=1e-3)
    parser.add_argument("--lambda-v", type=float, default=0.2)
    parser.add_argument("--lambda-gamma", type=float, default=0.0)
    parser.add_argument("--lambda-trust", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--gamma-relative", action="store_true")
    parser.add_argument("--certificate-stop", action="store_true")
    parser.add_argument("--certificate-gamma", type=float, default=0.0)
    parser.add_argument("--certificate-relative", action="store_true")
    parser.add_argument("--certificate-pass-fraction", type=float, default=0.95)
    parser.add_argument("--min-steps", type=int, default=1)
    parser.add_argument("--normalize-by-input", action="store_true")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
