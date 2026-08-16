#!/usr/bin/env python3
"""Evaluate E2 nuisance controls with frozen Standard/V-reg SAE checkpoints.

This is the missing bridge between text-only nuisance controls and
run_null_calibration.py. It reuses the E1 hidden extraction and SAE metric code so the
distance fields have the same meaning as E1/E1R per-pair metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch

REVISION_ROOT = Path(__file__).resolve().parents[1]
E1_ROOT = REVISION_ROOT / "E1_absolute_sensitivity"
for path in (REVISION_ROOT, E1_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from shared.path_registry import checkpoint_dir, load_manifest, sae_scaling_root  # noqa: E402
from eval_core import (  # noqa: E402
    clear_device_cache,
    collect_hidden_pairs,
    evaluate_sae_on_hidden,
    load_sae,
    module_param_info,
    setup_sae_scaling_imports,
)


PROFILE_RUNS: dict[str, tuple[str, str]] = {
    "gpt2": ("gpt2_standard_joint16_owt", "gpt2_vreg_joint16_owt"),
    "gemma-2-2b": ("gemma-2-2b_standard_joint16_owt", "gemma-2-2b_vreg_joint16_owt"),
    "qwen-2.5-3b": ("qwen-2.5-3b_standard_joint16_owt", "qwen-2.5-3b_vreg_joint16_owt"),
}

METRIC_KEYS = {
    "s": "s",
    "g": "g",
    "abs_dz": "abs_dz",
    "decode_resp": "decode_resp",
    "h_frac": "h_frac",
}

DEFAULT_GEMMA_E1R_VREG = (
    REVISION_ROOT
    / "gemma_true_last_protocol"
    / "checkpoints"
    / "gemma-2-2b"
    / "joint"
    / "vreg_joint16_owt_true_last"
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def checkpoint_sae_file(checkpoint: Path) -> Path:
    sae_path = checkpoint / "sae.pt"
    if not sae_path.is_file():
        raise FileNotFoundError(f"Missing SAE checkpoint file: {sae_path}")
    return sae_path


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def metric_value(arrays: dict[str, Any], metric: str, index: int) -> float:
    values = arrays[METRIC_KEYS[metric]]
    return float(values[index])


def validate_nuisance_payload(payload: dict[str, Any]) -> None:
    records = payload.get("records", [])
    if not records:
        raise ValueError("No nuisance records found")
    ids = [record["nuisance_id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate nuisance_id in nuisance payload")
    if payload.get("manual_audit_status") == "rejected":
        raise ValueError("Nuisance payload is marked rejected")


def profile_complete(records: list[dict[str, Any]], profile: str) -> bool:
    for record in records:
        profile_block = record.get("distances", {}).get(profile)
        if not profile_block:
            return False
        for representation in ("standard", "vreg"):
            rep_block = profile_block.get(representation)
            if not rep_block:
                return False
            if any(metric not in rep_block for metric in METRIC_KEYS):
                return False
    return True


def ensure_output_allowed(output_json: Path, resume: bool, force: bool) -> dict[str, Any] | None:
    if not output_json.exists():
        return None
    if resume:
        return read_json(output_json)
    if force:
        return None
    raise FileExistsError(
        f"Output already exists: {output_json}. Use --resume to continue or --force to overwrite."
    )


def checkpoint_metadata(
    checkpoint_id: str,
    checkpoint_path: Path,
    manifest: dict[str, Any],
    override_used: bool,
) -> dict[str, Any]:
    sae_path = checkpoint_sae_file(checkpoint_path)
    manifest_sha = None
    if not override_used and checkpoint_id in manifest.get("checkpoints", {}):
        manifest_sha = manifest["checkpoints"][checkpoint_id].get("sha256_sae_pt")
    actual_sha = sha256_file(sae_path)
    if manifest_sha and actual_sha.lower() != manifest_sha.lower():
        raise ValueError(
            f"Checkpoint SHA mismatch for {checkpoint_id}: expected {manifest_sha}, got {actual_sha}"
        )
    meta_path = checkpoint_path / "meta.json"
    return {
        "checkpoint_id": checkpoint_id,
        "checkpoint_path": str(checkpoint_path),
        "sae_path": str(sae_path),
        "sha256_sae_pt": actual_sha,
        "manifest_sha256_sae_pt": manifest_sha,
        "override_used": override_used,
        "meta_path": str(meta_path) if meta_path.is_file() else None,
    }


def evaluate_profile(
    profile: str,
    records: list[dict[str, Any]],
    manifest: dict[str, Any],
    device: str,
    lm_dtype: str,
    hidden_batch_size: int,
    max_length: int,
    gemma_vreg_checkpoint: Path,
) -> dict[str, Any]:
    setup_sae_scaling_imports(sae_scaling_root(manifest))
    from lm_loader import load_model_and_tokenizer

    model_cfg = manifest["models"][profile]
    layer = int(model_cfg["hf_hidden_state_index"])
    std_id, vreg_id = PROFILE_RUNS[profile]
    std_ckpt = checkpoint_dir(std_id, manifest)
    vreg_ckpt = gemma_vreg_checkpoint if profile == "gemma-2-2b" else checkpoint_dir(vreg_id, manifest)
    if not std_ckpt.is_dir():
        raise FileNotFoundError(std_ckpt)
    if not vreg_ckpt.is_dir():
        raise FileNotFoundError(vreg_ckpt)

    std_meta = checkpoint_metadata(std_id, std_ckpt, manifest, override_used=False)
    vreg_meta = checkpoint_metadata(
        "gemma_e1r_vreg_joint16_owt_true_last" if profile == "gemma-2-2b" else vreg_id,
        vreg_ckpt,
        manifest,
        override_used=profile == "gemma-2-2b",
    )

    pairs = [(record["source_text"], record["nuisance_text"]) for record in records]
    runtime: dict[str, Any] = {
        "profile": profile,
        "model_id": model_cfg["model_id"],
        "layer": layer,
        "device": device,
        "lm_dtype": lm_dtype,
        "hidden_batch_size": hidden_batch_size,
        "max_length": max_length,
        "extraction_protocol": "true_last",
        "checkpoint_standard": std_id,
        "checkpoint_standard_metadata": std_meta,
        "checkpoint_vreg": (
            "vreg_joint16_owt_true_last" if profile == "gemma-2-2b" else vreg_id
        ),
        "checkpoint_vreg_metadata": vreg_meta,
        "gemma_e1r_vreg_override_used": profile == "gemma-2-2b",
    }

    print(f"[{profile}] Loading LM {model_cfg['model_id']}")
    lm, tok = load_model_and_tokenizer(
        model_cfg["model_id"],
        device,
        dtype=lm_dtype,
        trust_remote_code=model_cfg.get("trust_remote_code", False),
    )
    lm.eval()
    runtime["lm"] = module_param_info(lm)
    runtime["tokenizer_padding_side"] = getattr(tok, "padding_side", "unknown")
    hidden = collect_hidden_pairs(
        lm,
        tok,
        pairs,
        layer,
        device,
        batch_size=hidden_batch_size,
        extraction_protocol="true_last",
        max_length=max_length,
    )
    del lm, tok
    clear_device_cache(device)

    print(f"[{profile}] Evaluating Standard SAE")
    std_sae = load_sae(std_ckpt, device)
    std_sae.eval()
    runtime["sae_standard"] = module_param_info(std_sae)
    std_arrays = evaluate_sae_on_hidden(std_sae, hidden)
    del std_sae
    clear_device_cache(device)

    print(f"[{profile}] Evaluating V-reg SAE")
    vreg_sae = load_sae(vreg_ckpt, device)
    vreg_sae.eval()
    runtime["sae_vreg"] = module_param_info(vreg_sae)
    vreg_arrays = evaluate_sae_on_hidden(vreg_sae, hidden)
    del vreg_sae
    clear_device_cache(device)

    for index, record in enumerate(records):
        record.setdefault("distances", {}).setdefault(profile, {})
        for representation, arrays in (("standard", std_arrays), ("vreg", vreg_arrays)):
            record["distances"][profile][representation] = {
                "s": metric_value(arrays, "s", index),
                "g": metric_value(arrays, "g", index),
                "abs_dz": metric_value(arrays, "abs_dz", index),
                "decode_resp": metric_value(arrays, "decode_resp", index),
                "h_frac": metric_value(arrays, "h_frac", index),
            }

    return runtime


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate E2 nuisance pairs with canonical true_last extraction"
    )
    parser.add_argument("--nuisance-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--profile", choices=[*PROFILE_RUNS.keys(), "all"], default="all")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--lm-dtype", default="float16")
    parser.add_argument("--hidden-batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--gemma-vreg-checkpoint", type=Path, default=DEFAULT_GEMMA_E1R_VREG)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    manifest = load_manifest()
    existing = ensure_output_allowed(args.output_json, args.resume, args.force)
    payload = existing if existing is not None else read_json(args.nuisance_json)
    validate_nuisance_payload(payload)
    records = payload.get("records", [])
    profiles = list(PROFILE_RUNS) if args.profile == "all" else [args.profile]
    device = resolve_device(args.device)

    current_nuisance_sha = sha256_file(args.nuisance_json)
    existing_sha = payload.get("distance_evaluation", {}).get("nuisance_json_sha256")
    if args.resume and existing_sha and existing_sha != current_nuisance_sha:
        raise ValueError(
            f"Resume input SHA mismatch: existing={existing_sha}, current={current_nuisance_sha}"
        )

    payload.setdefault("distance_evaluation", {})
    payload["distance_evaluation"].setdefault("runtime", {})
    payload["distance_evaluation"].setdefault("completed_profiles", [])
    payload["distance_evaluation"]["status"] = "running"
    payload["distance_evaluation"]["profiles_requested"] = profiles
    payload["distance_evaluation"]["metric_keys"] = sorted(METRIC_KEYS)
    payload["distance_evaluation"]["nuisance_json"] = str(args.nuisance_json)
    payload["distance_evaluation"]["nuisance_json_sha256"] = current_nuisance_sha

    for profile in profiles:
        if args.resume and profile_complete(records, profile):
            print(f"[{profile}] Existing distances complete; skipping due to --resume")
            if profile not in payload["distance_evaluation"]["completed_profiles"]:
                payload["distance_evaluation"]["completed_profiles"].append(profile)
            continue
        payload["distance_evaluation"]["runtime"][profile] = evaluate_profile(
            profile,
            records,
            manifest,
            device,
            args.lm_dtype,
            args.hidden_batch_size,
            args.max_length,
            args.gemma_vreg_checkpoint,
        )
        if profile not in payload["distance_evaluation"]["completed_profiles"]:
            payload["distance_evaluation"]["completed_profiles"].append(profile)
        payload["distance_evaluation"]["status"] = "partial"
        atomic_write_json(args.output_json, payload)
        print(f"[{profile}] Partial save -> {args.output_json}")

    missing = [profile for profile in profiles if not profile_complete(records, profile)]
    payload["distance_evaluation"]["status"] = "complete" if not missing else "partial"
    payload["distance_evaluation"]["missing_profiles"] = missing
    atomic_write_json(args.output_json, payload)
    print(f"Saved evaluated nuisance JSON -> {args.output_json}")


if __name__ == "__main__":
    main()
