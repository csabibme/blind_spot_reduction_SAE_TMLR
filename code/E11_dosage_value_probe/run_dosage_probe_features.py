#!/usr/bin/env python3
"""Extract frozen LM hidden states and SAE codes for held-out dosage pairs."""

from __future__ import annotations

import argparse
import hashlib
import sys
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dosage_probe_common import (  # noqa: E402
    REPRESENTATIONS,
    STATUS_FROZEN,
    read_json,
    sha256_file,
    sha256_text,
    write_json,
)
from revision_paths import E1_ROOT, REVISION_1_ROOT, ensure_import_paths, load_manifest  # noqa: E402

ensure_import_paths()
for path in (REVISION_1_ROOT, E1_ROOT):
    root = str(path.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)

from shared.path_registry import checkpoint_dir, sae_scaling_root  # noqa: E402
from eval_core import clear_device_cache, load_sae, setup_sae_scaling_imports  # noqa: E402

from dosage_probe_common import PROFILE_RUNS  # noqa: E402

PAIR_REPRESENTATIONS = tuple(f"{rep}_{side}" for rep in REPRESENTATIONS for side in ("left", "right"))

DEFAULT_GEMMA_E1R_VREG = (
    REVISION_1_ROOT
    / "E1R_gemma_protocol_repair"
    / "checkpoints"
    / "gemma-2-2b"
    / "joint"
    / "vreg_joint16_owt_true_last"
)


def checkpoint_sae_file(checkpoint: Path) -> Path:
    sae_path = checkpoint / "sae.pt"
    if not sae_path.is_file():
        raise FileNotFoundError(f"Missing SAE checkpoint file: {sae_path}")
    return sae_path


def checkpoint_metadata(
    checkpoint_id: str,
    checkpoint_path: Path,
    manifest: dict[str, Any],
    *,
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


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def extraction_protocol_for_profile(_profile: str) -> str:
    return "true_last"


def feature_cache_path(
    cache_dir: Path,
    profile: str,
    dataset_sha256: str,
    pair_ids: list[str],
) -> Path:
    digest = sha256_text("\n".join(pair_ids))
    return cache_dir / f"{profile}_{dataset_sha256[:16]}_{digest[:16]}.npz"


def normalized_cache_meta(meta: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(meta)
    for key in ("standard_checkpoint", "vreg_checkpoint"):
        if key not in normalized:
            continue
        path = Path(str(normalized[key])).expanduser()
        normalized[key] = unicodedata.normalize("NFC", str(path.resolve()))
    return normalized


def load_feature_cache(path: Path, expected_meta: dict[str, Any]) -> dict[str, np.ndarray] | None:
    if not path.is_file():
        return None
    cached = np.load(path, allow_pickle=True)
    meta_raw = cached["metadata"].item()
    if normalized_cache_meta(meta_raw) != normalized_cache_meta(expected_meta):
        return None
    return {name: cached[name] for name in PAIR_REPRESENTATIONS}


def save_feature_cache(path: Path, features: dict[str, np.ndarray], metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    arrays = {name: features[name] for name in PAIR_REPRESENTATIONS}
    arrays["metadata"] = np.array(metadata, dtype=object)
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    tmp.replace(path)


def collect_hidden_texts(
    lm,
    tok,
    texts: list[str],
    layer: int,
    device: str,
    batch_size: int,
    extraction_protocol: str,
    max_length: int,
    progress_prefix: str = "",
) -> torch.Tensor:
    if extraction_protocol != "true_last":
        raise ValueError(f"Unsupported extraction protocol: {extraction_protocol}")
    from activations import last_token_hidden_true_last

    bs = batch_size if batch_size > 0 else len(texts)
    n_batches = (len(texts) + bs - 1) // bs if texts else 0
    print(
        f"{progress_prefix}hidden batch_size={bs}, n_batches={n_batches}, n_texts={len(texts)}",
        flush=True,
    )
    progress_every = 5 if n_batches > 20 else 1
    return last_token_hidden_true_last(
        lm,
        tok,
        texts,
        layer,
        device,
        max_length=max_length,
        batch_size=bs,
        progress_every=progress_every,
        progress_prefix=progress_prefix,
    ).float().cpu()


def deduplicate_texts(texts: list[str]) -> tuple[list[str], list[int]]:
    unique: list[str] = []
    index_by_text: dict[str, int] = {}
    indices: list[int] = []
    for text in texts:
        if text not in index_by_text:
            index_by_text[text] = len(unique)
            unique.append(text)
        indices.append(index_by_text[text])
    return unique, indices


def default_hidden_batch_size(profile: str) -> int:
    # Gemma fp16 on MPS produces non-finite hidden states for these dosage texts
    # when multiple prompts are batched together. Singleton batches match the
    # finite diagnostic path while preserving the E3 device/dtype stack.
    if profile == "gemma-2-2b":
        return 1
    return 16


@torch.no_grad()
def encode_pair_representations(
    hidden_left: torch.Tensor,
    hidden_right: torch.Tensor,
    sae_standard,
    sae_vreg,
    encode_batch_size: int = 256,
) -> dict[str, np.ndarray]:
    device = next(sae_standard.parameters()).device

    def encode_batched(sae, hidden: torch.Tensor) -> np.ndarray:
        chunks: list[np.ndarray] = []
        for start in range(0, hidden.shape[0], encode_batch_size):
            batch = hidden[start : start + encode_batch_size].to(device)
            chunks.append(sae.encode(batch).float().cpu().numpy())
        return np.concatenate(chunks, axis=0)

    z_std_left = encode_batched(sae_standard, hidden_left)
    z_std_right = encode_batched(sae_standard, hidden_right)
    z_vreg_left = encode_batched(sae_vreg, hidden_left)
    z_vreg_right = encode_batched(sae_vreg, hidden_right)
    return {
        "hidden_left": hidden_left.float().cpu().numpy(),
        "hidden_right": hidden_right.float().cpu().numpy(),
        "sae_standard_code_left": z_std_left,
        "sae_standard_code_right": z_std_right,
        "sae_vreg_code_left": z_vreg_left,
        "sae_vreg_code_right": z_vreg_right,
    }


def extract_profile_features(
    profile: str,
    pairs: list[dict[str, Any]],
    manifest: dict[str, Any],
    device: str,
    lm_dtype: str,
    hidden_batch_size: int,
    max_length: int,
    gemma_vreg_checkpoint: Path,
    feature_cache_dir: Path,
    dataset_sha256: str,
) -> dict[str, Any]:
    setup_sae_scaling_imports(sae_scaling_root(manifest))
    from lm_loader import load_model_and_tokenizer

    print(f"[{profile}] Starting feature extraction ({len(pairs)} pairs)", flush=True)
    texts_left = [pair["text_left"] for pair in pairs]
    texts_right = [pair["text_right"] for pair in pairs]
    pair_ids = [pair["pair_id"] for pair in pairs]

    model_cfg = manifest["models"][profile]
    layer = int(model_cfg["hf_hidden_state_index"])
    std_id, vreg_id = PROFILE_RUNS[profile]
    std_ckpt = checkpoint_dir(std_id, manifest)
    vreg_ckpt = gemma_vreg_checkpoint if profile == "gemma-2-2b" else checkpoint_dir(vreg_id, manifest)
    extraction_protocol = extraction_protocol_for_profile(profile)
    vreg_override = profile == "gemma-2-2b"

    std_meta = checkpoint_metadata(std_id, std_ckpt, manifest, override_used=False)
    vreg_meta = checkpoint_metadata(
        vreg_id if not vreg_override else "gemma_e1r_vreg_override",
        vreg_ckpt,
        manifest,
        override_used=vreg_override,
    )

    cache_meta = {
        "profile": profile,
        "dataset_sha256": dataset_sha256,
        "model_id": model_cfg["model_id"],
        "layer": layer,
        "dtype": lm_dtype,
        "hidden_storage_dtype": "float32",
        "text_deduplication": True,
        "hidden_batch_size": hidden_batch_size,
        "max_length": max_length,
        "extraction_protocol": extraction_protocol,
        "standard_checkpoint": str(std_ckpt),
        "vreg_checkpoint": str(vreg_ckpt),
        "standard_checkpoint_sha256": std_meta["sha256_sae_pt"],
        "vreg_checkpoint_sha256": vreg_meta["sha256_sae_pt"],
        "n_pairs": len(pairs),
    }
    cache_path = feature_cache_path(feature_cache_dir, profile, dataset_sha256, pair_ids)
    features = load_feature_cache(cache_path, cache_meta)
    if features is None:
        print(f"[{profile}] Loading LM/tokenizer: {model_cfg['model_id']}", flush=True)
        lm, tok = load_model_and_tokenizer(
            model_cfg["model_id"],
            device=device,
            dtype=lm_dtype,
            trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
        )
        all_texts = texts_left + texts_right
        unique_texts, remap = deduplicate_texts(all_texts)
        n_left = len(texts_left)
        print(
            f"[{profile}] Collecting hidden states for {len(all_texts)} texts "
            f"({len(unique_texts)} unique), lm_dtype={lm_dtype}",
            flush=True,
        )
        hidden_all = collect_hidden_texts(
            lm,
            tok,
            unique_texts,
            layer,
            device,
            hidden_batch_size,
            extraction_protocol,
            max_length,
            progress_prefix=f"[{profile}] ",
        )
        if not torch.isfinite(hidden_all).all():
            bad = int((~torch.isfinite(hidden_all)).sum().item())
            raise ValueError(
                f"[{profile}] hidden extraction produced {bad} non-finite values "
                f"with device={device}, lm_dtype={lm_dtype}, batch_size={hidden_batch_size}; "
                "refusing to save a corrupted feature cache."
            )
        remap_tensor = torch.tensor(remap, dtype=torch.long)
        hidden_mapped = hidden_all[remap_tensor]
        hidden_left = hidden_mapped[:n_left]
        hidden_right = hidden_mapped[n_left:]
        del lm
        clear_device_cache(device)
        print(f"[{profile}] Loading Standard SAE", flush=True)
        sae_standard = load_sae(std_ckpt, device)
        print(f"[{profile}] Loading V-reg SAE", flush=True)
        sae_vreg = load_sae(vreg_ckpt, device)
        print(f"[{profile}] Encoding SAE representations", flush=True)
        features = encode_pair_representations(hidden_left, hidden_right, sae_standard, sae_vreg)
        del sae_standard, sae_vreg
        clear_device_cache(device)
        save_feature_cache(cache_path, features, cache_meta)
        print(f"[{profile}] Saved feature cache -> {cache_path}", flush=True)
    else:
        print(f"[{profile}] Loaded feature cache -> {cache_path}", flush=True)

    return {
        "profile": profile,
        "feature_cache_path": str(cache_path),
        "run_metadata": {
            "model_id": model_cfg["model_id"],
            "layer": layer,
            "dtype": lm_dtype,
            "device": device,
            "hidden_batch_size": hidden_batch_size,
            "max_length": max_length,
            "extraction_protocol": extraction_protocol,
            "standard_checkpoint": std_meta,
            "vreg_checkpoint": vreg_meta,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract dosage probe pair features")
    parser.add_argument("--dataset-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--profile", default="gpt2", choices=["gpt2", "gemma-2-2b", "qwen-2.5-3b", "all"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--lm-dtype", default="float16", choices=["float16", "float32", "bfloat16"])
    parser.add_argument("--hidden-batch-size", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument(
        "--feature-cache-dir",
        type=Path,
        default=SCRIPT_DIR.parent / "results" / "feature_cache",
    )
    parser.add_argument("--gemma-vreg-checkpoint", type=Path, default=DEFAULT_GEMMA_E1R_VREG)
    args = parser.parse_args()

    payload = read_json(args.dataset_json)
    if payload.get("status") != STATUS_FROZEN:
        raise ValueError(f"Unexpected dataset status: {payload.get('status')!r}")
    pairs = payload["pairs"]
    dataset_sha256 = sha256_file(args.dataset_json)
    manifest = load_manifest()
    device = resolve_device(args.device)
    profiles = list(PROFILE_RUNS) if args.profile == "all" else [args.profile]
    results = {
        "experiment": payload.get("experiment"),
        "dataset_json": str(args.dataset_json),
        "dataset_sha256": dataset_sha256,
        "profiles": {},
    }
    if args.output_json.is_file():
        existing = read_json(args.output_json)
        if existing.get("dataset_sha256") == dataset_sha256:
            results["profiles"].update(existing.get("profiles", {}))
            print(f"Loaded existing feature manifest -> {args.output_json}", flush=True)
        else:
            print(
                f"Ignoring stale feature manifest with dataset SHA {existing.get('dataset_sha256')}",
                flush=True,
            )
    for profile in profiles:
        hidden_batch_size = (
            args.hidden_batch_size
            if args.hidden_batch_size is not None
            else default_hidden_batch_size(profile)
        )
        print(
            f"[{profile}] extraction device={device}, lm_dtype={args.lm_dtype}, "
            f"hidden_batch_size={hidden_batch_size}",
            flush=True,
        )
        results["profiles"][profile] = extract_profile_features(
            profile,
            pairs,
            manifest,
            device,
            args.lm_dtype,
            hidden_batch_size,
            args.max_length,
            args.gemma_vreg_checkpoint,
            args.feature_cache_dir,
            dataset_sha256,
        )
        write_json(args.output_json, results)
        print(f"[{profile}] Updated feature extraction manifest -> {args.output_json}", flush=True)

    write_json(args.output_json, results)
    print(f"Wrote feature extraction manifest -> {args.output_json}", flush=True)


if __name__ == "__main__":
    main()
