#!/usr/bin/env python3
"""Extract hidden states and optional Standard/V-reg SAE codes for the stress test.

This script is intentionally conservative: it refuses to save non-finite arrays.
Use `--batch-size 1` for MPS/float16 if a larger batch gives NaN/Inf values.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(dtype: str):
    if dtype == "auto":
        return None
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def add_project_root(project_root: Optional[str]) -> None:
    if project_root:
        root = str(Path(project_root).resolve())
        if root not in sys.path:
            sys.path.insert(0, root)


def load_sae_checkpoint(path: Optional[str], device: str, project_root: Optional[str]):
    if not path:
        return None
    add_project_root(project_root)
    ckpt = Path(path)
    # Preferred path for Csaba's SAE_scaling code.
    try:
        from sae_model_v2 import load_any_sae  # type: ignore
        return load_any_sae(ckpt, device=device).eval()
    except Exception as exc1:
        try:
            obj = torch.load(ckpt / "sae.pt" if ckpt.is_dir() else ckpt, map_location=device)
            if hasattr(obj, "eval"):
                obj.eval()
            if not hasattr(obj, "encode"):
                raise TypeError("Loaded object has no encode(x) method")
            return obj
        except Exception as exc2:
            raise RuntimeError(f"Could not load SAE from {path}. load_any_sae error={exc1}; torch.load error={exc2}")


@torch.no_grad()
def extract_hidden(model, tok, texts, layer: int, device: str, batch_size: int, max_length: int) -> np.ndarray:
    vecs = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        input_ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)
        out = model(input_ids=input_ids, attention_mask=attn, output_hidden_states=True, return_dict=True, use_cache=False)
        h = out.hidden_states[layer]
        rows = torch.arange(h.shape[0], device=device)
        last_idx = attn.sum(dim=1).long() - 1
        pooled = h[rows, last_idx].float().cpu().numpy()
        if not np.isfinite(pooled).all():
            bad = int((~np.isfinite(pooled)).sum())
            raise ValueError(f"Non-finite hidden values in batch starting at {start}: {bad}. Try --batch-size 1 or --dtype float32.")
        vecs.append(pooled.astype(np.float32))
        if (start // batch_size + 1) % 20 == 0:
            print(f"hidden batches: {start + len(batch)}/{len(texts)}", flush=True)
    return np.concatenate(vecs, axis=0)


@torch.no_grad()
def encode_sae(sae, hidden: np.ndarray, device: str, batch_size: int) -> np.ndarray:
    if sae is None:
        raise ValueError("SAE is None")
    zs = []
    for start in range(0, len(hidden), batch_size):
        xb = torch.tensor(hidden[start : start + batch_size], dtype=torch.float32, device=device)
        z = sae.encode(xb).float().cpu().numpy()
        if not np.isfinite(z).all():
            bad = int((~np.isfinite(z)).sum())
            raise ValueError(f"Non-finite SAE code values in batch starting at {start}: {bad}")
        zs.append(z.astype(np.float32))
    return np.concatenate(zs, axis=0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", default="results/features.npz")
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--encode-batch-size", type=int, default=256)
    p.add_argument("--max-length", type=int, default=192)
    p.add_argument("--standard-checkpoint", default=None)
    p.add_argument("--vreg-checkpoint", default=None)
    p.add_argument("--project-root", default=None, help="Optional SAE_scaling root for load_any_sae imports")
    p.add_argument("--trust-remote-code", action="store_true")
    args = p.parse_args()

    device = resolve_device(args.device)
    torch_dtype = resolve_dtype(args.dtype)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kwargs: Dict[str, Any] = {"trust_remote_code": args.trust_remote_code}
    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs).to(device).eval()

    dataset = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    prompts = [it["prompt"] for it in dataset["items"]]
    hidden = extract_hidden(model, tok, prompts, args.layer, device, args.batch_size, args.max_length)

    arrays: Dict[str, np.ndarray] = {"hidden": hidden}
    metadata: Dict[str, Any] = {
        "experiment": "numeric_orientation_stress_features",
        "model": args.model,
        "layer": args.layer,
        "device": device,
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "dataset": args.dataset,
        "n_items": len(prompts),
        "nonfinite_sanitized": False,
        "finite_check_passed": True,
        "standard_checkpoint": args.standard_checkpoint,
        "vreg_checkpoint": args.vreg_checkpoint,
    }

    if args.standard_checkpoint:
        sae_std = load_sae_checkpoint(args.standard_checkpoint, device, args.project_root)
        arrays["standard_code"] = encode_sae(sae_std, hidden, device, args.encode_batch_size)
    if args.vreg_checkpoint:
        sae_vreg = load_sae_checkpoint(args.vreg_checkpoint, device, args.project_root)
        arrays["vreg_code"] = encode_sae(sae_vreg, hidden, device, args.encode_batch_size)

    for name, arr in arrays.items():
        if not np.isfinite(arr).all():
            raise ValueError(f"Refusing to save non-finite array: {name}")
        metadata[f"{name}_shape"] = list(arr.shape)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **arrays, metadata=np.array(metadata, dtype=object))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
