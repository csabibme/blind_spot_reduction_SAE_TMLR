#!/usr/bin/env python3
"""Build true-last hidden perturbation cache for E5 geometry by model profile."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

REVISION_ROOT = Path(__file__).resolve().parents[1]
E5_ROOT = Path(__file__).resolve().parent
for p in (REVISION_ROOT, E5_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from shared.path_registry import load_manifest, pairs_path, sae_scaling_root  # noqa: E402


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def main() -> None:
    p = argparse.ArgumentParser(description="Build E5 true-last hidden cache for one profile")
    p.add_argument("--profile", choices=["gpt2", "qwen-2.5-3b", "gemma-2-2b"], required=True)
    p.add_argument("--device", default="mps")
    p.add_argument("--dtype", default="float16")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--max-pairs", type=int, default=50)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()

    manifest = load_manifest()
    root = sae_scaling_root(manifest)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from activations import last_token_hidden_true_last
    from lm_loader import load_model_and_tokenizer
    from perturbation_data import load_perturbation_families

    cfg = manifest["models"][args.profile]
    pair_file = pairs_path(manifest)
    pf = load_perturbation_families(pair_file)
    families = sorted(k for k in pf if not k.startswith("_"))
    out_path = args.output or E5_ROOT / "results" / f"hidden_cache_{args.profile.replace('.', '_')}_true_last.pt"

    print(f"=== E5 hidden cache: {args.profile} ===")
    print(f"  model={cfg['model_id']} layer={cfg['hf_hidden_state_index']}")
    print(f"  extraction_protocol=true_last batch_size={args.batch_size}")
    lm, tok = load_model_and_tokenizer(
        cfg["model_id"],
        args.device,
        dtype=args.dtype,
        trust_remote_code=cfg.get("trust_remote_code", False),
    )
    layer = cfg["hf_hidden_state_index"]
    cache = {}
    for i, family in enumerate(families):
        pairs = [tuple(x) for x in pf[family]["pairs"][: args.max_pairs]]
        orig = [x[0] for x in pairs]
        pert = [x[1] for x in pairs]
        print(f"  [{i+1}/{len(families)}] {family} ({len(pairs)} pairs)")
        h_o = last_token_hidden_true_last(
            lm, tok, orig, layer, args.device,
            max_length=args.max_length, batch_size=args.batch_size,
        ).cpu()
        h_p = last_token_hidden_true_last(
            lm, tok, pert, layer, args.device,
            max_length=args.max_length, batch_size=args.batch_size,
        ).cpu()
        cache[family] = (h_o, h_p)

    payload = {
        "profile": args.profile,
        "model_id": cfg["model_id"],
        "layer": layer,
        "dtype": args.dtype,
        "max_length": args.max_length,
        "max_pairs": args.max_pairs,
        "hidden_batch_size": args.batch_size,
        "padding_side": getattr(tok, "padding_side", "unknown"),
        "extraction_protocol": "true_last",
        "attention_mask_passed": True,
        "pairs_file": str(pair_file),
        "pairs_sha256": sha256_file(pair_file),
        "n_families": len(families),
        "families": families,
        "git_commit": git_commit(),
        "built_at": datetime.now(timezone.utc).isoformat(),
        "cache": cache,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    meta = {k: v for k, v in payload.items() if k != "cache"}
    out_path.with_suffix(".meta.json").write_text(
        json.dumps(meta, indent=2, allow_nan=False), encoding="utf-8",
    )
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
