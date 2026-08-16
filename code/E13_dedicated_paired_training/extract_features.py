#!/usr/bin/env python3
"""Extract features for the dedicated GPT-2 negation evaluation (LM loaded once).

Produces:
  - results/features_negation.npz : hidden / std_code / vreg_code /
    std_recon / vreg_recon for every probe example, plus labels/splits/pair ids.
  - results/eval_phenomenon_noharm.json + .md : the V-Gini collapse profile
    (train + HELD-OUT test templates) and the OWT no-harm reconstruction check.

Downstream probe and behavioural analyses read the npz cache (no LM reload).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import common  # noqa: F401
import numpy as np
import torch

from activations import last_token_hidden
from lm_loader import load_model_and_tokenizer, resolve_device
from sae_model_v2 import load_any_sae

HERE = Path(__file__).resolve().parent
OWT_CACHE = HERE.parents[1] / "tmlr_revision/prepare/experiment_101_hybrid_owt/data/owt_cache_gpt2_l12_25k.pt"


def np_gini(arr: np.ndarray) -> float:
    s = np.sort(np.asarray(arr, dtype=np.float64))
    n = len(s)
    if n == 0 or s.sum() < 1e-10:
        return 0.0
    cumsum = np.cumsum(s)
    g = 1.0 - 2.0 * np.sum(cumsum) / (n * np.sum(s)) + 1.0 / n
    return float(max(0.0, g))


def rel_sensitivity(z_orig: np.ndarray, z_pert: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    num = np.linalg.norm(z_pert - z_orig, axis=-1)
    den = np.linalg.norm(z_orig, axis=-1) + eps
    return num / den


@torch.no_grad()
def encode_decode(sae, h: torch.Tensor, device: str, bs: int = 256):
    codes, recons = [], []
    for i in range(0, h.shape[0], bs):
        chunk = h[i:i + bs].to(device=device, dtype=torch.float32)
        z = sae.encode(chunk)
        x = sae.decode(z)
        codes.append(z.cpu())
        recons.append(x.cpu())
    return torch.cat(codes).numpy(), torch.cat(recons).numpy()


def recon_stats(h: np.ndarray, h_hat: np.ndarray) -> dict:
    diff = h - h_hat
    mse = float(np.mean(np.sum(diff ** 2, axis=1)))
    denom = float(np.mean(np.sum((h - h.mean(axis=0, keepdims=True)) ** 2, axis=1)))
    ev = float(1.0 - (np.mean(np.sum(diff ** 2, axis=1)) / (denom + 1e-12)))
    nmse = float(np.mean(np.sum(diff ** 2, axis=1)) / (np.mean(np.sum(h ** 2, axis=1)) + 1e-12))
    cos = float(np.mean(np.sum(h * h_hat, axis=1) /
                        (np.linalg.norm(h, axis=1) * np.linalg.norm(h_hat, axis=1) + 1e-12)))
    return {"mse": round(mse, 4), "nmse": round(nmse, 6), "ev": round(ev, 4), "cosine": round(cos, 4)}


def phenomenon_by_split(examples, codes: np.ndarray) -> dict:
    """V-Gini over affirmed->negated pairs, per split. codes: (N, d_sae)."""
    idx = {e["example_id"]: i for i, e in enumerate(examples)}
    pairs: dict[str, dict] = {}
    for e in examples:
        pairs.setdefault(e["pair_id"], {})[e["side"]] = e
    out = {}
    for split in ("train", "dev", "test", "all"):
        z_o, z_p = [], []
        for pid, sides in pairs.items():
            if "aff" not in sides or "neg" not in sides:
                continue
            if split != "all" and sides["aff"]["split"] != split:
                continue
            z_o.append(codes[idx[sides["aff"]["example_id"]]])
            z_p.append(codes[idx[sides["neg"]["example_id"]]])
        if not z_o:
            continue
        s = rel_sensitivity(np.stack(z_o), np.stack(z_p))
        out[split] = {"n_pairs": len(s), "V_gini": round(np_gini(s), 4),
                      "mean_sensitivity": round(float(s.mean()), 4)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--std-ckpt", type=Path, default=HERE / "checkpoints/negation_standard")
    ap.add_argument("--vreg-ckpt", type=Path, default=HERE / "checkpoints/negation_vreg")
    ap.add_argument("--probe-json", type=Path, default=HERE / "data/negation_probe.json")
    ap.add_argument("--owt-cache", type=Path, default=OWT_CACHE)
    ap.add_argument("--owt-n", type=int, default=5000)
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out-dir", type=Path, default=HERE / "results")
    args = ap.parse_args()

    device = resolve_device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads(args.probe_json.read_text(encoding="utf-8"))
    examples = payload["examples"]
    texts = [e["text"] for e in examples]

    std = load_any_sae(args.std_ckpt, device=device)
    vreg = load_any_sae(args.vreg_ckpt, device=device)

    lm, tok = load_model_and_tokenizer(args.model, device, "auto",
                                       trust_remote_code=args.trust_remote_code)
    hidden = last_token_hidden(lm, tok, texts, args.layer, device, batch_size=16).numpy()

    # OWT no-harm subset (held reconstruction of general text).
    owt = torch.load(Path(args.owt_cache).expanduser().resolve(), map_location="cpu", weights_only=False)
    owt_acts = owt["activations"][: args.owt_n].float()
    del lm, tok

    std_code, std_recon = encode_decode(std, torch.from_numpy(hidden), device)
    vreg_code, vreg_recon = encode_decode(vreg, torch.from_numpy(hidden), device)

    _, owt_std_hat = encode_decode(std, owt_acts, device)
    _, owt_vreg_hat = encode_decode(vreg, owt_acts, device)
    owt_np = owt_acts.numpy()

    labels = np.array([1 if e["label"] == "negated" else 0 for e in examples], dtype=np.int64)
    splits = np.array([e["split"] for e in examples])
    template_ids = np.array([e["template_id"] for e in examples], dtype=np.int64)
    pair_ids = np.array([e["pair_id"] for e in examples])
    sides = np.array([e["side"] for e in examples])

    np.savez_compressed(
        args.out_dir / "features_negation.npz",
        hidden=hidden, std_code=std_code, vreg_code=vreg_code,
        std_recon=std_recon, vreg_recon=vreg_recon,
        labels=labels, splits=splits, template_ids=template_ids,
        pair_ids=pair_ids, sides=sides, texts=np.array(texts, dtype=object),
    )

    phen = {
        "standard": phenomenon_by_split(examples, std_code),
        "vreg": phenomenon_by_split(examples, vreg_code),
    }
    noharm = {
        "owt_n": int(owt_np.shape[0]),
        "standard": recon_stats(owt_np, owt_std_hat),
        "vreg": recon_stats(owt_np, owt_vreg_hat),
    }
    result = {"phenomenon_V_gini": phen, "owt_no_harm": noharm,
              "std_ckpt": str(args.std_ckpt), "vreg_ckpt": str(args.vreg_ckpt)}
    (args.out_dir / "eval_phenomenon_noharm.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")

    # Markdown
    lines = ["# Phenomenon (V-Gini collapse) + OWT no-harm", "",
             "## V-Gini over affirmed->negated pairs (lower = flatter response profile)",
             "",
             "| Split | Standard V | V-reg V |", "|---|---:|---:|"]
    for split in ("train", "dev", "test", "all"):
        s = phen["standard"].get(split); v = phen["vreg"].get(split)
        if s and v:
            tag = " (HELD-OUT)" if split == "test" else ""
            lines.append(f"| {split}{tag} | {s['V_gini']} | {v['V_gini']} |")
    lines += ["", "## OWT reconstruction no-harm (general text, "
              f"{noharm['owt_n']} tokens)", "",
              "| Metric | Standard | V-reg |", "|---|---:|---:|"]
    for k in ("mse", "nmse", "ev", "cosine"):
        lines.append(f"| {k} | {noharm['standard'][k]} | {noharm['vreg'][k]} |")
    (args.out_dir / "eval_phenomenon_noharm.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
