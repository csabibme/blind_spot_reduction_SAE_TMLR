"""
E1 eval core — hidden-state cache + separate SAE evaluation.

Pipeline per profile:
  1. Load LM → collect all family hidden states on CPU → unload LM
  2. Load Standard SAE → evaluate all families → unload SAE
  3. Load V-reg SAE → evaluate all families → unload SAE
"""

from __future__ import annotations

import gc
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from shared.metrics import (
    absolute_code_distance,
    code_orig_norm,
    code_sparsity_stats,
    cosine_similarity_batch,
    decoded_perturbation_response,
    explained_variance,
    gini_coefficient,
    hidden_perturbation_fraction,
    input_normalised_gain,
    nmse,
    relative_sensitivity,
    summarize_distribution,
)


def setup_sae_scaling_imports(sae_scaling_root: Path) -> None:
    root = str(sae_scaling_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


def stable_family_seed(base_seed: int, family: str) -> int:
    digest = hashlib.sha256(family.encode("utf-8")).digest()
    offset = int.from_bytes(digest[:4], "big")
    return base_seed + offset


def subsample_pairs(
    pairs: list[tuple[str, str]],
    max_pairs: int,
    seed: int,
) -> tuple[list[tuple[str, str]], list[int]]:
    """Returns (selected_pairs, selected_indices)."""
    all_indices = list(range(len(pairs)))
    if max_pairs <= 0 or max_pairs >= len(pairs):
        return pairs, all_indices
    rng = random.Random(seed)
    idx = sorted(rng.sample(all_indices, max_pairs))
    return [pairs[i] for i in idx], idx


def tensor_info(t: torch.Tensor) -> dict[str, str]:
    return {"dtype": str(t.dtype), "device": str(t.device)}


def module_param_info(module: torch.nn.Module) -> dict[str, str]:
    p = next(module.parameters())
    return {"dtype": str(p.dtype), "device": str(p.device)}


def clear_device_cache(device: str) -> None:
    gc.collect()
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def load_sae(checkpoint_dir: Path, device: str):
    from sae_model_v2 import load_any_sae

    meta_path = checkpoint_dir / "meta.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("sae_class") == "MultiFactorSAE" or meta.get("sae_type") == "multifactor":
            from multifactor_sae import load_multifactor
            return load_multifactor(checkpoint_dir, device=device)
    sae = load_any_sae(checkpoint_dir, device=device)
    sae.eval()
    return sae


@torch.no_grad()
def collect_hidden_pairs(
    lm,
    tok,
    pairs: list[tuple[str, str]],
    layer: int,
    device: str,
    batch_size: int = 0,
    extraction_protocol: str = "archived",
    max_length: int = 128,
) -> dict[str, torch.Tensor]:
    """LM forward in microbatches; return h_orig/h_pert on CPU (float32).

    Args:
        batch_size: If >0, process texts in fixed microbatches of this size.
                    If 0, process all texts in a single batch.
        extraction_protocol: ``archived`` (E1 submitted) or ``true_last`` (true-last-token).
    """
    texts_o = [p[0] for p in pairs]
    texts_p = [p[1] for p in pairs]

    if extraction_protocol == "true_last":
        from activations import last_token_hidden_true_last

        def _collect(texts: list[str]) -> torch.Tensor:
            bs = batch_size if batch_size > 0 else len(texts)
            return last_token_hidden_true_last(
                lm, tok, texts, layer, device,
                max_length=max_length, batch_size=bs,
            ).float().cpu()

        return {"h_orig": _collect(texts_o), "h_pert": _collect(texts_p)}

    if extraction_protocol != "archived":
        raise ValueError(f"Unknown extraction_protocol: {extraction_protocol}")

    from activations import last_token_hidden_archived_protocol

    def _collect(texts: list[str]) -> torch.Tensor:
        if batch_size <= 0 or batch_size >= len(texts):
            return last_token_hidden_archived_protocol(
                lm, tok, texts, layer, device, max_length=max_length,
            ).float().cpu()
        chunks = []
        for i in range(0, len(texts), batch_size):
            chunk = last_token_hidden_archived_protocol(
                lm, tok, texts[i : i + batch_size], layer, device,
                max_length=max_length,
            ).float().cpu()
            chunks.append(chunk)
        return torch.cat(chunks, dim=0)

    h_orig = _collect(texts_o)
    h_pert = _collect(texts_p)
    return {"h_orig": h_orig, "h_pert": h_pert}


@torch.no_grad()
def batch_invariance_test(
    lm,
    tok,
    texts: list[str],
    layer: int,
    device: str,
) -> dict[str, float]:
    """Compare batch vs singleton hidden states — detect padding artifacts."""
    from activations import last_token_hidden

    h_batch = last_token_hidden(lm, tok, texts, layer, device).float().cpu()
    h_singles = []
    for t in texts:
        h_singles.append(last_token_hidden(lm, tok, [t], layer, device).float().cpu())
    h_single = torch.cat(h_singles, dim=0)
    diff = (h_batch - h_single).abs()
    rel = diff.norm().item() / (h_single.norm().item() + 1e-8)
    return {
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
        "relative_norm_diff": float(rel),
        "n_texts": len(texts),
    }


@torch.no_grad()
def evaluate_sae_on_hidden(
    sae,
    hidden: dict[str, torch.Tensor],
) -> dict[str, Any]:
    """Evaluate one SAE on precomputed hidden states."""
    sae_dev = next(sae.parameters()).device
    h_orig = hidden["h_orig"].to(sae_dev)
    h_pert = hidden["h_pert"].to(sae_dev)

    x_hat_o, z_o = sae(h_orig)
    x_hat_p, z_p = sae(h_pert)

    x_hat_o_f = x_hat_o.float()
    x_hat_p_f = x_hat_p.float()
    h_orig_f = h_orig.float()
    h_pert_f = h_pert.float()

    mse_orig = float(F.mse_loss(x_hat_o_f, h_orig_f).item())
    mse_pert = float(F.mse_loss(x_hat_p_f, h_pert_f).item())

    # Per-pair MSE (reduction='none' then mean over hidden dim)
    per_pair_mse_orig = (x_hat_o_f - h_orig_f).pow(2).mean(dim=-1).cpu().numpy()
    per_pair_mse_pert = (x_hat_p_f - h_pert_f).pow(2).mean(dim=-1).cpu().numpy()
    per_pair_mse = (per_pair_mse_orig + per_pair_mse_pert) / 2.0

    # Per-pair cosine similarity
    per_pair_cos_orig = F.cosine_similarity(x_hat_o_f, h_orig_f, dim=-1).cpu().numpy()
    per_pair_cos_pert = F.cosine_similarity(x_hat_p_f, h_pert_f, dim=-1).cpu().numpy()
    per_pair_cos = (per_pair_cos_orig + per_pair_cos_pert) / 2.0

    x_hat_o_np = x_hat_o_f.cpu().numpy()
    x_hat_p_np = x_hat_p_f.cpu().numpy()
    z_orig = z_o.float().cpu().numpy()
    z_pert = z_p.float().cpu().numpy()
    h_o_np = hidden["h_orig"].numpy()
    h_p_np = hidden["h_pert"].numpy()

    s = relative_sensitivity(z_orig, z_pert)
    g = input_normalised_gain(z_orig, z_pert, h_o_np, h_p_np)
    h_frac = hidden_perturbation_fraction(h_o_np, h_p_np)
    abs_dz = absolute_code_distance(z_orig, z_pert)
    z_o_norm = code_orig_norm(z_orig)
    decode_resp = decoded_perturbation_response(x_hat_o_np, x_hat_p_np)

    all_h = np.concatenate([h_o_np, h_p_np], axis=0)
    all_xhat = np.concatenate([x_hat_o_np, x_hat_p_np], axis=0)
    nmse_val = nmse(all_h, all_xhat)
    ev = explained_variance(all_h, all_xhat)
    cos_sim = cosine_similarity_batch(all_h, all_xhat)

    sp_orig = code_sparsity_stats(z_orig)
    sp_pert = code_sparsity_stats(z_pert)
    sp_all = code_sparsity_stats(np.concatenate([z_orig, z_pert], axis=0))

    return {
        "s": s,
        "g": g,
        "h_frac": h_frac,
        "abs_dz": abs_dz,
        "z_orig_norm": z_o_norm,
        "decode_resp": decode_resp,
        "per_pair_mse": per_pair_mse,
        "per_pair_cos": per_pair_cos,
        "mse_orig_mean": mse_orig,
        "mse_pert_mean": mse_pert,
        "mse_pair_mean": float((mse_orig + mse_pert) / 2.0),
        "nmse": nmse_val,
        "explained_variance": ev,
        "cosine_sim_mean": float(cos_sim.mean()),
        "cosine_sim_min": float(cos_sim.min()),
        "sparsity_orig": sp_orig,
        "sparsity_pert": sp_pert,
        "sparsity_all": sp_all,
    }


@torch.no_grad()
def verify_forward_matches_encode_decode(sae, h: torch.Tensor) -> float:
    sae_dev = next(sae.parameters()).device
    h = h.to(sae_dev)
    x_fwd, _ = sae(h)
    z = sae.encode(h)
    x_manual = sae.decode(z)
    return float((x_fwd - x_manual).abs().max().item())


def summarize_pair_arrays(arrays: dict[str, Any]) -> dict:
    s = arrays["s"]
    g = arrays["g"]
    abs_dz = arrays["abs_dz"]
    decode_resp = arrays["decode_resp"]
    sp = arrays["sparsity_all"]
    out: dict[str, Any] = {}
    out.update(summarize_distribution(s, prefix="s"))
    out.update(summarize_distribution(g, prefix="g"))
    out.update(summarize_distribution(abs_dz, prefix="abs_dz"))
    out.update(summarize_distribution(decode_resp, prefix="decode_resp"))
    out["V_gini_raw"] = gini_coefficient(s)
    out["V_gini_gain"] = gini_coefficient(g)
    out["h_frac_mean"] = float(np.mean(arrays["h_frac"]))
    out["h_frac_cv"] = float(
        np.std(arrays["h_frac"]) / (np.mean(arrays["h_frac"]) + 1e-8)
    )
    out["z_orig_norm_mean"] = float(np.mean(arrays["z_orig_norm"]))
    out["mse_orig_mean"] = arrays["mse_orig_mean"]
    out["mse_pert_mean"] = arrays["mse_pert_mean"]
    out["mse_pair_mean"] = arrays["mse_pair_mean"]
    out["nmse_mean"] = arrays["nmse"]
    out["explained_variance"] = arrays["explained_variance"]
    out["cosine_sim_mean"] = arrays["cosine_sim_mean"]
    out["cosine_sim_min"] = arrays["cosine_sim_min"]
    out["L0_mean"] = sp["L0_mean"]
    out["L0_orig_mean"] = arrays["sparsity_orig"]["L0_mean"]
    out["L0_pert_mean"] = arrays["sparsity_pert"]["L0_mean"]
    out["code_norm_mean"] = sp["code_norm_mean"]
    out["inactive_frac_mean"] = sp["inactive_frac_mean"]
    out["density_mean"] = sp["density_mean"]
    out["per_pair_s"] = [float(x) for x in s]
    out["per_pair_g"] = [float(x) for x in g]
    out["per_pair_abs_dz"] = [float(x) for x in abs_dz]
    out["per_pair_decode_resp"] = [float(x) for x in decode_resp]
    out["per_pair_z_orig_norm"] = [float(x) for x in arrays["z_orig_norm"]]
    out["per_pair_h_frac"] = [float(x) for x in arrays["h_frac"]]
    out["per_pair_mse"] = [float(x) for x in arrays["per_pair_mse"]]
    out["per_pair_cos"] = [float(x) for x in arrays["per_pair_cos"]]
    return out
