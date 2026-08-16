#!/usr/bin/env python3
"""Evaluate E7 freeze_v1 multi-seed checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

REVISION_ROOT = Path(__file__).resolve().parents[1]
E1_ROOT = REVISION_ROOT / "E1_absolute_sensitivity"
REPO_ROOT = REVISION_ROOT.parent.parent
SAE_ROOT = REPO_ROOT / "SAE_scaling"
for path in (REVISION_ROOT, E1_ROOT, SAE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from eval_core import evaluate_sae_on_hidden, load_sae, summarize_pair_arrays  # noqa: E402
from shared.metrics import code_sparsity_stats, lower_fraction_mean  # noqa: E402

FREEZE_ID = "E7_FREEZE_V1"
PROFILES = ("gpt2", "gemma-2-2b", "qwen-2.5-3b")
SEEDS = ("seed_000", "seed_001", "seed_002")
OBJECTIVES = ("standard", "vreg")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def mean_sd(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "sd": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "n": int(len(arr)),
    }


def metric_delta(vreg: dict[str, float], standard: dict[str, float], key: str) -> float:
    return float(vreg[key] - standard[key])


def checkpoint_dir(root: Path, profile: str, seed_label: str, objective: str) -> Path:
    pattern = f"e7_{profile}_{objective}_{seed_label}_freeze_v1"
    path = root / profile / seed_label / objective / pattern
    if not (path / "sae.pt").is_file():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    return path


def load_perturb_cache(path: Path) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    cache = {}
    for family, (h_orig, h_pert) in payload["cache"].items():
        cache[family] = {
            "h_orig": torch.as_tensor(h_orig, dtype=torch.float32),
            "h_pert": torch.as_tensor(h_pert, dtype=torch.float32),
        }
    meta = {key: value for key, value in payload.items() if key != "cache"}
    return cache, meta


@torch.no_grad()
def evaluate_owt(sae, owt_cache: Path, device: str, batch_size: int) -> dict[str, float]:
    payload = torch.load(owt_cache, map_location="cpu", weights_only=False)
    acts = payload["activations"].float()
    sae_device = next(sae.parameters()).device
    if str(sae_device) != device:
        device = str(sae_device)

    sse = 0.0
    ssx = 0.0
    n_elem = 0
    sum_x = 0.0
    sum_x2 = 0.0
    sum_res = 0.0
    sum_res2 = 0.0
    l0_values = []
    density_values = []
    code_norm_values = []
    for start in range(0, len(acts), batch_size):
        h = acts[start : start + batch_size].to(device)
        x_hat, z = sae(h)
        residual = (h.float() - x_hat.float()).cpu().numpy()
        h_np = h.float().cpu().numpy()
        z_np = z.float().cpu().numpy()
        sse += float(np.sum(residual**2))
        ssx += float(np.sum(h_np**2))
        n_elem += int(h_np.size)
        sum_x += float(np.sum(h_np))
        sum_x2 += float(np.sum(h_np**2))
        sum_res += float(np.sum(residual))
        sum_res2 += float(np.sum(residual**2))
        sp = code_sparsity_stats(z_np)
        l0_values.append(sp["L0_mean"])
        density_values.append(sp["density_mean"])
        code_norm_values.append(sp["code_norm_mean"])

    mean_x = sum_x / n_elem
    mean_res = sum_res / n_elem
    var_x = sum_x2 / n_elem - mean_x**2
    var_res = sum_res2 / n_elem - mean_res**2
    return {
        "nmse": float(sse / ssx) if ssx > 0 else float("nan"),
        "explained_variance": float(1.0 - var_res / var_x) if var_x > 0 else float("nan"),
        "L0_mean": float(np.mean(l0_values)),
        "density_mean": float(np.mean(density_values)),
        "code_norm_mean": float(np.mean(code_norm_values)),
    }


def macro_metrics(family_summaries: dict[str, dict[str, float]]) -> dict[str, float]:
    keys = [
        "s_L20",
        "abs_dz_L20",
        "V_gini_raw",
        "g_L20",
        "s_mean",
        "abs_dz_mean",
        "L0_mean",
        "density_mean",
        "nmse_mean",
        "explained_variance",
    ]
    return {
        key: float(np.mean([summary[key] for summary in family_summaries.values()]))
        for key in keys
    }


def evaluate_checkpoint(
    ckpt_dir: Path,
    perturb_cache: dict[str, dict[str, torch.Tensor]],
    owt_cache: Path,
    device: str,
    owt_batch_size: int,
) -> dict[str, Any]:
    sae = load_sae(ckpt_dir, device)
    sae.eval()
    family = {}
    for family_name, hidden in perturb_cache.items():
        arrays = evaluate_sae_on_hidden(sae, hidden)
        summary = summarize_pair_arrays(arrays)
        # Per-pair arrays are large and not needed for E7 headline JSON.
        for key in list(summary):
            if key.startswith("per_pair_"):
                summary.pop(key)
        family[family_name] = summary
    macro = macro_metrics(family)
    owt = evaluate_owt(sae, owt_cache, device, owt_batch_size)
    del sae
    if device == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()
    return {"family": family, "macro": macro, "owt": owt}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate E7 freeze_v1 checkpoints")
    parser.add_argument("--run-root", type=Path, default=REVISION_ROOT / "E7_multiseed" / "runs" / "freeze_v1")
    parser.add_argument("--results-dir", type=Path, default=REVISION_ROOT / "E7_multiseed" / "results")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--owt-batch-size", type=int, default=128)
    args = parser.parse_args()

    owt_caches = {
        "gpt2": REVISION_ROOT.parent / "tmlr_revision/prepare/experiment_101_hybrid_owt/data/owt_cache_gpt2_l12_25k.pt",
        "gemma-2-2b": REVISION_ROOT.parent / "tmlr_revision/prepare/experiment_101_hybrid_owt/data/owt_cache_gemma_l13_25k.pt",
        "qwen-2.5-3b": REVISION_ROOT.parent / "tmlr_revision/prepare/experiment_101_hybrid_owt/data/owt_cache_qwen_l18_25k.pt",
    }
    perturb_caches = {
        profile: args.run_root / "_caches" / f"{profile}_true_last_perturb_cache.pt"
        for profile in PROFILES
    }

    payload: dict[str, Any] = {
        "experiment": "E7_multiseed",
        "freeze_id": FREEZE_ID,
        "status": "complete",
        "run_root": str(args.run_root),
        "profiles": {},
        "headline": {},
    }

    for profile in PROFILES:
        print(f"=== Evaluating {profile} ===", flush=True)
        perturb_cache, perturb_meta = load_perturb_cache(perturb_caches[profile])
        profile_block = {
            "perturb_cache": str(perturb_caches[profile]),
            "perturb_cache_meta": perturb_meta,
            "checkpoints": {},
            "seed_deltas": {},
            "aggregate": {},
        }
        for seed_label in SEEDS:
            profile_block["checkpoints"].setdefault(seed_label, {})
            for objective in OBJECTIVES:
                ckpt = checkpoint_dir(args.run_root, profile, seed_label, objective)
                meta = read_json(ckpt / "meta.json")
                if meta.get("freeze_id") != FREEZE_ID:
                    raise ValueError(f"Freeze mismatch in {ckpt}")
                print(f"  {seed_label} {objective}", flush=True)
                profile_block["checkpoints"][seed_label][objective] = {
                    "checkpoint": str(ckpt),
                    "meta": meta,
                    "metrics": evaluate_checkpoint(
                        ckpt,
                        perturb_cache,
                        owt_caches[profile],
                        args.device,
                        args.owt_batch_size,
                    ),
                }

            std = profile_block["checkpoints"][seed_label]["standard"]["metrics"]
            vr = profile_block["checkpoints"][seed_label]["vreg"]["metrics"]
            deltas = {
                "delta_L20_s": metric_delta(vr["macro"], std["macro"], "s_L20"),
                "delta_L20_abs_dz": metric_delta(vr["macro"], std["macro"], "abs_dz_L20"),
                "delta_V_gini_raw": metric_delta(vr["macro"], std["macro"], "V_gini_raw"),
                "delta_L20_g": metric_delta(vr["macro"], std["macro"], "g_L20"),
                "delta_owt_nmse": metric_delta(vr["owt"], std["owt"], "nmse"),
                "delta_owt_ev": metric_delta(vr["owt"], std["owt"], "explained_variance"),
                "delta_owt_L0": metric_delta(vr["owt"], std["owt"], "L0_mean"),
            }
            family_positive = []
            for family_name in perturb_cache:
                f_std = std["family"][family_name]
                f_vr = vr["family"][family_name]
                family_positive.append(f_vr["s_L20"] - f_std["s_L20"] > 0)
            deltas["positive_family_fraction_L20_s"] = float(np.mean(family_positive))
            profile_block["seed_deltas"][seed_label] = deltas

        for key in next(iter(profile_block["seed_deltas"].values())):
            profile_block["aggregate"][key] = mean_sd([
                profile_block["seed_deltas"][seed][key]
                for seed in SEEDS
            ])
        payload["profiles"][profile] = profile_block
        payload["headline"][profile] = profile_block["aggregate"]

    out_json = args.results_dir / "multiseed_headline.json"
    write_json(out_json, payload)

    lines = [
        "# E7 Multi-Seed Headline",
        "",
        f"Freeze: `{FREEZE_ID}`",
        "",
        "| Model | ΔL20(s) mean±SD | ΔL20(abs dz) mean±SD | ΔV-Gini mean±SD | Positive-family fraction | ΔOWT NMSE mean±SD | ΔOWT EV mean±SD |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for profile in PROFILES:
        agg = payload["headline"][profile]

        def fmt(key: str) -> str:
            return f"{agg[key]['mean']:+.4f} ± {agg[key]['sd']:.4f}"

        lines.append(
            f"| {profile} | {fmt('delta_L20_s')} | {fmt('delta_L20_abs_dz')} | "
            f"{fmt('delta_V_gini_raw')} | {fmt('positive_family_fraction_L20_s')} | "
            f"{fmt('delta_owt_nmse')} | {fmt('delta_owt_ev')} |"
        )
    out_md = args.results_dir / "multiseed_headline_table.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved JSON -> {out_json}")
    print(f"Saved Markdown -> {out_md}")


if __name__ == "__main__":
    main()
