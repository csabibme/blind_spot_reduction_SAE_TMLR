#!/usr/bin/env python3
"""Evaluate E7b baseline extension against E7a Standard/V-reg comparators."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REVISION_ROOT = Path(__file__).resolve().parents[1]
E1_ROOT = REVISION_ROOT / "E1_absolute_sensitivity"
SAE_ROOT = REVISION_ROOT.parent.parent / "SAE_scaling"
for path in (REVISION_ROOT, E1_ROOT, SAE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from eval_core import evaluate_sae_on_hidden, load_sae, summarize_pair_arrays  # noqa: E402
from shared.metrics import code_sparsity_stats  # noqa: E402

FREEZE_ID = "E7B_BASELINE_FREEZE_V1"
E7A_FREEZE_ID = "E7_FREEZE_V1"
PROFILES = ("gpt2", "gemma-2-2b", "qwen-2.5-3b")
SEEDS = ("seed_000", "seed_001", "seed_002")
OBJECTIVES = ("standard", "jumprelu", "mdl", "vreg")


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


def ckpt_dir(e7a_root: Path, e7b_root: Path, profile: str, seed: str, objective: str) -> Path:
    if objective in {"standard", "vreg"}:
        pattern = f"e7_{profile}_{objective}_{seed}_freeze_v1"
        path = e7a_root / profile / seed / objective / pattern
    else:
        pattern = f"e7b_{profile}_{objective}_{seed}_freeze_v1"
        path = e7b_root / profile / seed / objective / pattern
    if not (path / "sae.pt").is_file():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    return path


def load_perturb_cache(path: Path) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    cache = {
        family: {
            "h_orig": torch.as_tensor(pair[0], dtype=torch.float32),
            "h_pert": torch.as_tensor(pair[1], dtype=torch.float32),
        }
        for family, pair in payload["cache"].items()
    }
    meta = {key: value for key, value in payload.items() if key != "cache"}
    return cache, meta


@torch.no_grad()
def evaluate_owt(sae, owt_cache: Path, device: str, batch_size: int) -> dict[str, float]:
    payload = torch.load(owt_cache, map_location="cpu", weights_only=False)
    acts = payload["activations"].float()
    sae_device = next(sae.parameters()).device
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
    mean_x = sum_x / n_elem
    mean_res = sum_res / n_elem
    var_x = sum_x2 / n_elem - mean_x**2
    var_res = sum_res2 / n_elem - mean_res**2
    return {
        "nmse": float(sse / ssx),
        "explained_variance": float(1.0 - var_res / var_x),
        "L0_mean": float(np.mean(l0_values)),
        "density_mean": float(np.mean(density_values)),
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
    return {key: float(np.mean([row[key] for row in family_summaries.values()])) for key in keys}


def evaluate_checkpoint(
    path: Path,
    perturb_cache: dict[str, dict[str, torch.Tensor]],
    owt_cache: Path,
    device: str,
    owt_batch_size: int,
) -> dict[str, Any]:
    sae = load_sae(path, device)
    family = {}
    for family_name, hidden in perturb_cache.items():
        summary = summarize_pair_arrays(evaluate_sae_on_hidden(sae, hidden))
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


def objective_aggregate(rows: dict[str, dict[str, Any]], objective: str) -> dict[str, Any]:
    metrics = ["s_L20", "abs_dz_L20", "V_gini_raw", "g_L20"]
    owt_metrics = ["nmse", "explained_variance", "L0_mean"]
    out = {
        metric: mean_sd([rows[seed][objective]["metrics"]["macro"][metric] for seed in SEEDS])
        for metric in metrics
    }
    out.update({
        f"owt_{metric}": mean_sd([rows[seed][objective]["metrics"]["owt"][metric] for seed in SEEDS])
        for metric in owt_metrics
    })
    return out


def delta_vs_standard(rows: dict[str, dict[str, Any]], objective: str) -> dict[str, Any]:
    keys = ["s_L20", "abs_dz_L20", "V_gini_raw", "g_L20"]
    out = {}
    for key in keys:
        out[f"delta_{key}"] = mean_sd([
            rows[seed][objective]["metrics"]["macro"][key] -
            rows[seed]["standard"]["metrics"]["macro"][key]
            for seed in SEEDS
        ])
    positives = []
    for seed in SEEDS:
        std_family = rows[seed]["standard"]["metrics"]["family"]
        obj_family = rows[seed][objective]["metrics"]["family"]
        positives.append(float(np.mean([
            obj_family[family]["s_L20"] - std_family[family]["s_L20"] > 0
            for family in std_family
        ])))
    out["positive_family_fraction_L20_s"] = mean_sd(positives)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate E7b baseline extension")
    parser.add_argument("--e7a-root", type=Path, default=REVISION_ROOT / "E7_multiseed" / "runs" / "freeze_v1")
    parser.add_argument("--e7b-root", type=Path, default=REVISION_ROOT / "E7b_baseline_extension" / "runs" / "freeze_v1")
    parser.add_argument("--results-dir", type=Path, default=REVISION_ROOT / "E7b_baseline_extension" / "results")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--owt-batch-size", type=int, default=128)
    args = parser.parse_args()

    owt_caches = {
        "gpt2": REVISION_ROOT.parent / "tmlr_revision/prepare/experiment_101_hybrid_owt/data/owt_cache_gpt2_l12_25k.pt",
        "gemma-2-2b": REVISION_ROOT.parent / "tmlr_revision/prepare/experiment_101_hybrid_owt/data/owt_cache_gemma_l13_25k.pt",
        "qwen-2.5-3b": REVISION_ROOT.parent / "tmlr_revision/prepare/experiment_101_hybrid_owt/data/owt_cache_qwen_l18_25k.pt",
    }

    payload: dict[str, Any] = {
        "experiment": "E7b_baseline_extension",
        "freeze_id": FREEZE_ID,
        "status": "complete",
        "e7a_root": str(args.e7a_root),
        "e7b_root": str(args.e7b_root),
        "profiles": {},
        "headline": {},
    }
    for profile in PROFILES:
        print(f"=== Evaluating {profile} ===", flush=True)
        perturb_cache, perturb_meta = load_perturb_cache(args.e7a_root / "_caches" / f"{profile}_true_last_perturb_cache.pt")
        profile_block = {
            "perturb_cache_meta": perturb_meta,
            "checkpoints": {},
            "aggregate_by_objective": {},
            "delta_vs_standard": {},
        }
        for seed in SEEDS:
            profile_block["checkpoints"].setdefault(seed, {})
            for objective in OBJECTIVES:
                path = ckpt_dir(args.e7a_root, args.e7b_root, profile, seed, objective)
                meta = read_json(path / "meta.json")
                expected_freeze = E7A_FREEZE_ID if objective in {"standard", "vreg"} else FREEZE_ID
                if meta.get("freeze_id") != expected_freeze:
                    raise ValueError(f"Freeze mismatch for {path}")
                print(f"  {seed} {objective}", flush=True)
                profile_block["checkpoints"][seed][objective] = {
                    "checkpoint": str(path),
                    "meta": meta,
                    "metrics": evaluate_checkpoint(
                        path,
                        perturb_cache,
                        owt_caches[profile],
                        args.device,
                        args.owt_batch_size,
                    ),
                }
        for objective in OBJECTIVES:
            profile_block["aggregate_by_objective"][objective] = objective_aggregate(
                profile_block["checkpoints"],
                objective,
            )
            if objective != "standard":
                profile_block["delta_vs_standard"][objective] = delta_vs_standard(
                    profile_block["checkpoints"],
                    objective,
                )
        payload["profiles"][profile] = profile_block
        payload["headline"][profile] = {
            objective: profile_block["aggregate_by_objective"][objective]
            for objective in OBJECTIVES
        }

    out_json = args.results_dir / "multiseed_baseline.json"
    write_json(out_json, payload)

    lines = [
        "# E7b Multi-Seed Baseline Extension",
        "",
        f"Freeze: `{FREEZE_ID}`",
        "",
    ]
    for profile in PROFILES:
        lines.extend([
            f"## {profile}",
            "",
            "| Objective | L20(s) mean±SD | L20(abs dz) mean±SD | V-Gini mean±SD | OWT NMSE mean±SD | OWT L0 mean±SD |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for objective in OBJECTIVES:
            agg = payload["profiles"][profile]["aggregate_by_objective"][objective]

            def fmt(key: str) -> str:
                return f"{agg[key]['mean']:.4f} ± {agg[key]['sd']:.4f}"

            lines.append(
                f"| {objective} | {fmt('s_L20')} | {fmt('abs_dz_L20')} | "
                f"{fmt('V_gini_raw')} | {fmt('owt_nmse')} | {fmt('owt_L0_mean')} |"
            )
        lines.append("")
        lines.extend([
            "| Objective vs Standard | ΔL20(s) | ΔL20(abs dz) | ΔV-Gini | Positive-family fraction |",
            "|---|---:|---:|---:|---:|",
        ])
        for objective in ("jumprelu", "mdl", "vreg"):
            delta = payload["profiles"][profile]["delta_vs_standard"][objective]

            def dfmt(key: str) -> str:
                return f"{delta[key]['mean']:+.4f} ± {delta[key]['sd']:.4f}"

            lines.append(
                f"| {objective} | {dfmt('delta_s_L20')} | {dfmt('delta_abs_dz_L20')} | "
                f"{dfmt('delta_V_gini_raw')} | {dfmt('positive_family_fraction_L20_s')} |"
            )
        lines.append("")

    out_md = args.results_dir / "multiseed_baseline_table.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved JSON -> {out_json}")
    print(f"Saved Markdown -> {out_md}")


if __name__ == "__main__":
    main()
