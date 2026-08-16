#!/usr/bin/env python3
"""Compute E5 family geometry statistics from profile-specific hidden caches."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REVISION_ROOT = Path(__file__).resolve().parents[1]
E5_ROOT = Path(__file__).resolve().parent
for p in (REVISION_ROOT, E5_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from template_cluster_utils import lexical_stats  # noqa: E402


def json_safe(obj):
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    return obj


def parse_profile_cache(items: list[str]) -> dict[str, Path]:
    out = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected PROFILE=PATH, got {item!r}")
        profile, path = item.split("=", 1)
        if profile in out:
            raise ValueError(f"Duplicate profile cache: {profile}")
        out[profile] = Path(path)
    return out


def load_cache(path: Path) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, Any]]:
    payload = torch.load(path.expanduser().resolve(), map_location="cpu", weights_only=False)
    cache = {}
    for family, (h_orig, h_pert) in payload["cache"].items():
        cache[family] = (
            torch.as_tensor(h_orig).float().numpy(),
            torch.as_tensor(h_pert).float().numpy(),
        )
    return cache, {k: v for k, v in payload.items() if k != "cache"}


def load_clusters(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))["families"]


def cosine_diversity(deltas: np.ndarray) -> dict[str, float | int | None]:
    norms = np.linalg.norm(deltas, axis=1)
    nonzero = norms > 1e-12
    zero_count = int((~nonzero).sum())
    usable = deltas[nonzero]
    n = usable.shape[0]
    if n < 2:
        return {
            "zero_delta_count": zero_count,
            "mean_pairwise_cosine": None,
            "cosine_diversity": None,
            "pairwise_cosine_q05": None,
            "pairwise_cosine_q95": None,
        }
    unit = usable / np.linalg.norm(usable, axis=1, keepdims=True)
    sim = unit @ unit.T
    tri = sim[np.triu_indices(n, k=1)]
    return {
        "zero_delta_count": zero_count,
        "mean_pairwise_cosine": float(np.mean(tri)),
        "cosine_diversity": float(1.0 - np.mean(tri)),
        "pairwise_cosine_q05": float(np.quantile(tri, 0.05)),
        "pairwise_cosine_q95": float(np.quantile(tri, 0.95)),
    }


def _spectral_from_matrix(matrix: np.ndarray, prefix: str, r_max: int) -> dict[str, float | None]:
    if matrix.shape[0] == 0 or r_max <= 0:
        return {
            f"{prefix}_effective_rank": None,
            f"{prefix}_effective_rank_norm": None,
            f"{prefix}_spectral_entropy": None,
            f"{prefix}_spectral_entropy_norm": None,
            f"{prefix}_top1_power_fraction": None,
            f"{prefix}_top5_power_fraction": None,
        }
    s = np.linalg.svd(matrix, full_matrices=False, compute_uv=False)
    power = s**2
    total = power.sum()
    if total <= 1e-12:
        return {
            f"{prefix}_effective_rank": 0.0,
            f"{prefix}_effective_rank_norm": 0.0,
            f"{prefix}_spectral_entropy": 0.0,
            f"{prefix}_spectral_entropy_norm": 0.0,
            f"{prefix}_top1_power_fraction": None,
            f"{prefix}_top5_power_fraction": None,
        }
    probs = power / total
    entropy = float(-(probs * np.log(np.clip(probs, 1e-12, None))).sum())
    eff = float(math.exp(entropy))
    log_r = math.log(max(2, r_max))
    return {
        f"{prefix}_effective_rank": eff,
        f"{prefix}_effective_rank_norm": float(eff / r_max),
        f"{prefix}_spectral_entropy": entropy,
        f"{prefix}_spectral_entropy_norm": float(entropy / log_r),
        f"{prefix}_top1_power_fraction": float(probs[0]),
        f"{prefix}_top5_power_fraction": float(probs[:5].sum()),
    }


def spectral_stats(deltas: np.ndarray) -> dict[str, float | None]:
    n, d = deltas.shape
    centered = deltas - deltas.mean(axis=0, keepdims=True)
    centered_r_max = min(max(0, n - 1), d)
    uncentered_r_max = min(n, d)
    out = {}
    out.update(_spectral_from_matrix(centered, "centered", centered_r_max))
    out.update(_spectral_from_matrix(deltas, "uncentered", uncentered_r_max))
    return out


def cv(values: np.ndarray) -> float:
    return float(np.std(values) / (np.mean(values) + 1e-12))


def hidden_subset_ablation(values: np.ndarray, sizes: list[int], n_repeats: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    out = {}
    n = len(values)
    for size in sizes:
        if size > n:
            continue
        means, l20s, cvs = [], [], []
        for _ in range(n_repeats):
            idx = rng.choice(n, size=size, replace=False)
            sub = np.sort(values[idx])
            k = max(1, math.ceil(0.2 * size))
            means.append(float(np.mean(sub)))
            l20s.append(float(np.mean(sub[:k])))
            cvs.append(cv(sub))
        out[str(size)] = {
            "n_repeats": n_repeats,
            "hidden_delta_norm_mean_mean": float(np.mean(means)),
            "hidden_delta_norm_mean_std": float(np.std(means)),
            "hidden_delta_norm_L20_mean": float(np.mean(l20s)),
            "hidden_delta_norm_L20_std": float(np.std(l20s)),
            "hidden_delta_norm_cv_mean": float(np.mean(cvs)),
            "hidden_delta_norm_cv_std": float(np.std(cvs)),
        }
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Analyze E5 family geometry")
    p.add_argument(
        "--profile-cache",
        nargs="+",
        required=True,
        help="One or more PROFILE=PATH hidden caches",
    )
    p.add_argument("--template-clusters", type=Path, default=E5_ROOT / "results" / "template_clusters.json")
    p.add_argument("--subset-sizes", type=int, nargs="*", default=[10, 20, 30, 50])
    p.add_argument("--n-repeats", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-json", type=Path, default=E5_ROOT / "results" / "family_geometry_profiles.json")
    p.add_argument("--output-md", type=Path, default=E5_ROOT / "results" / "family_geometry_profiles.md")
    p.add_argument(
        "--hidden-subset-output-json",
        type=Path,
        default=E5_ROOT / "results" / "hidden_geometry_subset_ablation.json",
    )
    args = p.parse_args()

    profile_caches = parse_profile_cache(args.profile_cache)
    clusters = load_clusters(args.template_clusters)
    results = {"template_clusters": str(args.template_clusters), "profiles": {}}
    subset_results = {"template_clusters": str(args.template_clusters), "profiles": {}}

    md = [
        "# E5 Family Geometry",
        "",
        "| Profile | Family | pairs | templates | h Δ mean | h Δ CV | cosine div. | centered eff. rank | centered H norm | zero Δh |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    cluster_family_set = set(clusters)
    for profile, path in sorted(profile_caches.items()):
        cache, cache_meta = load_cache(path)
        if set(cache) != cluster_family_set:
            raise ValueError(
                f"Family-set mismatch for {profile}: cache={sorted(cache)} clusters={sorted(cluster_family_set)}"
            )
        results["profiles"][profile] = {"hidden_cache": str(path), "cache_meta": cache_meta, "families": {}}
        subset_results["profiles"][profile] = {"hidden_cache": str(path), "families": {}}

        for family in sorted(cache):
            h_orig, h_pert = cache[family]
            deltas = h_pert - h_orig
            norms = np.linalg.norm(deltas, axis=1)
            fam_clusters = clusters[family]
            values = []
            for rec in fam_clusters["pairs"]:
                values.extend([rec["orig_value"], rec["pert_value"]])
            lex = lexical_stats(values)
            geom = {
                "n_pairs": int(len(norms)),
                "n_template_clusters": int(fam_clusters["n_template_clusters"]),
                "cluster_quality": fam_clusters.get("cluster_quality", {}),
                "template_cluster_sizes": fam_clusters["template_cluster_sizes"],
                "hidden_delta_norm_mean": float(np.mean(norms)),
                "hidden_delta_norm_std": float(np.std(norms)),
                "hidden_delta_norm_cv": cv(norms),
                "hidden_delta_norm_min": float(np.min(norms)),
                "hidden_delta_norm_max": float(np.max(norms)),
                "lexical_value_stats": lex,
            }
            geom.update(cosine_diversity(deltas))
            geom.update(spectral_stats(deltas))
            results["profiles"][profile]["families"][family] = geom
            subset_results["profiles"][profile]["families"][family] = hidden_subset_ablation(
                norms, args.subset_sizes, args.n_repeats, args.seed
            )
            md.append(
                f"| {profile} | {family} | {geom['n_pairs']} | {geom['n_template_clusters']} | "
                f"{geom['hidden_delta_norm_mean']:.3f} | {geom['hidden_delta_norm_cv']:.3f} | "
                f"{geom['cosine_diversity'] if geom['cosine_diversity'] is not None else 'NA'} | "
                f"{geom['centered_effective_rank'] if geom['centered_effective_rank'] is not None else 'NA'} | "
                f"{geom['centered_spectral_entropy_norm'] if geom['centered_spectral_entropy_norm'] is not None else 'NA'} | "
                f"{geom['zero_delta_count']} |"
            )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(json_safe(results), indent=2, allow_nan=False), encoding="utf-8")
    args.hidden_subset_output_json.write_text(
        json.dumps(json_safe(subset_results), indent=2, allow_nan=False), encoding="utf-8",
    )
    args.output_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Saved -> {args.output_json}")
    print(f"Saved -> {args.hidden_subset_output_json}")
    print(f"Saved -> {args.output_md}")


if __name__ == "__main__":
    main()
