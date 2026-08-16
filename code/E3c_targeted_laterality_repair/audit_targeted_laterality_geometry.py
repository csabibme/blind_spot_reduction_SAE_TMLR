#!/usr/bin/env python3
"""Direct geometry and fidelity audit for E3c targeted laterality repair."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

REVISION_ROOT = Path(__file__).resolve().parents[1]
SAE_ROOT = REVISION_ROOT.parent.parent / "SAE_scaling"
E3_ROOT = REVISION_ROOT / "E3_heldout_probes"
for path in (SAE_ROOT, E3_ROOT, REVISION_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_e3_negation_probes import deduplicate_examples, read_json, sha256_file  # noqa: E402
from sae_model_v2 import StandardSAE, load_any_sae  # noqa: E402
from shared.metrics import code_sparsity_stats, cosine_similarity_batch, explained_variance, nmse  # noqa: E402


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def lower_tail_mean(values: np.ndarray, frac: float = 0.20) -> float:
    ordered = np.sort(values.astype(np.float64))
    k = max(1, int(np.ceil(frac * len(ordered))))
    return float(np.mean(ordered[:k]))


@torch.no_grad()
def encode_decode(sae, hidden: np.ndarray, device: str, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    codes = []
    recons = []
    for start in range(0, len(hidden), batch_size):
        h = torch.as_tensor(hidden[start : start + batch_size], dtype=torch.float32, device=device)
        x_hat, z = sae(h)
        codes.append(z.float().cpu().numpy())
        recons.append(x_hat.float().cpu().numpy())
    return np.concatenate(codes, axis=0), np.concatenate(recons, axis=0)


def pair_indices(examples: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    by_pair: dict[str, dict[str, Any]] = defaultdict(dict)
    for index, example in enumerate(examples):
        if example["split"] != split:
            continue
        entry = by_pair[example["global_pair_id"]]
        entry["report_id"] = example["report_id"]
        entry["template_id"] = example["template_id"]
        entry[example["label"]] = index
    rows = []
    for pair_id, entry in sorted(by_pair.items()):
        if "left" not in entry or "right" not in entry:
            continue
        rows.append(
            {
                "global_pair_id": pair_id,
                "report_id": entry["report_id"],
                "template_id": entry["template_id"],
                "left_index": entry["left"],
                "right_index": entry["right"],
            }
        )
    return rows


def distance_rows(codes: np.ndarray, pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for pair in pairs:
        z_left = codes[pair["left_index"]]
        z_right = codes[pair["right_index"]]
        delta = z_right - z_left
        denom = 0.5 * (np.linalg.norm(z_left) + np.linalg.norm(z_right)) + 1e-8
        rows.append(
            {
                "global_pair_id": pair["global_pair_id"],
                "report_id": pair["report_id"],
                "template_id": pair["template_id"],
                "distance": float(np.linalg.norm(delta)),
                "relative_distance": float(np.linalg.norm(delta) / denom),
                "left_code_norm": float(np.linalg.norm(z_left)),
                "right_code_norm": float(np.linalg.norm(z_right)),
            }
        )
    return rows


def summarize_distances(rows: list[dict[str, Any]]) -> dict[str, float]:
    distance = np.asarray([row["distance"] for row in rows], dtype=np.float64)
    relative = np.asarray([row["relative_distance"] for row in rows], dtype=np.float64)
    norms = np.asarray(
        [0.5 * (row["left_code_norm"] + row["right_code_norm"]) for row in rows],
        dtype=np.float64,
    )
    return {
        "n_pairs": len(rows),
        "distance_mean": float(np.mean(distance)),
        "distance_median": float(np.median(distance)),
        "distance_l20": lower_tail_mean(distance),
        "relative_distance_mean": float(np.mean(relative)),
        "relative_distance_median": float(np.median(relative)),
        "relative_distance_l20": lower_tail_mean(relative),
        "mean_pair_code_norm": float(np.mean(norms)),
        "median_pair_code_norm": float(np.median(norms)),
    }


def anchored_tail_delta(
    standard_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    key: str,
) -> dict[str, Any]:
    std_by_id = {row["global_pair_id"]: row for row in standard_rows}
    cand_by_id = {row["global_pair_id"]: row for row in candidate_rows}
    common = sorted(set(std_by_id) & set(cand_by_id))
    ordered = sorted(common, key=lambda pair_id: std_by_id[pair_id][key])
    k = max(1, int(np.ceil(0.2 * len(ordered))))
    tail = ordered[:k]
    deltas = np.asarray([cand_by_id[pair_id][key] - std_by_id[pair_id][key] for pair_id in tail], dtype=np.float64)
    all_deltas = np.asarray([cand_by_id[pair_id][key] - std_by_id[pair_id][key] for pair_id in common], dtype=np.float64)
    return {
        "tail_size": k,
        "tail_delta_mean": float(np.mean(deltas)),
        "all_pair_delta_mean": float(np.mean(all_deltas)),
        "all_pair_delta_median": float(np.median(all_deltas)),
        "fraction_improved": float(np.mean(all_deltas > 0)),
    }


def fidelity(hidden: np.ndarray, recon: np.ndarray, codes: np.ndarray) -> dict[str, float]:
    mse = float(F.mse_loss(torch.as_tensor(recon), torch.as_tensor(hidden, dtype=torch.float32)).item())
    sparsity = code_sparsity_stats(codes)
    return {
        "mse": mse,
        "nmse": nmse(hidden, recon),
        "explained_variance": explained_variance(hidden, recon),
        "cosine_mean": float(cosine_similarity_batch(hidden, recon).mean()),
        "code_norm_mean": sparsity["code_norm_mean"],
        "L0_mean": sparsity["L0_mean"],
        "density_mean": sparsity["density_mean"],
    }


def load_owt_sample(path: Path, n_samples: int, seed: int) -> np.ndarray:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    acts = payload["activations"].float().numpy()
    rng = np.random.default_rng(seed)
    n = min(n_samples, len(acts))
    indices = rng.choice(len(acts), size=n, replace=False)
    return acts[indices]


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit E3c targeted laterality geometry/fidelity")
    parser.add_argument("--profile", choices=("gpt2", "gemma-2-2b", "qwen-2.5-3b"), default="gpt2")
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--feature-cache-npz", type=Path, required=True)
    parser.add_argument("--target-checkpoint", type=Path, required=True)
    parser.add_argument("--owt-cache", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--owt-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    split_payload = read_json(args.split_json)
    examples = deduplicate_examples(split_payload["examples"])
    with np.load(args.feature_cache_npz, allow_pickle=True) as data:
        hidden = data["hidden"].astype(np.float32)
        cache_meta = data["metadata"].item()
    standard_checkpoint = Path(cache_meta["standard_checkpoint"])
    vreg_checkpoint = Path(cache_meta["vreg_checkpoint"])
    owt_hidden = load_owt_sample(args.owt_cache, args.owt_samples, args.seed).astype(np.float32)

    saes = {
        "standard": load_any_sae(standard_checkpoint, device=args.device),
        "zero_shot_vreg": load_any_sae(vreg_checkpoint, device=args.device),
        "targeted": StandardSAE.load_checkpoint(args.target_checkpoint, device=args.device),
    }
    codes: dict[str, np.ndarray] = {}
    recons: dict[str, np.ndarray] = {}
    owt_codes: dict[str, np.ndarray] = {}
    owt_recons: dict[str, np.ndarray] = {}
    for name, sae in saes.items():
        sae.eval()
        codes[name], recons[name] = encode_decode(sae, hidden, args.device, args.batch_size)
        owt_codes[name], owt_recons[name] = encode_decode(sae, owt_hidden, args.device, args.batch_size)

    payload: dict[str, Any] = {
        "experiment": "E3c_targeted_laterality_repair",
        "analysis": "direct_geometry_and_fidelity_audit",
        "status": "complete",
        "split_json": str(args.split_json),
        "split_sha256": sha256_file(args.split_json),
        "feature_cache_npz": str(args.feature_cache_npz),
        "feature_cache_metadata": cache_meta,
        "profile": args.profile,
        "standard_checkpoint": str(standard_checkpoint),
        "vreg_checkpoint": str(vreg_checkpoint),
        "target_checkpoint": str(args.target_checkpoint),
        "owt_cache": str(args.owt_cache),
        "representations": {},
    }
    for split in ("train", "dev", "test"):
        pairs = pair_indices(examples, split)
        payload["representations"].setdefault("splits", {})[split] = {"n_pairs": len(pairs)}
        for name in saes:
            rows = distance_rows(codes[name], pairs)
            block = payload["representations"].setdefault(name, {})
            block.setdefault("geometry", {})[split] = {
                "summary": summarize_distances(rows),
                "rows": rows if split == "test" else [],
            }
            block.setdefault("fidelity_openi", {})[split] = fidelity(
                hidden[[idx for pair in pairs for idx in (pair["left_index"], pair["right_index"])]],
                recons[name][[idx for pair in pairs for idx in (pair["left_index"], pair["right_index"])]],
                codes[name][[idx for pair in pairs for idx in (pair["left_index"], pair["right_index"])]],
            )
        std_rows = distance_rows(codes["standard"], pairs)
        for name in ("zero_shot_vreg", "targeted"):
            cand_rows = distance_rows(codes[name], pairs)
            payload["representations"][name].setdefault("geometry_delta_vs_standard", {})[split] = {
                "distance": anchored_tail_delta(std_rows, cand_rows, "distance"),
                "relative_distance": anchored_tail_delta(std_rows, cand_rows, "relative_distance"),
            }
    for name in saes:
        payload["representations"][name]["fidelity_owt"] = fidelity(owt_hidden, owt_recons[name], owt_codes[name])

    write_json(args.output_json, payload)
    lines = [
        "# E3c Direct Geometry And Fidelity Audit",
        "",
        f"Target checkpoint: `{args.target_checkpoint}`",
        "",
        "## Test Geometry",
        "",
        "| Representation | Distance L20 | Relative L20 | Distance mean | Relative mean | Mean code norm |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ("standard", "zero_shot_vreg", "targeted"):
        summary = payload["representations"][name]["geometry"]["test"]["summary"]
        lines.append(
            f"| {name} | {summary['distance_l20']:.4f} | {summary['relative_distance_l20']:.4f} | "
            f"{summary['distance_mean']:.4f} | {summary['relative_distance_mean']:.4f} | "
            f"{summary['mean_pair_code_norm']:.4f} |"
        )
    lines.extend(["", "## Test Delta Vs Standard", ""])
    lines.append("| Candidate | Distance tail delta | Distance mean delta | Distance improved frac | Relative tail delta | Relative improved frac |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for name in ("zero_shot_vreg", "targeted"):
        dist = payload["representations"][name]["geometry_delta_vs_standard"]["test"]["distance"]
        rel = payload["representations"][name]["geometry_delta_vs_standard"]["test"]["relative_distance"]
        lines.append(
            f"| {name} | {dist['tail_delta_mean']:+.4f} | {dist['all_pair_delta_mean']:+.4f} | "
            f"{dist['fraction_improved']:.3f} | {rel['tail_delta_mean']:+.4f} | {rel['fraction_improved']:.3f} |"
        )
    lines.extend(["", "## Fidelity", ""])
    lines.append("| Representation | OWT NMSE | OWT EV | OWT L0 | OpenI test NMSE | OpenI test EV | OpenI test L0 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for name in ("standard", "zero_shot_vreg", "targeted"):
        owt = payload["representations"][name]["fidelity_owt"]
        test = payload["representations"][name]["fidelity_openi"]["test"]
        lines.append(
            f"| {name} | {owt['nmse']:.4f} | {owt['explained_variance']:.4f} | {owt['L0_mean']:.1f} | "
            f"{test['nmse']:.4f} | {test['explained_variance']:.4f} | {test['L0_mean']:.1f} |"
        )
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved JSON -> {args.output_json}")
    print(f"Saved Markdown -> {args.output_md}")


if __name__ == "__main__":
    main()
