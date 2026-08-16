#!/usr/bin/env python3
"""E1-style lower-tail and Standard-defined weak-set audit on real OpenI pairs.

Reuses E1's exact metric + extraction code (`FINAL/REVISION_1/eval_core.py`,
`shared/metrics.py`, `shared/bootstrap.py`) so numbers are directly comparable to
E1 — E1 itself is untouched. Runs the frozen general joint-16 Standard vs V-reg
checkpoints (experiment_101, zero-shot, not task-tuned) on the OpenI radiology
perturbation pairs (`REVISION_1/OpenI/pairs.jsonl`).

The dated protocol in this directory defines the added paired diagnostics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_external_root = os.environ.get("SAE_REPO_ROOT")
SAE_ROOT = (
    Path(_external_root).resolve()
    if _external_root
    else next(
        (
            parent for parent in HERE.parents
            if (parent / "FINAL" / "REVISION_1").is_dir()
        ),
        HERE.parents[1],
    )
)
REV1 = SAE_ROOT / "FINAL/REVISION_1"
SAE_SCALING = SAE_ROOT / "SAE_scaling"
for _p in (REV1, REV1 / "E1_absolute_sensitivity", SAE_SCALING):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from shared.bootstrap import paired_delta_bootstrap_ci  # noqa: E402
from shared.metrics import gini_coefficient, lower_fraction_mean  # noqa: E402

EXP101 = SAE_ROOT / "FINAL/tmlr_revision/prepare/experiment_101_hybrid_owt/checkpoints"
# corrected true_last V-reg override for gemma (matches E2 protocol)
GEMMA_VREG_OVERRIDE = (SAE_ROOT / "FINAL/REVISION_1/E1R_gemma_protocol_repair/checkpoints"
                       / "gemma-2-2b/joint/vreg_joint16_owt_true_last")


def public_repo_path(path: Path) -> str:
    """Return a repository-relative public path without local user details."""
    resolved = path.resolve()
    try:
        return f"<REPO_ROOT>/{resolved.relative_to(SAE_ROOT.resolve()).as_posix()}"
    except ValueError:
        return str(path)


def public_checkpoint_path(path: Path) -> str:
    """Return a checkpoint-root-relative public path when possible."""
    resolved = path.resolve()
    try:
        return f"<CKPT_ROOT>/{resolved.relative_to(EXP101.resolve()).as_posix()}"
    except ValueError:
        return public_repo_path(path)


PROFILES = {
    "gpt2": ("gpt2", 12, False),
    "qwen-2.5-3b": ("Qwen/Qwen2.5-3B", 18, True),
    "gemma-2-2b": ("google/gemma-2-2b", 13, True),
}
DEFAULT_MAX_LENGTHS = {
    "gpt2": 256,
    "qwen-2.5-3b": 128,
    "gemma-2-2b": 256,
}
DEFAULT_FAMILIES = ["negation", "laterality", "severity", "anatomical_direction"]
N_BOOT = 5000
PROTOCOL_DATE = "2026-08-06"
PROTOCOL_FILE = "OPENI_WEAK_SET_PROTOCOL_2026-08-06.md"
BOOTSTRAP_SEEDS = {
    "legacy_own_tail": 100,
    "fixed_w_std": 200,
    "selection_aware_w_std": 300,
    "reverse_w_vreg": 400,
}


def stable_family_seed(base_seed: int, family: str) -> int:
    """Match E1's stable SHA-256-derived family seed exactly."""
    digest = hashlib.sha256(family.encode("utf-8")).digest()
    return base_seed + int.from_bytes(digest[:4], "big")


def subsample_pairs(
    pairs: list[tuple[str, str]], max_pairs: int, seed: int,
) -> tuple[list[tuple[str, str]], list[int]]:
    """Match E1's sorted seeded subsample exactly."""
    all_indices = list(range(len(pairs)))
    if max_pairs <= 0 or max_pairs >= len(pairs):
        return pairs, all_indices
    idx = sorted(random.Random(seed).sample(all_indices, max_pairs))
    return [pairs[i] for i in idx], idx


def load_openi(path: Path, families: list[str]):
    by_fam: dict[str, list[tuple[str, str]]] = {}
    img_by_fam: dict[str, list[str]] = {}
    row_by_fam: dict[str, list[int]] = {}
    id_by_fam: dict[str, list[str]] = {}
    fam_counts: dict[str, int] = {}
    with path.open(encoding="utf-8") as fh:
        for row_index, line in enumerate(fh):
            d = json.loads(line)
            fam = d.get("family_id")
            if fam not in families:
                continue
            family_index = fam_counts.get(fam, 0)
            fam_counts[fam] = family_index + 1
            by_fam.setdefault(fam, []).append((d["text_orig"], d["text_pert"]))
            img_by_fam.setdefault(fam, []).append(str(d.get("image_id", "?")))
            row_by_fam.setdefault(fam, []).append(row_index)
            id_by_fam.setdefault(fam, []).append(f"{fam}:{family_index:04d}")
    return by_fam, img_by_fam, row_by_fam, id_by_fam


def d_l20(a, b):
    return float(lower_fraction_mean(b, 0.20) - lower_fraction_mean(a, 0.20))


def d_mean(a, b):
    return float(np.mean(b) - np.mean(a))


def d_gini(a, b):
    return float(gini_coefficient(b) - gini_coefficient(a))


def stable_bottom_indices(values: np.ndarray, fraction: float = 0.20) -> np.ndarray:
    """Bottom ceil(fraction*n), with original index as deterministic tie-break."""
    arr = np.asarray(values)
    n_keep = int(math.ceil(fraction * len(arr)))
    return np.argsort(arr, kind="stable")[:n_keep]


def verify_worst_subset_invariant(
    standard: np.ndarray,
    vreg: np.ndarray,
    w_std: np.ndarray,
    tolerance: float = 1e-12,
) -> dict[str, float | bool]:
    """Check the exact fixed-Standard-set consequence of own-tail L20."""
    standard = np.asarray(standard, dtype=np.float64)
    vreg = np.asarray(vreg, dtype=np.float64)
    fixed_delta = float(np.mean(vreg[w_std] - standard[w_std]))
    own_tail_delta = d_l20(standard, vreg)
    slack = fixed_delta - own_tail_delta
    if slack < -tolerance:
        raise AssertionError(
            "WORST_SUBSET_INVARIANT_FAILURE: fixed Standard-weak-set delta "
            "must be >= own-tail L20 delta under identical universe, weighting, "
            "response definition, and tail cardinality. "
            f"fixed_delta={fixed_delta:.17g}, "
            f"own_tail_delta={own_tail_delta:.17g}, slack={slack:.17g}"
        )
    return {
        "fixed_delta": fixed_delta,
        "own_tail_delta": own_tail_delta,
        "slack": slack,
        "passed": True,
        "tolerance": tolerance,
    }


def standard_quintile_indices(values: np.ndarray) -> list[np.ndarray]:
    """Five exhaustive Standard-ranked groups with deterministic tie handling."""
    order = np.argsort(np.asarray(values), kind="stable")
    return [np.asarray(x, dtype=np.int64) for x in np.array_split(order, 5)]


def _bootstrap_draw_indices(
    cluster_ids: np.ndarray, rng: np.random.Generator,
) -> tuple[np.ndarray, str]:
    """Draw paired indexes; use pairs only when all clusters are singletons."""
    clusters = np.asarray(cluster_ids)
    unique, counts = np.unique(clusters, return_counts=True)
    if np.all(counts == 1):
        return rng.integers(0, len(clusters), size=len(clusters)), "pair"
    cluster_to_idx = {c: np.flatnonzero(clusters == c) for c in unique}
    chosen = rng.choice(unique, size=len(unique), replace=True)
    return np.concatenate([cluster_to_idx[c] for c in chosen]), "cluster"


def _panel_values(
    rel_std: np.ndarray,
    rel_vreg: np.ndarray,
    abs_std: np.ndarray,
    abs_vreg: np.ndarray,
    indexes: np.ndarray,
) -> dict[str, float]:
    idx = np.asarray(indexes, dtype=np.int64)
    return {
        "relative_std": float(np.mean(rel_std[idx])),
        "relative_vreg": float(np.mean(rel_vreg[idx])),
        "relative_delta": float(np.mean(rel_vreg[idx] - rel_std[idx])),
        "absolute_std": float(np.mean(abs_std[idx])),
        "absolute_vreg": float(np.mean(abs_vreg[idx])),
        "absolute_delta": float(np.mean(abs_vreg[idx] - abs_std[idx])),
    }


def _add_bootstrap_intervals(
    point: dict[str, float], boots: dict[str, np.ndarray], kind: str,
) -> dict:
    n_boot = len(next(iter(boots.values())))
    out = {"bootstrap_kind": kind, "n_boot": n_boot}
    for key, value in point.items():
        lo, hi = np.quantile(boots[key], [0.025, 0.975])
        out[key] = {
            "mean": value,
            "ci": [float(lo), float(hi)],
        }
    return out


def fixed_set_panel(
    rel_std: np.ndarray,
    rel_vreg: np.ndarray,
    abs_std: np.ndarray,
    abs_vreg: np.ndarray,
    cluster_ids: np.ndarray,
    indexes: np.ndarray,
    *,
    seed: int,
    n_boot: int = N_BOOT,
) -> dict:
    """Paired panel; resampling is restricted to clusters in the fixed set."""
    idx = np.asarray(indexes, dtype=np.int64)
    point = _panel_values(rel_std, rel_vreg, abs_std, abs_vreg, idx)
    rng = np.random.default_rng(seed)
    boots = {key: np.empty(n_boot, dtype=np.float64) for key in point}
    kind = "pair"
    for b in range(n_boot):
        local, kind = _bootstrap_draw_indices(np.asarray(cluster_ids)[idx], rng)
        sampled = idx[local]
        values = _panel_values(rel_std, rel_vreg, abs_std, abs_vreg, sampled)
        for key, value in values.items():
            boots[key][b] = value
    out = _add_bootstrap_intervals(point, boots, kind)
    out["n"] = int(len(idx))
    out["n_clusters"] = int(len(np.unique(np.asarray(cluster_ids)[idx])))
    return out


def selection_aware_panel(
    rel_std: np.ndarray,
    rel_vreg: np.ndarray,
    abs_std: np.ndarray,
    abs_vreg: np.ndarray,
    cluster_ids: np.ndarray,
    *,
    seed: int,
    n_boot: int = N_BOOT,
) -> dict:
    """Resample the full family and reselect W_std in every resample."""
    w_std = stable_bottom_indices(rel_std)
    point = _panel_values(rel_std, rel_vreg, abs_std, abs_vreg, w_std)
    rng = np.random.default_rng(seed)
    boots = {key: np.empty(n_boot, dtype=np.float64) for key in point}
    kind = "pair"
    for b in range(n_boot):
        sampled, kind = _bootstrap_draw_indices(cluster_ids, rng)
        selected_local = stable_bottom_indices(rel_std[sampled])
        selected = sampled[selected_local]
        values = _panel_values(rel_std, rel_vreg, abs_std, abs_vreg, selected)
        for key, value in values.items():
            boots[key][b] = value
    out = _add_bootstrap_intervals(point, boots, kind)
    out["n_observed_w_std"] = int(len(w_std))
    out["resampling_frame"] = "full_family_then_reselect_w_std"
    return out


def quintile_profile(
    rel_std: np.ndarray,
    rel_vreg: np.ndarray,
    abs_std: np.ndarray,
    abs_vreg: np.ndarray,
    cluster_ids: np.ndarray,
    *,
    seed: int,
    n_boot: int = N_BOOT,
) -> dict[str, dict]:
    return {
        f"Q{i}": fixed_set_panel(
            rel_std,
            rel_vreg,
            abs_std,
            abs_vreg,
            cluster_ids,
            idx,
            seed=seed + i,
            n_boot=n_boot,
        )
        for i, idx in enumerate(standard_quintile_indices(rel_std), start=1)
    }


def main() -> None:
    from eval_core import collect_hidden_pairs, evaluate_sae_on_hidden, load_sae
    from lm_loader import load_model_and_tokenizer, resolve_device

    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True, choices=list(PROFILES))
    ap.add_argument("--families", nargs="*", default=DEFAULT_FAMILIES)
    ap.add_argument("--max-pairs", type=int, default=300)
    ap.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Token cap; defaults to the profile's canonical Table 9 value.",
    )
    ap.add_argument("--hidden-batch-size", type=int, default=16)
    ap.add_argument("--lm-dtype", default="float16")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pairs", type=Path, default=REV1 / "OpenI/pairs.jsonl")
    ap.add_argument("--out", type=Path, default=HERE / "results/openi_lowertail.json")
    args = ap.parse_args()

    model_id, layer, trc = PROFILES[args.profile]
    max_length = (
        args.max_length
        if args.max_length is not None
        else DEFAULT_MAX_LENGTHS[args.profile]
    )
    ckdir = args.profile
    std_ck = EXP101 / ckdir / "joint/standard_joint16_owt"
    vreg_ck = EXP101 / ckdir / "joint/vreg_joint16_owt"
    if args.profile == "gemma-2-2b" and GEMMA_VREG_OVERRIDE.is_dir():
        vreg_ck = GEMMA_VREG_OVERRIDE

    device = resolve_device(args.device)
    by_fam, img_by_fam, row_by_fam, id_by_fam = load_openi(
        args.pairs, args.families,
    )
    families = [f for f in args.families if f in by_fam]

    # --- Phase 1: cache hidden states per family ---
    lm, tok = load_model_and_tokenizer(model_id, device, dtype=args.lm_dtype,
                                       trust_remote_code=trc)
    lm.eval()
    hidden, sel_imgs, sel_rows, sel_ids = {}, {}, {}, {}
    for fam in families:
        fam_seed = stable_family_seed(args.seed, fam)
        pairs, idx = subsample_pairs(by_fam[fam], args.max_pairs, seed=fam_seed)
        sel_imgs[fam] = [img_by_fam[fam][i] for i in idx]
        sel_rows[fam] = [row_by_fam[fam][i] for i in idx]
        sel_ids[fam] = [id_by_fam[fam][i] for i in idx]
        print(f"  [{args.profile}] hidden: {fam} ({len(pairs)} pairs)", flush=True)
        hidden[fam] = collect_hidden_pairs(lm, tok, pairs, layer, device,
                                           batch_size=args.hidden_batch_size,
                                           extraction_protocol="true_last",
                                           max_length=max_length)
    del lm, tok

    # --- Phase 2/3: Standard then V-reg SAE ---
    std_sae = load_sae(std_ck, device)
    std_arr = {fam: evaluate_sae_on_hidden(std_sae, hidden[fam]) for fam in families}
    del std_sae
    vreg_sae = load_sae(vreg_ck, device)
    vreg_arr = {fam: evaluate_sae_on_hidden(vreg_sae, hidden[fam]) for fam in families}
    del vreg_sae

    fam_res = {}
    for fam in families:
        s_std = np.asarray(std_arr[fam]["s"], dtype=np.float64)
        s_vr = np.asarray(vreg_arr[fam]["s"], dtype=np.float64)
        dz_std = np.asarray(std_arr[fam]["abs_dz"], dtype=np.float64)
        dz_vr = np.asarray(vreg_arr[fam]["abs_dz"], dtype=np.float64)
        clusters = np.array(sel_imgs[fam])
        boot = paired_delta_bootstrap_ci(
            s_std, s_vr, d_l20, clusters, n_boot=N_BOOT,
            seed=BOOTSTRAP_SEEDS["legacy_own_tail"],
        )
        ci = boot.get("cluster", boot["pair"])
        w_std = stable_bottom_indices(s_std)
        w_vreg = stable_bottom_indices(s_vr)
        worst_subset_check = verify_worst_subset_invariant(
            s_std, s_vr, w_std,
        )
        w_std_set, w_vreg_set = set(w_std.tolist()), set(w_vreg.tolist())
        fixed = fixed_set_panel(
            s_std, s_vr, dz_std, dz_vr, clusters, w_std,
            seed=BOOTSTRAP_SEEDS["fixed_w_std"],
        )
        reverse = fixed_set_panel(
            s_std, s_vr, dz_std, dz_vr, clusters, w_vreg,
            seed=BOOTSTRAP_SEEDS["reverse_w_vreg"],
        )
        selection_aware = selection_aware_panel(
            s_std, s_vr, dz_std, dz_vr, clusters,
            seed=BOOTSTRAP_SEEDS["selection_aware_w_std"],
        )
        quintiles = standard_quintile_indices(s_std)
        quintile_membership = np.empty(len(s_std), dtype=np.int64)
        for q, q_idx in enumerate(quintiles, start=1):
            quintile_membership[q_idx] = q
        per_pair = []
        for i in range(len(s_std)):
            per_pair.append({
                "selected_index": i,
                "pair_id": sel_ids[fam][i],
                "source_row_index": int(sel_rows[fam][i]),
                "cluster_id": str(clusters[i]),
                "relative_D_std": float(s_std[i]),
                "relative_D_vreg": float(s_vr[i]),
                "absolute_dz_std": float(dz_std[i]),
                "absolute_dz_vreg": float(dz_vr[i]),
                "in_W_std": i in w_std_set,
                "in_W_vreg": i in w_vreg_set,
                "standard_quintile": f"Q{quintile_membership[i]}",
            })
        fam_res[fam] = {
            "n_pairs": int(len(s_std)),
            "n_reports": int(len(np.unique(clusters))),
            "L20_std": round(lower_fraction_mean(s_std, 0.20), 6),
            "L20_vreg": round(lower_fraction_mean(s_vr, 0.20), 6),
            "delta_L20_s": round(d_l20(s_std, s_vr), 6),
            "delta_L20_ci": [round(ci["lo"], 6), round(ci["hi"], 6)],
            "ci_kind": "cluster" if "cluster" in boot else "pair",
            "V_gini_std": round(gini_coefficient(s_std), 4),
            "V_gini_vreg": round(gini_coefficient(s_vr), 4),
            "delta_mean_s": round(d_mean(s_std, s_vr), 6),
            "nmse_std": round(float(std_arr[fam]["nmse"]), 6),
            "nmse_vreg": round(float(vreg_arr[fam]["nmse"]), 6),
            "own_tail": {
                "estimand": "separately_selected_bottom_ceil_20_percent",
                "n_std": int(len(w_std)),
                "n_vreg": int(len(w_vreg)),
                "L20_std": round(lower_fraction_mean(s_std, 0.20), 6),
                "L20_vreg": round(lower_fraction_mean(s_vr, 0.20), 6),
                "delta": round(d_l20(s_std, s_vr), 6),
                "delta_ci": [round(ci["lo"], 6), round(ci["hi"], 6)],
                "ci_kind": "cluster" if "cluster" in boot else "pair",
            },
            "W_std": {
                "definition": "bottom_ceil_20_percent_by_relative_D_std",
                "indexes": [int(i) for i in w_std],
                "pair_ids": [sel_ids[fam][i] for i in w_std],
                "paired_panel": fixed,
                "selection_aware_bootstrap_sensitivity": selection_aware,
                "fraction_improved": {
                    "relative_D": float(np.mean(s_vr[w_std] > s_std[w_std])),
                    "absolute_dz": float(np.mean(dz_vr[w_std] > dz_std[w_std])),
                },
            },
            "reverse_W_vreg_diagnostic": {
                "definition": "bottom_ceil_20_percent_by_relative_D_vreg",
                "indexes": [int(i) for i in w_vreg],
                "pair_ids": [sel_ids[fam][i] for i in w_vreg],
                "paired_panel": reverse,
            },
            "weak_set_overlap": {
                "n_intersection": len(w_std_set & w_vreg_set),
                "fraction_of_each_weak_set": (
                    len(w_std_set & w_vreg_set) / len(w_std_set)
                ),
                "jaccard": (
                    len(w_std_set & w_vreg_set) / len(w_std_set | w_vreg_set)
                ),
            },
            "worst_subset_invariant": worst_subset_check,
            "standard_defined_quintile_profile": quintile_profile(
                s_std, s_vr, dz_std, dz_vr, clusters,
                seed=500,
            ),
            "per_pair": per_pair,
            "interpretation": (
                "Diagnostic/noncausal: selection is coupled to observed model response."
            ),
        }

    dl20 = [fam_res[f]["delta_L20_s"] for f in families]
    macro = {
        "mean_delta_L20_s": round(float(np.mean(dl20)), 6),
        "frac_families_positive": round(float(np.mean([d > 0 for d in dl20])), 4),
        "n_families": len(families),
    }
    out = {
        "audit_version": "openi_standard_weak_set_v1",
        "protocol_date": PROTOCOL_DATE,
        "protocol_file": public_repo_path(HERE / PROTOCOL_FILE),
        "profile": args.profile,
        "model_id": model_id,
        "layer": layer,
        "std_ckpt": public_checkpoint_path(std_ck),
        "vreg_ckpt": public_checkpoint_path(vreg_ck),
        "input_pairs": public_repo_path(args.pairs),
        "max_pairs": args.max_pairs,
        "max_length": max_length,
        "lm_dtype": args.lm_dtype,
        "requested_device": args.device,
        "resolved_device": str(device),
        "extraction_protocol": "true_last",
        "hidden_batch_size": args.hidden_batch_size,
        "seed": args.seed,
        "family_subsampling_seeds": {
            fam: stable_family_seed(args.seed, fam) for fam in families
        },
        "bootstrap": {
            "n_boot": N_BOOT,
            "seeds": BOOTSTRAP_SEEDS,
            "cluster": "OpenI image/report ID",
            "pair_fallback": "only_when_all_clusters_in_frame_are_singletons",
        },
        "families": fam_res,
        "macro": macro,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    print(f"\n=== {args.profile}: OpenI real-clinical lower-tail (ΔL20 of s) ===", flush=True)
    print(f"{'family':22} {'n':>4} {'rep':>4} {'L20_std':>9} {'L20_vreg':>9} "
          f"{'ΔL20':>9} {'95% CI':>20} {'Vg_std':>7} {'Vg_vreg':>7}", flush=True)
    for fam in families:
        r = fam_res[fam]
        print(f"{fam:22} {r['n_pairs']:>4} {r['n_reports']:>4} {r['L20_std']:>9.5f} "
              f"{r['L20_vreg']:>9.5f} {r['delta_L20_s']:>+9.5f} "
              f"[{r['delta_L20_ci'][0]:+.4f},{r['delta_L20_ci'][1]:+.4f}] "
              f"{r['V_gini_std']:>7.3f} {r['V_gini_vreg']:>7.3f}", flush=True)
    print(f"macro ΔL20(s)={macro['mean_delta_L20_s']:+.5f}  "
          f"frac families>0={macro['frac_families_positive']:.0%}", flush=True)
    print("(no-harm NMSE per family in JSON)", flush=True)


if __name__ == "__main__":
    main()
