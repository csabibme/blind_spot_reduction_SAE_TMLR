#!/usr/bin/env python3
"""
E1 — Absolute sensitivity and lower-tail audit (v1.1).

Primary inference: ΔL20(s) = L20(s_vreg) - L20(s_std) per family.
Diagnostic: Standard-bottom paired lift (selection-biased — not primary).

See METRICS_SPEC.md v1.1.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np

REVISION_ROOT = Path(__file__).resolve().parents[1]
E1_ROOT = Path(__file__).resolve().parent
for p in (REVISION_ROOT, E1_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from eval_core import (  # noqa: E402
    clear_device_cache,
    collect_hidden_pairs,
    evaluate_sae_on_hidden,
    load_sae,
    module_param_info,
    setup_sae_scaling_imports,
    stable_family_seed,
    subsample_pairs,
    summarize_pair_arrays,
)
from shared.bootstrap import (  # noqa: E402
    cluster_bootstrap_ci,
    cluster_valid,
    default_cluster_ids,
    pair_bootstrap_ci,
    paired_delta_bootstrap_ci,
)
from shared.metrics import (  # noqa: E402
    gini_coefficient,
    lower_fraction_mean,
    paired_delta_summary,
    paired_lift_summary,
)
from shared.path_registry import (  # noqa: E402
    checkpoint_dir,
    load_manifest,
    pairs_path,
    repo_root,
    sae_scaling_root,
    verify_checkpoint,
)

PROFILE_RUNS: dict[str, tuple[str, str]] = {
    "gpt2": ("gpt2_standard_joint16_owt", "gpt2_vreg_joint16_owt"),
    "gemma-2-2b": ("gemma-2-2b_standard_joint16_owt", "gemma-2-2b_vreg_joint16_owt"),
    "qwen-2.5-3b": ("qwen-2.5-3b_standard_joint16_owt", "qwen-2.5-3b_vreg_joint16_owt"),
}


def atomic_write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _bootstrap_block(values: np.ndarray, cluster_ids: np.ndarray, seed: int) -> dict:
    is_valid = cluster_valid(cluster_ids, len(values))
    block = {
        "pair": {
            "mean_s": pair_bootstrap_ci(values, np.mean, seed=seed),
            "L20_s": pair_bootstrap_ci(
                values, lambda x: lower_fraction_mean(x, 0.20), seed=seed
            ),
            "V_gini": pair_bootstrap_ci(values, gini_coefficient, seed=seed),
        },
        "cluster_bootstrap_valid": is_valid,
    }
    if is_valid:
        block["cluster"] = {
            "mean_s": cluster_bootstrap_ci(values, cluster_ids, np.mean, seed=seed),
            "L20_s": cluster_bootstrap_ci(
                values, cluster_ids,
                lambda x: lower_fraction_mean(x, 0.20), seed=seed,
            ),
            "V_gini": cluster_bootstrap_ci(values, cluster_ids, gini_coefficient, seed=seed),
        }
    return block


def _paired_delta_bootstrap(
    std_arr: dict, vreg_arr: dict, cluster_ids: np.ndarray, seed: int,
) -> dict:
    s_std, s_vr = std_arr["s"], vreg_arr["s"]
    g_std, g_vr = std_arr["g"], vreg_arr["g"]
    dz_std, dz_vr = std_arr["abs_dz"], vreg_arr["abs_dz"]
    dr_std, dr_vr = std_arr["decode_resp"], vreg_arr["decode_resp"]

    def d_mean(a, b):
        return float(np.mean(b) - np.mean(a))

    def d_l20(a, b):
        return float(lower_fraction_mean(b, 0.20) - lower_fraction_mean(a, 0.20))

    def d_gini(a, b):
        return float(gini_coefficient(b) - gini_coefficient(a))

    return {
        "delta_mean_s": paired_delta_bootstrap_ci(s_std, s_vr, d_mean, cluster_ids, seed=seed),
        "delta_L20_s": paired_delta_bootstrap_ci(s_std, s_vr, d_l20, cluster_ids, seed=seed + 1),
        "delta_V_gini_raw": paired_delta_bootstrap_ci(s_std, s_vr, d_gini, cluster_ids, seed=seed + 2),
        "delta_L20_g": paired_delta_bootstrap_ci(g_std, g_vr, d_l20, cluster_ids, seed=seed + 3),
        "delta_mean_abs_dz": paired_delta_bootstrap_ci(dz_std, dz_vr, d_mean, cluster_ids, seed=seed + 4),
        "delta_L20_abs_dz": paired_delta_bootstrap_ci(dz_std, dz_vr, d_l20, cluster_ids, seed=seed + 5),
        "delta_mean_decode_resp": paired_delta_bootstrap_ci(dr_std, dr_vr, d_mean, cluster_ids, seed=seed + 6),
        "delta_L20_decode_resp": paired_delta_bootstrap_ci(dr_std, dr_vr, d_l20, cluster_ids, seed=seed + 7),
    }


def audit_profile(
    profile: str,
    families: list[str],
    max_pairs: int,
    device: str,
    lm_dtype: str,
    seed: int,
    manifest: dict,
    hidden_batch_size: int = 0,
    extraction_protocol: str = "archived",
    vreg_checkpoint_override: Path | None = None,
    std_checkpoint_override: Path | None = None,
    on_family_done: Callable[[str, dict], None] | None = None,
) -> dict:
    from lm_loader import load_model_and_tokenizer
    from perturbation_data import load_perturbation_families

    std_id, vreg_id = PROFILE_RUNS[profile]
    for cid in (std_id, vreg_id):
        if cid == vreg_id and vreg_checkpoint_override is not None:
            continue
        if cid == std_id and std_checkpoint_override is not None:
            continue
        verify_checkpoint(cid, manifest)

    std_ckpt = std_checkpoint_override or checkpoint_dir(std_id, manifest)
    vreg_ckpt = vreg_checkpoint_override or checkpoint_dir(vreg_id, manifest)

    model_cfg = manifest["models"][profile]
    pf = load_perturbation_families(pairs_path(manifest))
    if not families:
        families = sorted(k for k in pf if not k.startswith("_"))

    layer = model_cfg["hf_hidden_state_index"]
    runtime: dict = {
        "device": device,
        "lm_dtype_requested": lm_dtype,
        "hidden_batch_size": hidden_batch_size,
        "extraction_protocol": extraction_protocol,
        "tokenizer_padding_side": None,
    }

    # --- Phase 1: LM + hidden cache on CPU ---
    print(f"  [{profile}] Loading LM {model_cfg['model_id']} ...")
    lm, tok = load_model_and_tokenizer(
        model_cfg["model_id"], device, dtype=lm_dtype,
        trust_remote_code=model_cfg.get("trust_remote_code", False),
    )
    lm.eval()
    runtime["lm"] = module_param_info(lm)
    runtime["tokenizer_padding_side"] = getattr(tok, "padding_side", "unknown")
    print(f"  [{profile}] LM loaded: {runtime['lm']} padding={runtime['tokenizer_padding_side']}")

    hidden_cache: dict[str, dict] = {}
    family_pairs: dict[str, list] = {}
    family_pair_indices: dict[str, list[int]] = {}
    for fam in families:
        if fam not in pf:
            raise KeyError(f"Unknown family: {fam}")
        fam_seed = stable_family_seed(seed, fam)
        all_pairs = [tuple(p) for p in pf[fam]["pairs"]]
        pairs, indices = subsample_pairs(all_pairs, max_pairs, seed=fam_seed)
        family_pairs[fam] = pairs
        family_pair_indices[fam] = indices
        print(f"  [{profile}] Caching hidden: {fam} ({len(pairs)} pairs)")
        hidden_cache[fam] = collect_hidden_pairs(
            lm, tok, pairs, layer, device,
            batch_size=hidden_batch_size,
            extraction_protocol=extraction_protocol,
        )

    del lm, tok
    clear_device_cache(device)
    print(f"  [{profile}] LM unloaded, hidden cache on CPU ({len(hidden_cache)} families)")

    # --- Phase 2: Standard SAE ---
    print(f"  [{profile}] Loading Standard SAE ...")
    std_sae = load_sae(std_ckpt, device)
    std_sae.eval()
    runtime["sae_standard"] = module_param_info(std_sae)
    print(f"  [{profile}] Standard SAE loaded: {runtime['sae_standard']}")

    std_arrays: dict[str, dict] = {}
    for fam in families:
        std_arrays[fam] = evaluate_sae_on_hidden(std_sae, hidden_cache[fam])
    del std_sae
    clear_device_cache(device)
    print(f"  [{profile}] Standard SAE evaluated and unloaded")

    # --- Phase 3: V-reg SAE ---
    print(f"  [{profile}] Loading V-reg SAE ...")
    vreg_sae = load_sae(vreg_ckpt, device)
    vreg_sae.eval()
    runtime["sae_vreg"] = module_param_info(vreg_sae)
    print(f"  [{profile}] V-reg SAE loaded: {runtime['sae_vreg']}")

    family_results = {}
    for fam in families:
        print(f"  [{profile}] Evaluating V-reg / comparing: {fam}")
        vreg_arr = evaluate_sae_on_hidden(vreg_sae, hidden_cache[fam])
        std_arr = std_arrays[fam]

        cluster_ids = default_cluster_ids(len(family_pairs[fam]), fam)
        std_sum = summarize_pair_arrays(std_arr)
        vreg_sum = summarize_pair_arrays(vreg_arr)
        delta = paired_delta_summary(std_sum, vreg_sum)
        diagnostic = paired_lift_summary(std_arr["s"], vreg_arr["s"])

        family_results[fam] = {
            "n_pairs": len(family_pairs[fam]),
            "selected_pair_indices": family_pair_indices[fam],
            "standard": std_sum,
            "vreg": vreg_sum,
            "paired_delta": delta,
            "diagnostic_bottom_lift": diagnostic,
            "bootstrap": {
                "standard": _bootstrap_block(std_arr["s"], cluster_ids, seed),
                "vreg": _bootstrap_block(vreg_arr["s"], cluster_ids, seed + 1),
                "paired_delta": _paired_delta_bootstrap(
                    std_arr, vreg_arr, cluster_ids, seed + 2
                ),
            },
        }
        if on_family_done is not None:
            on_family_done(fam, family_results[fam])

    del vreg_sae, std_arrays, hidden_cache
    clear_device_cache(device)
    print(f"  [{profile}] V-reg SAE evaluated and unloaded")

    block = {
        "profile": profile,
        "model_id": model_cfg["model_id"],
        "layer": layer,
        "checkpoint_standard": (
            std_checkpoint_override.name if std_checkpoint_override is not None else std_id
        ),
        "checkpoint_vreg": (
            vreg_checkpoint_override.name if vreg_checkpoint_override is not None else vreg_id
        ),
        "checkpoint_standard_manifest_id": std_id,
        "checkpoint_vreg_manifest_id": vreg_id,
        "checkpoint_standard_override_used": std_checkpoint_override is not None,
        "checkpoint_vreg_override_used": vreg_checkpoint_override is not None,
        "checkpoint_standard_path": str(std_ckpt),
        "checkpoint_vreg_path": str(vreg_ckpt),
        "runtime": runtime,
        "families": family_results,
    }
    block["aggregate"] = aggregate_profile(block)
    return block


def aggregate_profile(profile_block: dict) -> dict:
    fams = profile_block["families"]
    deltas = [f["paired_delta"] for f in fams.values()]
    return {
        "mean_delta_L20_s": float(np.mean([d["delta_L20_s"] for d in deltas])),
        "frac_families_delta_L20_positive": float(np.mean([d["delta_L20_s"] > 0 for d in deltas])),
        "mean_delta_mean_s": float(np.mean([d["delta_mean_s"] for d in deltas])),
        "mean_delta_L20_g": float(np.mean([d["delta_L20_g"] for d in deltas])),
        "mean_delta_L20_abs_dz": float(np.mean([d["delta_L20_abs_dz"] for d in deltas])),
        "mean_delta_L20_decode_resp": float(np.mean([d["delta_L20_decode_resp"] for d in deltas])),
        "mean_delta_V_gini_raw": float(np.mean([d["delta_V_gini_raw"] for d in deltas])),
        "mean_mse_ratio": float(np.mean([d["mse_ratio"] for d in deltas])),
        "mean_nmse_std": float(np.mean([d["nmse_std"] for d in deltas])),
        "mean_nmse_vreg": float(np.mean([d["nmse_vreg"] for d in deltas])),
        "mean_code_norm_ratio": float(np.mean([d["code_norm_ratio"] for d in deltas])),
        "mean_std_L20": float(np.mean([f["standard"]["s_L20"] for f in fams.values()])),
        "mean_vreg_L20": float(np.mean([f["vreg"]["s_L20"] for f in fams.values()])),
        "mean_std_V_gini": float(np.mean([f["standard"]["V_gini_raw"] for f in fams.values()])),
        "mean_vreg_V_gini": float(np.mean([f["vreg"]["V_gini_raw"] for f in fams.values()])),
        "diagnostic_mean_bottom_lift": float(
            np.mean([f["diagnostic_bottom_lift"]["bottom_mean_lift"] for f in fams.values()])
        ),
    }


def write_markdown_table(audit: dict, out_path: Path) -> None:
    lines = [
        "# E1 absolute sensitivity audit",
        "",
        f"Generated: {audit['meta'].get('timestamp', 'unknown')}",
        f"Metrics spec: {audit['meta'].get('metrics_spec', 'unknown')}",
        f"Aggregation: unweighted mean across perturbation families.",
        "",
        "## Profile aggregates (primary: ΔL20)",
        "",
        "| Profile | ΔL20(s) | frac>0 | Δmean(s) | ΔL20(g) | ΔL20(|Δz|) | ΔL20(dec) | "
        "NMSE std | NMSE vreg | MSE ratio | norm ratio |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for profile, block in audit["profiles"].items():
        a = block["aggregate"]
        lines.append(
            f"| {profile} | {a['mean_delta_L20_s']:+.6f} | "
            f"{a['frac_families_delta_L20_positive']:.0%} | "
            f"{a['mean_delta_mean_s']:+.6f} | "
            f"{a['mean_delta_L20_g']:+.6f} | "
            f"{a['mean_delta_L20_abs_dz']:+.4f} | "
            f"{a['mean_delta_L20_decode_resp']:+.6f} | "
            f"{a['mean_nmse_std']:.6f} | "
            f"{a['mean_nmse_vreg']:.6f} | "
            f"{a['mean_mse_ratio']:.3f} | "
            f"{a['mean_code_norm_ratio']:.3f} |"
        )

    lines.extend(["", "## Per-family ΔL20(s) [primary]", ""])
    prof_names = list(audit["profiles"].keys())
    header = "| Family |" + "".join(f" {p} |" for p in prof_names)
    sep = "|---|" + "---:|" * len(prof_names)
    lines.extend([header, sep])
    all_fams = sorted({f for p in audit["profiles"].values() for f in p["families"]})
    for fam in all_fams:
        row = f"| {fam} |"
        for p in audit["profiles"].values():
            if fam in p["families"]:
                d = p["families"][fam]["paired_delta"]["delta_L20_s"]
                row += f" {d:+.6f} |"
            else:
                row += " — |"
        lines.append(row)

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="E1 absolute sensitivity audit")
    p.add_argument("--profile", choices=[*PROFILE_RUNS.keys(), "all"], default="all")
    p.add_argument("--families", nargs="*", default=None)
    p.add_argument("--max-pairs", type=int, default=50)
    p.add_argument("--hidden-batch-size", type=int, default=0,
                   help="LM microbatch size for hidden collection. 0=single batch.")
    p.add_argument(
        "--extraction-protocol",
        choices=["archived", "true_last"],
        default="archived",
        help="archived=E1 submitted (last_token_hidden); true_last=padding-aware last token",
    )
    p.add_argument("--device", default="auto")
    p.add_argument("--lm-dtype", default="float16")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-json", default=str(E1_ROOT / "results" / "absolute_sensitivity_audit.json"))
    p.add_argument("--output-md", default=str(E1_ROOT / "results" / "absolute_sensitivity_table.md"))
    p.add_argument("--vreg-checkpoint-override", type=Path, default=None)
    p.add_argument("--std-checkpoint-override", type=Path, default=None)
    args = p.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    manifest = load_manifest()
    setup_sae_scaling_imports(sae_scaling_root(manifest))
    from lm_loader import resolve_device

    device = resolve_device(args.device)
    profiles = list(PROFILE_RUNS) if args.profile == "all" else [args.profile]

    out_json = Path(args.output_json)
    audit = {
        "meta": {
            "experiment": "E1_absolute_sensitivity",
            "metrics_spec": "METRICS_SPEC.md v1.1",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "repo_root": str(repo_root()),
            "device": device,
            "lm_dtype": args.lm_dtype,
            "seed": args.seed,
            "max_pairs": args.max_pairs,
            "hidden_batch_size": args.hidden_batch_size,
            "extraction_protocol": args.extraction_protocol,
            "families_filter": args.families,
            "cluster_bootstrap_note": "cluster CI degenerate (all pairs = own cluster) until E5",
        },
        "profiles": {},
    }

    for profile in profiles:
        print(f"\n=== {profile} ===")

        def save_partial(_fam: str, _result: dict, _profile=profile) -> None:
            profile_block = audit["profiles"].setdefault(
                _profile, {"families": {}}
            )
            profile_block.setdefault("families", {})[_fam] = _result
            atomic_write_json(out_json, audit)

        block = audit_profile(
            profile, args.families or [], args.max_pairs,
            device, args.lm_dtype, args.seed, manifest,
            hidden_batch_size=args.hidden_batch_size,
            extraction_protocol=args.extraction_protocol,
            vreg_checkpoint_override=args.vreg_checkpoint_override,
            std_checkpoint_override=args.std_checkpoint_override,
            on_family_done=save_partial,
        )
        audit["profiles"][profile] = block
        atomic_write_json(out_json, audit)
        a = block["aggregate"]
        print(
            f"  aggregate: ΔL20={a['mean_delta_L20_s']:+.6f}  "
            f"frac>0={a['frac_families_delta_L20_positive']:.0%}  "
            f"MSE_ratio={a['mean_mse_ratio']:.3f}  "
            f"NMSE_std={a['mean_nmse_std']:.6f}  NMSE_vreg={a['mean_nmse_vreg']:.6f}"
        )

    write_markdown_table(audit, Path(args.output_md))
    print(f"\nSaved JSON -> {out_json}")
    print(f"Saved table -> {args.output_md}")


if __name__ == "__main__":
    main()
