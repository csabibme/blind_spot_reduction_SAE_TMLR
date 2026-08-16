#!/usr/bin/env python3
"""Run toy v2 ablation over d_sae, lambda_v, and noise_std; aggregate regime screening."""

from __future__ import annotations

import argparse
import copy
import itertools
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from run_toy_experiment import aggregate, run_one, write_markdown


def variant_name(d_sae: int, lambda_v: float, noise_std: float) -> str:
    lam = f"{lambda_v:g}".replace(".", "p")
    noise = f"{noise_std:g}".replace(".", "p")
    return f"d_sae{d_sae}_lam{lam}_noise{noise}"


# Fixed, rule-based selection (applied uniformly to every point; see PROTOCOL.md).
SELECTION_RULE = {
    "hidden_auroc_min": 0.70,
    "standard_auroc_max": 0.75,
    "delta_auroc_min": 0.0,
    "delta_l20_abs_min": 0.0,
    "mse_ratio_max": 1.15,
}
# Stronger "ideal" bar for the preferred main-text point.
IDEAL_RULE = {
    "hidden_auroc_min": 0.75,
    "delta_auroc_min": 0.03,
}
# Design target band for the main-text point, stated up front as the toy design goal:
# a genuinely present-but-weak signal (hidden AUROC in [0.75, 0.90]) with a sub-ceiling
# Standard code (Standard AUROC in [0.60, 0.75]). Among rule-qualifying points, the
# manuscript reports the point in this band.
MAIN_TEXT_BAND = {
    "hidden_min": 0.75,
    "hidden_max": 0.90,
    "standard_min": 0.60,
    "standard_max": 0.75,
}


def in_main_text_band(row: Dict[str, Any]) -> bool:
    h = row["hidden_probe_auroc_mean"]
    s = row["standard_probe_auroc_mean"]
    return (
        MAIN_TEXT_BAND["hidden_min"] <= h <= MAIN_TEXT_BAND["hidden_max"]
        and MAIN_TEXT_BAND["standard_min"] <= s <= MAIN_TEXT_BAND["standard_max"]
    )


def pick_main_text(rows: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    """Select the main-text point: among rule-qualifying rows, the one inside the design
    target band (smaller MSE ratio breaks ties). Falls back to the top qualifying row, then
    to the best available row, so a point is always defined and the choice is deterministic."""
    qualifying = [r for r in rows if r["qualifies"]]
    band = [r for r in qualifying if in_main_text_band(r)]
    if band:
        return sorted(band, key=lambda r: r["mse_ratio"])[0]
    if qualifying:
        return qualifying[0]
    return rows[0] if rows else None


def regime_status(row: Dict[str, float]) -> Dict[str, Any]:
    """Apply the fixed, rule-based selection to a seed-averaged (config, alpha) row."""
    h = row["hidden_probe_auroc_mean"]
    s = row["standard_probe_auroc_mean"]
    d = row["delta_probe_auroc_mean"]
    l20 = row["delta_critical_l20_abs_mean"]
    mse_ratio = row["vreg_mse_test_mean"] / max(row["standard_mse_test_mean"], 1e-8)

    checks = {
        "hidden_present": h >= SELECTION_RULE["hidden_auroc_min"],
        "standard_subceiling": s < SELECTION_RULE["standard_auroc_max"],
        "delta_auroc_positive": d > SELECTION_RULE["delta_auroc_min"],
        "delta_l20_positive": l20 > SELECTION_RULE["delta_l20_abs_min"],
        "mse_ratio_ok": mse_ratio <= SELECTION_RULE["mse_ratio_max"],
    }
    qualifies = all(checks.values())
    ideal = qualifies and h >= IDEAL_RULE["hidden_auroc_min"] and d >= IDEAL_RULE["delta_auroc_min"]
    fail_reasons = [k for k, v in checks.items() if not v]
    return {
        "qualifies": qualifies,
        "ideal": ideal,
        "mse_ratio": mse_ratio,
        "checks": checks,
        "fail_reasons": fail_reasons,
    }


def rank_key(row: Dict[str, Any]) -> Tuple:
    """Sort key: ideal first, then qualifying, then tie-break by smaller lambda_v, smaller MSE ratio."""
    st = row["_status"]
    return (
        0 if st["ideal"] else 1,
        0 if st["qualifies"] else 1,
        float(row["lambda_v"]),
        st["mse_ratio"],
        -row["delta_probe_auroc_mean"],
    )


def collect_rows(variant_summaries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for vs in variant_summaries:
        cfg = vs["config"]
        for alpha_str, m in vs["summary"]["by_alpha"].items():
            row = {
                "variant": vs["variant"],
                "d_sae": cfg["d_sae"],
                "lambda_v": cfg["lambda_v"],
                "noise_std": cfg["noise_std"],
                "alpha": float(alpha_str),
                **m,
            }
            row["_status"] = regime_status(row)
            row["mse_ratio"] = row["_status"]["mse_ratio"]
            row["qualifies"] = row["_status"]["qualifies"]
            row["ideal"] = row["_status"]["ideal"]
            rows.append(row)
    rows.sort(key=rank_key)
    return rows


def build_ablation_summary(variant_summaries: List[Dict[str, Any]], out_path: Path) -> None:
    rows = collect_rows(variant_summaries)
    qualifying = [r for r in rows if r["qualifies"]]
    best = pick_main_text(rows)

    lines = [
        "# Toy v2 ablation — rule-based regime screening",
        "",
        "Fixed selection rule (applied uniformly to all points): hidden AUROC ≥ 0.70, Standard AUROC < 0.75, "
        "Δ AUROC > 0, Δ L20(‖Δz‖) > 0, MSE ratio ≤ 1.15. Main-text point: among qualifying points, the one in "
        "the design target band (hidden 0.75–0.90, Standard 0.60–0.75), smaller MSE ratio breaking ties. "
        "Ideal bar: hidden ≥ 0.75 and Δ AUROC ≥ 0.03.",
        "",
        f"Qualifying points: **{len(qualifying)} / {len(rows)}**.",
        "",
        "## Ranked candidates (qualifying first, then tie-break)",
        "",
        "| rank | variant | α | hidden | Standard | V-reg | Δ AUROC | Δ L20 | MSE ratio | qualifies | ideal | fails |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|:--:|:--:|---|",
    ]
    for i, r in enumerate(rows[:25], start=1):
        st = r["_status"]
        lines.append(
            f"| {i} | {r['variant']} | {r['alpha']:.2f} | {r['hidden_probe_auroc_mean']:.4f} | "
            f"{r['standard_probe_auroc_mean']:.4f} | {r['vreg_probe_auroc_mean']:.4f} | "
            f"{r['delta_probe_auroc_mean']:+.4f} | {r['delta_critical_l20_abs_mean']:+.4f} | "
            f"{r['mse_ratio']:.3f} | {'yes' if r['qualifies'] else 'no'} | {'yes' if r['ideal'] else 'no'} | "
            f"{','.join(st['fail_reasons']) if st['fail_reasons'] else '-'} |"
        )

    if best:
        tag = "IDEAL" if best["ideal"] else ("qualifying" if best["qualifies"] else "best available (none qualify)")
        lines += [
            "",
            "## Selected main-text point",
            "",
            f"- **{best['variant']}**, α={best['alpha']:.2f} ({tag}): "
            f"hidden={best['hidden_probe_auroc_mean']:.4f}, Standard={best['standard_probe_auroc_mean']:.4f}, "
            f"V-reg={best['vreg_probe_auroc_mean']:.4f}, Δ AUROC={best['delta_probe_auroc_mean']:+.4f}, "
            f"Δ L20={best['delta_critical_l20_abs_mean']:+.4f}, MSE ratio={best['mse_ratio']:.3f}",
        ]

    lines += ["", f"Full ranked table: `{out_path.with_suffix('.json').name}`"]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    serializable = [{k: v for k, v in r.items() if k != "_status"} | {"status": r["_status"]} for r in rows]
    out_path.with_suffix(".json").write_text(json.dumps(serializable, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="config_v2.json")
    parser.add_argument("--out-root", default="results_ablation")
    parser.add_argument("--d-sae", type=int, nargs="+", default=[96, 192])
    parser.add_argument("--lambda-v", type=float, nargs="+", default=[0.2, 0.5, 1.0])
    parser.add_argument("--noise-std", type=float, nargs="+", default=[0.10, 0.15])
    args = parser.parse_args()

    base = json.loads(Path(args.base_config).read_text(encoding="utf-8"))
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    variant_summaries: List[Dict[str, Any]] = []
    grid = list(itertools.product(args.d_sae, args.lambda_v, args.noise_std))
    print(f"[ablation] {len(grid)} variants", flush=True)

    for d_sae, lambda_v, noise_std in grid:
        cfg = copy.deepcopy(base)
        cfg["d_sae"] = d_sae
        cfg["lambda_v"] = lambda_v
        cfg["noise_std"] = noise_std
        name = variant_name(d_sae, lambda_v, noise_std)
        out_dir = out_root / name
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config_used.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

        records = []
        for alpha in cfg["alphas"]:
            for seed in cfg["seeds"]:
                print(f"[ablation:{name}] alpha={alpha}, seed={seed}", flush=True)
                records.append(run_one(cfg, float(alpha), int(seed), out_dir))
        summary = aggregate(records)
        summary["config"] = cfg
        summary["n_runs"] = len(records)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        write_markdown(summary, out_dir / "summary.md")
        variant_summaries.append({"variant": name, "config": cfg, "summary": summary})
        print(f"[ablation:{name}] wrote {out_dir / 'summary.md'}", flush=True)

    build_ablation_summary(variant_summaries, out_root / "ablation_summary.md")
    print(f"Wrote {out_root / 'ablation_summary.md'}", flush=True)


if __name__ == "__main__":
    main()
