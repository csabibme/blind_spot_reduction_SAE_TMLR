#!/usr/bin/env python3
"""Apply the fixed, rule-based selection to finished toy summaries.

Reads seed-averaged summaries (ablation variants and/or the v2/v3 main runs), applies the
rule defined in run_ablation_grid.py (a single fixed rule applied uniformly to every
point), and writes a ranked table plus the recommended main-text point. No experiments
are re-run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from run_ablation_grid import IDEAL_RULE, MAIN_TEXT_BAND, SELECTION_RULE, collect_rows, pick_main_text


def variant_label(cfg: Dict[str, Any], fallback: str) -> str:
    try:
        lam = f"{cfg['lambda_v']:g}".replace(".", "p")
        noise = f"{cfg['noise_std']:g}".replace(".", "p")
        return f"d_sae{cfg['d_sae']}_lam{lam}_noise{noise}"
    except KeyError:
        return fallback


def load_variant_summaries(ablation_root: Path, extra_summaries: List[Path]) -> List[Dict[str, Any]]:
    variant_summaries: List[Dict[str, Any]] = []

    if ablation_root.is_dir():
        for summary_path in sorted(ablation_root.glob("*/summary.json")):
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            cfg = summary.get("config", {})
            variant_summaries.append(
                {
                    "variant": summary_path.parent.name,
                    "config": cfg,
                    "summary": summary,
                }
            )

    for path in extra_summaries:
        if not path.is_file():
            print(f"[select] skip missing summary: {path}")
            continue
        summary = json.loads(path.read_text(encoding="utf-8"))
        cfg = summary.get("config", {})
        name = f"{path.parent.name}:{variant_label(cfg, path.parent.name)}"
        variant_summaries.append({"variant": name, "config": cfg, "summary": summary})

    return variant_summaries


def write_selection(rows: List[Dict[str, Any]], out_md: Path) -> Dict[str, Any]:
    qualifying = [r for r in rows if r["qualifies"]]
    best = pick_main_text(rows)

    lines = [
        "# Toy — fixed, rule-based selection",
        "",
        "Fixed rule (applied uniformly to every point): "
        f"hidden AUROC ≥ {SELECTION_RULE['hidden_auroc_min']}, "
        f"Standard AUROC < {SELECTION_RULE['standard_auroc_max']}, "
        f"Δ AUROC > {SELECTION_RULE['delta_auroc_min']}, "
        f"Δ L20(‖Δz‖) > {SELECTION_RULE['delta_l20_abs_min']}, "
        f"MSE ratio ≤ {SELECTION_RULE['mse_ratio_max']}.",
        f"Main-text point: among qualifying points, the one in the design target band "
        f"(hidden {MAIN_TEXT_BAND['hidden_min']}–{MAIN_TEXT_BAND['hidden_max']}, "
        f"Standard {MAIN_TEXT_BAND['standard_min']}–{MAIN_TEXT_BAND['standard_max']}); "
        f"smaller MSE ratio breaks ties. "
        f"Ideal bar: hidden ≥ {IDEAL_RULE['hidden_auroc_min']} and Δ AUROC ≥ {IDEAL_RULE['delta_auroc_min']}.",
        "",
        f"Qualifying points: **{len(qualifying)} / {len(rows)}**.",
        "",
        "| rank | source | α | hidden | Standard | V-reg | Δ AUROC | Δ L20 | MSE ratio | qualifies | ideal | fails |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|:--:|:--:|---|",
    ]
    for i, r in enumerate(rows[:30], start=1):
        st = r["_status"]
        lines.append(
            f"| {i} | {r['variant']} | {r['alpha']:.2f} | {r['hidden_probe_auroc_mean']:.4f} | "
            f"{r['standard_probe_auroc_mean']:.4f} | {r['vreg_probe_auroc_mean']:.4f} | "
            f"{r['delta_probe_auroc_mean']:+.4f} | {r['delta_critical_l20_abs_mean']:+.4f} | "
            f"{r['mse_ratio']:.3f} | {'yes' if r['qualifies'] else 'no'} | {'yes' if r['ideal'] else 'no'} | "
            f"{','.join(st['fail_reasons']) if st['fail_reasons'] else '-'} |"
        )

    recommendation: Dict[str, Any] = {}
    if best:
        tag = "IDEAL" if best["ideal"] else ("qualifying" if best["qualifies"] else "best available (none qualify)")
        framing = "modest but consistent" if (best["qualifies"] and not best["ideal"]) else (
            "ideal main-text point" if best["ideal"] else "no qualifying point — do not over-claim"
        )
        in_band = (
            MAIN_TEXT_BAND["hidden_min"] <= best["hidden_probe_auroc_mean"] <= MAIN_TEXT_BAND["hidden_max"]
            and MAIN_TEXT_BAND["standard_min"] <= best["standard_probe_auroc_mean"] <= MAIN_TEXT_BAND["standard_max"]
        )
        band_note = (
            "in the design target band"
            if in_band
            else "top qualifying point (no qualifying point falls in the design target band)"
        )
        lines += [
            "",
            "## Recommended main-text point",
            "",
            "The ranked table above is ordered by the screening tie-break; the main-text point below "
            "is the qualifying point selected by the design target band (this can differ from table rank 1).",
            "",
            f"- **{best['variant']}**, α={best['alpha']:.2f} ({tag}, {band_note})",
            f"  - hidden AUROC = {best['hidden_probe_auroc_mean']:.4f}",
            f"  - Standard AUROC = {best['standard_probe_auroc_mean']:.4f}",
            f"  - V-reg AUROC = {best['vreg_probe_auroc_mean']:.4f}",
            f"  - Δ AUROC = {best['delta_probe_auroc_mean']:+.4f}",
            f"  - Δ L20(‖Δz‖) = {best['delta_critical_l20_abs_mean']:+.4f}",
            f"  - MSE ratio = {best['mse_ratio']:.3f}",
            f"  - suggested framing: _{framing}_",
        ]
        recommendation = {
            "variant": best["variant"],
            "alpha": best["alpha"],
            "tag": tag,
            "framing": framing,
            "hidden_auroc": best["hidden_probe_auroc_mean"],
            "standard_auroc": best["standard_probe_auroc_mean"],
            "vreg_auroc": best["vreg_probe_auroc_mean"],
            "delta_auroc": best["delta_probe_auroc_mean"],
            "delta_l20_abs": best["delta_critical_l20_abs_mean"],
            "mse_ratio": best["mse_ratio"],
        }

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    serializable = [{k: v for k, v in r.items() if k != "_status"} | {"status": r["_status"]} for r in rows]
    out_json = {
        "selection_rule": SELECTION_RULE,
        "ideal_rule": IDEAL_RULE,
        "n_qualifying": len(qualifying),
        "n_total": len(rows),
        "recommended": recommendation,
        "rows": serializable,
    }
    out_md.with_suffix(".json").write_text(json.dumps(out_json, indent=2), encoding="utf-8")
    return recommendation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation-root", default="results_ablation")
    parser.add_argument("--extra-summary", nargs="*", default=["results_v2/summary.json"])
    parser.add_argument("--out", default="results_ablation/selection.md")
    args = parser.parse_args()

    variant_summaries = load_variant_summaries(Path(args.ablation_root), [Path(p) for p in args.extra_summary])
    if not variant_summaries:
        raise SystemExit("No summaries found. Run the ablation and/or v2 first.")

    rows = collect_rows(variant_summaries)
    out_md = Path(args.out)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    rec = write_selection(rows, out_md)
    print(f"Wrote {out_md}")
    if rec:
        print(f"Recommended: {rec['variant']} alpha={rec['alpha']:.2f} ({rec['tag']}, {rec['framing']})")


if __name__ == "__main__":
    main()
