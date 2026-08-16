#!/usr/bin/env python3
"""Write a compact reviewer-facing summary of the weak-set audits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def fmt(value: float, digits: int = 6) -> str:
    return f"{value:+.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpt2", type=Path, required=True)
    parser.add_argument("--qwen", type=Path, required=True)
    parser.add_argument("--old-gpt2", type=Path, required=True)
    parser.add_argument("--old-qwen", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    new = {
        "GPT-2": json.loads(args.gpt2.read_text()),
        "Qwen 2.5": json.loads(args.qwen.read_text()),
    }
    old = {
        "GPT-2": json.loads(args.old_gpt2.read_text()),
        "Qwen 2.5": json.loads(args.old_qwen.read_text()),
    }
    lines = [
        "# Standard-defined weak-set audit results",
        "",
        "Protocols frozen 2026-08-06. OpenI conditional panels are diagnostic and "
        "noncausal; the Qwen probe is a separate cross-fitted extrinsic endpoint.",
        "",
        "**Reproduction status:** Both GPT-2 and Qwen exactly reproduce their "
        "released own-tail artifacts. The Qwen execution uses its canonical "
        "`max_length=128`; the initial protocol text's value of 256 was corrected "
        "from the preserved historical command before interpreting the paired audit.",
        "",
        "## OpenI own-tail reproduction and conditional panels",
        "",
        "| Model | Family | Released own-tail delta | Recomputed delta (difference) | n(Wstd) | "
        "D Std→V-reg; paired delta [95% CI] | abs delta [95% CI] | reverse D delta | overlap |",
        "|---|---|---:|---:|---:|---|---|---:|---:|",
    ]
    for model, payload in new.items():
        for family, row in payload["families"].items():
            released = old[model]["families"][family]["delta_L20_s"]
            current = row["delta_L20_s"]
            panel = row["W_std"]["paired_panel"]
            rel = panel["relative_delta"]
            absolute = panel["absolute_delta"]
            reverse = row["reverse_W_vreg_diagnostic"]["paired_panel"]["relative_delta"]
            lines.append(
                f"| {model} | {family} | {released:+.6f} | {current:+.6f} "
                f"({current-released:+.6f}) | {panel['n']} | "
                f"{panel['relative_std']['mean']:.6f}→{panel['relative_vreg']['mean']:.6f}; "
                f"{rel['mean']:+.6f} [{rel['ci'][0]:+.6f}, {rel['ci'][1]:+.6f}] | "
                f"{absolute['mean']:+.4f} [{absolute['ci'][0]:+.4f}, {absolute['ci'][1]:+.4f}] | "
                f"{reverse['mean']:+.6f} | "
                f"{row['weak_set_overlap']['fraction_of_each_weak_set']:.3f} |"
            )

    lines.extend(
        [
            "",
            "## Standard-defined OpenI quintile profiles",
            "",
            "| Model | Family | Quintile | n | relative delta [95% CI] | absolute delta [95% CI] |",
            "|---|---|---|---:|---|---|",
        ]
    )
    for model, payload in new.items():
        for family, row in payload["families"].items():
            for quintile, panel in row["standard_defined_quintile_profile"].items():
                rel = panel["relative_delta"]
                absolute = panel["absolute_delta"]
                lines.append(
                    f"| {model} | {family} | {quintile} | {panel['n']} | "
                    f"{rel['mean']:+.6f} [{rel['ci'][0]:+.6f}, {rel['ci'][1]:+.6f}] | "
                    f"{absolute['mean']:+.4f} [{absolute['ci'][0]:+.4f}, "
                    f"{absolute['ci'][1]:+.4f}] |"
                )

    probe = json.loads(args.probe.read_text())
    pooled = probe["subsets"]["pooled_Wstd"]
    nonweak = probe["subsets"]["pooled_nonweak"]
    assoc = probe["associations"]["pooled_Wstd"]
    p_ci = pooled["bootstrap"]["intervals"]["delta_probability_margin"]
    nonweak_ci = nonweak["bootstrap"]["intervals"]["delta_probability_margin"]
    a_ci = assoc["bootstrap"]["intervals"]["weak_minus_nonweak_probability_margin"]
    selection_ci = probe["selection_aware_sensitivity"]["intervals"]["delta_probability_margin"]
    cells = pooled["mcnemar"]
    lines.extend(
        [
            "",
            "## Qwen 78-pair nested cross-fitted probe",
            "",
            f"- Standard weak vs non-weak margin: {assoc['weak']['probability_margin']:.6f} vs "
            f"{assoc['nonweak']['probability_margin']:.6f}; weak−nonweak "
            f"{assoc['weak_minus_nonweak']['probability_margin']:+.6f} "
            f"[{a_ci['ci_low']:+.6f}, {a_ci['ci_high']:+.6f}].",
            f"- Fixed Wstd margin: {pooled['standard']['probability_margin']:.6f}→"
            f"{pooled['vreg']['probability_margin']:.6f}; delta "
            f"{pooled['paired_delta']['probability_margin']:+.6f} "
            f"[{p_ci['ci_low']:+.6f}, {p_ci['ci_high']:+.6f}].",
            f"- Remaining 80% margin: {nonweak['standard']['probability_margin']:.6f}→"
            f"{nonweak['vreg']['probability_margin']:.6f}; delta "
            f"{nonweak['paired_delta']['probability_margin']:+.6f} "
            f"[{nonweak_ci['ci_low']:+.6f}, {nonweak_ci['ci_high']:+.6f}].",
            f"- Selection-aware margin CI: [{selection_ci['ci_low']:+.6f}, "
            f"{selection_ci['ci_high']:+.6f}].",
            f"- Correct pairs on Wstd: {pooled['standard']['n_correct']}/{pooled['n']}→"
            f"{pooled['vreg']['n_correct']}/{pooled['n']}; McNemar cells "
            f"(both correct, Standard only, V-reg only, both wrong)="
            f"({cells['both_correct']}, {cells['standard_only_correct']}, "
            f"{cells['vreg_only_correct']}, {cells['both_wrong']}), "
            f"exact p={cells['exact_two_sided_p']:.4f}.",
            "",
            "## Calibrated interpretation",
            "",
            "- The confirmatory OpenI Standard-defined relative response delta is positive "
            "with an excluding-zero interval in all eight model-by-family cells.",
            "- The absolute OpenI endpoint is positive with an excluding-zero interval in all four "
            "GPT-2 cells, negative with an excluding-zero interval for Qwen negation and "
            "laterality, and uncertain for Qwen severity and anatomical direction.",
            "- In the cross-fitted Qwen probe, low Standard D identifies significantly lower "
            "probability margin, but the fixed weak-set Standard→V-reg margin improvement is "
            "uncertain and only one pair changes from incorrect to correct.",
            "- Therefore the new evidence supports relative conditional lifting and an association "
            "between weak response and lower semantic margin; it does not establish a general "
            "absolute repair or a statistically resolved extrinsic improvement on the same weak pairs.",
        ]
    )
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
