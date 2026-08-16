#!/usr/bin/env python3
"""Build held-out dosage numeric probe pairs (not used in V-loss training)."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

from dosage_probe_common import (
    DATASET_KIND,
    EXPERIMENT,
    LABEL_CRITICAL,
    LABEL_NUISANCE,
    STATUS_FROZEN,
    TASK,
    split_summary,
    stratified_shuffle,
    validate_pairs,
    write_json,
)

NUMBER_WORDS = {
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
}

DRUGS_FULL = [
    "cefazolin",
    "morphine",
    "metformin",
    "lisinopril",
    "heparin",
    "vancomycin",
    "acetaminophen",
    "furosemide",
    "insulin",
    "warfarin",
    "amoxicillin",
    "ondansetron",
]

DRUGS_SMOKE = ["cefazolin", "morphine"]

VALUE_PAIRS_FULL = [(3, 8), (2, 7), (4, 9), (5, 10), (1, 6)]
VALUE_PAIRS_SMOKE = [(3, 8), (2, 7)]

DOSE_COUNT_TEMPLATES: list[dict[str, Any]] = [
    {
        "template_id": "dose_count::received_today",
        "family": "dose_count",
        "pattern": "The patient received {n} doses of {drug} today.",
        "paraphrase": "Today, the patient received {n} doses of {drug}.",
    },
    {
        "template_id": "dose_count::chart_records",
        "family": "dose_count",
        "pattern": "The chart records {n} doses of {drug}.",
        "paraphrase": "According to the chart, the patient received {n} doses of {drug}.",
    },
    {
        "template_id": "dose_count::nursing_notes",
        "family": "dose_count",
        "pattern": "Nursing notes list {n} doses of {drug}.",
        "paraphrase": "The nursing notes indicate {n} doses of {drug}.",
    },
    {
        "template_id": "dose_count::administration_log",
        "family": "dose_count",
        "pattern": "The administration log shows {n} doses of {drug}.",
        "paraphrase": "Per the administration log, {n} doses of {drug} were given.",
    },
    {
        "template_id": "dose_count::medication_record",
        "family": "dose_count",
        "pattern": "The medication record documents {n} doses of {drug}.",
        "paraphrase": "Documentation lists {n} doses of {drug} on the medication record.",
    },
    {
        "template_id": "dose_count::discharge_summary",
        "family": "dose_count",
        "pattern": "The discharge summary states {n} doses of {drug}.",
        "paraphrase": "According to the discharge summary, the patient received {n} doses of {drug}.",
    },
    {
        "template_id": "dose_count::inpatient_orders",
        "family": "dose_count",
        "pattern": "Inpatient orders include {n} doses of {drug}.",
        "paraphrase": "The inpatient order set lists {n} doses of {drug}.",
    },
    {
        "template_id": "dose_count::emar_entry",
        "family": "dose_count",
        "pattern": "The eMAR entry confirms {n} doses of {drug}.",
        "paraphrase": "Electronic medication administration records show {n} doses of {drug}.",
    },
    {
        "template_id": "dose_count::pharmacy_label",
        "family": "dose_count",
        "pattern": "The pharmacy label indicates {n} doses of {drug}.",
        "paraphrase": "Pharmacy labeling documents {n} doses of {drug}.",
    },
    {
        "template_id": "dose_count::shift_handoff",
        "family": "dose_count",
        "pattern": "Shift handoff reports {n} doses of {drug}.",
        "paraphrase": "During handoff, nursing staff reported {n} doses of {drug}.",
    },
    {
        "template_id": "dose_count::progress_note",
        "family": "dose_count",
        "pattern": "The progress note mentions {n} doses of {drug}.",
        "paraphrase": "In today's progress note, the team recorded {n} doses of {drug}.",
    },
    {
        "template_id": "dose_count::order_set",
        "family": "dose_count",
        "pattern": "The order set specifies {n} doses of {drug}.",
        "paraphrase": "Orders in the chart specify {n} doses of {drug}.",
    },
]

DOSE_AMOUNT_TEMPLATES: list[dict[str, Any]] = [
    {
        "template_id": "dose_amount::given_mg",
        "family": "dose_amount",
        "pattern": "The patient was given {n} mg of {drug}.",
        "paraphrase": "The patient received {n} mg of {drug}.",
        "unit": "mg",
    },
    {
        "template_id": "dose_amount::administer_mg",
        "family": "dose_amount",
        "pattern": "Administer {n} mg of {drug} before discharge.",
        "paraphrase": "Before discharge, administer {n} mg of {drug}.",
        "unit": "mg",
    },
    {
        "template_id": "dose_amount::prescribed_mg",
        "family": "dose_amount",
        "pattern": "The prescribed dose was changed to {n} mg.",
        "paraphrase": "The prescription now lists {n} mg.",
        "unit": "mg",
        "drug_optional": True,
    },
    {
        "template_id": "dose_amount::infusion_mg",
        "family": "dose_amount",
        "pattern": "The infusion contained {n} mg of {drug}.",
        "paraphrase": "An infusion with {n} mg of {drug} was started.",
        "unit": "mg",
    },
    {
        "template_id": "dose_amount::tablet_mg",
        "family": "dose_amount",
        "pattern": "The patient took {n} mg tablets of {drug}.",
        "paraphrase": "Tablets of {drug} at {n} mg were taken.",
        "unit": "mg",
    },
    {
        "template_id": "dose_amount::bolus_mg",
        "family": "dose_amount",
        "pattern": "A bolus of {n} mg {drug} was given.",
        "paraphrase": "The team administered a {n} mg bolus of {drug}.",
        "unit": "mg",
    },
    {
        "template_id": "dose_amount::rate_ml",
        "family": "dose_amount",
        "pattern": "The infusion rate was set to {n} mL per hour.",
        "paraphrase": "Infusion was ordered at {n} mL per hour.",
        "unit": "mL/h",
        "drug_optional": True,
    },
    {
        "template_id": "dose_amount::volume_ml",
        "family": "dose_amount",
        "pattern": "The nurse drew up {n} mL of {drug}.",
        "paraphrase": "A volume of {n} mL of {drug} was prepared.",
        "unit": "mL",
    },
    {
        "template_id": "dose_amount::every_hours",
        "family": "dose_amount",
        "pattern": "The medication was given every {n} hours.",
        "paraphrase": "Dosing interval was set to every {n} hours.",
        "unit": "hours",
        "drug_optional": True,
    },
    {
        "template_id": "dose_amount::daily_mg",
        "family": "dose_amount",
        "pattern": "The daily dose totals {n} mg of {drug}.",
        "paraphrase": "Total daily dosing is {n} mg of {drug}.",
        "unit": "mg",
    },
    {
        "template_id": "dose_amount::loading_mg",
        "family": "dose_amount",
        "pattern": "The loading dose was {n} mg of {drug}.",
        "paraphrase": "A loading dose of {n} mg {drug} was administered.",
        "unit": "mg",
    },
    {
        "template_id": "dose_amount::maintenance_mg",
        "family": "dose_amount",
        "pattern": "The maintenance dose is {n} mg of {drug}.",
        "paraphrase": "Maintenance therapy continues at {n} mg of {drug}.",
        "unit": "mg",
    },
]


def fill_template(pattern: str, n: int, drug: str | None) -> str:
    text = pattern.format(n=n, drug=drug or "medication")
    if not text.endswith("."):
        text += "."
    return re.sub(r"\s+", " ", text).strip()


def format_word_variant(pattern: str, n: int, drug: str | None) -> str | None:
    word = NUMBER_WORDS.get(n)
    if word is None:
        return None
    text = pattern.replace("{n}", word, 1)
    if drug is not None:
        text = text.format(drug=drug)
    else:
        text = text.replace(" {drug}", "").replace(" of {drug}", "")
    if not text.endswith("."):
        text += "."
    return re.sub(r"\s+", " ", text).strip()


def assign_template_splits(templates: list[dict[str, Any]], seed: int, smoke: bool) -> dict[str, str]:
    by_family: dict[str, list[dict[str, Any]]] = {}
    for template in templates:
        by_family.setdefault(template["family"], []).append(template)

    if smoke:
        n_train_per_family, n_dev_per_family = 3, 1
    else:
        n_train_per_family, n_dev_per_family = 6, 3

    mapping: dict[str, str] = {}
    for family, family_templates in sorted(by_family.items()):
        ordered = stratified_shuffle(family_templates, seed + len(family))
        n_required = n_train_per_family + n_dev_per_family + 1
        if len(ordered) < n_required:
            raise ValueError(
                f"Not enough {family} templates for stratified train/dev/test split: "
                f"need at least {n_required}, got {len(ordered)}."
            )
        for index, template in enumerate(ordered):
            if index < n_train_per_family:
                split = "train"
            elif index < n_train_per_family + n_dev_per_family:
                split = "dev"
            else:
                split = "test"
            mapping[template["template_id"]] = split
    return mapping


def build_pairs_for_template(
    template: dict[str, Any],
    split: str,
    drugs: list[str],
    value_pairs: list[tuple[int, int]],
    pair_index_start: int,
) -> tuple[list[dict[str, Any]], int]:
    pairs: list[dict[str, Any]] = []
    pair_index = pair_index_start
    drug_values: list[str | None] = [None] if template.get("drug_optional", False) else drugs

    for drug in drug_values:
        for n_left, n_right in value_pairs:
            left = fill_template(template["pattern"], n_left, drug)
            right = fill_template(template["pattern"], n_right, drug)
            pairs.append(
                {
                    "pair_id": f"dosage::{template['template_id']}::{pair_index:05d}",
                    "template_id": template["template_id"],
                    "template_cluster_id": template["template_id"],
                    "split": split,
                    "family": template["family"],
                    "label": LABEL_CRITICAL,
                    "control_type": "numeric_change",
                    "text_left": left,
                    "text_right": right,
                    "numeric_left": n_left,
                    "numeric_right": n_right,
                    "drug": drug,
                }
            )
            pair_index += 1

            par_left = fill_template(template["pattern"], n_left, drug)
            par_right = fill_template(template["paraphrase"], n_left, drug)
            pairs.append(
                {
                    "pair_id": f"dosage::{template['template_id']}::{pair_index:05d}",
                    "template_id": template["template_id"],
                    "template_cluster_id": template["template_id"],
                    "split": split,
                    "family": template["family"],
                    "label": LABEL_NUISANCE,
                    "control_type": "paraphrase",
                    "text_left": par_left,
                    "text_right": par_right,
                    "numeric_left": n_left,
                    "numeric_right": n_left,
                    "drug": drug,
                }
            )
            pair_index += 1

            fmt_left = fill_template(template["pattern"], n_left, drug)
            fmt_right = format_word_variant(template["pattern"], n_left, drug)
            if fmt_right is not None:
                pairs.append(
                    {
                        "pair_id": f"dosage::{template['template_id']}::{pair_index:05d}",
                        "template_id": template["template_id"],
                        "template_cluster_id": template["template_id"],
                        "split": split,
                        "family": template["family"],
                        "label": LABEL_NUISANCE,
                        "control_type": "format_word",
                        "text_left": fmt_left,
                        "text_right": fmt_right,
                        "numeric_left": n_left,
                        "numeric_right": n_left,
                        "drug": drug,
                    }
                )
                pair_index += 1

    return pairs, pair_index


def build_dataset(seed: int, smoke: bool) -> dict[str, Any]:
    templates = DOSE_COUNT_TEMPLATES + DOSE_AMOUNT_TEMPLATES
    if smoke:
        templates = DOSE_COUNT_TEMPLATES[:5] + DOSE_AMOUNT_TEMPLATES[:5]
    split_map = assign_template_splits(templates, seed, smoke=smoke)
    drugs = DRUGS_SMOKE if smoke else DRUGS_FULL
    value_pairs = VALUE_PAIRS_SMOKE if smoke else VALUE_PAIRS_FULL

    all_pairs: list[dict[str, Any]] = []
    pair_index = 0
    for template in templates:
        split = split_map[template["template_id"]]
        new_pairs, pair_index = build_pairs_for_template(
            template, split, drugs, value_pairs, pair_index
        )
        all_pairs.extend(new_pairs)

    validate_pairs(all_pairs)
    summary = split_summary(all_pairs)
    return {
        "experiment": EXPERIMENT,
        "dataset_kind": DATASET_KIND,
        "task": TASK,
        "status": STATUS_FROZEN,
        "seed": seed,
        "smoke": smoke,
        "not_in_v_loss": True,
        "split_policy": {
            "split_before_pair_generation": True,
            "group_unit": "template_id",
            "test_templates_unseen_in_train_dev": True,
        },
        "label_rule": {
            "positive_label": LABEL_CRITICAL,
            "negative_label": LABEL_NUISANCE,
            "positive_control_type": "numeric_change",
            "negative_control_types": ["paraphrase", "format_word"],
        },
        "summary": {
            "n_templates": len(templates),
            "n_pairs": len(all_pairs),
            **summary,
        },
        "pairs": all_pairs,
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    lines = [
        "# Held-out dosage numeric probe",
        "",
        f"Status: `{payload['status']}`",
        f"Smoke: `{payload['smoke']}`",
        f"Pairs: {summary['n_pairs']}",
        f"Templates: {summary['n_templates']}",
        "",
        "## Split summary",
        "",
        "| Split | pairs | templates | critical | nuisance |",
        "|---|---:|---:|---:|---:|",
    ]
    for split in ("train", "dev", "test"):
        split_info = summary["splits"][split]
        labels = summary["labels"][split]
        lines.append(
            f"| {split} | {split_info['n_pairs']} | {split_info['n_templates']} | "
            f"{labels[LABEL_CRITICAL]} | {labels[LABEL_NUISANCE]} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Positive pairs change only the critical numeric value (e.g. 3 vs 8 doses).",
            "- Negative pairs are meaning-preserving paraphrase or digit/word formatting controls.",
            "- Templates in test are disjoint from train/dev.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build held-out dosage numeric probe dataset")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true", help="Small dataset for pipeline smoke tests")
    args = parser.parse_args()

    payload = build_dataset(seed=args.seed, smoke=args.smoke)
    write_json(args.output_json, payload)
    if args.output_md:
        write_markdown(args.output_md, payload)
    print(f"Wrote {len(payload['pairs'])} pairs -> {args.output_json}")


if __name__ == "__main__":
    main()
