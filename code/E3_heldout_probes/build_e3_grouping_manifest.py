#!/usr/bin/env python3
"""Build E3 general grouping manifest and task feasibility audit from E5 templates."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


TASK_TAGS_BY_FAMILY = {
    "negation": ["negation_state"],
    "condition_negation": ["negation_state"],
    "number_swap": ["numeric_candidate"],
    "frequency_change": ["frequency_candidate"],
    "unit_of_measure": ["unit_candidate"],
    "date_time_change": ["temporal_candidate"],
    "anatomical_direction": ["anatomical_direction_candidate"],
    "laterality": ["laterality_candidate"],
    "severity_change": ["severity_candidate"],
}

NUMERIC_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
NEGATION_RE = re.compile(r"\b(no|not|without|denies|denied|negative|never)\b", re.I)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def norm_value(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def stable_id(*parts: object) -> str:
    return hashlib.sha1("\n".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:12]


def unordered_value_pair_id(family: str, orig_value: str, pert_value: str) -> str:
    values = sorted([norm_value(orig_value), norm_value(pert_value)])
    return f"{family}::uvp_{stable_id(family, values[0], values[1])}"


def infer_candidate_task_tags(family: str, orig: str, pert: str, orig_value: str, pert_value: str) -> list[str]:
    tags = list(TASK_TAGS_BY_FAMILY.get(family, []))
    joined_values = f"{orig_value} {pert_value}"
    if NUMERIC_RE.search(joined_values):
        tags.append("numeric_value_present")
    if NEGATION_RE.search(orig) or NEGATION_RE.search(pert):
        tags.append("negation_token_present")
    return sorted(set(tags))


def build_records(template_payload: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for family, block in sorted(template_payload["families"].items()):
        for pair in block["pairs"]:
            pair_index = int(pair["pair_index"])
            uvp_id = unordered_value_pair_id(
                family,
                pair.get("orig_value", ""),
                pair.get("pert_value", ""),
            )
            records.append(
                {
                    "family": family,
                    "pair_index": pair_index,
                    "global_pair_id": f"{family}::{pair_index}",
                    "template_id": pair["template_id"],
                    "template_signature": pair["template_signature"],
                    "orig_value": pair.get("orig_value", ""),
                    "pert_value": pair.get("pert_value", ""),
                    "unordered_value_pair_id": uvp_id,
                    "source_sentence_ids": {
                        "orig": f"{family}::{pair_index}::orig",
                        "pert": f"{family}::{pair_index}::pert",
                    },
                    "candidate_task_tags": infer_candidate_task_tags(
                        family,
                        pair["orig"],
                        pair["pert"],
                        pair.get("orig_value", ""),
                        pair.get("pert_value", ""),
                    ),
                    "texts": {
                        "orig": pair["orig"],
                        "pert": pair["pert"],
                    },
                    "leakage_groups": {
                        "template": pair["template_id"],
                        "unordered_value_pair": uvp_id,
                        "source_pair": f"{family}::{pair_index}",
                    },
                }
            )
    return records


def feasibility(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_tag: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        for tag in record["candidate_task_tags"]:
            by_tag[tag].append(record)

    out: dict[str, Any] = {}
    for tag, tagged_records in sorted(by_tag.items()):
        families = Counter(record["family"] for record in tagged_records)
        templates = {record["leakage_groups"]["template"] for record in tagged_records}
        uvp_counts = Counter(record["unordered_value_pair_id"] for record in tagged_records)
        uvp_sizes = sorted(uvp_counts.values())
        out[tag] = {
            "n_pairs": len(tagged_records),
            "n_families": len(families),
            "families": dict(sorted(families.items())),
            "n_template_groups": len(templates),
            "n_unordered_value_pair_groups": len(uvp_counts),
            "largest_unordered_value_pair_group_size": max(uvp_sizes) if uvp_sizes else 0,
            "median_unordered_value_pair_group_size": median_int(uvp_sizes),
            "unordered_value_pair_group_size_distribution": dict(
                sorted(Counter(uvp_sizes).items())
            ),
            "task_split_status": task_split_status(tag),
            "notes": task_note(tag),
        }
    return out


def median_int(values: list[int]) -> float:
    if not values:
        return 0.0
    mid = len(values) // 2
    if len(values) % 2:
        return float(values[mid])
    return float((values[mid - 1] + values[mid]) / 2)


def task_note(tag: str) -> str:
    if tag == "negation_state":
        return "Do not use original-vs-perturbed as the label; define affirmed/negated labels explicitly."
    if tag in {"numeric_candidate", "numeric_value_present"}:
        return (
            "Do not merge number_swap, frequency_change, unit_of_measure, and date_time_change "
            "until the exact prediction label is frozen."
        )
    if tag == "frequency_candidate":
        return "Potential separate frequency task; do not merge into generic dosage without review."
    if tag == "unit_candidate":
        return "Potential unit task; label may be unit class rather than numeric comparison."
    return "Candidate task tag; requires manual label definition before split freeze."


def task_split_status(tag: str) -> str:
    if tag == "negation_token_present":
        return "diagnostic_only_not_a_task"
    if tag.endswith("_candidate") or tag in {"numeric_value_present"}:
        return "candidate_only_label_definition_required"
    return "candidate_ready_for_manual_label_review"


def validate_records(records: list[dict[str, Any]]) -> None:
    seen_global = set()
    for record in records:
        global_id = record["global_pair_id"]
        if global_id in seen_global:
            raise ValueError(f"Duplicate global pair ID: {global_id}")
        seen_global.add(global_id)
        if not record["template_id"]:
            raise ValueError(f"Missing template_id for {global_id}")
        if not record["unordered_value_pair_id"]:
            raise ValueError(f"Missing unordered value-pair ID for {global_id}")


def write_manifest_md(path: Path, records: list[dict[str, Any]], feasibility_block: dict[str, Any]) -> None:
    family_counts = Counter(record["family"] for record in records)
    lines = [
        "# E3 Grouping Manifest",
        "",
        f"Total pairs: {len(records)}",
        "",
        "## Family Counts",
        "",
        "| Family | pairs |",
        "|---|---:|",
    ]
    for family, count in sorted(family_counts.items()):
        lines.append(f"| {family} | {count} |")
    lines.extend(
        [
            "",
            "## Task Feasibility Summary",
            "",
            "| Candidate tag | pairs | families | template groups | value-pair groups | largest value group | median value group | status |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for tag, block in sorted(feasibility_block.items()):
        lines.append(
            f"| {tag} | {block['n_pairs']} | {block['n_families']} | "
            f"{block['n_template_groups']} | {block['n_unordered_value_pair_groups']} | "
            f"{block['largest_unordered_value_pair_group_size']} | "
            f"{block['median_unordered_value_pair_group_size']:.1f} | "
            f"{block['task_split_status']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_feasibility_md(path: Path, feasibility_block: dict[str, Any]) -> None:
    lines = [
        "# E3 Task Feasibility Audit",
        "",
        "No task-specific split is frozen here. This file only reports candidate feasibility.",
        "",
    ]
    for tag, block in sorted(feasibility_block.items()):
        lines.extend(
            [
                f"## {tag}",
                "",
                f"- Pairs: {block['n_pairs']}",
                f"- Families: {block['n_families']}",
                f"- Template groups: {block['n_template_groups']}",
                f"- Unordered value-pair groups: {block['n_unordered_value_pair_groups']}",
                f"- Largest unordered value-pair group: {block['largest_unordered_value_pair_group_size']}",
                f"- Median unordered value-pair group: {block['median_unordered_value_pair_group_size']:.1f}",
                f"- Value-pair group-size distribution: `{block['unordered_value_pair_group_size_distribution']}`",
                f"- Status: `{block['task_split_status']}`",
                f"- Note: {block['notes']}",
                "",
                "| Family | pairs |",
                "|---|---:|",
            ]
        )
        for family, count in sorted(block["families"].items()):
            lines.append(f"| {family} | {count} |")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build E3 grouping manifest and task feasibility audit"
    )
    parser.add_argument("--template-clusters-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--feasibility-json", type=Path, required=True)
    parser.add_argument("--feasibility-md", type=Path, required=True)
    args = parser.parse_args()

    template_payload = read_json(args.template_clusters_json)
    records = build_records(template_payload)
    validate_records(records)
    feasibility_block = feasibility(records)
    manifest = {
        "experiment": "E3_heldout_probes",
        "status": "general_grouping_manifest_only",
        "template_clusters_json": str(args.template_clusters_json),
        "pairs_sha256": template_payload.get("pairs_sha256"),
        "split_policy": {
            "level_1": "general grouping manifest",
            "level_2": "task-specific split deferred until label definitions are frozen",
            "primary_leakage_units": ["template_id", "unordered_value_pair_id", "source_sentence_id"],
        },
        "records": records,
    }
    feasibility_payload = {
        "experiment": "E3_heldout_probes",
        "status": "task_feasibility_only_no_final_splits",
        "template_clusters_json": str(args.template_clusters_json),
        "pairs_sha256": template_payload.get("pairs_sha256"),
        "task_feasibility": feasibility_block,
    }

    for path in (args.output_json, args.output_md, args.feasibility_json, args.feasibility_md):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    args.feasibility_json.write_text(
        json.dumps(feasibility_payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_manifest_md(args.output_md, records, feasibility_block)
    write_feasibility_md(args.feasibility_md, feasibility_block)
    print(f"Saved manifest JSON -> {args.output_json}")
    print(f"Saved manifest Markdown -> {args.output_md}")
    print(f"Saved feasibility JSON -> {args.feasibility_json}")
    print(f"Saved feasibility Markdown -> {args.feasibility_md}")


if __name__ == "__main__":
    main()
