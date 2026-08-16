#!/usr/bin/env python3
"""Build split-safe meaning-preserving nuisance controls for E2.

The important invariant is split-before-generation: source templates are assigned to
calibration/test first, then nuisance variants are generated only inside that split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable


TERMINAL_PUNCT = (".", "?", "!")
NEGATION_RE = re.compile(r"\b(no|not|without|denies|denied|negative|never)\b", re.I)
NUMERIC_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
SENSITIVE_TOKEN_RE = re.compile(
    r"\b(?:mg|mcg|g|ml|mmol|iu|units?|dose|q\d+h|bid|tid|qid|prn|cyp\w*|brca\w*)\b",
    re.I,
)
HIGH_RISK_AUDIT_FAMILIES = [
    "condition_negation",
    "date_time_change",
    "drug_name_swap",
    "frequency_change",
    "number_swap",
    "unit_of_measure",
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def stable_id(*parts: object) -> str:
    payload = "\n".join(str(part) for part in parts)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def normalise_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def has_internal_space(text: str) -> bool:
    stripped = text.strip()
    return " " in stripped


def transform_double_first_space(text: str) -> str | None:
    stripped = text.strip()
    if not has_internal_space(stripped):
        return None
    return stripped.replace(" ", "  ", 1)


def transform_terminal_period(text: str) -> str | None:
    stripped = text.strip()
    if not stripped:
        return None
    if stripped.endswith("."):
        return stripped[:-1]
    if stripped.endswith(("?", "!")):
        return None
    return stripped + "."


def transform_sentence_initial_case(text: str) -> str | None:
    stripped = text.strip()
    if not stripped or not stripped[0].isalpha():
        return None
    first_word = stripped.split(" ", 1)[0]
    if not (first_word[0].isupper() and first_word[1:].islower()):
        return None
    first = stripped[0]
    replacement = first.lower() if first.isupper() else first.upper()
    changed = replacement + stripped[1:]
    if changed == stripped:
        return None
    return changed


def transform_prefix_in_note(text: str) -> str | None:
    stripped = text.strip()
    if not stripped:
        return None
    lowered = stripped[0].lower() + stripped[1:] if stripped[0].isalpha() else stripped
    return "In the note, " + lowered


def transform_prefix_report_reads(text: str) -> str | None:
    stripped = text.strip()
    if not stripped:
        return None
    return "The report reads: " + stripped


Transform = tuple[str, str, Callable[[str], str | None]]

TRANSFORMS: list[Transform] = [
    ("tier1_formatting", "double_first_space", transform_double_first_space),
    ("tier1_formatting", "terminal_period_toggle", transform_terminal_period),
    ("tier1_formatting", "sentence_initial_case_toggle", transform_sentence_initial_case),
    ("tier2_lexical_nuisance", "prefix_in_note", transform_prefix_in_note),
    ("tier2_lexical_nuisance", "prefix_report_reads", transform_prefix_report_reads),
]


def admissibility(text: str, tier: str, transform_type: str) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if transform_type == "sentence_initial_case_toggle":
        first_word = text.strip().split(" ", 1)[0] if text.strip() else ""
        if first_word and not (first_word[0].isupper() and first_word[1:].islower()):
            reasons.append("sentence_initial_word_not_title_case")
    if transform_type == "sentence_initial_case_toggle" and SENSITIVE_TOKEN_RE.search(text):
        reasons.append("sensitive_abbreviation_or_unit_present")
    if tier == "tier2_lexical_nuisance":
        if NEGATION_RE.search(text):
            reasons.append("tier2_skips_negation_scope_risk")
        if NUMERIC_RE.search(text) or SENSITIVE_TOKEN_RE.search(text):
            reasons.append("tier2_skips_numeric_or_unit_risk")
    return (len(reasons) == 0), reasons


def split_templates(
    families: dict[str, Any],
    calibration_fraction: float,
    seed: int,
) -> dict[str, dict[str, str]]:
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("--calibration-fraction must be between 0 and 1")
    rng = random.Random(seed)
    split_by_family: dict[str, dict[str, str]] = {}

    for family, block in sorted(families.items()):
        template_ids = sorted(block["template_cluster_sizes"])
        shuffled = template_ids[:]
        rng.shuffle(shuffled)
        if len(shuffled) == 1:
            n_cal = 1
        else:
            n_cal = round(len(shuffled) * calibration_fraction)
            n_cal = min(max(1, n_cal), len(shuffled) - 1)
        cal_ids = set(shuffled[:n_cal])
        split_by_family[family] = {
            template_id: ("calibration" if template_id in cal_ids else "test")
            for template_id in template_ids
        }

    return split_by_family


def generate_records(
    families: dict[str, Any],
    split_by_family: dict[str, dict[str, str]],
) -> tuple[list[dict[str, Any]], Counter[str]]:
    records: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()

    for family, block in sorted(families.items()):
        for pair in block["pairs"]:
            template_id = pair["template_id"]
            split = split_by_family[family][template_id]
            for side in ("orig", "pert"):
                source_text = normalise_ws(pair[side])
                for tier, transform_type, fn in TRANSFORMS:
                    admissible, reasons = admissibility(source_text, tier, transform_type)
                    nuisance_text = fn(source_text) if admissible else None
                    if nuisance_text is None or nuisance_text == source_text:
                        skipped[f"{tier}:{transform_type}"] += 1
                        continue

                    transform_id = stable_id(
                        family,
                        pair["pair_index"],
                        side,
                        tier,
                        transform_type,
                        source_text,
                        nuisance_text,
                    )
                    records.append(
                        {
                            "nuisance_id": f"{family}::{pair['pair_index']}::{side}::{transform_id}",
                            "source_pair_index": int(pair["pair_index"]),
                            "source_side": side,
                            "source_sentence_id": f"{family}::{pair['pair_index']}::{side}",
                            "family": family,
                            "template_id": template_id,
                            "template_signature": pair["template_signature"],
                            "orig_value": pair.get("orig_value", ""),
                            "pert_value": pair.get("pert_value", ""),
                            "transform_tier": tier,
                            "transform_type": transform_type,
                            "deterministic_transform_id": transform_id,
                            "source_text": source_text,
                            "nuisance_text": nuisance_text,
                            "admissible": True,
                            "admissibility_reasons": reasons,
                            "split": split,
                        }
                    )

    return records, skipped


def validate_no_template_leakage(records: list[dict[str, Any]]) -> None:
    splits_by_template: dict[str, set[str]] = defaultdict(set)
    for record in records:
        key = f"{record['family']}::{record['template_id']}"
        splits_by_template[key].add(record["split"])
    leaked = {key: sorted(splits) for key, splits in splits_by_template.items() if len(splits) > 1}
    if leaked:
        raise ValueError(f"Template leakage across splits: {leaked}")


def build_summary(records: list[dict[str, Any]], skipped: Counter[str]) -> dict[str, Any]:
    by_split = Counter(record["split"] for record in records)
    by_tier = Counter(record["transform_tier"] for record in records)
    by_transform = Counter(
        f"{record['transform_tier']}:{record['transform_type']}" for record in records
    )
    by_family = Counter(record["family"] for record in records)
    return {
        "n_records": len(records),
        "by_split": dict(sorted(by_split.items())),
        "by_tier": dict(sorted(by_tier.items())),
        "by_transform": dict(sorted(by_transform.items())),
        "by_family": dict(sorted(by_family.items())),
        "skipped_transform_counts": dict(sorted(skipped.items())),
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    lines = [
        "# E2 Meaning-Preserving Nuisance Controls",
        "",
        f"Template clusters: `{payload['template_clusters_json']}`",
        f"Pairs SHA256: `{payload.get('pairs_sha256')}`",
        "",
        "## Summary",
        "",
        f"- Total nuisance records: {summary['n_records']}",
        f"- Split counts: {summary['by_split']}",
        f"- Tier counts: {summary['by_tier']}",
        "",
        "## Transform Counts",
        "",
        "| Transform | records | skipped |",
        "|---|---:|---:|",
    ]
    transforms = sorted(
        set(summary["by_transform"]) | set(summary["skipped_transform_counts"])
    )
    for transform in transforms:
        lines.append(
            f"| {transform} | {summary['by_transform'].get(transform, 0)} | "
            f"{summary['skipped_transform_counts'].get(transform, 0)} |"
        )
    lines.extend(
        [
            "",
            "## Audit Examples",
            "",
            "| Split | Tier | Transform | Family | Source | Nuisance |",
            "|---|---|---|---|---|---|",
        ]
    )
    audit_records = stratified_audit_records(payload["records"], per_bucket=len(HIGH_RISK_AUDIT_FAMILIES))
    for record in audit_records:
        source = record["source_text"].replace("|", "\\|")
        nuisance = record["nuisance_text"].replace("|", "\\|")
        lines.append(
            f"| {record['split']} | {record['transform_tier']} | "
            f"{record['transform_type']} | {record['family']} | {source} | {nuisance} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def stratified_audit_records(records: list[dict[str, Any]], per_bucket: int = 2) -> list[dict[str, Any]]:
    """Return deterministic audit examples across transform, split, side, and family.

    The Markdown audit table is meant for manual review, not statistical sampling. We keep
    examples spread across nuisance types so risky transforms are visible immediately.
    """
    buckets: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = (
            record["transform_tier"],
            record["transform_type"],
            record["split"],
            record["source_side"],
        )
        buckets[key].append(record)

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    for key in sorted(buckets):
        bucket = sorted(
            buckets[key],
            key=lambda record: (
                family_priority(record["family"]),
                record["family"],
                int(record["source_pair_index"]),
                record["nuisance_id"],
            ),
        )
        add_from_bucket(selected, selected_ids, bucket, key, per_bucket)
    return selected


def family_priority(family: str) -> tuple[int, int]:
    if family in HIGH_RISK_AUDIT_FAMILIES:
        return (0, HIGH_RISK_AUDIT_FAMILIES.index(family))
    return (1, 0)


def add_from_bucket(
    selected: list[dict[str, Any]],
    selected_ids: set[str],
    bucket: list[dict[str, Any]],
    key: tuple[str, str, str, str],
    per_bucket: int,
) -> None:
    seen_families = {record["family"] for record in selected if bucket_key(record) == key}
    for record in bucket:
        if bucket_count(selected, key) >= per_bucket:
            break
        if record["nuisance_id"] in selected_ids:
            continue
        if record["family"] in seen_families and len(seen_families) < len({r["family"] for r in bucket}):
            continue
        selected.append(record)
        selected_ids.add(record["nuisance_id"])
        seen_families.add(record["family"])
    for record in bucket:
        if bucket_count(selected, key) >= min(per_bucket, len(bucket)):
            break
        if record["nuisance_id"] in selected_ids:
            continue
        selected.append(record)
        selected_ids.add(record["nuisance_id"])


def bucket_count(records: list[dict[str, Any]], key: tuple[str, str, str, str]) -> int:
    return sum(1 for record in records if bucket_key(record) == key)


def bucket_key(record: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        record["transform_tier"],
        record["transform_type"],
        record["split"],
        record["source_side"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build split-safe meaning-preserving nuisance controls"
    )
    parser.add_argument("--template-clusters-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--calibration-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    template_payload = read_json(args.template_clusters_json)
    families = template_payload["families"]
    split_by_family = split_templates(families, args.calibration_fraction, args.seed)
    records, skipped = generate_records(families, split_by_family)
    validate_no_template_leakage(records)

    payload = {
        "experiment": "E2_null_calibration",
        "control_definition": "meaning_preserving_nuisance_controls",
        "null_hypothesis_note": (
            "Nuisance controls preserve the targeted clinical relation; they are not "
            "assumed to have zero hidden/code distance."
        ),
        "template_clusters_json": str(args.template_clusters_json),
        "pairs_sha256": template_payload.get("pairs_sha256"),
        "split_rule": {
            "unit": "family_template_id",
            "calibration_fraction": args.calibration_fraction,
            "seed": args.seed,
            "generation_order": "split_before_nuisance_generation",
        },
        "transform_tiers": {
            "tier1_formatting": "very safe formatting controls",
            "tier2_lexical_nuisance": "riskier lexical/stylistic controls; report separately",
        },
        "summary": build_summary(records, skipped),
        "split_by_family_template": split_by_family,
        "records": records,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_markdown(args.output_md, payload)
    print(f"Saved JSON -> {args.output_json}")
    print(f"Saved Markdown -> {args.output_md}")


if __name__ == "__main__":
    main()
