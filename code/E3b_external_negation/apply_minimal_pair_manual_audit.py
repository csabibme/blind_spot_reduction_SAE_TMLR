#!/usr/bin/env python3
"""Apply manual audit decisions to the E3b OpenI minimal-pair split."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

BAD_AFFIRMED_PATTERNS = [
    r"\bis seen is present\b",
    r"\bare seen is present\b",
    r"\bidentified is present\b",
    r"\bnoted is present\b",
    r"\bappreciated is present\b",
    r"\bdemonstrated is present\b",
    r"\bvisualized is present\b",
    r"\b(seen|identified|noted|appreciated|demonstrated|visualized)\b.*\bis present\b",
    r"\bair is prominent consolidation\b",
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def validate_examples(examples: list[dict[str, Any]]) -> dict[str, Any]:
    report_splits: dict[str, set[str]] = defaultdict(set)
    pair_labels: dict[str, set[str]] = defaultdict(set)
    for example in examples:
        report_splits[example["report_id"]].add(example["split"])
        pair_labels[example["global_pair_id"]].add(example["label"])
    report_leaks = {rid: sorted(splits) for rid, splits in report_splits.items() if len(splits) != 1}
    bad_pairs = [pair_id for pair_id, labels in pair_labels.items() if labels != {"affirmed", "negated"}]
    if report_leaks:
        raise ValueError(f"Report split leakage detected: {list(report_leaks.items())[:5]}")
    if bad_pairs:
        raise ValueError(f"Pairs without both labels: {bad_pairs[:5]}")
    split_summary = {}
    for split_name in ("train", "dev", "test"):
        split_examples = [example for example in examples if example["split"] == split_name]
        split_summary[split_name] = {
            "n_examples": len(split_examples),
            "n_pairs": len({example["global_pair_id"] for example in split_examples}),
            "n_reports": len({example["report_id"] for example in split_examples}),
            "label_counts": dict(Counter(example["label"] for example in split_examples)),
            "concept_counts": dict(Counter(example["concept"] for example in split_examples)),
        }
    return {
        "leakage_checks_passed": True,
        "split_summary": split_summary,
    }


def remove_probe_dedup_collisions(examples: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for example in examples:
        by_pair[example["global_pair_id"]].append(example)
    seen: set[tuple[str, str, str, str]] = set()
    kept: list[dict[str, Any]] = []
    removed_pairs = []
    for pair_id in sorted(by_pair):
        pair_examples = by_pair[pair_id]
        keys = {
            (example["split"], example["template_id"], example["text"], example["label"])
            for example in pair_examples
        }
        if any(key in seen for key in keys):
            removed_pairs.append(pair_id)
            continue
        seen.update(keys)
        kept.extend(pair_examples)
    return kept, {
        "deduplication_unit": "split_template_text_label",
        "removed_pair_count": len(removed_pairs),
        "removed_pair_ids": removed_pairs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply E3b minimal-pair manual audit")
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--audit-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    split = read_json(args.split_json)
    audit = read_json(args.audit_json)
    decisions = {row["global_pair_id"]: row for row in audit["rows"]}
    all_pair_ids = {example["global_pair_id"] for example in split["examples"]}
    missing_decisions = sorted(all_pair_ids - set(decisions))
    if missing_decisions:
        raise ValueError(
            f"Pairs without manual audit decision: {missing_decisions[:10]} "
            f"(total={len(missing_decisions)})"
        )
    undecided = [row["global_pair_id"] for row in audit["rows"] if row["accepted"] is None]
    if undecided:
        raise ValueError(f"Manual audit has undecided rows: {undecided[:10]}")

    rejected = {pair_id for pair_id, row in decisions.items() if row["accepted"] is not True}
    accepted_audited = {
        pair_id
        for pair_id, row in decisions.items()
        if row["accepted"] is True
    }
    filtered_examples_pre_dedup = [
        example
        for example in split["examples"]
        if decisions[example["global_pair_id"]]["accepted"] is True
    ]
    bad_retained = []
    for example in filtered_examples_pre_dedup:
        if example["label"] != "affirmed":
            continue
        for pattern in BAD_AFFIRMED_PATTERNS:
            if re.search(pattern, example["text"], flags=re.I):
                bad_retained.append(
                    {
                        "global_pair_id": example["global_pair_id"],
                        "pattern": pattern,
                        "text": example["text"],
                    }
                )
    if bad_retained:
        raise ValueError(f"Bad affirmed text retained after audit: {bad_retained[:10]}")
    filtered_examples, dedup_filter = remove_probe_dedup_collisions(filtered_examples_pre_dedup)
    source_pair_count = len(all_pair_ids)
    final_pair_count = len({example["global_pair_id"] for example in filtered_examples})
    removed_after_acceptance = dedup_filter["removed_pair_count"]
    if source_pair_count != len(rejected) + removed_after_acceptance + final_pair_count:
        raise ValueError(
            "Audit accounting mismatch: "
            f"source={source_pair_count} rejected={len(rejected)} "
            f"removed_after_acceptance={removed_after_acceptance} final={final_pair_count}"
        )
    reason_counts = Counter(
        row["rejection_reason"] or "accepted"
        for row in decisions.values()
    )

    split["status"] = "openi_external_negation_split_frozen_after_manual_audit"
    split["manual_audit"] = {
        "audit_json": str(args.audit_json),
        "audit_sha256": sha256_file(args.audit_json),
        "source_pairs": source_pair_count,
        "accepted_pairs": len(accepted_audited),
        "rejected_pairs": len(rejected),
        "removed_after_acceptance": removed_after_acceptance,
        "final_pairs": final_pair_count,
        "removed_after_acceptance_reason": {
            "probe_dedup_collision": removed_after_acceptance,
        },
        "n_audited_pairs": len(decisions),
        "n_accepted_audited_pairs": len(accepted_audited),
        "n_rejected_pairs": len(rejected),
        "all_source_pairs_had_decisions": True,
        "all_retained_pairs_manually_accepted": True,
        "bad_affirmed_guard_patterns": BAD_AFFIRMED_PATTERNS,
        "rejection_reason_counts": dict(reason_counts),
        "post_audit_probe_dedup_filter": dedup_filter,
    }
    split["examples"] = filtered_examples
    split["summary"]["n_examples"] = len(filtered_examples)
    split["summary"]["n_pairs"] = len({example["global_pair_id"] for example in filtered_examples})
    split["summary"]["n_reports"] = len({example["report_id"] for example in filtered_examples})
    split["validation"] = validate_examples(filtered_examples)
    write_json(args.output_json, split)
    print(f"Saved JSON -> {args.output_json}")


if __name__ == "__main__":
    main()
