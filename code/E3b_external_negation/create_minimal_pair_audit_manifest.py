#!/usr/bin/env python3
"""Create a manual-audit manifest for E3b OpenI minimal pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

REJECTION_REASONS = [
    "scope_ambiguity",
    "uncertainty",
    "grammatical_failure",
    "semantic_drift",
    "temporal_or_comparison_construct",
    "history_or_context",
    "duplicate",
    "other",
]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pair_examples(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_pair: dict[str, dict[str, Any]] = defaultdict(dict)
    meta: dict[str, dict[str, Any]] = {}
    for example in examples:
        by_pair[example["global_pair_id"]][example["label"]] = example
        meta[example["global_pair_id"]] = {
            "global_pair_id": example["global_pair_id"],
            "split": example["split"],
            "concept": example["concept"],
            "family": example["family"],
            "report_id": example["report_id"],
            "template_id": example["template_id"],
        }
    rows = []
    for pair_id in sorted(by_pair):
        pair = by_pair[pair_id]
        if set(pair) != {"affirmed", "negated"}:
            raise ValueError(f"Pair {pair_id} does not contain both labels")
        rows.append(
            {
                **meta[pair_id],
                "affirmed_text": pair["affirmed"]["text"],
                "negated_text": pair["negated"]["text"],
                "accepted": None,
                "rejection_reason": None,
                "auditor_note": None,
            }
        )
    return rows


def select_audit_rows(rows: list[dict[str, Any]], train_sample: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    dev_test = [row for row in rows if row["split"] in {"dev", "test"}]
    train = [row for row in rows if row["split"] == "train"]
    rng.shuffle(train)
    selected = dev_test + train[:train_sample]
    return sorted(selected, key=lambda row: (row["split"], row["concept"], row["global_pair_id"]))


def write_md(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# E3b Minimal Pair Manual Audit Manifest",
        "",
        "Fill `accepted`, `rejection_reason`, and `auditor_note` in the JSON.",
        "",
        "Allowed rejection reasons:",
        "",
    ]
    for reason in REJECTION_REASONS:
        lines.append(f"- `{reason}`")
    lines.extend(["", "## Audit Items", ""])
    for row in rows:
        lines.extend(
            [
                f"### {row['global_pair_id']}",
                "",
                f"- split: `{row['split']}`",
                f"- concept: `{row['concept']}`",
                f"- report_id: `{row['report_id']}`",
                f"- affirmed: {row['affirmed_text']}",
                f"- negated: {row['negated_text']}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create E3b minimal-pair manual audit manifest")
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--train-sample", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    payload = read_json(args.split_json)
    if payload.get("dataset_kind") != "minimal_pair":
        raise ValueError("Manual audit manifest is intended for minimal_pair split")
    rows = select_audit_rows(pair_examples(payload["examples"]), args.train_sample, args.seed)
    manifest = {
        "experiment": "E3b_external_negation",
        "audit": "manual_minimal_pair",
        "status": "pending_manual_audit",
        "split_json": str(args.split_json),
        "split_sha256": sha256_file(args.split_json),
        "train_sample": args.train_sample,
        "seed": args.seed,
        "rejection_reasons": REJECTION_REASONS,
        "n_items": len(rows),
        "rows": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_md(args.output_md, rows)
    print(f"Saved JSON -> {args.output_json}")
    print(f"Saved Markdown -> {args.output_md}")


if __name__ == "__main__":
    main()
