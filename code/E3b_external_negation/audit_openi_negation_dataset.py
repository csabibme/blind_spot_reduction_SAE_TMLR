#!/usr/bin/env python3
"""Audit frozen E3b OpenI negation split JSON files."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def audit(path: Path) -> dict:
    payload = read_json(path)
    examples = payload["examples"]
    require_paired = payload.get("dataset_kind") != "natural_classification"
    expected_labels = set(payload.get("label_rule", {}).get("labels", ["affirmed", "negated"]))
    report_splits: dict[str, set[str]] = defaultdict(set)
    pair_labels: dict[str, set[str]] = defaultdict(set)
    text_splits: dict[str, set[str]] = defaultdict(set)
    for example in examples:
        report_splits[example["report_id"]].add(example["split"])
        pair_labels[example["global_pair_id"]].add(example["label"])
        text_splits[example["text"].lower()].add(example["split"])

    split_blocks = {}
    for split in ("train", "dev", "test"):
        split_examples = [example for example in examples if example["split"] == split]
        split_blocks[split] = {
            "n_examples": len(split_examples),
            "n_pairs": len({example["global_pair_id"] for example in split_examples}),
            "n_reports": len({example["report_id"] for example in split_examples}),
            "labels": dict(Counter(example["label"] for example in split_examples)),
            "concepts": dict(Counter(example["concept"] for example in split_examples)),
        }

    return {
        "split_json": str(path),
        "status": payload.get("status"),
        "dataset_kind": payload.get("dataset_kind"),
        "summary": payload.get("summary", {}),
        "split_summary": split_blocks,
        "checks": {
            "report_split_leakage_count": sum(1 for splits in report_splits.values() if len(splits) > 1),
            "pair_label_errors_count": sum(
                1 for labels in pair_labels.values() if require_paired and labels != expected_labels
            ),
            "exact_text_cross_split_count": sum(1 for splits in text_splits.values() if len(splits) > 1),
            "external_exact_overlap_count": payload.get("overlap_audit", {}).get(
                "exact_normalized_overlap_count"
            ),
        },
        "overlap_audit": payload.get("overlap_audit", {}),
    }


def write_markdown(path: Path, report: dict) -> None:
    lines = [
        f"# E3b OpenI Dataset Audit — {report['dataset_kind']}",
        "",
        f"Source: `{report['split_json']}`",
        f"Status: `{report['status']}`",
        "",
        "## Split Summary",
        "",
        "| Split | examples | pairs | reports | labels |",
        "|---|---:|---:|---:|---|",
    ]
    for split, block in report["split_summary"].items():
        labels = block["labels"]
        lines.append(
            f"| {split} | {block['n_examples']} | {block['n_pairs']} | {block['n_reports']} | "
            f"`{labels}` |"
        )
    lines.extend(["", "## Checks", ""])
    for key, value in report["checks"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(
        [
            "",
            "## Externality",
            "",
            f"- exact normalized overlap with E3 600-pair source: `{report['checks']['external_exact_overlap_count']}`",
            f"- max 5-gram Jaccard max: `{report['overlap_audit'].get('max_5gram_jaccard_max')}`",
            f"- max 5-gram Jaccard q95: `{report['overlap_audit'].get('max_5gram_jaccard_q95')}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit E3b OpenI negation split")
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.split_json)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(args.output_md, report)
    print(f"Saved JSON -> {args.output_json}")
    print(f"Saved Markdown -> {args.output_md}")


if __name__ == "__main__":
    main()
