#!/usr/bin/env python3
"""Offline audit for frozen E3 negation task splits."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def cue_tokens(text: str) -> list[str]:
    import re

    pattern = re.compile(
        r"\b("
        r"no|not|never|denies|denied|negative|"
        r"cannot|can't|won't|isn't|aren't|wasn't|weren't|"
        r"doesn't|don't|didn't|hasn't|haven't|hadn't|"
        r"couldn't|wouldn't|shouldn't"
        r")\b",
        re.I,
    )
    return [match.group(0).lower() for match in pattern.finditer(text)]


def audit_split(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    examples = payload["examples"]
    report: dict[str, Any] = {
        "split_json": str(path),
        "status": payload.get("status"),
        "split_variant": payload.get("split_variant"),
        "primary_role": payload.get("primary_role"),
        "seed": payload.get("seed"),
        "splits": {},
        "global": {},
        "checks": {},
    }

    all_families = sorted({example["family"] for example in examples})
    for split in ("train", "dev", "test"):
        split_examples = [example for example in examples if example["split"] == split]
        text_counts = Counter(example["text"] for example in split_examples)
        cue_counter = Counter()
        for example in split_examples:
            for token in cue_tokens(example["text"]):
                cue_counter[token] += 1

        report["splits"][split] = {
            "n_examples": len(split_examples),
            "n_unique_texts": len(text_counts),
            "n_exact_text_duplicates": sum(count - 1 for count in text_counts.values() if count > 1),
            "n_pairs": len({example["global_pair_id"] for example in split_examples}),
            "n_templates": len({example["template_id"] for example in split_examples}),
            "families": dict(Counter(example["family"] for example in split_examples)),
            "labels": dict(Counter(example["label"] for example in split_examples)),
            "cue_tokens": dict(sorted(cue_counter.items())),
            "duplicate_texts": {
                text: count for text, count in sorted(text_counts.items()) if count > 1
            },
        }

    report["global"] = {
        "n_examples": len(examples),
        "n_unique_texts": len({example["text"] for example in examples}),
        "families": dict(Counter(example["family"] for example in examples)),
    }

    family_stratified = all(
        set(report["splits"][split]["families"]) == set(all_families)
        for split in ("train", "dev", "test")
    )
    report["checks"] = {
        "all_splits_have_both_families": family_stratified,
        "train_has_both_families": set(report["splits"]["train"]["families"]) == set(all_families),
        "dev_has_both_families": set(report["splits"]["dev"]["families"]) == set(all_families),
        "test_has_both_families": set(report["splits"]["test"]["families"]) == set(all_families),
    }
    return report


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        f"# E3 Negation Split Audit — {report['split_variant']}",
        "",
        f"Source: `{report['split_json']}`",
        f"Status: `{report['status']}`",
        f"Role: `{report['primary_role']}`",
        f"Seed: `{report['seed']}`",
        "",
        "## Split Breakdown",
        "",
        "| Split | examples | unique texts | dup rows | pairs | templates | negation | condition_negation |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split, block in report["splits"].items():
        families = block["families"]
        lines.append(
            f"| {split} | {block['n_examples']} | {block['n_unique_texts']} | "
            f"{block['n_exact_text_duplicates']} | {block['n_pairs']} | {block['n_templates']} | "
            f"{families.get('negation', 0)} | {families.get('condition_negation', 0)} |"
        )
    lines.extend(
        [
            "",
            "## Checks",
            "",
            f"- all_splits_have_both_families: `{report['checks']['all_splits_have_both_families']}`",
            "",
            "## Cue Tokens",
            "",
        ]
    )
    for split, block in report["splits"].items():
        lines.append(f"### {split}")
        lines.append("")
        if block["cue_tokens"]:
            for token, count in block["cue_tokens"].items():
                lines.append(f"- `{token}`: {count}")
        else:
            lines.append("- none")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit frozen E3 negation split JSON")
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    report = audit_split(args.split_json)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(args.output_md, report)
    print(f"Saved JSON -> {args.output_json}")
    print(f"Saved Markdown -> {args.output_md}")
    print(f"all_splits_have_both_families={report['checks']['all_splits_have_both_families']}")


if __name__ == "__main__":
    main()
