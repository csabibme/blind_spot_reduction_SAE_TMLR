#!/usr/bin/env python3
"""Model-free preflight for E3b OpenI negation splits."""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from pathlib import Path

import numpy as np

REVISION_ROOT = Path(__file__).resolve().parents[1]
E3_ROOT = REVISION_ROOT / "E3_heldout_probes"
if str(E3_ROOT) not in sys.path:
    sys.path.insert(0, str(E3_ROOT))

from run_e3_negation_probes import (  # noqa: E402
    ALLOWED_SPLIT_STATUSES,
    REPRESENTATIONS,
    deduplicate_examples,
    deduplication_summary,
    load_feature_cache,
    read_json,
    save_feature_cache,
)

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


def split_label_counts(examples: list[dict]) -> dict[str, dict[str, int]]:
    labels = sorted({example["label"] for example in examples})
    out = {split: {label: 0 for label in labels} for split in ("train", "dev", "test")}
    for example in examples:
        out[example["split"]][example["label"]] += 1
    return out


def check_feature_cache_roundtrip() -> dict[str, bool]:
    features = {name: np.zeros((3, 4), dtype=np.float32) for name in REPRESENTATIONS}
    meta = {"e3b_preflight": True}
    with tempfile.TemporaryDirectory() as tmpdir:
        test_path = Path(tmpdir) / "cache_test.npz"
        save_feature_cache(test_path, features, meta)
        loaded = load_feature_cache(test_path, meta)
        return {
            "cache_file_exists": test_path.is_file(),
            "cache_load_not_none": loaded is not None,
            "cache_all_representations": loaded is not None and all(name in loaded for name in REPRESENTATIONS),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Preflight E3b OpenI negation split")
    parser.add_argument("--split-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    payload = read_json(args.split_json)
    if payload.get("status") not in ALLOWED_SPLIT_STATUSES:
        raise ValueError(f"Unexpected split status: {payload.get('status')!r}")
    raw_examples = payload["examples"]
    examples = deduplicate_examples(raw_examples)
    summary = deduplication_summary(raw_examples, examples)
    labels = split_label_counts(examples)
    cache_checks = check_feature_cache_roundtrip()
    report_splits: dict[str, set[str]] = {}
    for example in examples:
        report_splits.setdefault(example["report_id"], set()).add(example["split"])
    manual_audit = payload.get("manual_audit", {})
    label_set = sorted({example["label"] for example in examples})
    is_negation = set(label_set) == {"affirmed", "negated"}
    bad_affirmed = [
        example["global_pair_id"]
        for example in examples
        if example["label"] == "affirmed"
        and any(re.search(pattern, example["text"], re.I) for pattern in BAD_AFFIRMED_PATTERNS)
    ]

    checks = {
        "no_deduplication_loss": len(raw_examples) == len(examples),
        "all_splits_nonempty": all(summary[f"{split}_examples"] > 0 for split in ("train", "dev", "test")),
        "labels_balanced_by_split": all(len(set(counts.values())) == 1 for counts in labels.values()),
        "no_report_split_leakage": all(len(splits) == 1 for splits in report_splits.values()),
        "all_retained_pairs_manually_accepted": (not is_negation)
        or manual_audit.get("all_retained_pairs_manually_accepted") is True,
        "all_source_pairs_had_decisions": (not is_negation) or manual_audit.get("all_source_pairs_had_decisions") is True,
        "no_bad_affirmed_guard_patterns": (not is_negation) or len(bad_affirmed) == 0,
        "random_label_seed_count": len(payload["random_label_control"]["seeds"]) == 5,
        "representation_count": len(payload.get("probe_protocol", {}).get("representations", REPRESENTATIONS)) == 5,
        "feature_cache_smoke": all(cache_checks.values()),
    }
    report = {
        "split_json": str(args.split_json),
        "status": payload.get("status"),
        "dataset_kind": payload.get("dataset_kind"),
        "example_summary": summary,
        "label_counts": labels,
        "checks": checks,
        "bad_affirmed_guard_pair_ids": bad_affirmed[:20],
        "feature_cache": cache_checks,
        "all_pass": all(checks.values()),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved JSON -> {args.output_json}")
    print(f"all_pass={report['all_pass']}")
    if not report["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
