#!/usr/bin/env python3
"""Model-free preflight checks for E3 negation probe runs."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_e3_negation_probes import (  # noqa: E402
    ALLOWED_SPLIT_STATUSES,
    REPRESENTATIONS,
    deduplicate_examples,
    deduplication_summary,
    load_feature_cache,
    read_json,
    save_feature_cache,
)

EXPECTED = {
    "raw_examples": 156,
    "deduplicated_examples": 96,
    "train_examples": 58,
    "dev_examples": 20,
    "test_examples": 18,
    "train_labels": {"affirmed": 29, "negated": 29},
    "dev_labels": {"affirmed": 10, "negated": 10},
    "test_labels": {"affirmed": 9, "negated": 9},
    "representations": len(REPRESENTATIONS),
    "random_label_seeds": 5,
}


def check_split_payload(path: Path) -> dict:
    payload = read_json(path)
    if payload.get("status") not in ALLOWED_SPLIT_STATUSES:
        raise ValueError(f"Unexpected split status: {payload.get('status')!r}")

    raw_examples = payload["examples"]
    examples = deduplicate_examples(raw_examples)
    summary = deduplication_summary(raw_examples, examples)
    random_label_seeds = payload["random_label_control"]["seeds"]
    probe_protocol = payload.get("probe_protocol", {})
    representations = probe_protocol.get("representations", list(REPRESENTATIONS))

    checks = {
        "raw_examples": summary["raw_examples"] == EXPECTED["raw_examples"],
        "deduplicated_examples": summary["deduplicated_examples"] == EXPECTED["deduplicated_examples"],
        "train_examples": summary["train_examples"] == EXPECTED["train_examples"],
        "dev_examples": summary["dev_examples"] == EXPECTED["dev_examples"],
        "test_examples": summary["test_examples"] == EXPECTED["test_examples"],
        "train_labels": summary["train_labels"] == EXPECTED["train_labels"],
        "dev_labels": summary["dev_labels"] == EXPECTED["dev_labels"],
        "test_labels": summary["test_labels"] == EXPECTED["test_labels"],
        "representations": len(representations) == EXPECTED["representations"],
        "random_label_seeds": len(random_label_seeds) == EXPECTED["random_label_seeds"],
    }
    return {
        "split_json": str(path),
        "summary": summary,
        "checks": checks,
        "all_pass": all(checks.values()),
    }


def check_feature_cache_roundtrip() -> dict[str, bool]:
    features = {name: np.zeros((3, 4), dtype=np.float32) for name in REPRESENTATIONS}
    meta = {"test": True}
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
    parser = argparse.ArgumentParser(description="Preflight E3 negation probe inputs")
    parser.add_argument(
        "--task-split-json",
        type=Path,
        default=SCRIPT_DIR / "results" / "e3_negation_task_split.json",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    split_report = check_split_payload(args.task_split_json)
    cache_checks = check_feature_cache_roundtrip()
    cache_pass = all(cache_checks.values())

    report = {
        "split": split_report,
        "feature_cache": {
            "checks": cache_checks,
            "all_pass": cache_pass,
        },
        "all_pass": split_report["all_pass"] and cache_pass,
    }

    print("Preflight summary")
    print(f"  raw examples:        {split_report['summary']['raw_examples']}")
    print(f"  unique examples:     {split_report['summary']['deduplicated_examples']}")
    print(
        "  train/dev/test:      "
        f"{split_report['summary']['train_examples']}/"
        f"{split_report['summary']['dev_examples']}/"
        f"{split_report['summary']['test_examples']}"
    )
    print(
        "  labels train:        "
        f"{split_report['summary']['train_labels']['affirmed']}/"
        f"{split_report['summary']['train_labels']['negated']}"
    )
    print(
        "  labels dev:          "
        f"{split_report['summary']['dev_labels']['affirmed']}/"
        f"{split_report['summary']['dev_labels']['negated']}"
    )
    print(
        "  labels test:         "
        f"{split_report['summary']['test_labels']['affirmed']}/"
        f"{split_report['summary']['test_labels']['negated']}"
    )
    print(f"  representations:     {EXPECTED['representations']}")
    print(f"  random-label seeds:  {EXPECTED['random_label_seeds']}")
    print(f"  feature cache smoke: {cache_pass}")

    for name, passed in split_report["checks"].items():
        print(f"  check {name}: {'PASS' if passed else 'FAIL'}")

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Saved JSON -> {args.output_json}")

    if not report["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
