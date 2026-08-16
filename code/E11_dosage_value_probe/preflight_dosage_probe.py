#!/usr/bin/env python3
"""Model-free preflight checks for held-out dosage numeric probe runs."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dosage_probe_common import (  # noqa: E402
    LABEL_CRITICAL,
    LABEL_NUISANCE,
    REPRESENTATIONS,
    STATUS_FROZEN,
    read_json,
    split_summary,
    validate_pairs,
    write_json,
)
from run_dosage_probe_features import (  # noqa: E402
    PAIR_REPRESENTATIONS,
    load_feature_cache,
    save_feature_cache,
)

SMOKE_EXPECTED = {
    "n_templates": 10,
    "n_pairs_min": 80,
    "train_templates_min": 6,
    "dev_templates_min": 2,
    "test_templates_min": 2,
}

FULL_EXPECTED = {
    "n_templates": 24,
    "n_pairs_min": 400,
    "train_templates_min": 12,
    "dev_templates_min": 6,
    "test_templates_min": 6,
}


def check_dataset_payload(path: Path, smoke: bool) -> dict:
    payload = read_json(path)
    if payload.get("status") != STATUS_FROZEN:
        raise ValueError(f"Unexpected dataset status: {payload.get('status')!r}")
    pairs = payload["pairs"]
    validate_pairs(pairs)
    summary = split_summary(pairs)
    expected = SMOKE_EXPECTED if smoke else FULL_EXPECTED

    test_templates = {p["template_id"] for p in pairs if p["split"] == "test"}
    train_templates = {p["template_id"] for p in pairs if p["split"] == "train"}
    dev_templates = {p["template_id"] for p in pairs if p["split"] == "dev"}

    n_templates = len({p["template_id"] for p in pairs})
    checks = {
        "status_frozen": payload.get("status") == STATUS_FROZEN,
        "not_in_v_loss": payload.get("not_in_v_loss") is True,
        "n_templates": n_templates >= expected["n_templates"],
        "n_pairs": len(pairs) >= expected["n_pairs_min"],
        "train_templates": summary["splits"]["train"]["n_templates"] >= expected["train_templates_min"],
        "dev_templates": summary["splits"]["dev"]["n_templates"] >= expected["dev_templates_min"],
        "test_templates": summary["splits"]["test"]["n_templates"] >= expected["test_templates_min"],
        "template_disjoint_test_train": not (test_templates & train_templates),
        "template_disjoint_test_dev": not (test_templates & dev_templates),
        "both_labels_present": all(
            summary["labels"][split][LABEL_CRITICAL] > 0 and summary["labels"][split][LABEL_NUISANCE] > 0
            for split in ("train", "dev", "test")
        ),
        "no_identical_pairs": all(p["text_left"] != p["text_right"] for p in pairs),
        "both_families_in_each_split": all(
            {"dose_count", "dose_amount"} <= set(summary["families"][split])
            for split in ("train", "dev", "test")
        ),
        "representations": len(REPRESENTATIONS) == 3,
    }
    return {
        "dataset_json": str(path),
        "smoke": smoke,
        "summary": {
            "n_pairs": len(pairs),
            "n_templates": n_templates,
            **summary,
        },
        "checks": checks,
        "all_pass": all(checks.values()),
    }


def check_feature_cache_roundtrip() -> dict[str, bool]:
    features = {name: np.zeros((3, 4), dtype=np.float32) for name in PAIR_REPRESENTATIONS}
    meta = {"test": True}
    with tempfile.TemporaryDirectory() as tmpdir:
        test_path = Path(tmpdir) / "cache_test.npz"
        save_feature_cache(test_path, features, meta)
        loaded = load_feature_cache(test_path, meta)
        return {
            "cache_file_exists": test_path.is_file(),
            "cache_load_not_none": loaded is not None,
            "cache_all_representations": loaded is not None
            and all(name in loaded for name in PAIR_REPRESENTATIONS),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Preflight held-out dosage numeric probe inputs")
    parser.add_argument(
        "--dataset-json",
        type=Path,
        required=True,
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    dataset_report = check_dataset_payload(args.dataset_json, smoke=args.smoke)
    cache_checks = check_feature_cache_roundtrip()
    cache_pass = all(cache_checks.values())

    report = {
        "dataset": dataset_report,
        "feature_cache": {
            "checks": cache_checks,
            "all_pass": cache_pass,
        },
        "all_pass": dataset_report["all_pass"] and cache_pass,
    }

    print("Preflight summary")
    print(f"  pairs:               {dataset_report['summary']['n_pairs']}")
    print(
        "  train/dev/test:      "
        f"{dataset_report['summary']['splits']['train']['n_pairs']}/"
        f"{dataset_report['summary']['splits']['dev']['n_pairs']}/"
        f"{dataset_report['summary']['splits']['test']['n_pairs']}"
    )
    print(f"  feature cache smoke: {cache_pass}")
    for name, passed in dataset_report["checks"].items():
        print(f"  check {name}: {'PASS' if passed else 'FAIL'}")

    if args.output_json is not None:
        write_json(args.output_json, report)
        print(f"Saved JSON -> {args.output_json}")

    if not report["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
