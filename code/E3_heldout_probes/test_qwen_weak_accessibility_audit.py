#!/usr/bin/env python3
"""Regression tests for the predeclared Qwen weak-accessibility audit."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from run_qwen_weak_accessibility_audit import (
    build_feature_indices,
    exact_binomial_two_sided,
    grouped_folds,
    mcnemar_cells,
    pair_map,
    quintile_analysis,
    run_analysis,
    sanitize_public_metadata,
    weak_ids_pooled,
    weak_ids_within_family,
    write_json,
)


def synthetic_inputs(root: Path) -> tuple[Path, Path, Path]:
    examples = []
    standard = []
    vreg = []
    for index in range(78):
        family = "condition_negation" if index < 28 else "negation"
        template = f"{family}::template_{index // 2:02d}"
        pair_id = f"{family}::pair_{index:02d}"
        orig = np.asarray([1.0, index / 100.0, 0.0, 0.0], dtype=np.float32)
        std_pert = orig + np.asarray([0.0, 0.0, 0.01 + index / 1000.0, 0.0], dtype=np.float32)
        vreg_pert = orig + np.asarray([0.0, 0.0, 0.02 + index / 800.0, 0.1], dtype=np.float32)
        for side, label, vector_std, vector_vreg in (
            ("orig", "affirmed", orig, orig),
            ("pert", "negated", std_pert, vreg_pert),
        ):
            examples.append(
                {
                    "example_id": f"{pair_id}::{side}",
                    "global_pair_id": pair_id,
                    "template_id": template,
                    "family": family,
                    "side": side,
                    "label": label,
                    "split": "train",
                    "text": f"{pair_id} {side}",
                }
            )
            standard.append(vector_std)
            vreg.append(vector_vreg)
    split_path = root / "split.json"
    split_path.write_text(json.dumps({"examples": examples}), encoding="utf-8")
    cache_path = root / "cache.npz"
    np.savez_compressed(
        cache_path,
        sae_standard_code=np.asarray(standard),
        sae_vreg_code=np.asarray(vreg),
        metadata=np.asarray({"synthetic": True}, dtype=object),
    )
    protocol = root / "protocol.md"
    protocol.write_text("predeclared\n", encoding="utf-8")
    return split_path, cache_path, protocol


class AuditUnitTests(unittest.TestCase):
    def test_public_metadata_sanitizes_local_paths(self) -> None:
        metadata = {
            "checkpoint": (
                "/" + "Users/example/repo/SAE/FINAL/tmlr_revision/prepare/"
                "experiment_101_hybrid_owt/checkpoints/qwen/model"
            ),
            "repo_file": "/" + "Users/example/repo/SAE/FINAL/data.json",
        }
        sanitized = sanitize_public_metadata(metadata)
        self.assertEqual(sanitized["checkpoint"], "<CKPT_ROOT>/qwen/model")
        self.assertEqual(sanitized["repo_file"], "<REPO_ROOT>/FINAL/data.json")

    def setUp(self) -> None:
        self.records = [
            {
                "pair_id": f"p{i:02d}",
                "template_id": f"t{i // 2:02d}",
                "family": "a" if i < 10 else "b",
                "D_standard": float(i),
                "D_vreg": float(19 - i),
                "standard_margin": i / 100.0 - 0.05,
                "vreg_margin": i / 100.0 - 0.04,
            }
            for i in range(20)
        ]

    def test_weak_partitions(self) -> None:
        self.assertEqual(weak_ids_pooled(self.records), ["p00", "p01", "p02", "p03"])
        self.assertEqual(
            weak_ids_within_family(self.records),
            ["p00", "p01", "p10", "p11"],
        )

    def test_quintiles_cover_once_and_are_ordered(self) -> None:
        quintiles = quintile_analysis(self.records)
        ids = [pair_id for block in quintiles for pair_id in block["pair_ids"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {row["pair_id"] for row in self.records})
        self.assertEqual([block["n"] for block in quintiles], [4, 4, 4, 4, 4])
        self.assertLess(quintiles[0]["D_standard_max"], quintiles[-1]["D_standard_min"])

    def test_mcnemar_cells_and_exact_test(self) -> None:
        std = np.asarray([True, True, False, False, True])
        vreg = np.asarray([True, False, True, False, True])
        cells = mcnemar_cells(std, vreg)
        self.assertEqual(
            [cells["both_correct"], cells["standard_only_correct"], cells["vreg_only_correct"], cells["both_wrong"]],
            [2, 1, 1, 1],
        )
        self.assertEqual(sum(cells[key] for key in ("both_correct", "standard_only_correct", "vreg_only_correct", "both_wrong")), 5)
        self.assertEqual(cells["exact_two_sided_p"], 1.0)
        self.assertAlmostEqual(exact_binomial_two_sided(0, 3), 0.25)

    def test_grouped_folds_keep_templates_intact(self) -> None:
        examples = []
        for row in self.records:
            for side in ("orig", "pert"):
                examples.append(
                    {
                        "global_pair_id": row["pair_id"],
                        "template_id": row["template_id"],
                        "family": row["family"],
                        "side": side,
                    }
                )
        pairs = pair_map(examples)
        folds = grouped_folds(pairs, pairs, 5, 7)
        self.assertEqual(set(folds), {row["template_id"] for row in self.records})
        self.assertEqual(set(folds.values()), set(range(5)))

    def test_index_alignment_reuses_only_exact_dedup_keys(self) -> None:
        examples = [
            {"split": "train", "template_id": "t", "text": "x", "label": "affirmed"},
            {"split": "train", "template_id": "t", "text": "x", "label": "affirmed"},
            {"split": "train", "template_id": "t", "text": "y", "label": "negated"},
        ]
        np.testing.assert_array_equal(build_feature_indices(examples, 2), [0, 0, 1])


class AuditIntegrationTests(unittest.TestCase):
    def test_exactly_once_oof_and_deterministic_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            split, cache, protocol = synthetic_inputs(root)
            first = run_analysis(
                split,
                cache,
                protocol,
                n_boot=20,
                seed=123,
                analysis_run_date="2099-01-01",
            )
            second = run_analysis(
                split,
                cache,
                protocol,
                n_boot=20,
                seed=123,
                analysis_run_date="2099-01-01",
            )
            self.assertEqual(first["analysis_run_date"], "2099-01-01")
            self.assertTrue(first["no_leakage"]["all_pass"])
            self.assertEqual(first["subsets"]["pooled_Wstd"]["n"], 16)
            self.assertEqual(first["subsets"]["pooled_nonweak"]["n"], 62)
            self.assertEqual(
                set(first["subsets"]["pooled_Wstd"]["pair_ids"])
                & set(first["subsets"]["pooled_nonweak"]["pair_ids"]),
                set(),
            )
            for method in ("standard", "vreg"):
                predictions = first["oof_predictions"][method]
                self.assertEqual(len(predictions), 156)
                self.assertEqual(len({row["example_id"] for row in predictions}), 156)
                self.assertTrue(all(0 <= row["outer_fold"] < 5 for row in predictions))
            first_path = root / "first.json"
            second_path = root / "second.json"
            write_json(first_path, first)
            write_json(second_path, second)
            self.assertEqual(first_path.read_bytes(), second_path.read_bytes())


if __name__ == "__main__":
    unittest.main()
