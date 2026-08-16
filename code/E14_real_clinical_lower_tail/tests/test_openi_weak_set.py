from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import openi_lower_tail as audit  # noqa: E402


def test_profile_default_token_caps_match_canonical_artifacts() -> None:
    assert audit.DEFAULT_MAX_LENGTHS["gpt2"] == 256
    assert audit.DEFAULT_MAX_LENGTHS["qwen-2.5-3b"] == 128


def test_own_tail_differs_from_fixed_standard_weak_set() -> None:
    std = np.array([1.0, 2.0, 100.0, 101.0, 102.0])
    vreg = np.array([100.0, 101.0, 1.0, 2.0, 3.0])
    own_tail = audit.d_l20(std, vreg)
    w_std = audit.stable_bottom_indices(std)
    fixed_delta = np.mean(vreg[w_std] - std[w_std])

    assert own_tail == 0.0
    assert fixed_delta == 99.0
    assert fixed_delta != own_tail


def test_worst_subset_invariant_is_exact_and_not_pointwise() -> None:
    std = np.array([1.0, 2.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0])
    vreg = np.array([0.5, 4.0, 3.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0])
    w_std = audit.stable_bottom_indices(std)

    check = audit.verify_worst_subset_invariant(std, vreg, w_std)

    assert check["passed"]
    assert check["fixed_delta"] >= check["own_tail_delta"]
    assert np.any(vreg[w_std] < std[w_std])


def test_worst_subset_invariant_failure_has_diagnostic_message() -> None:
    std = np.array([1.0, 2.0, 10.0, 11.0, 12.0])
    vreg = np.array([2.0, 3.0, 10.0, 11.0, 12.0])
    wrong_w_std = np.array([4])

    with np.testing.assert_raises_regex(
        AssertionError, "WORST_SUBSET_INVARIANT_FAILURE"
    ):
        audit.verify_worst_subset_invariant(std, vreg, wrong_w_std)


def test_w_std_indexes_are_shared_across_relative_and_absolute_endpoints() -> None:
    rel_std = np.array([0.3, 0.1, 0.2, 0.5, 0.4])
    rel_vreg = rel_std + 1.0
    abs_std = np.array([30.0, 10.0, 20.0, 50.0, 40.0])
    abs_vreg = abs_std + 2.0
    clusters = np.arange(5)
    w_std = audit.stable_bottom_indices(rel_std)

    panel = audit.fixed_set_panel(
        rel_std, rel_vreg, abs_std, abs_vreg, clusters, w_std,
        seed=7, n_boot=20,
    )

    assert w_std.tolist() == [1]
    assert panel["relative_std"]["mean"] == rel_std[1]
    assert panel["absolute_std"]["mean"] == abs_std[1]
    assert panel["relative_delta"]["mean"] == 1.0
    assert panel["absolute_delta"]["mean"] == 2.0


def test_reverse_selection_uses_vreg_ranking() -> None:
    std = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    vreg = np.array([0.5, 0.4, 0.3, 0.2, 0.1])

    assert audit.stable_bottom_indices(std).tolist() == [0]
    assert audit.stable_bottom_indices(vreg).tolist() == [4]


def test_standard_quintiles_are_exhaustive_ordered_partition() -> None:
    values = np.array([8, 3, 9, 0, 10, 1, 7, 2, 6, 4, 5], dtype=float)
    quintiles = audit.standard_quintile_indices(values)
    flattened = np.concatenate(quintiles)

    assert [len(q) for q in quintiles] == [3, 2, 2, 2, 2]
    assert sorted(flattened.tolist()) == list(range(len(values)))
    assert np.all(np.diff(values[flattened]) >= 0)


def test_ties_use_stable_original_index_order() -> None:
    tied = np.ones(10)

    assert audit.stable_bottom_indices(tied).tolist() == [0, 1]
    assert audit.standard_quintile_indices(tied)[0].tolist() == [0, 1]


def test_cluster_resampling_keeps_cluster_members_together() -> None:
    clusters = np.array(["a", "a", "b"])
    rng = np.random.default_rng(12)

    for _ in range(20):
        indexes, kind = audit._bootstrap_draw_indices(clusters, rng)
        assert kind == "cluster"
        assert np.sum(indexes == 0) == np.sum(indexes == 1)

    singleton_indexes, singleton_kind = audit._bootstrap_draw_indices(
        np.array(["a", "b", "c"]), rng,
    )
    assert singleton_kind == "pair"
    assert len(singleton_indexes) == 3


def test_quintile_profile_uses_fixed_partition_and_reports_intervals() -> None:
    rel_std = np.arange(10, dtype=float)
    rel_vreg = rel_std + 1.0
    abs_std = rel_std * 10.0
    abs_vreg = abs_std + 2.0
    clusters = np.array([f"r{i}" for i in range(10)])

    profile = audit.quintile_profile(
        rel_std,
        rel_vreg,
        abs_std,
        abs_vreg,
        clusters,
        seed=500,
        n_boot=20,
    )

    assert list(profile) == ["Q1", "Q2", "Q3", "Q4", "Q5"]
    assert sum(row["n"] for row in profile.values()) == 10
    for row in profile.values():
        assert row["relative_delta"]["mean"] == 1.0
        assert row["absolute_delta"]["mean"] == 2.0
        assert row["relative_delta"]["ci"] == [1.0, 1.0]
