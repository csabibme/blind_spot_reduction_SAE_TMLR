from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import verify_exact_lower_tail as audit  # noqa: E402


def brute_force_minimum(values: np.ndarray, m: int) -> float:
    return min(
        float(np.mean(values[list(indexes)]))
        for indexes in itertools.combinations(range(len(values)), m)
    )


def test_worst_subset_characterization_matches_brute_force() -> None:
    values = np.array([4.0, 1.0, 7.0, 2.0, 9.0, 3.0])
    m = audit.tail_size(len(values))

    assert audit.lower_tail_mean(values) == brute_force_minimum(values, m)


def test_ties_change_minimizer_identity_not_minimum_value() -> None:
    values = np.array([1.0, 1.0, 1.0, 5.0, 9.0, 10.0])
    m = audit.tail_size(len(values))
    minimizers = [
        indexes
        for indexes in itertools.combinations(range(len(values)), m)
        if np.mean(values[list(indexes)]) == audit.lower_tail_mean(values)
    ]

    assert len(minimizers) > 1
    assert audit.stable_bottom_indices(values).tolist() == [0, 1]
    assert all(np.mean(values[list(indexes)]) == 1.0 for indexes in minimizers)


def test_standard_fixed_delta_is_at_least_own_tail_delta() -> None:
    standard = np.array([1.0, 2.0, 10.0, 11.0, 12.0])
    vreg = np.array([1.5, 4.0, 3.0, 11.0, 12.0])
    weak = audit.stable_bottom_indices(standard)
    own_tail_delta = audit.lower_tail_mean(vreg) - audit.lower_tail_mean(standard)
    fixed_delta = float(np.mean(vreg[weak] - standard[weak]))

    assert fixed_delta >= own_tail_delta


def test_subset_average_guarantee_does_not_imply_pointwise_improvement() -> None:
    standard = np.array([1.0, 2.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0])
    vreg = np.array([0.5, 4.0, 3.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0])
    weak = audit.stable_bottom_indices(standard)
    own_tail_delta = audit.lower_tail_mean(vreg) - audit.lower_tail_mean(standard)
    fixed_delta = float(np.mean(vreg[weak] - standard[weak]))

    assert own_tail_delta > 0
    assert fixed_delta >= own_tail_delta
    assert np.any(vreg[weak] < standard[weak])


def test_lorenz_gini_identity_matches_pairwise_form() -> None:
    values = np.array([0.1, 0.2, 0.5, 1.3, 2.1])

    assert audit.gini_sorted(values) == pytest.approx(
        audit.gini_pairwise(values), abs=1e-15
    )
    assert audit.gini_from_cumulative_lower_tail(values) == pytest.approx(
        audit.gini_pairwise(values), abs=1e-15
    )


def test_sorted_gini_includes_double_sum_factor() -> None:
    values = np.array([0.0, 1.0])

    assert audit.gini_pairwise(values) == pytest.approx(0.5)
    assert audit.gini_sorted(values) == pytest.approx(0.5)


def test_zero_profile_uses_explicit_gini_convention() -> None:
    values = np.zeros(5)

    assert audit.gini_pairwise(values) == 0.0
    assert audit.gini_sorted(values) == 0.0
    assert audit.gini_from_cumulative_lower_tail(values) == 0.0


def test_all_openi_cells_satisfy_full_precision_invariant() -> None:
    payload = audit.run_audit(audit.DEFAULT_INPUTS, tolerance=1e-12)

    assert payload["status"] == "PASS"
    assert sum(
        len(model["families"]) for model in payload["models"].values()
    ) == 8
    for model in payload["models"].values():
        assert model["all_checks_pass"]
        for cell in model["families"].values():
            assert cell["relative"]["worst_subset_slack"] >= -1e-12
