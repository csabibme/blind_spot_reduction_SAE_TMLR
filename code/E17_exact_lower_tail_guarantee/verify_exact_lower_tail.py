#!/usr/bin/env python3
"""Verify the exact finite-family worst-subset lower-tail guarantee."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
RESULTS_ROOT = REPO_ROOT / "results"
E14_RESULTS = RESULTS_ROOT / "e14_real_clinical_lower_tail"
DEFAULT_INPUTS = {
    "GPT-2": E14_RESULTS / "openi_standard_weakset_v1_gpt2.json",
    "Qwen 2.5": E14_RESULTS / "openi_standard_weakset_v1_qwen_exact128.json",
}
DEFAULT_OUTPUT_DIR = RESULTS_ROOT / "e17_exact_lower_tail_guarantee"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def public_repo_path(path: Path) -> str:
    """Return a repository-relative public path without local user details."""
    resolved = path.resolve()
    try:
        return f"<REPO_ROOT>/{resolved.relative_to(REPO_ROOT.resolve()).as_posix()}"
    except ValueError:
        return str(path)


def tail_size(n: int, fraction: float = 0.20) -> int:
    if n <= 0:
        raise ValueError("The response profile must be non-empty")
    if not 0 < fraction <= 1:
        raise ValueError("fraction must lie in (0, 1]")
    return max(1, math.ceil(fraction * n))


def stable_bottom_indices(values: np.ndarray, fraction: float = 0.20) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return np.argsort(array, kind="stable")[: tail_size(len(array), fraction)]


def lower_tail_mean(values: np.ndarray, fraction: float = 0.20) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.sort(array)[: tail_size(len(array), fraction)].mean())


def gini_pairwise(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    if np.any(array < 0):
        raise ValueError("Gini response values must be nonnegative")
    total = float(array.sum())
    if total == 0:
        return 0.0
    n = len(array)
    return float(np.abs(array[:, None] - array[None, :]).sum() / (2 * n * total))


def gini_sorted(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    if np.any(array < 0):
        raise ValueError("Gini response values must be nonnegative")
    total = float(array.sum())
    if total == 0:
        return 0.0
    ordered = np.sort(array)
    n = len(ordered)
    ranks = np.arange(1, n + 1, dtype=np.float64)
    return float(2 * np.dot(ranks, ordered) / (n * total) - (n + 1) / n)


def gini_from_cumulative_lower_tail(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    if np.any(array < 0):
        raise ValueError("Gini response values must be nonnegative")
    total = float(array.sum())
    if total == 0:
        return 0.0
    n = len(array)
    shares = np.sort(array / total)
    cumulative = np.cumsum(shares)[:-1]
    return float((n - 1) / n - (2 / n) * cumulative.sum())


def audit_cell(
    family_payload: dict[str, Any],
    tolerance: float,
) -> dict[str, Any]:
    rows = family_payload["per_pair"]
    standard = np.asarray([row["relative_D_std"] for row in rows], dtype=np.float64)
    vreg = np.asarray([row["relative_D_vreg"] for row in rows], dtype=np.float64)
    w_std = stable_bottom_indices(standard)
    stored_w_std = np.asarray(family_payload["W_std"]["indexes"], dtype=np.int64)

    l20_standard = lower_tail_mean(standard)
    l20_vreg = lower_tail_mean(vreg)
    own_tail_delta = l20_vreg - l20_standard
    fixed_standard = float(standard[w_std].mean())
    fixed_vreg = float(vreg[w_std].mean())
    fixed_delta = fixed_vreg - fixed_standard
    slack = fixed_delta - own_tail_delta
    identity_error = fixed_standard - l20_standard
    stored_indexes_match = bool(np.array_equal(stored_w_std, w_std))

    if abs(identity_error) > tolerance:
        raise AssertionError(
            "WORST_SUBSET_IDENTITY_FAILURE: Standard weak-set mean must equal "
            f"Standard L20; error={identity_error:.17g}"
        )
    if slack < -tolerance:
        raise AssertionError(
            "WORST_SUBSET_INVARIANT_FAILURE: fixed Standard-weak-set delta "
            "must be >= own-tail L20 delta under identical universe, weighting, "
            f"response definition, and tail cardinality; slack={slack:.17g}"
        )
    if not stored_indexes_match:
        raise AssertionError(
            "WORST_SUBSET_INDEX_FAILURE: stored W_std does not match stable "
            "bottom-ceil-20-percent selection"
        )

    gini_standard_pairwise = gini_pairwise(standard)
    gini_standard_sorted = gini_sorted(standard)
    gini_standard_lorenz = gini_from_cumulative_lower_tail(standard)
    gini_vreg_pairwise = gini_pairwise(vreg)
    gini_vreg_sorted = gini_sorted(vreg)
    gini_vreg_lorenz = gini_from_cumulative_lower_tail(vreg)
    gini_error = max(
        abs(gini_standard_pairwise - gini_standard_sorted),
        abs(gini_standard_pairwise - gini_standard_lorenz),
        abs(gini_vreg_pairwise - gini_vreg_sorted),
        abs(gini_vreg_pairwise - gini_vreg_lorenz),
    )
    if gini_error > tolerance:
        raise AssertionError(
            "LORENZ_GINI_IDENTITY_FAILURE: pairwise and cumulative forms differ; "
            f"max_error={gini_error:.17g}"
        )

    fixed_panel = family_payload["W_std"]["paired_panel"]
    absolute = fixed_panel["absolute_delta"]
    fractions = family_payload["W_std"]["fraction_improved"]
    return {
        "n": len(rows),
        "m": len(w_std),
        "stored_W_std_matches": stored_indexes_match,
        "relative": {
            "L20_standard": l20_standard,
            "L20_vreg": l20_vreg,
            "own_tail_delta": own_tail_delta,
            "fixed_W_std_standard": fixed_standard,
            "fixed_W_std_vreg": fixed_vreg,
            "fixed_W_std_delta": fixed_delta,
            "fixed_W_std_delta_ci": fixed_panel["relative_delta"]["ci"],
            "worst_subset_slack": slack,
            "fraction_pairs_improved": fractions["relative_D"],
        },
        "absolute_on_relative_W_std": {
            "fixed_W_std_delta": absolute["mean"],
            "fixed_W_std_delta_ci": absolute["ci"],
            "fraction_pairs_improved": fractions["absolute_dz"],
            "status": "empirical_not_implied_by_relative_worst_subset_theorem",
        },
        "lorenz_gini": {
            "standard": gini_standard_lorenz,
            "vreg": gini_vreg_lorenz,
            "delta": gini_vreg_lorenz - gini_standard_lorenz,
            "max_identity_error": gini_error,
        },
        "checks": {
            "standard_minimizer_identity": abs(identity_error) <= tolerance,
            "fixed_delta_ge_own_tail_delta": slack >= -tolerance,
            "stored_W_std_matches": stored_indexes_match,
            "lorenz_gini_identity": gini_error <= tolerance,
        },
    }


def run_audit(inputs: dict[str, Path], tolerance: float = 1e-12) -> dict[str, Any]:
    models: dict[str, Any] = {}
    for model, path in inputs.items():
        payload = json.loads(path.read_text(encoding="utf-8"))
        families = {
            family: audit_cell(row, tolerance)
            for family, row in payload["families"].items()
        }
        models[model] = {
            "source": public_repo_path(path),
            "source_sha256": sha256_file(path),
            "families": families,
            "all_checks_pass": all(
                all(cell["checks"].values()) for cell in families.values()
            ),
        }
    return {
        "analysis": "Exact finite-family worst-subset lower-tail guarantee",
        "status": "PASS"
        if all(model["all_checks_pass"] for model in models.values())
        else "FAIL",
        "tolerance": tolerance,
        "assumptions": [
            "identical finite evaluated pair universe",
            "equal pair weighting",
            "identical response definition",
            "identical tail cardinality m=ceil(0.20*n)",
        ],
        "formulas": {
            "worst_subset": (
                "L20(D)=min_{W subset I, |W|=m} mean_{i in W} D_i"
            ),
            "corollary": (
                "mean_V(W_S)-mean_S(W_S) >= L20(D_V)-L20(D_S)"
            ),
            "lorenz_gini": (
                "G=(n-1)/n-(2/n)*sum_{k=1}^{n-1} Lambda_k, "
                "Lambda_k = cumulative share of k smallest normalised responses"
            ),
        },
        "scope": (
            "Subset-average guarantee only; no pointwise improvement claim. "
            "Absolute displacement and probe endpoints remain empirical."
        ),
        "models": models,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Exact finite-family worst-subset lower-tail audit",
        "",
        f"Status: **{payload['status']}**  ",
        f"Numerical tolerance: `{payload['tolerance']:.1e}`",
        "",
        "For the same equally weighted finite evaluation family and "
        "`m = ceil(0.20*n)`, the Standard-defined fixed-set relative delta "
        "must be at least the independently selected own-tail delta.",
        "",
        "| Model | Family | n | m | own-tail delta | fixed Wstd delta [95% CI] | "
        "slack | relative improved | absolute delta [95% CI] | absolute improved |",
        "|---|---|---:|---:|---:|---|---:|---:|---|---:|",
    ]
    for model, model_payload in payload["models"].items():
        for family, cell in model_payload["families"].items():
            relative = cell["relative"]
            absolute = cell["absolute_on_relative_W_std"]
            rel_ci = relative["fixed_W_std_delta_ci"]
            abs_ci = absolute["fixed_W_std_delta_ci"]
            lines.append(
                f"| {model} | {family} | {cell['n']} | {cell['m']} | "
                f"{relative['own_tail_delta']:+.6f} | "
                f"{relative['fixed_W_std_delta']:+.6f} "
                f"[{rel_ci[0]:+.6f}, {rel_ci[1]:+.6f}] | "
                f"{relative['worst_subset_slack']:+.3e} | "
                f"{relative['fraction_pairs_improved']:.3f} | "
                f"{absolute['fixed_W_std_delta']:+.4f} "
                f"[{abs_ci[0]:+.4f}, {abs_ci[1]:+.4f}] | "
                f"{absolute['fraction_pairs_improved']:.3f} |"
            )
    lines.extend(
        [
            "",
            "The relative fixed-set point estimate verifies a deterministic "
            "consequence of the own-tail endpoint; it is not independent evidence. "
            "The absolute endpoint, individual-pair fractions, uncertainty intervals, "
            "and downstream probe remain separate empirical quantities.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpt2", type=Path, default=DEFAULT_INPUTS["GPT-2"])
    parser.add_argument("--qwen", type=Path, default=DEFAULT_INPUTS["Qwen 2.5"])
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "exact_lower_tail_audit.json",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "RESULTS.md",
    )
    parser.add_argument("--tolerance", type=float, default=1e-12)
    args = parser.parse_args()

    payload = run_audit(
        {"GPT-2": args.gpt2, "Qwen 2.5": args.qwen},
        tolerance=args.tolerance,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(payload), encoding="utf-8")
    print(f"{payload['status']}: verified {sum(len(m['families']) for m in payload['models'].values())} cells")
    print(f"Saved JSON -> {args.output_json}")
    print(f"Saved Markdown -> {args.output_md}")


if __name__ == "__main__":
    main()
