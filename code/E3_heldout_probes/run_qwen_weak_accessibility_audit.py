#!/usr/bin/env python3
"""Offline, nested cross-fitted Qwen E3 weak-accessibility audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

C_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]
REPRESENTATIONS = {
    "standard": "sae_standard_code",
    "vreg": "sae_vreg_code",
}
EPSILON = 1e-8
OUTER_FOLDS = 5
INNER_FOLDS = 4
SEED = 20260806
N_BOOT = 5000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    path.write_text(text, encoding="utf-8")


def sanitize_public_metadata(value: Any) -> Any:
    """Recursively replace machine-local absolute paths in public artifacts."""
    if isinstance(value, dict):
        return {key: sanitize_public_metadata(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_public_metadata(item) for item in value]
    if not isinstance(value, str) or not value.startswith("/"):
        return value
    checkpoint_marker = (
        "/FINAL/tmlr_revision/prepare/experiment_101_hybrid_owt/checkpoints/"
    )
    if checkpoint_marker in value:
        return f"<CKPT_ROOT>/{value.split(checkpoint_marker, 1)[1]}"
    if "/SAE/" in value:
        return f"<REPO_ROOT>/{value.split('/SAE/', 1)[1]}"
    return "<ABSOLUTE_PATH_REDACTED>"


def dedup_key(example: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        example["split"],
        example["template_id"],
        example["text"],
        example["label"],
    )


def build_feature_indices(examples: list[dict[str, Any]], n_cached: int) -> np.ndarray:
    key_to_index: dict[tuple[str, str, str, str], int] = {}
    raw_to_cache = []
    for example in examples:
        key = dedup_key(example)
        if key not in key_to_index:
            key_to_index[key] = len(key_to_index)
        raw_to_cache.append(key_to_index[key])
    if len(key_to_index) != n_cached:
        raise ValueError(
            f"Cache/index mismatch: {len(key_to_index)} deduplicated examples, "
            f"{n_cached} cached rows"
        )
    return np.asarray(raw_to_cache, dtype=np.int64)


def label_vector(examples: list[dict[str, Any]]) -> np.ndarray:
    mapping = {"affirmed": 0, "negated": 1}
    try:
        return np.asarray([mapping[example["label"]] for example in examples], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"Unsupported label: {exc.args[0]}") from exc


def pair_map(examples: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    pairs: dict[str, dict[str, Any]] = {}
    for index, example in enumerate(examples):
        pair_id = example["global_pair_id"]
        entry = pairs.setdefault(
            pair_id,
            {
                "pair_id": pair_id,
                "template_id": example["template_id"],
                "family": example["family"],
            },
        )
        if entry["template_id"] != example["template_id"] or entry["family"] != example["family"]:
            raise ValueError(f"Inconsistent metadata within pair {pair_id}")
        side = example["side"]
        if side in entry:
            raise ValueError(f"Duplicate side {side} in pair {pair_id}")
        entry[side] = index
    for pair_id, entry in pairs.items():
        if "orig" not in entry or "pert" not in entry:
            raise ValueError(f"Incomplete pair {pair_id}")
    return dict(sorted(pairs.items()))


def grouped_folds(
    pair_entries: dict[str, dict[str, Any]],
    pair_ids: Iterable[str],
    n_splits: int,
    seed: int,
) -> dict[str, int]:
    """Greedily balance family-pair counts while never splitting templates."""
    by_template: dict[str, list[str]] = defaultdict(list)
    for pair_id in pair_ids:
        by_template[pair_entries[pair_id]["template_id"]].append(pair_id)
    if len(by_template) < n_splits:
        raise ValueError(f"Need at least {n_splits} template groups")

    family_totals = Counter(pair_entries[pid]["family"] for pid in pair_ids)
    targets = {family: count / n_splits for family, count in family_totals.items()}
    fold_family: list[Counter[str]] = [Counter() for _ in range(n_splits)]
    fold_pairs = [0] * n_splits
    fold_groups = [0] * n_splits
    assignment: dict[str, int] = {}

    ordered = sorted(
        by_template,
        key=lambda template: (
            -len(by_template[template]),
            stable_hash(f"{seed}:{template}"),
            template,
        ),
    )
    for template in ordered:
        ids = by_template[template]
        group_counts = Counter(pair_entries[pid]["family"] for pid in ids)

        def score(fold: int) -> tuple[float, int, int, int]:
            family_cost = sum(
                (fold_family[fold][family] + count - targets[family]) ** 2
                for family, count in group_counts.items()
            )
            return (family_cost, fold_pairs[fold], fold_groups[fold], fold)

        chosen = min(range(n_splits), key=score)
        assignment[template] = chosen
        fold_family[chosen].update(group_counts)
        fold_pairs[chosen] += len(ids)
        fold_groups[chosen] += 1
    if set(assignment) != set(by_template):
        raise AssertionError("Not every template was assigned")
    return assignment


def fit_model(x: np.ndarray, y: np.ndarray, c_value: float) -> tuple[StandardScaler, LogisticRegression]:
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    model = LogisticRegression(
        C=c_value,
        solver="lbfgs",
        max_iter=20000,
        class_weight="balanced",
        random_state=SEED,
    )
    model.fit(x_scaled, y)
    return scaler, model


def pair_correctness(
    probabilities: np.ndarray,
    pair_entries: dict[str, dict[str, Any]],
    pair_ids: Iterable[str],
) -> float:
    margins = [
        probabilities[pair_entries[pid]["pert"]] - probabilities[pair_entries[pid]["orig"]]
        for pid in pair_ids
    ]
    return float(np.mean(np.asarray(margins) > 0))


def nested_select_c(
    x: np.ndarray,
    y: np.ndarray,
    examples: list[dict[str, Any]],
    pair_entries: dict[str, dict[str, Any]],
    outer_train_pairs: list[str],
    outer_fold: int,
) -> tuple[float, list[dict[str, Any]], list[dict[str, Any]]]:
    inner_assignment = grouped_folds(
        pair_entries,
        outer_train_pairs,
        INNER_FOLDS,
        SEED + 1000 + outer_fold,
    )
    curves = []
    inner_checks = []
    for c_value in C_GRID:
        fold_scores = []
        for inner_fold in range(INNER_FOLDS):
            val_pairs = [
                pid
                for pid in outer_train_pairs
                if inner_assignment[pair_entries[pid]["template_id"]] == inner_fold
            ]
            train_pairs = [pid for pid in outer_train_pairs if pid not in set(val_pairs)]
            train_indices = np.asarray(
                [pair_entries[pid][side] for pid in train_pairs for side in ("orig", "pert")],
                dtype=np.int64,
            )
            val_indices = np.asarray(
                [pair_entries[pid][side] for pid in val_pairs for side in ("orig", "pert")],
                dtype=np.int64,
            )
            scaler, model = fit_model(x[train_indices], y[train_indices], c_value)
            probabilities = np.full(len(examples), np.nan, dtype=np.float64)
            probabilities[val_indices] = model.predict_proba(scaler.transform(x[val_indices]))[:, 1]
            fold_scores.append(pair_correctness(probabilities, pair_entries, val_pairs))
            if c_value == C_GRID[0]:
                train_templates = {pair_entries[pid]["template_id"] for pid in train_pairs}
                val_templates = {pair_entries[pid]["template_id"] for pid in val_pairs}
                inner_checks.append(
                    {
                        "inner_fold": inner_fold,
                        "n_train_pairs": len(train_pairs),
                        "n_validation_pairs": len(val_pairs),
                        "template_disjoint": train_templates.isdisjoint(val_templates),
                        "pair_disjoint": set(train_pairs).isdisjoint(val_pairs),
                        "scaler_fit_n_examples": int(len(train_indices)),
                    }
                )
        curves.append(
            {
                "C": c_value,
                "fold_pair_correctness": fold_scores,
                "mean_pair_correctness": float(np.mean(fold_scores)),
            }
        )
    selected = max(curves, key=lambda row: (row["mean_pair_correctness"], -C_GRID.index(row["C"])))
    return float(selected["C"]), curves, inner_checks


def cross_fit(
    x: np.ndarray,
    y: np.ndarray,
    examples: list[dict[str, Any]],
    pair_entries: dict[str, dict[str, Any]],
    outer_assignment: dict[str, int],
) -> dict[str, Any]:
    probabilities = np.full(len(examples), np.nan, dtype=np.float64)
    predictions = np.full(len(examples), -1, dtype=np.int64)
    example_fold = np.full(len(examples), -1, dtype=np.int64)
    prediction_count = np.zeros(len(examples), dtype=np.int64)
    fold_records = []
    all_pairs = list(pair_entries)
    for outer_fold in range(OUTER_FOLDS):
        test_pairs = [
            pid
            for pid in all_pairs
            if outer_assignment[pair_entries[pid]["template_id"]] == outer_fold
        ]
        train_pairs = [pid for pid in all_pairs if pid not in set(test_pairs)]
        train_indices = np.asarray(
            [pair_entries[pid][side] for pid in train_pairs for side in ("orig", "pert")],
            dtype=np.int64,
        )
        test_indices = np.asarray(
            [pair_entries[pid][side] for pid in test_pairs for side in ("orig", "pert")],
            dtype=np.int64,
        )
        selected_c, curve, inner_checks = nested_select_c(
            x, y, examples, pair_entries, train_pairs, outer_fold
        )
        scaler, model = fit_model(x[train_indices], y[train_indices], selected_c)
        probabilities[test_indices] = model.predict_proba(scaler.transform(x[test_indices]))[:, 1]
        predictions[test_indices] = model.predict(scaler.transform(x[test_indices]))
        example_fold[test_indices] = outer_fold
        prediction_count[test_indices] += 1
        train_templates = {pair_entries[pid]["template_id"] for pid in train_pairs}
        test_templates = {pair_entries[pid]["template_id"] for pid in test_pairs}
        fold_records.append(
            {
                "outer_fold": outer_fold,
                "selected_C": selected_c,
                "C_curve": curve,
                "n_train_pairs": len(train_pairs),
                "n_test_pairs": len(test_pairs),
                "train_templates": sorted(train_templates),
                "test_templates": sorted(test_templates),
                "template_disjoint": train_templates.isdisjoint(test_templates),
                "pair_disjoint": set(train_pairs).isdisjoint(test_pairs),
                "scaler_fit_n_examples": int(len(train_indices)),
                "inner_checks": inner_checks,
            }
        )
    if (
        np.any(~np.isfinite(probabilities))
        or np.any(predictions < 0)
        or np.any(example_fold < 0)
        or np.any(prediction_count != 1)
    ):
        raise AssertionError("OOF predictions are incomplete")
    records = []
    for index, example in enumerate(examples):
        records.append(
            {
                "example_id": example["example_id"],
                "pair_id": example["global_pair_id"],
                "template_id": example["template_id"],
                "family": example["family"],
                "side": example["side"],
                "y_true": int(y[index]),
                "y_pred": int(predictions[index]),
                "prob_positive": float(probabilities[index]),
                "outer_fold": int(example_fold[index]),
            }
        )
    return {
        "probabilities": probabilities,
        "example_fold": example_fold,
        "prediction_count": prediction_count,
        "prediction_records": records,
        "folds": fold_records,
    }


def displacement(x: np.ndarray, entry: dict[str, Any]) -> float:
    orig = x[entry["orig"]]
    pert = x[entry["pert"]]
    return float(np.linalg.norm(pert - orig) / (np.linalg.norm(orig) + EPSILON))


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    margins_std = np.asarray([row["standard_margin"] for row in records], dtype=np.float64)
    margins_vreg = np.asarray([row["vreg_margin"] for row in records], dtype=np.float64)
    correct_std = margins_std > 0
    correct_vreg = margins_vreg > 0
    return {
        "n": len(records),
        "standard": {
            "probability_margin": float(np.mean(margins_std)),
            "pair_correctness": float(np.mean(correct_std)),
            "pair_error": float(np.mean(~correct_std)),
            "n_correct": int(np.sum(correct_std)),
            "n_error": int(np.sum(~correct_std)),
        },
        "vreg": {
            "probability_margin": float(np.mean(margins_vreg)),
            "pair_correctness": float(np.mean(correct_vreg)),
            "pair_error": float(np.mean(~correct_vreg)),
            "n_correct": int(np.sum(correct_vreg)),
            "n_error": int(np.sum(~correct_vreg)),
        },
        "paired_delta": {
            "probability_margin": float(np.mean(margins_vreg - margins_std)),
            "pair_correctness": float(np.mean(correct_vreg.astype(float) - correct_std.astype(float))),
            "pair_error": float(np.mean((~correct_vreg).astype(float) - (~correct_std).astype(float))),
        },
        "mcnemar": mcnemar_cells(correct_std, correct_vreg),
    }


def exact_binomial_two_sided(standard_only: int, vreg_only: int) -> float:
    discordant = standard_only + vreg_only
    if discordant == 0:
        return 1.0
    k = min(standard_only, vreg_only)
    lower = sum(math.comb(discordant, j) for j in range(k + 1)) / (2**discordant)
    return float(min(1.0, 2.0 * lower))


def mcnemar_cells(correct_std: np.ndarray, correct_vreg: np.ndarray) -> dict[str, Any]:
    both_correct = int(np.sum(correct_std & correct_vreg))
    standard_only = int(np.sum(correct_std & ~correct_vreg))
    vreg_only = int(np.sum(~correct_std & correct_vreg))
    both_wrong = int(np.sum(~correct_std & ~correct_vreg))
    return {
        "both_correct": both_correct,
        "standard_only_correct": standard_only,
        "vreg_only_correct": vreg_only,
        "both_wrong": both_wrong,
        "discordant": standard_only + vreg_only,
        "exact_two_sided_p": exact_binomial_two_sided(standard_only, vreg_only),
    }


def weak_ids_pooled(records: list[dict[str, Any]], field: str = "D_standard") -> list[str]:
    ordered = sorted(records, key=lambda row: (row[field], row["pair_id"]))
    return [row["pair_id"] for row in ordered[: math.ceil(0.20 * len(ordered))]]


def weak_ids_within_family(records: list[dict[str, Any]]) -> list[str]:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_family[row["family"]].append(row)
    selected = []
    for family in sorted(by_family):
        ordered = sorted(by_family[family], key=lambda row: (row["D_standard"], row["pair_id"]))
        selected.extend(row["pair_id"] for row in ordered[: math.ceil(0.20 * len(ordered))])
    return sorted(selected)


def percentile_ci(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "ci_low": float(np.quantile(array, 0.025)),
        "ci_high": float(np.quantile(array, 0.975)),
    }


def sample_template_clusters(
    records: list[dict[str, Any]], rng: np.random.Generator
) -> list[dict[str, Any]]:
    by_template: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_template[row["template_id"]].append(row)
    templates = sorted(by_template)
    sampled = rng.choice(templates, size=len(templates), replace=True)
    return [row for template in sampled for row in by_template[str(template)]]


def bootstrap_fixed(
    all_records: list[dict[str, Any]],
    selected_ids: set[str],
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    selected = [row for row in all_records if row["pair_id"] in selected_ids]
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = defaultdict(list)
    for _ in range(n_boot):
        sample = sample_template_clusters(selected, rng)
        summary = summarize(sample)
        for method in ("standard", "vreg"):
            for metric in ("probability_margin", "pair_correctness", "pair_error"):
                draws[f"{method}_{metric}"].append(summary[method][metric])
        for metric in ("probability_margin", "pair_correctness", "pair_error"):
            draws[f"delta_{metric}"].append(summary["paired_delta"][metric])
    return {
        "n_boot": n_boot,
        "seed": seed,
        "resampling_unit": "template_id",
        "intervals": {key: percentile_ci(values) for key, values in sorted(draws.items())},
    }


def bootstrap_association(
    records: list[dict[str, Any]],
    weak_ids: set[str],
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = defaultdict(list)
    for _ in range(n_boot):
        sample = sample_template_clusters(records, rng)
        weak = [row for row in sample if row["pair_id"] in weak_ids]
        nonweak = [row for row in sample if row["pair_id"] not in weak_ids]
        if not weak or not nonweak:
            continue
        weak_summary = summarize(weak)["standard"]
        nonweak_summary = summarize(nonweak)["standard"]
        for metric in ("probability_margin", "pair_correctness", "pair_error"):
            draws[f"weak_minus_nonweak_{metric}"].append(
                weak_summary[metric] - nonweak_summary[metric]
            )
    return {
        "n_boot": n_boot,
        "seed": seed,
        "resampling_unit": "template_id",
        "intervals": {key: percentile_ci(values) for key, values in sorted(draws.items())},
    }


def selection_aware_bootstrap(
    records: list[dict[str, Any]], n_boot: int, seed: int
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = defaultdict(list)
    selected_sizes = []
    for _ in range(n_boot):
        sample = sample_template_clusters(records, rng)
        ordered = sorted(sample, key=lambda row: (row["D_standard"], row["pair_id"]))
        selected = ordered[: math.ceil(0.20 * len(ordered))]
        selected_sizes.append(len(selected))
        summary = summarize(selected)["paired_delta"]
        for metric in ("probability_margin", "pair_correctness", "pair_error"):
            draws[f"delta_{metric}"].append(summary[metric])
    return {
        "n_boot": n_boot,
        "seed": seed,
        "selection": "bottom_ceil_20_percent_D_standard_within_each_resample",
        "selected_n_range": [int(min(selected_sizes)), int(max(selected_sizes))],
        "intervals": {key: percentile_ci(values) for key, values in sorted(draws.items())},
    }


def fixed_subset_analysis(
    records: list[dict[str, Any]],
    ids: list[str],
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    id_set = set(ids)
    subset = [row for row in records if row["pair_id"] in id_set]
    point = summarize(subset)
    point["pair_ids"] = sorted(ids)
    point["template_count"] = len({row["template_id"] for row in subset})
    point["bootstrap"] = bootstrap_fixed(records, id_set, n_boot, seed)
    return point


def weak_nonweak_analysis(
    records: list[dict[str, Any]],
    weak_ids: list[str],
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    weak_set = set(weak_ids)
    weak = [row for row in records if row["pair_id"] in weak_set]
    nonweak = [row for row in records if row["pair_id"] not in weak_set]
    weak_standard = summarize(weak)["standard"]
    nonweak_standard = summarize(nonweak)["standard"]
    difference = {
        metric: weak_standard[metric] - nonweak_standard[metric]
        for metric in ("probability_margin", "pair_correctness", "pair_error")
    }
    return {
        "weak": weak_standard,
        "nonweak": nonweak_standard,
        "weak_minus_nonweak": difference,
        "bootstrap": bootstrap_association(records, weak_set, n_boot, seed),
    }


def quintile_analysis(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(records, key=lambda row: (row["D_standard"], row["pair_id"]))
    output = []
    for index, indices in enumerate(np.array_split(np.arange(len(ordered)), 5), start=1):
        subset = [ordered[int(i)] for i in indices]
        summary = summarize(subset)
        output.append(
            {
                "quintile": f"Q{index}",
                "n": len(subset),
                "D_standard_min": float(min(row["D_standard"] for row in subset)),
                "D_standard_max": float(max(row["D_standard"] for row in subset)),
                "standard": summary["standard"],
                "vreg": summary["vreg"],
                "paired_delta": summary["paired_delta"],
                "mcnemar": summary["mcnemar"],
                "pair_ids": [row["pair_id"] for row in subset],
            }
        )
    return output


def no_leakage_checks(
    examples: list[dict[str, Any]],
    pairs: dict[str, dict[str, Any]],
    outer_assignment: dict[str, int],
    fits: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    pair_folds = {
        pid: outer_assignment[entry["template_id"]]
        for pid, entry in pairs.items()
    }
    template_fold_unique = all(
        len({pair_folds[pid] for pid, entry in pairs.items() if entry["template_id"] == template}) == 1
        for template in outer_assignment
    )
    checks = {
        "pair_count_78": len(pairs) == 78,
        "each_pair_has_two_sides": all(
            "orig" in entry and "pert" in entry for entry in pairs.values()
        ),
        "template_assigned_once": len(outer_assignment) == len({e["template_id"] for e in examples}),
        "template_fold_unique": template_fold_unique,
        "standard_vreg_identical_outer_folds": all(
            fits["standard"]["example_fold"][i] == fits["vreg"]["example_fold"][i]
            for i in range(len(examples))
        ),
        "every_standard_example_oof_once": bool(np.all(fits["standard"]["example_fold"] >= 0)),
        "every_vreg_example_oof_once": bool(np.all(fits["vreg"]["example_fold"] >= 0)),
        "standard_prediction_count_exactly_one": bool(
            np.all(fits["standard"]["prediction_count"] == 1)
        ),
        "vreg_prediction_count_exactly_one": bool(
            np.all(fits["vreg"]["prediction_count"] == 1)
        ),
        "outer_train_test_disjoint": all(
            fold["template_disjoint"] and fold["pair_disjoint"]
            for method in fits.values()
            for fold in method["folds"]
        ),
        "inner_train_validation_disjoint": all(
            check["template_disjoint"] and check["pair_disjoint"]
            for method in fits.values()
            for fold in method["folds"]
            for check in fold["inner_checks"]
        ),
        "outer_labels_unused_for_scaling_or_selection": True,
    }
    return {"checks": checks, "all_pass": all(checks.values())}


def render_method(summary: dict[str, Any]) -> str:
    return (
        f"{summary['probability_margin']:.6f} | "
        f"{summary['pair_correctness']:.3f} ({summary['n_correct']}/{summary['n_correct'] + summary['n_error']}) | "
        f"{summary['pair_error']:.3f}"
    )


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Qwen E3 cross-fitted weak/non-weak semantic accessibility audit",
        "",
        f"Protocol date: `{payload['protocol_date']}`  ",
        f"Analysis rerun date: `{payload['analysis_run_date']}`  ",
        f"Pairs/templates: {payload['data']['n_pairs']}/{payload['data']['n_templates']}  ",
        f"Outer folds / bootstrap replicates: {OUTER_FOLDS}/{payload['bootstrap']['n_replicates']}  ",
        f"No-leakage checks: `{'PASS' if payload['no_leakage']['all_pass'] else 'FAIL'}`",
        "",
        "## Fixed Standard-defined subsets",
        "",
        "| Subset | n | Method | Mean probability margin | Pair correctness | Pair error |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for name in ("pooled_Wstd", "pooled_nonweak", "within_family_Wstd"):
        block = payload["subsets"][name]
        for method in ("standard", "vreg"):
            summary = block[method]
            lines.append(
                f"| {name} | {block['n']} | {method} | {summary['probability_margin']:.6f} | "
                f"{summary['pair_correctness']:.3f} ({summary['n_correct']}/{block['n']}) | "
                f"{summary['pair_error']:.3f} |"
            )
        ci = block["bootstrap"]["intervals"]["delta_probability_margin"]
        lines.append(
            f"| {name} | {block['n']} | V-reg−Standard | "
            f"{block['paired_delta']['probability_margin']:.6f} "
            f"[{ci['ci_low']:.6f}, {ci['ci_high']:.6f}] | "
            f"{block['paired_delta']['pair_correctness']:+.3f} | "
            f"{block['paired_delta']['pair_error']:+.3f} |"
        )
    lines.extend(
        [
            "",
            "## Paired correctness transitions on fixed weak sets",
            "",
            "| Subset | both correct | Standard only | V-reg only | both wrong | exact McNemar p |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name in ("pooled_Wstd", "within_family_Wstd"):
        cells = payload["subsets"][name]["mcnemar"]
        lines.append(
            f"| {name} | {cells['both_correct']} | {cells['standard_only_correct']} | "
            f"{cells['vreg_only_correct']} | {cells['both_wrong']} | "
            f"{cells['exact_two_sided_p']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Standard weak versus non-weak association",
            "",
            "| Definition | Group | n | Mean margin | Pair correctness | Pair error |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for name in ("pooled_Wstd", "within_family_Wstd"):
        association = payload["associations"][name]
        for group in ("weak", "nonweak"):
            row = association[group]
            lines.append(
                f"| {name} | {group} | {row['n_correct'] + row['n_error']} | "
                f"{row['probability_margin']:.6f} | {row['pair_correctness']:.3f} | "
                f"{row['pair_error']:.3f} |"
            )
        diff = association["weak_minus_nonweak"]
        ci = association["bootstrap"]["intervals"][
            "weak_minus_nonweak_probability_margin"
        ]
        lines.append(
            f"| {name} | weak−nonweak | — | {diff['probability_margin']:+.6f} "
            f"[{ci['ci_low']:.6f}, {ci['ci_high']:.6f}] | "
            f"{diff['pair_correctness']:+.3f} | {diff['pair_error']:+.3f} |"
        )
    lines.extend(
        [
            "",
            "## Standard displacement quintiles",
            "",
            "| Quintile | n | Std margin | V-reg margin | Delta | Std correct | V-reg correct |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["quintiles"]:
        lines.append(
            f"| {row['quintile']} | {row['n']} | {row['standard']['probability_margin']:.6f} | "
            f"{row['vreg']['probability_margin']:.6f} | "
            f"{row['paired_delta']['probability_margin']:+.6f} | "
            f"{row['standard']['pair_correctness']:.3f} | "
            f"{row['vreg']['pair_correctness']:.3f} |"
        )
    reverse = payload["reverse_Wvreg"]
    reverse_ci = reverse["bootstrap"]["intervals"]["delta_probability_margin"]
    lines.extend(
        [
            "",
            "## Reverse diagnostic",
            "",
            f"`W_vreg` n={reverse['n']}: Standard margin "
            f"{reverse['standard']['probability_margin']:.6f}, V-reg margin "
            f"{reverse['vreg']['probability_margin']:.6f}, delta "
            f"{reverse['paired_delta']['probability_margin']:+.6f} "
            f"[{reverse_ci['ci_low']:.6f}, {reverse_ci['ci_high']:.6f}].",
            "",
            "## Interpretation limits",
            "",
            "- Standard displacement and Standard margin share a representation source.",
            "- Separate representation-specific probes make this a representation-plus-readout comparison.",
            "- Probability margins also include calibration differences.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_analysis(
    split_path: Path,
    cache_path: Path,
    protocol_path: Path,
    n_boot: int = N_BOOT,
    seed: int = SEED,
    analysis_run_date: str | None = None,
) -> dict[str, Any]:
    split = read_json(split_path)
    examples = split["examples"]
    pairs = pair_map(examples)
    if len(examples) != 156 or len(pairs) != 78:
        raise ValueError(f"Expected 156 examples/78 pairs, got {len(examples)}/{len(pairs)}")
    with np.load(cache_path, allow_pickle=True) as cache:
        arrays_cached = {name: np.asarray(cache[key], dtype=np.float64) for name, key in REPRESENTATIONS.items()}
        metadata = sanitize_public_metadata(cache["metadata"].item())
    n_cached = next(iter(arrays_cached.values())).shape[0]
    if any(array.shape[0] != n_cached for array in arrays_cached.values()):
        raise ValueError("Representation cache row mismatch")
    raw_to_cache = build_feature_indices(examples, n_cached)
    arrays = {name: array[raw_to_cache] for name, array in arrays_cached.items()}
    y = label_vector(examples)
    outer_assignment = grouped_folds(pairs, pairs.keys(), OUTER_FOLDS, seed)
    fits = {
        name: cross_fit(array, y, examples, pairs, outer_assignment)
        for name, array in arrays.items()
    }
    rows = []
    for pair_id, entry in pairs.items():
        row = {
            "pair_id": pair_id,
            "template_id": entry["template_id"],
            "family": entry["family"],
            "outer_fold": int(outer_assignment[entry["template_id"]]),
            "orig_example_id": examples[entry["orig"]]["example_id"],
            "pert_example_id": examples[entry["pert"]]["example_id"],
        }
        for method in ("standard", "vreg"):
            row[f"D_{method}"] = displacement(arrays[method], entry)
            probs = fits[method]["probabilities"]
            row[f"{method}_prob_orig"] = float(probs[entry["orig"]])
            row[f"{method}_prob_pert"] = float(probs[entry["pert"]])
            row[f"{method}_margin"] = float(probs[entry["pert"]] - probs[entry["orig"]])
            row[f"{method}_correct"] = bool(row[f"{method}_margin"] > 0)
        rows.append(row)

    pooled_ids = weak_ids_pooled(rows)
    pooled_id_set = set(pooled_ids)
    pooled_nonweak_ids = sorted(
        row["pair_id"] for row in rows if row["pair_id"] not in pooled_id_set
    )
    family_ids = weak_ids_within_family(rows)
    reverse_ids = weak_ids_pooled(rows, "D_vreg")
    no_leakage = no_leakage_checks(examples, pairs, outer_assignment, fits)
    if not no_leakage["all_pass"]:
        raise AssertionError(f"No-leakage checks failed: {no_leakage}")

    payload: dict[str, Any] = {
        "analysis": "Qwen E3 cross-fitted weak/non-weak semantic accessibility audit",
        "status": "complete",
        "protocol_date": "2026-08-06",
        "analysis_run_date": analysis_run_date,
        "protocol_path": sanitize_public_metadata(str(protocol_path.resolve())),
        "protocol_sha256": sha256_file(protocol_path),
        "source": {
            "task_split": sanitize_public_metadata(str(split_path.resolve())),
            "task_split_sha256": sha256_file(split_path),
            "feature_cache": sanitize_public_metadata(str(cache_path.resolve())),
            "feature_cache_sha256": sha256_file(cache_path),
            "feature_cache_metadata": metadata,
            "offline_only": True,
        },
        "data": {
            "n_examples": len(examples),
            "n_cached_deduplicated_examples": n_cached,
            "n_pairs": len(pairs),
            "n_templates": len(outer_assignment),
            "family_pair_counts": dict(sorted(Counter(row["family"] for row in rows).items())),
        },
        "protocol": {
            "D_formula": "||z_pert-z_orig||_2/(||z_orig||_2+1e-8)",
            "outer_folds": OUTER_FOLDS,
            "inner_folds": INNER_FOLDS,
            "C_GRID": C_GRID,
            "C_selection": "mean inner-fold pair correctness; first C_GRID value wins ties",
            "scaling": "inner-train-only during selection; outer-train-only for final fold fit",
            "group": "template_id",
            "primary_weak_n": math.ceil(0.20 * len(rows)),
        },
        "bootstrap": {
            "n_replicates": n_boot,
            "seed": seed,
            "unit": "template_id",
            "interval": "percentile 95%",
        },
        "outer_template_folds": dict(sorted(outer_assignment.items())),
        "folds": {
            method: [
                {
                    **{key: value for key, value in fold.items() if key != "inner_checks"},
                    "inner_checks": fold["inner_checks"],
                }
                for fold in fit["folds"]
            ]
            for method, fit in fits.items()
        },
        "no_leakage": no_leakage,
        "oof_predictions": {
            method: fit["prediction_records"] for method, fit in fits.items()
        },
        "per_pair": rows,
        "subsets": {
            "pooled_Wstd": fixed_subset_analysis(rows, pooled_ids, n_boot, seed + 1),
            "pooled_nonweak": fixed_subset_analysis(
                rows, pooled_nonweak_ids, n_boot, seed + 7
            ),
            "within_family_Wstd": fixed_subset_analysis(rows, family_ids, n_boot, seed + 2),
        },
        "associations": {
            "pooled_Wstd": weak_nonweak_analysis(rows, pooled_ids, n_boot, seed + 3),
            "within_family_Wstd": weak_nonweak_analysis(rows, family_ids, n_boot, seed + 4),
        },
        "selection_aware_sensitivity": selection_aware_bootstrap(rows, n_boot, seed + 5),
        "quintiles": quintile_analysis(rows),
        "reverse_Wvreg": fixed_subset_analysis(rows, reverse_ids, n_boot, seed + 6),
        "limitations": [
            "Standard D and Standard OOF margin share a representation source.",
            "Separate representation-specific probes estimate representation plus readout.",
            "Probability-margin comparisons include calibration differences.",
        ],
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-split", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--analysis-run-date",
        required=True,
        help="Explicit ISO date for the generated result artifact (YYYY-MM-DD)",
    )
    args = parser.parse_args()
    payload = run_analysis(
        args.task_split,
        args.feature_cache,
        args.protocol,
        n_boot=args.n_boot,
        seed=args.seed,
        analysis_run_date=args.analysis_run_date,
    )
    write_json(args.output_json, payload)
    write_markdown(args.output_md, payload)
    print(f"Saved JSON -> {args.output_json}")
    print(f"Saved Markdown -> {args.output_md}")


if __name__ == "__main__":
    main()
