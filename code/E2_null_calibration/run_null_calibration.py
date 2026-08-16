#!/usr/bin/env python3
"""Aggregate E2 nuisance-calibrated detectability after nuisance distances exist.

This script does not evaluate language models or SAEs. It expects nuisance records that
already contain per-profile/per-representation distance metrics.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any


METRICS = {
    "abs_dz": "per_pair_abs_dz",
    "decode_resp": "per_pair_decode_resp",
    "s": "per_pair_s",
    "g": "per_pair_g",
}

PRIMARY_METRIC = "abs_dz"
SECONDARY_METRIC = "decode_resp"

ANCHOR_PRESERVING_TRANSFORMS = {
    "double_first_space",
    "sentence_initial_case_toggle",
}
PUNCTUATION_STRESS_TRANSFORMS = {
    "terminal_period_toggle",
}
TIER2_TRANSFORMS = {
    "prefix_in_note",
    "prefix_report_reads",
}

ENDPOINT_HIERARCHY = {
    "frozen_before_final_inference": True,
    "primary_analysis_group": "primary_anchor_preserving",
    "primary_metric": "abs_dz",
    "primary_endpoint": "delta_family_macro_auroc",
    "primary_comparison": "vreg_minus_standard",
    "supporting_endpoints": [
        "delta_pooled_auroc",
        "delta_source_matched_fraction",
        "delta_fixed_ratio_auprc",
        "delta_family_macro_target_coverage",
        "delta_family_macro_nuisance_fpr",
        "delta_shared_standard_threshold_target_coverage",
        "delta_shared_standard_threshold_nuisance_fpr",
    ],
    "secondary_representation_metric": "decode_resp",
    "diagnostic_metrics": [
        "s",
        "conditional_hfrac_residual_auroc",
        "punctuation_stress",
        "tier2_lexical_nuisance",
    ],
    "stress_test": "global_pooled_Q0.95_shared_standard_threshold_coverage",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def quantile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("Cannot compute quantile of an empty list")
    if not 0.0 <= q <= 1.0:
        raise ValueError("Quantile must be in [0, 1]")
    xs = sorted(values)
    pos = (len(xs) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(xs[lo])
    frac = pos - lo
    return float(xs[lo] * (1.0 - frac) + xs[hi] * frac)


def fraction_ge(values: list[float], threshold: float) -> float | None:
    if not values:
        return None
    return sum(value >= threshold for value in values) / len(values)


def finite_float(value: Any, path: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Non-numeric value at {path}: {value!r}") from exc
    if not math.isfinite(out):
        raise ValueError(f"Non-finite value at {path}: {value!r}")
    return out


def validate_nuisance_distances(payload: dict[str, Any]) -> None:
    records = payload.get("records", [])
    if not records:
        raise ValueError("Nuisance payload contains no records")
    missing = [record["nuisance_id"] for record in records if "distances" not in record]
    if missing:
        raise ValueError(
            "Nuisance records do not yet contain evaluated distance metrics. "
            "Evaluate the nuisance texts with the frozen Standard/V-reg checkpoints first. "
            f"Missing distances for {len(missing)} records; first={missing[0]}"
        )
    for record_index, record in enumerate(records):
        for profile, profile_block in record["distances"].items():
            for representation in ("standard", "vreg"):
                if representation not in profile_block:
                    raise ValueError(
                        f"Missing {representation} distances in record {record_index}/{profile}"
                    )
                for metric in METRICS:
                    if metric not in profile_block[representation]:
                        raise ValueError(
                            f"Missing {metric} in record {record_index}/{profile}/{representation}"
                        )
                    finite_float(
                        profile_block[representation][metric],
                        f"records[{record_index}].distances.{profile}.{representation}.{metric}",
                    )


def validate_target_payload(
    target_payload: dict[str, Any],
    template_payload: dict[str, Any],
) -> None:
    template_families = set(template_payload["families"])
    for profile, profile_block in target_payload["profiles"].items():
        profile_families = set(profile_block["families"])
        if profile_families != template_families:
            raise ValueError(
                f"Family mismatch for target profile {profile}: "
                f"{sorted(profile_families)} vs {sorted(template_families)}"
            )
        for family, family_block in profile_block["families"].items():
            selected = family_block["selected_pair_indices"]
            for representation in ("standard", "vreg"):
                if representation not in family_block:
                    raise ValueError(f"Missing {representation} block in {profile}/{family}")
                rep_block = family_block[representation]
                for metric, key in METRICS.items():
                    if key not in rep_block:
                        raise ValueError(f"Missing {key} in {profile}/{family}/{representation}")
                    values = rep_block[key]
                    if len(values) != len(selected):
                        raise ValueError(
                            f"Length mismatch for {profile}/{family}/{representation}/{key}: "
                            f"{len(values)} vs selected {len(selected)}"
                        )
                    for index, value in enumerate(values):
                        finite_float(value, f"{profile}.{family}.{representation}.{key}[{index}]")


def target_values_by_split(
    target_payload: dict[str, Any],
    split_by_family_template: dict[str, dict[str, str]],
    template_payload: dict[str, Any],
) -> dict[str, dict[str, dict[str, dict[str, list[float]]]]]:
    """Return profile -> representation -> metric -> split -> values."""
    out: dict[str, dict[str, dict[str, dict[str, list[float]]]]] = {}

    for profile, profile_block in target_payload["profiles"].items():
        out.setdefault(profile, {})
        for representation in ("standard", "vreg"):
            out[profile].setdefault(representation, {metric: {"calibration": [], "test": []} for metric in METRICS})
        for family, family_block in profile_block["families"].items():
            template_records = {
                int(pair["pair_index"]): pair
                for pair in template_payload["families"][family]["pairs"]
            }
            selected = [int(i) for i in family_block["selected_pair_indices"]]
            for position, pair_index in enumerate(selected):
                template_id = template_records[pair_index]["template_id"]
                split = split_by_family_template[family][template_id]
                for representation in ("standard", "vreg"):
                    rep_block = family_block[representation]
                    for metric, key in METRICS.items():
                        out[profile][representation][metric][split].append(
                            finite_float(rep_block[key][position], f"{profile}.{family}.{representation}.{key}[{position}]")
                        )

    return out


def target_records_by_metric(
    target_payloads: list[dict[str, Any]],
    split_by_family_template: dict[str, dict[str, str]],
    template_payload: dict[str, Any],
) -> dict[str, dict[str, dict[str, list[dict[str, Any]]]]]:
    """Return profile -> representation -> metric -> target records."""
    out: dict[str, dict[str, dict[str, list[dict[str, Any]]]]] = {}
    for target_payload in target_payloads:
        validate_target_payload(target_payload, template_payload)
        for profile, profile_block in target_payload["profiles"].items():
            out.setdefault(profile, {})
            for representation in ("standard", "vreg"):
                out[profile].setdefault(representation, {metric: [] for metric in METRICS})
            for family, family_block in profile_block["families"].items():
                template_records = {
                    int(pair["pair_index"]): pair
                    for pair in template_payload["families"][family]["pairs"]
                }
                selected = [int(i) for i in family_block["selected_pair_indices"]]
                for position, pair_index in enumerate(selected):
                    template_id = template_records[pair_index]["template_id"]
                    split = split_by_family_template[family][template_id]
                    for representation in ("standard", "vreg"):
                        rep_block = family_block[representation]
                        h_frac = finite_float(
                            rep_block["per_pair_h_frac"][position],
                            f"{profile}.{family}.{representation}.per_pair_h_frac[{position}]",
                        )
                        for metric, key in METRICS.items():
                            out[profile][representation][metric].append(
                                {
                                    "family": family,
                                    "pair_index": pair_index,
                                    "template_id": template_id,
                                    "split": split,
                                    "h_frac": h_frac,
                                    "value": finite_float(
                                        rep_block[key][position],
                                        f"{profile}.{family}.{representation}.{key}[{position}]",
                                    ),
                                }
                            )
    return out


def nuisance_records_by_metric(
    nuisance_payload: dict[str, Any],
) -> dict[str, dict[str, dict[str, dict[str, list[dict[str, Any]]]]]]:
    """Return profile -> representation -> metric -> analysis_group -> nuisance records."""
    out: dict[str, dict[str, dict[str, dict[str, list[dict[str, Any]]]]]] = {}
    for record in nuisance_payload["records"]:
        groups = analysis_groups_for_record(record)
        for profile, profile_block in record["distances"].items():
            for representation, rep_block in profile_block.items():
                for metric, value in rep_block.items():
                    numeric = finite_float(
                        value,
                        f"{record['nuisance_id']}.{profile}.{representation}.{metric}",
                    )
                    base = {
                        "nuisance_id": record["nuisance_id"],
                        "family": record["family"],
                        "pair_index": int(record["source_pair_index"]),
                        "template_id": record["template_id"],
                        "split": record["split"],
                        "source_side": record["source_side"],
                        "transform_type": record["transform_type"],
                        "h_frac": finite_float(
                            rep_block["h_frac"],
                            f"{record['nuisance_id']}.{profile}.{representation}.h_frac",
                        ),
                        "value": numeric,
                    }
                    for group in groups:
                        out.setdefault(profile, {}).setdefault(representation, {}).setdefault(metric, {}).setdefault(
                            group, []
                        ).append(base)
    return out


def nuisance_values_by_split(
    nuisance_payload: dict[str, Any],
) -> dict[str, dict[str, dict[str, dict[str, dict[str, list[float]]]]]]:
    """Return profile -> representation -> metric -> analysis_group -> split -> values."""
    out: dict[str, dict[str, dict[str, dict[str, dict[str, list[float]]]]]] = {}
    for record in nuisance_payload["records"]:
        split = record["split"]
        groups = analysis_groups_for_record(record)
        for profile, profile_block in record["distances"].items():
            for representation, rep_block in profile_block.items():
                for metric, value in rep_block.items():
                    numeric = finite_float(
                        value,
                        f"{record['nuisance_id']}.{profile}.{representation}.{metric}",
                    )
                    for group in groups:
                        out.setdefault(profile, {}).setdefault(representation, {}).setdefault(metric, {}).setdefault(
                            group, {"calibration": [], "test": []}
                        )[split].append(numeric)
    return out


def analysis_groups_for_record(record: dict[str, Any]) -> list[str]:
    transform_type = record["transform_type"]
    tier = record["transform_tier"]
    groups = [f"transform::{transform_type}"]
    if transform_type in ANCHOR_PRESERVING_TRANSFORMS:
        groups.append("primary_anchor_preserving")
        groups.append("tier1_punctuation_inclusive")
    elif transform_type in PUNCTUATION_STRESS_TRANSFORMS:
        groups.append("punctuation_stress")
        groups.append("tier1_punctuation_inclusive")
    elif transform_type in TIER2_TRANSFORMS:
        groups.append("tier2_lexical_nuisance")
    else:
        groups.append(tier)
    return groups


def validate_profile_coverage(
    nuisance: dict[str, Any],
    targets: dict[str, Any],
) -> None:
    nuisance_profiles = set(nuisance)
    target_profiles = set(targets)
    if nuisance_profiles != target_profiles:
        raise ValueError(
            f"Profile mismatch between nuisance and target data: "
            f"nuisance={sorted(nuisance_profiles)}, target={sorted(target_profiles)}"
        )
    for profile in sorted(target_profiles):
        for representation in ("standard", "vreg"):
            if representation not in nuisance[profile]:
                raise ValueError(f"Missing nuisance representation {profile}/{representation}")
            if representation not in targets[profile]:
                raise ValueError(f"Missing target representation {profile}/{representation}")
            for metric in METRICS:
                if metric not in nuisance[profile][representation]:
                    raise ValueError(f"Missing nuisance metric {profile}/{representation}/{metric}")
                if metric not in targets[profile][representation]:
                    raise ValueError(f"Missing target metric {profile}/{representation}/{metric}")


def calibrate(
    nuisance_payload: dict[str, Any],
    target_payloads: list[dict[str, Any]],
    template_payload: dict[str, Any],
    q: float,
    q_secondary: float,
) -> dict[str, Any]:
    validate_nuisance_distances(nuisance_payload)
    nuisance = nuisance_values_by_split(nuisance_payload)
    split_by_family_template = nuisance_payload["split_by_family_template"]

    targets: dict[str, dict[str, dict[str, dict[str, list[float]]]]] = {}
    for target_payload in target_payloads:
        validate_target_payload(target_payload, template_payload)
        block = target_values_by_split(target_payload, split_by_family_template, template_payload)
        for profile, profile_block in block.items():
            if profile in targets:
                raise ValueError(f"Duplicate target profile: {profile}")
            targets[profile] = profile_block
    validate_profile_coverage(nuisance, targets)

    results: dict[str, Any] = {}
    for profile, profile_block in sorted(targets.items()):
        results[profile] = {}
        for metric in METRICS:
            results[profile][metric] = {}
            groups = sorted(
                {
                    group
                    for representation in nuisance.get(profile, {})
                    for group in nuisance[profile][representation].get(metric, {})
                }
            )
            for group in groups:
                std_null_cal = nuisance[profile]["standard"][metric][group]["calibration"]
                tau_std = quantile(std_null_cal, q)
                tau_std_secondary = quantile(std_null_cal, q_secondary)
                tier_result: dict[str, Any] = {
                    "analysis_group": group,
                    "analysis_group_role": analysis_group_role(group),
                    "primary_shared_standard_threshold": {
                        "threshold_quantile": q,
                        "tau_standard": tau_std,
                        "tau_standard_secondary": tau_std_secondary,
                        "metric_role": (
                            "primary"
                            if metric == PRIMARY_METRIC
                            else "secondary"
                            if metric == SECONDARY_METRIC
                            else "diagnostic"
                        ),
                    },
                    "representations": {},
                }
                for representation in ("standard", "vreg"):
                    null_block = nuisance[profile][representation][metric][group]
                    target_block = profile_block[representation][metric]
                    tau_rep = quantile(null_block["calibration"], q)
                    tau_rep_secondary = quantile(null_block["calibration"], q_secondary)
                    tier_result["representations"][representation] = {
                        "shared_standard_threshold": {
                            "target_test_coverage": fraction_ge(target_block["test"], tau_std),
                            "nuisance_test_false_positive_rate": fraction_ge(null_block["test"], tau_std),
                            "target_test_coverage_secondary_q": fraction_ge(
                                target_block["test"], tau_std_secondary
                            ),
                            "nuisance_test_false_positive_rate_secondary_q": fraction_ge(
                                null_block["test"], tau_std_secondary
                            ),
                        },
                        "representation_specific_threshold": {
                            "tau": tau_rep,
                            "tau_secondary": tau_rep_secondary,
                            "target_test_coverage": fraction_ge(target_block["test"], tau_rep),
                            "nuisance_test_false_positive_rate": fraction_ge(null_block["test"], tau_rep),
                            "target_test_coverage_secondary_q": fraction_ge(
                                target_block["test"], tau_rep_secondary
                            ),
                            "nuisance_test_false_positive_rate_secondary_q": fraction_ge(
                                null_block["test"], tau_rep_secondary
                            ),
                        },
                        "counts": {
                            "nuisance_calibration": len(null_block["calibration"]),
                            "nuisance_test": len(null_block["test"]),
                            "target_calibration": len(target_block["calibration"]),
                            "target_test": len(target_block["test"]),
                        },
                    }
                results[profile][metric][group] = tier_result
                std_cov = tier_result["representations"]["standard"]["shared_standard_threshold"][
                    "target_test_coverage"
                ]
                vreg_cov = tier_result["representations"]["vreg"]["shared_standard_threshold"][
                    "target_test_coverage"
                ]
                if std_cov is not None and vreg_cov is not None:
                    tier_result["shared_standard_threshold_coverage_delta_vreg_minus_standard"] = (
                        vreg_cov - std_cov
                    )

    return results


def auroc(pos: list[float], neg: list[float]) -> float | None:
    if not pos or not neg:
        return None
    scored = [(value, 1) for value in pos] + [(value, 0) for value in neg]
    scored.sort(key=lambda item: item[0])
    rank_sum_pos = 0.0
    rank = 1
    i = 0
    while i < len(scored):
        j = i + 1
        while j < len(scored) and scored[j][0] == scored[i][0]:
            j += 1
        avg_rank = (rank + rank + (j - i) - 1) / 2.0
        rank_sum_pos += avg_rank * sum(label for _, label in scored[i:j])
        rank += j - i
        i = j
    n_pos = len(pos)
    n_neg = len(neg)
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def average_precision(pos: list[float], neg: list[float]) -> float | None:
    if not pos or not neg:
        return None
    scored = [(value, 1) for value in pos] + [(value, 0) for value in neg]
    scored.sort(key=lambda item: item[0], reverse=True)
    total_pos = sum(label for _, label in scored)
    if total_pos == 0:
        return None
    tp = 0
    precision_sum = 0.0
    for rank, (_, label) in enumerate(scored, start=1):
        if label:
            tp += 1
            precision_sum += tp / rank
    return precision_sum / total_pos


def record_sort_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        family_key(record),
        record["pair_index"],
        record.get("source_side", ""),
        record.get("transform_type", ""),
        record.get("nuisance_id", ""),
    )


def stratified_fixed_ratio_negatives(
    pos_records: list[dict[str, Any]],
    neg_records: list[dict[str, Any]],
    rng: random.Random | None = None,
) -> list[dict[str, Any]]:
    if not pos_records or not neg_records:
        return []
    n = min(len(pos_records), len(neg_records))
    pos_counts: dict[str, int] = {}
    neg_by_family: dict[str, list[dict[str, Any]]] = {}
    for record in pos_records:
        pos_counts[family_key(record)] = pos_counts.get(family_key(record), 0) + 1
    for record in neg_records:
        neg_by_family.setdefault(family_key(record), []).append(record)

    selected: list[dict[str, Any]] = []
    remaining: list[dict[str, Any]] = []
    for family in sorted(neg_by_family):
        records = sorted(neg_by_family[family], key=record_sort_key)
        take = min(pos_counts.get(family, 0), len(records))
        if rng is None:
            selected.extend(records[:take])
            remaining.extend(records[take:])
        else:
            sampled_indices = set(rng.sample(range(len(records)), take)) if take else set()
            selected.extend(records[index] for index in sorted(sampled_indices))
            remaining.extend(records[index] for index in range(len(records)) if index not in sampled_indices)

    if len(selected) < n:
        remaining_sorted = sorted(remaining, key=record_sort_key)
        need = n - len(selected)
        if rng is None:
            selected.extend(remaining_sorted[:need])
        else:
            selected.extend(rng.sample(remaining_sorted, min(need, len(remaining_sorted))))
    return selected[:n]


def average_precision_fixed_ratio(
    pos_records: list[dict[str, Any]],
    neg_records: list[dict[str, Any]],
    rng: random.Random | None = None,
) -> float | None:
    neg_sample = stratified_fixed_ratio_negatives(pos_records, neg_records, rng)
    if not pos_records or not neg_sample:
        return None
    n = min(len(pos_records), len(neg_sample))
    pos_balanced = sorted(pos_records, key=record_sort_key)[:n]
    return average_precision(
        [record["value"] for record in pos_balanced],
        [record["value"] for record in neg_sample],
    )


def macro_mean(values: list[float]) -> float | None:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def threshold_free_diagnostics(
    nuisance_payload: dict[str, Any],
    target_payloads: list[dict[str, Any]],
    template_payload: dict[str, Any],
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    split_by_family_template = nuisance_payload["split_by_family_template"]
    target_records = target_records_by_metric(target_payloads, split_by_family_template, template_payload)
    nuisance_records = nuisance_records_by_metric(nuisance_payload)
    diagnostics: dict[str, Any] = {}

    for profile, profile_block in sorted(target_records.items()):
        diagnostics[profile] = {}
        for metric in METRICS:
            diagnostics[profile][metric] = {}
            for group in sorted(nuisance_records[profile]["standard"][metric]):
                diagnostics[profile][metric][group] = {}
                for representation in ("standard", "vreg"):
                    targets = [
                        record for record in profile_block[representation][metric]
                        if record["split"] == "test"
                    ]
                    nuisances = [
                        record for record in nuisance_records[profile][representation][metric][group]
                        if record["split"] == "test"
                    ]
                    pooled_pos = [record["value"] for record in targets]
                    pooled_neg = [record["value"] for record in nuisances]
                    family_aurocs = []
                    family_aps = []
                    for family in sorted({record["family"] for record in targets}):
                        pos_f_records = [record for record in targets if record["family"] == family]
                        pos_f = [record["value"] for record in pos_f_records]
                        neg_f_records = [record for record in nuisances if record["family"] == family]
                        family_aurocs.append(auroc(pos_f, [record["value"] for record in neg_f_records]))
                        family_aps.append(average_precision_fixed_ratio(pos_f_records, neg_f_records))
                    diagnostics[profile][metric][group][representation] = {
                        "pooled_auroc": auroc(pooled_pos, pooled_neg),
                        "family_macro_auroc": macro_mean(family_aurocs),
                        "fixed_ratio_auprc": average_precision_fixed_ratio(targets, nuisances),
                        "family_macro_fixed_ratio_auprc": macro_mean(family_aps),
                        "source_matched": source_matched_summary(targets, nuisances),
                        "family_balanced_representation_specific_q_coverage": family_balanced_q_coverage(
                            targets,
                            nuisance_records[profile][representation][metric][group],
                        ),
                        "conditional_hfrac_residual": conditional_hfrac_summary(
                            targets,
                            nuisance_records[profile][representation][metric][group],
                        ),
                        "counts": {
                            "target_test": len(targets),
                            "nuisance_test": len(nuisances),
                        },
                    }
                std_auc = diagnostics[profile][metric][group]["standard"]["pooled_auroc"]
                vreg_auc = diagnostics[profile][metric][group]["vreg"]["pooled_auroc"]
                std_macro = diagnostics[profile][metric][group]["standard"]["family_macro_auroc"]
                vreg_macro = diagnostics[profile][metric][group]["vreg"]["family_macro_auroc"]
                std_ap = diagnostics[profile][metric][group]["standard"]["fixed_ratio_auprc"]
                vreg_ap = diagnostics[profile][metric][group]["vreg"]["fixed_ratio_auprc"]
                std_matched = diagnostics[profile][metric][group]["standard"]["source_matched"][
                    "target_gt_max_nuisance_fraction"
                ]
                vreg_matched = diagnostics[profile][metric][group]["vreg"]["source_matched"][
                    "target_gt_max_nuisance_fraction"
                ]
                std_family_cov = diagnostics[profile][metric][group]["standard"][
                    "family_balanced_representation_specific_q_coverage"
                ]["family_macro_target_coverage"]
                vreg_family_cov = diagnostics[profile][metric][group]["vreg"][
                    "family_balanced_representation_specific_q_coverage"
                ]["family_macro_target_coverage"]
                std_family_fpr = diagnostics[profile][metric][group]["standard"][
                    "family_balanced_representation_specific_q_coverage"
                ]["family_macro_nuisance_fpr"]
                vreg_family_fpr = diagnostics[profile][metric][group]["vreg"][
                    "family_balanced_representation_specific_q_coverage"
                ]["family_macro_nuisance_fpr"]
                std_resid_auc = diagnostics[profile][metric][group]["standard"]["conditional_hfrac_residual"][
                    "residual_pooled_auroc"
                ]
                vreg_resid_auc = diagnostics[profile][metric][group]["vreg"]["conditional_hfrac_residual"][
                    "residual_pooled_auroc"
                ]
                diagnostics[profile][metric][group]["delta_vreg_minus_standard"] = {
                    "pooled_auroc": None if std_auc is None or vreg_auc is None else vreg_auc - std_auc,
                    "family_macro_auroc": None if std_macro is None or vreg_macro is None else vreg_macro - std_macro,
                    "fixed_ratio_auprc": None if std_ap is None or vreg_ap is None else vreg_ap - std_ap,
                    "source_matched_target_gt_max_nuisance_fraction": (
                        None if std_matched is None or vreg_matched is None else vreg_matched - std_matched
                    ),
                    "family_balanced_q_target_coverage": (
                        None if std_family_cov is None or vreg_family_cov is None else vreg_family_cov - std_family_cov
                    ),
                    "family_balanced_q_nuisance_fpr": (
                        None if std_family_fpr is None or vreg_family_fpr is None else vreg_family_fpr - std_family_fpr
                    ),
                    "conditional_hfrac_residual_pooled_auroc": (
                        None if std_resid_auc is None or vreg_resid_auc is None else vreg_resid_auc - std_resid_auc
                    ),
                }
                if group == "primary_anchor_preserving" and metric in (PRIMARY_METRIC, "s", SECONDARY_METRIC):
                    diagnostics[profile][metric][group]["cluster_bootstrap_delta_ci"] = bootstrap_delta_ci(
                        std_targets=profile_block["standard"][metric],
                        std_nuisances=nuisance_records[profile]["standard"][metric][group],
                        vreg_targets=profile_block["vreg"][metric],
                        vreg_nuisances=nuisance_records[profile]["vreg"][metric][group],
                        n_boot=n_boot,
                        seed=seed + stable_seed_offset(profile, metric, group),
                    )
    return diagnostics


def stable_seed_offset(*parts: str) -> int:
    text = "::".join(parts)
    return sum((index + 1) * ord(char) for index, char in enumerate(text))


def family_key(record: dict[str, Any]) -> str:
    return str(record.get("_bootstrap_family", record["family"]))


def cluster_key(record: dict[str, Any]) -> str:
    return f"{record['family']}::{record['template_id']}"


def cluster_map(records: list[dict[str, Any]]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    out: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for record in records:
        out.setdefault(record["split"], {}).setdefault(cluster_key(record), []).append(record)
    return out


def split_cluster_keys_by_family(records: list[dict[str, Any]]) -> dict[str, dict[str, list[str]]]:
    out: dict[str, dict[str, set[str]]] = {}
    for record in records:
        out.setdefault(record["family"], {}).setdefault(record["split"], set()).add(cluster_key(record))
    return {
        family: {split: sorted(keys) for split, keys in split_block.items()}
        for family, split_block in out.items()
    }


def merge_family_split_keys(*record_sets: list[dict[str, Any]]) -> dict[str, dict[str, list[str]]]:
    merged: dict[str, dict[str, set[str]]] = {}
    for records in record_sets:
        for family, split_block in split_cluster_keys_by_family(records).items():
            for split, keys in split_block.items():
                merged.setdefault(family, {}).setdefault(split, set()).update(keys)
    return {
        family: {split: sorted(keys) for split, keys in split_block.items()}
        for family, split_block in merged.items()
    }


def expand_sampled_clusters(
    mapping: dict[str, dict[str, list[dict[str, Any]]]],
    sampled: list[dict[str, str]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in sampled:
        for record in mapping.get(item["split"], {}).get(item["cluster_key"], []):
            copied = dict(record)
            copied["_bootstrap_family"] = item["bootstrap_family"]
            out.append(copied)
    return out


def ci95(values: list[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    xs = sorted(values)
    return {
        "mean": sum(xs) / len(xs),
        "q025": quantile(xs, 0.025),
        "q500": quantile(xs, 0.5),
        "q975": quantile(xs, 0.975),
        "n_boot_effective": len(xs),
    }


def bootstrap_delta_ci(
    std_targets: list[dict[str, Any]],
    std_nuisances: list[dict[str, Any]],
    vreg_targets: list[dict[str, Any]],
    vreg_nuisances: list[dict[str, Any]],
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    if n_boot <= 0:
        return {"status": "not_requested", "n_boot": n_boot}
    std_target_map = cluster_map(std_targets)
    std_nuisance_map = cluster_map(std_nuisances)
    vreg_target_map = cluster_map(vreg_targets)
    vreg_nuisance_map = cluster_map(vreg_nuisances)
    family_split_keys = merge_family_split_keys(
        std_targets,
        std_nuisances,
        vreg_targets,
        vreg_nuisances,
    )
    if not family_split_keys:
        return {"status": "insufficient_data", "n_boot": n_boot}

    return {
        "fixed_family": run_bootstrap_mode(
            mode="fixed_family",
            std_target_map=std_target_map,
            std_nuisance_map=std_nuisance_map,
            vreg_target_map=vreg_target_map,
            vreg_nuisance_map=vreg_nuisance_map,
            family_split_keys=family_split_keys,
            n_boot=n_boot,
            seed=seed,
        ),
        "family_resampled": run_bootstrap_mode(
            mode="family_resampled",
            std_target_map=std_target_map,
            std_nuisance_map=std_nuisance_map,
            vreg_target_map=vreg_target_map,
            vreg_nuisance_map=vreg_nuisance_map,
            family_split_keys=family_split_keys,
            n_boot=n_boot,
            seed=seed + 1_000_003,
        ),
    }


def empty_delta_accumulator() -> dict[str, list[float]]:
    return {
        "delta_pooled_auroc": [],
        "delta_family_macro_auroc": [],
        "delta_fixed_ratio_auprc": [],
        "delta_source_matched_fraction": [],
        "delta_family_macro_target_coverage": [],
        "delta_family_macro_nuisance_fpr": [],
        "delta_conditional_hfrac_residual_auroc": [],
        "delta_shared_standard_threshold_target_coverage": [],
        "delta_shared_standard_threshold_nuisance_fpr": [],
    }


def run_bootstrap_mode(
    mode: str,
    std_target_map: dict[str, dict[str, list[dict[str, Any]]]],
    std_nuisance_map: dict[str, dict[str, list[dict[str, Any]]]],
    vreg_target_map: dict[str, dict[str, list[dict[str, Any]]]],
    vreg_nuisance_map: dict[str, dict[str, list[dict[str, Any]]]],
    family_split_keys: dict[str, dict[str, list[str]]],
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    families = sorted(family_split_keys)
    deltas = empty_delta_accumulator()
    diagnostics = {
        "n_families_total": len(families),
        "n_families_per_replicate_min": None,
        "n_families_per_replicate_max": None,
        "n_family_draws_min": None,
        "n_family_draws_max": None,
        "n_unique_families_min": None,
        "n_unique_families_max": None,
        "calibration_clusters_per_replicate_min": None,
        "test_clusters_per_replicate_min": None,
        "n_replicates_with_all_fixed_families": 0,
        "shared_resampling_for_standard_and_vreg": True,
        "shared_auprc_rng_for_standard_and_vreg": True,
        "tau_recalibrated_each_replicate": True,
    }
    for _ in range(n_boot):
        sampled = sample_bootstrap_clusters(mode, family_split_keys, rng)
        n_family_draws = len({item["bootstrap_family"] for item in sampled})
        n_unique_families = len({item["family"] for item in sampled})
        n_cal = sum(item["split"] == "calibration" for item in sampled)
        n_test = sum(item["split"] == "test" for item in sampled)
        diagnostics["n_families_per_replicate_min"] = min_or_value(
            diagnostics["n_families_per_replicate_min"], n_family_draws
        )
        diagnostics["n_families_per_replicate_max"] = max_or_value(
            diagnostics["n_families_per_replicate_max"], n_family_draws
        )
        diagnostics["n_family_draws_min"] = min_or_value(
            diagnostics["n_family_draws_min"], n_family_draws
        )
        diagnostics["n_family_draws_max"] = max_or_value(
            diagnostics["n_family_draws_max"], n_family_draws
        )
        diagnostics["n_unique_families_min"] = min_or_value(
            diagnostics["n_unique_families_min"], n_unique_families
        )
        diagnostics["n_unique_families_max"] = max_or_value(
            diagnostics["n_unique_families_max"], n_unique_families
        )
        diagnostics["calibration_clusters_per_replicate_min"] = min_or_value(
            diagnostics["calibration_clusters_per_replicate_min"], n_cal
        )
        diagnostics["test_clusters_per_replicate_min"] = min_or_value(
            diagnostics["test_clusters_per_replicate_min"], n_test
        )
        if mode == "fixed_family" and n_unique_families == len(families):
            diagnostics["n_replicates_with_all_fixed_families"] += 1

        std_t = expand_sampled_clusters(std_target_map, sampled)
        std_n = expand_sampled_clusters(std_nuisance_map, sampled)
        vreg_t = expand_sampled_clusters(vreg_target_map, sampled)
        vreg_n = expand_sampled_clusters(vreg_nuisance_map, sampled)
        ap_seed = rng.randrange(2**31)
        std_stats = diagnostic_scalar_block(std_t, std_n, random.Random(ap_seed))
        vreg_stats = diagnostic_scalar_block(vreg_t, vreg_n, random.Random(ap_seed))
        shared_stats = shared_standard_threshold_scalar_block(std_t, std_n, vreg_t, vreg_n)
        scalar_names = {
            "pooled_auroc": "delta_pooled_auroc",
            "family_macro_auroc": "delta_family_macro_auroc",
            "fixed_ratio_auprc": "delta_fixed_ratio_auprc",
            "source_matched_target_gt_max_nuisance_fraction": "delta_source_matched_fraction",
            "family_balanced_q_target_coverage": "delta_family_macro_target_coverage",
            "family_balanced_q_nuisance_fpr": "delta_family_macro_nuisance_fpr",
            "conditional_hfrac_residual_pooled_auroc": "delta_conditional_hfrac_residual_auroc",
        }
        for stat_name, delta_name in scalar_names.items():
            if std_stats[stat_name] is not None and vreg_stats[stat_name] is not None:
                deltas[delta_name].append(vreg_stats[stat_name] - std_stats[stat_name])
        for stat_name, delta_name in {
            "target_coverage_delta": "delta_shared_standard_threshold_target_coverage",
            "nuisance_fpr_delta": "delta_shared_standard_threshold_nuisance_fpr",
        }.items():
            if shared_stats[stat_name] is not None:
                deltas[delta_name].append(shared_stats[stat_name])
    ci_block = {
        name: ci95(values)
        for name, values in deltas.items()
    }
    if ci_block.get("delta_shared_standard_threshold_target_coverage") is not None:
        ci_block["shared_threshold_coverage_delta"] = ci_block[
            "delta_shared_standard_threshold_target_coverage"
        ]
    if ci_block.get("delta_shared_standard_threshold_nuisance_fpr") is not None:
        ci_block["shared_threshold_fpr_delta"] = ci_block[
            "delta_shared_standard_threshold_nuisance_fpr"
        ]
    return {
        "status": f"computed_{mode}_template_cluster_bootstrap",
        "n_boot_requested": n_boot,
        "seed": seed,
        "resampling_unit": "family_template_id",
        "resampling_mode": mode,
        "split_resampling": "calibration_and_test_clusters_sampled_separately",
        "standard_vreg_pairing": "same_family_split_cluster_draws_and_same_auprc_rng",
        "tau_recalibration": "thresholds recalibrated inside every bootstrap replicate",
        "diagnostics": diagnostics,
        "delta_vreg_minus_standard_ci95": ci_block,
    }


def min_or_value(current: int | None, value: int) -> int:
    return value if current is None else min(current, value)


def max_or_value(current: int | None, value: int) -> int:
    return value if current is None else max(current, value)


def sample_bootstrap_clusters(
    mode: str,
    family_split_keys: dict[str, dict[str, list[str]]],
    rng: random.Random,
) -> list[dict[str, str]]:
    families = sorted(family_split_keys)
    family_draws = families if mode == "fixed_family" else [rng.choice(families) for _ in families]
    sampled: list[dict[str, str]] = []
    for draw_index, family in enumerate(family_draws):
        bootstrap_family = family if mode == "fixed_family" else f"{family}::draw_{draw_index:02d}"
        for split in ("calibration", "test"):
            keys = family_split_keys.get(family, {}).get(split, [])
            if not keys:
                continue
            for _ in keys:
                sampled.append(
                    {
                        "family": family,
                        "bootstrap_family": bootstrap_family,
                        "split": split,
                        "cluster_key": rng.choice(keys),
                    }
                )
    return sampled


def diagnostic_scalar_block(
    targets_all_splits: list[dict[str, Any]],
    nuisances_all_splits: list[dict[str, Any]],
    rng: random.Random | None = None,
) -> dict[str, float | None]:
    targets = [record for record in targets_all_splits if record["split"] == "test"]
    nuisances = [record for record in nuisances_all_splits if record["split"] == "test"]
    pos = [record["value"] for record in targets]
    neg = [record["value"] for record in nuisances]
    family_aurocs = []
    for family in sorted({family_key(record) for record in targets}):
        pos_f = [record["value"] for record in targets if family_key(record) == family]
        neg_f = [record["value"] for record in nuisances if family_key(record) == family]
        family_aurocs.append(auroc(pos_f, neg_f))
    source_matched = source_matched_summary(targets, nuisances)
    family_coverage = family_balanced_q_coverage(targets, nuisances_all_splits)
    conditional = conditional_hfrac_summary(targets, nuisances_all_splits)
    return {
        "pooled_auroc": auroc(pos, neg),
        "family_macro_auroc": macro_mean(family_aurocs),
        "fixed_ratio_auprc": average_precision_fixed_ratio(targets, nuisances, rng),
        "source_matched_target_gt_max_nuisance_fraction": source_matched[
            "target_gt_max_nuisance_fraction"
        ],
        "family_balanced_q_target_coverage": family_coverage["family_macro_target_coverage"],
        "family_balanced_q_nuisance_fpr": family_coverage["family_macro_nuisance_fpr"],
        "conditional_hfrac_residual_pooled_auroc": conditional.get("residual_pooled_auroc"),
    }


def shared_standard_threshold_scalar_block(
    std_targets_all_splits: list[dict[str, Any]],
    std_nuisances_all_splits: list[dict[str, Any]],
    vreg_targets_all_splits: list[dict[str, Any]],
    vreg_nuisances_all_splits: list[dict[str, Any]],
    q: float = 0.95,
) -> dict[str, float | None]:
    per_family = []
    for family in sorted({family_key(record) for record in std_targets_all_splits}):
        std_null_cal = [
            record["value"]
            for record in std_nuisances_all_splits
            if family_key(record) == family and record["split"] == "calibration"
        ]
        if not std_null_cal:
            continue
        tau = quantile(std_null_cal, q)
        std_target_test = [
            record["value"]
            for record in std_targets_all_splits
            if family_key(record) == family and record["split"] == "test"
        ]
        vreg_target_test = [
            record["value"]
            for record in vreg_targets_all_splits
            if family_key(record) == family and record["split"] == "test"
        ]
        std_nuisance_test = [
            record["value"]
            for record in std_nuisances_all_splits
            if family_key(record) == family and record["split"] == "test"
        ]
        vreg_nuisance_test = [
            record["value"]
            for record in vreg_nuisances_all_splits
            if family_key(record) == family and record["split"] == "test"
        ]
        per_family.append(
            {
                "std_target_coverage": fraction_ge(std_target_test, tau),
                "vreg_target_coverage": fraction_ge(vreg_target_test, tau),
                "std_nuisance_fpr": fraction_ge(std_nuisance_test, tau),
                "vreg_nuisance_fpr": fraction_ge(vreg_nuisance_test, tau),
            }
        )
    target_deltas = [
        block["vreg_target_coverage"] - block["std_target_coverage"]
        for block in per_family
        if block["std_target_coverage"] is not None and block["vreg_target_coverage"] is not None
    ]
    fpr_deltas = [
        block["vreg_nuisance_fpr"] - block["std_nuisance_fpr"]
        for block in per_family
        if block["std_nuisance_fpr"] is not None and block["vreg_nuisance_fpr"] is not None
    ]
    return {
        "target_coverage_delta": macro_mean(target_deltas),
        "nuisance_fpr_delta": macro_mean(fpr_deltas),
    }


def family_balanced_q_coverage(
    targets: list[dict[str, Any]],
    nuisances_all_splits: list[dict[str, Any]],
    q: float = 0.95,
) -> dict[str, Any]:
    per_family = {}
    for family in sorted({family_key(record) for record in targets}):
        null_cal = [
            record["value"]
            for record in nuisances_all_splits
            if family_key(record) == family and record["split"] == "calibration"
        ]
        target_test = [
            record["value"]
            for record in targets
            if family_key(record) == family and record["split"] == "test"
        ]
        nuisance_test = [
            record["value"]
            for record in nuisances_all_splits
            if family_key(record) == family and record["split"] == "test"
        ]
        if not null_cal or not target_test:
            continue
        tau = quantile(null_cal, q)
        per_family[family] = {
            "tau": tau,
            "target_test_coverage": fraction_ge(target_test, tau),
            "nuisance_test_false_positive_rate": fraction_ge(nuisance_test, tau),
            "nuisance_calibration": len(null_cal),
            "target_test": len(target_test),
            "nuisance_test": len(nuisance_test),
        }
    coverages = [
        block["target_test_coverage"]
        for block in per_family.values()
        if block["target_test_coverage"] is not None
    ]
    fprs = [
        block["nuisance_test_false_positive_rate"]
        for block in per_family.values()
        if block["nuisance_test_false_positive_rate"] is not None
    ]
    return {
        "threshold_quantile": q,
        "family_macro_target_coverage": macro_mean(coverages),
        "family_macro_nuisance_fpr": macro_mean(fprs),
        "n_families": len(per_family),
        "per_family": per_family,
    }


def conditional_hfrac_summary(
    targets: list[dict[str, Any]],
    nuisances_all_splits: list[dict[str, Any]],
) -> dict[str, Any]:
    calibration = [
        record for record in nuisances_all_splits
        if record["split"] == "calibration"
    ]
    nuisance_test = [
        record for record in nuisances_all_splits
        if record["split"] == "test"
    ]
    if len(calibration) < 2 or not targets or not nuisance_test:
        return {"status": "insufficient_data"}
    x_mean = sum(record["h_frac"] for record in calibration) / len(calibration)
    y_mean = sum(record["value"] for record in calibration) / len(calibration)
    denom = sum((record["h_frac"] - x_mean) ** 2 for record in calibration)
    beta = 0.0 if denom == 0.0 else sum(
        (record["h_frac"] - x_mean) * (record["value"] - y_mean)
        for record in calibration
    ) / denom
    alpha = y_mean - beta * x_mean

    def residual(record: dict[str, Any]) -> float:
        return record["value"] - (alpha + beta * record["h_frac"])

    target_residuals = [residual(record) for record in targets]
    nuisance_residuals = [residual(record) for record in nuisance_test]
    family_residual_aurocs = []
    for family in sorted({family_key(record) for record in targets}):
        pos_f = [residual(record) for record in targets if family_key(record) == family]
        neg_f = [residual(record) for record in nuisance_test if family_key(record) == family]
        family_residual_aurocs.append(auroc(pos_f, neg_f))
    return {
        "calibration_model": "ols_value_on_h_frac_fit_on_nuisance_calibration",
        "alpha": alpha,
        "beta": beta,
        "residual_pooled_auroc": auroc(target_residuals, nuisance_residuals),
        "residual_family_macro_auroc": macro_mean(family_residual_aurocs),
        "target_residual_median": quantile(target_residuals, 0.5),
        "nuisance_residual_median": quantile(nuisance_residuals, 0.5),
    }


def source_matched_summary(
    targets: list[dict[str, Any]],
    nuisances: list[dict[str, Any]],
) -> dict[str, Any]:
    nuisance_by_pair: dict[tuple[str, int], list[float]] = {}
    for record in nuisances:
        nuisance_by_pair.setdefault((family_key(record), record["pair_index"]), []).append(record["value"])
    comparisons = []
    ratios = []
    margins = []
    for target in targets:
        vals = nuisance_by_pair.get((family_key(target), target["pair_index"]), [])
        if not vals:
            continue
        max_nuisance = max(vals)
        comparisons.append(target["value"] > max_nuisance)
        ratios.append(target["value"] / (max_nuisance + 1e-12))
        margins.append(target["value"] - max_nuisance)
    if not comparisons:
        return {"n_pairs": 0, "target_gt_max_nuisance_fraction": None}
    ratios_sorted = sorted(ratios)
    return {
        "n_pairs": len(comparisons),
        "target_gt_max_nuisance_fraction": sum(comparisons) / len(comparisons),
        "median_target_over_max_nuisance": ratios_sorted[len(ratios_sorted) // 2],
        "mean_target_minus_max_nuisance": sum(margins) / len(margins),
    }


def analysis_group_role(group: str) -> str:
    if group == "primary_anchor_preserving":
        return "primary"
    if group == "punctuation_stress":
        return "stress_test"
    if group == "tier1_punctuation_inclusive":
        return "conservative_punctuation_inclusive_supplement"
    if group == "tier2_lexical_nuisance":
        return "secondary_lexical_nuisance"
    if group.startswith("transform::"):
        return "diagnostic_transform_specific"
    return "diagnostic"


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# E2 Null-Calibrated Detectability",
        "",
        "This report aggregates pre-evaluated meaning-preserving nuisance controls.",
        "",
        "The shared Standard-threshold Q0.95 table is a high-specificity operating-point stress test,",
        "not the sole E2 endpoint.",
        "",
        "| Profile | Metric | Analysis group | Role | Shared tau(std) | Std target coverage | V-reg target coverage | V-reg null FPR |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for profile, profile_block in sorted(payload["profiles"].items()):
        for metric, metric_block in sorted(profile_block.items()):
            for tier, tier_block in sorted(metric_block.items()):
                shared = tier_block["primary_shared_standard_threshold"]["tau_standard"]
                std = tier_block["representations"]["standard"]["shared_standard_threshold"]
                vreg = tier_block["representations"]["vreg"]["shared_standard_threshold"]
                lines.append(
                    f"| {profile} | {metric} | {tier} | {tier_block['analysis_group_role']} | "
                    f"{shared:.6f} | "
                    f"{std['target_test_coverage']:.6f} | "
                    f"{vreg['target_test_coverage']:.6f} | "
                    f"{vreg['nuisance_test_false_positive_rate']:.6f} |"
                )
    lines.extend(
        [
            "",
            "## Threshold-Free Anchor-Preserving Diagnostics",
            "",
            "| Profile | Metric | Std pooled AUROC | V-reg pooled AUROC | Delta | Std family-macro AUROC | V-reg family-macro AUROC | Std source-matched | V-reg source-matched |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for profile, profile_block in sorted(payload["threshold_free_diagnostics"].items()):
        for metric in ("abs_dz", "s", "decode_resp"):
            block = profile_block[metric]["primary_anchor_preserving"]
            std = block["standard"]
            vreg = block["vreg"]
            delta = block["delta_vreg_minus_standard"]["pooled_auroc"]
            lines.append(
                f"| {profile} | {metric} | "
                f"{std['pooled_auroc']:.6f} | "
                f"{vreg['pooled_auroc']:.6f} | "
                f"{delta:.6f} | "
                f"{std['family_macro_auroc']:.6f} | "
                f"{vreg['family_macro_auroc']:.6f} | "
                f"{std['source_matched']['target_gt_max_nuisance_fraction']:.6f} | "
                f"{vreg['source_matched']['target_gt_max_nuisance_fraction']:.6f} |"
            )
    if payload.get("n_boot", 0) > 0:
        lines.extend(
            [
                "",
                "## Primary Bootstrap Delta CIs",
                "",
                "| Profile | Metric | Endpoint | Point delta | Fixed-family 95% CI | Family-resampled 95% CI |",
                "|---|---|---|---:|---:|---:|",
            ]
        )
        endpoint_map = {
            "delta_family_macro_auroc": "family_macro_auroc",
            "delta_pooled_auroc": "pooled_auroc",
            "delta_source_matched_fraction": "source_matched_target_gt_max_nuisance_fraction",
            "delta_fixed_ratio_auprc": "fixed_ratio_auprc",
            "delta_family_macro_target_coverage": "family_balanced_q_target_coverage",
            "delta_family_macro_nuisance_fpr": "family_balanced_q_nuisance_fpr",
            "delta_conditional_hfrac_residual_auroc": "conditional_hfrac_residual_pooled_auroc",
        }
        for profile, profile_block in sorted(payload["threshold_free_diagnostics"].items()):
            for metric in ("abs_dz", "decode_resp"):
                block = profile_block[metric]["primary_anchor_preserving"]
                boot = block.get("cluster_bootstrap_delta_ci", {})
                for endpoint, point_key in endpoint_map.items():
                    point = block["delta_vreg_minus_standard"].get(point_key)
                    fixed = boot.get("fixed_family", {}).get("delta_vreg_minus_standard_ci95", {}).get(endpoint)
                    family = boot.get("family_resampled", {}).get("delta_vreg_minus_standard_ci95", {}).get(endpoint)
                    lines.append(
                        f"| {profile} | {metric} | {endpoint} | "
                        f"{format_optional(point)} | "
                        f"{format_ci(fixed)} | "
                        f"{format_ci(family)} |"
                    )
                for endpoint in (
                    "delta_shared_standard_threshold_target_coverage",
                    "delta_shared_standard_threshold_nuisance_fpr",
                ):
                    fixed = boot.get("fixed_family", {}).get("delta_vreg_minus_standard_ci95", {}).get(endpoint)
                    family = boot.get("family_resampled", {}).get("delta_vreg_minus_standard_ci95", {}).get(endpoint)
                    lines.append(
                        f"| {profile} | {metric} | {endpoint} | n/a | "
                        f"{format_ci(fixed)} | {format_ci(family)} |"
                    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_optional(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6f}"


def format_ci(ci: dict[str, Any] | None) -> str:
    if not ci:
        return "n/a"
    return f"[{ci['q025']:.6f}, {ci['q975']:.6f}]"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate null-calibrated detectability from evaluated nuisance records"
    )
    parser.add_argument("--nuisance-json", type=Path, required=True)
    parser.add_argument("--target-json", type=Path, nargs="+", required=True)
    parser.add_argument("--template-clusters-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--quantile", type=float, default=0.95)
    parser.add_argument("--secondary-quantile", type=float, default=0.99)
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    nuisance_payload = read_json(args.nuisance_json)
    target_payloads = [read_json(path) for path in args.target_json]
    template_payload = read_json(args.template_clusters_json)
    profiles = calibrate(
        nuisance_payload,
        target_payloads,
        template_payload,
        args.quantile,
        args.secondary_quantile,
    )
    diagnostics = threshold_free_diagnostics(
        nuisance_payload,
        target_payloads,
        template_payload,
        args.n_boot,
        args.seed,
    )
    payload = {
        "experiment": "E2_null_calibration",
        "status": "aggregated_from_pre_evaluated_nuisance_distances",
        "analysis_scope": (
            "point_estimates_with_threshold_free_diagnostics_and_primary_cluster_bootstrap_delta_ci"
            if args.n_boot > 0
            else "point_estimates_with_threshold_free_diagnostics_no_bootstrap_ci"
        ),
        "primary_analysis_group": "primary_anchor_preserving",
        "punctuation_policy": (
            "terminal_period_toggle is reported as punctuation_stress; "
            "tier1_punctuation_inclusive is supplementary/conservative."
        ),
        "quantile": args.quantile,
        "secondary_quantile": args.secondary_quantile,
        "bootstrap_status": (
            "computed_fixed_family_and_family_resampled_for_primary_anchor_preserving_abs_dz_s_decode_resp_deltas"
            if args.n_boot > 0
            else "not_computed"
        ),
        "n_boot": args.n_boot,
        "seed": args.seed,
        "endpoint_hierarchy": ENDPOINT_HIERARCHY,
        "cluster_bootstrap_note": (
            "Cluster-aware CIs include fixed-family and family-resampled hierarchical "
            "template-cluster bootstraps for primary anchor-preserving abs_dz, s, and "
            "decode_resp delta diagnostics. Calibration and test clusters are resampled "
            "separately; thresholds are recalibrated inside every bootstrap replicate."
        ),
        "profiles": profiles,
        "threshold_free_diagnostics": diagnostics,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_markdown(args.output_md, payload)
    print(f"Saved JSON -> {args.output_json}")
    print(f"Saved Markdown -> {args.output_md}")


if __name__ == "__main__":
    main()
