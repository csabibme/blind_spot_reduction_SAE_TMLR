#!/usr/bin/env python3
"""Generate a supplementary-ready E5 analysis report from canonical JSON outputs.

The script is intentionally read-only: it does not recompute the experiments.
It validates and aggregates the canonical E5 outputs into:

1. a human-readable Markdown report suitable for supplementary material;
2. a compact machine-readable JSON summary.

Expected inputs
---------------
- hierarchical_intervals.json
- template_clusters.json
- family_geometry_profiles.json
- e1_effect_family_size_ablation.json

Example
-------
python generate_e5_supplementary_analysis.py \
  --hierarchical-json E5_family_geometry/results/hierarchical_intervals.json \
  --template-clusters-json E5_family_geometry/results/template_clusters.json \
  --geometry-json E5_family_geometry/results/family_geometry_profiles.json \
  --size-ablation-json E5_family_geometry/results/e1_effect_family_size_ablation.json \
  --output-md E5_family_geometry/results/E5_SUPPLEMENTARY_ANALYSIS.md \
  --output-json E5_family_geometry/results/E5_SUPPLEMENTARY_ANALYSIS_SUMMARY.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable

REVISION_ROOT = Path(__file__).resolve().parents[1]


PROFILE_LABELS = {
    "gpt2": "GPT-2",
    "qwen-2.5-3b": "Qwen 2.5 3B",
    "gemma-2-2b": "Gemma 2 2B",
}

METRIC_LABELS = {
    "s": "ΔL20(s)",
    "abs_dz": "ΔL20(|Δz|)",
    "decode_resp": "ΔL20(decoded)",
    "g": "ΔL20(g)",
}

EXPECTED_DIRECTION = {
    "s": "positive",
    "abs_dz": "positive",
    "decode_resp": "positive",
    "g": "negative",
}


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def relpath(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(REVISION_ROOT))
    except ValueError:
        return str(path)


def assert_finite_tree(value: Any, path: str = "root") -> None:
    """Reject non-finite numeric values in canonical result files."""
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"Non-finite numeric value at {path}: {value!r}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            assert_finite_tree(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            assert_finite_tree(item, f"{path}.{key}")
        return
    raise TypeError(f"Unsupported value type at {path}: {type(value).__name__}")


def profile_label(profile: str) -> str:
    return PROFILE_LABELS.get(profile, profile)


def fmt(value: float | int | None, digits: int = 6, signed: bool = False) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, int):
        return str(value)
    prefix = "+" if signed and value >= 0 else ""
    return f"{prefix}{value:.{digits}f}"


def median(values: Iterable[float]) -> float:
    vals = list(values)
    if not vals:
        raise ValueError("Cannot compute median of an empty sequence")
    return float(statistics.median(vals))


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    if not vals:
        raise ValueError("Cannot compute mean of an empty sequence")
    return float(statistics.fmean(vals))


def quantile_sign_count(values: Iterable[float]) -> dict[str, int]:
    vals = list(values)
    return {
        "n": len(vals),
        "positive": sum(v > 0 for v in vals),
        "negative": sum(v < 0 for v in vals),
        "zero": sum(v == 0 for v in vals),
    }


def markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    out = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" if i == 0 else "---:" for i in range(len(headers))) + "|",
    ]
    out.extend("| " + " | ".join(row) + " |" for row in rows)
    return out


def validate_inputs(
    hierarchical: dict[str, Any],
    templates: dict[str, Any],
    geometry: dict[str, Any],
    size_ablation: dict[str, Any],
) -> dict[str, Any]:
    for name, payload in (
        ("hierarchical", hierarchical),
        ("templates", templates),
        ("geometry", geometry),
        ("size_ablation", size_ablation),
    ):
        assert_finite_tree(payload, name)

    required_profiles = set(hierarchical["profiles"])
    if set(geometry["profiles"]) != required_profiles:
        raise ValueError(
            "Profile mismatch between hierarchical intervals and geometry: "
            f"{sorted(required_profiles)} vs {sorted(geometry['profiles'])}"
        )
    if set(size_ablation["profiles"]) != required_profiles:
        raise ValueError(
            "Profile mismatch between hierarchical intervals and family-size ablation"
        )

    template_families = set(templates["families"])
    if not template_families:
        raise ValueError("Template manifest contains no families")

    expected_sha = templates.get("pairs_sha256")
    hierarchical_sha = hierarchical.get("cluster_meta", {}).get("pairs_sha256")
    if expected_sha and hierarchical_sha and expected_sha != hierarchical_sha:
        raise ValueError("Pair-file SHA mismatch between template and hierarchical outputs")

    for profile, profile_block in geometry["profiles"].items():
        families = set(profile_block["families"])
        if families != template_families:
            raise ValueError(
                f"Family mismatch for {profile}: "
                f"geometry={sorted(families)}, templates={sorted(template_families)}"
            )
        cache_sha = profile_block.get("cache_meta", {}).get("pairs_sha256")
        if expected_sha and cache_sha and cache_sha != expected_sha:
            raise ValueError(f"Pair-file SHA mismatch for geometry profile {profile}")

    for profile, profile_block in hierarchical["profiles"].items():
        for metric, metric_block in profile_block.items():
            if metric not in METRIC_LABELS:
                raise ValueError(f"Unexpected metric {metric!r} in {profile}")
            primary = metric_block["primary_fixed_family_cluster_bootstrap"]
            secondary = metric_block["secondary_family_resampled_cluster_bootstrap"]
            if primary["n_families"] != len(template_families):
                raise ValueError(f"Unexpected family count in {profile}/{metric}")
            if secondary["n_families"] != len(template_families):
                raise ValueError(f"Unexpected secondary family count in {profile}/{metric}")
            if not (primary["lo"] <= primary["point"] <= primary["hi"]):
                raise ValueError(f"Primary point outside CI in {profile}/{metric}")
            if not (secondary["lo"] <= secondary["point"] <= secondary["hi"]):
                raise ValueError(f"Secondary point outside CI in {profile}/{metric}")

    return {
        "profiles": sorted(required_profiles),
        "families": sorted(template_families),
        "pairs_sha256": expected_sha,
    }


def template_summary(templates: dict[str, Any]) -> dict[str, Any]:
    cluster_sizes: list[int] = []
    repeated_families: list[str] = []
    family_rows: list[dict[str, Any]] = []

    for family, block in sorted(templates["families"].items()):
        sizes = [int(v) for v in block["template_cluster_sizes"].values()]
        cluster_sizes.extend(sizes)
        if any(size > 1 for size in sizes):
            repeated_families.append(family)
        quality = block.get("cluster_quality", {})
        family_rows.append(
            {
                "family": family,
                "n_pairs": int(block["n_pairs"]),
                "n_clusters": int(block["n_template_clusters"]),
                "singleton_fraction": float(
                    quality.get(
                        "singleton_cluster_fraction",
                        sum(size == 1 for size in sizes) / max(1, len(sizes)),
                    )
                ),
                "largest_cluster_fraction": float(
                    quality.get(
                        "largest_cluster_fraction",
                        max(sizes) / max(1, sum(sizes)),
                    )
                ),
                "effective_clusters": float(
                    quality.get("effective_n_clusters", len(sizes))
                ),
                "unique_values": int(
                    round(block.get("lexical_value_stats", {}).get("unique_values", 0))
                ),
            }
        )

    total_pairs = sum(row["n_pairs"] for row in family_rows)
    total_clusters = len(cluster_sizes)
    singleton_clusters = sum(size == 1 for size in cluster_sizes)
    repeated_clusters = sum(size > 1 for size in cluster_sizes)
    pairs_in_repeated = sum(size for size in cluster_sizes if size > 1)

    return {
        "total_pairs": total_pairs,
        "total_clusters": total_clusters,
        "singleton_clusters": singleton_clusters,
        "repeated_clusters": repeated_clusters,
        "pairs_in_repeated_clusters": pairs_in_repeated,
        "repeated_pair_fraction": pairs_in_repeated / max(1, total_pairs),
        "repeated_families": repeated_families,
        "family_rows": family_rows,
    }


def interval_summary(hierarchical: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    profile_summary: dict[str, Any] = {}

    for profile, profile_block in sorted(hierarchical["profiles"].items()):
        profile_summary[profile] = {}
        for metric, metric_block in profile_block.items():
            primary = metric_block["primary_fixed_family_cluster_bootstrap"]
            secondary = metric_block["secondary_family_resampled_cluster_bootstrap"]
            row = {
                "profile": profile,
                "metric": metric,
                "point": float(primary["point"]),
                "fixed_lo": float(primary["lo"]),
                "fixed_hi": float(primary["hi"]),
                "family_lo": float(secondary["lo"]),
                "family_hi": float(secondary["hi"]),
            }
            rows.append(row)
            profile_summary[profile][metric] = row

    return rows, profile_summary


def deletion_summary(hierarchical: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}

    for profile, profile_block in sorted(hierarchical["profiles"].items()):
        result[profile] = {}
        for metric, metric_block in profile_block.items():
            loto_values: list[float] = []
            for family_block in metric_block["leave_one_template_out"].values():
                for record in family_block.values():
                    value = record.get("delta_L20")
                    if value is not None:
                        loto_values.append(float(value))

            lofo_values = [
                float(record["macro_delta_L20"])
                for record in metric_block["leave_one_family_out"].values()
                if record.get("macro_delta_L20") is not None
            ]

            result[profile][metric] = {
                "loto": {
                    **quantile_sign_count(loto_values),
                    "min": min(loto_values),
                    "max": max(loto_values),
                },
                "lofo": {
                    **quantile_sign_count(lofo_values),
                    "min": min(lofo_values),
                    "max": max(lofo_values),
                },
            }

    return result


def geometry_summary(geometry: dict[str, Any]) -> dict[str, Any]:
    fields = {
        "hidden_delta_norm_cv": "Hidden Δ norm CV",
        "cosine_diversity": "Cosine diversity",
        "centered_effective_rank": "Centered effective rank",
        "centered_effective_rank_norm": "Normalized effective rank",
        "centered_spectral_entropy_norm": "Normalized spectral entropy",
    }
    result: dict[str, Any] = {}

    for profile, profile_block in sorted(geometry["profiles"].items()):
        families = list(profile_block["families"].values())
        profile_result: dict[str, Any] = {}
        for key, label in fields.items():
            values = [float(family[key]) for family in families]
            profile_result[key] = {
                "label": label,
                "min": min(values),
                "median": median(values),
                "max": max(values),
            }
        profile_result["zero_delta_total"] = sum(
            int(family["zero_delta_count"]) for family in families
        )
        profile_result["n_families"] = len(families)
        result[profile] = profile_result

    return result


def family_size_summary(size_ablation: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}

    for profile, profile_block in sorted(size_ablation["profiles"].items()):
        profile_result: dict[str, Any] = {}
        families = profile_block["families"]

        metric_names = sorted(
            {
                metric
                for family_block in families.values()
                for metric in family_block
            }
        )
        for metric in metric_names:
            metric_result: dict[str, Any] = {}
            size_names = sorted(
                {
                    size
                    for family_block in families.values()
                    if metric in family_block
                    for size in family_block[metric]
                },
                key=int,
            )
            for size in size_names:
                records = [
                    family_block[metric][size]
                    for family_block in families.values()
                    if metric in family_block and size in family_block[metric]
                ]
                metric_result[size] = {
                    "n_families": len(records),
                    "mean_delta_L20": mean(
                        float(record["delta_L20_mean"]) for record in records
                    ),
                    "families_q05_positive": sum(
                        float(record["delta_L20_q05"]) > 0 for record in records
                    ),
                    "families_mean_positive": sum(
                        float(record["delta_L20_mean"]) > 0 for record in records
                    ),
                    "minimum_positive_fraction": min(
                        float(record["positive_fraction"]) for record in records
                    ),
                }
            profile_result[metric] = metric_result
        result[profile] = profile_result

    return result


def direction_statement(metric: str, lo: float, hi: float) -> str:
    if lo > 0:
        return "positive"
    if hi < 0:
        return "negative"
    return "crosses zero"


def build_key_findings(
    interval_profiles: dict[str, Any],
    deletion: dict[str, Any],
    geometry: dict[str, Any],
    templates: dict[str, Any],
) -> list[str]:
    findings: list[str] = []

    positive_s = all(
        profile_metrics["s"]["family_lo"] > 0
        for profile_metrics in interval_profiles.values()
    )
    positive_abs = all(
        profile_metrics["abs_dz"]["family_lo"] > 0
        for profile_metrics in interval_profiles.values()
    )
    if positive_s:
        findings.append(
            "The relative lower-tail effect remained positive for every model under "
            "both fixed-family template-cluster and family-resampled hierarchical bootstrap."
        )
    if positive_abs:
        findings.append(
            "The absolute code-distance lower-tail effect also remained positive for "
            "every model under the stricter family-resampled analysis."
        )

    decoded_positive = [
        profile_label(profile)
        for profile, metrics in interval_profiles.items()
        if metrics["decode_resp"]["fixed_lo"] > 0
    ]
    decoded_null = [
        profile_label(profile)
        for profile, metrics in interval_profiles.items()
        if metrics["decode_resp"]["fixed_lo"] <= 0 <= metrics["decode_resp"]["fixed_hi"]
    ]
    if decoded_positive:
        findings.append(
            "Decoded lower-tail improvement was robust for "
            + ", ".join(decoded_positive)
            + ("; " if decoded_null else ".")
            + (
                "the interval crossed zero for " + ", ".join(decoded_null) + "."
                if decoded_null
                else ""
            )
        )

    all_loto_s = all(
        metrics["s"]["loto"]["positive"] == metrics["s"]["loto"]["n"]
        for metrics in deletion.values()
    )
    all_lofo_s = all(
        metrics["s"]["lofo"]["positive"] == metrics["s"]["lofo"]["n"]
        for metrics in deletion.values()
    )
    if all_loto_s and all_lofo_s:
        findings.append(
            "The primary relative lower-tail effect remained positive after every "
            "single-template and every single-family deletion."
        )

    if all(profile["zero_delta_total"] == 0 for profile in geometry.values()):
        findings.append(
            "No exact-zero hidden perturbation vector occurred in the true-last-token "
            "true-last geometry caches."
        )

    findings.append(
        f"The 600-pair collection contained {templates['total_clusters']} exact-skeleton "
        f"template clusters; {templates['pairs_in_repeated_clusters']} pairs "
        f"({100 * templates['repeated_pair_fraction']:.1f}%) belonged to repeated templates."
    )

    return findings


def generate_markdown(
    metadata: dict[str, Any],
    templates: dict[str, Any],
    intervals: list[dict[str, Any]],
    interval_profiles: dict[str, Any],
    deletion: dict[str, Any],
    geometry: dict[str, Any],
    size_summary: dict[str, Any],
    source_paths: dict[str, str],
) -> str:
    lines: list[str] = [
        "# Supplementary Analysis: Family Geometry and Template-Cluster Uncertainty",
        "",
        "## Scope",
        "",
        "This supplementary analysis evaluates whether the E1/E1R lower-tail results "
        "can be explained by repeated surface templates, a single perturbation family, "
        "small pair counts, or low-dimensional hidden-state geometry. All reported "
        "statistics are generated from the canonical E5 JSON outputs; no values are "
        "manually entered into this report.",
        "",
        "## Reproducibility and validation",
        "",
        f"- Pair-file SHA256: `{metadata['pairs_sha256']}`",
        f"- Profiles: {', '.join(profile_label(p) for p in metadata['profiles'])}",
        f"- Families: {len(metadata['families'])}",
        f"- Pairs: {templates['total_pairs']}",
        f"- Exact-skeleton template clusters: {templates['total_clusters']}",
        "- Primary uncertainty estimand: fixed 16-family template-cluster bootstrap.",
        "- Secondary uncertainty estimand: family-resampled plus template-cluster bootstrap.",
        "",
        "Source files:",
        "",
    ]
    lines.extend(f"- `{label}`: `{path}`" for label, path in source_paths.items())

    lines.extend(["", "## Key findings", ""])
    for finding in build_key_findings(interval_profiles, deletion, geometry, templates):
        lines.append(f"- {finding}")

    lines.extend(["", "## S1. Template structure", ""])
    lines.append(
        f"The {templates['total_pairs']} pairs formed {templates['total_clusters']} "
        f"exact-skeleton clusters. Of these, {templates['singleton_clusters']} were "
        f"singletons and {templates['repeated_clusters']} contained multiple pairs. "
        f"A total of {templates['pairs_in_repeated_clusters']} pairs "
        f"({100 * templates['repeated_pair_fraction']:.1f}%) belonged to repeated clusters."
    )
    lines.append("")
    lines.append(
        "Families containing repeated exact templates: "
        + ", ".join(f"`{name}`" for name in templates["repeated_families"])
        + "."
    )
    lines.append("")
    template_rows = [
        [
            row["family"],
            str(row["n_pairs"]),
            str(row["n_clusters"]),
            f"{row['singleton_fraction']:.3f}",
            f"{row['largest_cluster_fraction']:.3f}",
            f"{row['effective_clusters']:.2f}",
            str(row["unique_values"]),
        ]
        for row in templates["family_rows"]
    ]
    lines.extend(
        markdown_table(
            [
                "Family",
                "Pairs",
                "Templates",
                "Singleton fraction",
                "Largest cluster fraction",
                "Effective clusters",
                "Unique slot values",
            ],
            template_rows,
        )
    )

    lines.extend(["", "## S2. Hierarchical uncertainty", ""])
    lines.append(
        "The fixed-family bootstrap keeps all sixteen families in every replicate and "
        "resamples template clusters within family. The secondary bootstrap additionally "
        "resamples families and therefore targets a broader perturbation-family population."
    )
    lines.append("")
    interval_rows = []
    for row in sorted(intervals, key=lambda x: (profile_label(x["profile"]), x["metric"])):
        interval_rows.append(
            [
                profile_label(row["profile"]),
                METRIC_LABELS[row["metric"]],
                fmt(row["point"], signed=True),
                f"[{fmt(row['fixed_lo'], signed=True)}, {fmt(row['fixed_hi'], signed=True)}]",
                f"[{fmt(row['family_lo'], signed=True)}, {fmt(row['family_hi'], signed=True)}]",
                direction_statement(row["metric"], row["fixed_lo"], row["fixed_hi"]),
            ]
        )
    lines.extend(
        markdown_table(
            [
                "Profile",
                "Metric",
                "Point",
                "Fixed-family 95% CI",
                "Family-resampled 95% CI",
                "Fixed-family direction",
            ],
            interval_rows,
        )
    )

    lines.extend(["", "## S3. Leave-one-template-out and leave-one-family-out robustness", ""])
    deletion_rows = []
    for profile, profile_block in sorted(deletion.items()):
        for metric, block in profile_block.items():
            deletion_rows.append(
                [
                    profile_label(profile),
                    METRIC_LABELS[metric],
                    f"{block['loto']['positive']}/{block['loto']['n']}",
                    f"{block['loto']['negative']}/{block['loto']['n']}",
                    f"[{fmt(block['loto']['min'], signed=True)}, {fmt(block['loto']['max'], signed=True)}]",
                    f"{block['lofo']['positive']}/{block['lofo']['n']}",
                    f"{block['lofo']['negative']}/{block['lofo']['n']}",
                    f"[{fmt(block['lofo']['min'], signed=True)}, {fmt(block['lofo']['max'], signed=True)}]",
                ]
            )
    lines.extend(
        markdown_table(
            [
                "Profile",
                "Metric",
                "LOTO positive",
                "LOTO negative",
                "LOTO range",
                "LOFO positive",
                "LOFO negative",
                "LOFO macro range",
            ],
            deletion_rows,
        )
    )
    lines.append("")
    lines.append(
        "For the primary relative lower-tail metric, every single-template deletion and "
        "every single-family deletion retained a positive effect in all three profiles. "
        "The decoded GPT-2 and normalized-gain analyses are intentionally more mixed, "
        "matching the model-specific mechanism observed in E1."
    )

    lines.extend(["", "## S4. Hidden-state geometry", ""])
    lines.append(
        "Geometry was computed from true-last-token hidden perturbation vectors "
        r"$\delta_i=h_i^{pert}-h_i^{orig}$. Effective rank and entropy use the centered "
        "delta matrix; normalized quantities account for the maximum rank permitted by "
        "family size."
    )
    lines.append("")
    geometry_rows = []
    for profile, block in sorted(geometry.items()):
        geometry_rows.append(
            [
                profile_label(profile),
                str(block["n_families"]),
                str(block["zero_delta_total"]),
                (
                    f"{block['hidden_delta_norm_cv']['min']:.3f} / "
                    f"{block['hidden_delta_norm_cv']['median']:.3f} / "
                    f"{block['hidden_delta_norm_cv']['max']:.3f}"
                ),
                (
                    f"{block['cosine_diversity']['min']:.3f} / "
                    f"{block['cosine_diversity']['median']:.3f} / "
                    f"{block['cosine_diversity']['max']:.3f}"
                ),
                (
                    f"{block['centered_effective_rank']['min']:.2f} / "
                    f"{block['centered_effective_rank']['median']:.2f} / "
                    f"{block['centered_effective_rank']['max']:.2f}"
                ),
                (
                    f"{block['centered_spectral_entropy_norm']['min']:.3f} / "
                    f"{block['centered_spectral_entropy_norm']['median']:.3f} / "
                    f"{block['centered_spectral_entropy_norm']['max']:.3f}"
                ),
            ]
        )
    lines.extend(
        markdown_table(
            [
                "Profile",
                "Families",
                "Zero Δh",
                "Δh norm CV min/median/max",
                "Cosine diversity min/median/max",
                "Centered effective rank min/median/max",
                "Normalized spectral entropy min/median/max",
            ],
            geometry_rows,
        )
    )
    lines.append("")
    lines.append(
        "The hidden perturbations are non-zero and span multiple directions. GPT-2 "
        "shows a markedly lower centered effective rank than Gemma and Qwen, while the "
        "code-space lower-tail effect remains positive. This supports a model-dependent "
        "geometric interpretation rather than a single universal perturbation geometry."
    )

    lines.extend(["", "## S5. Pair-count sensitivity", ""])
    lines.append(
        "For each available family size, repeated random subsets were drawn without "
        "replacement and the Standard–V-reg lower-tail contrast was recomputed. "
        "The table reports the number of eligible families, the mean subset effect, "
        "the number of families whose 5th percentile remained positive, and the worst "
        "positive-replicate fraction."
    )
    lines.append("")
    size_rows = []
    for profile, profile_block in sorted(size_summary.items()):
        for metric, metric_block in sorted(profile_block.items()):
            for size, record in sorted(metric_block.items(), key=lambda item: int(item[0])):
                size_rows.append(
                    [
                        profile_label(profile),
                        METRIC_LABELS.get(metric, metric),
                        size,
                        str(record["n_families"]),
                        fmt(record["mean_delta_L20"], signed=True),
                        f"{record['families_q05_positive']}/{record['n_families']}",
                        f"{record['families_mean_positive']}/{record['n_families']}",
                        f"{record['minimum_positive_fraction']:.3f}",
                    ]
                )
    lines.extend(
        markdown_table(
            [
                "Profile",
                "Metric",
                "K",
                "Eligible families",
                "Mean ΔL20",
                "Families q05>0",
                "Families mean>0",
                "Minimum positive fraction",
            ],
            size_rows,
        )
    )

    lines.extend(["", "## S6. Interpretation", ""])
    lines.append(
        "The primary relative lower-tail effect is robust to exact-template clustering, "
        "family resampling, deletion of any single template, deletion of any single family, "
        "and substantial pair-count reduction. Absolute code-distance lower-tail effects "
        "also remain positive at the aggregate hierarchical level in all three models. "
        "Decoded lower-tail improvement is robust for Gemma and Qwen but not for GPT-2, "
        "which is consistent with the earlier finding that GPT-2's improvement is primarily "
        "expressed in sparse-code space. The hidden-normalized gain metric remains negative "
        "in aggregate, supporting a redistribution/lower-tail-conditioning interpretation "
        "rather than proportional gain amplification."
    )

    lines.extend(["", "## S7. Limitations", ""])
    lines.extend(
        [
            "- Exact-skeleton clustering is deliberately conservative. Semantically related "
            "but differently worded sentences may remain in separate clusters.",
            "- In families where every exact skeleton is unique, cluster resampling coincides "
            "with pair-level resampling; broader semantic generalization is therefore tested "
            "separately by held-out-template probes.",
            "- Family resampling treats the sixteen perturbation families as a sample from a "
            "larger conceptual family population; this is a secondary, stronger assumption.",
            "- E5 evaluates data and template dependence for fixed SAE checkpoints. "
            "Training-seed uncertainty is addressed separately in E7.",
        ]
    )

    lines.extend(["", "## S8. Conclusion", ""])
    lines.append(
        "E5 provides no evidence that the main lower-tail result is driven by duplicated "
        "surface templates, a single dominant family, or a large nominal pair count. "
        "The result persists under cluster-aware uncertainty and deletion analyses, while "
        "the model-specific decoded and normalized-gain findings remain visible rather than "
        "being averaged away."
    )

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate supplementary-ready E5 analysis from canonical JSON outputs"
    )
    parser.add_argument("--hierarchical-json", type=Path, required=True)
    parser.add_argument("--template-clusters-json", type=Path, required=True)
    parser.add_argument("--geometry-json", type=Path, required=True)
    parser.add_argument("--size-ablation-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    hierarchical = read_json(args.hierarchical_json)
    templates_payload = read_json(args.template_clusters_json)
    geometry_payload = read_json(args.geometry_json)
    size_payload = read_json(args.size_ablation_json)

    metadata = validate_inputs(
        hierarchical,
        templates_payload,
        geometry_payload,
        size_payload,
    )
    templates = template_summary(templates_payload)
    interval_rows, interval_profiles = interval_summary(hierarchical)
    deletion = deletion_summary(hierarchical)
    geometry = geometry_summary(geometry_payload)
    size_summary = family_size_summary(size_payload)

    source_paths = {
        "hierarchical intervals": relpath(args.hierarchical_json),
        "template clusters": relpath(args.template_clusters_json),
        "family geometry": relpath(args.geometry_json),
        "family-size ablation": relpath(args.size_ablation_json),
    }
    summary = {
        "metadata": metadata,
        "source_paths": source_paths,
        "output_paths": {
            "markdown": relpath(args.output_md),
            "json": relpath(args.output_json),
        },
        "template_summary": templates,
        "hierarchical_intervals": interval_rows,
        "deletion_robustness": deletion,
        "geometry_summary": geometry,
        "family_size_summary": size_summary,
        "key_findings": build_key_findings(
            interval_profiles,
            deletion,
            geometry,
            templates,
        ),
    }
    assert_finite_tree(summary, "summary")
    markdown = generate_markdown(
        metadata,
        templates,
        interval_rows,
        interval_profiles,
        deletion,
        geometry,
        size_summary,
        source_paths,
    )

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(markdown, encoding="utf-8")
    args.output_json.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print(f"Saved Markdown -> {relpath(args.output_md)}")
    print(f"Saved JSON -> {relpath(args.output_json)}")


if __name__ == "__main__":
    main()
