#!/usr/bin/env python3
"""Build frozen negation_state task splits for E3 held-out probes.

Two split variants:

- ``primary``: family-stratified unseen-template split (template-only grouping).
- ``stress``: UVP-connected unseen-cue / cross-family stress split.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

CLAUSE_NEGATION_RE = re.compile(
    r"\b("
    r"no|not|never|denies|denied|negative|"
    r"cannot|can't|won't|isn't|aren't|wasn't|weren't|"
    r"doesn't|don't|didn't|hasn't|haven't|hadn't|"
    r"couldn't|wouldn't|shouldn't"
    r")\b",
    re.I,
)

PRIMARY_EXCLUSIONS = {
    "condition_negation::13",
    "condition_negation::16",
}

RANDOM_LABEL_SEEDS = [11, 23, 37, 53, 71]
REAL_LABEL_PROBE_SEED = 42
SPLIT_RATIOS = {"train": 0.60, "dev": 0.20, "test": 0.20}

SPLIT_VARIANTS = {
    "primary": {
        "status": "negation_task_split_frozen",
        "split_variant": "family_stratified_unseen_template",
        "primary_role": "primary_downstream_probe",
        "policy_primary": "family_stratified_unseen_template",
        "assignment_unit": "template_id",
        "assignment_method": "seeded_random_order_greedy_balance_within_family",
        "leakage_units": ["template_id", "source_sentence_id"],
    },
    "stress": {
        "status": "negation_stress_split_frozen",
        "split_variant": "unseen_cue_uvp_safe",
        "primary_role": "secondary_stress_test",
        "policy_primary": "unseen_cue_cross_family_stress",
        "assignment_unit": "connected_components_of_template_and_unordered_value_pair",
        "assignment_method": "greedy_balance_by_pair_count",
        "leakage_units": ["template_id", "unordered_value_pair_id", "source_sentence_id"],
    },
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def negation_state_label(text: str) -> str:
    return "negated" if CLAUSE_NEGATION_RE.search(text) else "affirmed"


def select_primary_negation_pairs(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = [
        record
        for record in records
        if "negation_state" in record["candidate_task_tags"]
        and record["global_pair_id"] not in PRIMARY_EXCLUSIONS
    ]
    if not selected:
        raise ValueError("No primary negation_state records found")
    return selected


def template_units(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    units: dict[str, dict[str, Any]] = {}
    for record in records:
        template_id = record["leakage_groups"]["template"]
        if template_id not in units:
            units[template_id] = {
                "template_id": template_id,
                "family": record["family"],
                "pair_count": 0,
            }
        units[template_id]["pair_count"] += 1
    return units


def split_targets(total_pairs: int) -> dict[str, int]:
    raw = {name: SPLIT_RATIOS[name] * total_pairs for name in SPLIT_RATIOS}
    floored = {name: int(raw[name]) for name in SPLIT_RATIOS}
    remainder = total_pairs - sum(floored.values())
    order = sorted(SPLIT_RATIOS, key=lambda name: raw[name] - floored[name], reverse=True)
    targets = dict(floored)
    for name in order[:remainder]:
        targets[name] += 1
    return targets


def assign_primary_family_stratified(
    records: list[dict[str, Any]],
    seed: int,
) -> dict[str, str]:
    rng = random.Random(seed)
    units = template_units(records)
    template_to_split: dict[str, str] = {}
    families = sorted({unit["family"] for unit in units.values()})

    for family in families:
        family_units = [unit for unit in units.values() if unit["family"] == family]
        family_total = sum(unit["pair_count"] for unit in family_units)
        targets = split_targets(family_total)
        split_counts = {name: 0 for name in SPLIT_RATIOS}

        ordered = sorted(family_units, key=lambda unit: unit["pair_count"], reverse=True)
        rng.shuffle(ordered)

        for unit in ordered:
            pair_count = unit["pair_count"]
            best_split = min(
                SPLIT_RATIOS,
                key=lambda name: (split_counts[name] + pair_count) / max(targets[name], 1),
            )
            template_to_split[unit["template_id"]] = best_split
            split_counts[best_split] += pair_count

        if not all(split_counts[name] > 0 for name in SPLIT_RATIOS):
            raise ValueError(
                f"Primary split failed to populate all splits for family={family}: "
                f"{split_counts} targets={targets}"
            )

    all_families = {record["family"] for record in records}
    for split_name in SPLIT_RATIOS:
        split_families = {
            units[template_id]["family"]
            for template_id, split in template_to_split.items()
            if split == split_name
        }
        if split_families != all_families:
            raise ValueError(
                f"Primary split {split_name} missing families: "
                f"have={sorted(split_families)} need={sorted(all_families)}"
            )
    return template_to_split


def build_leakage_components(records: list[dict[str, Any]]) -> dict[str, set[str]]:
    templates = {record["leakage_groups"]["template"] for record in records}
    parent = {template: template for template in templates}

    def find(node: str) -> str:
        while parent[node] != node:
            node = parent[node]
        return node

    def union(left: str, right: str) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    uvp_to_templates: dict[str, set[str]] = defaultdict(set)
    for record in records:
        uvp_to_templates[record["leakage_groups"]["unordered_value_pair"]].add(
            record["leakage_groups"]["template"]
        )
    for template_group in uvp_to_templates.values():
        ordered = sorted(template_group)
        for template in ordered[1:]:
            union(ordered[0], template)

    components: dict[str, set[str]] = defaultdict(set)
    for template in templates:
        components[find(template)].add(template)
    return dict(components)


def assign_stress_uvp_greedy(
    records: list[dict[str, Any]],
    components: dict[str, set[str]],
) -> dict[str, str]:
    template_pair_counts = Counter(record["leakage_groups"]["template"] for record in records)
    component_items: list[tuple[int, set[str]]] = []
    for templates in components.values():
        pair_count = sum(template_pair_counts[template] for template in templates)
        component_items.append((pair_count, templates))
    component_items.sort(key=lambda item: item[0], reverse=True)

    split_names = ("train", "dev", "test")
    split_pair_counts = {name: 0 for name in split_names}
    template_to_split: dict[str, str] = {}
    for pair_count, templates in component_items:
        target_split = min(split_names, key=lambda name: split_pair_counts[name])
        for template in templates:
            template_to_split[template] = target_split
        split_pair_counts[target_split] += pair_count
    if not all(split_pair_counts[name] > 0 for name in split_names):
        raise ValueError(
            "Stress split assignment failed to populate all splits; "
            f"component_count={len(component_items)} pair_counts={split_pair_counts}"
        )
    return template_to_split


def build_examples(
    records: list[dict[str, Any]],
    template_to_split: dict[str, str],
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for record in records:
        template_id = record["leakage_groups"]["template"]
        split = template_to_split[template_id]
        for side in ("orig", "pert"):
            text = record["texts"][side]
            label = negation_state_label(text)
            examples.append(
                {
                    "example_id": record["source_sentence_ids"][side],
                    "task": "negation_state",
                    "label": label,
                    "split": split,
                    "family": record["family"],
                    "pair_index": record["pair_index"],
                    "global_pair_id": record["global_pair_id"],
                    "side": side,
                    "text": text,
                    "template_id": template_id,
                    "template_signature": record["template_signature"],
                    "unordered_value_pair_id": record["leakage_groups"]["unordered_value_pair"],
                    "source_sentence_id": record["source_sentence_ids"][side],
                    "leakage_groups": {
                        "template": template_id,
                        "unordered_value_pair": record["leakage_groups"]["unordered_value_pair"],
                        "source_sentence_id": record["source_sentence_ids"][side],
                        "source_pair": record["leakage_groups"]["source_pair"],
                    },
                }
            )
    return examples


def split_audit(examples: list[dict[str, Any]]) -> dict[str, Any]:
    audit: dict[str, Any] = {"splits": {}, "global": {}}
    text_counter = Counter(example["text"] for example in examples)
    duplicate_texts = {text: count for text, count in text_counter.items() if count > 1}

    for split in ("train", "dev", "test"):
        split_examples = [example for example in examples if example["split"] == split]
        unique_texts = {example["text"] for example in split_examples}
        audit["splits"][split] = {
            "n_examples": len(split_examples),
            "n_unique_texts": len(unique_texts),
            "n_exact_text_duplicates": sum(
                count - 1 for text, count in Counter(example["text"] for example in split_examples).items() if count > 1
            ),
            "n_pairs": len({example["global_pair_id"] for example in split_examples}),
            "n_templates": len({example["template_id"] for example in split_examples}),
            "families": dict(Counter(example["family"] for example in split_examples)),
            "label_counts": dict(Counter(example["label"] for example in split_examples)),
            "templates_by_family": {
                family: len(
                    {
                        example["template_id"]
                        for example in split_examples
                        if example["family"] == family
                    }
                )
                for family in sorted({example["family"] for example in split_examples})
            },
        }

    audit["global"] = {
        "n_examples": len(examples),
        "n_unique_texts": len(text_counter),
        "n_exact_duplicate_texts": len(duplicate_texts),
        "duplicate_text_counts": dict(sorted(duplicate_texts.items(), key=lambda item: (-item[1], item[0]))[:20]),
    }
    return audit


def validate_examples(
    examples: list[dict[str, Any]],
    *,
    require_both_families_per_split: bool,
    require_uvp_leakage_safe: bool,
) -> dict[str, Any]:
    template_splits: dict[str, set[str]] = defaultdict(set)
    uvp_splits: dict[str, set[str]] = defaultdict(set)
    sentence_splits: dict[str, set[str]] = defaultdict(set)
    for example in examples:
        split = example["split"]
        template_splits[example["template_id"]].add(split)
        uvp_splits[example["unordered_value_pair_id"]].add(split)
        sentence_splits[example["source_sentence_id"]].add(split)

    def leakage_errors(mapping: dict[str, set[str]], name: str) -> list[str]:
        errors = []
        for key, splits in mapping.items():
            if len(splits) != 1:
                errors.append(f"{name} {key} spans {sorted(splits)}")
        return errors

    errors = []
    errors.extend(leakage_errors(template_splits, "template"))
    if require_uvp_leakage_safe:
        errors.extend(leakage_errors(uvp_splits, "unordered_value_pair"))
    errors.extend(leakage_errors(sentence_splits, "source_sentence_id"))
    if errors:
        raise ValueError("Leakage validation failed:\n" + "\n".join(errors[:20]))

    pair_labels: dict[str, set[str]] = defaultdict(set)
    for example in examples:
        pair_labels[example["global_pair_id"]].add(example["label"])
    same_label_pairs = sorted(
        pair_id for pair_id, labels in pair_labels.items() if len(labels) == 1
    )
    if same_label_pairs:
        raise ValueError(
            "Primary negation pairs with identical affirmed/negated labels on both sides: "
            + ", ".join(same_label_pairs[:10])
        )

    split_summary = {}
    for split in ("train", "dev", "test"):
        split_examples = [example for example in examples if example["split"] == split]
        split_summary[split] = {
            "n_examples": len(split_examples),
            "n_unique_texts": len({example["text"] for example in split_examples}),
            "n_pairs": len({example["global_pair_id"] for example in split_examples}),
            "n_templates": len({example["template_id"] for example in split_examples}),
            "families": dict(Counter(example["family"] for example in split_examples)),
            "label_counts": dict(Counter(example["label"] for example in split_examples)),
        }
        if require_both_families_per_split:
            families = set(split_summary[split]["families"])
            expected = {example["family"] for example in examples}
            if families != expected:
                raise ValueError(
                    f"Split {split} is not family-stratified: have={sorted(families)} "
                    f"need={sorted(expected)}"
                )

    return {
        "leakage_checks_passed": True,
        "split_summary": split_summary,
        "split_audit": split_audit(examples),
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["validation"]["split_summary"]
    audit = payload["validation"]["split_audit"]
    lines = [
        f"# E3 Negation Task Split ({payload['split_variant']})",
        "",
        f"Task: `{payload['task']}`",
        f"Status: `{payload['status']}`",
        f"Role: `{payload['primary_role']}`",
        f"Primary pairs: {payload['summary']['n_primary_pairs']}",
        f"Examples: {payload['summary']['n_examples']}",
        f"Templates: {payload['summary']['n_templates']}",
        "",
        "## Split Summary",
        "",
        "| Split | examples | unique texts | pairs | templates | negation | condition_negation | affirmed | negated |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split, block in summary.items():
        labels = block["label_counts"]
        families = block["families"]
        lines.append(
            f"| {split} | {block['n_examples']} | {block['n_unique_texts']} | {block['n_pairs']} | "
            f"{block['n_templates']} | {families.get('negation', 0)} | "
            f"{families.get('condition_negation', 0)} | "
            f"{labels.get('affirmed', 0)} | {labels.get('negated', 0)} |"
        )
    lines.extend(
        [
            "",
            "## Duplicate Text Audit",
            "",
            f"- Global unique texts: {audit['global']['n_unique_texts']}",
            f"- Global exact duplicate text groups: {audit['global']['n_exact_duplicate_texts']}",
            "",
            "## Exclusions",
            "",
            "Primary probe excludes:",
            "",
        ]
    )
    for pair_id in payload["primary_exclusions"]:
        lines.append(f"- `{pair_id}`")
    lines.extend(
        [
            "",
            "## Label Rule",
            "",
            payload["label_rule"]["description"],
            "",
            "Frozen regex:",
            "",
            "```text",
            payload["label_rule"]["regex"],
            "```",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_payload(
    manifest: dict[str, Any],
    grouping_manifest_json: Path,
    seed: int,
    split_variant: str,
) -> dict[str, Any]:
    if split_variant not in SPLIT_VARIANTS:
        raise ValueError(f"Unknown split variant: {split_variant}")

    variant = SPLIT_VARIANTS[split_variant]
    primary_pairs = select_primary_negation_pairs(manifest["records"])

    if split_variant == "primary":
        template_to_split = assign_primary_family_stratified(primary_pairs, seed)
        components = None
    else:
        components = build_leakage_components(primary_pairs)
        template_to_split = assign_stress_uvp_greedy(primary_pairs, components)

    examples = build_examples(primary_pairs, template_to_split)
    validation = validate_examples(
        examples,
        require_both_families_per_split=(split_variant == "primary"),
        require_uvp_leakage_safe=(split_variant == "stress"),
    )

    return {
        "experiment": "E3_heldout_probes",
        "task": "negation_state",
        "status": variant["status"],
        "split_variant": variant["split_variant"],
        "primary_role": variant["primary_role"],
        "grouping_manifest_json": str(grouping_manifest_json),
        "pairs_sha256": manifest.get("pairs_sha256"),
        "seed": seed,
        "split_policy": {
            "primary": variant["policy_primary"],
            "leakage_units": variant["leakage_units"],
            "assignment_unit": variant["assignment_unit"],
            "assignment_method": variant["assignment_method"],
            "target_ratios": SPLIT_RATIOS,
            "splits": ["train", "dev", "test"],
        },
        "label_rule": {
            "task_label": "semantic_negation_state",
            "labels": ["affirmed", "negated"],
            "description": (
                "Each sentence receives affirmed/negated from clause-level negation cues. "
                "Standalone prepositional 'without' is excluded from negation detection. "
                "The probe label is never original-vs-perturbed."
            ),
            "regex": CLAUSE_NEGATION_RE.pattern,
        },
        "primary_exclusions": sorted(PRIMARY_EXCLUSIONS),
        "sensitivity_analysis_pairs": sorted(PRIMARY_EXCLUSIONS),
        "random_label_control": {
            "shuffle_scope": "train_only",
            "dev_test_labels": "real",
            "seeds": RANDOM_LABEL_SEEDS,
        },
        "probe_protocol": {
            "representations": [
                "hidden",
                "sae_standard_code",
                "sae_vreg_code",
                "sae_standard_reconstruction",
                "sae_vreg_reconstruction",
            ],
            "primary_comparison": "sae_vreg_code_vs_sae_standard_code",
            "classifier": "linear_probe",
            "standardize_features": "train_only",
            "hyperparameter_selection": "dev_only",
            "test_evaluation": "once",
            "real_label_probe_seed": REAL_LABEL_PROBE_SEED,
            "probe_seeds": [REAL_LABEL_PROBE_SEED],
            "probe_seed_note": (
                "Single deterministic real-label fit; lbfgs logistic regression is effectively "
                "deterministic across random_state values. Uncertainty comes from offline "
                "template-cluster bootstrap over exported per-example predictions."
            ),
            "metrics": ["balanced_accuracy", "auroc", "macro_f1"],
            "export_predictions": ["dev", "test"],
        },
        "summary": {
            "n_primary_pairs": len(primary_pairs),
            "n_examples": len(examples),
            "n_templates": len({example["template_id"] for example in examples}),
            "n_leakage_components": len(components) if components is not None else None,
        },
        "template_to_split": {
            template: template_to_split[template]
            for template in sorted(template_to_split)
        },
        "validation": validation,
        "examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build frozen E3 negation task split")
    parser.add_argument("--grouping-manifest-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument(
        "--split-variant",
        choices=sorted(SPLIT_VARIANTS),
        default="primary",
        help="primary=family-stratified unseen-template; stress=UVP unseen-cue stress split",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    manifest = read_json(args.grouping_manifest_json)
    payload = build_payload(manifest, args.grouping_manifest_json, args.seed, args.split_variant)

    write_json_atomic(args.output_json, payload)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    write_markdown(args.output_md, payload)
    print(f"Saved JSON -> {args.output_json}")
    print(f"Saved Markdown -> {args.output_md}")


if __name__ == "__main__":
    main()
