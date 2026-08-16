#!/usr/bin/env python3
"""Build E3b external OpenI laterality minimal-pair probe dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPORT_SPLITS = {"train": 0.70, "dev": 0.15, "test": 0.15}
RANDOM_LABEL_SEEDS = [11, 23, 37, 53, 71]
LEFT_RE = re.compile(r"\bleft(?:-sided)?\b", re.I)
RIGHT_RE = re.compile(r"\bright(?:-sided)?\b", re.I)
EXCLUDE_RE = re.compile(
    r"\b(bilateral|bilaterally|both|left\s+and\s+right|right\s+and\s+left|"
    r"left/right|right/left|midline|rightward|leftward)\b",
    re.I,
)
ANATOMY_PATTERNS = {
    "lung": r"\b(lung|lungs|pulmonary)\b",
    "base": r"\b(base|basilar|lower lobe|lower lung)\b",
    "apex": r"\b(apex|apical|upper lobe|upper lung)\b",
    "pleural": r"\b(pleural|effusion|costophrenic)\b",
    "hilar": r"\b(hilar|hilum|perihilar)\b",
    "rib": r"\b(rib|ribs)\b",
    "clavicle": r"\b(clavicle|clavicular)\b",
    "line_or_tube": r"\b(line|tube|catheter|picc|port)\b",
    "heart_mediastinum": r"\b(heart|cardiac|mediastinum|mediastinal)\b",
    "other_laterality": r".*",
}
ANATOMY_RES = {name: re.compile(pattern, re.I) for name, pattern in ANATOMY_PATTERNS.items()}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"xxxx", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def sentence_split(report: str) -> list[str]:
    report = report.replace("..", ".")
    report = re.sub(r"\b\d+\.\s+", ". ", report)
    pieces = re.split(r"(?<=[.!?])\s+", report)
    sentences = []
    for piece in pieces:
        sent = piece.strip(" ;")
        sent = re.sub(r"\s+", " ", sent)
        if sent and not sent.endswith((".", "!", "?")):
            sent += "."
        if 5 <= len(sent.split()) <= 35:
            sentences.append(sent)
    return sentences


def read_index(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def split_reports(report_ids: list[str], seed: int) -> dict[str, str]:
    rng = random.Random(seed)
    ordered = sorted(report_ids)
    rng.shuffle(ordered)
    n_total = len(ordered)
    n_train = int(round(REPORT_SPLITS["train"] * n_total))
    n_dev = int(round(REPORT_SPLITS["dev"] * n_total))
    train_ids = set(ordered[:n_train])
    dev_ids = set(ordered[n_train : n_train + n_dev])
    return {
        report_id: "train" if report_id in train_ids else "dev" if report_id in dev_ids else "test"
        for report_id in ordered
    }


def laterality_label(sentence: str) -> str | None:
    if "XXXX" in sentence or EXCLUDE_RE.search(sentence):
        return None
    has_left = LEFT_RE.search(sentence) is not None
    has_right = RIGHT_RE.search(sentence) is not None
    if has_left == has_right:
        return None
    return "left" if has_left else "right"


def swap_laterality(sentence: str) -> str:
    left_plain = "__E3B_LEFT__"
    left_sided = "__E3B_LEFT_SIDED__"
    right_plain = "__E3B_RIGHT__"
    right_sided = "__E3B_RIGHT_SIDED__"
    sentence = LEFT_RE.sub(lambda m: left_sided if m.group(0).lower().endswith("-sided") else left_plain, sentence)
    sentence = RIGHT_RE.sub(lambda m: right_sided if m.group(0).lower().endswith("-sided") else right_plain, sentence)
    return (
        sentence.replace(left_sided, "right-sided")
        .replace(left_plain, "right")
        .replace(right_sided, "left-sided")
        .replace(right_plain, "left")
    )


def concept_for_sentence(sentence: str) -> str:
    for name, pattern in ANATOMY_RES.items():
        if pattern.search(sentence):
            return name
    return "other_laterality"


def extract_candidates(rows: list[dict[str, str]], report_to_split: dict[str, str]) -> list[dict[str, Any]]:
    candidates = []
    for row in rows:
        report_id = row["sample_id"]
        for sent_index, sentence in enumerate(sentence_split(row["caption"])):
            label = laterality_label(sentence)
            if label is None:
                continue
            candidates.append(
                {
                    "candidate_id": f"openi::{report_id}::sent_{sent_index:03d}::laterality",
                    "report_id": report_id,
                    "image_filename": row.get("image_filename"),
                    "split": report_to_split[report_id],
                    "concept": concept_for_sentence(sentence),
                    "label": label,
                    "text": sentence,
                    "normalized_text": normalize_text(sentence),
                }
            )
    return candidates


def load_training_texts(grouping_manifest_json: Path) -> set[str]:
    payload = json.loads(grouping_manifest_json.read_text(encoding="utf-8"))
    texts = set()
    for record in payload["records"]:
        for side in ("orig", "pert"):
            texts.add(normalize_text(record["texts"][side]))
    return texts


def overlap_audit(
    candidates: list[dict[str, Any]],
    training_texts: set[str],
    grouping_manifest_json: Path,
) -> dict[str, Any]:
    exact = [c for c in candidates if c["normalized_text"] in training_texts]
    train_grams = []
    for text in training_texts:
        toks = text.split()
        if len(toks) >= 5:
            train_grams.append(set(tuple(toks[i : i + 5]) for i in range(len(toks) - 4)))
    max_jaccards = []
    for candidate in candidates[:5000]:
        toks = candidate["normalized_text"].split()
        grams = set(tuple(toks[i : i + 5]) for i in range(max(0, len(toks) - 4)))
        if not grams or not train_grams:
            max_jaccards.append(0.0)
            continue
        max_jaccards.append(max(len(grams & tg) / len(grams | tg) for tg in train_grams))
    return {
        "training_text_source": str(grouping_manifest_json),
        "n_candidates_checked": len(candidates),
        "exact_normalized_overlap_count": len(exact),
        "exact_overlap_candidate_ids": [c["candidate_id"] for c in exact[:20]],
        "max_5gram_jaccard_checked": len(max_jaccards),
        "max_5gram_jaccard_max": max(max_jaccards) if max_jaccards else 0.0,
        "max_5gram_jaccard_q95": float(sorted(max_jaccards)[int(0.95 * (len(max_jaccards) - 1))]) if max_jaccards else 0.0,
    }


def build_minimal_pairs(candidates: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    ordered = list(candidates)
    rng.shuffle(ordered)
    pairs = []
    for cand in ordered:
        swapped_text = swap_laterality(cand["text"])
        swapped_label = "right" if cand["label"] == "left" else "left"
        if swapped_text == cand["text"] or laterality_label(swapped_text) != swapped_label:
            continue
        left_item = cand if cand["label"] == "left" else {
            **cand,
            "candidate_id": cand["candidate_id"] + "::synthetic_left",
            "label": "left",
            "text": swapped_text,
            "normalized_text": normalize_text(swapped_text),
        }
        right_item = cand if cand["label"] == "right" else {
            **cand,
            "candidate_id": cand["candidate_id"] + "::synthetic_right",
            "label": "right",
            "text": swapped_text,
            "normalized_text": normalize_text(swapped_text),
        }
        pairs.append({"split": cand["split"], "concept": cand["concept"], "left": left_item, "right": right_item})
    return pairs


def remove_cross_split_text_pairs(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    text_splits: dict[str, set[str]] = defaultdict(set)
    for pair in pairs:
        for label in ("left", "right"):
            text_splits[normalize_text(pair[label]["text"])].add(pair["split"])
    crossing_texts = {text for text, splits in text_splits.items() if len(splits) > 1}
    return [
        pair
        for pair in pairs
        if not any(normalize_text(pair[label]["text"]) in crossing_texts for label in ("left", "right"))
    ]


def downsample_pairs(
    pairs: list[dict[str, Any]],
    seed: int,
    max_train_pairs: int,
    max_dev_pairs: int,
    max_test_pairs: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    caps = {"train": max_train_pairs, "dev": max_dev_pairs, "test": max_test_pairs}
    output = []
    for split, cap in caps.items():
        split_pairs = [p for p in pairs if p["split"] == split]
        by_concept: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for pair in split_pairs:
            by_concept[pair["concept"]].append(pair)
        chosen = []
        while len(chosen) < cap:
            progressed = False
            for concept in sorted(by_concept):
                if by_concept[concept] and len(chosen) < cap:
                    chosen.append(by_concept[concept].pop())
                    progressed = True
            if not progressed:
                break
        rng.shuffle(chosen)
        output.extend(chosen)
    return output


def pair_to_examples(pair: dict[str, Any], pair_index: int) -> list[dict[str, Any]]:
    global_pair_id = f"openi_laterality_minimal_pair::{pair['concept']}::{pair_index:05d}"
    template_id = f"openi::laterality_minimal_pair::{pair['concept']}"
    examples = []
    for label in ("left", "right"):
        item = pair[label]
        examples.append(
            {
                "example_id": item["candidate_id"],
                "global_pair_id": global_pair_id,
                "template_id": template_id,
                "task": "external_laterality_state",
                "label": label,
                "split": pair["split"],
                "family": "openi_radiology_laterality",
                "concept": pair["concept"],
                "report_id": item["report_id"],
                "source_sentence_id": item["candidate_id"],
                "text": item["text"],
                "leakage_groups": {
                    "report_id": item["report_id"],
                    "template": template_id,
                    "source_sentence_id": item["candidate_id"],
                    "source_pair": global_pair_id,
                },
            }
        )
    return examples


def remove_probe_dedup_collisions(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str, str]] = set()
    kept = []
    for pair in pairs:
        template_id = f"openi::laterality_minimal_pair::{pair['concept']}"
        keys = {
            (pair["split"], template_id, pair[label]["text"], label)
            for label in ("left", "right")
        }
        if seen & keys:
            continue
        seen.update(keys)
        kept.append(pair)
    return kept


def validate_examples(examples: list[dict[str, Any]]) -> dict[str, Any]:
    report_splits: dict[str, set[str]] = defaultdict(set)
    pair_labels: dict[str, set[str]] = defaultdict(set)
    for example in examples:
        report_splits[example["report_id"]].add(example["split"])
        pair_labels[example["global_pair_id"]].add(example["label"])
    report_leaks = {rid: sorted(splits) for rid, splits in report_splits.items() if len(splits) != 1}
    bad_pairs = [pair_id for pair_id, labels in pair_labels.items() if labels != {"left", "right"}]
    if report_leaks:
        raise ValueError(f"Report split leakage detected: {list(report_leaks.items())[:5]}")
    if bad_pairs:
        raise ValueError(f"Pairs without both labels: {bad_pairs[:5]}")
    split_summary = {}
    for split in ("train", "dev", "test"):
        split_examples = [e for e in examples if e["split"] == split]
        split_summary[split] = {
            "n_examples": len(split_examples),
            "n_pairs": len({e["global_pair_id"] for e in split_examples}),
            "n_reports": len({e["report_id"] for e in split_examples}),
            "label_counts": dict(Counter(e["label"] for e in split_examples)),
            "concept_counts": dict(Counter(e["concept"] for e in split_examples)),
        }
    return {"leakage_checks_passed": True, "split_summary": split_summary}


def write_payload(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# E3b OpenI External Laterality Dataset",
        "",
        f"Status: `{payload['status']}`",
        f"Examples: {payload['summary']['n_examples']}",
        f"Pairs: {payload['summary']['n_pairs']}",
        f"Reports: {payload['summary']['n_reports']}",
        "",
        "## Split Summary",
        "",
        "| Split | examples | pairs | reports | left | right |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split, block in payload["validation"]["split_summary"].items():
        labels = block["label_counts"]
        lines.append(
            f"| {split} | {block['n_examples']} | {block['n_pairs']} | {block['n_reports']} | "
            f"{labels.get('left', 0)} | {labels.get('right', 0)} |"
        )
    lines.extend(
        [
            "",
            "## Externality Audit",
            "",
            f"- exact normalized overlap with E3 600-pair source: `{payload['overlap_audit']['exact_normalized_overlap_count']}`",
            f"- max 5-gram Jaccard q95: `{payload['overlap_audit']['max_5gram_jaccard_q95']:.4f}`",
            f"- max 5-gram Jaccard max: `{payload['overlap_audit']['max_5gram_jaccard_max']:.4f}`",
            "",
            "## Pair Construction",
            "",
            "Each pair is generated by a strict whole-word `left`/`right` or `left-sided`/`right-sided` swap. Sentences with bilateral, both-sided, or mixed left/right cues are excluded.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build E3b OpenI external laterality dataset")
    parser.add_argument("--openi-index-csv", type=Path, required=True)
    parser.add_argument("--e3-grouping-manifest-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-pairs", type=int, default=300)
    parser.add_argument("--max-dev-pairs", type=int, default=80)
    parser.add_argument("--max-test-pairs", type=int, default=80)
    args = parser.parse_args()

    rows = read_index(args.openi_index_csv)
    report_to_split = split_reports([row["sample_id"] for row in rows], args.seed)
    candidates = extract_candidates(rows, report_to_split)
    training_texts = load_training_texts(args.e3_grouping_manifest_json)
    overlap = overlap_audit(candidates, training_texts, args.e3_grouping_manifest_json)
    pairs = build_minimal_pairs(candidates, args.seed)
    pairs = remove_cross_split_text_pairs(pairs)
    pairs = downsample_pairs(pairs, args.seed, args.max_train_pairs, args.max_dev_pairs, args.max_test_pairs)
    pairs_before_probe_dedup = len(pairs)
    pairs = remove_probe_dedup_collisions(pairs)
    examples = []
    for index, pair in enumerate(pairs):
        examples.extend(pair_to_examples(pair, index))
    validation = validate_examples(examples)
    payload = {
        "experiment": "E3b_external_laterality",
        "dataset_kind": "laterality_minimal_pair",
        "task": "external_laterality_state",
        "status": "openi_external_laterality_split_frozen",
        "openi_index_csv": str(args.openi_index_csv),
        "openi_index_sha256": sha256_file(args.openi_index_csv),
        "e3_grouping_manifest_json": str(args.e3_grouping_manifest_json),
        "e3_grouping_manifest_sha256": sha256_file(args.e3_grouping_manifest_json),
        "seed": args.seed,
        "source": {
            "name": "OpenI NLMCXR / Indiana University chest X-ray reports",
            "sections": "FINDINGS + IMPRESSION as prepared in OpenI/index.csv",
            "images_used": False,
        },
        "split_policy": {
            "split_before_sentence_extraction": True,
            "group_unit": "report_id/sample_id",
            "ratios": REPORT_SPLITS,
        },
        "label_rule": {
            "labels": ["left", "right"],
            "positive_class_for_margin": "right",
            "baseline_class_for_margin": "left",
            "excluded_patterns": EXCLUDE_RE.pattern,
            "left_regex": LEFT_RE.pattern,
            "right_regex": RIGHT_RE.pattern,
        },
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
            "probe": "logistic_regression_balanced_class_weight",
            "probe_seeds": [42, 43, 44, 45, 46],
            "primary_endpoint": "paired probability-margin L20, right minus left",
            "cluster_unit_for_inference": "report_id",
        },
        "summary": {
            "n_candidates": len(candidates),
            "n_examples": len(examples),
            "n_pairs": len({e["global_pair_id"] for e in examples}),
            "n_reports": len({e["report_id"] for e in examples}),
            "pairs_removed_for_probe_dedup_collisions": pairs_before_probe_dedup - len(pairs),
        },
        "validation": validation,
        "overlap_audit": overlap,
        "examples": examples,
    }
    write_payload(args.output_json, payload)
    write_markdown(args.output_md, payload)
    print(f"Saved JSON -> {args.output_json}")
    print(f"Saved Markdown -> {args.output_md}")


if __name__ == "__main__":
    main()
