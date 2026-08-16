#!/usr/bin/env python3
"""Build E3b external OpenI negation probe datasets.

This is split-before-extraction and model-free. It creates external, SAE-training-unseen
probe examples from OpenI report text. Images are not used.
"""

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

NEGATION_RE = re.compile(
    r"\b(no|not|never|without|denies|denied|negative|absent|cannot|can't|isn't|aren't|"
    r"doesn't|don't|didn't|hasn't|haven't|hadn't)\b",
    re.I,
)
UNCERTAINTY_RE = re.compile(
    r"\b(questionable|possible|possibly|probable|probably|may|might|could|suggest|"
    r"suspicious|suspect|unknown|if|differential|cannot exclude|difficult to exclude|versus|vs\.?)\b",
    re.I,
)
REPORT_SPLITS = {"train": 0.70, "dev": 0.15, "test": 0.15}
RANDOM_LABEL_SEEDS = [11, 23, 37, 53, 71]
REAL_LABEL_PROBE_SEED = 42

CONCEPT_PATTERNS = {
    "pleural_effusion": r"\b(pleural effusion|effusion|effusions)\b",
    "pneumothorax": r"\bpneumothorax\b",
    "consolidation": r"\b(consolidation|airspace disease|air space disease|airspace opacity|infiltrate)\b",
    "edema": r"\b(edema|pulmonary edema)\b",
    "cardiomegaly": r"\b(cardiomegaly|cardiac enlargement|enlarged cardiac silhouette)\b",
    "atelectasis": r"\batelectasis\b",
    "opacity": r"\b(opacity|opacities)\b",
    "nodule": r"\b(nodule|nodular density|nodular opacity|mass)\b",
    "fracture": r"\b(fracture|fractures)\b",
    "pneumonia": r"\bpneumonia\b",
}
CONCEPT_RES = {name: re.compile(pattern, re.I) for name, pattern in CONCEPT_PATTERNS.items()}


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


def concepts_for_sentence(sentence: str) -> list[str]:
    return [name for name, pattern in CONCEPT_RES.items() if pattern.search(sentence)]


def sentence_allowed(sentence: str) -> bool:
    if "XXXX" in sentence:
        return False
    if UNCERTAINTY_RE.search(sentence):
        return False
    return True


def label_for_concept(sentence: str, concept: str) -> str | None:
    if not sentence_allowed(sentence):
        return None
    match = CONCEPT_RES[concept].search(sentence)
    if match is None:
        return None
    before = sentence[: match.start()].lower()
    after = sentence[match.end() :].lower()
    # Concept-specific negation: cue must govern the target mention, not another later clause.
    if re.search(r"\b(no|without|not|absent|negative for|no evidence of|denies|denied)\b", before):
        return "negated"
    if re.match(
        r"\s*(is|are|was|were)?\s*(not\b|absent\b|negative\b|excluded\b|"
        r"not\s+(seen|identified|demonstrated)\b)",
        after,
    ):
        return "negated"
    return "affirmed"


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
    test_ids = set(ordered[n_train + n_dev :])
    return {
        report_id: "train" if report_id in train_ids else "dev" if report_id in dev_ids else "test"
        for report_id in ordered
    }


def extract_candidates(rows: list[dict[str, str]], report_to_split: dict[str, str]) -> list[dict[str, Any]]:
    candidates = []
    for row in rows:
        report_id = row["sample_id"]
        for sent_index, sentence in enumerate(sentence_split(row["caption"])):
            concepts = concepts_for_sentence(sentence)
            if not concepts:
                continue
            for concept in concepts:
                label = label_for_concept(sentence, concept)
                if label is None:
                    continue
                candidates.append(
                    {
                        "candidate_id": f"openi::{report_id}::sent_{sent_index:03d}::{concept}",
                        "report_id": report_id,
                        "image_filename": row.get("image_filename"),
                        "split": report_to_split[report_id],
                        "concept": concept,
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
    # Lightweight 5-gram Jaccard screen for near-duplicate audit.
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


def build_natural_pairs(candidates: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    by_key: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for c in candidates:
        by_key[(c["split"], c["concept"], c["label"])].append(c)
    pairs = []
    for split in ("train", "dev", "test"):
        for concept in sorted(CONCEPT_PATTERNS):
            affirmed = by_key[(split, concept, "affirmed")]
            negated = by_key[(split, concept, "negated")]
            rng.shuffle(affirmed)
            rng.shuffle(negated)
            used_reports: set[str] = set()
            n = min(len(affirmed), len(negated))
            made = 0
            for aff in affirmed:
                neg = next(
                    (
                        cand
                        for cand in negated
                        if cand["report_id"] != aff["report_id"]
                        and cand["candidate_id"] not in used_reports
                    ),
                    None,
                )
                if neg is None or made >= n:
                    continue
                used_reports.add(neg["candidate_id"])
                pairs.append({"split": split, "concept": concept, "affirmed": aff, "negated": neg})
                made += 1
    return pairs


def minimal_transform(sentence: str) -> str | None:
    rules = [
        (r"^There is no evidence of (.+)\.$", r"There is evidence of \1."),
        (r"^There is no (.+)\.$", r"There is \1."),
        (r"^There are no (.+)\.$", r"There are \1."),
        (r"^No evidence of (.+)\.$", r"Evidence of \1 is present."),
        (r"^No (.+)\.$", r"\1 is present."),
        (r"^There is not (.+)\.$", r"There is \1."),
        (r"^There are not (.+)\.$", r"There are \1."),
    ]
    for pattern, repl in rules:
        out = re.sub(pattern, repl, sentence, flags=re.I)
        if out != sentence and not NEGATION_RE.search(out) and 5 <= len(out.split()) <= 35:
            return out[0].upper() + out[1:]
    return None


def build_minimal_pairs(candidates: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    negated = [c for c in candidates if c["label"] == "negated"]
    rng.shuffle(negated)
    pairs = []
    for cand in negated:
        affirmed = minimal_transform(cand["text"])
        if affirmed is None:
            continue
        pairs.append(
            {
                "split": cand["split"],
                "concept": cand["concept"],
                "negated": cand,
                "affirmed": {
                    **cand,
                    "candidate_id": cand["candidate_id"] + "::synthetic_affirmed",
                    "label": "affirmed",
                    "text": affirmed,
                    "normalized_text": normalize_text(affirmed),
                },
                "manual_audit_required": True,
            }
        )
    return pairs


def pair_to_examples(pair: dict[str, Any], pair_index: int, dataset_kind: str) -> list[dict[str, Any]]:
    global_pair_id = f"openi_{dataset_kind}::{pair['concept']}::{pair_index:05d}"
    template_id = f"openi::{dataset_kind}::{pair['concept']}"
    examples = []
    for label in ("affirmed", "negated"):
        item = pair[label]
        examples.append(
            {
                "example_id": item["candidate_id"],
                "global_pair_id": global_pair_id,
                "template_id": template_id,
                "task": "external_negation_state",
                "label": label,
                "split": pair["split"],
                "family": "openi_radiology",
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


def candidate_to_example(candidate: dict[str, Any], index: int) -> dict[str, Any]:
    global_id = f"openi_natural_classification::{index:06d}"
    return {
        "example_id": candidate["candidate_id"],
        "global_pair_id": global_id,
        "template_id": f"openi::natural_classification::{candidate['concept']}",
        "task": "external_negation_state",
        "label": candidate["label"],
        "split": candidate["split"],
        "family": "openi_radiology",
        "concept": candidate["concept"],
        "report_id": candidate["report_id"],
        "source_sentence_id": candidate["candidate_id"],
        "text": candidate["text"],
        "leakage_groups": {
            "report_id": candidate["report_id"],
            "template": f"openi::natural_classification::{candidate['concept']}",
            "source_sentence_id": candidate["candidate_id"],
            "source_pair": global_id,
        },
    }


def remove_cross_split_text_pairs(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    text_splits: dict[str, set[str]] = defaultdict(set)
    for pair in pairs:
        for label in ("affirmed", "negated"):
            text_splits[normalize_text(pair[label]["text"])].add(pair["split"])
    crossing_texts = {text for text, splits in text_splits.items() if len(splits) > 1}
    if not crossing_texts:
        return pairs
    filtered = []
    for pair in pairs:
        if any(normalize_text(pair[label]["text"]) in crossing_texts for label in ("affirmed", "negated")):
            continue
        filtered.append(pair)
    return filtered


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


def validate_examples(examples: list[dict[str, Any]], *, require_paired: bool) -> dict[str, Any]:
    report_splits: dict[str, set[str]] = defaultdict(set)
    pair_labels: dict[str, set[str]] = defaultdict(set)
    for example in examples:
        report_splits[example["report_id"]].add(example["split"])
        pair_labels[example["global_pair_id"]].add(example["label"])
    report_leaks = {rid: sorted(splits) for rid, splits in report_splits.items() if len(splits) != 1}
    bad_pairs = [
        pair_id
        for pair_id, labels in pair_labels.items()
        if require_paired and labels != {"affirmed", "negated"}
    ]
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
    return {
        "leakage_checks_passed": True,
        "split_summary": split_summary,
    }


def write_payload(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# E3b OpenI External Negation Dataset — {payload['dataset_kind']}",
        "",
        f"Status: `{payload['status']}`",
        f"Examples: {payload['summary']['n_examples']}",
        f"Pairs: {payload['summary']['n_pairs']}",
        f"Reports: {payload['summary']['n_reports']}",
        "",
        "## Split Summary",
        "",
        "| Split | examples | pairs | reports | affirmed | negated |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split, block in payload["validation"]["split_summary"].items():
        labels = block["label_counts"]
        lines.append(
            f"| {split} | {block['n_examples']} | {block['n_pairs']} | {block['n_reports']} | "
            f"{labels.get('affirmed', 0)} | {labels.get('negated', 0)} |"
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
            "## License / Use Note",
            "",
            "OpenI/NLMCXR licensing is item/source dependent. E3b stores and reports aggregate text-probe statistics only; images are not redistributed or used.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_dataset(
    rows: list[dict[str, str]],
    candidates: list[dict[str, Any]],
    dataset_kind: str,
    seed: int,
) -> list[dict[str, Any]]:
    if dataset_kind == "natural_paired":
        pairs = build_natural_pairs(candidates, seed)
        pairs = remove_cross_split_text_pairs(pairs)
        pairs = downsample_pairs(pairs, seed, max_train_pairs=300, max_dev_pairs=80, max_test_pairs=80)
        examples = []
        for index, pair in enumerate(pairs):
            examples.extend(pair_to_examples(pair, index, dataset_kind))
    elif dataset_kind == "minimal_pair":
        pairs = build_minimal_pairs(candidates, seed)
        pairs = remove_cross_split_text_pairs(pairs)
        pairs = downsample_pairs(pairs, seed, max_train_pairs=300, max_dev_pairs=80, max_test_pairs=80)
        examples = []
        for index, pair in enumerate(pairs):
            examples.extend(pair_to_examples(pair, index, dataset_kind))
    elif dataset_kind == "natural_classification":
        text_splits: dict[str, set[str]] = defaultdict(set)
        for candidate in candidates:
            text_splits[candidate["normalized_text"]].add(candidate["split"])
        crossing = {text for text, splits in text_splits.items() if len(splits) > 1}
        filtered = [c for c in candidates if c["normalized_text"] not in crossing]
        examples = [candidate_to_example(candidate, index) for index, candidate in enumerate(filtered)]
    else:
        raise ValueError(f"Unknown dataset kind: {dataset_kind}")
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Build E3b OpenI external negation dataset")
    parser.add_argument("--openi-index-csv", type=Path, required=True)
    parser.add_argument("--e3-grouping-manifest-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument(
        "--dataset-kind",
        choices=["natural_paired", "minimal_pair", "natural_classification"],
        required=True,
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = read_index(args.openi_index_csv)
    report_to_split = split_reports([row["sample_id"] for row in rows], args.seed)
    candidates = extract_candidates(rows, report_to_split)
    training_texts = load_training_texts(args.e3_grouping_manifest_json)
    overlap = overlap_audit(candidates, training_texts, args.e3_grouping_manifest_json)
    examples = build_dataset(rows, candidates, args.dataset_kind, args.seed)
    validation = validate_examples(examples, require_paired=args.dataset_kind != "natural_classification")
    payload = {
        "experiment": "E3b_external_negation",
        "dataset_kind": args.dataset_kind,
        "task": "external_negation_state",
        "status": "openi_external_negation_split_frozen",
        "openi_index_csv": str(args.openi_index_csv),
        "openi_index_sha256": sha256_file(args.openi_index_csv),
        "e3_grouping_manifest_json": str(args.e3_grouping_manifest_json),
        "e3_grouping_manifest_sha256": sha256_file(args.e3_grouping_manifest_json),
        "seed": args.seed,
        "source": {
            "name": "OpenI NLMCXR / Indiana University chest X-ray reports",
            "sections": "FINDINGS + IMPRESSION as prepared in OpenI/index.csv",
            "images_used": False,
            "license_note": (
                "OpenI does not provide a single blanket reuse license; individual images/source "
                "articles have item-level terms. E3b uses text-derived aggregate statistics and "
                "does not redistribute images."
            ),
        },
        "split_policy": {
            "split_before_sentence_extraction": True,
            "group_unit": "report_id/sample_id",
            "ratios": REPORT_SPLITS,
        },
        "label_rule": {
            "labels": ["affirmed", "negated"],
            "negation_regex": NEGATION_RE.pattern,
            "uncertainty_exclusion_regex": UNCERTAINTY_RE.pattern,
            "concepts": CONCEPT_PATTERNS,
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
            "primary_comparison": "sae_vreg_code_vs_sae_standard_code",
            "classifier": "linear_probe",
            "standardize_features": "train_only",
            "hyperparameter_selection": "dev_only",
            "test_evaluation": "once",
            "real_label_probe_seed": REAL_LABEL_PROBE_SEED,
            "probe_seeds": [REAL_LABEL_PROBE_SEED],
            "metrics": ["balanced_accuracy", "auroc", "macro_f1", "pair_margin_l20"],
        },
        "summary": {
            "n_reports_input": len(rows),
            "n_candidates": len(candidates),
            "n_examples": len(examples),
            "n_pairs": len({e["global_pair_id"] for e in examples}),
            "n_reports": len({e["report_id"] for e in examples}),
        },
        "overlap_audit": overlap,
        "validation": validation,
        "examples": examples,
    }
    write_payload(args.output_json, payload)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    write_markdown(args.output_md, payload)
    print(f"Saved JSON -> {args.output_json}")
    print(f"Saved Markdown -> {args.output_md}")


if __name__ == "__main__":
    main()
