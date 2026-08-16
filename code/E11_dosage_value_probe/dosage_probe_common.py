"""Shared constants and helpers for held-out dosage numeric probe."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np

EXPERIMENT = "exp_dosage"
DATASET_KIND = "heldout_dosage_numeric_probe"
TASK = "dosage_numeric_probe"
STATUS_FROZEN = "heldout_dosage_numeric_probe_frozen"

LABEL_CRITICAL = "critical_numeric_change"
LABEL_NUISANCE = "nuisance"
POSITIVE_LABELS = {LABEL_CRITICAL}
NEGATIVE_LABELS = {LABEL_NUISANCE}

REPRESENTATIONS = ("hidden", "sae_standard_code", "sae_vreg_code")

PROFILE_RUNS: dict[str, tuple[str, str]] = {
    "gpt2": ("gpt2_standard_joint16_owt", "gpt2_vreg_joint16_owt"),
    "gemma-2-2b": ("gemma-2-2b_standard_joint16_owt", "gemma-2-2b_vreg_joint16_owt"),
    "qwen-2.5-3b": ("qwen-2.5-3b_standard_joint16_owt", "qwen-2.5-3b_vreg_joint16_owt"),
}

C_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]
PROBE_SEED = 42


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def label_to_int(label: str) -> int:
    if label == LABEL_CRITICAL:
        return 1
    if label == LABEL_NUISANCE:
        return 0
    raise ValueError(f"Unknown label: {label!r}")


def split_summary(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"splits": {}, "labels": {}, "families": {}}
    for split in ("train", "dev", "test"):
        subset = [p for p in pairs if p["split"] == split]
        out["splits"][split] = {
            "n_pairs": len(subset),
            "n_templates": len({p["template_id"] for p in subset}),
            "n_clusters": len({p["template_cluster_id"] for p in subset}),
        }
        out["labels"][split] = {
            LABEL_CRITICAL: sum(1 for p in subset if p["label"] == LABEL_CRITICAL),
            LABEL_NUISANCE: sum(1 for p in subset if p["label"] == LABEL_NUISANCE),
        }
        families: dict[str, int] = {}
        for pair in subset:
            families[pair["family"]] = families.get(pair["family"], 0) + 1
        out["families"][split] = families
    return out


def validate_pairs(pairs: list[dict[str, Any]]) -> None:
    required = {
        "pair_id",
        "template_id",
        "template_cluster_id",
        "split",
        "family",
        "label",
        "control_type",
        "text_left",
        "text_right",
        "numeric_left",
        "numeric_right",
    }
    seen_ids: set[str] = set()
    for pair in pairs:
        missing = required - set(pair)
        if missing:
            raise ValueError(f"Pair {pair.get('pair_id')} missing fields: {sorted(missing)}")
        if pair["pair_id"] in seen_ids:
            raise ValueError(f"Duplicate pair_id: {pair['pair_id']}")
        seen_ids.add(pair["pair_id"])
        if pair["label"] not in POSITIVE_LABELS | NEGATIVE_LABELS:
            raise ValueError(f"Bad label on {pair['pair_id']}: {pair['label']}")
        if pair["split"] not in {"train", "dev", "test"}:
            raise ValueError(f"Bad split on {pair['pair_id']}: {pair['split']}")

    test_templates = {p["template_id"] for p in pairs if p["split"] == "test"}
    train_templates = {p["template_id"] for p in pairs if p["split"] == "train"}
    dev_templates = {p["template_id"] for p in pairs if p["split"] == "dev"}
    if test_templates & train_templates:
        raise ValueError(f"Template leakage train/test: {sorted(test_templates & train_templates)[:5]}")
    if test_templates & dev_templates:
        raise ValueError(f"Template leakage dev/test: {sorted(test_templates & dev_templates)[:5]}")


def pair_indices_by_split(pairs: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    by_split: dict[str, list[int]] = {"train": [], "dev": [], "test": []}
    for index, pair in enumerate(pairs):
        by_split[pair["split"]].append(index)
    return {split: np.asarray(idxs, dtype=np.int64) for split, idxs in by_split.items()}


def labels_array(pairs: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([label_to_int(p["label"]) for p in pairs], dtype=np.int64)


def lower_tail_mean(values: np.ndarray, frac: float = 0.20) -> float:
    ordered = np.sort(values.astype(np.float64))
    k = max(1, int(np.ceil(frac * len(ordered))))
    return float(np.mean(ordered[:k]))


def absolute_code_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.linalg.norm(right - left, axis=1)


def relative_code_distance(left: np.ndarray, right: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    denom = np.linalg.norm(left, axis=1) + eps
    return absolute_code_distance(left, right) / denom


def pair_feature_abs_diff(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.abs(right - left)


def stratified_shuffle(items: list[Any], seed: int) -> list[Any]:
    rng = random.Random(seed)
    ordered = list(items)
    rng.shuffle(ordered)
    return ordered
