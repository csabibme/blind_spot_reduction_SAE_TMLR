#!/usr/bin/env python3
"""Train frozen E3 negation probes on held-out representations.

Uses a frozen negation task split. Feature extraction loads LM/SAE checkpoints; probe
training is lightweight and reuses cached features when available.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import unicodedata
from pathlib import Path
from typing import Any
from collections import Counter

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

REVISION_ROOT = Path(__file__).resolve().parents[1]
E1_ROOT = REVISION_ROOT / "E1_absolute_sensitivity"
for path in (REVISION_ROOT, E1_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from shared.path_registry import checkpoint_dir, load_manifest, sae_scaling_root  # noqa: E402
from eval_core import (  # noqa: E402
    clear_device_cache,
    load_sae,
    setup_sae_scaling_imports,
)

PROFILE_RUNS: dict[str, tuple[str, str]] = {
    "gpt2": ("gpt2_standard_joint16_owt", "gpt2_vreg_joint16_owt"),
    "gemma-2-2b": ("gemma-2-2b_standard_joint16_owt", "gemma-2-2b_vreg_joint16_owt"),
    "qwen-2.5-3b": ("qwen-2.5-3b_standard_joint16_owt", "qwen-2.5-3b_vreg_joint16_owt"),
}

REPRESENTATIONS = (
    "hidden",
    "sae_standard_code",
    "sae_vreg_code",
    "sae_standard_reconstruction",
    "sae_vreg_reconstruction",
)

DEFAULT_GEMMA_E1R_VREG = (
    REVISION_ROOT
    / "gemma_true_last_protocol"
    / "checkpoints"
    / "gemma-2-2b"
    / "joint"
    / "vreg_joint16_owt_true_last"
)

ALLOWED_SPLIT_STATUSES = {
    "negation_task_split_frozen",
    "negation_stress_split_frozen",
    "openi_external_negation_split_frozen",
    "openi_external_negation_split_frozen_after_manual_audit",
    "openi_external_laterality_split_frozen",
}

C_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]
REAL_LABEL_PROBE_SEED = 42


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def checkpoint_sae_file(checkpoint: Path) -> Path:
    sae_path = checkpoint / "sae.pt"
    if not sae_path.is_file():
        raise FileNotFoundError(f"Missing SAE checkpoint file: {sae_path}")
    return sae_path


def checkpoint_metadata(
    checkpoint_id: str,
    checkpoint_path: Path,
    manifest: dict[str, Any],
    *,
    override_used: bool,
) -> dict[str, Any]:
    sae_path = checkpoint_sae_file(checkpoint_path)
    manifest_sha = None
    if not override_used and checkpoint_id in manifest.get("checkpoints", {}):
        manifest_sha = manifest["checkpoints"][checkpoint_id].get("sha256_sae_pt")
    actual_sha = sha256_file(sae_path)
    if manifest_sha and actual_sha.lower() != manifest_sha.lower():
        raise ValueError(
            f"Checkpoint SHA mismatch for {checkpoint_id}: expected {manifest_sha}, got {actual_sha}"
        )
    meta_path = checkpoint_path / "meta.json"
    return {
        "checkpoint_id": checkpoint_id,
        "checkpoint_path": str(checkpoint_path),
        "sae_path": str(sae_path),
        "sha256_sae_pt": actual_sha,
        "manifest_sha256_sae_pt": manifest_sha,
        "override_used": override_used,
        "meta_path": str(meta_path) if meta_path.is_file() else None,
    }


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def extraction_protocol_for_profile(_profile: str) -> str:
    return "true_last"


def feature_cache_path(
    cache_dir: Path,
    profile: str,
    task_split_sha256: str,
    texts: list[str],
) -> Path:
    text_digest = sha256_text("\n".join(texts))
    return cache_dir / f"{profile}_{task_split_sha256[:16]}_{text_digest[:16]}.npz"


def normalized_cache_meta(meta: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(meta)
    for key in ("standard_checkpoint", "vreg_checkpoint"):
        if key not in normalized:
            continue
        path = Path(str(normalized[key])).expanduser()
        normalized[key] = unicodedata.normalize("NFC", str(path.resolve()))
    return normalized


def load_feature_cache(path: Path, expected_meta: dict[str, Any]) -> dict[str, np.ndarray] | None:
    if not path.is_file():
        return None
    cached = np.load(path, allow_pickle=True)
    meta_raw = cached["metadata"].item()
    if normalized_cache_meta(meta_raw) != normalized_cache_meta(expected_meta):
        return None
    return {name: cached[name] for name in REPRESENTATIONS}


def deduplicate_examples(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str, str]] = set()
    unique: list[dict[str, Any]] = []
    for example in examples:
        key = (
            example["split"],
            example["template_id"],
            example["text"],
            example["label"],
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(example)
    return unique


def deduplication_summary(
    raw_examples: list[dict[str, Any]],
    examples: list[dict[str, Any]],
) -> dict[str, Any]:
    split_counts = {split: 0 for split in ("train", "dev", "test")}
    split_labels: dict[str, Counter[str]] = {split: Counter() for split in ("train", "dev", "test")}
    for example in examples:
        split = example["split"]
        split_counts[split] += 1
        split_labels[split][example["label"]] += 1
    return {
        "raw_examples": len(raw_examples),
        "deduplicated_examples": len(examples),
        "deduplication_unit": "split_template_text_label",
        "train_examples": split_counts["train"],
        "dev_examples": split_counts["dev"],
        "test_examples": split_counts["test"],
        "train_labels": dict(split_labels["train"]),
        "dev_labels": dict(split_labels["dev"]),
        "test_labels": dict(split_labels["test"]),
    }


def save_feature_cache(path: Path, features: dict[str, np.ndarray], metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    arrays = {name: features[name] for name in REPRESENTATIONS}
    arrays["metadata"] = np.array(metadata, dtype=object)
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    tmp.replace(path)


def collect_hidden_texts(
    lm,
    tok,
    texts: list[str],
    layer: int,
    device: str,
    batch_size: int,
    extraction_protocol: str,
    max_length: int,
) -> torch.Tensor:
    if extraction_protocol != "true_last":
        raise ValueError(f"Unsupported extraction protocol: {extraction_protocol}")
    from activations import last_token_hidden_true_last

    bs = batch_size if batch_size > 0 else len(texts)
    return last_token_hidden_true_last(
        lm, tok, texts, layer, device, max_length=max_length, batch_size=bs
    ).float().cpu()


@torch.no_grad()
def encode_representations(
    hidden: torch.Tensor,
    sae_standard,
    sae_vreg,
) -> dict[str, np.ndarray]:
    device = next(sae_standard.parameters()).device
    h = hidden.to(device)
    z_std = sae_standard.encode(h).float().cpu().numpy()
    z_vreg = sae_vreg.encode(h).float().cpu().numpy()
    x_std = sae_standard.decode(torch.tensor(z_std, device=device)).float().cpu().numpy()
    x_vreg = sae_vreg.decode(torch.tensor(z_vreg, device=device)).float().cpu().numpy()
    return {
        "hidden": hidden.numpy(),
        "sae_standard_code": z_std,
        "sae_vreg_code": z_vreg,
        "sae_standard_reconstruction": x_std,
        "sae_vreg_reconstruction": x_vreg,
    }


def label_to_int(label: str) -> int:
    if label in {"negated", "right"}:
        return 1
    if label in {"affirmed", "left"}:
        return 0
    raise ValueError(f"Unknown binary probe label: {label!r}")


def split_arrays(
    examples: list[dict[str, Any]],
    features: dict[str, np.ndarray],
    representation: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    by_split: dict[str, list[int]] = {"train": [], "dev": [], "test": []}
    for index, example in enumerate(examples):
        by_split[example["split"]].append(index)
    x = features[representation]
    y = np.array([label_to_int(example["label"]) for example in examples], dtype=np.int64)
    train_idx = np.array(by_split["train"], dtype=np.int64)
    dev_idx = np.array(by_split["dev"], dtype=np.int64)
    test_idx = np.array(by_split["test"], dtype=np.int64)
    return (
        x[train_idx],
        y[train_idx],
        x[dev_idx],
        y[dev_idx],
        x[test_idx],
        y[test_idx],
        [examples[i] for i in train_idx.tolist()],
        [examples[i] for i in dev_idx.tolist()],
        [examples[i] for i in test_idx.tolist()],
    )


def fit_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_dev: np.ndarray,
    y_dev: np.ndarray,
    probe_seed: int,
) -> tuple[StandardScaler, LogisticRegression, float]:
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_dev_scaled = scaler.transform(x_dev)
    best_c = C_GRID[0]
    best_score = -1.0
    for c in C_GRID:
        model = LogisticRegression(
            C=c,
            solver="lbfgs",
            max_iter=20000,
            class_weight="balanced",
            random_state=probe_seed,
        )
        model.fit(x_train_scaled, y_train)
        preds = model.predict(x_dev_scaled)
        score = balanced_accuracy_score(y_dev, preds)
        if score > best_score:
            best_score = score
            best_c = c
    final_model = LogisticRegression(
        C=best_c,
        solver="lbfgs",
        max_iter=20000,
        class_weight="balanced",
        random_state=probe_seed,
    )
    final_model.fit(x_train_scaled, y_train)
    return scaler, final_model, best_c


def prediction_records(
    scaler: StandardScaler,
    model: LogisticRegression,
    x: np.ndarray,
    y: np.ndarray,
    split_examples: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    x_scaled = scaler.transform(x)
    preds = model.predict(x_scaled)
    prob = model.predict_proba(x_scaled)[:, 1]
    records = []
    for index, example in enumerate(split_examples):
        records.append(
            {
                "example_id": example["example_id"],
                "global_pair_id": example["global_pair_id"],
                "template_id": example["template_id"],
                "report_id": example.get("report_id"),
                "family": example["family"],
                "y_true": int(y[index]),
                "y_pred": int(preds[index]),
                "prob_negated": float(prob[index]),
                "prob_positive": float(prob[index]),
            }
        )
    return records


def evaluate_probe(
    scaler: StandardScaler,
    model: LogisticRegression,
    x: np.ndarray,
    y: np.ndarray,
    split_examples: list[dict[str, Any]],
) -> dict[str, Any]:
    x_scaled = scaler.transform(x)
    preds = model.predict(x_scaled)
    prob = model.predict_proba(x_scaled)[:, 1]
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y, preds)),
        "macro_f1": float(f1_score(y, preds, average="macro")),
        "auroc": float(roc_auc_score(y, prob)),
        "predictions": prediction_records(scaler, model, x, y, split_examples),
    }


def shuffled_train_labels(y_train: np.ndarray, seed: int) -> np.ndarray:
    rng = random.Random(seed)
    labels = y_train.tolist()
    rng.shuffle(labels)
    return np.asarray(labels, dtype=np.int64)


def profile_complete(
    profile_block: dict[str, Any],
    *,
    real_label_probe_seed: int,
    random_label_seeds: list[int],
) -> bool:
    if not profile_block:
        return False
    reps = profile_block.get("representations", {})
    if len(reps) != len(REPRESENTATIONS):
        return False
    for representation in REPRESENTATIONS:
        rep_block = reps.get(representation, {})
        real_labels = rep_block.get("real_labels", {})
        random_controls = rep_block.get("random_label_controls", {})
        if str(real_label_probe_seed) not in real_labels:
            return False
        real_entry = real_labels[str(real_label_probe_seed)]
        if "dev" not in real_entry or "test" not in real_entry:
            return False
        if "predictions" not in real_entry["dev"] or "predictions" not in real_entry["test"]:
            return False
        if set(random_controls) != {str(seed) for seed in random_label_seeds}:
            return False
        for shuffle_seed in random_label_seeds:
            control = random_controls[str(shuffle_seed)]
            if "dev" not in control or "test" not in control:
                return False
            if "predictions" not in control["dev"] or "predictions" not in control["test"]:
                return False
    return True


def ensure_output_allowed(output_json: Path, resume: bool, force: bool) -> dict[str, Any] | None:
    if not output_json.exists():
        return None
    if resume:
        return read_json(output_json)
    if force:
        return None
    raise FileExistsError(
        f"Output already exists: {output_json}. Use --resume to continue or --force to overwrite."
    )


def write_results(output_json: Path, payload: dict[str, Any]) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_json.with_suffix(output_json.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    tmp.replace(output_json)


def evaluate_profile(
    profile: str,
    examples: list[dict[str, Any]],
    example_summary: dict[str, Any],
    manifest: dict[str, Any],
    device: str,
    lm_dtype: str,
    hidden_batch_size: int,
    max_length: int,
    gemma_vreg_checkpoint: Path,
    real_label_probe_seed: int,
    random_label_seeds: list[int],
    feature_cache_dir: Path,
    task_split_sha256: str,
) -> dict[str, Any]:
    setup_sae_scaling_imports(sae_scaling_root(manifest))
    from lm_loader import load_model_and_tokenizer

    texts = [example["text"] for example in examples]
    model_cfg = manifest["models"][profile]
    layer = int(model_cfg["hf_hidden_state_index"])
    std_id, vreg_id = PROFILE_RUNS[profile]
    std_ckpt = checkpoint_dir(std_id, manifest)
    vreg_ckpt = gemma_vreg_checkpoint if profile == "gemma-2-2b" else checkpoint_dir(vreg_id, manifest)
    extraction_protocol = extraction_protocol_for_profile(profile)
    vreg_override = profile == "gemma-2-2b"

    std_meta = checkpoint_metadata(std_id, std_ckpt, manifest, override_used=False)
    vreg_meta = checkpoint_metadata(
        vreg_id if not vreg_override else "gemma_e1r_vreg_override",
        vreg_ckpt,
        manifest,
        override_used=vreg_override,
    )

    cache_meta = {
        "profile": profile,
        "task_split_sha256": task_split_sha256,
        "model_id": model_cfg["model_id"],
        "layer": layer,
        "dtype": lm_dtype,
        "hidden_batch_size": hidden_batch_size,
        "max_length": max_length,
        "extraction_protocol": extraction_protocol,
        "standard_checkpoint": str(std_ckpt),
        "vreg_checkpoint": str(vreg_ckpt),
        "standard_checkpoint_sha256": std_meta["sha256_sae_pt"],
        "vreg_checkpoint_sha256": vreg_meta["sha256_sae_pt"],
        "deduplication_unit": example_summary["deduplication_unit"],
        "deduplicated_examples": example_summary["deduplicated_examples"],
    }
    cache_path = feature_cache_path(feature_cache_dir, profile, task_split_sha256, texts)
    features = load_feature_cache(cache_path, cache_meta)
    if features is None:
        lm, tok = load_model_and_tokenizer(
            model_cfg["model_id"],
            device=device,
            dtype=lm_dtype,
        )
        hidden = collect_hidden_texts(
            lm, tok, texts, layer, device, hidden_batch_size, extraction_protocol, max_length
        )
        sae_standard = load_sae(std_ckpt, device)
        sae_vreg = load_sae(vreg_ckpt, device)
        features = encode_representations(hidden, sae_standard, sae_vreg)
        del lm, sae_standard, sae_vreg
        clear_device_cache(device)
        save_feature_cache(cache_path, features, cache_meta)
        print(f"[{profile}] Saved feature cache -> {cache_path}")
    else:
        print(f"[{profile}] Loaded feature cache -> {cache_path}")

    profile_results: dict[str, Any] = {
        "profile": profile,
        "run_metadata": {
            "model_id": model_cfg["model_id"],
            "layer": layer,
            "dtype": lm_dtype,
            "device": device,
            "hidden_batch_size": hidden_batch_size,
            "max_length": max_length,
            "extraction_protocol": extraction_protocol,
            "feature_cache_path": str(cache_path),
            "example_summary": example_summary,
            "standard_checkpoint": std_meta,
            "vreg_checkpoint": vreg_meta,
        },
        "representations": {},
    }
    for representation in REPRESENTATIONS:
        (
            x_train,
            y_train,
            x_dev,
            y_dev,
            x_test,
            y_test,
            _train_examples,
            dev_examples,
            test_examples,
        ) = split_arrays(examples, features, representation)
        rep_block: dict[str, Any] = {"real_labels": {}, "random_label_controls": {}}

        scaler, model, best_c = fit_probe(
            x_train, y_train, x_dev, y_dev, real_label_probe_seed
        )
        dev_eval = evaluate_probe(scaler, model, x_dev, y_dev, dev_examples)
        test_eval = evaluate_probe(scaler, model, x_test, y_test, test_examples)
        rep_block["real_labels"][str(real_label_probe_seed)] = {
            "selected_C": best_c,
            "dev": dev_eval,
            "test": test_eval,
        }

        for shuffle_seed in random_label_seeds:
            y_train_shuffled = shuffled_train_labels(y_train, shuffle_seed)
            control_scaler, control_model, control_c = fit_probe(
                x_train,
                y_train_shuffled,
                x_dev,
                y_dev,
                probe_seed=real_label_probe_seed,
            )
            rep_block["random_label_controls"][str(shuffle_seed)] = {
                "selected_C": control_c,
                "dev": evaluate_probe(
                    control_scaler, control_model, x_dev, y_dev, dev_examples
                ),
                "test": evaluate_probe(
                    control_scaler, control_model, x_test, y_test, test_examples
                ),
            }
        profile_results["representations"][representation] = rep_block
    return profile_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen E3 negation probes")
    parser.add_argument("--task-split-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--profile", choices=["gpt2", "gemma-2-2b", "qwen-2.5-3b", "all"], default="all")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--lm-dtype", default="float16")
    parser.add_argument("--hidden-batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--gemma-vreg-checkpoint", type=Path, default=DEFAULT_GEMMA_E1R_VREG)
    parser.add_argument(
        "--feature-cache-dir",
        type=Path,
        default=REVISION_ROOT / "E3_heldout_probes" / "results" / "feature_cache",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    split_payload = read_json(args.task_split_json)
    if split_payload.get("status") not in ALLOWED_SPLIT_STATUSES:
        raise ValueError(
            f"Task split status must be one of {sorted(ALLOWED_SPLIT_STATUSES)}; "
            f"got {split_payload.get('status')!r}"
        )
    manifest = load_manifest()
    device = resolve_device(args.device)
    profiles = ["gpt2", "gemma-2-2b", "qwen-2.5-3b"] if args.profile == "all" else [args.profile]
    probe_protocol = split_payload.get("probe_protocol", {})
    real_label_probe_seed = int(
        probe_protocol.get("real_label_probe_seed", probe_protocol.get("probe_seeds", [REAL_LABEL_PROBE_SEED])[0])
    )
    random_label_seeds = split_payload["random_label_control"]["seeds"]
    task_split_sha256 = sha256_file(args.task_split_json)

    raw_examples = split_payload["examples"]
    examples = deduplicate_examples(raw_examples)
    example_summary = deduplication_summary(raw_examples, examples)

    existing = ensure_output_allowed(args.output_json, args.resume, args.force)
    if existing and existing.get("task_split_sha256") and existing["task_split_sha256"] != task_split_sha256:
        raise ValueError(
            "Existing output was built from a different task split. "
            "Use --force to overwrite or point --output-json elsewhere."
        )

    results = existing or {
        "experiment": "E3_heldout_probes",
        "task": "negation_state",
        "status": "negation_probe_results",
        "task_split_json": str(args.task_split_json),
        "task_split_sha256": task_split_sha256,
        "split_variant": split_payload.get("split_variant"),
        "example_summary": example_summary,
        "profiles": {},
    }
    results["task_split_json"] = str(args.task_split_json)
    results["task_split_sha256"] = task_split_sha256
    results["split_variant"] = split_payload.get("split_variant")
    results["example_summary"] = example_summary
    results["requested_profiles"] = profiles
    results.setdefault("profiles", {})

    for profile in profiles:
        if args.resume and profile_complete(
            results["profiles"].get(profile, {}),
            real_label_probe_seed=real_label_probe_seed,
            random_label_seeds=random_label_seeds,
        ):
            print(f"[{profile}] Existing probe results complete; skipping due to --resume")
            continue
        print(f"[{profile}] Running negation probes...")
        results["profiles"][profile] = evaluate_profile(
            profile,
            examples,
            example_summary,
            manifest,
            device,
            args.lm_dtype,
            args.hidden_batch_size,
            args.max_length,
            args.gemma_vreg_checkpoint,
            real_label_probe_seed,
            random_label_seeds,
            args.feature_cache_dir,
            task_split_sha256,
        )
        requested_missing = [
            p
            for p in profiles
            if not profile_complete(
                results["profiles"].get(p, {}),
                real_label_probe_seed=real_label_probe_seed,
                random_label_seeds=random_label_seeds,
            )
        ]
        results["requested_profiles_status"] = "complete" if not requested_missing else "partial"
        results["status"] = (
            "complete_for_requested_profiles"
            if not requested_missing and args.profile != "all"
            else "complete"
            if not requested_missing
            else "partial"
        )
        write_results(args.output_json, results)

    requested_missing = [
        p
        for p in profiles
        if not profile_complete(
            results["profiles"].get(p, {}),
            real_label_probe_seed=real_label_probe_seed,
            random_label_seeds=random_label_seeds,
        )
    ]
    all_missing = [
        p
        for p in ["gpt2", "gemma-2-2b", "qwen-2.5-3b"]
        if not profile_complete(
            results["profiles"].get(p, {}),
            real_label_probe_seed=real_label_probe_seed,
            random_label_seeds=random_label_seeds,
        )
    ]
    results["requested_profiles_status"] = "complete" if not requested_missing else "partial"
    results["all_profiles_status"] = "complete" if not all_missing else "partial"
    results["status"] = (
        "complete"
        if args.profile == "all" and not all_missing
        else "complete_for_requested_profiles"
        if not requested_missing
        else "partial"
    )
    write_results(args.output_json, results)
    print(f"Saved JSON -> {args.output_json} (status={results['status']})")


if __name__ == "__main__":
    main()
