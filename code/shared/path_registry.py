"""
Portable path resolution for TMLR TMLR submission.

All manifests store repo-relative paths. Resolve with SAE_REPO_ROOT
(absolute path to the SAE repository root).
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = PACKAGE_ROOT / "manifest.yaml"


def repo_root() -> Path:
    env = os.environ.get("SAE_REPO_ROOT")
    if env:
        return Path(env).resolve()
    # Walk up from submission to SAE repo root (parent of FINAL/)
    candidate = REVISION_ROOT.parent.parent
    if (candidate / "SAE_scaling").is_dir() and (candidate / "FINAL").is_dir():
        return candidate.resolve()
    raise RuntimeError(
        "Set SAE_REPO_ROOT to the SAE repository root "
        "(directory containing SAE_scaling/ and FINAL/)."
    )


def load_manifest(path: Path | None = None) -> dict:
    manifest_path = path or DEFAULT_MANIFEST
    with open(manifest_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_rel(rel_path: str, root: Path | None = None) -> Path:
    return (root or repo_root()) / rel_path


def checkpoint_dir(checkpoint_id: str, manifest: dict | None = None) -> Path:
    m = manifest or load_manifest()
    ckpt = m["checkpoints"][checkpoint_id]
    return resolve_rel(ckpt["rel_path"])


def checkpoint_sae_path(checkpoint_id: str, manifest: dict | None = None) -> Path:
    m = manifest or load_manifest()
    ckpt = m["checkpoints"][checkpoint_id]
    return checkpoint_dir(checkpoint_id, m) / ckpt.get("sae_pt", "sae.pt")


def pairs_path(manifest: dict | None = None) -> Path:
    m = manifest or load_manifest()
    rel = m["data_files"]["joint16_pairs"]["rel_path"]
    return resolve_rel(rel)


def sae_scaling_root(manifest: dict | None = None) -> Path:
    m = manifest or load_manifest()
    return resolve_rel(m["sources"]["sae_code"]["rel_root"])


def verify_sha256(path: Path, expected: str) -> bool:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest() == expected.lower()


def verify_checkpoint(checkpoint_id: str, manifest: dict | None = None) -> None:
    m = manifest or load_manifest()
    ckpt = m["checkpoints"][checkpoint_id]
    sae_path = checkpoint_sae_path(checkpoint_id, m)
    if not sae_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {sae_path}")
    expected = ckpt.get("sha256_sae_pt")
    if expected and not verify_sha256(sae_path, expected):
        raise ValueError(f"SHA256 mismatch for {checkpoint_id}: {sae_path}")
