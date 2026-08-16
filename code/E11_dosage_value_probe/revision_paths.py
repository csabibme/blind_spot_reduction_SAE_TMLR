"""Path helpers for Revision 2 experiments (reuse Revision 1 manifest + eval stack)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
_external_root = os.environ.get("SAE_REPO_ROOT")
SAE_REPO_ROOT = Path(_external_root).resolve() if _external_root else PACKAGE_ROOT
REVISION_1_ROOT = SAE_REPO_ROOT / "FINAL/REVISION_1"
E1_ROOT = REVISION_1_ROOT / "E1_absolute_sensitivity"


def ensure_import_paths() -> None:
    for path in (REVISION_1_ROOT, E1_ROOT):
        root = str(path.resolve())
        if root not in sys.path:
            sys.path.insert(0, root)


def repo_root() -> Path:
    ensure_import_paths()
    from shared.path_registry import repo_root as _repo_root

    return _repo_root()


def load_manifest():
    ensure_import_paths()
    from shared.path_registry import load_manifest as _load_manifest

    return _load_manifest(PACKAGE_ROOT / "manifest.yaml")


def default_env() -> dict[str, str]:
    root = repo_root()
    env = dict(os.environ)
    env.setdefault("SAE_REPO_ROOT", str(root))
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    return env
