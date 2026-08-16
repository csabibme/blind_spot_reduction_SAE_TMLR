"""Shared path setup for the dedicated GPT-2 V-reg experiment.

Reuses the frozen phase2 helper modules (lm_loader, model_profiles, activations,
sae_model_v2, v_gini_loss_v2, perturbation_hidden_cache, perturbation_data)
WITHOUT copying them, so this experiment stays consistent with the training
code that produced the existing checkpoints. Nothing here writes to those dirs.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def phase2_scripts() -> Path:
    """Locate FINAL/phase2_joint16_gemma_sweep/scripts above this file."""
    for parent in HERE.parents:
        cand = parent / "phase2_joint16_gemma_sweep" / "scripts"
        if (cand / "lm_loader.py").is_file():
            return cand
    raise FileNotFoundError("phase2_joint16_gemma_sweep/scripts not found")


_P2 = phase2_scripts()
if str(_P2) not in sys.path:
    sys.path.insert(0, str(_P2))
