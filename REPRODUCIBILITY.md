# Reproducibility

## Verification without model checkpoints

The files under `results/` contain the point estimates, intervals, provenance
fields, and—in the principal audits—pair-level records needed to verify the
reported tables. The lightweight tests and E17 exact lower-tail verifier run
directly from this repository.

From the repository root:

```bash
python -m pytest -q \
  code/E3_heldout_probes/test_qwen_weak_accessibility_audit.py \
  code/E14_real_clinical_lower_tail/tests \
  code/E17_exact_lower_tail_guarantee/tests
python code/E17_exact_lower_tail_guarantee/verify_exact_lower_tail.py
```

## Full model reruns

End-to-end feature extraction and SAE training additionally require:

- Python dependencies in `requirements.txt`;
- GPT-2, Qwen-2.5-3B, and Gemma-2-2B model weights;
- the base SAE implementation used by the project (`SAE_scaling/`);
- the original repository manifest and checkpoint registry;
- regenerated joint-16 SAE checkpoints and activation/feature caches.

Set `SAE_REPO_ROOT` to the external training-workspace root when running scripts
that use those resources. Checkpoints and caches are omitted because of their
size; their identities and hashes are recorded in protocols and result JSONs.
The bundled `manifest.yaml` is a provenance snapshot; its paths are resolved
against that external workspace.

## Scope

This repository is the public E-series artifact. It excludes the original
submission phase folders, manuscript/rebuttal files, compiled PDFs, local logs,
temporary outputs, and invalidated Gemma padding-artifact results.
