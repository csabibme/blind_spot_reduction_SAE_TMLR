# Perturbation-Awareness SAE Experiments (E1–E17)

Code, frozen evaluation data, and machine-readable results accompanying:

> **A Symmetry-Matching Approach to Blind-Spot Reduction in Sparse Autoencoders**  
> Transactions on Machine Learning Research, 2026  
> OpenReview: https://openreview.net/forum?id=NWWpKC9CZH

The artifact separates the central concepts used in the paper:

- **representational blind spot:** the failure mode under study;
- **perturbation awareness:** the graded capacity of an encoding to respond to
  task-relevant changes;
- **perturbation-response hierarchy:** the evaluation framework;
- **V-Gini:** the directly optimized response-profile imbalance component.

## Repository layout

- `code/` — analysis, training, audit, and verification scripts;
- `data/` — frozen minimal-pair and synthetic evaluation inputs;
- `results/` — frozen JSON/Markdown outputs used by the paper;
- `REPRODUCIBILITY.md` — verification and full-rerun requirements;
- `ARTIFACT_MANIFEST.json` — SHA-256 inventory of this snapshot.
- `manifest.yaml` — checkpoint/model provenance registry;
- `docs/METRICS_SPEC.md` — frozen E1 metric specification.

The experiment labels follow the manuscript history and are intentionally
non-consecutive. The public series comprises E1, E2, E3/E3b/E3c, E5, E7/E7b,
and E10–E17.

## Quick verification

```bash
python -m pip install -r requirements.txt
python -m pytest -q \
  code/E3_heldout_probes/test_qwen_weak_accessibility_audit.py \
  code/E14_real_clinical_lower_tail/tests \
  code/E17_exact_lower_tail_guarantee/tests
python code/E17_exact_lower_tail_guarantee/verify_exact_lower_tail.py
```

Large model checkpoints and activation/feature caches are intentionally
excluded. Frozen result files are self-contained; full model reruns require the
external artifacts described in `REPRODUCIBILITY.md`.

## License

The original software in this repository is released under the
[MIT License](LICENSE). Bundled datasets and source-derived material may remain
subject to their source-specific terms; see the corresponding data README files.
