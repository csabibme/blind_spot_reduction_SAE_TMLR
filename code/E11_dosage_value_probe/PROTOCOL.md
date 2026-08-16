# exp_dosage Protocol — Held-out dosage numeric probe

**Status:** `DRAFT / smoke-validated`

---

## Purpose

Answer reviewer R2: if V-regularisation improves the response profile, does the motivating
dosage/numeric distinction become more accessible in frozen SAE codes — measured by endpoints
**other than** $V_{\mathrm{Gini}}$ (AUROC, balanced accuracy, probability margin)?

---

## Dataset

- Name: `heldout_dosage_numeric_probe`
- **Not** in V-loss training data
- Families: `dose_count`, `dose_amount`
- Positive label: `critical_numeric_change` (numeric value changes, e.g. 3 vs 8)
- Negative label: `nuisance` (paraphrase or digit/word format control, same numeric value)

### Split policy

- Group unit: `template_id` (assigned before pair generation)
- Test templates disjoint from train/dev
- Full: 24 templates (~80–120 target at scale; expand in future if needed)
- Smoke: 10 templates, 6/2/2 train/dev/test

---

## Representations

Identical perturbation pairs for all representations:

| Key | Source |
|---|---|
| `hidden` | Base LM, `true_last` extraction |
| `sae_standard_code` | Manifest Standard joint-16 OWT |
| `sae_vreg_code` | Manifest V-reg joint-16 OWT |

No new SAE training. Gemma V-reg uses E1R override path when present (same as E3).

---

## Probe policy

- Pair feature: $\phi_i = |z_i^{(2)} - z_i^{(1)}|$
- Classifier: L2-regularised logistic regression, `class_weight=balanced`
- Hyperparameter grid: `C ∈ {0.01, 0.1, 1, 10, 100}`
- Select `C` on dev by balanced accuracy
- Fixed probe seed: 42

---

## Primary endpoints (test)

| Endpoint | Definition |
|---|---|
| Distance AUROC | AUROC of $\|z^{(2)}-z^{(1)}\|$ for critical vs nuisance |
| Probe AUROC | AUROC of held-out logistic probe |
| Balanced accuracy | On test, selected threshold |
| Probe margin L20 | Lower 20% mean of signed probability margin |
| Critical response L20 | Lower 20% mean of $\|z^{(2)}-z^{(1)}\|$ on critical numeric-change pairs only |
| Critical response mean | Mean $\|z^{(2)}-z^{(1)}\|$ on critical numeric-change pairs only |
| Critical relative response L20 | Lower 20% mean of $\|z^{(2)}-z^{(1)}\| / (\|z^{(1)}\|+\epsilon)$ on critical pairs |
| Hard direction probe | Signed-difference probe, increase vs reversed decrease, on critical pairs only |
| Δ vs Standard | V-reg minus Standard on same pairs |

Bootstrap: 1000 resamples over **template clusters** on test pairs; 95% CI for Δ endpoints.

Interpretation priority if the standard positive-vs-nuisance probe saturates: report it as a
ceiling result, and use the critical-only response endpoints as the primary evidence for
whether held-out dosage perturbations receive stronger SAE-code displacement under V-reg.

---

## Non-tautological claim checklist

- Probe set excluded from V-loss
- Metrics are AUROC / BA / margin, not $V_{\mathrm{Gini}}$
- Test templates held out from probe train/dev
- Standard and V-reg share base LM, width, and training protocol (manifest)
- Paired comparison on identical perturbations

---

## Smoke acceptance

GPT-2 smoke must complete:

1. `build_heldout_dosage_probe.py --smoke`
2. `preflight_dosage_probe.py --smoke` → exit 0
3. `run_dosage_probe_features.py --profile gpt2`
4. `analyze_dosage_probe.py --profile gpt2` → JSON + MD under `results/`
