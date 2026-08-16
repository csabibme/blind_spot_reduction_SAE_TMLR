# exp_dosage — Held-out dosage numeric probe

Non-tautological evaluation of whether V-reg SAE codes make clinically critical numeric
changes (dose count / dose amount) more accessible than Standard SAE codes on a probe set
**not used in V-loss training**.

## Setup

From the public repository root, install `requirements.txt`. For
full feature extraction, point `SAE_REPO_ROOT` to the external training
workspace containing `FINAL/REVISION_1/` and `SAE_scaling/`:

```bash
export SAE_REPO_ROOT=/path/to/external/training-workspace
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
```

## Step 1 — Build dataset

Full (24 templates):

```bash
python code/E11_dosage_value_probe/build_heldout_dosage_probe.py \
  --output-json data/E11_dosage_value_probe/heldout_dosage_numeric_probe.json \
  --output-md data/E11_dosage_value_probe/heldout_dosage_numeric_probe.md \
  --seed 42
```

Smoke (10 templates, GPT-2 pilot):

```bash
python code/E11_dosage_value_probe/build_heldout_dosage_probe.py \
  --output-json data/E11_dosage_value_probe/heldout_dosage_numeric_probe_smoke.json \
  --output-md data/E11_dosage_value_probe/heldout_dosage_numeric_probe_smoke.md \
  --seed 42 \
  --smoke
```

## Step 2 — Preflight (no LM/SAE)

```bash
python code/E11_dosage_value_probe/preflight_dosage_probe.py \
  --dataset-json data/E11_dosage_value_probe/heldout_dosage_numeric_probe_smoke.json \
  --smoke \
  --output-json results/e11_dosage_value_probe/dosage_probe_preflight_smoke.json
```

## Step 3 — Feature extraction

GPT-2 smoke:

```bash
python code/E11_dosage_value_probe/run_dosage_probe_features.py \
  --dataset-json data/E11_dosage_value_probe/heldout_dosage_numeric_probe_smoke.json \
  --output-json results/e11_dosage_value_probe/dosage_probe_features_smoke.json \
  --profile gpt2 \
  --device auto \
  --hidden-batch-size 16 \
  --max-length 128
```

All three models (full run):

```bash
python code/E11_dosage_value_probe/run_dosage_probe_features.py \
  --dataset-json data/E11_dosage_value_probe/heldout_dosage_numeric_probe.json \
  --output-json results/e11_dosage_value_probe/dosage_probe_features.json \
  --profile all \
  --device auto
```

## Step 4 — Analysis (cached features only)

```bash
python code/E11_dosage_value_probe/analyze_dosage_probe.py \
  --dataset-json data/E11_dosage_value_probe/heldout_dosage_numeric_probe_smoke.json \
  --features-json results/e11_dosage_value_probe/dosage_probe_features_smoke.json \
  --output-json results/e11_dosage_value_probe/dosage_probe_analysis_smoke.json \
  --output-md results/e11_dosage_value_probe/dosage_probe_analysis_smoke.md \
  --profile gpt2
```

## Representations

- `hidden` — base LM last-token hidden state (`true_last`)
- `sae_standard_code` — frozen Standard joint-16 OWT checkpoint
- `sae_vreg_code` — frozen V-reg joint-16 OWT checkpoint

Checkpoint identities come from the bundled `manifest.yaml`; files are resolved under `SAE_REPO_ROOT`.

Extraction defaults match E3: `device=auto` (MPS on Mac), `lm_dtype=float16`, `hidden_batch_size=16`.

## Metrics (test split)

1. **Distance AUROC** — `||z_right - z_left||` separates critical vs nuisance pairs
2. **Held-out logistic probe** — feature `|z_right - z_left|`, L2-regularised, `C` selected on dev
3. **Critical-only response** — lower-tail and mean `||z_right - z_left||` on critical numeric changes only
4. **Hard direction probe** — signed feature `z_right - z_left`, increase vs reversed decrease on critical pairs
5. **Template-cluster bootstrap** — 95% CI for V-reg minus Standard deltas

## Outputs

```text
data/E11_dosage_value_probe/heldout_dosage_numeric_probe.json
data/E11_dosage_value_probe/heldout_dosage_numeric_probe_smoke.json
results/e11_dosage_value_probe/feature_cache/
results/e11_dosage_value_probe/dosage_probe_features_smoke.json
results/e11_dosage_value_probe/dosage_probe_analysis_smoke.json
results/e11_dosage_value_probe/dosage_probe_analysis_smoke.md
```

See `PROTOCOL.md` for frozen split and probe rules.
