# Experiment 1 — Controlled non-saturated toy SAE

This package implements a small synthetic experiment for the SAE revision.

Goal: create a controlled regime where the task-relevant numeric direction is present in the hidden state, but weak relative to nuisance/template variation. In this non-saturated regime, a standard reconstruction+sparsity SAE can under-preserve the weak direction, while a perturbation-response regularized SAE can improve held-out probe accessibility.

The experiment is intentionally not a clinical benchmark. It is a mechanism check:

> When the downstream task has headroom, does perturbation-aware SAE training translate response-profile changes into probe improvement?

## Quick start

```bash
python run_toy_experiment.py --config config_smoke.json --out results_smoke
```

For the fuller seed/alpha sweep:

```bash
python run_toy_experiment.py --config config_full.json --out results_full
```

The script writes:

- `summary.json`
- `summary.md`
- per-run JSON records under `runs/`

## Design notes

- **Paired runs.** The Standard and V-reg SAEs are paired by initialization and
  reconstruction-minibatch stream; the V-reg run uses an additional independent
  generator only for perturbation-pair sampling. This common-randomness design
  isolates the effect of the V-regulariser from optimizer noise.
- **Overcomplete dictionary.** `d_sae >= d_in` so the toy probes SAE-like feature
  selection rather than a compressive bottleneck.
- **Fixed selection rule.** The main-text configuration is chosen by a single fixed rule
  applied uniformly to every point (see `PROTOCOL.md`): qualification requires hidden
  AUROC ≥ 0.70, Standard AUROC < 0.75, Δ AUROC > 0, Δ L20(‖Δz‖) > 0, MSE ratio ≤ 1.15; the
  main-text point is the qualifying point in the design target band (hidden 0.75–0.90,
  Standard 0.60–0.75), smaller MSE ratio breaking ties. Described as a rule-based (not
  formally pre-registered) procedure.

## Main metrics

For each signal strength `alpha` and random seed, the experiment reports:

- hidden-state probe AUROC / balanced accuracy
- Standard SAE code probe AUROC / balanced accuracy
- V-reg SAE code probe AUROC / balanced accuracy
- `V-reg minus Standard` probe AUROC
- critical perturbation lower-tail `L20(||Δz||)`
- reconstruction MSE
- response-profile Gini on held-out perturbation pairs

Expected qualitative regimes:

1. Very small `alpha`: hidden itself may be weak; no method can recover much.
2. Intermediate `alpha`: hidden has signal, Standard SAE is sub-ceiling, V-reg can improve probe accessibility.
3. Large `alpha`: all representations saturate; downstream differences disappear.

## Interpretation to use in the paper

Method (paired design):

> We use a paired synthetic control in which the Standard and \(V\)-regularised SAEs share the same initialization and reconstruction minibatch sequence; the \(V\)-regularised run uses an additional independent generator only for perturbation-pair sampling. This common-randomness design isolates the effect of the \(V\)-regulariser from optimizer noise.

Result:

> In the intermediate non-saturated regime, the task-relevant direction is present in the hidden state but only partially preserved by the Standard SAE code. \(V\)-regularisation yields a modest but consistent improvement in held-out probe AUROC, while increasing the lower-tail critical code displacement. The effect is therefore not merely a decrease of the trained response-imbalance statistic.

Limitation:

> The controlled experiment is not intended as a clinical benchmark. It is a mechanism check showing that response-profile regularisation can translate into downstream probe gains when the task is not already saturated.

Claim discipline: report a **modest but consistent** downstream-probe improvement, not a *large* downstream improvement.

## Notes

- The toy is not used to claim clinical downstream performance.
- It complements the real-model dosage probe, where modern models are often already near ceiling.
- The data split is template-held-out, so the probe is tested on templates not used by the downstream classifier.
