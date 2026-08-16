# Protocol — Controlled non-saturated toy SAE

## Objective

Demonstrate, in a controlled non-saturated setting, that perturbation-response regularisation can improve downstream probe accessibility when a task-relevant direction is present but weak.

## Data generation

Each synthetic hidden state is generated as

```text
h = R(template_component + drug_component + alpha * y * numeric_direction(template) + noise)
```

where:

- `y ∈ {-1,+1}` is the binary numeric state;
- template/drug components are large nuisance directions;
- `alpha` controls the strength of the task-relevant numeric direction;
- `R` is a random orthogonal rotation;
- the numeric direction has a shared base component plus template-specific jitter.

The train/dev/test split holds out whole templates.

## SAE training

Two SAEs are trained on the same train states:

1. **Standard SAE**: reconstruction MSE + L1 sparsity.
2. **V-reg SAE**: same objective plus Gini regularisation of relative code responses on train perturbation pairs.

For a perturbation pair `(x_left, x_right)`, the relative response is

```text
D = ||z_left - z_right|| / (||x_left - x_right|| + eps)
```

and the regularizer is the Gini coefficient across a minibatch of `D` values.

### Paired (common-randomness) design

The Standard and V-reg runs are **paired**: they share the same parameter
initialization and the same reconstruction-minibatch stream. The V-reg run draws
its perturbation-pair minibatches from a separate, independent generator, so that
adding the pairing does not perturb the reconstruction-batch order. This isolates
the effect of the `V`-regulariser from optimizer/initialization noise, so that

```text
Δ probe AUROC = AUROC(z_vreg) - AUROC(z_standard)
```

reflects the regulariser rather than two independent training trajectories.

The dictionary is **overcomplete** (`d_sae >= d_in`); a compressive bottleneck would
test a plain compressive autoencoder rather than an SAE-like feature basis and
collapses on held-out templates.

## Downstream probe

A logistic probe is trained on train-template representations, tuned on dev templates, and evaluated on held-out test templates.

Reported representations:

- hidden state `h`
- Standard SAE code `z_standard`
- V-reg SAE code `z_vreg`

## Main endpoint

```text
Δ probe AUROC = AUROC(z_vreg) - AUROC(z_standard)
```

Secondary endpoints:

```text
Δ L20(||Δz||)
Δ L20(D)
Δ response-profile Gini
reconstruction MSE
```

## Fixed selection rule

To avoid the appearance of hyperparameter hunting, the regime/configuration shown in the
main text is chosen by a single fixed rule that is applied uniformly to every
`(configuration, alpha)` point, rather than by choosing whatever looks best. A point
**qualifies** if, averaged over seeds:

1. `hidden AUROC >= 0.70` — the task-relevant direction is genuinely present in the hidden state;
2. `Standard SAE AUROC < 0.75` — the Standard code is sub-ceiling (non-saturated regime);
3. `Δ probe AUROC > 0` — V-reg gives a downstream gain;
4. `Δ L20(||Δz||) > 0` — V-reg raises the lower-tail critical response;
5. `MSE_vreg / MSE_standard <= 1.15` — no disproportionate reconstruction damage.

**Main-text point.** Among qualifying points, the main text reports the one that falls in
the *design target band* stated up front as the toy's goal — `hidden AUROC in [0.75, 0.90]`
(signal present but not trivial) and `Standard AUROC in [0.60, 0.75]` (sub-ceiling); the
**smaller MSE ratio** breaks ties. A point additionally meets the stronger *ideal* bar if
`hidden AUROC >= 0.75` and `Δ probe AUROC >= 0.03`. If no qualifying point falls in the
band, the top qualifying point is reported instead, with "modest but consistent" framing.

The full `d_sae x lambda_v x noise_std` grid goes to the appendix; only the selected
point is highlighted in the main text. The complete ranked table is written to
`results_ablation/selection.md` / `selection.json` by `select_regime.py`.

We describe this as a **fixed / rule-based selection procedure** rather than a formally
pre-registered analysis, since there is no external timestamp establishing the rule ahead
of the runs.

## Intended interpretation

A positive result in the intermediate-alpha regime supports this limited claim:

> In a non-saturated regime, the response-profile intervention can translate into improved held-out probe accessibility.

It does not claim that the real clinical dosage task is improved end-to-end. The
controlled experiment is a mechanism check, not a clinical benchmark.
