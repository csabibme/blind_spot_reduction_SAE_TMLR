# Protocol — Numeric-orientation stress test

## Objective

Create a real-model stress test with more headroom than the simple dosage perturbation probe.

The central design principle is:

> The larger surface number is sometimes the wrong answer.

This distinguishes simple numeric accessibility from contextually oriented numeric reasoning.

## Families

### 1. total_daily_dose

Compares total 24-hour exposure.

Example trap:

```text
A receives 2 mg every 6 hours.
B receives 7 mg once daily.
Who receives the larger total daily dose?
```

A is correct because `2 × 24/6 = 8 mg/day`, despite B having the larger surface amount.

### 2. concentration_volume

Compares active amount after `concentration × volume`.

Example trap:

```text
A contains 2 mg/mL and 6 mL is given.
B contains 9 mg/mL and 1 mL is given.
```

A is correct because `12 mg > 9 mg`.

### 3. unit_conversion

Compares doses after converting units.

Example trap:

```text
A receives 0.8 mg.
B receives 650 micrograms.
```

A is correct because `0.8 mg = 800 micrograms`.

### 4. change_direction

Tests whether the model binds old/new values correctly under order changes.

Example:

```text
The prescription changed to 3 mg from 8 mg.
Was this increased or decreased?
```

Correct answer: decreased.

## Raw output evaluation

Use deterministic candidate log-probability scoring.

For A/B tasks:

```text
score(A) = log p(" A" | prompt)
score(B) = log p(" B" | prompt)
```

For direction tasks:

```text
score(increased) = log p(" increased" | prompt)
score(decreased) = log p(" decreased" | prompt)
```

Reported output-level endpoints:

- accuracy
- correct-answer logprob margin
- trap surface-error rate
- family/regime breakdown

## SAE-code probe evaluation

Use the prompt hidden state at the final prompt token (`Answer:`) as the input representation.

Representations:

- raw hidden
- Standard SAE code
- V-reg SAE code

A logistic probe is trained on train-template items, tuned on dev-template items, and evaluated on test-template items.

Main endpoint:

```text
Δ probe AUROC = AUROC(V-reg code) - AUROC(Standard code)
```

Secondary endpoints:

```text
Δ balanced accuracy
Δ probability-margin L20
family/regime breakdown
```

## Interpretation

If the trap regime is not at ceiling, then Standard vs V-reg can be compared in a real-model setting with headroom.

Possible paper wording:

> The simple held-out dosage probe saturates in modern pretrained models. We therefore add a numeric-orientation stress test in which the larger surface number is not necessarily the task-relevant quantity. This creates a non-saturated real-model regime for evaluating whether perturbation-aware SAE codes improve held-out probe accessibility.

## Reproducibility notes

- Use `--batch-size 1` for Gemma on MPS/float16 if larger batches produce non-finite hidden states.
- The extraction script refuses to save non-finite arrays and records `nonfinite_sanitized=false` in metadata.
- Do not use sanitized hidden caches for paper results.
