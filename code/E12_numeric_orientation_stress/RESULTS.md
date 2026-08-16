# Results — Numeric-orientation stress test

**Status: contextual negative result.** This experiment is reported as a headroom
*gate*, not as an SAE comparison. After removing a benchmark confound, a 3B base model
shows **no clean non-saturated numeric-reasoning regime**: it either saturates or falls
back to a position/surface heuristic. There is therefore nothing for a downstream
SAE-code probe to discriminate, so we do not run the Standard-vs-V-reg probe here. The
result is used to motivate the controlled toy (`exp_toy`).

## What the gate measures

`eval_raw_logprob.py` scores each item by deterministic candidate log-probability
(` A` vs ` B`, or ` increased` vs ` decreased`) appended to the prompt. No sampling,
no SAE. The question is purely: *does the base model have headroom on contextually
oriented numeric traps, where the larger surface number is not the task-relevant
quantity?* If a model is at ceiling (or at chance) there is no regime in which a
Standard-vs-V-reg SAE probe could be informative.

## Benchmark confound found and fixed

The original generator coupled the correct answer to answer **position** and to the
per-cell **label distribution**:

- In `total_daily_dose`, the frequency-dosed arm was always option A and was always the
  correct trap answer, so the **trap cell was single-class** (correct = A for 100% of
  items). A model that simply prefers the token " A" scored 100% on that cell.
- Across the A/B families the label was imbalanced per cell, so raw accuracy reflected a
  position prior times the label prior, producing a spurious **trap > easy** inversion.

**Fix (`build_stress_benchmark.py`, arm-based construction):**

1. Each item is built from two *arms* (each with a surface number and a true quantity).
2. The correct arm is placed at position A or B by an alternating **per-template**
   counter, so the correct answer is balanced ~50/50 **within every template** (hence
   within every split, since the split is template-held-out).
3. `regime` (easy/trap) is a property of the arm pair only (does the larger surface
   number carry the larger true quantity?), independent of A/B placement.

After the fix, an "always A" predictor scores exactly 50% in every A/B cell, so any
above-chance accuracy reflects genuine comparison rather than a position prior.

## Clean gate result (Qwen2.5-3B base, test split, debiased)

Overall accuracy 0.703 over 300 held-out-template items. The breakdown is the point:

| Family | easy | trap | Predicted distribution | Reading |
|---|---:|---:|---|---|
| `change_direction` | 1.000 | 1.000 | balanced (53 / 47) | genuine reasoning, **saturated** |
| `total_daily_dose` | 0.500 | 0.515 | almost always A (67 / 0) | **chance** — not solved |
| `concentration_volume` | 0.576 | 0.515 | almost always A (64 / 2) | **chance** — not solved |
| `unit_conversion` | 0.382 | 0.848 | mixed (36 / 31) | non-normalizing heuristic, not clean reasoning |

- The only family the model genuinely solves (`change_direction`, a semantic
  increased/decreased judgement) is **at ceiling** for both regimes — no headroom.
- On the A/B arithmetic comparisons the model defaults to the " A" token
  (`total_daily_dose`, `concentration_volume` sit at chance) or follows a non-normalizing
  surface heuristic (`unit_conversion`), i.e. it does **not** compute the task-relevant
  quantity.

## Why this rules out an SAE comparison here

The downstream probe target is the correct answer decoded from the prompt hidden state.
If the base model does not compute that answer (A/B arithmetic families) the hidden state
does not encode it, so all representations — raw hidden, Standard code, V-reg code — are
near chance and a probe measures noise. Where the model does encode the answer
(`change_direction`) every representation is near ceiling, so there is no headroom for a
Standard-vs-V-reg difference. Either way there is no non-saturated regime in which the
SAE-code comparison is informative at the base-3B scale.

This is consistent with the dosage experiment: a frozen, generally-trained SAE re-encodes
whatever the base model already computed, and the V-regulariser has no structural reason
to improve a real-model reasoning probe. The clean mechanism demonstration lives in the
controlled toy (`exp_toy`), where the SAE is trained on the relevant perturbation
distribution.

## Manuscript framing

> To test whether a non-saturated regime exists in a real pretrained model, we built a
> rule-based numeric-orientation stress test in which the larger surface number is not
> necessarily the task-relevant quantity, and balanced the answer position per template so
> that a position prior scores at chance. A 3B base model either saturates on the items it
> can solve (semantic increase/decrease judgements) or falls back to a position/surface
> heuristic on the arithmetic comparisons, leaving no regime in which the correct answer
> is partially but not fully accessible. We therefore use the controlled synthetic setting
> of Section [toy] for the mechanism check, where the sparse code is trained on the
> relevant perturbation distribution, and we report the real-model dosage probe with its
> ceiling/limitations acknowledged.

Claim discipline: this is a **negative/contextual** result (a headroom gate), not evidence
for or against the V-regulariser. It is **not** a clinical benchmark.

## Reproduce

```bash
python build_stress_benchmark.py --config config_stress_full.json --out data/numeric_orientation_stress.json
python eval_raw_logprob.py --model Qwen/Qwen2.5-3B --dataset data/numeric_orientation_stress.json \
  --out results/qwen_raw_logprob_debiased.json --split test --dtype float16
```

Artifacts: `results/qwen_raw_logprob_debiased.json` / `.md` (clean gate);
`results/qwen_raw_logprob.json` / `.md` (pre-debias, kept only to document the confound).
