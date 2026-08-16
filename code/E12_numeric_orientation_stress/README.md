# Experiment 2 — Numeric-orientation stress test

This package builds a rule-based stress test for large language models and SAE codes.

The ordinary dosage probe is often near ceiling: the model can distinguish simple numeric changes. This stress test creates cases where the larger surface number is not the task-relevant quantity. The goal is to create headroom and test whether Standard vs V-reg SAE codes differ on a harder numeric-orientation task.

## Families

1. `total_daily_dose` — compares total daily dose after frequency normalization.
2. `concentration_volume` — compares active drug amount after concentration × volume.
3. `unit_conversion` — compares doses after unit conversion.
4. `change_direction` — decides whether a dose increased or decreased despite wording/order traps.

Each family contains:

- `easy` items where the surface heuristic agrees with the correct answer;
- `trap` items where the surface heuristic points to the wrong answer;
- template-held-out train/dev/test splits.

## Quick start

Generate the benchmark:

```bash
python build_stress_benchmark.py --config config_stress_smoke.json --out data/numeric_orientation_stress.json
```

Evaluate raw model output by next-token candidate log-probability:

```bash
python eval_raw_logprob.py \
  --model gpt2 \
  --dataset data/numeric_orientation_stress.json \
  --out results/gpt2_raw_logprob.json
```

Extract hidden/SAE representations:

```bash
python extract_representations.py \
  --model gpt2 \
  --layer 12 \
  --dataset data/numeric_orientation_stress.json \
  --standard-checkpoint /path/to/standard_checkpoint \
  --vreg-checkpoint /path/to/vreg_checkpoint \
  --out results/gpt2_features.npz
```

Analyze hidden/SAE-code downstream probes:

```bash
python analyze_probe.py \
  --dataset data/numeric_orientation_stress.json \
  --features results/gpt2_features.npz \
  --out results/gpt2_probe_analysis.json
```

## Debiased construction (A/B families)

The A/B comparison families (`total_daily_dose`, `concentration_volume`,
`unit_conversion`) are built from two *arms* and the correct arm is placed at position A
or B by an alternating **per-template** counter. This balances the correct answer ~50/50
within every template (and therefore every split), so a model that simply prefers the
token " A" scores at chance rather than at the label prior. `regime` (easy/trap) is a
property of the arm pair only and is independent of A/B placement.

## Claims this experiment can support

If the raw model is not at ceiling **and not at chance** on the trap regime, this
experiment can support:

> The easy dosage distinction saturates, but harder contextually oriented numeric traps create headroom. In this non-saturated regime, Standard and V-reg SAE codes can be compared on a held-out downstream probe.

It should not be used to claim clinical validity. The benchmark is rule-based and synthetic.

## Outcome on this run — see `RESULTS.md`

The debiased gate (Qwen2.5-3B base) shows **no clean non-saturated regime**: the model
saturates on the one family it genuinely solves (`change_direction`, 1.0/1.0) and sits at
chance / follows a surface heuristic on the A/B arithmetic families. The experiment is
therefore reported as a **contextual headroom gate** motivating the controlled toy, not as
an SAE comparison. Full diagnosis, numbers, and manuscript framing are in `RESULTS.md`.

## Recommended first-pass workflow

1. Run `build_stress_benchmark.py`.
2. Run `eval_raw_logprob.py` on GPT-2, Gemma, and Qwen to identify non-ceiling families.
3. Only then run `extract_representations.py` and `analyze_probe.py` on the most informative model/family.
