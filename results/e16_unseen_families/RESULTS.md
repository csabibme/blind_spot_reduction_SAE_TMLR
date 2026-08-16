# E16 — Held-out family generalization audit (results)

Frozen general joint-16 Standard vs V-reg checkpoints (experiment_101),
zero-shot, evaluated on the held-out families of `pairs_100.json`
(prepared before reviewer feedback; the template families were never used in
any submitted training run). Endpoint ΔL20(s) per family with 95% bootstrap CI
(5000 replicates, up to 30 pairs/family, seed 42). Tier rule pre-registered in
`PROTOCOL.md`. A two-pass duplicate audit (disclosed in `PROTOCOL.md`)
excluded 12 shard families whose pair sets are identical to a retained
representative — in particular, all eight negation shards contain the same
8-pair set (they differ only in pair order), so they count as one negation
measurement, not seven. This leaves 72 distinct held-out families
(Tier A = 54, Tier B = 18); the dedup changes only aggregation weights, no
per-family value. All families reported.

## Headline

| Model | Tier | n families | mean ΔL20(s) | ΔL20 > 0 | CI excludes zero |
|---|---|---:|---:|---:|---:|
| GPT-2 | A (unseen instances of trained kinds) | 54 | +0.0160 | 98% | 91% |
| GPT-2 | B (kinds unseen in nature) | 18 | +0.0256 | **100%** | **100%** |
| Qwen-2.5-3B | A | 54 | +0.0356 | 100% | 89% |
| Qwen-2.5-3B | B | 18 | +0.0402 | 100% | 89% |
| Gemma-2-2B | A | 54 | +0.0260 | **100%** | **100%** |
| Gemma-2-2B | B | 18 | +0.0334 | **100%** | **100%** |

Gemma uses the corrected E1R protocol throughout: `true_last` extraction with
batch size 1 (the padding-sensitivity erratum), float16 LM, and the
`vreg_joint16_owt_true_last` checkpoint from E1R against the experiment_101
Standard checkpoint.

The lower-tail lift is not specific to the 16 trained families. It appears on
unseen instances and lexicons of trained kinds (Tier A) and extends to family
kinds that differ in nature from everything in training (Tier B): operator-
level (modal, hedging, quantifier, comparative), inflectional (tense,
pluralization, article), structural (voice, clause order), and discourse-level
(coreference, connective) transformations. On GPT-2 and Gemma, all 18 Tier-B
families are individually CI-positive; on Gemma all 72 held-out families are.

## Not CI-positive (reported, per the fixed rule)

- GPT-2, Tier A (5/54): `semantic_location`, `semantic_pathology`,
  `semantic_social`, `semantic_specimen`, `semantic_sub_shard_02` — point
  estimates ≥ 0 except pathology (−0.0004), CIs cross zero.
- Qwen, Tier A (6/54): all six are `number_shard_*` — consistent with the
  surface-numeric ceiling documented in the dosage probe (E11/Tier D).
- Qwen, Tier B (2/18): `preposition_swap` (+0.005, CI crosses zero),
  `tense_shift` (+0.018, CI [−0.002, +0.042]).
- Gemma: none — every held-out family is CI-positive.

No family is significantly negative on any model.

## Reconstruction (honest, perturbation-pair domain)

- GPT-2: median NMSE 5.4e-5 (Standard) → 6.6e-4 (V-reg); negligible in
  absolute terms.
- Qwen: median NMSE 0.009 (Standard) → 0.130 (V-reg) on these template pairs.
  This is the same model-dependent perturbation-domain reconstruction cost seen
  in RC-LT/E14 (general OWT reconstruction is unchanged; see the dedicated
  paired experiment). The lift is a real effect with a reconstruction price on
  the larger model in this domain, and we report the two together.
- Gemma: median NMSE 0.011 (Standard) → 0.065 (Tier A) / 0.075 (Tier B, V-reg)
  on these template pairs — same pattern as Qwen, milder in magnitude.

## Reading

- **Family-choice sensitivity (eFS4 §4a):** the conclusions do not hinge on the
  16 trained families; the effect reproduces at the same order of magnitude on
  72 distinct held-out families spanning both tiers.
- Very tight CIs on some families (e.g. Qwen `spelling_variant`,
  [+0.0148, +0.0148]) reflect internal template homogeneity: the bootstrap
  resamples near-identical lower-tail pairs. This narrows the interval; it does
  not affect the sign or the macro.
- **Broader blind-spot classes (eFS4 §4b):** the Tier-B result is the direct
  answer — the lift extends to transformation kinds that share no mechanics
  with training (no content-word substitution), which is what the family-blind
  gradient argument (`experiment_100/theory/proposition_family_blind.tex`)
  predicts.
- Scope: where the response is carried by a trivially decodable surface token
  (Qwen number shards), there is no lower-tail headroom — the same boundary as
  E11.

## Files

- `unseen_lower_tail.py` — audit script (reuses E1 `eval_core` + shared metrics).
- `PROTOCOL.md` — pre-registered design and tier rule.
- `results/unseen_lowertail_gpt2.json`, `results/unseen_lowertail_qwen-2.5-3b.json`,
  `results/unseen_lowertail_gemma-2-2b.json`
  — per-family tables (ΔL20, CI, V-Gini, NMSE) + tier macros.
- `results/run_gpt2.log`, `results/run_qwen.log`, `results/run_gemma.log` —
  full run logs.
