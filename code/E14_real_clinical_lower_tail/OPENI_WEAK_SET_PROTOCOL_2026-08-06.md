# OpenI Standard-defined weak-set audit protocol

**Predeclared:** 2026-08-06, before computation of the results defined here.

**Execution-parameter correction (2026-08-07):** The initial protocol text
incorrectly assigned `max_length=256` to both profiles. The preserved command
that generated the released Qwen Table 9 artifact explicitly used
`max_length=128`; GPT-2 used `max_length=256`. The Qwen value below is corrected
to match that pre-existing canonical execution, before interpreting the paired
audit. A rerun at 128 exactly reproduced every released Qwen own-tail value.

## Scope and estimands

The existing own-tail estimand is retained and reproduced exactly:

`L20(V-reg) - L20(Standard)`, where each model's `L20` is the mean of its own
bottom `ceil(0.20 n)` relative responses `D = ||Delta z|| / (||z_orig|| + eps)`.

The primary added estimand is paired and Standard-defined. Within each
model-by-family cell, rank pairs by Standard relative response `D_std` and let
`W_std` be the bottom `ceil(0.20 n)` pairs. On this one fixed set, report:

- mean `D_std`, mean `D_vreg`, and paired delta `mean(D_vreg - D_std)`;
- mean `||Delta z||_std`, mean `||Delta z||_vreg`, and paired delta
  `mean(||Delta z||_vreg - ||Delta z||_std)`.

The same `W_std` indexes must be used for both relative and absolute endpoints.
Report the overlap of `W_std` with the reverse set `W_vreg`, and the fraction of
selected pairs improved (`V-reg > Standard`) for each endpoint.

## Diagnostics

The reverse-selection diagnostic repeats the paired relative and absolute
means/deltas on `W_vreg`, the bottom `ceil(0.20 n)` pairs ranked by V-reg
relative response.

The Standard-defined profile stably ranks all pairs by `D_std`, partitions that
ordering into five consecutive, exhaustive, disjoint groups with `numpy.array_split`,
and labels them Q1 through Q5. For each quintile report `n`, both model means,
and the paired V-reg-minus-Standard delta for relative `D` and absolute
`||Delta z||`.

All weak-set, reverse-set, quintile, overlap, fraction-improved, and
selection-aware analyses are diagnostic and noncausal because set membership
is coupled to an observed model response.

## Ties and identity

Ranking uses `numpy.argsort(values, kind="stable")`; ties therefore retain the
original selected-input order. Every exported pair has a stable family-local
pair ID, original JSONL row index, image/report cluster ID, selected-input
index, both relative responses, both absolute responses, and membership labels.

## Uncertainty

All intervals are percentile 95% intervals from 5,000 paired resamples.

- Existing own-tail intervals retain the legacy implementation and seed 100.
- Fixed `W_std` panels resample only report/image clusters represented in the
  fixed set, with seed 200.
- Selection-aware sensitivity resamples clusters from the full family, then
  stably reselects the bottom `ceil(0.20 n*)` Standard responses in each
  resample, with seed 300.
- Reverse `W_vreg` panels resample only represented clusters, with seed 400.
- A pair bootstrap is used only when every represented cluster is a singleton.
  Otherwise the report/image cluster bootstrap is used, including the
  degenerate one-cluster case.
- Relative and absolute endpoints use the same sampled indexes within every
  joint panel bootstrap.

## Frozen inputs and execution

- Input: `FINAL/REVISION_1/OpenI/pairs.jsonl`
  (SHA-256 `7ac937e7e6c1096c26fba4e8803f03452d658aa18b619dbbf2e06bb38af90a41`).
- Families, in order: `negation`, `laterality`, `severity`,
  `anatomical_direction`.
- GPT-2: profile `gpt2`, model `gpt2`, layer 12, `max_pairs=300`.
- Qwen: profile `qwen-2.5-3b`, model `Qwen/Qwen2.5-3B`, layer 18,
  `max_pairs=80`, `trust_remote_code=true`.
- Canonical token caps: GPT-2 `max_length=256`; Qwen `max_length=128`.
- Both: extraction protocol `true_last`, `lm_dtype=float16`, hidden batch
  size 16, base sampling seed 42. The confirmatory executions requested
  Apple MPS explicitly.
- Family subsampling seeds:
  `negation=4129248117`, `laterality=3648580089`, `severity=3381613839`,
  `anatomical_direction=3371242558`.
- Standard checkpoints:
  `FINAL/tmlr_revision/prepare/experiment_101_hybrid_owt/checkpoints/{profile}/joint/standard_joint16_owt`.
- V-reg checkpoints:
  `FINAL/tmlr_revision/prepare/experiment_101_hybrid_owt/checkpoints/{profile}/joint/vreg_joint16_owt`.
- Checkpoint metadata SHA-256 (GPT-2 Standard, GPT-2 V-reg, Qwen Standard,
  Qwen V-reg): `893187ad5da4eddee72e64c5cdedf58629255ac95cb7203ed30ca973b44dbb97`,
  `37ee11bf9918fb566366430f7ec7ae41962fabbaa370c29901c579d690a8f8dd`,
  `495d24a3a55f21333ce627b7889664c0d84dd762608e087d8e06bf20b21a29ca`,
  `b036994d88085f7f33b256020abd9a20bbbdcb942043a435b046c5574843c424`.
- Qwen SAE file SHA-256 (Standard, V-reg):
  `3bb65c319592777642f47cbbce2deacf42b8262890be69eeb92fb9259a9925d5`,
  `442710c76d8c4aa47e8c1377474d204d585c5470ba64b0fbdb0b98cf4d51e9d8`.
- Qwen local Hugging Face snapshot:
  `3aab1f1954e9cc14eb9509a215f9e5ca08227a9b`.

Versioned outputs must not replace the existing canonical JSON files.
