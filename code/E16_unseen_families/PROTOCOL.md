# E16 — Held-out family generalization audit (protocol)

Fixed before running the evaluation; applied uniformly; all families reported.

## Question

eFS4 asks how sensitive the conclusions are to the choice of perturbation
family, and whether the approach generalizes to broader classes of
representational blind spots.

## Data provenance (why this is not post-hoc)

`experiment_100/data/pairs_100.json` was prepared **before** reviewer feedback
(see `experiment_100/README.md`). Of its 100 families, the 84 template families
were never used in any training run submitted with the paper. They are held-out
family kinds by construction, not families constructed after seeing the reviews.

## Design

- Checkpoints: the frozen general joint-16 Standard and V-reg SAEs
  (`experiment_101_hybrid_owt`), zero-shot. Neither checkpoint saw any of the
  evaluated families during training.
- Extraction: `true_last`, identical to E1/RC-LT (`eval_core.collect_hidden_pairs`).
- Endpoint (per family): ΔL20(s) = L20(s_vreg) − L20(s_std), where
  s = ‖Δz‖/(‖z_orig‖+ε); 95% bootstrap CI over pairs (5000 replicates).
- Sample: up to 30 pairs per family, deterministic per-family seed (42).

## Pre-registered tier rule

- **Excluded**: the 16 trained families themselves.
- **Tier A — unseen instances / lexicons of trained kinds.** Shard variants of
  trained kinds (negation_shard_*, typo_shard_*, synonym_shard_*,
  number_shard_*, word_order_shard_*, semantic_sub_shard_*) and content-word
  substitution families on new lexicons (semantic_*, medical_auto swaps,
  homophone/spelling surface edits). Same mechanics as training, new instances
  and domains.
- **Tier B — kinds unseen in nature.** Operator-, inflection-, structure- and
  discourse-level transformations that are not content-word substitutions:
  modal_swap, hedging_swap, tense_shift, voice_swap, clause_order,
  coreference_swap, conjunction_swap, determiner_quantifier, comparative_swap,
  degree_intensifier, pluralization, article_swap, preposition_swap,
  adverb_swap, contraction_expand, punctuation_change, capitalization,
  polarity_flip.

The exact lists are frozen in `unseen_lower_tail.py` (`TRAINED_16`, `TIER_B`).

## Duplicate audit (data-integrity amendment, disclosed)

A content audit of `pairs_100.json` found duplicate shard families that would
double-count the same measurement in the macro. The audit was run in two
passes, both disclosed:

1. **Order-sensitive pass (before the first evaluation run):** 6 families are
   byte-identical to another family (`negation_shard_03`,
   `semantic_sub_shard_04`, `synonym_shard_04..07`).
2. **Set-level pass (after inspecting the first results):** identical per-family
   L20 values across all `negation_shard_*` revealed that all eight negation
   shards contain the **same 8-pair set** — the shards differ only in pair
   order, which the order-sensitive pass missed. Since L20 is computed on the
   set of pair sensitivities, these are one measurement, not seven.

Rule: one representative per set-equal group is retained
(`negation_shard_00`, `semantic_sub_shard_00`, `synonym_shard_00..03`); the
other 12 families are excluded from the macro (`DUPLICATE_FAMILIES` in the
script). This changes only the aggregation weights, not any per-family value;
headline conclusions are unchanged under either counting.

## Reporting rule

- A family counts as positive when ΔL20(s) > 0 **and** its 95% CI excludes zero.
- Macro per tier: mean ΔL20(s), fraction of families > 0, fraction CI-positive.
- Every held-out family is reported regardless of outcome. A partial or null
  result is reported as a scope finding, not suppressed.
- Order of models: GPT-2 first, then Qwen-2.5-3B. No family- or model-level
  selection after seeing results.

## Framing

Appendix material. The claim carriers remain E2 (task-level extrinsic) and
RC-LT / E14 (real-clinical). E16 contextualizes family-choice sensitivity:
Tier A tests instance/domain generalization of trained kinds, Tier B tests
generalization to family kinds unseen in nature.
