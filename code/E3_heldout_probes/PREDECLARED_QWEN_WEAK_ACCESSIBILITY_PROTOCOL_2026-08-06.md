# Predeclared Qwen E3 cross-fitted weak/non-weak semantic accessibility audit

Date frozen: **2026-08-06**, before generation of the results covered by this protocol.

## Frozen inputs and estimand

- Source feature cache: the existing frozen Qwen cache `qwen-2.5-3b_70f78770026d89ac_4223d08c6a767b9c.npz`.
- Task data: the existing frozen 78-pair E3 task split. No examples, labels, families, templates, or feature vectors will be regenerated or altered.
- The audit is offline: it will not load a language model or SAE.
- For pair \(i\) and each representation separately, semantic displacement is
  \(D_i=\lVert z_{\mathrm{pert}}-z_{\mathrm{orig}}\rVert_2/
  (\lVert z_{\mathrm{orig}}\rVert_2+\epsilon)\), with \(\epsilon=10^{-8}\).

Protocol erratum recorded 2026-08-07: the first frozen draft printed
`10^-12` here, while the manuscript and the pre-existing shared metric
implementation both fix epsilon at `10^-8`. The executable and protocol were
corrected to the already-published definition before final interpretation.
This correction is not selected from the audit outcome.

## Cross-fitted probes

- All 78 pairs enter a deterministic five-fold outer cross-fit.
- Outer folds are template-grouped; both pair sides and every member sharing a template are always assigned together. Group integrity is absolute. Family stratification is used where feasible.
- Standard and V-reg use exactly the same outer folds.
- A separate balanced logistic-regression probe is fit for the Standard and V-reg representations. Accordingly, their comparison estimates representation-plus-readout performance, not a representation-only causal effect.
- In every outer fold, the final standard scaler is fit on outer-training examples only. During nested selection, each candidate is evaluated with a scaler fit only on that inner-training partition.
- Logistic-regression `C` is nested-selected inside each outer-training set from the existing `C_GRID`. Inner validation is template-grouped, keeps all pair sides/template members together, and uses family stratification where feasible.
- The predeclared primary `C` selection criterion is mean inner-fold pair correctness (probability paired margin \(>0\)); ties select the first value in `C_GRID` (strongest regularization in the existing ordering).
- Held-out outer-fold labels are never used for scaling, `C` selection, or fitting.
- Every example receives exactly one out-of-fold prediction; every pair receives one out-of-fold probability paired margin.

## Predeclared subsets and endpoints

- Primary weak set \(W_{\mathrm{std}}\): the 16 pairs with the smallest Standard \(D_i\), because \(\lceil0.20\times78\rceil=16\). Remaining pairs are non-weak. Stable ties are resolved by pair ID.
- Sensitivity weak set: within each family, the bottom \(\lceil20\%\rceil\) by Standard \(D_i\), with stable pair-ID tie breaking.
- Primary endpoints on each fixed subset are mean probability paired margin and pair correctness/error (paired margin \(>0\) / \(\le0\)).
- Association endpoint: Standard weak versus non-weak differences in Standard probability paired margin and pair correctness/error.
- Intervention-comparison endpoint: paired Standard-to-V-reg changes on the fixed Standard-defined subsets.
- Uncertainty uses a deterministic 5,000-replicate template-cluster bootstrap. Templates, including all their pairs, are the resampling unit.
- Selection-aware sensitivity repeats weak-set selection within each bootstrap replicate before computing Standard-to-V-reg differences.

## Predeclared diagnostics

- Reverse analysis defines \(W_{\mathrm{vreg}}\) from the bottom 16 V-reg displacements and reports Standard/V-reg endpoints and paired changes on that subset.
- Standard-displacement quintiles Q1--Q5 are formed deterministically in ascending \(D_{\mathrm{std}}\) order, with sizes differing by at most one, and report the endpoint profile.
- For every fixed subset comparison, McNemar cells are reported as: both correct, Standard-only correct, V-reg-only correct, both wrong. The exact two-sided binomial p-value is computed on discordant pairs.

## Leakage and interpretation limits

- The implementation must verify index alignment, identical outer folds across representations, exactly-once out-of-fold prediction, and no template/pair overlap across outer or inner train/test partitions.
- Standard \(D_i\) and Standard out-of-fold margins share a representation source. Their association is descriptive and may reflect common-source dependence.
- Standard and V-reg use separate representation-specific probes. Differences therefore combine representation, fitted readout, and probability calibration; probability-margin changes are not pure geometry effects.
- The 78 frozen pairs and their template/family structure bound the scope of inference.
