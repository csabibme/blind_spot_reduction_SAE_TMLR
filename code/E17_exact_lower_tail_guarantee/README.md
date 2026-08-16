# E17 — Exact finite-family lower-tail guarantee

This package verifies the exact worst-subset interpretation of the lower-tail
endpoint used in the paper. It also checks equality of the pairwise, sorted,
and cumulative-share forms of the Gini coefficient, including the factor of
two induced by double-counting unordered pairs. It performs no model inference
and consumes the full-precision per-pair OpenI artifacts produced by E14.

## Statement

For a finite response profile \(D=(D_1,\ldots,D_n)\), let
\(m=\lceil0.20n\rceil\). Then

\[
L_{20}(D)
=
\min_{\substack{W\subseteq\{1,\ldots,n\}\\|W|=m}}
\frac1m\sum_{i\in W}D_i.
\]

If \(W_S\) is any Standard minimizer, then

\[
\overline D_V(W_S)-\overline D_S(W_S)
\ge
L_{20}(D_V)-L_{20}(D_S).
\]

The guarantee assumes the same finite evaluated pair universe, equal pair
weights, the same response definition, and the same tail cardinality. It is a
subset-average statement, not a claim that every pair in \(W_S\) improves.

The relative fixed-set point estimate is therefore a deterministic consequence
of the independently computed own-tail endpoint. The absolute
\(\|\Delta z\|\) endpoint, individual-pair improvement fractions, confidence
intervals, and downstream probe outcomes remain separate empirical evidence.

## Lorenz–Gini relation

For a nonzero response profile, define
\(p_i=D_i/\sum_jD_j\) and the cumulative lower-tail shares
\(\Lambda_k=\sum_{i=1}^k p_{(i)}\). The verifier also checks the standard
discrete identity

\[
V_{\mathrm{Gini}}
=
\frac{n-1}{n}
-
\frac{2}{n}\sum_{k=1}^{n-1}\Lambda_k.
\]

This aggregate identity does not imply pointwise Lorenz dominance or identify
which perturbation carries a response.

## Reproduction

From the repository root:

```bash
python code/E17_exact_lower_tail_guarantee/verify_exact_lower_tail.py
python -m pytest -q code/E17_exact_lower_tail_guarantee/tests
```

Outputs are written to
`results/e17_exact_lower_tail_guarantee/`.
The invariant is evaluated from full-precision per-pair arrays, never from
six-decimal summary fields.
