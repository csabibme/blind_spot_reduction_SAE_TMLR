# E1 — frozen metric specification v1.1

**Version:** 1.1 (amended 2026-06-19)  
**Change from v1.0:** primary inference moved from Standard-bottom paired lift to ΔL20.

## Primary family-level comparison

\[
\Delta L_{20}(s) = L_{20}(s^{\mathrm{vreg}}) - L_{20}(s^{\mathrm{std}})
\]

Ask: is the **weakest 20% of the full V-reg distribution** stronger than the weakest 20% of Standard?

Also report paired bootstrap CI on:

- Δmean(s)
- ΔL20(s) **(primary)**
- ΔV_Gini(raw)
- ΔL20(g)

## Diagnostic (secondary — not primary inference)

Standard-bottom paired lift: mean(s_vreg - s_std) on pairs in Standard's lowest 20%.
**Selection-biased** — use for scatter plots only.

## Per-method aggregates (each SAE alone)

mean, median, Q0.10, L20, min (diagnostic), V_Gini on s and g; MSE (forward path, float32); L0; inactive_frac_mean; code_norm_mean.

## Pipeline (memory)

1. LM → hidden cache on CPU → unload LM  
2. Standard SAE → eval → unload  
3. V-reg SAE → eval (same hidden states)

## MSE

Use `sae.forward(h)` path; compute `F.mse_loss(x_hat.float(), h.float())`.

## Incremental save

JSON written after each family and each profile (atomic replace).
