# Qwen E3 cross-fitted weak/non-weak semantic accessibility audit

Protocol date: `2026-08-06`  
Analysis rerun date: `2026-08-08`  
Pairs/templates: 78/48  
Outer folds / bootstrap replicates: 5/5000  
No-leakage checks: `PASS`

## Fixed Standard-defined subsets

| Subset | n | Method | Mean probability margin | Pair correctness | Pair error |
|---|---:|---|---:|---:|---:|
| pooled_Wstd | 16 | standard | 0.302213 | 0.938 (15/16) | 0.062 |
| pooled_Wstd | 16 | vreg | 0.357597 | 1.000 (16/16) | 0.000 |
| pooled_Wstd | 16 | V-reg−Standard | 0.055384 [-0.087147, 0.257443] | +0.062 | -0.062 |
| pooled_nonweak | 62 | standard | 0.589584 | 0.887 (55/62) | 0.113 |
| pooled_nonweak | 62 | vreg | 0.575493 | 0.968 (60/62) | 0.032 |
| pooled_nonweak | 62 | V-reg−Standard | -0.014090 [-0.123362, 0.117782] | +0.081 | -0.081 |
| within_family_Wstd | 16 | standard | 0.338102 | 1.000 (16/16) | 0.000 |
| within_family_Wstd | 16 | vreg | 0.340171 | 1.000 (16/16) | 0.000 |
| within_family_Wstd | 16 | V-reg−Standard | 0.002069 [-0.105445, 0.145084] | +0.000 | +0.000 |

## Paired correctness transitions on fixed weak sets

| Subset | both correct | Standard only | V-reg only | both wrong | exact McNemar p |
|---|---:|---:|---:|---:|---:|
| pooled_Wstd | 15 | 0 | 1 | 0 | 1.0000 |
| within_family_Wstd | 16 | 0 | 0 | 0 | 1.0000 |

## Standard weak versus non-weak association

| Definition | Group | n | Mean margin | Pair correctness | Pair error |
|---|---|---:|---:|---:|---:|
| pooled_Wstd | weak | 16 | 0.302213 | 0.938 | 0.062 |
| pooled_Wstd | nonweak | 62 | 0.589584 | 0.887 | 0.113 |
| pooled_Wstd | weak−nonweak | — | -0.287370 [-0.508607, -0.065577] | +0.050 | -0.050 |
| within_family_Wstd | weak | 16 | 0.338102 | 1.000 | 0.000 |
| within_family_Wstd | nonweak | 62 | 0.580322 | 0.871 | 0.129 |
| within_family_Wstd | weak−nonweak | — | -0.242221 [-0.438007, -0.032071] | +0.129 | -0.129 |

## Standard displacement quintiles

| Quintile | n | Std margin | V-reg margin | Delta | Std correct | V-reg correct |
|---|---:|---:|---:|---:|---:|---:|
| Q1 | 16 | 0.302213 | 0.357597 | +0.055384 | 0.938 | 1.000 |
| Q2 | 16 | 0.526107 | 0.713468 | +0.187360 | 1.000 | 1.000 |
| Q3 | 16 | 0.568851 | 0.423310 | -0.145541 | 0.812 | 0.938 |
| Q4 | 15 | 0.528705 | 0.556216 | +0.027510 | 0.800 | 0.933 |
| Q5 | 15 | 0.740285 | 0.609928 | -0.130357 | 0.933 | 1.000 |

## Reverse diagnostic

`W_vreg` n=16: Standard margin 0.389156, V-reg margin 0.535937, delta +0.146781 [-0.130906, 0.531351].

## Interpretation limits

- Standard displacement and Standard margin share a representation source.
- Separate representation-specific probes make this a representation-plus-readout comparison.
- Probability margins also include calibration differences.
