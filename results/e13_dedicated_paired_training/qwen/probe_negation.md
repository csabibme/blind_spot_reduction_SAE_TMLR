# Held-out-template negation probe (TEST templates unseen by V-loss)

| Representation | test AUROC | test BA | C |
|---|---:|---:|---:|
| hidden | 1.0 | 1.0 | 0.01 |
| sae_standard_code | 1.0 | 1.0 | 0.01 |
| sae_vreg_code | 0.811 | 0.75 | 1.0 |
| sae_standard_reconstruction | 1.0 | 1.0 | 0.01 |
| sae_vreg_reconstruction | 1.0 | 0.9861 | 0.01 |

Standard code AUROC -> V-reg code AUROC: 1.0 -> 0.811 (Δ -0.189)

Standard code BA -> V-reg code BA: 1.0 -> 0.75 (Δ -0.25)
