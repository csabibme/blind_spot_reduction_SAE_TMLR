# Held-out-template negation probe (TEST templates unseen by V-loss)

| Representation | test AUROC | test BA | C |
|---|---:|---:|---:|
| hidden | 0.9954 | 0.8333 | 0.01 |
| sae_standard_code | 0.9954 | 0.8611 | 0.01 |
| sae_vreg_code | 0.9799 | 0.8333 | 0.01 |
| sae_standard_reconstruction | 0.9954 | 0.8333 | 0.01 |
| sae_vreg_reconstruction | 0.9684 | 0.8333 | 0.01 |

Standard code AUROC -> V-reg code AUROC: 0.9954 -> 0.9799 (Δ -0.0155)

Standard code BA -> V-reg code BA: 0.8611 -> 0.8333 (Δ -0.0278)
