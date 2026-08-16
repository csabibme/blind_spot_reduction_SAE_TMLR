# E3c Targeted Laterality Repair Evaluation

Target checkpoint: `E3c_targeted_laterality_repair/checkpoints/qwen-2.5-3b/qwen_laterality_recon_only_seed42_150`

| Representation | BA | AUROC | F1 | Pair acc | L20 | Logit L20 | Geom L20 |
|---|---:|---:|---:|---:|---:|---:|---:|
| standard_code | 0.7215 | 0.7752 | 0.7208 | 0.9620 | 0.0285 | 0.1786 | 0.1600 |
| zero_shot_vreg_code | 0.6962 | 0.7664 | 0.6958 | 0.9747 | 0.0426 | 0.2752 | 0.2620 |
| targeted_repair_code | 0.7152 | 0.7792 | 0.7138 | 0.9747 | 0.0361 | 0.3543 | 0.3189 |

## Report-Cluster Deltas Vs Standard

| Candidate | Delta L20 | 95% CI | Delta BA | Delta F1 |
|---|---:|---:|---:|---:|
| zero_shot_vreg_code_minus_standard | +0.0141 | [-0.0115, 0.0350] | -0.0253 | -0.0250 |
| targeted_repair_code_minus_standard | +0.0076 | [-0.0010, 0.0165] | -0.0063 | -0.0070 |
