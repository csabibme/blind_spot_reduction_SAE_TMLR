# E3c Targeted Laterality Repair Evaluation

Target checkpoint: `E3c_targeted_laterality_repair/checkpoints/gemma-2-2b/gemma_laterality_recon_only_seed42_50`

| Representation | BA | AUROC | F1 | Pair acc | L20 | Logit L20 | Geom L20 |
|---|---:|---:|---:|---:|---:|---:|---:|
| standard_code | 0.7658 | 0.8253 | 0.7656 | 0.9747 | 0.0823 | 0.6346 | 0.5530 |
| zero_shot_vreg_code | 0.7532 | 0.7920 | 0.7532 | 0.9241 | 0.0077 | -0.6316 | -0.5511 |
| targeted_repair_code | 0.7532 | 0.8286 | 0.7527 | 0.9747 | 0.0840 | 0.6404 | 0.5580 |

## Report-Cluster Deltas Vs Standard

| Candidate | Delta L20 | 95% CI | Delta BA | Delta F1 |
|---|---:|---:|---:|---:|
| zero_shot_vreg_code_minus_standard | -0.0747 | [-0.1630, -0.0120] | -0.0127 | -0.0124 |
| targeted_repair_code_minus_standard | +0.0017 | [-0.0013, 0.0053] | -0.0127 | -0.0129 |
