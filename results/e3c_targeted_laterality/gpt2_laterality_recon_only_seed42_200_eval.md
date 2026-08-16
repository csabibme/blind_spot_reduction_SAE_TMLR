# E3c Targeted Laterality Repair Evaluation

Target checkpoint: `E3c_targeted_laterality_repair/checkpoints/gpt2/gpt2_laterality_recon_only_seed42_200`

| Representation | BA | AUROC | F1 | Pair acc | L20 | Logit L20 | Geom L20 |
|---|---:|---:|---:|---:|---:|---:|---:|
| standard_code | 0.8291 | 0.9172 | 0.8283 | 0.9873 | 0.1578 | 1.0398 | 0.9414 |
| zero_shot_vreg_code | 0.8734 | 0.9112 | 0.8733 | 0.9873 | 0.1600 | 1.0022 | 0.9395 |
| targeted_repair_code | 0.8291 | 0.9258 | 0.8283 | 1.0000 | 0.1779 | 1.1535 | 1.0463 |

## Report-Cluster Deltas Vs Standard

| Candidate | Delta L20 | 95% CI | Delta BA | Delta F1 |
|---|---:|---:|---:|---:|
| zero_shot_vreg_code_minus_standard | +0.0022 | [-0.0343, 0.0302] | +0.0443 | +0.0451 |
| targeted_repair_code_minus_standard | +0.0200 | [-0.0002, 0.0447] | +0.0000 | +0.0000 |
