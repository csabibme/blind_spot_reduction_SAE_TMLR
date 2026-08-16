# E3c Targeted Laterality Repair Evaluation

Target checkpoint: `E3c_targeted_laterality_repair/checkpoints/gpt2/gpt2_laterality_certificate_rel002_seed42`

| Representation | BA | AUROC | F1 | Pair acc | L20 | Logit L20 | Geom L20 |
|---|---:|---:|---:|---:|---:|---:|---:|
| standard_code | 0.8291 | 0.9172 | 0.8283 | 0.9873 | 0.1578 | 1.0398 | 0.9414 |
| zero_shot_vreg_code | 0.8734 | 0.9112 | 0.8733 | 0.9873 | 0.1600 | 1.0022 | 0.9395 |
| targeted_repair_code | 0.8544 | 0.9172 | 0.8543 | 1.0000 | 0.1493 | 1.2507 | 1.2054 |

## Report-Cluster Deltas Vs Standard

| Candidate | Delta L20 | 95% CI | Delta BA | Delta F1 |
|---|---:|---:|---:|---:|
| zero_shot_vreg_code_minus_standard | +0.0022 | [-0.0351, 0.0295] | +0.0443 | +0.0451 |
| targeted_repair_code_minus_standard | -0.0085 | [-0.0478, 0.0384] | +0.0253 | +0.0260 |
