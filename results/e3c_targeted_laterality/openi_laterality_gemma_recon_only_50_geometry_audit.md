# E3c Direct Geometry And Fidelity Audit

Target checkpoint: `E3c_targeted_laterality_repair/checkpoints/gemma-2-2b/gemma_laterality_recon_only_seed42_50`

## Test Geometry

| Representation | Distance L20 | Relative L20 | Distance mean | Relative mean | Mean code norm |
|---|---:|---:|---:|---:|---:|
| standard | 5.0872 | 0.0272 | 21.5909 | 0.1226 | 179.0969 |
| zero_shot_vreg | 4.6352 | 0.0318 | 19.8009 | 0.1398 | 142.4455 |
| targeted | 5.0676 | 0.0275 | 21.5090 | 0.1239 | 176.6187 |

## Test Delta Vs Standard

| Candidate | Distance tail delta | Distance mean delta | Distance improved frac | Relative tail delta | Relative improved frac |
|---|---:|---:|---:|---:|---:|
| zero_shot_vreg | -0.4290 | -1.7900 | 0.025 | +0.0046 | 1.000 |
| targeted | -0.0196 | -0.0820 | 0.000 | +0.0003 | 1.000 |

## Fidelity

| Representation | OWT NMSE | OWT EV | OWT L0 | OpenI test NMSE | OpenI test EV | OpenI test L0 |
|---|---:|---:|---:|---:|---:|---:|
| standard | 0.0017 | 0.9983 | 3666.5 | 0.0105 | 0.9895 | 3617.1 |
| zero_shot_vreg | 0.0025 | 0.9975 | 3694.5 | 0.0640 | 0.9360 | 2919.8 |
| targeted | 0.0008 | 0.9992 | 3645.9 | 0.0098 | 0.9902 | 3598.0 |
