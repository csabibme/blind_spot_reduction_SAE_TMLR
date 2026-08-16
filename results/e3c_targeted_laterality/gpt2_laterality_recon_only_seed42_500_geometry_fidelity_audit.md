# E3c Direct Geometry And Fidelity Audit

Target checkpoint: `E3c_targeted_laterality_repair/checkpoints/gpt2/gpt2_laterality_recon_only_seed42_500`

## Test Geometry

| Representation | Distance L20 | Relative L20 | Distance mean | Relative mean | Mean code norm |
|---|---:|---:|---:|---:|---:|
| standard | 1.3564 | 0.0100 | 3.9937 | 0.0311 | 132.0248 |
| zero_shot_vreg | 1.7530 | 0.0132 | 4.7272 | 0.0392 | 125.2018 |
| targeted | 1.3567 | 0.0099 | 3.9978 | 0.0311 | 132.3967 |

## Test Delta Vs Standard

| Candidate | Distance tail delta | Distance mean delta | Distance improved frac | Relative tail delta | Relative improved frac |
|---|---:|---:|---:|---:|---:|
| zero_shot_vreg | +0.4486 | +0.7335 | 0.886 | +0.0039 | 0.962 |
| targeted | +0.0003 | +0.0042 | 0.595 | -0.0000 | 0.266 |

## Fidelity

| Representation | OWT NMSE | OWT EV | OWT L0 | OpenI test NMSE | OpenI test EV | OpenI test L0 |
|---|---:|---:|---:|---:|---:|---:|
| standard | 0.0001 | 0.9999 | 2373.3 | 0.0000 | 1.0000 | 2376.5 |
| zero_shot_vreg | 0.0001 | 0.9999 | 2397.4 | 0.0001 | 0.9999 | 2403.6 |
| targeted | 0.0000 | 1.0000 | 2390.2 | 0.0000 | 1.0000 | 2395.8 |
