# E3c Direct Geometry And Fidelity Audit

Target checkpoint: `E3c_targeted_laterality_repair/checkpoints/gpt2/gpt2_laterality_certificate_rel002_seed42`

## Test Geometry

| Representation | Distance L20 | Relative L20 | Distance mean | Relative mean | Mean code norm |
|---|---:|---:|---:|---:|---:|
| standard | 1.3564 | 0.0100 | 3.9937 | 0.0311 | 132.0248 |
| zero_shot_vreg | 1.7530 | 0.0132 | 4.7272 | 0.0392 | 125.2018 |
| targeted | 3.1248 | 0.0230 | 5.9229 | 0.0463 | 130.4225 |

## Test Delta Vs Standard

| Candidate | Distance tail delta | Distance mean delta | Distance improved frac | Relative tail delta | Relative improved frac |
|---|---:|---:|---:|---:|---:|
| zero_shot_vreg | +0.4486 | +0.7335 | 0.886 | +0.0039 | 0.962 |
| targeted | +1.8990 | +1.9293 | 0.899 | +0.0141 | 0.899 |

## Fidelity

| Representation | OWT NMSE | OWT EV | OWT L0 | OpenI test NMSE | OpenI test EV | OpenI test L0 |
|---|---:|---:|---:|---:|---:|---:|
| standard | 0.0001 | 0.9999 | 2361.2 | 0.0000 | 1.0000 | 2376.5 |
| zero_shot_vreg | 0.0001 | 0.9999 | 2385.3 | 0.0001 | 0.9999 | 2403.6 |
| targeted | 0.0000 | 1.0000 | 2378.6 | 0.0001 | 0.9999 | 2384.9 |
