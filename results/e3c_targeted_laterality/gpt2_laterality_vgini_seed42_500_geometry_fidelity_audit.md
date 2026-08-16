# E3c Direct Geometry And Fidelity Audit

Target checkpoint: `E3c_targeted_laterality_repair/checkpoints/gpt2/gpt2_laterality_vgini_seed42_500`

## Test Geometry

| Representation | Distance L20 | Relative L20 | Distance mean | Relative mean | Mean code norm |
|---|---:|---:|---:|---:|---:|
| standard | 1.3564 | 0.0100 | 3.9937 | 0.0311 | 132.0248 |
| zero_shot_vreg | 1.7530 | 0.0132 | 4.7272 | 0.0392 | 125.2018 |
| targeted | 3.7110 | 0.0281 | 6.1303 | 0.0484 | 128.3072 |

## Test Delta Vs Standard

| Candidate | Distance tail delta | Distance mean delta | Distance improved frac | Relative tail delta | Relative improved frac |
|---|---:|---:|---:|---:|---:|
| zero_shot_vreg | +0.4486 | +0.7335 | 0.886 | +0.0039 | 0.962 |
| targeted | +2.6354 | +2.1366 | 0.848 | +0.0200 | 0.886 |

## Fidelity

| Representation | OWT NMSE | OWT EV | OWT L0 | OpenI test NMSE | OpenI test EV | OpenI test L0 |
|---|---:|---:|---:|---:|---:|---:|
| standard | 0.0001 | 0.9999 | 2373.3 | 0.0000 | 1.0000 | 2376.5 |
| zero_shot_vreg | 0.0001 | 0.9999 | 2397.4 | 0.0001 | 0.9999 | 2403.6 |
| targeted | 0.0000 | 1.0000 | 2386.1 | 0.0003 | 0.9997 | 2368.1 |
