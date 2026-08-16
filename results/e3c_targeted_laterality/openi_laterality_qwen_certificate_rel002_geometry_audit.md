# E3c Direct Geometry And Fidelity Audit

Target checkpoint: `E3c_targeted_laterality_repair/checkpoints/qwen-2.5-3b/qwen_laterality_certificate_rel002_seed42`

## Test Geometry

| Representation | Distance L20 | Relative L20 | Distance mean | Relative mean | Mean code norm |
|---|---:|---:|---:|---:|---:|
| standard | 1.5906 | 0.0196 | 6.6118 | 0.0797 | 82.4956 |
| zero_shot_vreg | 1.4452 | 0.0261 | 5.9924 | 0.1010 | 57.4690 |
| targeted | 1.8956 | 0.0277 | 5.8269 | 0.0797 | 70.8657 |

## Test Delta Vs Standard

| Candidate | Distance tail delta | Distance mean delta | Distance improved frac | Relative tail delta | Relative improved frac |
|---|---:|---:|---:|---:|---:|
| zero_shot_vreg | -0.1309 | -0.6194 | 0.152 | +0.0067 | 0.975 |
| targeted | +0.3086 | -0.7849 | 0.671 | +0.0084 | 0.835 |

## Fidelity

| Representation | OWT NMSE | OWT EV | OWT L0 | OpenI test NMSE | OpenI test EV | OpenI test L0 |
|---|---:|---:|---:|---:|---:|---:|
| standard | 0.0016 | 0.9984 | 3365.3 | 0.0079 | 0.9921 | 3312.2 |
| zero_shot_vreg | 0.0013 | 0.9987 | 3364.8 | 0.1196 | 0.8803 | 2016.3 |
| targeted | 0.0003 | 0.9997 | 3359.1 | 0.0444 | 0.9556 | 3216.5 |
