# Controlled non-saturated toy SAE results

| alpha | hidden AUROC | Standard AUROC | V-reg AUROC | Δ AUROC | Δ L20 `||dz||` | Δ L20 rel D | Δ Gini rel | Std MSE | V-reg MSE |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.2000 | 0.6087 | 0.5536 | 0.5568 | +0.0032 | -0.0594 | -0.0350 | -0.0009 | 2.8336 | 3.2321 |
| 0.4000 | 0.6560 | 0.5780 | 0.5969 | +0.0189 | -0.0167 | -0.0060 | -0.0034 | 2.8343 | 3.1610 |
| 0.6000 | 0.6820 | 0.6059 | 0.6246 | +0.0188 | +0.0498 | +0.0278 | -0.0041 | 2.8279 | 3.0689 |
| 0.8000 | 0.7051 | 0.6429 | 0.6665 | +0.0237 | +0.1396 | +0.0649 | -0.0048 | 2.8194 | 3.0105 |
| 1.0000 | 0.7265 | 0.6609 | 0.6851 | +0.0242 | +0.2588 | +0.1063 | -0.0068 | 2.8090 | 2.9618 |

Interpretation: look for intermediate alpha values where hidden is above chance, Standard SAE is sub-ceiling, and V-reg improves held-out probe AUROC and/or lower-tail critical response.
