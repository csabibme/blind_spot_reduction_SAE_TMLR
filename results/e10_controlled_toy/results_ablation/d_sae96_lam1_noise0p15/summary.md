# Controlled non-saturated toy SAE results

| alpha | hidden AUROC | Standard AUROC | V-reg AUROC | Δ AUROC | Δ L20 `||dz||` | Δ L20 rel D | Δ Gini rel | Std MSE | V-reg MSE |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.2000 | 0.6012 | 0.5447 | 0.5436 | -0.0011 | -0.2266 | -0.1132 | +0.0058 | 3.6535 | 5.0225 |
| 0.4000 | 0.6549 | 0.5722 | 0.5921 | +0.0199 | -0.1914 | -0.0885 | +0.0022 | 3.6358 | 4.9120 |
| 0.6000 | 0.6809 | 0.6008 | 0.6303 | +0.0295 | -0.1167 | -0.0464 | -0.0028 | 3.6332 | 4.7329 |
| 0.8000 | 0.7066 | 0.6225 | 0.6619 | +0.0394 | -0.0450 | -0.0140 | -0.0051 | 3.6279 | 4.5522 |
| 1.0000 | 0.7265 | 0.6578 | 0.6968 | +0.0390 | +0.0269 | +0.0132 | -0.0055 | 3.6242 | 4.4266 |

Interpretation: look for intermediate alpha values where hidden is above chance, Standard SAE is sub-ceiling, and V-reg improves held-out probe AUROC and/or lower-tail critical response.
