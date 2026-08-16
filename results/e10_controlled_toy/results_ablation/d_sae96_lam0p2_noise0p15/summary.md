# Controlled non-saturated toy SAE results

| alpha | hidden AUROC | Standard AUROC | V-reg AUROC | Δ AUROC | Δ L20 `||dz||` | Δ L20 rel D | Δ Gini rel | Std MSE | V-reg MSE |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.2000 | 0.6012 | 0.5447 | 0.5448 | +0.0000 | -0.0957 | -0.0480 | -0.0023 | 3.6535 | 3.9197 |
| 0.4000 | 0.6549 | 0.5722 | 0.5875 | +0.0153 | -0.0824 | -0.0364 | -0.0037 | 3.6358 | 3.8999 |
| 0.6000 | 0.6809 | 0.6008 | 0.6120 | +0.0112 | -0.0337 | -0.0107 | -0.0059 | 3.6332 | 3.8756 |
| 0.8000 | 0.7066 | 0.6225 | 0.6366 | +0.0141 | +0.0437 | +0.0221 | -0.0088 | 3.6279 | 3.8499 |
| 1.0000 | 0.7265 | 0.6578 | 0.6663 | +0.0085 | +0.1328 | +0.0527 | -0.0105 | 3.6242 | 3.8022 |

Interpretation: look for intermediate alpha values where hidden is above chance, Standard SAE is sub-ceiling, and V-reg improves held-out probe AUROC and/or lower-tail critical response.
