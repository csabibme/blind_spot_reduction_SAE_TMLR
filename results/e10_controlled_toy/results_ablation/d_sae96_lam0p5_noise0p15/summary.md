# Controlled non-saturated toy SAE results

| alpha | hidden AUROC | Standard AUROC | V-reg AUROC | Δ AUROC | Δ L20 `||dz||` | Δ L20 rel D | Δ Gini rel | Std MSE | V-reg MSE |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.2000 | 0.6012 | 0.5447 | 0.5461 | +0.0014 | -0.1567 | -0.0784 | +0.0006 | 3.6535 | 4.2530 |
| 0.4000 | 0.6549 | 0.5722 | 0.5878 | +0.0156 | -0.1387 | -0.0638 | -0.0007 | 3.6358 | 4.2212 |
| 0.6000 | 0.6809 | 0.6008 | 0.6197 | +0.0188 | -0.0960 | -0.0385 | -0.0031 | 3.6332 | 4.1735 |
| 0.8000 | 0.7066 | 0.6225 | 0.6484 | +0.0259 | -0.0135 | -0.0008 | -0.0080 | 3.6279 | 4.1186 |
| 1.0000 | 0.7265 | 0.6578 | 0.6727 | +0.0149 | +0.0765 | +0.0324 | -0.0106 | 3.6242 | 4.0542 |

Interpretation: look for intermediate alpha values where hidden is above chance, Standard SAE is sub-ceiling, and V-reg improves held-out probe AUROC and/or lower-tail critical response.
