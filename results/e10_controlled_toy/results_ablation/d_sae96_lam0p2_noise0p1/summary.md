# Controlled non-saturated toy SAE results

| alpha | hidden AUROC | Standard AUROC | V-reg AUROC | Δ AUROC | Δ L20 `||dz||` | Δ L20 rel D | Δ Gini rel | Std MSE | V-reg MSE |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.2000 | 0.6072 | 0.5399 | 0.5515 | +0.0116 | -0.0275 | -0.0201 | -0.0028 | 4.7967 | 5.3133 |
| 0.4000 | 0.6553 | 0.5752 | 0.5850 | +0.0098 | -0.0077 | -0.0026 | -0.0076 | 4.7780 | 5.2607 |
| 0.6000 | 0.6834 | 0.6029 | 0.6117 | +0.0088 | +0.0220 | +0.0139 | -0.0097 | 4.7670 | 5.1834 |
| 0.8000 | 0.7063 | 0.6330 | 0.6458 | +0.0128 | +0.0727 | +0.0378 | -0.0119 | 4.7489 | 5.0969 |
| 1.0000 | 0.7264 | 0.6646 | 0.6630 | -0.0016 | +0.1393 | +0.0606 | -0.0133 | 4.7289 | 5.0179 |

Interpretation: look for intermediate alpha values where hidden is above chance, Standard SAE is sub-ceiling, and V-reg improves held-out probe AUROC and/or lower-tail critical response.
