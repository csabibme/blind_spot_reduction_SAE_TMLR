# Controlled non-saturated toy SAE results

| alpha | hidden AUROC | Standard AUROC | V-reg AUROC | Δ AUROC | Δ L20 `||dz||` | Δ L20 rel D | Δ Gini rel | Std MSE | V-reg MSE |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.2000 | 0.6072 | 0.5399 | 0.5454 | +0.0055 | -0.0488 | -0.0347 | -0.0001 | 4.7967 | 5.8238 |
| 0.4000 | 0.6553 | 0.5752 | 0.5895 | +0.0143 | -0.0196 | -0.0118 | -0.0047 | 4.7780 | 5.6993 |
| 0.6000 | 0.6834 | 0.6029 | 0.6172 | +0.0144 | +0.0251 | +0.0158 | -0.0069 | 4.7670 | 5.5313 |
| 0.8000 | 0.7063 | 0.6330 | 0.6614 | +0.0284 | +0.0641 | +0.0327 | -0.0079 | 4.7489 | 5.4140 |
| 1.0000 | 0.7264 | 0.6646 | 0.6854 | +0.0208 | +0.1358 | +0.0584 | -0.0110 | 4.7289 | 5.3123 |

Interpretation: look for intermediate alpha values where hidden is above chance, Standard SAE is sub-ceiling, and V-reg improves held-out probe AUROC and/or lower-tail critical response.
