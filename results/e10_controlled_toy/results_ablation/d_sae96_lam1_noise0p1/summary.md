# Controlled non-saturated toy SAE results

| alpha | hidden AUROC | Standard AUROC | V-reg AUROC | Δ AUROC | Δ L20 `||dz||` | Δ L20 rel D | Δ Gini rel | Std MSE | V-reg MSE |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.2000 | 0.6072 | 0.5399 | 0.5402 | +0.0003 | -0.0452 | -0.0333 | +0.0037 | 4.7967 | 6.2487 |
| 0.4000 | 0.6553 | 0.5752 | 0.5926 | +0.0173 | -0.0102 | -0.0044 | -0.0041 | 4.7780 | 6.1445 |
| 0.6000 | 0.6834 | 0.6029 | 0.6178 | +0.0150 | +0.0470 | +0.0290 | -0.0089 | 4.7670 | 6.0198 |
| 0.8000 | 0.7063 | 0.6330 | 0.6640 | +0.0311 | +0.1026 | +0.0516 | -0.0102 | 4.7489 | 5.8300 |
| 1.0000 | 0.7264 | 0.6646 | 0.6902 | +0.0257 | +0.1582 | +0.0686 | -0.0071 | 4.7289 | 5.6823 |

Interpretation: look for intermediate alpha values where hidden is above chance, Standard SAE is sub-ceiling, and V-reg improves held-out probe AUROC and/or lower-tail critical response.
