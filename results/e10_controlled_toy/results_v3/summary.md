# Controlled non-saturated toy SAE results

| alpha | hidden AUROC | Standard AUROC | V-reg AUROC | Δ AUROC | Δ L20 `||dz||` | Δ L20 rel D | Δ Gini rel | Std MSE | V-reg MSE |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.8000 | 0.7060 | 0.6444 | 0.6574 | +0.0130 | +0.1530 | +0.0766 | -0.0034 | 3.1467 | 3.7698 |
| 1.2000 | 0.7488 | 0.6859 | 0.7099 | +0.0241 | +0.3314 | +0.1221 | +0.0003 | 3.1272 | 3.5054 |
| 1.6000 | 0.7873 | 0.7252 | 0.7417 | +0.0164 | +0.5076 | +0.1470 | +0.0056 | 3.1157 | 3.3806 |
| 2.0000 | 0.8222 | 0.7701 | 0.7786 | +0.0085 | +0.7538 | +0.1792 | +0.0025 | 3.1008 | 3.2860 |
| 2.5000 | 0.8807 | 0.8183 | 0.8382 | +0.0199 | +1.0259 | +0.1988 | +0.0034 | 3.0972 | 3.2210 |

Interpretation: look for intermediate alpha values where hidden is above chance, Standard SAE is sub-ceiling, and V-reg improves held-out probe AUROC and/or lower-tail critical response.
