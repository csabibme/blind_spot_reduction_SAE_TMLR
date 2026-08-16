# Held-out dosage numeric VALUE-accessibility probe (confound-free)

Single-side probe: classify high-dose (value >= 6) vs low-dose (value <= 5) from one representation, trained on train templates, evaluated on disjoint test templates. Avoids the edit-magnitude confound of the pair-difference probe.

## Test AUROC by profile

| Profile | hidden | Standard | V-reg | Δ AUROC (V-reg−Std) | 95% CI (template cluster) |
|---|---:|---:|---:|---:|---|
| gpt2 | 0.9803 | 0.9722 | 0.8733 | -0.0990 | [-0.1607, +0.0062] |
| gemma-2-2b | 0.8651 | 0.8518 | 0.8125 | -0.0393 | [-0.1058, +0.0335] |
| qwen-2.5-3b | 0.9208 | 0.9121 | 0.7558 | -0.1563 | [-0.2827, -0.0597] |

## Balanced accuracy (test)

| Profile | hidden | Standard | V-reg | Δ BA |
|---|---:|---:|---:|---:|
| gpt2 | 0.9282 | 0.9079 | 0.8120 | -0.0958 |
| gemma-2-2b | 0.8106 | 0.7630 | 0.7444 | -0.0185 |
| qwen-2.5-3b | 0.8185 | 0.8310 | 0.7139 | -0.1171 |

