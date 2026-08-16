# Rev3 pre-flight headroom screen

Gate: hidden AUROC ≥ 0.7 (signal present) AND Standard SAE-code AUROC ≤ 0.95 (sub-ceiling headroom).
Splits per point: 5 template-held-out (pair-grouped) splits, mean reported.

## severity_change (n_pairs=30, n_sides=60)

| Model | hidden AUROC | Standard code AUROC | V-reg code AUROC | Δ AUROC (V−S) | headroom? |
|---|---:|---:|---:|---:|:--:|
| gpt2 | 0.917 | 0.922 | 0.767 | -0.156 | yes |
| gemma-2-2b | 1.000 | 0.994 | 0.933 | -0.061 | no |
| qwen-2.5-3b | 0.972 | 0.983 | 0.967 | -0.017 | no |

