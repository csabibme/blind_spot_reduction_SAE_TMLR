# Behavioural case study: SAE-in-the-loop decision flips (held-out templates)

Readout trained on true hidden states, then applied to reconstructions.

| Input to readout | test accuracy | test BA |
|---|---:|---:|
| true hidden (upper bound) | 1.0 | 1.0 |
| Standard SAE reconstruction | 1.0 | 1.0 |
| V-reg SAE reconstruction | 1.0 | 1.0 |

Test minimal pairs: 36
- collapsed (aff/neg decided the same) by Standard recon: **0**
- collapsed by V-reg recon: **0**
- V-reg saves (hidden correct, Standard wrong, V-reg correct): **0**

