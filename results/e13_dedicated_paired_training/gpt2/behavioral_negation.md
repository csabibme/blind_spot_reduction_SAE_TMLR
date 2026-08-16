# Behavioural case study: SAE-in-the-loop decision flips (held-out templates)

Readout trained on true hidden states, then applied to reconstructions.

| Input to readout | test accuracy | test BA |
|---|---:|---:|
| true hidden (upper bound) | 0.8333 | 0.8333 |
| Standard SAE reconstruction | 0.8333 | 0.8333 |
| V-reg SAE reconstruction | 0.8333 | 0.8333 |

Test minimal pairs: 36
- collapsed (aff/neg decided the same) by Standard recon: **12**
- collapsed by V-reg recon: **12**
- V-reg saves (hidden correct, Standard wrong, V-reg correct): **0**

