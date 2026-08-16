# Phenomenon (V-Gini collapse) + OWT no-harm

## V-Gini over affirmed->negated pairs (lower = flatter response profile)

| Split | Standard V | V-reg V |
|---|---:|---:|
| train | 0.2326 | 0.0009 |
| dev | 0.23 | 0.1518 |
| test (HELD-OUT) | 0.2726 | 0.2665 |
| all | 0.2578 | 0.1297 |

## OWT reconstruction no-harm (general text, 5000 tokens)

| Metric | Standard | V-reg |
|---|---:|---:|
| mse | 3.3093 | 3.183 |
| nmse | 5.4e-05 | 5.2e-05 |
| ev | 0.9997 | 0.9997 |
| cosine | 0.9999 | 0.9999 |
