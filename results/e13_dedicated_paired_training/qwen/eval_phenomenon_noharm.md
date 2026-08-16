# Phenomenon (V-Gini collapse) + OWT no-harm

## V-Gini over affirmed->negated pairs (lower = flatter response profile)

| Split | Standard V | V-reg V |
|---|---:|---:|
| train | 0.1134 | 0.0008 |
| dev | 0.081 | 0.0618 |
| test (HELD-OUT) | 0.1306 | 0.1145 |
| all | 0.1166 | 0.0663 |

## OWT reconstruction no-harm (general text, 5000 tokens)

| Metric | Standard | V-reg |
|---|---:|---:|
| mse | 568.3879 | 470.7276 |
| nmse | 0.004887 | 0.004047 |
| ev | 0.995 | 0.9958 |
| cosine | 0.9977 | 0.9978 |
