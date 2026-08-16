# Held-out dosage numeric probe analysis

Dataset: `data/heldout_dosage_numeric_probe.json`

## Test metrics by profile

| Profile | Representation | Distance AUROC | Probe AUROC | Balanced acc. | Probe L20 margin | Critical L20 `||dz||` | Critical mean `||dz||` | Hard-dir AUROC |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| gpt2 | hidden | 0.1167 | 1.0000 | 1.0000 | 0.9768 | 2.6034 | 5.2956 | 1.0000 |
| gpt2 | sae_standard_code | 0.0841 | 1.0000 | 1.0000 | 0.9805 | 2.0799 | 3.4615 | 1.0000 |
| gpt2 | sae_vreg_code | 0.0513 | 1.0000 | 1.0000 | 0.9861 | 2.5971 | 4.0574 | 1.0000 |
| gemma-2-2b | hidden | 0.1807 | 0.9973 | 0.9542 | 0.6254 | 7.9601 | 21.8666 | 1.0000 |
| gemma-2-2b | sae_standard_code | 0.1762 | 0.9862 | 0.8979 | 0.3622 | 7.9851 | 21.5822 | 1.0000 |
| gemma-2-2b | sae_vreg_code | 0.1721 | 0.9867 | 0.8917 | 0.3516 | 7.8882 | 20.2745 | 1.0000 |
| qwen-2.5-3b | hidden | 0.1841 | 0.9995 | 0.9576 | 0.6605 | 2.6985 | 8.2429 | 1.0000 |
| qwen-2.5-3b | sae_standard_code | 0.1773 | 0.9845 | 0.9653 | 0.7081 | 2.8626 | 8.5778 | 1.0000 |
| qwen-2.5-3b | sae_vreg_code | 0.2202 | 0.9757 | 0.9382 | 0.5495 | 2.4409 | 7.5018 | 1.0000 |

## V-reg minus Standard (test)

### gpt2

- Δ probe AUROC: +0.0000
- Δ balanced accuracy: +0.0000
- Δ distance AUROC: -0.0328
- Δ critical L20 `||dz||`: +0.5172
- Δ critical mean `||dz||`: +0.5959
- Δ critical L20 relative response `D`: +0.0059
- Δ hard-direction AUROC: +0.0000
- 95% CI Δ probe AUROC (template cluster): [-0.0000, +0.0000]
- 95% CI Δ critical L20 `||dz||` (template cluster): [+0.2722, +0.8826]

### gemma-2-2b

- Δ probe AUROC: +0.0005
- Δ balanced accuracy: -0.0063
- Δ distance AUROC: -0.0041
- Δ critical L20 `||dz||`: -0.0969
- Δ critical mean `||dz||`: -1.3077
- Δ critical L20 relative response `D`: +0.0086
- Δ hard-direction AUROC: +0.0000
- 95% CI Δ probe AUROC (template cluster): [-0.0294, +0.0160]
- 95% CI Δ critical L20 `||dz||` (template cluster): [-1.2110, +0.0125]

### qwen-2.5-3b

- Δ probe AUROC: -0.0089
- Δ balanced accuracy: -0.0271
- Δ distance AUROC: +0.0430
- Δ critical L20 `||dz||`: -0.4218
- Δ critical mean `||dz||`: -1.0760
- Δ critical L20 relative response `D`: +0.0078
- Δ hard-direction AUROC: +0.0000
- 95% CI Δ probe AUROC (template cluster): [-0.0541, +0.0282]
- 95% CI Δ critical L20 `||dz||` (template cluster): [-0.8996, -0.3647]

