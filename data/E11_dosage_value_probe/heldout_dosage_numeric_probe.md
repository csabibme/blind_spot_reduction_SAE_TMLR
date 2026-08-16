# Held-out dosage numeric probe

Status: `heldout_dosage_numeric_probe_frozen`
Smoke: `False`
Pairs: 3825
Templates: 24

## Split summary

| Split | pairs | templates | critical | nuisance |
|---|---:|---:|---:|---:|
| train | 1830 | 12 | 610 | 1220 |
| dev | 915 | 6 | 305 | 610 |
| test | 1080 | 6 | 360 | 720 |

## Notes

- Positive pairs change only the critical numeric value (e.g. 3 vs 8 doses).
- Negative pairs are meaning-preserving paraphrase or digit/word formatting controls.
- Templates in test are disjoint from train/dev.
