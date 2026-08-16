# Toy — fixed, rule-based selection

Fixed rule (applied uniformly to every point): hidden AUROC ≥ 0.7, Standard AUROC < 0.75, Δ AUROC > 0.0, Δ L20(‖Δz‖) > 0.0, MSE ratio ≤ 1.15.
Main-text point: among qualifying points, the one in the design target band (hidden 0.75–0.9, Standard 0.6–0.75); smaller MSE ratio breaks ties. Ideal bar: hidden ≥ 0.75 and Δ AUROC ≥ 0.03.

Qualifying points: **10 / 40**.

| rank | source | α | hidden | Standard | V-reg | Δ AUROC | Δ L20 | MSE ratio | qualifies | ideal | fails |
|---:|---|---:|---:|---:|---:|---:|---:|---:|:--:|:--:|---|
| 1 | d_sae96_lam0p2_noise0p15 | 1.00 | 0.7265 | 0.6578 | 0.6663 | +0.0085 | +0.1328 | 1.049 | yes | no | - |
| 2 | d_sae96_lam0p2_noise0p15 | 0.80 | 0.7066 | 0.6225 | 0.6366 | +0.0141 | +0.0437 | 1.061 | yes | no | - |
| 3 | d_sae96_lam0p2_noise0p1 | 0.80 | 0.7063 | 0.6330 | 0.6458 | +0.0128 | +0.0727 | 1.073 | yes | no | - |
| 4 | results_v2:d_sae192_lam0p5_noise0p12 | 1.00 | 0.7265 | 0.6609 | 0.6851 | +0.0242 | +0.2588 | 1.054 | yes | no | - |
| 5 | results_v2:d_sae192_lam0p5_noise0p12 | 0.80 | 0.7051 | 0.6429 | 0.6665 | +0.0237 | +0.1396 | 1.068 | yes | no | - |
| 6 | results_v3:d_sae192_lam0p5_noise0p1 | 1.60 | 0.7873 | 0.7252 | 0.7417 | +0.0164 | +0.5076 | 1.085 | yes | no | - |
| 7 | d_sae96_lam0p5_noise0p15 | 1.00 | 0.7265 | 0.6578 | 0.6727 | +0.0149 | +0.0765 | 1.119 | yes | no | - |
| 8 | results_v3:d_sae192_lam0p5_noise0p1 | 1.20 | 0.7488 | 0.6859 | 0.7099 | +0.0241 | +0.3314 | 1.121 | yes | no | - |
| 9 | d_sae96_lam0p5_noise0p1 | 1.00 | 0.7264 | 0.6646 | 0.6854 | +0.0208 | +0.1358 | 1.123 | yes | no | - |
| 10 | d_sae96_lam0p5_noise0p1 | 0.80 | 0.7063 | 0.6330 | 0.6614 | +0.0284 | +0.0641 | 1.140 | yes | no | - |
| 11 | d_sae96_lam0p2_noise0p1 | 1.00 | 0.7264 | 0.6646 | 0.6630 | -0.0016 | +0.1393 | 1.061 | no | no | delta_auroc_positive |
| 12 | d_sae96_lam0p2_noise0p15 | 0.60 | 0.6809 | 0.6008 | 0.6120 | +0.0112 | -0.0337 | 1.067 | no | no | hidden_present,delta_l20_positive |
| 13 | d_sae96_lam0p2_noise0p15 | 0.40 | 0.6549 | 0.5722 | 0.5875 | +0.0153 | -0.0824 | 1.073 | no | no | hidden_present,delta_l20_positive |
| 14 | d_sae96_lam0p2_noise0p15 | 0.20 | 0.6012 | 0.5447 | 0.5448 | +0.0000 | -0.0957 | 1.073 | no | no | hidden_present,delta_l20_positive |
| 15 | d_sae96_lam0p2_noise0p1 | 0.60 | 0.6834 | 0.6029 | 0.6117 | +0.0088 | +0.0220 | 1.087 | no | no | hidden_present |
| 16 | d_sae96_lam0p2_noise0p1 | 0.40 | 0.6553 | 0.5752 | 0.5850 | +0.0098 | -0.0077 | 1.101 | no | no | hidden_present,delta_l20_positive |
| 17 | d_sae96_lam0p2_noise0p1 | 0.20 | 0.6072 | 0.5399 | 0.5515 | +0.0116 | -0.0275 | 1.108 | no | no | hidden_present,delta_l20_positive |
| 18 | results_v3:d_sae192_lam0p5_noise0p1 | 2.50 | 0.8807 | 0.8183 | 0.8382 | +0.0199 | +1.0259 | 1.040 | no | no | standard_subceiling |
| 19 | results_v3:d_sae192_lam0p5_noise0p1 | 2.00 | 0.8222 | 0.7701 | 0.7786 | +0.0085 | +0.7538 | 1.060 | no | no | standard_subceiling |
| 20 | results_v2:d_sae192_lam0p5_noise0p12 | 0.60 | 0.6820 | 0.6059 | 0.6246 | +0.0188 | +0.0498 | 1.085 | no | no | hidden_present |
| 21 | results_v2:d_sae192_lam0p5_noise0p12 | 0.40 | 0.6560 | 0.5780 | 0.5969 | +0.0189 | -0.0167 | 1.115 | no | no | hidden_present,delta_l20_positive |
| 22 | d_sae96_lam0p5_noise0p15 | 0.80 | 0.7066 | 0.6225 | 0.6484 | +0.0259 | -0.0135 | 1.135 | no | no | delta_l20_positive |
| 23 | results_v2:d_sae192_lam0p5_noise0p12 | 0.20 | 0.6087 | 0.5536 | 0.5568 | +0.0032 | -0.0594 | 1.141 | no | no | hidden_present,delta_l20_positive |
| 24 | d_sae96_lam0p5_noise0p15 | 0.60 | 0.6809 | 0.6008 | 0.6197 | +0.0188 | -0.0960 | 1.149 | no | no | hidden_present,delta_l20_positive |
| 25 | d_sae96_lam0p5_noise0p1 | 0.60 | 0.6834 | 0.6029 | 0.6172 | +0.0144 | +0.0251 | 1.160 | no | no | hidden_present,mse_ratio_ok |
| 26 | d_sae96_lam0p5_noise0p15 | 0.40 | 0.6549 | 0.5722 | 0.5878 | +0.0156 | -0.1387 | 1.161 | no | no | hidden_present,delta_l20_positive,mse_ratio_ok |
| 27 | d_sae96_lam0p5_noise0p15 | 0.20 | 0.6012 | 0.5447 | 0.5461 | +0.0014 | -0.1567 | 1.164 | no | no | hidden_present,delta_l20_positive,mse_ratio_ok |
| 28 | d_sae96_lam0p5_noise0p1 | 0.40 | 0.6553 | 0.5752 | 0.5895 | +0.0143 | -0.0196 | 1.193 | no | no | hidden_present,delta_l20_positive,mse_ratio_ok |
| 29 | results_v3:d_sae192_lam0p5_noise0p1 | 0.80 | 0.7060 | 0.6444 | 0.6574 | +0.0130 | +0.1530 | 1.198 | no | no | mse_ratio_ok |
| 30 | d_sae96_lam0p5_noise0p1 | 0.20 | 0.6072 | 0.5399 | 0.5454 | +0.0055 | -0.0488 | 1.214 | no | no | hidden_present,delta_l20_positive,mse_ratio_ok |

## Recommended main-text point

The ranked table above is ordered by the screening tie-break; the main-text point below is the qualifying point selected by the design target band (this can differ from table rank 1).

- **results_v3:d_sae192_lam0p5_noise0p1**, α=1.60 (qualifying, in the design target band)
  - hidden AUROC = 0.7873
  - Standard AUROC = 0.7252
  - V-reg AUROC = 0.7417
  - Δ AUROC = +0.0164
  - Δ L20(‖Δz‖) = +0.5076
  - MSE ratio = 1.085
  - suggested framing: _modest but consistent_
