# Exact finite-family worst-subset lower-tail audit

Status: **PASS**  
Numerical tolerance: `1.0e-12`

For the same equally weighted finite evaluation family and `m = ceil(0.20*n)`, the Standard-defined fixed-set relative delta must be at least the independently selected own-tail delta.

| Model | Family | n | m | own-tail delta | fixed Wstd delta [95% CI] | slack | relative improved | absolute delta [95% CI] | absolute improved |
|---|---|---:|---:|---:|---|---:|---:|---|---:|
| GPT-2 | negation | 300 | 60 | +0.004215 | +0.004736 [+0.003720, +0.005731] | +5.215e-04 | 0.917 | +0.6337 [+0.5151, +0.7537] | 0.933 |
| GPT-2 | laterality | 300 | 60 | +0.000945 | +0.000995 [+0.000886, +0.001104] | +4.933e-05 | 0.983 | +0.1202 [+0.1053, +0.1348] | 0.983 |
| GPT-2 | severity | 300 | 60 | +0.003231 | +0.004157 [+0.003558, +0.004791] | +9.265e-04 | 1.000 | +0.5350 [+0.4577, +0.6183] | 1.000 |
| GPT-2 | anatomical_direction | 131 | 27 | +0.000807 | +0.000949 [+0.000771, +0.001149] | +1.419e-04 | 1.000 | +0.1161 [+0.0954, +0.1388] | 1.000 |
| Qwen 2.5 | negation | 80 | 16 | +0.009887 | +0.009887 [+0.008415, +0.011205] | +0.000e+00 | 1.000 | -0.1080 [-0.1963, -0.0271] | 0.250 |
| Qwen 2.5 | laterality | 80 | 16 | +0.000888 | +0.000960 [+0.000769, +0.001181] | +7.178e-05 | 1.000 | -0.0066 [-0.0132, -0.0008] | 0.375 |
| Qwen 2.5 | severity | 80 | 16 | +0.003506 | +0.003601 [+0.002771, +0.004417] | +9.571e-05 | 1.000 | -0.0029 [-0.0295, +0.0266] | 0.375 |
| Qwen 2.5 | anatomical_direction | 80 | 16 | +0.000556 | +0.000562 [+0.000408, +0.000705] | +6.555e-06 | 0.875 | -0.0015 [-0.0051, +0.0021] | 0.375 |

The relative fixed-set point estimate verifies a deterministic consequence of the own-tail endpoint; it is not independent evidence. The absolute endpoint, individual-pair fractions, uncertainty intervals, and downstream probe remain separate empirical quantities.
