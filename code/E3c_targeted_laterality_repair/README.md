# E3c — Targeted Laterality Repair Pilot

**Status:** `STARTED / MULTI-MODEL CERTIFICATE-GATED PILOTS COMPLETE`

E3c tests the constructive symmetry-matched claim after E3b identified OpenI laterality as a
non-ceiling external boundary case. This is not zero-shot transfer: the OpenI laterality train
reports are explicitly included in the repair objective, while dev/test reports remain held
out.

## Question

Can a targeted perturbation-aware repair improve held-out OpenI left/right separability after
the laterality weak family has been identified?

## Current Answer

Yes, with model-dependent strength. GPT-2 and Qwen provide constructive proof-of-mechanism
evidence: certificate-gated targeted repair raised the identified laterality direction to a
predeclared dev-set geometric floor and transferred to held-out test geometry and several
downstream readouts. Qwen is the strongest case because probability L20, BA, pair direction,
logit L20, and geometric L20 all improve over Standard. Gemma passes the certificate and shows
small held-out geometry gains, but not downstream BA/logit/geometric gains, so it remains a
boundary case rather than a positive downstream repair.

## Inputs

- Base checkpoint: `gpt2_standard_joint16_owt`
- Frozen split: `E3b_external_negation/results/openi_laterality_minimal_pair_split.json`
- Frozen hidden cache: `E3b_external_negation/results/feature_cache_laterality/gpt2_cadc6ce3f8c1ec8d_0726cab7bed0ce0e.npz`
- OWT reconstruction cache: `FINAL/tmlr_revision/prepare/experiment_101_hybrid_owt/data/owt_cache_gpt2_l12_25k.pt`

## GPT-2 Smoke Runs

| Run | Steps | Objective | Test BA | Test pair acc | Prob L20 | Logit L20 | Geom L20 |
|---|---:|---|---:|---:|---:|---:|---:|
| Standard baseline | — | — | 0.8291 | 0.9873 | 0.1578 | 1.0398 | 0.9414 |
| Zero-shot V-reg baseline | — | previous E16 V-reg | 0.8734 | 0.9873 | 0.1600 | 1.0022 | 0.9395 |
| `gpt2_laterality_recon_only_seed42_200` | 200 | OWT recon continuation | 0.8291 | 1.0000 | 0.1779 | 1.1535 | 1.0463 |
| `gpt2_laterality_recon_only_seed42_500` | 500 | OWT recon continuation | 0.8291 | 0.9873 | 0.1713 | 1.0742 | 0.9744 |
| `gpt2_laterality_vgini_seed42_smoke200` | 200 | V-Gini | 0.8481 | 1.0000 | 0.1488 | 1.2303 | 1.1863 |
| `gpt2_laterality_vgini_seed42_500` | 500 | V-Gini | 0.8165 | 0.9873 | 0.1493 | 1.2354 | 1.1845 |
| `gpt2_laterality_vgini_mingain_seed42_smoke500` | 500 | V-Gini + relative min-gain | 0.7975 | 0.9747 | 0.1334 | 1.0234 | 0.9816 |
| `gpt2_laterality_certificate_rel002_seed42` | 525 / certificate | V-Gini + saturated relative min-gain + trust region | 0.8544 | 1.0000 | 0.1493 | 1.2507 | 1.2054 |

## Frozen Multi-Model Certificate Runs

All three models used the same frozen certificate protocol: relative `gamma = 0.02`,
required dev pass fraction `0.95`, saturated relative min-gain hinge, trust-region penalty,
and max 1000 steps.

| Model | Run | Stop step | Observed dev pass | Dev q05 | Dev L20 | Dev mean |
|---|---|---:|---:|---:|---:|---:|
| GPT-2 | `gpt2_laterality_certificate_rel002_seed42` | 525 | 0.950 | 0.0202 | 0.0238 | 0.0480 |
| Gemma | `gemma_laterality_certificate_rel002_seed42` | 50 | 1.000 | 0.0276 | 0.0299 | 0.0914 |
| Qwen | `qwen_laterality_certificate_rel002_seed42` | 150 | 0.963 | 0.0206 | 0.0243 | 0.0586 |

Downstream test readout:

| Model | Representation | BA | Pair acc | Prob L20 | Logit L20 | Geom L20 |
|---|---|---:|---:|---:|---:|---:|
| GPT-2 | Standard | 0.8291 | 0.9873 | 0.1578 | 1.0398 | 0.9414 |
| GPT-2 | Targeted certificate | 0.8544 | 1.0000 | 0.1493 | 1.2507 | 1.2054 |
| Gemma | Standard | 0.7658 | 0.9747 | 0.0823 | 0.6346 | 0.5530 |
| Gemma | Targeted certificate | 0.7468 | 0.9620 | 0.0851 | 0.5497 | 0.4759 |
| Qwen | Standard | 0.7215 | 0.9620 | 0.0285 | 0.1786 | 0.1600 |
| Qwen | Targeted certificate | 0.7532 | 1.0000 | 0.0641 | 0.7357 | 0.6398 |

Report-cluster probability-L20 deltas vs Standard:

| Model | Delta probability L20 | 95% CI | Delta BA | Delta F1 |
|---|---:|---:|---:|---:|
| GPT-2 | -0.0085 | [-0.0478, 0.0384] | +0.0253 | +0.0260 |
| Gemma | +0.0027 | [-0.0459, 0.0398] | -0.0190 | -0.0191 |
| Qwen | +0.0356 | [0.0028, 0.0738] | +0.0316 | +0.0319 |

## Direct Geometry/Fidelity Audits

CPU audits were used for stability; an earlier MPS loop was interrupted before writing output.

| Run | Test distance L20 | Test relative L20 | Distance mean | Relative mean | OpenI test NMSE | OWT NMSE |
|---|---:|---:|---:|---:|---:|---:|
| Standard baseline | 1.3564 | 0.0100 | 3.9937 | 0.0311 | 0.0000 | 0.0001 |
| Zero-shot V-reg baseline | 1.7530 | 0.0132 | 4.7272 | 0.0392 | 0.0001 | 0.0001 |
| `gpt2_laterality_recon_only_seed42_500` | 1.3567 | 0.0099 | 3.9978 | 0.0311 | 0.0000 | 0.0000 |
| `gpt2_laterality_vgini_seed42_500` | 3.7110 | 0.0281 | 6.1303 | 0.0484 | 0.0003 | 0.0000 |
| `gpt2_laterality_certificate_rel002_seed42` | 3.1248 | 0.0230 | 5.9229 | 0.0463 | 0.0001 | 0.0000 |

Multi-model certificate geometry:

| Model | Standard relative L20 | Target relative L20 | Delta | Standard distance L20 | Target distance L20 | Delta |
|---|---:|---:|---:|---:|---:|---:|
| GPT-2 | 0.0100 | 0.0230 | +0.0141 | 1.3564 | 3.1248 | +1.8990 |
| Gemma | 0.0272 | 0.0305 | +0.0033 | 5.0872 | 5.2345 | +0.1705 |
| Qwen | 0.0196 | 0.0277 | +0.0084 | 1.5906 | 1.8956 | +0.3086 |

Matched recon-only controls at the same certificate stop steps:

| Model | Run | Test relative L20 | Test distance L20 | BA | Prob L20 | Logit L20 | Geom L20 |
|---|---|---:|---:|---:|---:|---:|---:|
| Gemma | Standard | 0.0272 | 5.0872 | 0.7658 | 0.0823 | 0.6346 | 0.5530 |
| Gemma | Recon-only 50 | 0.0275 | 5.0676 | 0.7532 | 0.0840 | 0.6404 | 0.5580 |
| Gemma | Certificate 50 | 0.0305 | 5.2345 | 0.7468 | 0.0851 | 0.5497 | 0.4759 |
| Qwen | Standard | 0.0196 | 1.5906 | 0.7215 | 0.0285 | 0.1786 | 0.1600 |
| Qwen | Recon-only 150 | 0.0197 | 1.5877 | 0.7152 | 0.0361 | 0.3543 | 0.3189 |
| Qwen | Certificate 150 | 0.0277 | 1.8956 | 0.7532 | 0.0641 | 0.7357 | 0.6398 |

## Certificate-Gated Pilot

The certificate-gated run replaces fixed-step stopping with a dev floor condition:
relative pair distance `d_i >= 0.02` for at least 95% of dev pairs, with a saturated
relative min-gain hinge and a checkpoint trust-region penalty. The run requested up to 1000
steps and stopped at step 525 when the dev pass fraction reached 0.950.

Final dev certificate:

| Gamma | Relative | Required pass | Observed pass | Dev q05 | Dev L20 | Dev mean |
|---:|---|---:|---:|---:|---:|---:|
| 0.0200 | yes | 0.950 | 0.950 | 0.0202 | 0.0238 | 0.0480 |

## Interpretation

Matched continuation controls are now available. Recon-only continuation explains part of the
probability/logit/geometric L20 improvement, so those gains cannot be attributed to V-Gini
without the matched baseline. However, recon-only barely changes held-out direct code
distance, while V-Gini 500 substantially increases held-out left/right code-distance L20 and
relative-distance L20 with only a small OpenI reconstruction cost. This is the clearest
V-Gini-specific signal.

The V-Gini-only repair improves logit/geometric lower-tail margins, but not probability L20 or
BA at 500 steps. The 200-step V-Gini checkpoint has better BA/F1 and perfect pairwise
direction, suggesting that unconstrained V-Gini can keep reshaping the code space after the
useful laterality repair has largely occurred.

The certificate-gated pilot is a stronger constructive smoke. It stopped by a predeclared
mathematical floor condition rather than by picking the empirically best step count. Held-out
test geometry improved substantially over Standard (relative L20 0.0100 -> 0.0230), pairwise
direction reached 1.0000, and BA improved to 0.8544. Probability L20 remains below the
Standard baseline, so this is not yet a closed downstream-performance claim. It is evidence
that saturated min-gain plus trust-region can serve as a disciplined repair stopping rule.

The GPT-2 certificate-gated checkpoint is also stronger than fixed-step V-Gini 500 on downstream
readout: it preserves the logit/geometric lower-tail gains while recovering BA (0.8165 ->
0.8544) and pairwise direction (0.9873 -> 1.0000). It does not dominate the prior zero-shot
V-reg checkpoint on conventional BA/F1, so the claim should remain constructive and
mechanistic rather than "best overall downstream classifier."

The frozen multi-model extension sharpens the picture. Qwen becomes the cleanest positive
case: probability L20 has a positive report-cluster CI, BA/F1 improve, pairwise direction is
1.0000, and logit/geometric L20 rise sharply. GPT-2 remains a constructive geometry/readout
case with probability-L20 caveat. Gemma passes the dev certificate and gains a small amount of
held-out geometry, but downstream BA and logit/geometric L20 decline; this should be reported
as a model-dependent boundary, not a positive downstream repair.

The Gemma/Qwen recon-only controls reduce the main continuation confound. Qwen recon-only at
150 steps barely changes held-out geometry and does not improve BA, while certificate-gated
repair gives a much larger geometry lift and positive downstream deltas. Gemma recon-only
mostly improves reconstruction and leaves geometry nearly unchanged; the certificate run adds
a small geometry/probability-tail lift, but its downstream BA/logit/geometric readout remains
weaker than Standard. Thus the shared E3c claim is constructive reorientation of an identified
weak family, with model-dependent downstream consequences.

## Frozen Protocol

The multi-model runs used this frozen protocol:

- `gamma_rel = 0.02`
- `required_dev_pass = 0.95`
- saturated relative min-gain hinge
- trust-region penalty to the Standard SAE checkpoint
- stop by dev certificate, with max 1000 steps
- primary geometry: test relative L20 and distance L20
- secondary downstream: BA, pair accuracy, probability L20, logit L20, geometric L20
- fidelity: OWT and OpenI NMSE/EV/L0
