# Track 1 Prompt-Mismatch Accounting

Run: `track1_prompt_mismatch_accounting_mert_l3`

Representation: `mert_v1_95m` / `m_a_p_mert_v1_95m` / `3`

Dataset: 4000 same-singer same-text pairs, 20 speakers, 5 speaker-disjoint folds.

## Model Metrics

| model | covariates | delta cosine | MSE red. vs M0 | residual norm red. | R@1 after | control |
|---|---:|---:|---:|---:|---:|---|
| M0 | 0 | 0.5628 | 0.0000 | 0.1681 | 0.5285 | false |
| M1 | 429 | 0.4727 | -0.3620 | 0.0381 | 0.3985 | false |
| M2 | 433 | 0.4725 | -0.3626 | 0.0380 | 0.3980 | false |
| M3 | 439 | 0.4871 | -0.2589 | 0.0741 | 0.4555 | false |
| M4 | 457 | 0.5083 | -0.1883 | 0.1000 | 0.4630 | false |
| M5 | 463 | 0.5126 | -0.1785 | 0.1034 | 0.4587 | false |
| C_f0_duration_energy_only | 10 | 0.5731 | 0.0541 | 0.1937 | 0.5665 | true |
| C_metadata_only | 12 | 0.5394 | -0.0949 | 0.1319 | 0.5360 | true |
| C_m5_shuffle_technique | 463 | 0.5080 | -0.1891 | 0.0996 | 0.4652 | true |
| C_shuffle_singing_within_speaker | 463 | 0.4757 | -0.3471 | 0.0701 | 0.4265 | true |

## Factor Increments

| increment | MSE change | MSE red. vs previous | R@1 change | delta cosine change |
|---|---:|---:|---:|---:|
| added F0/prosody | -0.000002 | -0.0004 | -0.0005 | -0.0001 |
| added timing/energy | 0.000426 | 0.0761 | 0.0575 | 0.0145 |
| added acoustic/phonation proxy | 0.000290 | 0.0561 | 0.0075 | 0.0213 |
| added technique labels | 0.000040 | 0.0082 | -0.0043 | 0.0043 |

## Seed-VC Prompt Gap Link

Matched Seed-VC prompt-gap pairs: 10

| predictor | n | Pearson | Spearman |
|---|---:|---:|---:|
| raw_delta_norm | 10 | 0.7620 | 0.4182 |
| m5_residual_norm | 10 | 0.6179 | 0.2606 |
| m5_predicted_delta_norm | 10 | 0.3370 | 0.4061 |
| technique_increment_norm_m5_minus_m4 | 10 | -0.1668 | -0.1636 |
| acoustic_increment_norm_m4_minus_m3 | 10 | -0.3147 | -0.7818 |

## Notes

- Values are out-of-fold over speaker-disjoint splits.
- Technique is a label-associated covariate, not a causal factor.
- Acoustic-baseline covariates are included as a shortcut/phonation proxy check.
