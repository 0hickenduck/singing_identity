# Track 1 Prompt-Mismatch Accounting

Run: `track1_prompt_mismatch_accounting_wavlm_l9`

Representation: `wavlm_base_plus` / `microsoft_wavlm_base_plus` / `9`

Dataset: 4000 same-singer same-text pairs, 20 speakers, 5 speaker-disjoint folds.

## Model Metrics

| model | covariates | delta cosine | MSE red. vs M0 | residual norm red. | R@1 after | control |
|---|---:|---:|---:|---:|---:|---|
| M0 | 0 | 0.7677 | 0.0000 | 0.3385 | 0.5443 | false |
| M1 | 429 | 0.7328 | -0.2628 | 0.2642 | 0.4135 | false |
| M2 | 433 | 0.7325 | -0.2630 | 0.2642 | 0.4155 | false |
| M3 | 439 | 0.7400 | -0.1378 | 0.3002 | 0.5000 | false |
| M4 | 457 | 0.7481 | -0.0662 | 0.3217 | 0.5255 | false |
| M5 | 463 | 0.7492 | -0.0620 | 0.3229 | 0.5172 | false |
| C_f0_duration_energy_only | 10 | 0.7780 | 0.0943 | 0.3761 | 0.6342 | true |
| C_metadata_only | 12 | 0.7611 | -0.0488 | 0.3242 | 0.5535 | true |
| C_m5_shuffle_technique | 463 | 0.7479 | -0.0669 | 0.3215 | 0.5232 | true |
| C_shuffle_singing_within_speaker | 463 | 0.6935 | -0.4489 | 0.2515 | 0.4605 | true |

## Factor Increments

| increment | MSE change | MSE red. vs previous | R@1 change | delta cosine change |
|---|---:|---:|---:|---:|
| added F0/prosody | -0.000001 | -0.0001 | 0.0020 | -0.0002 |
| added timing/energy | 0.000386 | 0.0991 | 0.0845 | 0.0075 |
| added acoustic/phonation proxy | 0.000220 | 0.0629 | 0.0255 | 0.0081 |
| added technique labels | 0.000013 | 0.0039 | -0.0082 | 0.0011 |

## Seed-VC Prompt Gap Link

Matched Seed-VC prompt-gap pairs: 10

| predictor | n | Pearson | Spearman |
|---|---:|---:|---:|
| raw_delta_norm | 10 | 0.5200 | 0.5030 |
| m5_residual_norm | 10 | -0.3821 | -0.1394 |
| m5_predicted_delta_norm | 10 | -0.0647 | 0.0424 |
| technique_increment_norm_m5_minus_m4 | 10 | -0.5065 | -0.6242 |
| acoustic_increment_norm_m4_minus_m3 | 10 | -0.3696 | -0.7818 |

## Notes

- Values are out-of-fold over speaker-disjoint splits.
- Technique is a label-associated covariate, not a causal factor.
- Acoustic-baseline covariates are included as a shortcut/phonation proxy check.
