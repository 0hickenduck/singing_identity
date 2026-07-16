# Track 1 Prompt-Mismatch Accounting

Run: `track1_prompt_mismatch_accounting_wavlm_l6_smoke100`

Representation: `wavlm_base_plus` / `microsoft_wavlm_base_plus` / `6`

Dataset: 100 same-singer same-text pairs, 20 speakers, 5 speaker-disjoint folds.

## Model Metrics

| model | covariates | delta cosine | MSE red. vs M0 | residual norm red. | R@1 after | control |
|---|---:|---:|---:|---:|---:|---|
| M0 | 0 | 0.7599 | 0.0000 | 0.3320 | 0.3700 | false |
| M1 | 211 | 0.7446 | -0.0975 | 0.2968 | 0.2900 | false |
| M2 | 215 | 0.7440 | -0.1032 | 0.2951 | 0.2900 | false |
| M3 | 221 | 0.7469 | -0.0786 | 0.3037 | 0.3400 | false |
| M4 | 239 | 0.7504 | -0.0467 | 0.3144 | 0.4100 | false |
| M5 | 241 | 0.7502 | -0.0475 | 0.3142 | 0.4100 | false |
| C_f0_duration_energy_only | 10 | 0.7496 | -0.1560 | 0.3127 | 0.4500 | true |
| C_metadata_only | 12 | 0.7584 | -0.0029 | 0.3312 | 0.3300 | true |
| C_m5_shuffle_technique | 241 | 0.7504 | -0.0457 | 0.3148 | 0.4000 | true |
| C_shuffle_singing_within_speaker | 241 | 0.7236 | -0.2088 | 0.2867 | 0.3500 | true |

## Factor Increments

| increment | MSE change | MSE red. vs previous | R@1 change | delta cosine change |
|---|---:|---:|---:|---:|
| added F0/prosody | -0.000016 | -0.0052 | 0.0000 | -0.0006 |
| added timing/energy | 0.000070 | 0.0223 | 0.0500 | 0.0028 |
| added acoustic/phonation proxy | 0.000090 | 0.0296 | 0.0700 | 0.0035 |
| added technique labels | -0.000002 | -0.0008 | 0.0000 | -0.0002 |

## Seed-VC Prompt Gap Link

Matched Seed-VC prompt-gap pairs: 10

| predictor | n | Pearson | Spearman |
|---|---:|---:|---:|
| raw_delta_norm | 10 | 0.5672 | 0.5273 |
| m5_residual_norm | 10 | -0.2183 | -0.2242 |
| m5_predicted_delta_norm | 10 | -0.2877 | -0.0424 |
| technique_increment_norm_m5_minus_m4 | 10 | -0.1869 | -0.4909 |
| acoustic_increment_norm_m4_minus_m3 | 10 | 0.0184 | -0.1030 |

## Notes

- Values are out-of-fold over speaker-disjoint splits.
- Technique is a label-associated covariate, not a causal factor.
- Acoustic-baseline covariates are included as a shortcut/phonation proxy check.
