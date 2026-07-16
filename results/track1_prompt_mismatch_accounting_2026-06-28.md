# Track 1 Prompt-Mismatch Accounting

Run: `track1_prompt_mismatch_accounting_wavlm_l6`

Representation: `wavlm_base_plus` / `microsoft_wavlm_base_plus` / `6`

Dataset: 4000 same-singer same-text pairs, 20 speakers, 5 speaker-disjoint folds.

## Model Metrics

| model | covariates | delta cosine | MSE red. vs M0 | residual norm red. | R@1 after | control |
|---|---:|---:|---:|---:|---:|---|
| M0 | 0 | 0.7681 | 0.0000 | 0.3421 | 0.5225 | false |
| M1 | 429 | 0.7238 | -0.3152 | 0.2543 | 0.3560 | false |
| M2 | 433 | 0.7236 | -0.3147 | 0.2545 | 0.3570 | false |
| M3 | 439 | 0.7317 | -0.1827 | 0.2915 | 0.4318 | false |
| M4 | 457 | 0.7417 | -0.1032 | 0.3153 | 0.4622 | false |
| M5 | 463 | 0.7439 | -0.0952 | 0.3177 | 0.4545 | false |
| C_f0_duration_energy_only | 10 | 0.7792 | 0.0780 | 0.3755 | 0.6038 | true |
| C_metadata_only | 12 | 0.7550 | -0.0774 | 0.3181 | 0.4953 | true |
| C_m5_shuffle_technique | 463 | 0.7415 | -0.1038 | 0.3150 | 0.4630 | true |
| C_shuffle_singing_within_speaker | 463 | 0.6924 | -0.4656 | 0.2477 | 0.4093 | true |

## Factor Increments

| increment | MSE change | MSE red. vs previous | R@1 change | delta cosine change |
|---|---:|---:|---:|---:|
| added F0/prosody | 0.000001 | 0.0004 | 0.0010 | -0.0002 |
| added timing/energy | 0.000381 | 0.1005 | 0.0748 | 0.0081 |
| added acoustic/phonation proxy | 0.000229 | 0.0672 | 0.0305 | 0.0100 |
| added technique labels | 0.000023 | 0.0072 | -0.0077 | 0.0022 |

## Seed-VC Prompt Gap Link

Matched Seed-VC prompt-gap pairs: 10

| predictor | n | Pearson | Spearman |
|---|---:|---:|---:|
| raw_delta_norm | 10 | 0.5672 | 0.5273 |
| m5_residual_norm | 10 | -0.2851 | -0.1394 |
| m5_predicted_delta_norm | 10 | 0.0264 | 0.3818 |
| technique_increment_norm_m5_minus_m4 | 10 | -0.3783 | -0.3818 |
| acoustic_increment_norm_m4_minus_m3 | 10 | -0.3761 | -0.7697 |

## Notes

- Values are out-of-fold over speaker-disjoint splits.
- Technique is a label-associated covariate, not a causal factor.
- Acoustic-baseline covariates are included as a shortcut/phonation proxy check.
