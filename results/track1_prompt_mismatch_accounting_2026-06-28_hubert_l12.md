# Track 1 Prompt-Mismatch Accounting

Run: `track1_prompt_mismatch_accounting_hubert_l12`

Representation: `hubert_base` / `facebook_hubert_base_ls960` / `12`

Dataset: 4000 same-singer same-text pairs, 20 speakers, 5 speaker-disjoint folds.

## Model Metrics

| model | covariates | delta cosine | MSE red. vs M0 | residual norm red. | R@1 after | control |
|---|---:|---:|---:|---:|---:|---|
| M0 | 0 | 0.6639 | 0.0000 | 0.2423 | 0.6302 | false |
| M1 | 429 | 0.6043 | -0.2880 | 0.1483 | 0.5072 | false |
| M2 | 433 | 0.6039 | -0.2889 | 0.1481 | 0.5088 | false |
| M3 | 439 | 0.6122 | -0.2267 | 0.1693 | 0.5470 | false |
| M4 | 457 | 0.6235 | -0.1681 | 0.1885 | 0.5655 | false |
| M5 | 463 | 0.6258 | -0.1620 | 0.1905 | 0.5657 | false |
| C_f0_duration_energy_only | 10 | 0.6733 | 0.0352 | 0.2611 | 0.6960 | true |
| C_metadata_only | 12 | 0.6466 | -0.0521 | 0.2231 | 0.6168 | true |
| C_m5_shuffle_technique | 463 | 0.6232 | -0.1690 | 0.1882 | 0.5670 | true |
| C_shuffle_singing_within_speaker | 463 | 0.5724 | -0.4650 | 0.1489 | 0.5430 | true |

## Factor Increments

| increment | MSE change | MSE red. vs previous | R@1 change | delta cosine change |
|---|---:|---:|---:|---:|
| added F0/prosody | -0.000005 | -0.0007 | 0.0015 | -0.0003 |
| added timing/energy | 0.000365 | 0.0483 | 0.0383 | 0.0082 |
| added acoustic/phonation proxy | 0.000344 | 0.0478 | 0.0185 | 0.0113 |
| added technique labels | 0.000036 | 0.0053 | 0.0002 | 0.0024 |

## Seed-VC Prompt Gap Link

Matched Seed-VC prompt-gap pairs: 10

| predictor | n | Pearson | Spearman |
|---|---:|---:|---:|
| raw_delta_norm | 10 | 0.5860 | 0.5879 |
| m5_residual_norm | 10 | -0.0922 | -0.1030 |
| m5_predicted_delta_norm | 10 | 0.3555 | 0.4424 |
| technique_increment_norm_m5_minus_m4 | 10 | -0.4176 | -0.3818 |
| acoustic_increment_norm_m4_minus_m3 | 10 | -0.2719 | -0.4909 |

## Notes

- Values are out-of-fold over speaker-disjoint splits.
- Technique is a label-associated covariate, not a causal factor.
- Acoustic-baseline covariates are included as a shortcut/phonation proxy check.
