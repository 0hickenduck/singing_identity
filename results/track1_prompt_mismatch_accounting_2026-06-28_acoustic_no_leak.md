# Track 1 Prompt-Mismatch Accounting

Run: `track1_prompt_mismatch_accounting_acoustic_no_leak`

Representation: `acoustic_baseline` / `local_wave_v1` / `frame25ms_hop20ms`

Dataset: 4000 same-singer same-text pairs, 20 speakers, 5 speaker-disjoint folds.

## Model Metrics

| model | covariates | delta cosine | MSE red. vs M0 | residual norm red. | R@1 after | control |
|---|---:|---:|---:|---:|---:|---|
| M0 | 0 | 0.6253 | 0.0000 | 0.1966 | 0.2080 | false |
| M1 | 429 | 0.5661 | -0.3108 | 0.0718 | 0.2042 | false |
| M2 | 433 | 0.5652 | -0.3164 | 0.0702 | 0.1968 | false |
| M3 | 439 | 0.5762 | -0.2140 | 0.1044 | 0.1938 | false |
| M4 | 439 | 0.5762 | -0.2140 | 0.1044 | 0.1938 | false |
| M5 | 445 | 0.5777 | -0.2198 | 0.1017 | 0.1943 | false |
| C_f0_duration_energy_only | 10 | 0.6136 | 0.0151 | 0.2042 | 0.2218 | true |
| C_metadata_only | 12 | 0.6069 | -0.1319 | 0.1372 | 0.1898 | true |
| C_m5_shuffle_technique | 445 | 0.5760 | -0.2121 | 0.1050 | 0.1953 | true |
| C_shuffle_singing_within_speaker | 445 | 0.4941 | -0.6408 | 0.0769 | 0.1980 | true |

## Factor Increments

| increment | MSE change | MSE red. vs previous | R@1 change | delta cosine change |
|---|---:|---:|---:|---:|
| added F0/prosody | -1091.304126 | -0.0043 | -0.0075 | -0.0009 |
| added timing/energy | 19819.338828 | 0.0778 | -0.0030 | 0.0109 |
| added acoustic/phonation proxy | 0.000000 | 0.0000 | 0.0000 | 0.0000 |
| added technique labels | -1132.183324 | -0.0048 | 0.0005 | 0.0015 |

## Seed-VC Prompt Gap Link

Matched Seed-VC prompt-gap pairs: 10

| predictor | n | Pearson | Spearman |
|---|---:|---:|---:|
| raw_delta_norm | 10 | 0.1767 | 0.0788 |
| m5_residual_norm | 10 | -0.1084 | -0.2970 |
| m5_predicted_delta_norm | 10 | -0.3437 | -0.2485 |
| technique_increment_norm_m5_minus_m4 | 10 | 0.1229 | -0.0424 |
| acoustic_increment_norm_m4_minus_m3 | 10 | nan | nan |

## Notes

- Values are out-of-fold over speaker-disjoint splits.
- Technique is a label-associated covariate, not a causal factor.
- Acoustic-baseline covariates are included as a shortcut/phonation proxy check.
