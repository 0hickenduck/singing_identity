# Track 1 Prompt-Mismatch Accounting

Run: `track1_prompt_mismatch_accounting_acoustic`

Representation: `acoustic_baseline` / `local_wave_v1` / `frame25ms_hop20ms`

Dataset: 4000 same-singer same-text pairs, 20 speakers, 5 speaker-disjoint folds.

## Model Metrics

| model | covariates | delta cosine | MSE red. vs M0 | residual norm red. | R@1 after | control |
|---|---:|---:|---:|---:|---:|---|
| M0 | 0 | 0.6253 | 0.0000 | 0.1966 | 0.2080 | false |
| M1 | 429 | 0.5661 | -0.3108 | 0.0718 | 0.2042 | false |
| M2 | 433 | 0.5652 | -0.3164 | 0.0702 | 0.1968 | false |
| M3 | 439 | 0.5762 | -0.2140 | 0.1044 | 0.1938 | false |
| M4 | 457 | 1.0000 | 1.0000 | 0.9999 | 1.0000 | false |
| M5 | 463 | 1.0000 | 1.0000 | 0.9999 | 1.0000 | false |
| C_f0_duration_energy_only | 10 | 0.6136 | 0.0151 | 0.2042 | 0.2218 | true |
| C_metadata_only | 12 | 0.6069 | -0.1319 | 0.1372 | 0.1898 | true |
| C_m5_shuffle_technique | 463 | 1.0000 | 1.0000 | 0.9999 | 1.0000 | true |
| C_shuffle_singing_within_speaker | 463 | 0.5797 | -0.1420 | 0.2290 | 0.2367 | true |

## Factor Increments

| increment | MSE change | MSE red. vs previous | R@1 change | delta cosine change |
|---|---:|---:|---:|---:|
| added F0/prosody | -1091.304126 | -0.0043 | -0.0075 | -0.0009 |
| added timing/energy | 19819.338828 | 0.0778 | -0.0030 | 0.0109 |
| added acoustic/phonation proxy | 234794.533288 | 1.0000 | 0.8063 | 0.4238 |
| added technique labels | -0.000030 | -0.0130 | 0.0000 | -0.0000 |

## Seed-VC Prompt Gap Link

Matched Seed-VC prompt-gap pairs: 10

| predictor | n | Pearson | Spearman |
|---|---:|---:|---:|
| raw_delta_norm | 10 | 0.1767 | 0.0788 |
| m5_residual_norm | 10 | -0.4067 | -0.4182 |
| m5_predicted_delta_norm | 10 | 0.1767 | 0.0788 |
| technique_increment_norm_m5_minus_m4 | 10 | -0.3611 | -0.6242 |
| acoustic_increment_norm_m4_minus_m3 | 10 | -0.1518 | -0.3212 |

## Notes

- Values are out-of-fold over speaker-disjoint splits.
- Technique is a label-associated covariate, not a causal factor.
- Acoustic-baseline covariates are included as a shortcut/phonation proxy check.
