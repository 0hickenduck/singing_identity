# Track 1 Prompt-Mismatch Accounting

Run: `track1_prompt_mismatch_accounting_mert_l12`

Representation: `mert_v1_95m` / `m_a_p_mert_v1_95m` / `12`

Dataset: 4000 same-singer same-text pairs, 20 speakers, 5 speaker-disjoint folds.

## Model Metrics

| model | covariates | delta cosine | MSE red. vs M0 | residual norm red. | R@1 after | control |
|---|---:|---:|---:|---:|---:|---|
| M0 | 0 | 0.5536 | 0.0000 | 0.1638 | 0.4258 | false |
| M1 | 429 | 0.4588 | -0.3726 | 0.0314 | 0.3730 | false |
| M2 | 433 | 0.4584 | -0.3739 | 0.0310 | 0.3745 | false |
| M3 | 439 | 0.4796 | -0.2704 | 0.0685 | 0.4310 | false |
| M4 | 457 | 0.4963 | -0.2213 | 0.0868 | 0.4490 | false |
| M5 | 463 | 0.4997 | -0.2143 | 0.0893 | 0.4432 | false |
| C_f0_duration_energy_only | 10 | 0.5780 | 0.0616 | 0.1949 | 0.4918 | true |
| C_metadata_only | 12 | 0.5227 | -0.0979 | 0.1265 | 0.4640 | true |
| C_m5_shuffle_technique | 463 | 0.4960 | -0.2223 | 0.0864 | 0.4447 | true |
| C_shuffle_singing_within_speaker | 463 | 0.4759 | -0.3214 | 0.0720 | 0.4068 | true |

## Factor Increments

| increment | MSE change | MSE red. vs previous | R@1 change | delta cosine change |
|---|---:|---:|---:|---:|
| added F0/prosody | -0.000011 | -0.0010 | 0.0015 | -0.0004 |
| added timing/energy | 0.000847 | 0.0753 | 0.0565 | 0.0212 |
| added acoustic/phonation proxy | 0.000402 | 0.0387 | 0.0180 | 0.0167 |
| added technique labels | 0.000058 | 0.0058 | -0.0058 | 0.0034 |

## Seed-VC Prompt Gap Link

Matched Seed-VC prompt-gap pairs: 10

| predictor | n | Pearson | Spearman |
|---|---:|---:|---:|
| raw_delta_norm | 10 | 0.2221 | -0.0061 |
| m5_residual_norm | 10 | 0.3300 | 0.2727 |
| m5_predicted_delta_norm | 10 | 0.2439 | 0.3697 |
| technique_increment_norm_m5_minus_m4 | 10 | -0.0998 | -0.0788 |
| acoustic_increment_norm_m4_minus_m3 | 10 | -0.3012 | -0.7333 |

## Notes

- Values are out-of-fold over speaker-disjoint splits.
- Technique is a label-associated covariate, not a causal factor.
- Acoustic-baseline covariates are included as a shortcut/phonation proxy check.
