# Track 1 Accounting v2: Low-Dim Input, Low-Rank Target

Run: `track1_prompt_mismatch_accounting_v2_wavlm_l6_smoke`
Representation: `wavlm_base_plus` / `microsoft_wavlm_base_plus` / `6`
Pairs: 100; speakers: 20

## Target PCA K=8

| model | covariates | delta cosine | MSE red. vs M0 | R@1 before | R@1 after |
|---|---:|---:|---:|---:|---:|
| M0_mean_delta | 0 | 0.7599 | 0.0000 | 0.1500 | 0.3700 |
| L1_nuisance | 10 | 0.7500 | -0.1103 | 0.1500 | 0.4200 |
| L2_nuisance_metadata | 22 | 0.7508 | -0.0999 | 0.1500 | 0.3700 |
| L3_low_content | 28 | 0.7502 | -0.1058 | 0.1500 | 0.3700 |
| L4_technique | 30 | 0.7506 | -0.0928 | 0.1500 | 0.3500 |
| L5_acoustic_proxy | 48 | 0.7598 | -0.0035 | 0.1500 | 0.4200 |

## Target PCA K=16

| model | covariates | delta cosine | MSE red. vs M0 | R@1 before | R@1 after |
|---|---:|---:|---:|---:|---:|
| M0_mean_delta | 0 | 0.7599 | 0.0000 | 0.1500 | 0.3700 |
| L1_nuisance | 10 | 0.7490 | -0.1332 | 0.1500 | 0.4100 |
| L2_nuisance_metadata | 22 | 0.7502 | -0.1157 | 0.1500 | 0.3800 |
| L3_low_content | 28 | 0.7501 | -0.1198 | 0.1500 | 0.3900 |
| L4_technique | 30 | 0.7502 | -0.1044 | 0.1500 | 0.3700 |
| L5_acoustic_proxy | 48 | 0.7603 | -0.0009 | 0.1500 | 0.4400 |
