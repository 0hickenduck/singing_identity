# Track 1 Accounting v2: Low-Dim Input, Low-Rank Target

Run: `track1_prompt_mismatch_accounting_v2_wavlm_l6`
Representation: `wavlm_base_plus` / `microsoft_wavlm_base_plus` / `6`
Pairs: 4000; speakers: 20

## Target PCA K=8

| model | covariates | delta cosine | MSE red. vs M0 | R@1 before | R@1 after |
|---|---:|---:|---:|---:|---:|
| M0_mean_delta | 0 | 0.7681 | 0.0000 | 0.2245 | 0.5225 |
| L1_nuisance | 10 | 0.7775 | 0.0713 | 0.2245 | 0.5860 |
| L2_nuisance_metadata | 22 | 0.7657 | 0.0315 | 0.2245 | 0.5595 |
| L3_low_content | 28 | 0.7692 | 0.0482 | 0.2245 | 0.5665 |
| L4_technique | 34 | 0.7716 | 0.0582 | 0.2245 | 0.5745 |
| L5_acoustic_proxy | 52 | 0.7651 | 0.0628 | 0.2245 | 0.5487 |

## Target PCA K=16

| model | covariates | delta cosine | MSE red. vs M0 | R@1 before | R@1 after |
|---|---:|---:|---:|---:|---:|
| M0_mean_delta | 0 | 0.7681 | 0.0000 | 0.2245 | 0.5225 |
| L1_nuisance | 10 | 0.7775 | 0.0719 | 0.2245 | 0.5830 |
| L2_nuisance_metadata | 22 | 0.7671 | 0.0390 | 0.2245 | 0.5695 |
| L3_low_content | 28 | 0.7692 | 0.0489 | 0.2245 | 0.5623 |
| L4_technique | 34 | 0.7718 | 0.0572 | 0.2245 | 0.5577 |
| L5_acoustic_proxy | 52 | 0.7642 | 0.0564 | 0.2245 | 0.5497 |

## Target PCA K=32

| model | covariates | delta cosine | MSE red. vs M0 | R@1 before | R@1 after |
|---|---:|---:|---:|---:|---:|
| M0_mean_delta | 0 | 0.7681 | 0.0000 | 0.2245 | 0.5225 |
| L1_nuisance | 10 | 0.7776 | 0.0720 | 0.2245 | 0.5853 |
| L2_nuisance_metadata | 22 | 0.7674 | 0.0404 | 0.2245 | 0.5695 |
| L3_low_content | 28 | 0.7699 | 0.0511 | 0.2245 | 0.5663 |
| L4_technique | 34 | 0.7726 | 0.0596 | 0.2245 | 0.5580 |
| L5_acoustic_proxy | 52 | 0.7644 | 0.0539 | 0.2245 | 0.5493 |

## Target PCA K=64

| model | covariates | delta cosine | MSE red. vs M0 | R@1 before | R@1 after |
|---|---:|---:|---:|---:|---:|
| M0_mean_delta | 0 | 0.7681 | 0.0000 | 0.2245 | 0.5225 |
| L1_nuisance | 10 | 0.7777 | 0.0725 | 0.2245 | 0.5863 |
| L2_nuisance_metadata | 22 | 0.7675 | 0.0405 | 0.2245 | 0.5677 |
| L3_low_content | 28 | 0.7700 | 0.0515 | 0.2245 | 0.5645 |
| L4_technique | 34 | 0.7728 | 0.0600 | 0.2245 | 0.5623 |
| L5_acoustic_proxy | 52 | 0.7640 | 0.0505 | 0.2245 | 0.5530 |
