# Track 1 Accounting v2: Low-Dim Input, Low-Rank Target

Run: `track1_prompt_mismatch_accounting_v2_controls_wavlm_l6`
Representation: `wavlm_base_plus` / `microsoft_wavlm_base_plus` / `6`
Pairs: 4000; speakers: 20

## Target PCA K=16

| model | covariates | delta cosine | MSE red. vs M0 | R@1 before | R@1 after |
|---|---:|---:|---:|---:|---:|
| M0_mean_delta | 0 | 0.7681 | 0.0000 | 0.2245 | 0.5225 |
| L1_nuisance | 10 | 0.7775 | 0.0719 | 0.2245 | 0.5830 |
| L2_nuisance_metadata | 22 | 0.7671 | 0.0390 | 0.2245 | 0.5695 |
| L3_low_content | 28 | 0.7692 | 0.0489 | 0.2245 | 0.5623 |
| L4_technique | 34 | 0.7718 | 0.0572 | 0.2245 | 0.5577 |
| C_metadata_only | 12 | 0.7680 | 0.0002 | 0.2245 | 0.5215 |
| C_content_only | 6 | 0.7698 | 0.0195 | 0.2245 | 0.5160 |
| C_shuffled_nuisance | 10 | 0.7681 | -0.0001 | 0.2245 | 0.5208 |
| C_l4_shuffle_technique | 34 | 0.7691 | 0.0484 | 0.2245 | 0.5615 |
| L5_acoustic_proxy | 52 | 0.7642 | 0.0564 | 0.2245 | 0.5497 |
| C_acoustic_only | 18 | 0.7790 | 0.0903 | 0.2245 | 0.5950 |

## Target PCA K=64

| model | covariates | delta cosine | MSE red. vs M0 | R@1 before | R@1 after |
|---|---:|---:|---:|---:|---:|
| M0_mean_delta | 0 | 0.7681 | 0.0000 | 0.2245 | 0.5225 |
| L1_nuisance | 10 | 0.7777 | 0.0725 | 0.2245 | 0.5863 |
| L2_nuisance_metadata | 22 | 0.7675 | 0.0405 | 0.2245 | 0.5677 |
| L3_low_content | 28 | 0.7700 | 0.0515 | 0.2245 | 0.5645 |
| L4_technique | 34 | 0.7728 | 0.0600 | 0.2245 | 0.5623 |
| C_metadata_only | 12 | 0.7682 | 0.0007 | 0.2245 | 0.5202 |
| C_content_only | 6 | 0.7702 | 0.0209 | 0.2245 | 0.5200 |
| C_shuffled_nuisance | 10 | 0.7681 | -0.0001 | 0.2245 | 0.5215 |
| C_l4_shuffle_technique | 34 | 0.7698 | 0.0508 | 0.2245 | 0.5633 |
| L5_acoustic_proxy | 52 | 0.7640 | 0.0505 | 0.2245 | 0.5530 |
| C_acoustic_only | 18 | 0.7792 | 0.0899 | 0.2245 | 0.5913 |
