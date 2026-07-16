# Track 1 Accounting v2: Low-Dim Input, Low-Rank Target

Run: `track1_prompt_mismatch_accounting_v2_controls_wavlm_l9`
Representation: `wavlm_base_plus` / `microsoft_wavlm_base_plus` / `9`
Pairs: 4000; speakers: 20

## Target PCA K=16

| model | covariates | delta cosine | MSE red. vs M0 | R@1 before | R@1 after |
|---|---:|---:|---:|---:|---:|
| M0_mean_delta | 0 | 0.7677 | 0.0000 | 0.2402 | 0.5443 |
| L1_nuisance | 10 | 0.7758 | 0.0835 | 0.2402 | 0.6128 |
| L2_nuisance_metadata | 22 | 0.7700 | 0.0560 | 0.2402 | 0.6010 |
| L3_low_content | 28 | 0.7724 | 0.0691 | 0.2402 | 0.6102 |
| L4_technique | 34 | 0.7737 | 0.0728 | 0.2402 | 0.6075 |
| C_metadata_only | 12 | 0.7689 | 0.0051 | 0.2402 | 0.5555 |
| C_content_only | 6 | 0.7714 | 0.0317 | 0.2402 | 0.5523 |
| C_shuffled_nuisance | 10 | 0.7677 | -0.0001 | 0.2402 | 0.5427 |
| C_l4_shuffle_technique | 34 | 0.7722 | 0.0686 | 0.2402 | 0.6092 |
| L5_acoustic_proxy | 52 | 0.7695 | 0.0865 | 0.2402 | 0.5970 |
| C_acoustic_only | 18 | 0.7704 | 0.0679 | 0.2402 | 0.6172 |

## Target PCA K=64

| model | covariates | delta cosine | MSE red. vs M0 | R@1 before | R@1 after |
|---|---:|---:|---:|---:|---:|
| M0_mean_delta | 0 | 0.7677 | 0.0000 | 0.2402 | 0.5443 |
| L1_nuisance | 10 | 0.7764 | 0.0860 | 0.2402 | 0.6220 |
| L2_nuisance_metadata | 22 | 0.7702 | 0.0570 | 0.2402 | 0.6005 |
| L3_low_content | 28 | 0.7731 | 0.0714 | 0.2402 | 0.6148 |
| L4_technique | 34 | 0.7746 | 0.0754 | 0.2402 | 0.6100 |
| C_metadata_only | 12 | 0.7691 | 0.0055 | 0.2402 | 0.5573 |
| C_content_only | 6 | 0.7719 | 0.0332 | 0.2402 | 0.5547 |
| C_shuffled_nuisance | 10 | 0.7677 | -0.0001 | 0.2402 | 0.5435 |
| C_l4_shuffle_technique | 34 | 0.7729 | 0.0706 | 0.2402 | 0.6160 |
| L5_acoustic_proxy | 52 | 0.7692 | 0.0817 | 0.2402 | 0.5985 |
| C_acoustic_only | 18 | 0.7707 | 0.0684 | 0.2402 | 0.6170 |
