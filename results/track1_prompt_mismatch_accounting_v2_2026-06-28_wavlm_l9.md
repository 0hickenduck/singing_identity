# Track 1 Accounting v2: Low-Dim Input, Low-Rank Target

Run: `track1_prompt_mismatch_accounting_v2_wavlm_l9`
Representation: `wavlm_base_plus` / `microsoft_wavlm_base_plus` / `9`
Pairs: 4000; speakers: 20

## Target PCA K=8

| model | covariates | delta cosine | MSE red. vs M0 | R@1 before | R@1 after |
|---|---:|---:|---:|---:|---:|
| M0_mean_delta | 0 | 0.7677 | 0.0000 | 0.2402 | 0.5443 |
| L1_nuisance | 10 | 0.7759 | 0.0847 | 0.2402 | 0.6185 |
| L2_nuisance_metadata | 22 | 0.7680 | 0.0477 | 0.2402 | 0.5975 |
| L3_low_content | 28 | 0.7721 | 0.0672 | 0.2402 | 0.6100 |
| L4_technique | 34 | 0.7729 | 0.0697 | 0.2402 | 0.6032 |
| L5_acoustic_proxy | 52 | 0.7681 | 0.0823 | 0.2402 | 0.6018 |

## Target PCA K=16

| model | covariates | delta cosine | MSE red. vs M0 | R@1 before | R@1 after |
|---|---:|---:|---:|---:|---:|
| M0_mean_delta | 0 | 0.7677 | 0.0000 | 0.2402 | 0.5443 |
| L1_nuisance | 10 | 0.7758 | 0.0835 | 0.2402 | 0.6128 |
| L2_nuisance_metadata | 22 | 0.7700 | 0.0560 | 0.2402 | 0.6010 |
| L3_low_content | 28 | 0.7724 | 0.0691 | 0.2402 | 0.6102 |
| L4_technique | 34 | 0.7737 | 0.0728 | 0.2402 | 0.6075 |
| L5_acoustic_proxy | 52 | 0.7695 | 0.0865 | 0.2402 | 0.5970 |

## Target PCA K=32

| model | covariates | delta cosine | MSE red. vs M0 | R@1 before | R@1 after |
|---|---:|---:|---:|---:|---:|
| M0_mean_delta | 0 | 0.7677 | 0.0000 | 0.2402 | 0.5443 |
| L1_nuisance | 10 | 0.7764 | 0.0861 | 0.2402 | 0.6245 |
| L2_nuisance_metadata | 22 | 0.7705 | 0.0578 | 0.2402 | 0.6030 |
| L3_low_content | 28 | 0.7731 | 0.0714 | 0.2402 | 0.6132 |
| L4_technique | 34 | 0.7746 | 0.0754 | 0.2402 | 0.6088 |
| L5_acoustic_proxy | 52 | 0.7698 | 0.0854 | 0.2402 | 0.5985 |

## Target PCA K=64

| model | covariates | delta cosine | MSE red. vs M0 | R@1 before | R@1 after |
|---|---:|---:|---:|---:|---:|
| M0_mean_delta | 0 | 0.7677 | 0.0000 | 0.2402 | 0.5443 |
| L1_nuisance | 10 | 0.7764 | 0.0860 | 0.2402 | 0.6220 |
| L2_nuisance_metadata | 22 | 0.7702 | 0.0570 | 0.2402 | 0.6005 |
| L3_low_content | 28 | 0.7731 | 0.0714 | 0.2402 | 0.6148 |
| L4_technique | 34 | 0.7746 | 0.0754 | 0.2402 | 0.6100 |
| L5_acoustic_proxy | 52 | 0.7692 | 0.0817 | 0.2402 | 0.5985 |
