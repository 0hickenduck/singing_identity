# Track 1 Accounting v2: Low-Dim Input, Low-Rank Target

Run: `track1_prompt_mismatch_accounting_v2_hubert_l12`
Representation: `hubert_base` / `facebook_hubert_base_ls960` / `12`
Pairs: 4000; speakers: 20

## Target PCA K=8

| model | covariates | delta cosine | MSE red. vs M0 | R@1 before | R@1 after |
|---|---:|---:|---:|---:|---:|
| M0_mean_delta | 0 | 0.6639 | 0.0000 | 0.3140 | 0.6302 |
| L1_nuisance | 10 | 0.6699 | 0.0271 | 0.3140 | 0.6920 |
| L2_nuisance_metadata | 22 | 0.6692 | 0.0234 | 0.3140 | 0.6933 |
| L3_low_content | 28 | 0.6736 | 0.0416 | 0.3140 | 0.6955 |
| L4_technique | 34 | 0.6756 | 0.0458 | 0.3140 | 0.6925 |
| L5_acoustic_proxy | 52 | 0.6531 | 0.0071 | 0.3140 | 0.6528 |

## Target PCA K=16

| model | covariates | delta cosine | MSE red. vs M0 | R@1 before | R@1 after |
|---|---:|---:|---:|---:|---:|
| M0_mean_delta | 0 | 0.6639 | 0.0000 | 0.3140 | 0.6302 |
| L1_nuisance | 10 | 0.6693 | 0.0264 | 0.3140 | 0.6977 |
| L2_nuisance_metadata | 22 | 0.6726 | 0.0370 | 0.3140 | 0.6867 |
| L3_low_content | 28 | 0.6747 | 0.0441 | 0.3140 | 0.6923 |
| L4_technique | 34 | 0.6772 | 0.0494 | 0.3140 | 0.6875 |
| L5_acoustic_proxy | 52 | 0.6530 | 0.0075 | 0.3140 | 0.6590 |

## Target PCA K=32

| model | covariates | delta cosine | MSE red. vs M0 | R@1 before | R@1 after |
|---|---:|---:|---:|---:|---:|
| M0_mean_delta | 0 | 0.6639 | 0.0000 | 0.3140 | 0.6302 |
| L1_nuisance | 10 | 0.6708 | 0.0301 | 0.3140 | 0.6913 |
| L2_nuisance_metadata | 22 | 0.6728 | 0.0379 | 0.3140 | 0.6830 |
| L3_low_content | 28 | 0.6749 | 0.0451 | 0.3140 | 0.6900 |
| L4_technique | 34 | 0.6776 | 0.0507 | 0.3140 | 0.6850 |
| L5_acoustic_proxy | 52 | 0.6812 | 0.0695 | 0.3140 | 0.6863 |

## Target PCA K=64

| model | covariates | delta cosine | MSE red. vs M0 | R@1 before | R@1 after |
|---|---:|---:|---:|---:|---:|
| M0_mean_delta | 0 | 0.6639 | 0.0000 | 0.3140 | 0.6302 |
| L1_nuisance | 10 | 0.6707 | 0.0299 | 0.3140 | 0.6920 |
| L2_nuisance_metadata | 22 | 0.6728 | 0.0376 | 0.3140 | 0.6805 |
| L3_low_content | 28 | 0.6750 | 0.0453 | 0.3140 | 0.6905 |
| L4_technique | 34 | 0.6777 | 0.0509 | 0.3140 | 0.6873 |
| L5_acoustic_proxy | 52 | 0.6821 | 0.0717 | 0.3140 | 0.6943 |
