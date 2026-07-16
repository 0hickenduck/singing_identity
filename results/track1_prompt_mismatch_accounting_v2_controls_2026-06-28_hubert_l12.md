# Track 1 Accounting v2: Low-Dim Input, Low-Rank Target

Run: `track1_prompt_mismatch_accounting_v2_controls_hubert_l12`
Representation: `hubert_base` / `facebook_hubert_base_ls960` / `12`
Pairs: 4000; speakers: 20

## Target PCA K=16

| model | covariates | delta cosine | MSE red. vs M0 | R@1 before | R@1 after |
|---|---:|---:|---:|---:|---:|
| M0_mean_delta | 0 | 0.6639 | 0.0000 | 0.3140 | 0.6302 |
| L1_nuisance | 10 | 0.6693 | 0.0264 | 0.3140 | 0.6977 |
| L2_nuisance_metadata | 22 | 0.6726 | 0.0370 | 0.3140 | 0.6867 |
| L3_low_content | 28 | 0.6747 | 0.0441 | 0.3140 | 0.6923 |
| L4_technique | 34 | 0.6772 | 0.0494 | 0.3140 | 0.6875 |
| C_metadata_only | 12 | 0.6600 | -0.0096 | 0.3140 | 0.6275 |
| C_content_only | 6 | 0.6679 | 0.0174 | 0.3140 | 0.6565 |
| C_shuffled_nuisance | 10 | 0.6639 | -0.0000 | 0.3140 | 0.6300 |
| C_l4_shuffle_technique | 34 | 0.6745 | 0.0437 | 0.3140 | 0.6947 |
| L5_acoustic_proxy | 52 | 0.6530 | 0.0075 | 0.3140 | 0.6590 |
| C_acoustic_only | 18 | 0.6660 | 0.0188 | 0.3140 | 0.6690 |

## Target PCA K=64

| model | covariates | delta cosine | MSE red. vs M0 | R@1 before | R@1 after |
|---|---:|---:|---:|---:|---:|
| M0_mean_delta | 0 | 0.6639 | 0.0000 | 0.3140 | 0.6302 |
| L1_nuisance | 10 | 0.6707 | 0.0299 | 0.3140 | 0.6920 |
| L2_nuisance_metadata | 22 | 0.6728 | 0.0376 | 0.3140 | 0.6805 |
| L3_low_content | 28 | 0.6750 | 0.0453 | 0.3140 | 0.6905 |
| L4_technique | 34 | 0.6777 | 0.0509 | 0.3140 | 0.6873 |
| C_metadata_only | 12 | 0.6659 | 0.0061 | 0.3140 | 0.6330 |
| C_content_only | 6 | 0.6680 | 0.0176 | 0.3140 | 0.6573 |
| C_shuffled_nuisance | 10 | 0.6639 | -0.0000 | 0.3140 | 0.6295 |
| C_l4_shuffle_technique | 34 | 0.6748 | 0.0447 | 0.3140 | 0.6907 |
| L5_acoustic_proxy | 52 | 0.6821 | 0.0717 | 0.3140 | 0.6943 |
| C_acoustic_only | 18 | 0.6669 | 0.0204 | 0.3140 | 0.6703 |
