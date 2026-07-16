# Track 1 Accounting v2: Low-Dim Input, Low-Rank Target

Run: `track1_prompt_mismatch_accounting_v2_controls_mert_l3`
Representation: `mert_v1_95m` / `m_a_p_mert_v1_95m` / `3`
Pairs: 4000; speakers: 20

## Target PCA K=16

| model | covariates | delta cosine | MSE red. vs M0 | R@1 before | R@1 after |
|---|---:|---:|---:|---:|---:|
| M0_mean_delta | 0 | 0.5628 | 0.0000 | 0.3332 | 0.5285 |
| L1_nuisance | 10 | 0.5700 | 0.0494 | 0.3332 | 0.5547 |
| L2_nuisance_metadata | 22 | 0.5698 | 0.0339 | 0.3332 | 0.5645 |
| L3_low_content | 28 | 0.5693 | 0.0316 | 0.3332 | 0.5790 |
| L4_technique | 34 | 0.5740 | 0.0401 | 0.3332 | 0.5750 |
| C_metadata_only | 12 | 0.5625 | -0.0002 | 0.3332 | 0.5278 |
| C_content_only | 6 | 0.5613 | -0.0018 | 0.3332 | 0.5295 |
| C_shuffled_nuisance | 10 | 0.5627 | -0.0000 | 0.3332 | 0.5290 |
| C_l4_shuffle_technique | 34 | 0.5692 | 0.0315 | 0.3332 | 0.5770 |
| L5_acoustic_proxy | 52 | 0.5769 | 0.0575 | 0.3332 | 0.5543 |
| C_acoustic_only | 18 | 0.5849 | 0.0749 | 0.3332 | 0.5450 |

## Target PCA K=64

| model | covariates | delta cosine | MSE red. vs M0 | R@1 before | R@1 after |
|---|---:|---:|---:|---:|---:|
| M0_mean_delta | 0 | 0.5628 | 0.0000 | 0.3332 | 0.5285 |
| L1_nuisance | 10 | 0.5711 | 0.0510 | 0.3332 | 0.5513 |
| L2_nuisance_metadata | 22 | 0.5703 | 0.0347 | 0.3332 | 0.5735 |
| L3_low_content | 28 | 0.5703 | 0.0334 | 0.3332 | 0.5815 |
| L4_technique | 34 | 0.5752 | 0.0422 | 0.3332 | 0.5753 |
| C_metadata_only | 12 | 0.5626 | -0.0001 | 0.3332 | 0.5262 |
| C_content_only | 6 | 0.5624 | -0.0004 | 0.3332 | 0.5330 |
| C_shuffled_nuisance | 10 | 0.5627 | -0.0000 | 0.3332 | 0.5288 |
| C_l4_shuffle_technique | 34 | 0.5702 | 0.0332 | 0.3332 | 0.5817 |
| L5_acoustic_proxy | 52 | 0.5796 | 0.0621 | 0.3332 | 0.5647 |
| C_acoustic_only | 18 | 0.5878 | 0.0789 | 0.3332 | 0.5533 |
