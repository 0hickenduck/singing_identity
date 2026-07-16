# Track 1 Accounting v2: Low-Dim Input, Low-Rank Target

Run: `track1_prompt_mismatch_accounting_v2_controls_mert_l12`
Representation: `mert_v1_95m` / `m_a_p_mert_v1_95m` / `12`
Pairs: 4000; speakers: 20

## Target PCA K=16

| model | covariates | delta cosine | MSE red. vs M0 | R@1 before | R@1 after |
|---|---:|---:|---:|---:|---:|
| M0_mean_delta | 0 | 0.5536 | 0.0000 | 0.3155 | 0.4258 |
| L1_nuisance | 10 | 0.5730 | 0.0549 | 0.3155 | 0.4665 |
| L2_nuisance_metadata | 22 | 0.5692 | 0.0451 | 0.3155 | 0.4773 |
| L3_low_content | 28 | 0.5660 | 0.0413 | 0.3155 | 0.4855 |
| L4_technique | 34 | 0.5693 | 0.0464 | 0.3155 | 0.4903 |
| C_metadata_only | 12 | 0.5504 | -0.0042 | 0.3155 | 0.4373 |
| C_content_only | 6 | 0.5549 | 0.0038 | 0.3155 | 0.4325 |
| C_shuffled_nuisance | 10 | 0.5536 | -0.0000 | 0.3155 | 0.4260 |
| C_l4_shuffle_technique | 34 | 0.5659 | 0.0413 | 0.3155 | 0.4858 |
| L5_acoustic_proxy | 52 | 0.5716 | 0.0538 | 0.3155 | 0.5080 |
| C_acoustic_only | 18 | 0.5688 | 0.0435 | 0.3155 | 0.4915 |

## Target PCA K=64

| model | covariates | delta cosine | MSE red. vs M0 | R@1 before | R@1 after |
|---|---:|---:|---:|---:|---:|
| M0_mean_delta | 0 | 0.5536 | 0.0000 | 0.3155 | 0.4258 |
| L1_nuisance | 10 | 0.5739 | 0.0563 | 0.3155 | 0.4723 |
| L2_nuisance_metadata | 22 | 0.5697 | 0.0461 | 0.3155 | 0.4820 |
| L3_low_content | 28 | 0.5716 | 0.0495 | 0.3155 | 0.4840 |
| L4_technique | 34 | 0.5745 | 0.0545 | 0.3155 | 0.4863 |
| C_metadata_only | 12 | 0.5505 | -0.0041 | 0.3155 | 0.4380 |
| C_content_only | 6 | 0.5550 | 0.0039 | 0.3155 | 0.4350 |
| C_shuffled_nuisance | 10 | 0.5536 | -0.0000 | 0.3155 | 0.4260 |
| C_l4_shuffle_technique | 34 | 0.5715 | 0.0494 | 0.3155 | 0.4850 |
| L5_acoustic_proxy | 52 | 0.5738 | 0.0574 | 0.3155 | 0.5118 |
| C_acoustic_only | 18 | 0.5711 | 0.0464 | 0.3155 | 0.4955 |
