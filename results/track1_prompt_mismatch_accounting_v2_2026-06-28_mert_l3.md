# Track 1 Accounting v2: Low-Dim Input, Low-Rank Target

Run: `track1_prompt_mismatch_accounting_v2_mert_l3`
Representation: `mert_v1_95m` / `m_a_p_mert_v1_95m` / `3`
Pairs: 4000; speakers: 20

## Target PCA K=8

| model | covariates | delta cosine | MSE red. vs M0 | R@1 before | R@1 after |
|---|---:|---:|---:|---:|---:|
| M0_mean_delta | 0 | 0.5628 | 0.0000 | 0.3332 | 0.5285 |
| L1_nuisance | 10 | 0.5662 | 0.0437 | 0.3332 | 0.5473 |
| L2_nuisance_metadata | 22 | 0.5682 | 0.0308 | 0.3332 | 0.5530 |
| L3_low_content | 28 | 0.5671 | 0.0278 | 0.3332 | 0.5615 |
| L4_technique | 34 | 0.5710 | 0.0352 | 0.3332 | 0.5610 |
| L5_acoustic_proxy | 52 | 0.5748 | 0.0487 | 0.3332 | 0.5525 |

## Target PCA K=16

| model | covariates | delta cosine | MSE red. vs M0 | R@1 before | R@1 after |
|---|---:|---:|---:|---:|---:|
| M0_mean_delta | 0 | 0.5628 | 0.0000 | 0.3332 | 0.5285 |
| L1_nuisance | 10 | 0.5700 | 0.0494 | 0.3332 | 0.5547 |
| L2_nuisance_metadata | 22 | 0.5698 | 0.0339 | 0.3332 | 0.5645 |
| L3_low_content | 28 | 0.5693 | 0.0316 | 0.3332 | 0.5790 |
| L4_technique | 34 | 0.5740 | 0.0401 | 0.3332 | 0.5750 |
| L5_acoustic_proxy | 52 | 0.5769 | 0.0575 | 0.3332 | 0.5543 |

## Target PCA K=32

| model | covariates | delta cosine | MSE red. vs M0 | R@1 before | R@1 after |
|---|---:|---:|---:|---:|---:|
| M0_mean_delta | 0 | 0.5628 | 0.0000 | 0.3332 | 0.5285 |
| L1_nuisance | 10 | 0.5709 | 0.0507 | 0.3332 | 0.5507 |
| L2_nuisance_metadata | 22 | 0.5701 | 0.0341 | 0.3332 | 0.5720 |
| L3_low_content | 28 | 0.5700 | 0.0326 | 0.3332 | 0.5805 |
| L4_technique | 34 | 0.5748 | 0.0414 | 0.3332 | 0.5747 |
| L5_acoustic_proxy | 52 | 0.5787 | 0.0607 | 0.3332 | 0.5590 |

## Target PCA K=64

| model | covariates | delta cosine | MSE red. vs M0 | R@1 before | R@1 after |
|---|---:|---:|---:|---:|---:|
| M0_mean_delta | 0 | 0.5628 | 0.0000 | 0.3332 | 0.5285 |
| L1_nuisance | 10 | 0.5711 | 0.0510 | 0.3332 | 0.5513 |
| L2_nuisance_metadata | 22 | 0.5703 | 0.0347 | 0.3332 | 0.5735 |
| L3_low_content | 28 | 0.5703 | 0.0334 | 0.3332 | 0.5815 |
| L4_technique | 34 | 0.5752 | 0.0422 | 0.3332 | 0.5753 |
| L5_acoustic_proxy | 52 | 0.5796 | 0.0621 | 0.3332 | 0.5647 |
