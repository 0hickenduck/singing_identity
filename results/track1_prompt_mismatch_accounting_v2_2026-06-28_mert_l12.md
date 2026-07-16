# Track 1 Accounting v2: Low-Dim Input, Low-Rank Target

Run: `track1_prompt_mismatch_accounting_v2_mert_l12`
Representation: `mert_v1_95m` / `m_a_p_mert_v1_95m` / `12`
Pairs: 4000; speakers: 20

## Target PCA K=8

| model | covariates | delta cosine | MSE red. vs M0 | R@1 before | R@1 after |
|---|---:|---:|---:|---:|---:|
| M0_mean_delta | 0 | 0.5536 | 0.0000 | 0.3155 | 0.4258 |
| L1_nuisance | 10 | 0.5723 | 0.0555 | 0.3155 | 0.4733 |
| L2_nuisance_metadata | 22 | 0.5692 | 0.0453 | 0.3155 | 0.4770 |
| L3_low_content | 28 | 0.5657 | 0.0416 | 0.3155 | 0.4845 |
| L4_technique | 34 | 0.5684 | 0.0458 | 0.3155 | 0.4940 |
| L5_acoustic_proxy | 52 | 0.5694 | 0.0521 | 0.3155 | 0.5118 |

## Target PCA K=16

| model | covariates | delta cosine | MSE red. vs M0 | R@1 before | R@1 after |
|---|---:|---:|---:|---:|---:|
| M0_mean_delta | 0 | 0.5536 | 0.0000 | 0.3155 | 0.4258 |
| L1_nuisance | 10 | 0.5730 | 0.0549 | 0.3155 | 0.4665 |
| L2_nuisance_metadata | 22 | 0.5692 | 0.0451 | 0.3155 | 0.4773 |
| L3_low_content | 28 | 0.5660 | 0.0413 | 0.3155 | 0.4855 |
| L4_technique | 34 | 0.5693 | 0.0464 | 0.3155 | 0.4903 |
| L5_acoustic_proxy | 52 | 0.5716 | 0.0538 | 0.3155 | 0.5080 |

## Target PCA K=32

| model | covariates | delta cosine | MSE red. vs M0 | R@1 before | R@1 after |
|---|---:|---:|---:|---:|---:|
| M0_mean_delta | 0 | 0.5536 | 0.0000 | 0.3155 | 0.4258 |
| L1_nuisance | 10 | 0.5738 | 0.0561 | 0.3155 | 0.4685 |
| L2_nuisance_metadata | 22 | 0.5697 | 0.0460 | 0.3155 | 0.4813 |
| L3_low_content | 28 | 0.5714 | 0.0492 | 0.3155 | 0.4848 |
| L4_technique | 34 | 0.5743 | 0.0542 | 0.3155 | 0.4863 |
| L5_acoustic_proxy | 52 | 0.5734 | 0.0568 | 0.3155 | 0.5130 |

## Target PCA K=64

| model | covariates | delta cosine | MSE red. vs M0 | R@1 before | R@1 after |
|---|---:|---:|---:|---:|---:|
| M0_mean_delta | 0 | 0.5536 | 0.0000 | 0.3155 | 0.4258 |
| L1_nuisance | 10 | 0.5739 | 0.0563 | 0.3155 | 0.4723 |
| L2_nuisance_metadata | 22 | 0.5697 | 0.0461 | 0.3155 | 0.4820 |
| L3_low_content | 28 | 0.5716 | 0.0495 | 0.3155 | 0.4840 |
| L4_technique | 34 | 0.5745 | 0.0545 | 0.3155 | 0.4863 |
| L5_acoustic_proxy | 52 | 0.5738 | 0.0574 | 0.3155 | 0.5118 |
