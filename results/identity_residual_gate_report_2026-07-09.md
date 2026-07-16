# Identity Residual Gate Report

This report summarizes the computational gate sequence run on 2026-07-09.

## Outputs

- JVS full representation sweep: `/home/bowen/bowen_lab/projects/singing_identity/results/identity_residual_jvs_fullsweep_2026-07-09`
- JVS duration-balanced sensitivity: `/home/bowen/bowen_lab/projects/singing_identity/results/identity_residual_expA_duration_balanced_2026-07-09`
- GTSinger B1 reliability: `/home/bowen/bowen_lab/projects/singing_identity/results/identity_residual_gtsinger_b1_2026-07-09`
- GTSinger B2 mapper: `/home/bowen/bowen_lab/projects/singing_identity/results/identity_residual_gtsinger_b2_2026-07-09`

## Gate A: JVS Residualized Retrieval

JVS/JVS-MuSiC full sweep used 20 speaker-disjoint seeds with 20 held-out test speakers per split.

Main pattern:

- ECAPA is already near ceiling raw: 98.8% / 99.5% R@1.
- SSL representations improve strongly after mode-dummy or duration-related residualization.
- Random, shuffled, and random-low-rank controls do not reproduce the gains.
- B0 global residual fractions are high for SSL models: HuBERT 0.804-0.849, MERT 0.654-0.749, WavLM 0.836-0.873.

Selected JVS R@1, S->G / G->S:

- HuBERT L6: raw 19.5% / 29.2%; mode-dummy 60.5% / 75.8%; duration 51.5% / 63.8%
- MERT L3: raw 36.5% / 36.0%; mode-dummy 66.7% / 78.0%; duration 62.2% / 73.8%
- WavLM L6: raw 9.0% / 20.8%; mode-dummy 41.2% / 66.2%; duration 37.0% / 60.8%
- WavLM L12: raw 13.5% / 19.5%; mode-dummy 36.7% / 59.3%; duration 34.0% / 48.5%

Gate A outcome: pass for global mode/duration-axis account; not a clean F0/timbre-removal result.

## Gate B1: GTSinger Split-Half Reliability

GTSinger has utterance-disjoint speech and singing halves, so it is a useful exploratory B1 dataset despite only 20 singers.

Residual `r` split-half reliability over 50 seeds:

- ECAPA: R@1 mean 97.6%, range 90.0-100.0%; same-minus-diff margin 0.728
- MERT L3: R@1 mean 94.4%, range 80.0-100.0%; margin 0.692
- WavLM L12: R@1 mean 92.0%, range 80.0-100.0%; margin 0.579

Gate B1 outcome: exploratory pass. The residual is stable within GTSinger, but this does not override JVS as the main dataset.

## Gate B2: GTSinger Mapper Evaluation

Mapper evaluation used 50 GTSinger 10/10 speaker-disjoint splits. The mandatory baseline is `global_mean_residual`.

ECAPA:

- global R@1: 95.6%
- speech ridge full R@1: 96.6%; residual MSE reduction vs global 14.6%, CI 4.9-22.3%
- speech PCA ridge R@1: 97.4%; MSE reduction 13.6%, CI 5.8-19.1%
- acoustic-only, wrong-speaker, random, and shuffle controls all have negative MSE reduction.

MERT L3:

- global R@1: 62.2%
- speech PCA ridge R@1: 68.8%; MSE reduction 16.9%, CI -9.2 to 31.8%
- speech+acoustic PCA ridge R@1: 69.8%; MSE reduction 17.6%, CI -7.4 to 32.2%
- controls are negative, but the MSE CI includes 0.

WavLM L12:

- global R@1: 70.0%
- learned mappers do not beat global retrieval; MSE reduction CIs include 0.

Gate B2 outcome: weak/exploratory only. ECAPA passes, but SSL evidence does not meet the strong threshold of stable improvement in at least two SSL settings. Therefore, do not claim a deployable individualized SSL mapper.

## Decision

The computational gates now support this story:

Frozen SSL representations contain a strong global speech-to-singing mode/duration residual. Removing that train-estimated direction improves held-out cross-mode identity retrieval. GTSinger exploratory split-half reliability shows a stable residual can exist, but mapper evidence beyond global residual is not yet strong for SSL representations.

Not supported:

- pure timbre isolation
- full F0/prosody removal
- strong deployable person-specific SSL residual mapper
- SeedVC downstream claim

## Stop Point

This is the point for human judgment. The next decision is conceptual, not computational:

1. Use the current evidence for a paper story centered on global mode residual accounting.
2. Search for stronger independent singing material before pursuing individualized mapper claims.
3. Only run SeedVC/human listening if we decide a global residual downstream sanity check is worth the engineering cost.
