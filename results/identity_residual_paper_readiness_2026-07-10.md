# Identity Residual Paper-Readiness Update

Date: 2026-07-10

## Decision

The required content-control gate passes. The paper may retain the term `global speech-to-singing mode residual`, provided it remains explicitly representation-level and is not described as pure timbre, pitch removal, or a waveform transformation.

No further mapper tuning or cross-dataset transfer is required before writing. Cross-dataset transfer remains an optional appendix experiment, not a blocker.

## Why This Experiment Was Needed

The main JVS/JVS-MuSiC evaluation is same-person but not same-text: singing uses the common song `katatsumuri`, while speech uses JVS `parallel100` sentences. Its global vector can therefore contain mode, content distribution, segmentation, and recording-domain effects.

The new supporting experiment uses GTSinger pairs with:

- `technique == control`;
- independently checked lexical equality after NFKC normalization;
- `<AP>/<SP>` boundary markers and whitespace ignored;
- unique speech and singing utterance IDs;
- speaker-disjoint 10-train/10-test splits;
- one speaker, one vote when estimating the train global vector.

The manifest `same_text_flag` was not trusted because it is hard-coded by the builder. The resulting clean subset contains 1,953 pairs, 3,906 distinct utterances, all 20 singers, and nine languages. All selected WavLM L12, MERT L3, HuBERT L6, and ECAPA feature files were present.

## Same-Text Global Adapter Result

Fifty speaker-disjoint seeds and 50 same-norm random-vector draws per seed were used. Test galleries contain 10 held-out singers, so chance R@1 is 10%.

| Model | Raw S->G R@1 | Correct global | Wrong sign | Random same norm | Paired delta vs raw, 95% CI | Gate |
|---|---:|---:|---:|---:|---:|---|
| WavLM L12 | 19.8% | 73.8% | 14.4% | 20.3% | +54.0 pp [51.0, 57.0] | pass |
| MERT L3 | 27.4% | 60.4% | 19.6% | 28.9% | +33.0 pp [28.2, 37.6] | pass |
| HuBERT L6 | 19.8% | 76.2% | 13.6% | 19.6% | +56.4 pp [53.2, 59.6] | pass |
| ECAPA sanity | 92.8% | 98.6% | 83.2% | 92.5% | +5.8 pp [4.2, 7.4] | sanity only |

The reverse G->S direction also improves: WavLM L12 +45.0 pp [41.2, 48.8], MERT L3 +44.0 pp [40.6, 47.4], and HuBERT L6 +46.6 pp [42.8, 50.4].

The preregistered gate required at least two SSL models with at least +5 pp R@1 improvement, paired CI excluding zero, and wrong-sign/random controls not reproducing the gain. All three SSL models pass.

## Mode Mechanism

The train-only L2-penalized logistic mode probe remains perfectly separable before correction (mean AUC 1.0) and falls near chance after symmetric global correction:

| Model | Raw mode AUC | Corrected mode AUC |
|---|---:|---:|
| WavLM L12 | 1.000 | 0.466 |
| MERT L3 | 1.000 | 0.538 |
| HuBERT L6 | 1.000 | 0.513 |
| ECAPA sanity | 1.000 | 0.506 |

This content-controlled result supports the mechanism claim that a shared mode-dominant direction obscures cross-mode same-person matching. It does not show that timbre was isolated or that all mode information was removed.

## Corrected Final-Validation Artifacts

The earlier final-validation implementation was also rerun after four paper-readiness fixes:

1. Mode AUC now uses train-only L2-penalized logistic regression rather than a difference-of-means probe.
2. Restricted-gallery chance is the mean query-specific `1/gallery_size`, with gallery-size distributions recorded.
3. Per-speaker raw/corrected ranks and nearest impostors are populated.
4. Headline global-adapter comparisons include paired-across-split bootstrap CIs.

The corrected JVS conclusions are unchanged:

- WavLM L12 global-adapter delta +22.8 pp, CI [19.8, 25.8].
- MERT L3 delta +29.5 pp, CI [25.0, 34.0].
- HuBERT L6 delta +41.0 pp, CI [36.5, 45.7].
- Corrected mode AUC: WavLM L12 0.497, MERT L3 0.474, HuBERT L6 0.466.
- Within-gender query-specific chance is about 10%, not the previously reported blanket 5%; the correction gain remains.

## Final Supported Claim

Frozen SSL audio representations contain a strong train-estimable global speech-to-singing mode residual. A speaker-balanced global vector estimated only from training speakers substantially improves cross-mode same-person retrieval for held-out speakers. The effect survives a same-text, control-technique validation and is not reproduced by wrong-sign or same-norm random vectors. Correcting the vector also reduces linear mode separability to near chance. Current evidence still does not support a deployable individualized SSL residual mapper beyond the global baseline.

## Remaining Limitations

- GTSinger has only 20 singers, and language is strongly coupled to singer identity.
- The JVS main result remains content-unmatched across speech and singing, although content distributions are shared across speakers.
- The correction is representation-space alignment, not waveform-domain conversion.
- No pure timbre, causal duration, complete pitch/prosody removal, or SeedVC claim is supported.

## Reproducibility Paths

- Same-text report: `results/identity_residual_same_text_2026-07-10/README_results.md`
- Same-text paired deltas: `results/identity_residual_same_text_2026-07-10/paired_delta_summary.csv`
- Same-text pair audit: `results/identity_residual_same_text_2026-07-10/pair_selection_audit.csv`
- Corrected final validation: `results/identity_residual_final_validation_corrected_2026-07-10/`
- Same-text run root and caches: `/localdisk/bowen/singing_identity/runs/identity_residual_same_text_2026-07-10`
