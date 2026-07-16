# Identity Residual: Protocol Robustness and Matched-Frame Follow-Up

## Technical Summary

The global speech-to-singing displacement survives the requested protocol stress tests. On held-out speakers, correct-sign correction improves verification EER and train-calibrated TMR@FMR=1% across JVS/JVS-MuSiC and clean GTSinger same-text control data, including when the speech side is a single utterance rather than a full centroid. It also survives a strict GTSinger equal-100-frame, voiced-ratio-matched reevaluation.

This strengthens the paper claim to `train-estimable global speech-to-singing displacement improves identity-relevant verification and retrieval geometry`. It does not establish pure timbre, duration removal, causal segmentation, a one-dimensional residual, or a deployable individualized mapper.

## Verification Is Not Only A Small-Gallery Ranking Effect

All TMR thresholds were selected from train-speaker impostor trials. Held-out EER is reported as descriptive ROC crossing; no test labels select a deployment threshold. Genuine trials are same-speaker cross-mode pairs and every off-diagonal cross-speaker pair is an impostor; train speakers never enter test trials.

| Dataset | Model | Raw R@1 -> corrected | Raw EER -> corrected | Raw TMR@1% -> corrected |
|---|---:|---:|---:|---:|
| JVS/JVS-MuSiC | WavLM L12 | 13.5% -> 36.3% | 42.1% -> 26.6% | 3.0% -> 14.5% |
| JVS/JVS-MuSiC | MERT L3 | 36.5% -> 66.0% | 28.5% -> 16.1% | 14.3% -> 38.8% |
| JVS/JVS-MuSiC | HuBERT L6 | 19.5% -> 60.5% | 37.1% -> 18.6% | 2.0% -> 30.0% |
| GTSinger same-text | WavLM L12 | 19.8% -> 73.8% | 37.5% -> 20.9% | 5.4% -> 33.4% |
| GTSinger same-text | MERT L3 | 27.4% -> 60.4% | 35.4% -> 23.7% | 8.4% -> 20.2% |
| GTSinger same-text | HuBERT L6 | 19.8% -> 76.2% | 38.2% -> 18.5% | 4.0% -> 28.2% |

Wrong-sign correction degrades performance and same-norm random directions remain near raw. ROC/DET point tables are in [protocol robustness results](/home/bowen/bowen_lab/projects/singing_identity/results/identity_residual_protocol_robustness_2026-07-10); no rendered figure is claimed when the optional plotting backend is unavailable.

The mean genuine-minus-impostor cosine separation is not uniformly larger after correction, despite the consistent EER/TMR improvements. This is a useful qualification: the result is a better score distribution and ranking geometry, not a claim that every simple cosine-margin summary must monotonically grow.

## Limited Reference And Gallery Robustness

One speech utterance remains informative. For example, one-reference S->G R@1 changes from 13.0% to 28.6% (WavLM), 35.0% to 59.8% (MERT), and 18.2% to 50.8% (HuBERT) on JVS; on same-text GTSinger it changes from 21.3% to 61.1%, 28.6% to 52.1%, and 20.5% to 66.8% respectively.

The gain remains at gallery sizes 5, 10, and 20 where supported by the fixed held-out split. JVS 50-way galleries were intentionally not added because the main 60/20/20 split has only 20 held-out speakers; changing that split would confound the comparison. Full per-budget and per-gallery outputs are in `reference_budget_*` and `gallery_size_*` CSVs.

## Matched-Frame Control

The GTSinger subset starts with 1,953 lexical-equal, control-technique pairs across 20 singers. Under the predeclared 100-frame requirement, WavLM and HuBERT each retain 1,930 pairs and MERT retains 1,951; all 20 singers remain represented. The model-specific too-short pairs are explicitly logged rather than padded. Each retained speech and singing crop has exactly 100 representation frames. Among deterministic candidate crops, the selected pair minimizes the difference in voiced-frame fraction; pair-level exclusions and matching diagnostics are retained in `matched_frame_crop_audit.csv`.

| Model | Raw R@1 -> corrected | Paired EER delta | Paired R@1 delta |
|---|---:|---:|---:|
| WavLM L12 | 29.2% -> 76.8% | -11.4 pp | +47.6 pp |
| MERT L3 | 38.6% -> 71.0% | -5.0 pp | +32.4 pp |
| HuBERT L6 | 31.8% -> 72.2% | -12.9 pp | +40.4 pp |

The matched global direction remains closely aligned with the full-utterance direction (WavLM splitwise cosine roughly 0.94-0.96). Fixed-frame symmetric correction reduces GTSinger mode AUC from about 1.0 to 0.48-0.53. This is a matched-frame reevaluation, not a claim that duration was removed.

## Held-Out Structure And Uncertainty

Held-out residuals align positively with the train global vector: mean alignment cosine ranges from 0.78 to 0.92 across the six dataset/model settings. Speaker-level aggregation, which does not treat repeated appearances of the same speaker as independent subjects, finds positive rank improvements for 71%-85% of speakers. The speaker-bootstrap intervals exclude zero and paired sign-permutation p-values are at most 0.0005.

A shared mean explains 0.65-0.83 of residual energy, but the centered residual is not one-dimensional. Effective rank is about 6 on GTSinger and 23-28 on JVS. The complete eigenvalue and cumulative-variance outputs are in [residual spectrum results](/home/bowen/bowen_lab/projects/singing_identity/results/identity_residual_spectrum_2026-07-10). No low-rank identity adapter was applied: a non-oracle test-speech-only coefficient rule was not pre-specified, and learning one would restart the stopped mapper line.

## Duration Audit Cleanup

The cleaned duration table is [duration_audit_cleaned_summary.csv](/home/bowen/bowen_lab/projects/singing_identity/results/identity_residual_protocol_robustness_2026-07-10/duration_audit_cleaned_summary.csv). It has one row per dataset/model/target/condition and explicitly records raw versus corrected numerical basis, within- versus between-mode status, and the classification label. The interpretation remains that duration is a strong mode/segmentation proxy, not a causal explanation.

## Remaining Limits

- JVS speech and singing content remain unmatched, although the GTSinger support set is independently lexical-equal.
- GTSinger has 20 singers and language is correlated with identity; it supports content control but is not broad demographic generalization.
- EER is a descriptive held-out statistic; TMR thresholds are train-calibrated. GTSinger cannot estimate TMR@0.1% with only 90 train impostor trials, so those cells are deliberately marked insufficient.
- The result does not justify an individualized residual mapper, SeedVC conclusion, pitch removal, pure timbre interpretation, or a one-dimensional latent claim.

## Recommended Paper Package

Use verification EER/TMR, one-reference retrieval, the matched-frame result, held-out alignment, speaker-level uncertainty, and the residual-spectrum qualification as the core robustness package. No further representation, mapper, or cross-dataset-transfer experiment is currently necessary for the stated global-displacement paper claim.
