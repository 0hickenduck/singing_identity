# Identity residual paper-closure gate report

## Selected paper framing

**F2 — Dominant-direction masking**, with U1-a and U2-a scope modifiers.

> A shared, train-estimable mode displacement aligned with the dominant variance directions masks cross-mode identity; translation, top-PC removal, and whitening are functionally equivalent repairs of the same subspace in the evaluated frozen SSL representations. The recovery includes single-utterance matching and is stable across disjoint GTSinger song partitions, subject to GTSinger's language–singer confound.

## Gate table

| Experiment | Status | Evidence |
|---|---|---|
| Baseline reproduction | **PASS** | Six OAS-whitened JVS headline rows reproduce 2026-07-13 to 0.00 pp |
| W1 diagonal vs full whitening | **PASS — W1-b: correlations required** | JVS A(diag): wavlm_l12=0.231, hubert_l6=0.224, mert_l3=0.060 |
| W2 spectrum + ABTT | **PASS — W2-a: concentrated + recoverable** | E_d(8): wavlm_l12=0.9999, hubert_l6=0.9998, mert_l3=0.9990; k≤8 recovery fractions: wavlm_l12=0.834, hubert_l6=0.914, mert_l3=0.824 |
| G2x layerwise | **PASS (descriptive)** | Layer 3 is best for corrected JVS R@1 in all three families; WavLM/HuBERT headline layers were not optimal |
| U1 single utterance | **PASS — U1-a: survives** | OAS R@1: wavlm_l12=0.193, hubert_l6=0.525, mert_l3=0.639 vs 0.05 chance; all CIs above chance |
| U2 cross-song | **PASS — U2-a: stable across song sets** | OAS A/B R@1: wavlm_l12=0.916/0.902, hubert_l6=0.910/0.910, mert_l3=0.788/0.782 |
| C1 Chowdhury 2022 | **PASS** | User-supplied PDF confirms DeepCORAL/CORAL+ second-moment adaptation; trial counts and calibration source are not reported in the paper |
| N1 nonlinear mode probe | **NOT RUN (predeclared)** | Final wording is no stronger than dominant linear centroid-level separability; nonlinear search is not triggered |

## Interpretation

Diagonal rescaling is weak and translation still helps after it, so the mechanism is not per-dimension variance imbalance. In contrast, essentially all displacement energy lies in the top eight pooled-covariance PCs, removing k≤8 PCs recovers at least 82% of the full-whitening gain for all three JVS models, and translation has no positive speaker-cluster interval after the selected removal. Random lower-variance/null directions recover approximately zero gain.

The strongest raw OAS backend remains the practical result: 72.8% WavLM, 93.3% HuBERT, and 89.5% MERT centroid R@1. Single-utterance performance drops, especially for WavLM, but remains well above chance under the centroid-fitted transform. Because JVS has only one singing item, its utterance→centroid and utterance→utterance protocols coincide on the gallery side; GTSinger supplies the genuinely varying singing-utterance and cross-song support.

## Boundaries

- Do not call the displacement a pure timbre, identity, or singing vector.
- Do not claim all mode information disappears; only the dominant linear centroid-level separation was tested previously.
- Do not claim protocol-matched superiority over Chowdhury et al. 2022; the inspected PDF confirms that dataset, trial unit, and calibration are not matched or not fully reported.
- GTSinger cross-song support remains language-confounded with singer.
- N1, synthesis, mapper, SeedVC, PLDA reimplementation, and professional/amateur branches are not reopened.

## Closest-work positioning

Chowdhury et al. train a 1D-CNN with DeepCORAL covariance matching and adapt i/x-vector PLDA systems with CORAL+. This converges with the present backend-absorption finding at the level of second-order compensation, but the scientific objects differ: trained domain alignment versus frozen-SSL geometric diagnosis. Their PDF reports severe cross-modal results (DA EER 42.11–44.64%) and says DA does not significantly improve that cross-domain table; those values are not numerically comparable to this controlled 20-speaker centroid gallery.

Run root: `/localdisk/bowen/singing_identity/runs/identity_residual_paper_closure_2026-07-15`
