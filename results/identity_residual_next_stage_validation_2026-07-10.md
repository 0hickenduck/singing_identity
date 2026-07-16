## Validation Report

### Overall Assessment: Share with caveats

### Methodology Review

The follow-up analysis answers the requested question: whether the previously observed global correction survives verification metrics, limited speech references, matched-frame conditions, speaker-level aggregation, and gallery changes. All reported correction vectors are estimated from train speakers. Test speakers are excluded from training trials; TMR thresholds are train-calibrated. The analysis does not fit an individualized mapper or use test singing targets to select a residual direction.

### Calculation Spot-Checks

- Verification coverage: verified. `verification_per_split.csv` contains 11,130 rows: JVS has 3 models x 20 splits x 53 variants/draws and GTSinger has 3 models x 50 splits x 53 variants/draws.
- Reference coverage: verified. `reference_budget_per_split.csv` contains 42,420 rows, corresponding to the predeclared 1/2/5/10/full budgets and same raw/corrected reference samples.
- Gallery coverage: verified. `gallery_size_per_split.csv` contains 24,000 rows for only feasible held-out gallery sizes; omitted 50-way JVS and 20-way GTSinger are documented rather than silently substituted.
- Speaker grain: verified. `heldout_residual_alignment_per_speaker_split.csv` has 2,700 rows, while `speaker_level_uncertainty.csv` aggregates repeated appearances before bootstrapping and testing.
- Matched-frame eligibility: verified. The crop audit records frame counts, accepted crop count, and voiced-ratio difference for each selected pair. The full run retained all 20 singers.
- Duration audit: verified. The cleaned table has one row per dataset/model/target/condition rather than repeated headline model rows.

### Issues Found

1. Medium: JVS speech and singing contents are unmatched. The GTSinger lexical-equal subset addresses this as supporting evidence, but not as a replacement for a large same-text Japanese corpus.
2. Medium: GTSinger has only 20 singers and language is associated with singer identity. Its intervals are speaker-aggregated, but generalization should remain qualified.
3. Low: Mean genuine-minus-impostor cosine separation is not uniformly larger after correction even when EER/TMR improve. The report therefore avoids a monotone mean-margin claim.
4. Low: TMR@FMR=0.1% is unavailable on GTSinger because its 10 training speakers yield only 90 impostor calibration trials. The output marks this insufficient rather than reporting a misleading estimate.

### Visualization Review

The report links the full ROC/DET point tables. No static ROC image is represented as evidence when the optional plotting backend is unavailable. The report tables state population, split basis, and whether an operating point is train-calibrated.

### Required Caveats for Stakeholders

- Describe the finding as a representation-space global displacement and verification/retrieval improvement.
- Do not claim duration removal, pitch removal, pure timbre, causal segmentation, one-dimensionality, or an individualized deployable mapper.
- Treat GTSinger as controlled supporting evidence with small-sample and language-confounding limitations.
