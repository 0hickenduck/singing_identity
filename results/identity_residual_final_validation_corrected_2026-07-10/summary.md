# Identity Residual Final Validation

## Technical Summary

The final validation supports the paper-ready global residual claim: direct train-speaker speaker-balanced global adaptation reproduces much of the mode-dummy retrieval gain, while wrong-sign and random-vector controls do not. The result remains strongest for JVS/JVS-MuSiC SSL representations and remains framed as global mode/duration-correlated correction, not pitch or timbre removal.

The specified shuffled-global control is non-diagnostic: shuffling train singing centroids before averaging leaves the same mean vector when every train speaker has both modes. It is included in the CSV and marked as mathematically degenerate.

## F1 Global Adapter

- JVS_JVSMuSiC wavlm_l12: raw S->G R@1 13.5%; speech+global R@1 36.2%; paired split delta 22.8% (95% bootstrap CI 19.8% to 25.8%, n=20); wrong-sign R@1 10.2%; paired delta -3.3% (95% bootstrap CI -5.0% to -1.7%).
- JVS_JVSMuSiC mert_l3: raw S->G R@1 36.5%; speech+global R@1 66.0%; paired split delta 29.5% (95% bootstrap CI 25.0% to 34.0%, n=20); wrong-sign R@1 18.0%; paired delta -18.5% (95% bootstrap CI -22.2% to -14.8%).
- JVS_JVSMuSiC hubert_l6: raw S->G R@1 19.5%; speech+global R@1 60.5%; paired split delta 41.0% (95% bootstrap CI 36.5% to 45.7%, n=20); wrong-sign R@1 10.8%; paired delta -8.8% (95% bootstrap CI -11.0% to -6.5%).
- GTSinger wavlm_l12: raw S->G R@1 19.4%; speech+global R@1 70.0%; paired split delta 50.6% (95% bootstrap CI 47.0% to 54.4%, n=50); wrong-sign R@1 14.2%; paired delta -5.2% (95% bootstrap CI -6.6% to -3.8%).
- GTSinger mert_l3: raw S->G R@1 33.2%; speech+global R@1 62.2%; paired split delta 29.0% (95% bootstrap CI 24.4% to 33.6%, n=50); wrong-sign R@1 18.0%; paired delta -15.2% (95% bootstrap CI -17.8% to -12.6%).

Paired deltas for all deterministic headline variants are in `global_adapter_paired_delta_summary.csv`. Each delta pairs target and baseline R@1 on the same split seed; the CI bootstraps the mean of those paired split deltas. Repeated split partitions may share speakers, so this is split-robustness uncertainty rather than an independent-subject CI.

## F2 Speaker-Balanced Direction

The speaker-balanced `mu_delta` rows and utterance-weighted mode-vector rows are saved in `speaker_balanced_mode_vector_results.csv`. The key audit fields are `cos_utterance_to_speaker_balanced` and `cos_duration_weighted_to_speaker_balanced`; high positive alignment means the simpler global-vector interpretation is not an utterance-count artifact.

## F3 Mode Probe

Mode results use L2-penalized logistic regression fit to train-speaker centroids only. Standardization, coefficients, intercept, and the balanced-accuracy/F1 operating threshold are all learned on train speakers; held-out test speakers are used only for AUC, balanced accuracy, and F1 evaluation.

- JVS_JVSMuSiC wavlm_l12: centroid mode AUC changes from 1.000 to 0.497 after speaker-balanced correction.
- JVS_JVSMuSiC mert_l3: centroid mode AUC changes from 1.000 to 0.474 after speaker-balanced correction.
- JVS_JVSMuSiC hubert_l6: centroid mode AUC changes from 1.000 to 0.466 after speaker-balanced correction.

## F4 Nuisance Decodability

Nuisance decodability rows use speaker-mode centroids and train-speaker-only ridge probes. Interpret them as linear decodability changes, not causal removal of pitch, prosody, or timbre.

## F5 Duration Proxy Audit

- JVS_JVSMuSiC hubert_l6: duration-only mode AUC mean 1.000; classification `mostly_mode_proxy`.
- JVS_JVSMuSiC hubert_l6: duration-only mode AUC mean 1.000; classification `within_mode_nuisance_or_unresolved`.
- JVS_JVSMuSiC mert_l3: duration-only mode AUC mean 1.000; classification `mostly_mode_proxy`.
- JVS_JVSMuSiC mert_l3: duration-only mode AUC mean 1.000; classification `within_mode_nuisance_or_unresolved`.
- JVS_JVSMuSiC wavlm_l12: duration-only mode AUC mean 1.000; classification `mostly_mode_proxy`.
- JVS_JVSMuSiC wavlm_l12: duration-only mode AUC mean 1.000; classification `within_mode_nuisance_or_unresolved`.

## F6 Restricted Gallery

Restricted-gallery rows are saved in `restricted_gallery_results.csv`. JVS within-gender retrieval is the main confound check. Rows state when no query has an eligible gallery; small-gallery interpretation should use the reported gallery-size distribution rather than a blanket denominator rule.
Chance R@1 is the mean across eligible queries of `1 / query-specific gallery size`; each row also reports the minimum, median, maximum, and frequency distribution of query-specific gallery sizes.

## F7 JVS Mapper Status

JVS B2 mapper status is reported separately in the final validation report when the minimal mapper output is present. If no JVS mapper output exists, the correct status is that the global-residual claim does not rely on a positive mapper result.

## Supported Claim

Frozen SSL representations contain a strong train-estimable speech-to-singing global mode residual. Correcting this global mode/duration-correlated direction improves cross-mode same-person retrieval on held-out speakers. The effect is not reproduced by shuffled nuisance, random Gaussian nuisance, or random low-rank controls from the main suite, and the direct global-adapter validation supports the same interpretation. Current evidence does not support a deployable individualized SSL residual mapper beyond the global baseline.

## Unsupported Claims

- We removed pitch.
- We isolated timbre.
- We learned singing identity residuals.
- Duration is causal.
- SeedVC behavior is solved.
- GTSinger proves general person-specific residuals.
