# Identity Residual Final Validation

## Technical Summary

The final validation supports the paper-ready global residual claim: direct train-speaker speaker-balanced global adaptation reproduces much of the mode-dummy retrieval gain, while wrong-sign and random-vector controls do not. The result remains strongest for JVS/JVS-MuSiC SSL representations and remains framed as global mode/duration-correlated correction, not pitch or timbre removal.

The specified shuffled-global control is non-diagnostic: shuffling train singing centroids before averaging leaves the same mean vector when every train speaker has both modes. It is included in the CSV and marked as mathematically degenerate.

## F1 Global Adapter

- JVS_JVSMuSiC wavlm_l12: raw S->G R@1 13.5%; speech+global R@1 36.2%; wrong-sign R@1 10.2%.
- JVS_JVSMuSiC mert_l3: raw S->G R@1 36.5%; speech+global R@1 66.0%; wrong-sign R@1 18.0%.
- JVS_JVSMuSiC hubert_l6: raw S->G R@1 19.5%; speech+global R@1 60.5%; wrong-sign R@1 10.8%.
- GTSinger wavlm_l12: raw S->G R@1 19.4%; speech+global R@1 70.0%; wrong-sign R@1 14.2%.
- GTSinger mert_l3: raw S->G R@1 33.2%; speech+global R@1 62.2%; wrong-sign R@1 18.0%.

## F2 Speaker-Balanced Direction

The speaker-balanced vector reproduces the utterance-weighted mode-vector result. On JVS/JVS-MuSiC, `cos_utterance_to_speaker_balanced` is 1.000 for WavLM L12, MERT L3, HuBERT L6, and ECAPA. The duration-weighted vector is also almost collinear with the speaker-balanced vector: WavLM L12 0.99994, MERT L3 0.99994, HuBERT L6 0.99996.

This supports the simpler interpretation that the useful correction is a train-estimated global speech-to-singing vector, not an artifact of utterance-count imbalance.

## F3 Mode Probe

- JVS_JVSMuSiC wavlm_l12: centroid mode AUC changes from 1.000 to 0.500 after speaker-balanced correction.
- JVS_JVSMuSiC mert_l3: centroid mode AUC changes from 1.000 to 0.462 after speaker-balanced correction.
- JVS_JVSMuSiC hubert_l6: centroid mode AUC changes from 1.000 to 0.453 after speaker-balanced correction.

## F4 Nuisance Decodability

Nuisance decodability rows use speaker-mode centroids and train-speaker-only ridge probes. Interpret them as linear decodability changes, not causal removal of pitch, prosody, or timbre.

For duration decodability on JVS/JVS-MuSiC, raw and random low-rank corrected embeddings remain highly duration-decodable, while speaker-balanced global correction nearly removes linear duration decodability:

- WavLM L12: raw R2 0.992, random low-rank 0.992, speaker-balanced global 0.025.
- MERT L3: raw R2 0.977, random low-rank 0.977, speaker-balanced global 0.019.
- HuBERT L6: raw R2 0.991, random low-rank 0.991, speaker-balanced global 0.029.

This is consistent with the mode/duration-correlated direction being removed by the global correction. It is not evidence that duration is causal or that all duration/prosody information is removed.

## F5 Duration Proxy Audit

Duration is mostly a mode/segmentation proxy in JVS/JVS-MuSiC. Duration alone predicts mode with AUC 1.000 for WavLM L12, MERT L3, and HuBERT L6 splits because the local speech and singing segments have very different duration distributions.

The duration-predicted embedding direction is also highly aligned with the global mode vector:

- WavLM L12: cos(delta_duration, mu_delta) 0.978.
- MERT L3: cos(delta_duration, mu_delta) 0.979.
- HuBERT L6: cos(delta_duration, mu_delta) 0.969.

Within-mode duration residualization does not reproduce the global adapter gain: WavLM L12 raw 13.5% vs within-mode duration 12.8% vs global adapter 36.2%; MERT L3 raw 36.5% vs within-mode duration 36.8% vs global 66.0%; HuBERT L6 raw 19.5% vs within-mode duration 18.8% vs global 60.5%. The final classification is therefore `mostly_mode_proxy`.

## F6 Restricted Gallery

JVS within-gender retrieval preserves the main correction effect:

- WavLM L12: raw 16.0%, speech+global 40.0%.
- MERT L3: raw 37.5%, speech+global 66.5%.
- HuBERT L6: raw 24.5%, speech+global 60.7%.

This argues against the JVS result collapsing when coarse gender information is controlled in the gallery. GTSinger within-language rows are saved, but many same-language test galleries are too small; GTSinger remains exploratory and potentially language-confounded.

## F7 JVS Mapper Status

JVS B2 was not present before this validation, so I ran the requested minimal JVS mapper check for WavLM L12, MERT L3, and HuBERT L6 over the same 20 speaker-disjoint splits. Output: `expB_mapper_eval/summary.csv`.

The mapper accounting loop is now closed, but it does not change the final claim. Learned SSL mappers reduce residual MSE versus global with positive CIs, but retrieval gains over `global_mean_residual` are small or inconsistent, acoustic-only is competitive for MERT L3, and JVS still lacks an adequate independent split-half singing reliability ceiling.

- WavLM L12: global R@1 36.2%; speech PCA mapper R@1 36.5%; MSE reduction vs global 15.8%, CI 11.4-21.9%.
- MERT L3: global R@1 66.0%; speech PCA mapper R@1 70.5%; acoustic-only R@1 71.5%; MSE reduction vs global 19.0%, CI 15.0-24.2%.
- HuBERT L6: global R@1 60.5%; speech PCA mapper R@1 63.3%; MSE reduction vs global 21.5%, CI 16.4-28.0%.

This is at most weak mapper evidence. It does not satisfy the original strong individualized mapper standard because the JVS reliability target is not independently validated and the retrieval improvements do not robustly clear the +5 pp threshold across two SSL settings while beating acoustic-only controls.

## Supported Claim

Frozen SSL representations contain a strong train-estimable speech-to-singing global mode residual. Correcting this global mode/duration-correlated direction improves cross-mode same-person retrieval on held-out speakers. The effect is not reproduced by shuffled nuisance, random Gaussian nuisance, or random low-rank controls from the main suite, and the direct global-adapter validation supports the same interpretation. Current evidence does not support a deployable individualized SSL residual mapper beyond the global baseline.

## Unsupported Claims

- We removed pitch.
- We isolated timbre.
- We learned singing identity residuals.
- Duration is causal.
- SeedVC behavior is solved.
- GTSinger proves general person-specific residuals.
