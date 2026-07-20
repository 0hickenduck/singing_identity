# Final Experiment Report

Interim minimal strict suite for the speech-to-singing identity residual project. This is not the full 20-seed/layer sweep.

## 1. Data Actually Used

- dataset: `JVS_JVSMuSiC`
- manifest: `/localdisk/bowen/singing_identity/runs/jvs_music_retrieval_2026-07-08/manifests/jvs_music_utterances.jsonl`
- feature root: `/localdisk/bowen/singing_identity/features/jvs_music_2026-07-08`
- run root: `/localdisk/bowen/singing_identity/runs/identity_residual_suite_2026-07-09_minimal`
- compact results: `/home/bowen/bowen_lab/projects/singing_identity/results/identity_residual_suite_2026-07-09_minimal`
- manifest audit found 10,061 utterances, 100 speakers, and one singing utterance per speaker.

## 2. Speaker Splits Actually Used

- split seeds: `13,17,19`
- protocol: 60/20/20 speaker-disjoint
- test gallery size: 20 speakers per split, so chance R@1 = 5.0%
- speech cap for this minimal run: first 20 speech utterances per speaker

## 3. Residualization Implementation Details

- residualizers and nuisance scalers were fit on train-speaker utterances only.
- primary formula reported here: mean-preserving residualization.
- variants included raw, center-only classic, mode-dummy-only, acoustic-full-no-mode, acoustic-full-plus-mode, row-shuffled nuisance, random Gaussian nuisance, and random low-rank projection.
- nuisance columns `rms_std` and `voiced_rate` were entirely missing in the JVS manifest and should not be interpreted as active controls.

## 4. Synthetic Test Result

Synthetic gate passed at `/home/bowen/bowen_lab/projects/singing_identity/results/synthetic_tests/identity_residual_synthetic_2026-07-09`.

The six cases passed: no nuisance/mode dominance, mode nuisance dominance, mode proxy, within-mode nuisance, random nuisance, and leakage trap.

## 5. Experiment A Result

Full table: `expA_residualization_audit/summary.csv`.

Mean-preserving WavLM L12:

- raw S->G R@1: 18.3%; G->S R@1: 23.3%
- mode-dummy-only S->G R@1: 45.0%; G->S R@1: 70.0%
- acoustic-full-no-mode S->G R@1: 21.7%; G->S R@1: 35.0%
- acoustic-full-plus-mode S->G R@1: 18.3%; G->S R@1: 41.7%

Mean-preserving MERT L3:

- raw S->G R@1: 36.7%; G->S R@1: 36.7%
- mode-dummy-only S->G R@1: 71.7%; G->S R@1: 86.7%
- acoustic-full-no-mode S->G R@1: 61.7%; G->S R@1: 78.3%
- acoustic-full-plus-mode S->G R@1: 63.3%; G->S R@1: 80.0%

## 6. Experiment A Controls

Random controls did not reproduce the main gains:

- WavLM L12 row-shuffled nuisance: S->G 13.3%, G->S 16.7%
- WavLM L12 random Gaussian nuisance: S->G 13.3%, G->S 18.3%
- WavLM L12 random low-rank projection: S->G 18.3%, G->S 23.3%
- MERT L3 row-shuffled nuisance: S->G 35.0%, G->S 36.7%
- MERT L3 random Gaussian nuisance: S->G 35.0%, G->S 38.3%
- MERT L3 random low-rank projection: S->G 36.7%, G->S 36.7%

The strongest pattern is that mode-dummy-only explains much or all of the improvement, especially for WavLM L12. This supports a global speech-vs-singing mode-axis interpretation more than a narrow acoustic nuisance interpretation.

## 7. Global Residual Accounting

Full table: `expB_global_and_reliability/b0_global_residual_accounting.csv`.

- WavLM L12 global fraction by seed: 0.901, 0.962, 0.803
- WavLM L12 mean cosine between held-out delta and train global delta: 0.919, 0.918, 0.913
- MERT L3 global fraction by seed: 0.664, 0.746, 0.616
- MERT L3 mean cosine between held-out delta and train global delta: 0.814, 0.808, 0.819

This is strong evidence that the speech-to-singing shift is substantially global in these representations.

## 8. Split-Half Reliability

Full table: `expB_global_and_reliability/b1_split_half_reliability.csv`.

All B1 rows are labeled `underpowered_single_singing_file` because the local JVS-MuSiC manifest has one singing utterance per speaker. The script used non-overlapping frame halves for singing, so these numbers are within-file reliability checks, not robust song-disjoint or take-disjoint residual reliability.

Observed within-file residual R@1 was 20/20 for both WavLM L12 and MERT L3 across all three seeds, but this cannot support the strong B1 claim required by the long spec.

## 9. Mapper Result

B2 mapper evaluation was intentionally not run in this minimal suite. The reason is methodological: B1 is underpowered for JVS because the singing side is a single file per speaker, and the spec says not to spend effort on complex mappers before checking residual stability.

## 10. Confound Checks

The minimal run did not include within-gender, duration-balanced, leave-one-group-out, or full bootstrap checks. Those remain required before final claims.

## 11. Final Decision Tree Outcome

Current outcome is closest to Case B / global mode residual:

Mode-dummy-only and B0 global residual accounting explain a large part of the retrieval improvement. Random/shuffled controls do not explain the gain. The current evidence supports global mode-axis removal more strongly than individualized residual mapping.

## 12. What Claim Is Currently Supported

Allowed interim claim:

Train-speaker-only removal of a speech-vs-singing mode direction substantially improves held-out cross-mode retrieval in WavLM L12 and MERT L3 on JVS/JVS-MuSiC. The effect is not reproduced by row-shuffled nuisance, random Gaussian nuisance, or random low-rank controls in this minimal run.

## 13. What Claim Is Not Yet Supported

Not supported yet:

- pure timbre isolation
- full F0/prosody removal
- stable person-specific residual under independent singing takes
- deployable mapper improvement beyond global residual
- final paper-level claim from this 3-seed minimal suite

## 14. Exact Next Experiment Recommended

Run the expanded Exp A audit with 20 seeds and bootstrap CIs for MERT L3 plus WavLM L12, then add leave-one-group-out and duration-balanced controls. For B1, use a dataset or segmentation protocol with genuinely independent singing material; otherwise keep JVS B1 labeled as within-file reliability only and do not proceed to strong B2 mapper claims.
