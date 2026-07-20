# Expanded Exp A Report

20-seed JVS/JVS-MuSiC residualization audit for the speech-to-singing identity residual project.

## Data And Outputs

- dataset: `JVS_JVSMuSiC`
- manifest: `/localdisk/bowen/singing_identity/runs/jvs_music_retrieval_2026-07-08/manifests/jvs_music_utterances.jsonl`
- feature root: `/localdisk/bowen/singing_identity/features/jvs_music_2026-07-08`
- run root: `/localdisk/bowen/singing_identity/runs/identity_residual_expA_expanded_2026-07-09`
- default compact results: `/home/bowen/bowen_lab/projects/singing_identity/results/identity_residual_expA_expanded_2026-07-09`
- duration-balanced compact results: `/home/bowen/bowen_lab/projects/singing_identity/results/identity_residual_expA_duration_balanced_2026-07-09`
- split seeds: `13 17 19 23 29 31 37 41 43 47 53 59 61 67 71 73 79 83 89 97`
- test gallery size: 20 speakers per split, chance R@1 = 5.0%

## Synthetic Gate

The prior synthetic gate passed at `/home/bowen/bowen_lab/projects/singing_identity/results/synthetic_tests/identity_residual_synthetic_2026-07-09`.

## Default Exp A

Full table: `expA_residualization_audit/summary.csv`.

WavLM L12:

- raw: S->G R@1 13.5%, G->S R@1 19.5%
- mode-dummy-only mean-preserving: S->G 36.7% (delta +23.3 pp), G->S 59.3% (delta +39.8 pp)
- acoustic-full-no-mode mean-preserving: S->G 22.5% (delta +9.0 pp; split CI includes 0), G->S 45.8% (delta +26.3 pp; split CI includes 0)
- acoustic-full-plus-mode mean-preserving: S->G 23.5% (delta +10.0 pp; split CI includes 0), G->S 46.7% (delta +27.3 pp; split CI low +2.4 pp)

MERT L3:

- raw: S->G R@1 36.5%, G->S R@1 36.0%
- mode-dummy-only mean-preserving: S->G 66.7% (delta +30.3 pp), G->S 78.0% (delta +42.0 pp)
- acoustic-full-no-mode mean-preserving: S->G 49.3% (delta +12.8 pp; split CI includes 0), G->S 70.5% (delta +34.5 pp)
- acoustic-full-plus-mode mean-preserving: S->G 51.0% (delta +14.5 pp; split CI includes 0), G->S 70.3% (delta +34.2 pp)

## Controls

Random and shuffled controls did not reproduce the gains.

- WavLM L12 row-shuffled: S->G 13.3%, G->S 17.0%
- WavLM L12 random Gaussian: S->G 12.0%, G->S 17.8%
- WavLM L12 random low-rank: S->G 13.0%, G->S 19.2%
- MERT L3 row-shuffled: S->G 36.0%, G->S 36.2%
- MERT L3 random Gaussian: S->G 37.5%, G->S 35.8%
- MERT L3 random low-rank: S->G 36.5%, G->S 36.0%

This passes the non-specific geometry control: the improvement is not reproduced by removing arbitrary/random subspaces or shuffled nuisance.

## Nuisance Group Findings

The strongest single measured group is duration:

- WavLM L12 single duration: S->G 34.0%, G->S 48.5%
- MERT L3 single duration: S->G 62.2%, G->S 73.8%

Removing duration from the full nuisance set removes much of the gain:

- WavLM L12 full-minus-duration: S->G 10.5%, G->S 18.7%
- MERT L3 full-minus-duration: S->G 42.2%, G->S 60.3%

F0-only did not help and often hurt retrieval. Voicing is unavailable in this manifest, so voicing-only is identical to raw.

Interpretation: measured duration, likely as a proxy for the speech-vs-singing data construction and global mode difference, explains more of the gain than F0-specific nuisance removal.

## Duration-Balanced Sensitivity

Full table: `/home/bowen/bowen_lab/projects/singing_identity/results/identity_residual_expA_duration_balanced_2026-07-09/expA_residualization_audit/summary.csv`.

Duration-balanced WavLM L12:

- raw: S->G 14.3%, G->S 19.3%
- mode-dummy-only: S->G 37.5%, G->S 59.3%
- acoustic-full-no-mode: S->G 29.0%, G->S 45.3%
- single duration: S->G 39.5%, G->S 56.2%
- full-minus-duration: S->G 17.5%, G->S 21.5%

Duration-balanced MERT L3:

- raw: S->G 43.5%, G->S 40.5%
- mode-dummy-only: S->G 71.0%, G->S 75.7%
- acoustic-full-no-mode: S->G 53.0%, G->S 69.5%
- single duration: S->G 70.8%, G->S 77.0%
- full-minus-duration: S->G 44.7%, G->S 62.5%

The duration-balanced run does not remove the duration/mode effect. It makes the same interpretation stronger: the useful correction is a global mode/duration-related axis, not a clean isolated F0 or timbre control.

## Broad Speaker Effect

Per-speaker rows are in `expA_residualization_audit/per_speaker.csv`.

Default run, rank-improvement share over 800 evaluated query cases per model/variant:

- WavLM L12 mode-dummy-only: 71.3% improved, 16.3% degraded, median rank improvement +4
- WavLM L12 acoustic-full-plus-mode: 59.8% improved, 26.8% degraded, median +2
- MERT L3 mode-dummy-only: 58.8% improved, 6.8% degraded, median +1
- MERT L3 acoustic-full-plus-mode: 46.8% improved, 19.8% degraded, median 0

The broadest improvement is mode-dummy-only; acoustic-full improvements are less uniformly speaker-wide.

## B0 Global Residual

Full table: `expB_global_and_reliability/b0_global_residual_accounting.csv`.

Default:

- WavLM L12 global fraction mean 0.845, range 0.762-0.962; mean cosine with train global delta 0.916
- MERT L3 global fraction mean 0.654, range 0.581-0.754; mean cosine 0.804

Duration-balanced:

- WavLM L12 global fraction mean 0.829, range 0.748-0.945; mean cosine 0.908
- MERT L3 global fraction mean 0.605, range 0.537-0.711; mean cosine 0.770

The global residual remains strong after duration balancing.

## Decision

Current evidence supports a global mode-axis account:

Train-speaker-only removal of speech-vs-singing mode/duration-related directions improves held-out cross-mode retrieval, and random/shuffled controls do not explain the gain. The effect is not strong evidence for a stable individualized residual or pure timbre disentanglement.

## Not Supported Yet

- “F0 removed” or “pitch removed”
- pure timbre isolation
- stable person-specific residual under independent singing takes
- deployable mapper beyond global residual

## Next Recommended Step

Do not run a large B2 mapper claim on current JVS alone. The next useful step is to seek or construct a genuinely independent singing split for B1, likely using GTSinger same-singer multi-utterance material as exploratory evidence, while keeping JVS as the main global-mode residual result.
