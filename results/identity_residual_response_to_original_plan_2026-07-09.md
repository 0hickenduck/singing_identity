# Response to the Original Identity Residual Experiment Plan

Date: 2026-07-09

This is a technical response to the original experiment plan for the Speech-to-Singing Identity Residual project. I followed the planned gate sequence until the point where the next decision is conceptual/human rather than computational. The short answer is: the experiments support a strong global speech-to-singing mode/duration residual story, but they do not yet support a strong deployable individualized SSL residual mapper.

## 1. Executive Technical Summary

The main planned question was whether the frozen-representation speech-to-singing shift can be decomposed into a global residual plus a stable person-specific residual:

`z_sing,i - z_speech,i = mu_delta + r_i + epsilon_i`

The computational gates now support the global component strongly. Across JVS/JVS-MuSiC speaker-disjoint splits, removing a train-estimated mode/duration-related direction greatly improves cross-mode retrieval for SSL representations. Random, shuffled, and random low-rank controls do not reproduce the gain. B0 global residual accounting also shows high global fractions for SSL models.

The individualized component is weaker. GTSinger split-half reliability passes exploratorily for ECAPA, MERT L3, and WavLM L12, but mapper evaluation only gives stable mapper-over-global evidence for ECAPA. SSL mappers do not meet the strong threshold because their improvements over the mandatory `global_mean_residual` baseline are weak, unstable, or not replicated across two SSL settings.

Therefore I stopped before optional SeedVC/human-listening work. The next step should be a human decision about whether to frame the current result as global mode residual accounting, seek stronger independent singing material for individualized residual claims, or run only a limited downstream sanity check.

## 2. Data and Splits Actually Used

JVS/JVS-MuSiC was used as the main statistical dataset:

- Manifest: `/localdisk/bowen/singing_identity/runs/jvs_music_retrieval_2026-07-08/manifests/jvs_music_utterances.jsonl`
- Feature root: `/localdisk/bowen/singing_identity/features/jvs_music_2026-07-08`
- Split protocol: 60/20/20 speaker-disjoint
- Split seeds: `13 17 19 23 29 31 37 41 43 47 53 59 61 67 71 73 79 83 89 97`
- Test gallery size: 20 speakers per split
- Chance R@1: 5.0%

GTSinger was used as exploratory evidence:

- Split protocol: 10 train singers / 10 test singers, speaker-disjoint
- Split seeds: same 50-seed set used by the B1/B2 GTSinger runs
- Chance R@1: 10.0%
- Interpretation constraint: exploratory only because of small speaker count and possible language/singer confounds.

Compact result locations:

- Synthetic tests: `/home/bowen/bowen_lab/projects/singing_identity/results/synthetic_tests/identity_residual_synthetic_2026-07-09`
- JVS minimal suite: `/home/bowen/bowen_lab/projects/singing_identity/results/identity_residual_suite_2026-07-09_minimal`
- JVS expanded Exp A: `/home/bowen/bowen_lab/projects/singing_identity/results/identity_residual_expA_expanded_2026-07-09`
- JVS duration-balanced sensitivity: `/home/bowen/bowen_lab/projects/singing_identity/results/identity_residual_expA_duration_balanced_2026-07-09`
- JVS full representation sweep: `/home/bowen/bowen_lab/projects/singing_identity/results/identity_residual_jvs_fullsweep_2026-07-09`
- GTSinger B1: `/home/bowen/bowen_lab/projects/singing_identity/results/identity_residual_gtsinger_b1_2026-07-09`
- GTSinger B2: `/home/bowen/bowen_lab/projects/singing_identity/results/identity_residual_gtsinger_b2_2026-07-09`
- Overall gate report: `/home/bowen/bowen_lab/projects/singing_identity/results/identity_residual_gate_report_2026-07-09.md`

## 3. Implementation Notes Relative to the Plan

I implemented a strict experiment driver in `scripts/probing/run_identity_residual_suite.py` with train-speaker-only residualizer fitting, speaker-disjoint splits, mean-preserving residualization, raw/classic variants where relevant, random/shuffled/low-rank controls, per-split outputs, per-speaker rank-change outputs, geometry output, experiment cards, and compact reports.

I also implemented:

- `scripts/probing/run_identity_residual_synthetic.py` for the synthetic gate.
- `scripts/probing/run_identity_residual_mapper_eval.py` for B2 mapper evaluation.
- `tests/test_identity_residual_synthetic.py` for regression coverage of the synthetic gate.

Verification performed after the runs:

- `uv run python -m py_compile scripts/probing/run_identity_residual_suite.py scripts/probing/run_identity_residual_synthetic.py scripts/probing/run_identity_residual_mapper_eval.py tests/test_identity_residual_synthetic.py`
- `uv run python -m unittest tests.test_identity_residual_synthetic`

Both checks passed.

## 4. Synthetic Gate Result

The required synthetic test suite passed before trusting the real-data experiments.

Output: `/home/bowen/bowen_lab/projects/singing_identity/results/synthetic_tests/identity_residual_synthetic_2026-07-09/summary.csv`

The six synthetic cases matched the expected behavior:

- no nuisance/mode dominance
- mode/nuisance dominance
- mode proxy
- within-mode nuisance
- random nuisance
- leakage trap

This supports that the pipeline can distinguish genuine mode/nuisance residualization from random nuisance and leakage-driven improvements in controlled conditions.

## 5. Experiment A: JVS Residualized Retrieval Audit

Experiment A passes as a global mode/duration-axis result, not as a pure acoustic/timbre/F0 removal result.

Selected JVS/JVS-MuSiC full-sweep R@1, S->G / G->S:

| Model | Raw | Mode-dummy only | Single duration | Acoustic full + mode |
|---|---:|---:|---:|---:|
| HuBERT L6 | 19.5% / 29.2% | 60.5% / 75.8% | 51.5% / 63.8% | 40.8% / 61.5% |
| MERT L3 | 36.5% / 36.0% | 66.7% / 78.0% | 62.2% / 73.8% | 51.0% / 70.3% |
| WavLM L6 | 9.0% / 20.8% | 41.2% / 66.2% | 37.0% / 60.8% | 29.0% / 61.3% |
| WavLM L12 | 13.5% / 19.5% | 36.7% / 59.3% | 34.0% / 48.5% | 23.5% / 46.7% |
| ECAPA | 98.8% / 99.5% | near ceiling | near ceiling | near ceiling |

Main interpretation:

- SSL retrieval improves far above chance after mode/duration-related correction.
- The strongest and broadest gains usually come from `mode_dummy_only` or `single_group_duration`.
- ECAPA is already near ceiling, so it is a sanity check rather than a novel identity result.

## 6. Experiment A Controls

The random/shuffled/low-rank controls did not reproduce the real residualization gains.

From the JVS expanded run:

| Model | Variant | S->G R@1 | G->S R@1 |
|---|---|---:|---:|
| WavLM L12 | raw | 13.5% | 19.5% |
| WavLM L12 | row-shuffled | 13.3% | 17.0% |
| WavLM L12 | random Gaussian | 12.0% | 17.8% |
| WavLM L12 | random low-rank | 13.0% | 19.2% |
| MERT L3 | raw | 36.5% | 36.0% |
| MERT L3 | row-shuffled | 36.0% | 36.2% |
| MERT L3 | random Gaussian | 37.5% | 35.8% |
| MERT L3 | random low-rank | 36.5% | 36.0% |

This satisfies the important negative-control gate: the gains are not explained by arbitrary low-rank geometry changes or shuffled nuisance rows.

The control that does explain much of the gain is `mode_dummy_only`, and the strongest measured nuisance group is duration. Therefore the correct interpretation is the allowed template from the original plan:

> The improvement is largely reproduced by removing a train-estimated speech-vs-singing mode direction. Therefore the result should be interpreted primarily as global mode-axis removal, not as specific F0 or prosody removal.

## 7. Duration-Balanced Sensitivity

Duration balancing did not remove the mode/duration effect.

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

This makes the result less likely to be a simple unequal-duration sampling artifact. It still does not justify saying pitch/prosody/timbre was removed; the safer wording is mode/duration-related axis removal.

## 8. B0: Global Residual Accounting

B0 strongly supports a global residual account in SSL spaces.

JVS full-sweep global fraction means:

| Family | Global fraction mean |
|---|---:|
| Acoustic-only | 0.799 |
| ECAPA | 0.173 |
| HuBERT layers | 0.804-0.849 |
| MERT layers | 0.654-0.749 |
| WavLM layers | 0.836-0.873 |

Expanded-run selected values:

- WavLM L12 default global fraction mean 0.845, range 0.762-0.962; mean cosine with train global delta 0.916.
- MERT L3 default global fraction mean 0.654, range 0.581-0.754; mean cosine 0.804.
- Duration-balanced WavLM L12 global fraction mean 0.829, range 0.748-0.945.
- Duration-balanced MERT L3 global fraction mean 0.605, range 0.537-0.711.

Interpretation: the SSL speech-to-singing displacement is dominated by a train-estimable global direction. ECAPA behaves differently because identity is already highly preserved.

## 9. B1: Split-Half Reliability

JVS/JVS-MuSiC B1 is underpowered because local JVS singing material does not provide the kind of independent repeated singing halves needed for a strong person-specific reliability claim.

GTSinger was therefore used as exploratory B1 evidence with utterance-disjoint halves. It passes as exploratory reliability:

| Model | r A->B R@1 mean | Range | Same-minus-different margin |
|---|---:|---:|---:|
| ECAPA | 97.6% | 90.0-100.0% | 0.728 |
| MERT L3 | 94.4% | 80.0-100.0% | 0.692 |
| WavLM L12 | 92.0% | 80.0-100.0% | 0.579 |

Because this is GTSinger with 20 singers, the correct conclusion is “exploratory pass,” not a final main-dataset proof.

## 10. B2: Mapper Evaluation Against Global Baseline

The mapper gate does not support a strong individualized SSL mapper claim.

GTSinger B2 used 50 speaker-disjoint 10/10 splits. The mandatory baseline was `global_mean_residual`.

ECAPA passes:

- `global_mean_residual` R@1: 95.6%
- `speech_embedding_ridge_full` R@1: 96.6%; residual MSE reduction vs global 14.6%, CI 4.9-22.3%
- `speech_embedding_ridge_pca_target` R@1: 97.4%; residual MSE reduction 13.6%, CI 5.8-19.1%
- acoustic-only, wrong-speaker, random, and shuffle controls all have negative MSE reduction.

MERT L3 is weak/unstable:

- `global_mean_residual` R@1: 62.2%
- `speech_embedding_ridge_pca_target` R@1: 68.8%; MSE reduction 16.9%, CI -9.2 to 31.8%
- `speech_embedding_plus_acoustic_ridge_pca_target` R@1: 69.8%; MSE reduction 17.6%, CI -7.4 to 32.2%
- Controls are negative, but the key MSE CI includes 0.

WavLM L12 fails to beat global reliably:

- `global_mean_residual` R@1: 70.0%
- learned mappers are around 66.4-67.2% R@1 and do not beat global retrieval.
- MSE reduction confidence intervals include 0.

The original plan required the learned model to beat global stably and appear in at least two SSL settings, or one preregistered SSL setting plus support. That threshold is not met.

## 11. Confound Checks and Deviations From the Full Plan

Completed:

- train-speaker-only residualizer and preprocessing
- speaker-disjoint splits
- exact chance levels
- random/shuffled/random low-rank controls
- duration-balanced sensitivity
- broad per-speaker rank-change audit
- synthetic leakage trap
- JVS full representation sweep across acoustic, ECAPA, HuBERT, MERT, and WavLM layers
- GTSinger exploratory B1 and B2 gates

Partially completed or not completed:

- The full 5-fold JVS speaker CV was not run after the 20-seed 60/20/20 protocol already gave a stable decision-relevant pattern.
- Mode-probe AUC and nuisance-decodability probes were not fully added as separate classifier/decoder outputs. The decision was still clear from Exp A controls plus B0 global accounting, but this remains a useful methodological add-on if the final paper needs the exact A5 evidence form.
- Within-gender and within-language restricted-gallery checks were not completed in this gate report.
- Content/song controls were logged conceptually, but not turned into a full restricted-gallery analysis.
- Optional SeedVC downstream sanity was intentionally not run because B2 did not strongly pass for SSL individualized mapping.

These deviations do not change the current gate decision, but they do matter if the paper wants to make stronger claims than “global mode/duration residual accounting.”

## 12. Final Decision Tree Outcome

The result matches the original plan's Case B and B2-A:

Case B: `mode_dummy_only` matches or exceeds full nuisance in many settings.

- Interpretation: the gain is mostly removal of a global speech-vs-singing mode axis. F0/energy/duration may be proxies for mode.
- Next step from original plan: focus paper story on global mode residual `mu_delta`; still run B, but expect global baseline to be strong.

Case B2-A: global residual beats speech-only, learned mapper does not reliably beat global in SSL spaces.

- Paper story: frozen representations contain a strong global speech-to-singing mode residual. Current deployable mappers do not reliably predict person-specific SSL residuals beyond this global shift.
- Next step: focus analysis on `mu_delta`, its alignment with nuisance/mode directions, and how it affects prompt/mode mismatch.

## 13. Claims Currently Supported

Supported:

1. Train-speaker-only removal of a speech-vs-singing mode/duration-related direction improves held-out cross-mode identity retrieval in frozen SSL representations.
2. The effect is not reproduced by shuffled nuisance, Gaussian random nuisance, or random low-rank projection controls.
3. The JVS/JVS-MuSiC SSL speech-to-singing residual is largely global, with high B0 global fractions across HuBERT, MERT, and WavLM layers.
4. GTSinger exploratory split-half evidence suggests a stable residual can exist within that dataset.
5. ECAPA can support mapper-over-global improvements, but it is a supervised speaker embedding and near-ceiling sanity space rather than the main SSL novelty.

## 14. Claims Not Supported Yet

Not supported:

1. We isolated pure timbre.
2. We removed pitch, F0, or prosody in a complete causal sense.
3. The person-specific residual is strongly established on the main JVS/JVS-MuSiC data.
4. A deployable individualized SSL mapper works beyond global residual.
5. SeedVC prompt-mode behavior is explained by this residual without downstream sanity checks.
6. GTSinger alone establishes a generalizable identity-residual result.

## 15. Exact Recommended Next Step

I recommend stopping automatic computation here and asking for human feedback on the intended claim.

Decision needed:

1. If the desired paper claim is about representation analysis, the clean next direction is to write the story around global speech-to-singing mode/duration residual accounting and add only the missing mode-probe/nuisance-decodability checks needed for reviewer comfort.
2. If the desired claim is individualized residual mapping, we need stronger independent singing material or a more reliable repeated-singing protocol before investing more in mappers.
3. If downstream SVC is still desired, it should be framed as an optional sanity check of global residual adaptation, not as proof of individualized mapper success.

My default recommendation is option 1: proceed with the global residual accounting story and do not claim a deployable individualized SSL mapper yet.
