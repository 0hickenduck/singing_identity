# Experiment Registry

This file is the compact index of meaningful runs. It is not a log dump.
Every new experiment should add a short entry with the run name, question,
script/config, run root, result, and next action.

## stage1_repaired_200_fresh_local

- date: 2026-06
- question: Build a repaired local-scratch Stage 1 base with manifests and cached features.
- run_root: `/localdisk/bowen/singing_identity/runs/stage1_repaired_200_fresh_local`
- feature_root: `/localdisk/bowen/singing_identity/features/stage1_repaired_200_fresh_local`
- report: `/home/bowen/bowen_lab/projects/singing_identity/results/stage1_repaired_200_fresh_local_report.md`
- status: usable base run
- next_action: use as the base for controlled Track 1 / Track 2 probes.

## track1_prompt_mismatch_accounting_v2_2026-06-28

- date: 2026-06-28
- question: Account for speech-to-singing prompt/mode mismatch using low-dimensional covariates and residual prediction.
- run_root: `/localdisk/bowen/singing_identity/runs/track1_prompt_mismatch_accounting_v2_2026-06-28`
- reports: `/home/bowen/bowen_lab/projects/singing_identity/results/track1_prompt_mismatch_accounting_v2_2026-06-28_*.md`
- status: completed
- conclusion: useful accounting run, but interpretation depends on controls and representation/layer.
- next_action: compare against controls and avoid claiming clean identity residual without stronger evidence.

## track1_prompt_mismatch_accounting_v2_controls_2026-06-28

- date: 2026-06-28
- question: Run control variants for prompt/mode mismatch accounting.
- run_root: `/localdisk/bowen/singing_identity/runs/track1_prompt_mismatch_accounting_v2_controls_2026-06-28`
- reports: `/home/bowen/bowen_lab/projects/singing_identity/results/track1_prompt_mismatch_accounting_v2_controls_2026-06-28_*.md`
- status: completed
- conclusion: control evidence is required context for Track 1 claims.
- next_action: use alongside the v2 run when writing conclusions.

## track1_seedvc_prompt_baseline_30pairs

- date: 2026-06-28
- question: Does SeedVC respond differently to speech-prompt versus singing-prompt conditions?
- run_root: `/localdisk/bowen/singing_identity/runs/track1_seedvc_prompt_baseline_30pairs`
- reports:
  - `/home/bowen/bowen_lab/projects/singing_identity/results/seedvc_prompt_30pair_acoustic_objective_zh_2026-06-28.md`
  - `/home/bowen/bowen_lab/projects/singing_identity/results/seedvc_30pair_prompt_gap_posthoc_zh_2026-06-28.md`
- review_notebook: `/home/bowen/bowen_lab/projects/singing_identity/notebooks/track1_seedvc_prompt_review.py`
- status: completed, needs careful human-facing interpretation
- conclusion: prompt mode has measurable/audible effects, but the effect is not automatically an identity result.
- next_action: use human review and posthoc accounting before deeper SeedVC integration.

## track1_seedvc_component_ablation_12pairs

- date: 2026-06-28
- question: Which SeedVC components carry the prompt/mode effect?
- run_root: `/localdisk/bowen/singing_identity/runs/track1_seedvc_component_ablation_12pairs`
- report: `/home/bowen/bowen_lab/projects/singing_identity/results/seedvc_component_12pair_acoustic_objective_zh_2026-06-28.md`
- review_notebook: `/home/bowen/bowen_lab/projects/singing_identity/notebooks/track1_seedvc_component_ablation_review.py`
- status: completed
- conclusion: component differences exist, but identity-vs-technique/domain interpretation remains open.
- next_action: only promote claims after human listening and objective controls agree.

## track2_breathy_detection_probe_2026-06-28

- date: 2026-06-28
- question: Is breathy technique detectable in frozen representations?
- run_root: `/localdisk/bowen/singing_identity/runs/track2_breathy_detection_probe_2026-06-28`
- status: completed
- conclusion: breathy signal appears encoded, but this is not the same as a clean global steering direction.
- next_action: compare with local/subspace probes and wrong-technique controls.

## track2_breathy_subspace_probe_2026-06-28

- date: 2026-06-28
- question: Does breathy technique form a stable global or local subspace direction?
- run_root: `/localdisk/bowen/singing_identity/runs/track2_breathy_subspace_probe_2026-06-28`
- status: completed
- conclusion: current evidence weakens the simple clean-global-direction story.
- next_action: prefer local controls and narrower claims.

## hook_gate_probe_no_registry_2026-07-01

- date: 2026-07-01
- question: Verify that the Codex stop hook blocks when an experiment marker exists but the registry is not updated.
- script/config: `scripts/mark_experiment.py`
- run_root: `/localdisk/bowen/singing_identity/runs/hook_gate_probe_no_registry_2026-07-01`
- status: finished; hook probe only, no real experiment or run artifacts were produced.
- conclusion: stop hook correctly blocked the turn until this compact registry entry was added.
- next_action: no scientific follow-up; keep as an operational test of the registry gate.

## track1_residualized_retrieval_2026-07-08

- date: 2026-07-08
- question: Does same-person speech-to-singing retrieval survive after removing low-dimensional acoustic nuisance variables?
- script/config: `scripts/probing/run_speaker_retrieval.py --residualize-nuisance`
- run_root: `/localdisk/bowen/singing_identity/runs/track1_residualized_retrieval_2026-07-08`
- feature_root: `/localdisk/bowen/singing_identity/features/stage1_repaired_200_fresh_local`
- report: `/home/bowen/bowen_lab/projects/singing_identity/results/track1_residualized_retrieval_2026-07-08.md`
- status: completed
- conclusion: WavLM L12 residualized retrieval remains above threshold (S->G R@1 40%, 8x chance; G->S R@1 60%, 12x chance), but the acoustic baseline is also strong and needs ECAPA/JVS follow-up before a narrow identity claim.
- next_action: run independent ECAPA speaker embeddings, then expand retrieval and mode probes to JVS/JVS-MuSiC.

## track1_ecapa_gtsinger_2026-07-08

- date: 2026-07-08
- question: Does an independent speaker embedding model recover same-person identity across speech and singing on GTSinger?
- script/config: `scripts/data_prep/extract_ecapa_features.py`; `scripts/probing/run_speaker_retrieval.py`
- run_root: `/localdisk/bowen/singing_identity/runs/track1_ecapa_gtsinger_2026-07-08`
- feature_root: `/localdisk/bowen/singing_identity/features/stage1_repaired_200_fresh_local`
- report: `/home/bowen/bowen_lab/projects/singing_identity/results/track1_ecapa_gtsinger_2026-07-08.md`
- status: completed
- conclusion: ECAPA retrieves same-person speech/singing pairs strongly on 20 GTSinger speakers (raw and residualized S->G R@1 90%, 18x chance; G->S R@1 85%, 17x chance).
- next_action: use as independent evidence that identity crosses modes; compare with larger JVS/JVS-MuSiC results.

## jvs_music_retrieval_mapper_2026-07-08

- date: 2026-07-08
- question: Do residualized cross-mode identity retrieval and centroid residual mappers hold up on 100 JVS+JVS-MuSiC speakers?
- script/config: `scripts/run_jvs_music_priority3_2026_07_08.sh`; `scripts/probing/run_jvs_centroid_residual_predictor.py`
- run_root: `/localdisk/bowen/singing_identity/runs/jvs_music_retrieval_2026-07-08`
- feature_root: `/localdisk/bowen/singing_identity/features/jvs_music_2026-07-08`
- report: `/home/bowen/bowen_lab/projects/singing_identity/results/jvs_music_retrieval_mapper_2026-07-08.md`
- status: completed
- conclusion: Retrieval remains far above 1% chance after residualization (ECAPA 97%/95%, MERT L3 54%/46%, HuBERT L3 43%/34%, WavLM L3 42%/32%). Mapper baselines are now meaningful with 70/15/15 speakers; speech_embedding_ridge usually gives the best residual MSE reduction for SSL representations.
- next_action: use JVS results for the main mapper claim; inspect whether near-perfect residualized mode probes require additional domain controls before making a narrow identity-only claim.

## identity_residual_suite_minimal_2026-07-09

- date: 2026-07-09
- question: Why does residualization improve cross-mode retrieval, and is the next mapper step justified after synthetic and global/reliability gates?
- script/config: `scripts/probing/run_identity_residual_synthetic.py`; `scripts/probing/run_identity_residual_suite.py --models wavlm_l12 mert_l3 --seeds 13 17 19 --max-speech-per-speaker 20`
- run_root: `/localdisk/bowen/singing_identity/runs/identity_residual_suite_2026-07-09_minimal`
- report: `/home/bowen/bowen_lab/projects/singing_identity/results/identity_residual_suite_2026-07-09_minimal/final_experiment_report.md`
- synthetic_report: `/home/bowen/bowen_lab/projects/singing_identity/results/synthetic_tests/identity_residual_synthetic_2026-07-09/README_results.md`
- status: completed minimal strict suite
- conclusion: Synthetic gate passed. On JVS/JVS-MuSiC, random/shuffled controls did not reproduce gains, but mode-dummy-only explained much of the retrieval improvement. B0 showed a strong global residual fraction (WavLM L12 0.803-0.962; MERT L3 0.616-0.746). B1 is labeled underpowered because the local JVS singing side has one file per speaker, so no strong independent split-half residual claim is supported.
- next_action: expand Exp A to 20 seeds with bootstrap CIs and leave-one/duration-balanced controls; do not run strong B2 mapper claims until B1 has genuinely independent singing material or is explicitly framed as within-file reliability only.

## identity_residual_expA_expanded_2026-07-09

- date: 2026-07-09
- question: Does the residualized retrieval improvement survive 20 speaker-disjoint seeds, bootstrap CIs, nuisance-group controls, and duration-balanced sensitivity?
- script/config: `scripts/probing/run_identity_residual_suite.py --models wavlm_l12 mert_l3 --seeds 13 17 19 23 29 31 37 41 43 47 53 59 61 67 71 73 79 83 89 97 --max-speech-per-speaker 20 --bootstrap-samples 1000`; duration sensitivity rerun with `--duration-balanced`
- run_root: `/localdisk/bowen/singing_identity/runs/identity_residual_expA_expanded_2026-07-09`
- report: `/home/bowen/bowen_lab/projects/singing_identity/results/identity_residual_expA_expanded_2026-07-09/final_experiment_report.md`
- duration_report_dir: `/home/bowen/bowen_lab/projects/singing_identity/results/identity_residual_expA_duration_balanced_2026-07-09`
- status: completed expanded Exp A audit
- conclusion: Random/shuffled/random-low-rank controls still do not reproduce the gain. The strongest and broadest gains are from mode-dummy-only and duration-related residualization: WavLM L12 mode-dummy mean-preserving R@1 reaches 36.7%/59.3% vs raw 13.5%/19.5%; MERT L3 reaches 66.7%/78.0% vs raw 36.5%/36.0%. Single duration is the strongest measured nuisance group, and full-minus-duration loses much of the gain. B0 remains strongly global (default global fraction mean WavLM L12 0.845, MERT L3 0.654; duration-balanced WavLM L12 0.829, MERT L3 0.605).
- next_action: frame the JVS result as global mode/duration-axis removal, not pure acoustic/timbre residualization. Do not run a strong B2 mapper claim on current JVS alone; first seek genuinely independent singing material for B1, likely with GTSinger exploratory split-half reliability.

## identity_residual_full_gate_2026-07-09

- date: 2026-07-09
- question: After completing the planned gates, should the project proceed to individualized mapper or downstream SeedVC/human listening?
- script/config: `scripts/probing/run_identity_residual_suite.py` full JVS sweep over acoustic, ECAPA, HuBERT L6/L9/L12, MERT L3/L6/L9/L12, WavLM L6/L9/L12; GTSinger B1 with `--split-protocol gtsinger_10_0_10`; `scripts/probing/run_identity_residual_mapper_eval.py` for GTSinger B2
- run_roots:
  - `/localdisk/bowen/singing_identity/runs/identity_residual_jvs_fullsweep_2026-07-09`
  - `/localdisk/bowen/singing_identity/runs/identity_residual_gtsinger_b1_2026-07-09`
  - `/localdisk/bowen/singing_identity/runs/identity_residual_gtsinger_b2_2026-07-09`
- report: `/home/bowen/bowen_lab/projects/singing_identity/results/identity_residual_gate_report_2026-07-09.md`
- status: completed computational gate sequence; human decision point reached
- conclusion: JVS full sweep confirms a global mode/duration residual story across SSL layers; ECAPA is already near ceiling. GTSinger B1 passes exploratory split-half reliability for ECAPA, MERT L3, and WavLM L12. GTSinger B2 only gives stable mapper-over-global evidence for ECAPA; SSL mappers are weak/unstable and do not meet the strong threshold for individualized deployable residual mapping.
- next_action: stop automatic computation and decide with the human whether to frame the current result as global mode residual accounting, seek stronger independent singing material for individualized residuals, or run an optional SeedVC/global-residual listening sanity check.

## identity_residual_final_validation_2026-07-09

- date: 2026-07-09
- question: Is the global speech-to-singing residual claim paper-ready after direct global-adapter, mode-probe, nuisance-decodability, duration-proxy, restricted-gallery, and JVS B2 accounting checks?
- script/config:
  - `scripts/probing/run_identity_residual_final_validation.py --control-draws 50`
  - `scripts/probing/run_identity_residual_mapper_eval.py --dataset JVS_JVSMuSiC --models wavlm_l12 mert_l3 hubert_l6 --seeds 13 17 19 23 29 31 37 41 43 47 53 59 61 67 71 73 79 83 89 97 --split-protocol jvs_60_20_20 --control-draws 50 --shuffle-draws 20`
- run_roots:
  - `/localdisk/bowen/singing_identity/runs/identity_residual_final_validation_2026-07-09`
  - `/localdisk/bowen/singing_identity/runs/identity_residual_final_validation_2026-07-09/jvs_b2_mapper`
- report: `/home/bowen/bowen_lab/projects/singing_identity/results/identity_residual_final_validation_2026-07-09/final_validation_report.md`
- status: completed final validation suite
- conclusion: Direct `speech + mu_delta_train` reproduces the mode-dummy/global correction gains on JVS SSL models (WavLM L12 13.5% -> 36.2%, MERT L3 36.5% -> 66.0%, HuBERT L6 19.5% -> 60.5% S->G R@1), while wrong-sign and random-vector controls do not. Speaker-balanced, utterance-weighted, and duration-weighted global vectors are nearly collinear. Mode AUC drops from 1.0 to near chance after global correction while retrieval rises. Duration is best classified as a mode/segmentation proxy: duration-only mode AUC is 1.0 and within-mode duration residualization does not reproduce the gain. JVS within-gender restricted galleries preserve the improvement. Minimal JVS B2 shows positive residual-MSE reductions for SSL mappers but only weak retrieval evidence and no adequate independent JVS B1 reliability ceiling, so it does not support a strong individualized mapper claim.
- next_action: use the paper claim template centered on train-estimable global mode/duration-correlated residual accounting. Do not claim pitch removal, timbre isolation, causal duration, SeedVC explanation, or deployable individualized SSL residual mapping without stronger independent singing material and reliability evidence.

## identity_residual_same_text_and_statistical_correction_2026-07-10

- date: 2026-07-10
- question: Does the global mode residual survive an actual same-text, control-technique evaluation, and do corrected paper-readiness statistics preserve the earlier conclusion?
- script/config:
  - `scripts/probing/run_identity_residual_same_text.py --control-draws 50 --bootstrap-samples 10000`
  - `scripts/probing/run_identity_residual_final_validation.py --run-root /localdisk/bowen/singing_identity/runs/identity_residual_final_validation_2026-07-09 --results-dir results/identity_residual_final_validation_corrected_2026-07-10 --control-draws 50`
- run_roots:
  - `/localdisk/bowen/singing_identity/runs/identity_residual_same_text_2026-07-10`
  - `/localdisk/bowen/singing_identity/runs/identity_residual_final_validation_2026-07-09`
- report: `/home/bowen/bowen_lab/projects/singing_identity/results/identity_residual_paper_readiness_2026-07-10.md`
- same_text_results: `/home/bowen/bowen_lab/projects/singing_identity/results/identity_residual_same_text_2026-07-10`
- corrected_validation_results: `/home/bowen/bowen_lab/projects/singing_identity/results/identity_residual_final_validation_corrected_2026-07-10`
- status: completed; content-control gate passed
- data_audit: the GTSinger manifest `same_text_flag` is hard-coded and was not trusted. The experiment selected `technique=control` pairs only when independently stored speech/singing text matched after NFKC and boundary-marker normalization, yielding 1,953 pairs, 3,906 distinct utterances, and all 20 singers.
- conclusion: The same-text global adapter strongly improves held-out S->G R@1 for all primary SSL models over 50 speaker-disjoint 10/10 splits: WavLM L12 19.8% -> 73.8% (delta +54.0 pp, CI 51.0-57.0), MERT L3 27.4% -> 60.4% (+33.0 pp, CI 28.2-37.6), and HuBERT L6 19.8% -> 76.2% (+56.4 pp, CI 53.2-59.6). Wrong-sign decreases performance and same-norm random controls stay near raw. Train-only logistic mode AUC falls from 1.0 to near chance after correction. The corrected JVS rerun fixes restricted-gallery chance, nearest-impostor rows, train-only logistic probing, and paired split CIs without changing the earlier conclusion.
- next_action: stop new representation experiments and cross-dataset transfer; begin the paper package. Retain `global speech-to-singing mode residual` as a representation-level term, while keeping GTSinger small-sample/language confounding and JVS content mismatch explicit. Do not resume individualized mapper tuning or SeedVC as a main claim.

## identity_residual_protocol_robustness_2026-07-10

- date: 2026-07-10
- question: Does the train-estimated global displacement improve cross-mode verification, limited-reference retrieval, gallery robustness, and held-out speaker geometry beyond centroid R@1 alone?
- script/config: `scripts/probing/run_identity_residual_robustness.py` over JVS/JVS-MuSiC and clean GTSinger same-text control pairs; WavLM L12, MERT L3, HuBERT L6; train-only verification thresholds; 25 reference/gallery draws; 50 random direction draws.
- run_root: `/localdisk/bowen/singing_identity/runs/identity_residual_protocol_robustness_2026-07-10`
- report/results: `/home/bowen/bowen_lab/projects/singing_identity/results/identity_residual_protocol_robustness_2026-07-10`
- status: completed
- conclusion: Correct global correction reduces held-out EER and raises train-calibrated TMR@FMR=1% on all six dataset/model combinations, including with one speech reference and at 5/10/20 held-out galleries. Speaker-aggregate rank improvements are positive with bootstrap intervals excluding zero and sign-permutation p <= 0.0005. Mean cosine separation itself is not uniformly increased, so the supported claim is improved verification/ranking geometry, not monotone mean-margin expansion. Held-out residual alignment is strongly positive (mean cosine 0.78-0.92) and a shared mean explains 0.65-0.83 residual energy; centered effective rank remains multi-dimensional.
- next_action: use verification/TMR/EER and speaker-level uncertainty as the main robustness evidence. Keep residual spectrum descriptive and do not claim one-dimensionality or resume an individualized mapper.

## identity_residual_matched_frames_2026-07-10

- date: 2026-07-10
- question: Does the global correction survive an equal-frame-count, voiced-ratio-matched reevaluation on clean GTSinger same-text control pairs?
- script/config: `scripts/probing/run_identity_residual_matched_frames.py --fixed-frames 100 --crops-per-pair 5 --candidate-count 8 --control-draws 50`.
- run_root: `/localdisk/bowen/singing_identity/runs/identity_residual_matched_frames_2026-07-10`
- report/results: `/home/bowen/bowen_lab/projects/singing_identity/results/identity_residual_matched_frames_2026-07-10`
- status: completed
- conclusion: All 20 singers remain after frame eligibility. Correct global improves R@1 over raw by +47.6 pp WavLM, +32.4 pp MERT, and +40.4 pp HuBERT; paired EER deltas are -11.4, -5.0, and -12.9 pp respectively. The matched global direction remains aligned with the full-utterance global vector (WavLM splitwise cosine about 0.94-0.96), and symmetric correction moves mode AUC from about 1.0 to near chance.
- next_action: retain the term `matched-frame reevaluation`; do not describe it as duration removal or causal proof about segmentation.

## identity_residual_spectrum_2026-07-10

- date: 2026-07-10
- question: How much train-speaker residual energy is explained by the shared mean, and what centered residual dimensionality remains without fitting an oracle or individualized adapter?
- script/config: `scripts/probing/run_identity_residual_spectrum.py` using cached representations and the same train/test speaker splits.
- report/results: `/home/bowen/bowen_lab/projects/singing_identity/results/identity_residual_spectrum_2026-07-10`
- status: completed
- conclusion: Shared mean explained energy is 0.65-0.83 depending on dataset/model, while centered effective rank is approximately 6 on 20-singer GTSinger and 23-28 on JVS. The centered residual is therefore not one-dimensional; no low-rank adapter sweep was run because a non-oracle test-speech-only coefficient rule was not pre-specified and fitting one would revive the stopped mapper line.
- next_action: report the spectrum as structural description only and retain global-mean correction as the intervention actually validated on held-out identity metrics.

## identity_residual_metric_robustness_2026-07-13

- date: 2026-07-13
- question: Is the speech-to-singing identity residual a genuine origin-invariant relation, or is its functional benefit absorbed by centering and stronger train-only metric backends?
- script/config: `scripts/probing/run_identity_residual_metric_robustness.py`; JVS 60/20/20 over 20 fixed seeds and clean same-text GTSinger 10/10 over 50 fixed seeds; WavLM L12, HuBERT L6, and MERT L3; original/pooled/mode origins; cosine, unnormalized Euclidean, OAS-whitened cosine, OAS Mahalanobis, and shrinkage-LDA cosine.
- run_root: `/localdisk/bowen/singing_identity/runs/identity_residual_metric_robustness_2026-07-13`
- report/results: `/home/bowen/bowen_lab/projects/singing_identity/results/identity_residual_metric_robustness_2026-07-13`
- status: completed representation/metric gate; closest-work PDF audit remains blocked because the five-page Chowdhury et al. (2022) IEEE paper was inaccessible, so PDF-dependent protocol fields are explicitly `PDF_REQUIRED`.
- conclusion: R0 reproduces the three JVS baselines exactly. Held-out residual magnitude generalizes for all six dataset/model combinations (`E_test` 0.583-0.830; 95-100% of splits have `rho < 1`). Pooled centering explains only 9-24% of the original cosine gain, and unnormalized Euclidean retains large gains, so the result is not adequately described as a pure cosine-origin artifact. In contrast, train-only OAS whitening produces strong raw JVS R@1 (WavLM 72.8%, HuBERT 93.3%, MERT 89.5%) and makes the incremental translation effect approximately zero; regularized LDA also shows no stable increment. GTSinger utterance-disjoint analogy ordering and offset-consistency gates pass (AUC 0.754-0.869), supporting an analogy-like shared relation but not exact phonological arithmetic. The selected final position is `1. Origin-dominated functional effect (backend-absorbed arm)`, with the explicit caveat that the predeclared descriptive `origin-dominated` criterion itself is not met.
- next_action: use the backend-absorption result to narrow the paper claim and obtain the Chowdhury et al. (2022) PDF to close C1. Do not automatically begin synthesis, PLDA, nonlinear mapping, or professional-singer follow-up branches.

## identity_residual_paper_closure_2026-07-15

- date: 2026-07-15
- question: What mechanism lets train-only whitening absorb the speech–singing identity displacement, and does the recovery survive layer, single-utterance, and cross-song scope tests?
- script/config: `scripts/probing/run_identity_residual_paper_closure.py`; W1 diagonal versus OAS whitening, W2 displacement/covariance alignment and ABTT, G2x JVS layers 3/6/9/12, U1 25 seeded utterance draws per split, U2 clean-control GTSinger song-disjoint halves, and PDF-backed C1 protocol closure.
- run_root: `/localdisk/bowen/singing_identity/runs/identity_residual_paper_closure_2026-07-15`
- report/results: `/home/bowen/bowen_lab/projects/singing_identity/results/identity_residual_paper_closure_2026-07-15`
- status: completed paper-closure suite; all experimental gates pass, C1 closed from the user-supplied IEEE PDF, and N1 was not triggered by the retained narrow linear-mode wording.
- conclusion: Baseline OAS rows reproduce exactly. W1 is `correlations required` (JVS diagonal absorption 0.060-0.231). W2 is `concentrated + recoverable`: 99.90-99.99% of displacement energy lies in the top eight train covariance PCs, and k<=8 ABTT recovers 82.4-91.4% of the OAS R@1 gain with near-zero random-direction controls and no positive post-removal translation interval. Layer 3 is best for corrected JVS R@1 in all families. OAS single-utterance JVS R@1 remains 19.3%, 52.5%, and 63.9% versus 5% chance, and GTSinger OAS cross-song A/B results are stable. Chowdhury et al. (2022) explicitly use DeepCORAL/CORAL+ covariance alignment, providing convergent second-order compensation evidence under a non-comparable protocol.
- next_action: use framing `F2 — Dominant-direction masking` with single-utterance and within-corpus cross-song modifiers; write the paper and stop new representation experiments. Do not reopen mapper, SeedVC, synthesis, PLDA implementation, nonlinear probe search, or professional/amateur branches without a new human decision.

## identity_residual_paper_supplement_2026-07-16

- date: 2026-07-16
- question: Does full train-only OAS whitening improve at the G2x-optimal JVS layer 3, and does the previously positive WavLM ABTT split trend survive the predeclared speaker-cluster analysis?
- script/config: `scripts/probing/run_identity_residual_paper_supplement.py`; R0' exact closure reproduction, X1 OAS at WavLM/HuBERT L3 over the fixed 20 JVS splits, X2 10,000-sample unique-speaker bootstrap and sign-flip tests from saved closure score bundles with Holm correction, and X3 log-scale alignment figure.
- run_root: `/localdisk/bowen/singing_identity/runs/identity_residual_paper_supplement_2026-07-16`
- report/results: `/home/bowen/bowen_lab/projects/singing_identity/results/identity_residual_paper_supplement_2026-07-16`
- status: completed additive supplement; R0', X1, X2, and X3 pass, all saved W2 matrices reconstruct their logged R@1 within 1e-12, and the 2026-07-15 closure artifacts remain unchanged.
- conclusion: X1 is `X1-a` for WavLM: OAS-whitened L3 R@1 is 93.25% versus 72.75% at L12, with speaker-cluster D=+20.82 pp (95% CI +13.84 to +28.18 pp; Holm-significant). HuBERT L3 is 92.00% versus 93.25% at L6 and is `X1-b` (D=-0.60 pp; CI -4.73 to +3.50 pp). X2 is `X2-b`: WavLM k=1 has a +3.75 pp split mean (normal 95% interval +0.83 to +6.67 pp) but only +2.46 pp after unique-speaker aggregation (95% bootstrap CI -0.52 to +5.53 pp; Holm p=0.368), and neither HuBERT nor MERT is positive at its fixed k*. Framing remains `F2 — Dominant-direction masking` with U1-a/U2-a; add a WavLM OAS@L3 row to the main table, retain the HuBERT L6 and MERT L3 headline rows, and reaffirm the closure sentence that translation has no positive speaker-cluster interval after selected removal.
- next_action: write the paper from the 2026-07-15 closure plus this addendum, then stop. Do not reopen representation, mapper, SeedVC, synthesis, PLDA, nonlinear-probe, or professional/amateur experiments without a new human decision.
