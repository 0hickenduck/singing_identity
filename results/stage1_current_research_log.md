# Stage 1 Current Research Log

Generated: 2026-06-18

## Current State

The robust Stage 1 infrastructure exists and has completed one stable real-data run:

- Completed run: `results/stage1_reusemanifest_wavlm_tmpfeatures_run`
- Completed report: `results/stage1_reusemanifest_wavlm_tmpfeatures_report.md`
- Data actually used: local GTSinger subset, 378 utterances, 18 speakers
- Model actually completed: WavLM Base+
- Earlier acoustic-only run also completed: `results/stage1_overnight_robust_partial_report.md`

A new tmux job is currently running:

- Session: `singing_stage1_200`
- First step: targeted repair of missing wavs for the 200-row-per-singer GTSinger Stage 1 slice
- Second step: rerun Stage 1 with `acoustic,wavlm,hubert,mert`
- New report target: `results/stage1_repaired_200_allmodels_report.md`
- New feature root: `/home/bowen/bowen_lab/projects/singing_identity_stage1_features/repaired_200_allmodels`
- Current repair supervisor: retry-capable wrapper with up to 8 repair cycles and 30-minute cooldowns after rate limits
- Current observed repair progress after supervisor restart: original missing queue fell from 5601 to 4700, so 901 repaired files were preserved
- Current observed repair rate after restart: about 628 files/hour, with an estimated 7.4 hours remaining for the repair stage at that moment
- CUDA environment prepared at `/tmp/singing_identity_cuda_venv`; verified `torch 2.8.0+cu128`, CUDA 12.8, and NVIDIA RTX A6000.
- Active supervisor was restarted with `/tmp/singing_identity_cuda_venv/bin/python` and `--device cuda`; after restart the remaining queue was 4580, preserving about 120 additional repaired files from the previous repair cycle.
- The feature root was moved out of `/tmp` because `/tmp` only had about 5.5 GB free after CUDA setup; the all-model feature cache would likely exceed that.
- The repair runner now uses `--resume-existing-queue --skip-final-scan`, so restarts load the persistent missing-file queue instead of rescanning all metadata and wav paths over NFS.
- Current patched repair queue after the storage/resume restart: 4539 remaining files, with downloads continuing from the prior breakpoint.
- Latest checked repair progress: 100/4539 queue positions, with 92 new downloads and 8 already-existing files; recent rate was about 482 files/hour and the repair ETA was about 9.2 hours.
- A separate CUDA all-model smoke probe completed successfully:
  - Run: `results/stage1_cuda_allmodels_smoke_probe_run`
  - Report: `results/stage1_cuda_allmodels_smoke_probe_report.md`
  - Jobs: 18 successful, 0 failed, 0 skipped
  - Models validated: acoustic baseline, WavLM Base+, HuBERT base, MERT v1 95M
  - This validates extractor/model dependency health before the repaired-data run reaches Stage 1 extraction.
- Latest checked repair progress after the smoke probe and runner patch: 3857/4539 queue positions, with 3849 new downloads and 8 already-existing files; recent rate was about 880 files/hour and the repair ETA was about 0.8 hours.
- The live repair process was started before the queue-compaction patch, so `missing_queue.json` still contains the original 4539 paths; this is expected for the active process, and `status.json` index/total is the authoritative live progress.
- The repair saw one transient `IncompleteRead` download error, but retried the same file successfully on attempt 2 and the wav exists at the expected local path.
- 2026-06-18 22:03 JST status: repair is still healthy at 3930/4539 queue positions, with 3922 newly downloaded files, 8 already existing files, 1 transient failed attempt, and no active rate limit. Recent rate is about 869 files/hour, with about 0.7 hours remaining before Stage 1 should start.
- Full unit tests in the CUDA environment now pass again: `/tmp/singing_identity_cuda_venv/bin/python -m unittest` ran 19 tests successfully after the latest Track 2 probe/report fixes.
- 2026-06-18 22:05 JST status: repair is still healthy at 3965/4539 queue positions, with 3957 newly downloaded files, 8 already existing files, and no active rate limit. Stage 1 has not started yet.
- 2026-06-18 22:07 JST status: repair is still healthy at 3989/4539 queue positions, with 3981 newly downloaded files, 8 already existing files, and no active rate limit. Stage 1 has not started yet.
- 2026-06-18 22:08 JST status: repair is still healthy at 4005/4539 queue positions, with 3997 newly downloaded files, 8 already existing files, and no active rate limit. Stage 1 has not started yet.
- 2026-06-18 22:45 JST status: targeted GTSinger repair completed for the active 200-row-per-singer queue. Final repair counters were 4531 downloaded, 8 already existing, and 1 transient failed attempt that had already recovered.
- 2026-06-18 22:48 JST status: Stage 1 manifest preparation completed. The repaired manifest has 6016 utterances, 20 speakers, 9 languages, 4000 speech/singing pairs, 168172 phone examples, and 81917 technique pairs. The manifest report has `skipped: {}`, so missing wavs no longer block this run.
- 2026-06-18 22:49 JST status: smoke extraction and smoke validation succeeded for acoustic, WavLM Base+, HuBERT base, and MERT v1 95M.
- 2026-06-18 22:53 JST status: full acoustic extraction is running. It has written 678 acoustic feature files out of 6016 utterances, with recent feature timestamps advancing; current NFS wait states look like I/O pressure, not a stalled job.
- 2026-06-19 17:40 JST status: full feature extraction succeeded for acoustic, WavLM Base+, HuBERT base, and MERT v1 95M. Track 1 is running. One optional control job, `track1_mode_shuffled_wavlm_l6`, timed out after 7200 seconds; the runner continued. Current active job is WavLM layer 9 speaker retrieval.
- 2026-06-20 12:11 JST status: the `/home` run is considered stopped/incomplete. It has no final report, 41 successful jobs, 2 timed-out WavLM mode-probe jobs (`track1_mode_raw_wavlm_l12`, `track1_mode_shuffled_wavlm_l6`), and one stale `running` status (`track1_mode_controlled_wavlm_l12`). Recovery should move run state and feature cache to `/work/bowen`, then resume without `--force`.
- 2026-06-20 partial audit: `results/stage1_repaired_200_partial_audit.md` summarizes the incomplete `/home` run. Completed Track 1 results show acoustic R@1/R@5 = 0.350/0.600; WavLM completed layers have stronger residual mapper cosines than acoustic but worse speaker retrieval so far. Track 2 has not run yet, so no final verdict is possible.

## What We Wanted To Test

Track 1 asks whether speaker identity survives from speech to singing, and whether the speech-to-singing residual is global, personalized, or mostly acoustic.

Track 2 asks whether singing technique directions are stable enough across phones and singers before trying latent steering.

## What We Obtained So Far

### Track 1: Timbre / Speech-to-Singing

WavLM controlled mode AUC was 1.000 for layers 3, 6, 9, and 12. This does not mean the experiment is finished. It means WavLM can very strongly separate speech from singing in the current subset even after the nuisance residualization used by the probe.

Important interpretation: AUC below 0.5 is not automatically bad. AUC far from 0.5 means strong separability; the sign tells orientation. The report now separates signed AUC from separability AUC with `max(AUC, 1 - AUC)`.

Identity retrieval is the more important Stage 2 gate. Best current WavLM result is layer 12:

| Layer | Controlled signed AUC | Separability AUC | R@1 | R@5 | Mapper cosine | Global residual cosine |
|---:|---:|---:|---:|---:|---:|---:|
| 3 | 1.000 | 1.000 | 0.222 | 0.500 | 0.715 | 0.781 |
| 6 | 1.000 | 1.000 | 0.056 | 0.611 | 0.799 | 0.804 |
| 9 | 1.000 | 1.000 | 0.056 | 0.722 | 0.779 | 0.784 |
| 12 | 1.000 | 1.000 | 0.333 | 0.778 | 0.690 | 0.676 |

This is not a negative result. Cross-mode identity retrieval is above chance in at least some layers, especially WavLM layer 12. Global residual is also strong and should be treated as a candidate method, not as a failure.

### Track 2: Technique Directions

Current WavLM technique evidence is concentrated in Chinese breathy groups.

| Layer | Reliable groups | Passed groups |
|---:|---:|---:|
| 3 | 78 | 7 |
| 6 | 78 | 9 |
| 9 | 78 | 5 |
| 12 | 78 | 8 |

Promising examples include Chinese breathy directions for phones like `d_zh`, `j_zh`, `x_zh`, and `z_zh`. This is promising, but not yet broad enough to claim general technique transfer.

## Known Flaws

1. Data is still incomplete. The current completed WavLM report skipped 3762 missing wavs. A repair job is now running to reduce this.
2. The data is imbalanced. The usable subset is dominated by Chinese and breathy technique examples.
3. Shuffled-label checks are not perfectly near chance in the current WavLM run. This suggests remaining split imbalance, acoustic confound, or probe/data leakage that must be checked before strong claims.
4. Only WavLM completed in the strongest current SSL run. HuBERT and MERT comparisons are being attempted in the new run.
5. The original project venv hangs on `import torch` on this machine, so the running job uses `/tmp/singing_identity_cuda_venv/bin/python`.
6. The controlled probe residualizes the listed nuisance variables using train-only fitting, but this does not remove every possible dataset confound.
7. Generated experiment artifacts were not ignored by git, and a large `git add` over feature/result files caused extra NFS contention.
8. The first technique analogy implementation used the group mean direction including the query item itself. That made `analogy_top1` optimistic and was not a strict train-direction transfer test.

## Bug Fixes Made

- Avoided full Hugging Face `snapshot_download`, which caused `429 Too Many Requests`.
- Added targeted missing-file repair with single-file downloads, request pacing, exponential backoff, rate-limit stop status, and a persistent missing queue.
- Added a repair supervisor loop so future `429 Too Many Requests` events wait and resume instead of immediately proceeding with partial repair.
- Disabled Xet for Hugging Face downloads where possible.
- Added fast manifest building that avoids scanning full wav payloads.
- Added runner job timeouts and optional model failure handling.
- Added existing-manifest reuse and clean `/tmp` feature roots to avoid NFS stalls.
- Moved the active all-model feature root from `/tmp` to an outside-repo project sibling with larger available storage.
- Started the active long job inside tmux so it persists if Codex disconnects.
- Added `scripts/data_prep/summarize_stage1_supervisor.py` for quick progress/status checks.
- Extended the supervisor summary with recent download rate and ETA.
- Added resume-from-existing-queue repair mode to avoid expensive NFS metadata scans on every restart.
- Hardened the supervisor summary against transient partial `status.json` reads while the repair process is writing.
- Added periodic queue compaction for future repair restarts, so completed files can be dropped from the resume queue at checkpoints.
- Added `.gitignore` rules for generated feature caches, Stage 1 run directories, repair logs, and generated track result directories while leaving top-level Markdown reports trackable.
- Terminated a stale `git add` process that was trying to stage generated experiment outputs and competing with the repair job on NFS.
- Added a unit test for the repair queue resume helpers.
- Re-ran `/tmp/singing_identity_cuda_venv/bin/python -m unittest`: 16 tests passed after the resume, monitor, ignore-rule, and queue-test changes.
- Ran a real 10-utterance CUDA smoke extraction/validation for every enabled extractor; all enabled extractors passed.
- Fixed Stage 1 global failure signaling: `scripts/run_stage1_overnight.py` now exits nonzero when a required/global failure is caught, so the repair wrapper cannot mark a failed Stage 1 as complete.
- Added a regression test confirming global Stage 1 failures return nonzero while still writing a report.
- Re-ran `/tmp/singing_identity_cuda_venv/bin/python -m unittest`: 17 tests passed after the failure-signaling fix.
- Extended `scripts/data_prep/summarize_stage1_supervisor.py` to summarize Stage 1 job status counts and list failed/timeout jobs once the run moves past repair.
- Re-ran `/tmp/singing_identity_cuda_venv/bin/python -m unittest`: 17 tests passed after the Stage 1 monitor improvement.
- Updated `scripts/run_targeted_repair_then_stage1.py` so future wrapper runs preserve Stage 1 `complete_with_failures` instead of flattening all zero-returncode Stage 1 runs to `complete`.
- Added wrapper status regression tests.
- Re-ran `/tmp/singing_identity_cuda_venv/bin/python -m unittest`: 19 tests passed after the wrapper status fix.
- Fixed the Track 2 technique analogy test so `analogy_top1` now uses a leave-one-out train direction for each query instead of a direction estimated from the query itself.
- Added `shuffled_direction_top1` to the Track 2 technique metrics and made the Stage 1 report count passed groups only when analogy beats both wrong-technique and shuffled-direction baselines.
- Re-ran `/tmp/singing_identity_cuda_venv/bin/python -m unittest`: 19 tests passed after the Track 2 analogy/baseline fix.
- Expanded the final Stage 1 report table for Track 1 so it directly shows raw signed AUC, raw separability AUC, controlled signed AUC, controlled separability AUC, shuffled-label separability AUC, retrieval, mapper cosine, and global residual cosine.
- Expanded the Track 2 report table so best groups include phone, technique, pair count, speaker count, language count, bootstrap angle, analogy score, wrong-technique baseline, and shuffled-direction baseline.
- Added per-job `summary.md` files to the Stage 1 runner so overnight failures can be inspected without first reading full raw logs.
- Re-ran `/tmp/singing_identity_cuda_venv/bin/python -m unittest tests.test_stage1_runner`: 4 tests passed after the report-summary changes.
- Re-ran the full suite after all report-summary changes: `/tmp/singing_identity_cuda_venv/bin/python -m unittest` ran 19 tests successfully.
- Added regression assertions that optional failed and timed-out jobs write readable `summary.md` files with status, missing output, and timeout details.
- Re-ran `/tmp/singing_identity_cuda_venv/bin/python -m unittest tests.test_stage1_runner`: 4 tests passed after the job-summary regression assertions.
- Fixed a runtime issue for future shuffled-label probes: `scripts/probing/run_mode_probe.py` now uses 400 optimizer steps instead of 1200 when `--shuffle-labels` is set and no explicit `--steps` is provided. This keeps shuffled controls useful as near-chance sanity checks without spending full classifier training time.
- Added `optimization_steps` and `learning_rate` to mode-probe metrics for auditability.
- Re-ran `/tmp/singing_identity_cuda_venv/bin/python -m unittest tests.test_research_utils tests.test_synthetic_pipeline`: 12 tests passed after the shuffled-probe speed fix.
- Added repository-level cluster rules in `AGENTS.md`: code/docs may live in `/home`, but data, run roots, feature caches, model caches, checkpoints, prediction dumps, and overnight outputs must live under `/work/bowen`; long jobs must run in `tmux`/`nohup`; non-interactive conda must be initialized explicitly or bypassed with an absolute Python path.
- Added Stage 1 preflight protection: real non-smoke runs now refuse heavy `/home/bowen` paths for `run_root`, `feature_root`, `local_cache_root`, and fresh `gtsinger_root` unless `--allow-home-heavy-io` is explicitly passed.
- Changed repair wrappers so Stage 1 resume no longer uses `--force` by default. Successful jobs can now be skipped and timeout/missing jobs rerun.
- Changed repair-wrapper defaults to `/work/bowen/singing_identity/...` and `/work/bowen/.cache/singing_identity`.
- Replaced the slow numpy logistic optimizer with sklearn `LogisticRegression` while preserving the same score interface; synthetic mode-probe separability returned to 1.000 after this fix.
- Added `scripts/resume_stage1_repaired_200_on_work.sh`, a tmux-safe recovery script that copies existing run state and feature cache to `/work`, then resumes Stage 1 without `--force`.
- Added `scripts/audit_stage1_results.py` to summarize job failures and completed Track 1/Track 2 metrics for either the incomplete `/home` run or the resumed `/work` run.
- Verified `bash -n scripts/resume_stage1_repaired_200_on_work.sh` and `/tmp/singing_identity_cuda_venv/bin/python -m unittest`: 19 tests passed after the cluster/path/resume/logistic fixes.
- Ran `/tmp/singing_identity_cuda_venv/bin/python scripts/audit_stage1_results.py --run-root results/stage1_repaired_200_allmodels_run --out results/stage1_repaired_200_partial_audit.md`; the audit completed successfully.
- Verified unit tests with `/tmp/singing_identity_ssl_venv/bin/python -m unittest`: 15 tests passed.
- Confirmed the active repaired-data run passed all SSL model smoke extraction and feature-cache validation jobs before full extraction started.
- Created and verified a CUDA-capable local venv, then reran `/tmp/singing_identity_cuda_venv/bin/python -m unittest`: 15 tests passed.

## Human Check List

- Current repaired-data run checkpoint as of 2026-06-19 08:05 JST:
  - Data repair completed and manifest build has `skipped: {}`.
  - Full feature extraction succeeded for acoustic, WavLM Base+, HuBERT base, and MERT v1 95M.
  - Acoustic Track 1 finished: raw separability AUC 0.912, controlled separability AUC 0.912, shuffled separability AUC 0.507, singing-to-speech R@1 0.350, R@5 0.600, ridge mapper cosine 0.617, global residual cosine 0.629.
  - WavLM layer 3 raw mode probe finished: separability AUC 0.999. WavLM layer 3 controlled probe is currently marked running and has not written metrics yet.
  - Interpretation: mode AUC is not the Stage 2 gate because acoustic-only mode separability is already high. The decisive Track 1 checks are cross-mode identity retrieval and whether learned/global residual baselines beat acoustic-only controls.

- Check whether the new repair job finishes or stops with `status: rate_limited`.
- If it rate-limits, the supervisor should now wait and retry automatically. If repeated cycles fail, add an HF token at `/tmp/singing_identity_hf_token`.
- After the new Stage 1 report appears, check whether shuffled-label AUC is closer to chance.
- Check whether HuBERT and MERT complete or fail at smoke extraction.
- For Track 1, focus on retrieval and residual metrics, not raw mode AUC alone.
- For Track 2, continue only technique groups that pass both bootstrap angle and analogy retrieval against wrong-technique controls.
- For Track 2, inspect the new `shuffled_direction_top1` column/group metric; a technique group should not be treated as promising if analogy beats wrong-technique but not shuffled-direction.

## Current Decision

Do not desert the project yet.

Track 1 has enough evidence to continue, but only if the repaired-data run confirms identity retrieval above chance and residual baselines remain stronger than acoustic-only controls.

Track 2 should continue narrowly for breathy technique directions first. It should not proceed to broad latent steering until the repaired/all-model run shows stable directions beyond one imbalanced subset, or until a cleaner dataset such as VocalSet is added.

## Codex Entry Context (2026-06-23)

- **Old Reference Deleted**: The old baseline directory  has been deleted to prevent confusion.
- **Stage 1 Complete**: The data repair wrapper completed successfully.  reflects .
- **Features Cached**: All heavy features (acoustic, WavLM Base+, HuBERT, MERT v1) are safely cached on .
- **Metrics Available**: The full Stage 1 metrics are available in .
- **Next Steps**: Please review the Stage 1 metrics report and prepare to move forward with the next phases of the  project: Track 1 MicroMapper (Stage B/C) and Track 2 Latent Steering, as defined in .


## Codex Entry Context (2026-06-23)

- **Old Reference Deleted**: The old baseline directory arti6_linearvc has been deleted to prevent confusion.
- **Stage 1 Complete**: The data repair wrapper completed successfully. status.json reflects complete.
- **Features Cached**: All heavy features (acoustic, WavLM Base+, HuBERT, MERT v1) are safely cached on /localdisk/bowen/singing_identity/features.
- **Metrics Available**: The full Stage 1 metrics are available in results/stage1_repaired_200_fresh_local_report.md.
- **Next Steps**: Please review the Stage 1 metrics report and prepare to move forward with the next phases of the singing_identity project: Track 1 MicroMapper (Stage B/C) and Track 2 Latent Steering, as defined in design/CODEX_INSTRUCTIONS.md.

## Codex Entry Context (2026-06-24)

- **Seed-VC Checkout**: Cloned the public Seed-VC repository to `/localdisk/bowen/singing_identity/external/seed-vc` so third-party model code and Hugging Face caches stay off the source tree.
- **Track 1 Vector Materialization**: Added `scripts/intervention/run_seedvc_inject.py` for MicroMapper-conditioned WavLM vector batches. It writes baseline speech-prompt, mapped singing-prompt, and optional oracle singing-prompt condition vectors. This is ready for an internal frozen-decoder adapter, but the public Seed-VC CLI does not accept arbitrary WavLM vectors.
- **Track 1 Black-Box Seed-VC Baseline**: Added `scripts/intervention/run_seedvc_prompt_baseline.py`, `scripts/intervention/summarize_seedvc_prompt_outputs.py`, and `scripts/intervention/evaluate_seedvc_prompt_outputs.py`.
- **Balanced Prompt-Mode Batch**: Generated 10 heldout target pairs and 20 Seed-VC SVC WAVs under `/localdisk/bowen/singing_identity/runs/track1_seedvc_prompt_baseline_10pairs_rr` with `diffusion_steps=10`, `fp16=true`, and two conditions per pair: target speech prompt vs target singing prompt.
- **Audio Sanity**: `audio_summary.json` reports 20 valid WAVs, 10 pairs, no missing outputs, mean duration 8.40 seconds, mean RMS -18.61 dB.
- **Automatic Proxy Result**: Resemblyzer triage found singing-prompt outputs moved toward target-singing references in 10/10 pairs and away from target-speech references in 10/10 pairs. Mean deltas: target singing `+0.1103`, target speech `-0.0750`, source singing `+0.0358`.
- **Human Review Surface**: Added Marimo notebook `notebooks/track1_seedvc_prompt_review.py`; open it with `uv run --with marimo marimo edit notebooks/track1_seedvc_prompt_review.py`. The notebook exposes source audio, target references, generated speech-prompt/singing-prompt outputs, and the exact `continue` / `rerun` / `stop` decision requested from the human reviewer.
- **Human Review Protocol**: Added `design/HUMAN_REVIEW_PROTOCOL.md`; future subjective listening or visual checks should produce a Marimo notebook first, plus a clear decision request.
- **Track 2 Materialization**: Added `scripts/intervention/run_latent_steering.py` and generated MERT L3 breathy three-condition vectors under `/localdisk/bowen/singing_identity/runs/track2_latent_steering_mert_l3_breathy`. Audio synthesis for latent steering still requires a decoder adapter that accepts the latent condition vectors.
- **Validation**: Full test suite passes with `TMPDIR=/tmp PYTHONPYCACHEPREFIX=/tmp/singing_identity_pycache /tmp/singing_identity_cuda_venv/bin/python -m unittest` (`19` tests).

## Codex Entry Context (2026-06-24 Seed-VC Native Latent Audit)

- **Research framing correction**: Track 1 should be framed as decomposing Seed-VC target conditioning, not as a generic "predict singing voice from speech" task. Working decomposition: target singing condition is approximately identity core plus a general singing-domain residual plus technique residual. Track 1 estimates the general singing-domain correction; Track 2 estimates technique corrections.
- **Native Prompt Latent Extraction**: Added `scripts/intervention/extract_seedvc_prompt_latents.py`. It loads Seed-VC once and extracts prompt-side native latents for paired target speech/singing prompts: CAMPPlus `style`, `semantic_mean/std`, `prompt_mean/std`, `mel_mean/std`, F0 summary, audio summary, and a pooled vector.
- **Native Latent Audit**: Added `scripts/intervention/audit_seedvc_prompt_latents.py`. It trains a global residual and a ridge residual on train speakers, evaluates heldout speakers, and reports speech-to-singing cosine, mapped-to-singing cosine, delta-to-true-delta cosine, delta norm, and nuisance correlations.
- **All200 Run**: Extracted 200 paired Seed-VC prompt latents, 400 conditions total, under `/localdisk/bowen/singing_identity/runs/seedvc_native_latents_all200`. Audit outputs are under `/localdisk/bowen/singing_identity/runs/seedvc_native_latents_all200/audit`.
- **Heldout Split**: The all200 audit uses 160 train pairs and 40 test pairs across 20 speakers. Heldout speakers are `KO-Tenor-1`, `RU-Alto-1`, `ZH-Alto-1`, and `ZH-Tenor-1`.
- **Key Audit Results**:
  - `style`: speech-to-singing cosine `0.451`; ridge delta-to-true-delta cosine `0.519`. Seed-VC speaker/style embeddings strongly separate speech and singing, but direct style mapping is risky.
  - `semantic_stats`: speech-to-singing cosine `0.944`; ridge mapped-to-singing cosine `0.976`; ridge delta-to-true-delta cosine `0.705`. This is the best first intervention target because the residual is both present and predictable.
  - `prompt_stats`: speech-to-singing cosine `0.944`; ridge mapped-to-singing cosine `0.959`; ridge delta-to-true-delta cosine `0.575`. This is the secondary intervention target.
  - `mel_stats`: speech-to-singing cosine `0.993`; ridge delta-to-true-delta cosine `0.476`. It is too close at baseline and likely less informative as a causal intervention target.
  - `pooled`: ridge delta-to-true-delta cosine `0.866`, but it mixes style, semantic, mel, F0, and audio summaries, so it is useful for diagnosis rather than clean injection.
- **Nuisance Caveat**: `semantic_stats` residual norm correlates with duration delta (`0.608`), F0 voiced-pct delta (`0.426`), and RMS delta (`0.406`). The result is promising but not pure timbre; the next intervention must include oracle and mapped conditions and should not claim identity-only correction.
- **Human/Visual Review Surface**: Added Marimo notebook `notebooks/seedvc_native_latent_audit.py`. Run from `/localdisk/bowen/singing_identity` with `uv run --with marimo marimo run --headless --host 127.0.0.1 --port 2718 notebooks/seedvc_native_latent_audit.py`, then connect through SSH tunneling from the local machine.
- **Next Experiment**: Implement Seed-VC decoder intervention using `semantic_stats` first, with three listening conditions: baseline speech prompt, mapped speech prompt, and oracle singing prompt. If semantic injection is technically unstable, repeat the same design with `prompt_stats`.

## Codex Entry Context (2026-06-24 Seed-VC Semantic Intervention)

- **Intervention Script**: Added `scripts/intervention/run_seedvc_prompt_intervention.py`. It trains a ridge residual from all200 paired native latents, then synthesizes three Seed-VC listening conditions: `baseline_speech_prompt`, `mapped_semantic_stats` or `mapped_prompt_stats`, and `oracle_singing_prompt`.
- **Intervention Method**: For `semantic_stats`, the script predicts target singing semantic mean/std from the target speech semantic stats, then affine-shifts the target speech semantic prompt sequence before Seed-VC length regulation. It intentionally leaves `mel2` and CAMPPlus `style2` as target speech for the mapped condition, so this first causal check isolates the semantic prompt sequence only.
- **Technical Caveat**: Because Seed-VC also conditions CFM on prompt mel and CAMPPlus style, semantic-only intervention may be weak even when the residual direction is mathematically valid. If listening shows weak movement, the next technical step is a combined prompt-condition/style or prompt-condition/mel ablation, not an immediate rejection of the Track 1 framing.
- **Small Audio Batch**: Generated a 4-pair heldout semantic intervention batch under `/localdisk/bowen/singing_identity/runs/track1_seedvc_semantic_intervention_4pairs`. The batch has 12 valid WAVs, 4 pairs, mean duration `7.54` seconds, mean RMS `-18.61` dB, and peak max `1.0`.
- **Human Listening Surface**: Added Marimo notebook `notebooks/track1_seedvc_semantic_intervention_review.py`. It presents source singing, target speech reference, target singing reference, and the three generated conditions for direct listening.
- **Human Decision Needed Next**: When ready, listen to the semantic intervention notebook and report one of `semantic works`, `semantic weak`, or `semantic harmful`, plus pair-level notes. Do not evaluate this by filesystem browsing; use the notebook review surface.

## Human Feedback (2026-06-24 Semantic Intervention)

- Human listening decision: `semantic weak`.
- Interpretation: semantic-only affine shifting produced valid audio, but did not create an obvious or reliable audible movement from speech-prompt output toward singing-prompt/oracle output. The target timbre itself was also hard to judge in the small batch.
- Consequence: Do not scale semantic-only mapping yet. The next experiment should be an oracle component ablation inside Seed-VC target conditioning: swap singing-vs-speech `prompt_condition`, `mel2`, and CAMPPlus `style2` independently to identify which native component actually carries the audible singing-prompt advantage.

## Codex Entry Context (2026-06-24 Seed-VC Component Ablation)

- **Purpose**: Directly answer what constitutes the audible speech-to-singing residual inside Seed-VC target conditioning before fitting another predictive mapper.
- **Ablation Script**: Added `scripts/intervention/run_seedvc_component_ablation.py`.
- **Native Components Tested**:
  - `prompt_seq`: length-regulated Seed-VC prompt condition from Whisper/semantic features plus F0.
  - `mel`: target prompt mel context passed into CFM.
  - `style`: CAMPPlus target speaker/style embedding.
- **Conditions Generated**: `baseline_all_speech`, `oracle_all_singing`, `singing_prompt_seq_only`, `singing_mel_only`, `singing_style_only`, `singing_prompt_seq_mel`, `singing_prompt_seq_style`, and `singing_mel_style`.
- **Audio Batch**: Generated 4 heldout pairs, 32 WAVs total, under `/localdisk/bowen/singing_identity/runs/track1_seedvc_component_ablation_4pairs`. Mean duration is `7.54` seconds, mean RMS is `-18.57` dB, and peak max is `1.0`.
- **Human Review Surface**: Added Marimo notebook `notebooks/track1_seedvc_component_ablation_review.py`. It presents references, baseline/oracle, the three single-component swaps, and optional two-component swaps.
- **Human Decision Needed Next**: Listen to the component ablation notebook and report which single component is most audible: `prompt_seq`, `mel`, `style`, or `none`; also report whether any two-component combo approaches `oracle_all_singing`.
- **Next Branching Rule**: If `style` dominates, map/steer CAMPPlus style. If `mel` dominates, treat the residual as mostly prompt acoustic context. If `prompt_seq` dominates, improve the semantic/F0 path. If no single component dominates but a combo does, build a combined adapter.

## Human Feedback (2026-06-25 Component Ablation)

- Human listening decision: component differences are generally small. `oracle_all_singing` is still the best condition, but among the three single-component swaps, `style` seems to have the largest effect. The evidence is weak because only one of four pairs made the style effect clearly audible.
- Interpretation: Seed-VC's audible speech-vs-singing prompt advantage does not decompose cleanly into one obvious native component in this small batch. There may be a distributed effect across `style`, `mel`, and `prompt_seq`, or the effect size may be too small/noisy for this Seed-VC setup and dataset slice.
- Consequence: Do not invest immediately in a large predictive mapper for one Seed-VC native component. Treat the current Seed-VC intervention path as an exploratory diagnostic, not the main thesis result yet.
- Recommended next pivot: return to the higher-level scientific question and formalize a residual taxonomy with measurable probes: acoustic/prosody residual, speaker/style residual, and technique residual. Use Seed-VC only as a listening sanity check after a residual factor shows strong quantitative evidence.
