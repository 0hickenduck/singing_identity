# Benchmark Structure & Provenance Migration Record

**Date:** 2026-09-06 JST  
**Author:** Bowen / Antigravity  
**Branch:** `reorg/conventional-layout`  
**Git Archive Tag:** `pre-conventional-reorg` (`f1bde5ce3896ae88ef43b804e31c97902f1b246b`)  
**Goal:** Reorganize `singing_identity` into a conventional benchmark-style ML research repository with machine-readable experiment provenance, retiring `legacy/` from the working tree into Git history.

---

## 1. Classification and Retirement of `legacy/` Tree

Git history, pinned at tag `pre-conventional-reorg`, serves as the permanent archive for obsolete historical implementations. All active methods, metrics, probing tools, and runners have been migrated into the canonical Python package `singing_identity` and single canonical CLI entrypoints under `scripts/`.

### 1.1 Migrated Scientific Implementations

The following active components were extracted from legacy scripts and integrated into `src/singing_identity/`:

| Old Script / Component | Canonical Package Destination | Last Relevant Tag | Role |
|---|---|---|---|
| `scripts/probing/run_identity_residual_metric_robustness.py` (`fit_oas_dual`, `OASDual`) | `singing_identity.methods.identity_residual` | `pre-conventional-reorg` | Exact dual OAS covariance estimation |
| `scripts/probing/run_identity_residual_paper_closure.py` (`remove_subspace`, `fit_diagonal_whitener`) | `singing_identity.methods.identity_residual` | `pre-conventional-reorg` | ABTT subspace removal & diagonal whitening |
| `scripts/probing/run_identity_residual_final_validation.py` (`mode_probe_centroid`, `restricted_retrieval`, `add_global_rows`, `paired_split_delta_summaries`) | `singing_identity.methods.identity_residual` | `pre-conventional-reorg` | Mode probe & restricted gallery evaluation |
| `scripts/probing/run_identity_residual_paper_supplement.py` (`speaker_interval`) | `singing_identity.methods.identity_residual` | `pre-conventional-reorg` | Speaker-level bootstrap confidence interval |
| `scripts/probing/run_identity_residual_robustness.py` (`eer`, `roc_det_points`, `train_threshold_at_fmr`, `verification_metrics`) | `singing_identity.evaluation.metrics` | `pre-conventional-reorg` | Verification calibration and metrics |
| `scripts/probing/run_identity_residual_same_text.py` (`normalize_lexical_text`, `select_clean_control_pairs`) | `singing_identity.data.manifests` | `pre-conventional-reorg` | Text normalization & pair filtering |
| `scripts/probing/run_identity_residual_synthetic.py` | `singing_identity.probing.synthetic_gate` | `pre-conventional-reorg` | Synthetic invariance test gate |
| `scripts/probing/run_mode_probe.py` | `singing_identity.probing.mode_probe` | `pre-conventional-reorg` | Cross-validated mode classification probe |
| `scripts/probing/run_speaker_retrieval.py` | `singing_identity.probing.speaker_retrieval` | `pre-conventional-reorg` | Speaker identity retrieval probe |
| `scripts/probing/run_residual_control.py` | `singing_identity.probing.residual_control` | `pre-conventional-reorg` | Residual-controlled acoustic probe |
| `scripts/probing/run_technique_directions.py` | `singing_identity.probing.technique_directions` | `pre-conventional-reorg` | Vocal technique direction probe |
| `scripts/probing/run_technique_subspace_probe.py` | `singing_identity.probing.technique_subspace` | `pre-conventional-reorg` | Multi-technique subspace probe |
| `scripts/probing/run_breathy_detection_probe.py` | `singing_identity.probing.breathy_probe` | `pre-conventional-reorg` | Phonation/breathy technique probe |
| `scripts/intervention/run_micro_mapper.py` | `singing_identity.intervention.micro_mapper` | `pre-conventional-reorg` | Residual mapping intervention |
| `scripts/intervention/run_seedvc_inject.py` | `singing_identity.intervention.seedvc_inject` | `pre-conventional-reorg` | Latent injection probe |
| `scripts/intervention/run_latent_steering.py` | `singing_identity.intervention.latent_steering` | `pre-conventional-reorg` | Representation steering probe |
| `scripts/data_prep/make_synthetic_experiment.py` | `singing_identity.data.synthetic` | `pre-conventional-reorg` | Synthetic dataset generator |
| `scripts/data_prep/validate_feature_cache.py` | `singing_identity.data.features` | `pre-conventional-reorg` | Feature cache validator |
| `scripts/data_prep/extract_acoustic_features.py` | `singing_identity.data.extractors` | `pre-conventional-reorg` | Acoustic baseline feature extractor |
| `scripts/run_stage1_overnight.py` (`Stage1Runner`) | `singing_identity.runner` | `pre-conventional-reorg` | Stage 1 multi-job benchmark runner |

### 1.2 Obsolete Historical Implementations Deleted from Working Tree

All of the following files were classified as obsolete historical implementations, ad-hoc wrappers, or one-off data repair scripts. They have been deleted from the active tree and are recoverable on demand from tag `pre-conventional-reorg`:

- **Ad-hoc Shell & Cluster Scripts:**
  - `legacy/check_lab_environment.sh`
  - `legacy/migrate_stage1_features_slow.sh`
  - `legacy/resume_stage1_repaired_200_on_localdisk.sh`
  - `legacy/resume_stage1_repaired_200_on_work.sh`
  - `legacy/run_fresh_stage1_on_localdisk.sh`
  - `legacy/run_fresh_stage1_on_work.sh`
  - `legacy/run_jvs_music_priority3_2026_07_08.sh`
  - `legacy/setup_public_tools.sh`
  - `legacy/sync-large-data.sh`
  - `legacy/sync_push.sh`
  - `legacy/valkyrie-status.sh`
  - `legacy/review/open_lab_tunnel.ps1`
  - `legacy/review/start_review_servers.sh`
  - `legacy/review/stop_review_servers.sh`
- **Data Repair Scripts (Retired):**
  - `legacy/data_prep/repair_gtsinger_missing_wavs.py`
  - `legacy/data_prep/repair_gtsinger_and_rerun_stage1.py`
  - `legacy/data_prep/summarize_repair_progress.py`
  - `legacy/data_prep/summarize_stage1_supervisor.py`
  - `legacy/run_targeted_repair_then_stage1.py`
  - (Unit tests `tests/test_repair_queue.py` and `tests/test_repair_wrapper.py` retired)
- **One-off Probing & Analysis Scripts (Archived in Git):**
  - `legacy/probing/run_identity_residual_suite.py`
  - `legacy/probing/run_identity_residual_spectrum.py`
  - `legacy/probing/run_identity_residual_mapper_eval.py`
  - `legacy/probing/run_jvs_centroid_residual_predictor.py`
  - `legacy/probing/run_prompt_mismatch_accounting.py`
  - `legacy/probing/run_prompt_mismatch_accounting_v2.py`
  - `legacy/audit_stage1.py`
  - `legacy/audit_stage1_results.py`
  - `legacy/build_demo.py`
  - `legacy/mark_experiment.py`
  - `legacy/prepare_raw_wavs.py`
  - `legacy/extract_features.py`
  - `legacy/data_prep/build_manifest.py`
  - `legacy/data_prep/build_gtsinger_manifests.py`
  - `legacy/data_prep/build_jvs_music_manifests.py`
  - `legacy/data_prep/extract_ecapa_features.py`
  - `legacy/intervention/run_seedvc_prompt_intervention.py`
  - `legacy/intervention/run_seedvc_prompt_baseline.py`
  - `legacy/intervention/run_seedvc_component_ablation.py`
  - `legacy/intervention/evaluate_seedvc_acoustic_objective.py`
  - `legacy/intervention/evaluate_seedvc_multicondition_outputs.py`
  - `legacy/intervention/evaluate_seedvc_prompt_outputs.py`
  - `legacy/intervention/extract_seedvc_prompt_latents.py`
  - `legacy/intervention/audit_seedvc_prompt_latents.py`
  - `legacy/intervention/analyze_seedvc_prompt_gap_posthoc.py`
  - `legacy/intervention/build_listening_review.py`
  - `legacy/intervention/build_multicondition_listening_review.py`
  - `legacy/intervention/summarize_seedvc_prompt_outputs.py`

---

## 2. Canonical Entry Points

Under `scripts/`, exactly one canonical script is maintained per operation:

1. **Prepare manifests & feature caches:** `scripts/data/prepare_manifests.py`
2. **Run experiments:** `scripts/run/run_stage1.py`
3. **Evaluate verification metrics:** `scripts/evaluate/evaluate_verification.py`
4. **Summarize benchmark results:** `scripts/summarize/summarize_results.py`
5. **Lab setup:** `scripts/wt_setup.sh` (required by lab worktree conventions)

---

## 3. Provenance Core Implementation

### 3.1 Provenance Invariant
> Every reported benchmark result must be traceable to one concrete run, and every run must identify the exact code, resolved configuration, data/features, execution command, and relevant environment that produced it.

$$\text{RESULT} \longrightarrow \text{RUN ID} \longrightarrow \begin{cases} \text{Canonical Resolved Configuration } (\texttt{config.json}) \\ \text{Git State (40-char SHA + dirty flag + } \texttt{artifacts/uncommitted.patch}) \\ \text{Data \& Feature Fingerprint (manifest SHA-256 + cache root)} \\ \text{Execution Context (command, seed, host, Python)} \end{cases}$$

### 3.2 Standard Run Directory Layout
```text
runs/<experiment>/<approach>/<run_id>/
├── config.json               # Single canonical snapshot of resolved hyperparameters
├── metadata.json             # Git SHA, branch, dirty flag, manifest SHA-256, seed, host
├── command.txt              # Exact CLI command string executed
├── metrics.json              # Standardized metric payload
├── logs/                     # Console stdout/stderr logs
└── artifacts/                # Run outputs
    └── uncommitted.patch     # Unified git diff when run on a dirty working tree
```

### 3.3 Synthetic & Feature Provenance Fixes
- **Synthetic Manifest Hash:** Recorded from the actual consumed synthetic manifest generated during the run, computing its actual SHA-256 (not defaulting to `gtsinger_utterances.jsonl`).
- **Explicit Feature Provenance:** `feature_manifest_sha256` is `null` (not `""`) when not pre-computed; `feature_provenance` is explicitly set to `"not_applicable"` or `"cached"`.
- **Config Unification:** Duplicate `resolved_config.json` is removed; a single canonical `config.json` holds the complete configuration.
- **Tracked Provenance Independence:** `results/summary_provenance.json` embeds the resolved configuration dictionary directly, keeping results scientifically interpretable even when `runs/` scratch is purged.

---

## 4. Test Suite Verification

All 51 test cases in `tests/` pass with zero regressions:
- 43 algorithmic, metric, feature, and runner unit tests (all testing `singing_identity` directly).
- 8 provenance and reproducibility unit tests in `tests/test_provenance.py`.
- Zero dependencies on `legacy/`.
