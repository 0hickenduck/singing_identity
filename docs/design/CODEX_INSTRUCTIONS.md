# Codex Instructions: Singing Representation Research

> [!IMPORTANT]
> **START HERE.** This is the entry point for Codex. Read this file completely before writing any code.

---

## 0. What This Project Is

This is a master's-level research project studying **same-person speech-to-singing vocal identity shift** and **singing technique representation** in frozen SSL/codec models.

There are two parallel research tracks:
- **Track 1 (Timbre):** Estimate the "mode residual" $\Delta = T_{\text{singing}} - T_{\text{speech}}$ and use it to improve speech-reference zero-shot SVC.
- **Track 2 (Technique):** Find linear directions for singing techniques (vibrato, falsetto, etc.) in frozen SSL representations and steer downstream synthesis.

**Compute constraint:** We have a few GPUs. We do NOT train large base models from scratch. Everything is frozen encoders + small adapters/probes.

---

## 1. Required Reading (in order)

Read ALL of these before writing code. They contain the exact engineering specifications.

### Cluster Workflow

| Priority | Document | What to read | Why |
|----------|----------|-------------|-----|
| 🔴 MUST | [LAB_SERVER_WORKFLOW.md](../LAB_SERVER_WORKFLOW.md) | Full document | Defines where code, manuscript material, datasets, caches, feature stores, run roots, notebooks, and cleanup belong on the lab cluster. |
| 🔴 MUST | [HUMAN_REVIEW_PROTOCOL.md](HUMAN_REVIEW_PROTOCOL.md) | Full document | Defines how listening/visual checkpoints should be packaged as Marimo notebooks with explicit human decisions. |

### Primary Design Documents

| Priority | Document | What to read | Why |
|----------|----------|-------------|-----|
| 🔴 MUST | [pro3_voice_representation_idea_review.md](../pro_suggestions/pro3_voice_representation_idea_review.md) | Part 3 (Stage A: frozen feature audit), Part 4 (Stage B/C: residual modeling + intervention), Part 6 (AI-coding engineering checkpoints, task decomposition, file/artifact contracts, manifest schema, feature cache schema, smoke tests, unit tests) | This is the engineering-ready full experiment spec. The manifest schema, `.npz` schema, probe targets, negative controls, and stop/go criteria are all here. |
| 🔴 MUST | [pro5_research_direction_ranking.md](../pro_suggestions/pro5_research_direction_ranking.md) | "The single experiment to run this week" (≈line 108), "30-day plan" (≈line 253), "Go/no-go criteria" (≈line 184) | Contains the exact first experiment, weekly deliverables, and the 3 conditions that must hold before continuing. |
| 🟠 SHOULD | [pro7_ml_audio_research_workflow.md](../pro_suggestions/pro7_ml_audio_research_workflow.md) | Full document | Research methodology template; defines what AI can help with vs. what the human must verify. |
| 🟡 CONTEXT | [nuisance_control_methodologies.md](../free_recall/nuisance_control_methodologies.md) | Full document | Math for residualization, GRL+LoRA, FiLM vs. Cross-Attention. |
| 🟡 CONTEXT | [project_blueprint_tracks.md](../free_recall/project_blueprint_tracks.md) | Full document | High-level validated design of both tracks. |

### Reference Code (existing implementation to build on)

| File | What it does |
|------|-------------|
| [scripts/intervention/run_micro_mapper.py](../scripts/intervention/run_micro_mapper.py) | Current lightweight speech-to-singing residual mapper over cached Stage 1 features. Use this for Track 1 Stage B/C baselines. |
| [scripts/probing/run_technique_directions.py](../scripts/probing/run_technique_directions.py) | Current Track 2 technique-direction estimator with bootstrap, wrong-technique, and shuffled-direction controls. |
| [scripts/data_prep/build_gtsinger_manifests.py](../scripts/data_prep/build_gtsinger_manifests.py) | Current GTSinger manifest builder for utterance, pair, phone-example, and phoneme-pair manifests. |

---

## 2. File Structure for This Research

All new code goes under `research/singing_representation/`. Do NOT scatter files in the repo root.

```
research/singing_representation/
├── design/
│   └── CODEX_INSTRUCTIONS.md        # This file
├── experiments/
│   ├── track1_timbre/               # Track 1 outputs and configs
│   └── track2_technique/            # Track 2 outputs and configs
├── scripts/
│   ├── data_prep/                   # Data extraction, manifest building
│   │   ├── build_manifest.py        # [TO CREATE] Build utterances.parquet
│   │   └── extract_features.py      # [TO CREATE] Extract frozen SSL features
│   ├── probing/                     # Probe training and evaluation
│   │   ├── run_mode_probe.py        # [TO CREATE] Speech vs singing classifier
│   │   ├── run_speaker_retrieval.py # [TO CREATE] Cross-mode Recall@1/5
│   │   └── run_residual_control.py  # [TO CREATE] F0/duration/energy regress-out
│   └── intervention/                # Downstream injection
│       ├── run_micro_mapper.py      # [TO CREATE] MicroMapper training wrapper
│       └── run_seedvc_inject.py     # [TO CREATE] Seed-VC injection with mapped embedding
├── notebooks/                       # Exploratory marimo notebooks
├── results/                         # Compact reports, tables, and small final figures
└── README.md
```

---

## 3. Track 1: Timbre — What to Implement First

Follow **pro3 Part 3** and **pro5 "experiment to run this week"** exactly.

### Step 1: Build the Manifest (`scripts/data_prep/build_manifest.py`)

**Dataset:** JVS + JVS-MuSiC first (50–100 same-person speech+singing pairs). Add GTSinger paired speech if setup is easy.

**Output:** `experiments/track1_timbre/manifests/utterances.parquet` with these columns (from pro3 Part 3 "Feature cache schema"):
```
utt_id, singer_id, language, mode (speech/singing), technique, song_id,
wav_path, start_sec, end_sec, duration_sec, sample_rate,
text, phone_seq, alignment_path, paired_utt_id,
f0_mean_hz, f0_std_hz, f0_min_hz, f0_max_hz, f0_voiced_pct,
energy_mean, energy_std, rms_db, snr_db, split_group_key
```

**Split invariant:** All utterances from the same song/phrase stay in the same split. Singer-disjoint splits for generalization tests.

### Step 2: Extract Frozen Features (`scripts/data_prep/extract_features.py`)

**Models to extract from (in priority order):**
1. WavLM Base+ — layers 3, 6, 9, 12 (segment-level mean + std)
2. ContentVec — final layer
3. ECAPA-TDNN (SpeechBrain pretrained) — utterance-level speaker embedding
4. Seed-VC speaker encoder — if easy to access

**Output path pattern** (from pro3 Part 3 "Feature cache schema"):
```
/localdisk/bowen/singing_identity/features/<run_name>/{extractor}/{checkpoint_hash}/{layer}/{utt_id}.npz
```
Each `.npz` must contain: `x: float32 [T, D]`, `times_sec: float32 [T]`, `voiced_mask: bool [T]`, `phone_id: int [T]`, `f0_hz: float32 [T]`, `energy: float32 [T]`

### Step 3: Run Probes (`scripts/probing/`)

From pro5 "Probes" section and pro3 Part 3 "Probe targets":
1. **Mode probe:** speech vs singing, speaker-disjoint split. Report AUC.
2. **Speaker leakage probe:** train on speech ID → test on singing; train on singing ID → test on speech. Report Recall@1/5.
3. **Residual control:** regress out mean F0, F0 std, energy, duration, voiced ratio. Re-run probes. If AUC collapses, pivot.
4. **Tiny Seed-VC smoke test:** 5–10 speakers, speech-prompt vs singing-prompt. Compare ECAPA similarity.

**Go/no-go after Step 3 (from pro5):**
- Continue if: mode AUC ≥ 0.75 after controls, cross-mode Recall@1 ≥ 10% with 100 speakers, Seed-VC responds to prompt mode.
- Pivot to Track 2 if any of these fail.

### Step 4: MicroMapper Training (`scripts/intervention/run_micro_mapper.py`)

Use [scripts/intervention/run_micro_mapper.py](../scripts/intervention/run_micro_mapper.py) to train a lightweight mapper on paired $(T_{\text{speech}}, T_{\text{singing}})$ embeddings from the manifest.

---

## 4. Track 2: Technique — What to Implement

From pro5 "Backup direction" and pro3 Part 3 "Probe targets (technique)".

### Step 1: GTSinger Phone-Level Feature Extraction

Use the existing Stage 1 GTSinger manifests from `/localdisk/bowen/singing_identity/runs/stage1_repaired_200_fresh_local/manifests` as the base.
Extract frame-level features for **same phoneme, different technique** (e.g., Normal vs. Vibrato).
Use GTSinger's six technique labels: mixed voice, falsetto, breathy, pharyngeal, vibrato, glissando.

### Step 2: Analogy Probing

For each (phoneme, technique_A, technique_B) triple:
- Compute $\Delta Z = \text{mean}(Z_{\text{tech\_A}}) - \text{mean}(Z_{\text{tech\_B}})$
- Bootstrap: subsample 80% of examples, repeat 1000x, measure variance of $\Delta Z$ direction
- Report: which model/layer has the most stable direction (lowest angular variance)?

### Step 3: Latent Steering

$Z_{\text{new}} = Z + \lambda \cdot \hat{\Delta Z}$, feed to frozen decoder. Evaluate if technique character changes while identity is preserved.

---

## 5. Absolute Engineering Rules

1. **Never train base models.** Only train probes (linear/MLP), the `MicroMapper` MLP, or LoRA adapters.
2. **Always log tensor shapes** during development: `print(f"z shape: {z.shape}")`.
3. **Negative controls are mandatory.** Always run label permutation and F0/duration-only baselines before claiming a "timbre residual."
4. **Split invariant:** No singer/speaker can appear in both train and test for generalization experiments.
5. **Feature cache first.** Cache all frozen features to disk before training probes. Do not re-extract on every run.
6. **Use the exact `.npz` schema from pro3.** Do not invent a different format.
7. **Keep heavy I/O off NFS.** Active datasets, caches, features, predictions, checkpoints, and long logs must be under verified `/localdisk/bowen`.
8. **Use scripts for final claims.** Marimo notebooks are for exploration and figures; final results must be reproducible from scripts/configs.

---

## 6. Deliverables (Week 1 target from pro5)

- [ ] `utterances.parquet` manifest for JVS/JVS-MuSiC
- [ ] Cached WavLM + ECAPA features for 50–100 speakers
- [ ] Layer-wise mode AUC + cross-mode Recall@1 figure (layer on x-axis)
- [ ] 10 Seed-VC audio pairs: same source, same target, speech-prompt vs singing-prompt
- [ ] One-page decision memo: continue Track 1, narrow, or pivot to Track 2
