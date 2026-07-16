# Experiment Implementation Task Context

**Objective:** Implement the two parallel research tracks (Timbre Mode Residual & Technique Probing) for a master's thesis on speech/singing vocal identity and style representation.

> [!IMPORTANT]
> The **primary source of truth** for experiment design is the Pro model's detailed design documents in `pro_suggestions/`. Read them in full before writing any code. The sections below summarize what is in each document and where to find it.

---

## 1. Primary Design Documents (MUST READ IN FULL)

These are the detailed, engineering-ready experiment designs written by the Pro model. They contain dataset split strategies, feature cache schemas, probe targets, metrics, negative controls, stop/go criteria, and 30-day plans.

### Core Experiment Design

| Document | What it contains | Key sections |
|----------|-----------------|--------------|
| [pro3_voice_representation_idea_review.md](file:///Users/bowen/research/project/pro_suggestions/pro3_voice_representation_idea_review.md) | **The most detailed document.** Full experiment design for Stage A (frozen-feature audit), Stage B (residual modeling), Stage C (downstream SVC intervention). Includes dataset split strategy, segment/example construction, feature cache schema, representation variants, temporal statistics, probe targets, metrics, negative controls, cheap acoustic shortcut baselines, F0/phone/duration controls, statistical analysis plan, stop/go criteria, compute/storage estimates, AI-coding task decomposition, and paper strategy. | Part 3: Stage A Experiment Design (line ~387), Part 4: Stage B and C Design, Part 5: Baseline and Code Feasibility, Part 6: AI-Coding and Engineering Checkpoints |
| [pro5_research_direction_ranking.md](file:///Users/bowen/research/project/pro_suggestions/pro5_research_direction_ranking.md) | Risk-adjusted ranking of all 6 candidate directions. Contains the "experiment to run this week," go/no-go criteria, 30-day plan with weekly deliverables, and the 12 verification questions with detailed answers. | "The single experiment to run this week" (line ~108), "30-day plan" (line ~253), "Answers to the 12 questions" (line ~379) |
| [pro7_ml_audio_research_workflow.md](file:///Users/bowen/research/project/pro_suggestions/pro7_ml_audio_research_workflow.md) | Full research methodology workflow using the speech/singing project as the running example. Covers: turning intuition into research question, literature scanning, novelty matrix, data/baseline choice, hypothesis writing, pilot design, confound detection, main experiment, ablations, downstream intervention, subjective evaluation, mixed-result interpretation, honest claims, and pivot decisions. | Each section has "concrete deliverable," "bad version," "good version," and "what AI can help with" |

### Supporting Context

| Document | What it contains |
|----------|-----------------|
| [pro1_speech_singing_field_map.md](file:///Users/bowen/research/project/pro_suggestions/pro1_speech_singing_field_map.md) | 2024-2026 SVC/speech-singing field landscape |
| [pro2_speech_singing_research_advisor.md](file:///Users/bowen/research/project/pro_suggestions/pro2_speech_singing_research_advisor.md) | Low-compute research direction exploration |
| [pro4_singing_skill_prediction_review.md](file:///Users/bowen/research/project/pro_suggestions/pro4_singing_skill_prediction_review.md) | Why singing skill/quality evaluation was deprioritized |
| [pro6_literature_citation_chain.md](file:///Users/bowen/research/project/pro_suggestions/pro6_literature_citation_chain.md) | GTSinger/Seed-VC/FACodec citation chain and each tool's role |
| [README.md (pro_suggestions)](file:///Users/bowen/research/project/pro_suggestions/README.md) | Index and summary of all 9 Pro documents |

---

## 2. Free Recall & Architecture Notes

These were created during conversation sessions to consolidate understanding:

| Document | What it contains |
|----------|-----------------|
| [project_blueprint_tracks.md](file:///Users/bowen/research/project/free_recall/project_blueprint_tracks.md) | High-level blueprint of Track 1 (Timbre/Mode Residual) and Track 2 (Technique Probing). Summarizes the validated two-track design. |
| [nuisance_control_methodologies.md](file:///Users/bowen/research/project/free_recall/nuisance_control_methodologies.md) | Math for nuisance regression/residualization, Gradient Reversal Layer (GRL) + LoRA, and the "Golden Rule of Conditioning" (FiLM vs. Cross-Attention). |
| [latent_space_architecture_free_recall.md](file:///Users/bowen/research/project/free_recall/latent_space_architecture_free_recall.md) | Architecture background linking latent space concepts to conditioning choices. |
| [pro1_free_recall.md](file:///Users/bowen/research/project/free_recall/pro1_free_recall.md) | Free recall notes from pro1 session |
| [pro3_free_recall.md](file:///Users/bowen/research/project/free_recall/pro3_free_recall.md) | Free recall notes from pro3 session |

---

## 3. Reference Code (Existing Implementation)

Review these files to understand the current codebase:

| File | What it contains |
|------|-----------------|
| [run_timbre_shift_mapper.py](file:///Users/bowen/research/project/arti6_linearvc_demo/run_timbre_shift_mapper.py) | `MicroMapper` class (Residual MLP) and training logic for mapping $T_{\text{speech}} \to T_{\text{singing}}$ |
| [run_seedvc_svc_demo.py](file:///Users/bowen/research/project/arti6_linearvc_demo/run_seedvc_svc_demo.py) | Inference logic for Seed-VC conversion. Shows where speaker embeddings are injected into the frozen decoder. |

---

## 4. Two-Track Task Breakdown

### Track 1: Timbre Shift (Mode Residual Mapper)

**Goal:** Map the vocal identity shift when a person transitions from speaking to singing, and use it to improve zero-shot cross-domain SVC.

**Detailed design:** See [pro3](file:///Users/bowen/research/project/pro_suggestions/pro3_voice_representation_idea_review.md) Part 3 (Stage A) and Part 4 (Stages B/C). Also see [pro5](file:///Users/bowen/research/project/pro_suggestions/pro5_research_direction_ranking.md) "The single experiment to run this week" section.

**Implementation steps:**
1. **Data Preparation:** Build the manifest described in pro3 (Part 3, "Feature cache schema"). Use JVS + JVS-MuSiC first (50-100 same people, speech + singing crops). Store paired embeddings with metadata (F0, energy, duration, etc.).
2. **Feature Extraction:** Extract frozen embeddings from WavLM Base+ (layers 3/6/9/12), ContentVec, ECAPA-TDNN speaker embeddings, and the Seed-VC speaker encoder. Follow the `.npz` schema in pro3.
3. **Probing (Stage A):** Run mode probe (speech vs singing), speaker leakage probe (cross-mode), same-person retrieval (Recall@1/5), and residual control (regress out F0/loudness/duration). See pro3 Part 3 "Probe targets" and "Negative controls."
4. **MicroMapper Training (Stage B):** Train the residual MLP from [run_timbre_shift_mapper.py](file:///Users/bowen/research/project/arti6_linearvc_demo/run_timbre_shift_mapper.py). Loss = cosine similarity / MSE between predicted and ground truth singing embedding.
5. **Downstream Injection (Stage C):** Connect trained MicroMapper output to [run_seedvc_svc_demo.py](file:///Users/bowen/research/project/arti6_linearvc_demo/run_seedvc_svc_demo.py). Test: speech-only prompt → predicted singing embedding → Seed-VC generation. Compare against speech-prompt baseline and oracle singing-prompt.

### Track 2: Technique Probing & Reconstruction

**Goal:** Discover linear directions for singing techniques (vibrato, falsetto, etc.) in frozen SSL models, and use them to control singing style synthesis.

**Detailed design:** See [pro5](file:///Users/bowen/research/project/pro_suggestions/pro5_research_direction_ranking.md) "Backup direction" section. Also see [pro3](file:///Users/bowen/research/project/pro_suggestions/pro3_voice_representation_idea_review.md) Part 3 "Probe targets" (technique classification).

**Implementation steps:**
1. **Analogy Probing (Stage A):** Load GTSinger phone-level technique labels. Extract frame-level features for the **same phoneme** under **different techniques** (Normal vs. Vibrato, etc.). Calculate $\Delta Z = Z_{\text{vibrato}} - Z_{\text{normal}}$ and bootstrap to measure direction stability. Compare across WavLM, HuBERT, ContentVec, MERT.
2. **Cross-Model Comparison:** Rank models/layers by how linearly and stably they encode technique directions. Hypothesis: music-trained model (MERT) may encode techniques better than speech-only SSL.
3. **Latent Steering (Stage B):** Apply $Z_{\text{new}} = Z + \lambda \cdot \Delta Z$ and feed to a frozen decoder. Evaluate whether technique character changes while identity is preserved.
4. **Optional LoRA + GRL (Stage C):** If latent steering alone is insufficient, implement LoRA adapters with GRL to strip technique from content branch. See [nuisance_control_methodologies.md](file:///Users/bowen/research/project/free_recall/nuisance_control_methodologies.md) for the math.

---

## 5. Design Guidelines

* **Keep Decoders Frozen:** Do not end-to-end train massive synthesizer models. Train lightweight adapters only.
* **Control for Confounds:** Always regress out F0, energy, duration, voiced ratio before claiming a "timbre residual" or "technique direction." This is non-negotiable (see pro3 negative controls and pro5 go/no-go criteria).
* **Modularity:** Decouple data extraction, feature caching, probe training, and inference.
* **Logging:** Print tensor shapes explicitly during development. Follow the `.npz` feature cache schema from pro3.
* **Splits:** Use held-out-singer splits for publishability. Within-language subsets are the most trustworthy (see pro3 Split 3).

---

## 6. Go/No-Go Criteria (from pro5)

**Continue if ALL THREE hold:**
1. A nontrivial layer-wise tradeoff: some layers preserve cross-mode identity while others encode speech/singing mode.
2. The effect survives controls: after regressing out F0, energy, voiced ratio, and duration, mode classification AUC ≥ 0.75 and cross-mode Recall@1 ≥ 10% with 100 speakers.
3. Seed-VC cares about prompt mode: speech-prompt vs singing-prompt outputs show consistent differences in target-singing similarity.

**Pivot to Track 2 (Technique) if:**
- The speech/singing residual vanishes after F0/loudness/duration controls.
- Cross-mode speaker retrieval is near chance for every representation.
- Seed-VC output is insensitive to prompt mode.
