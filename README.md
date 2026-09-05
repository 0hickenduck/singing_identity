# 🎙️ Singing Identity & Technique Representation Benchmark

**Research question:** Where does singer identity survive when transitioning between speech and singing, and how can we leverage that to disentangle technique and timbre in zero-shot singing voice conversion?

This repository contains the benchmark code, declarative experiment configurations, manifests, and evaluation protocols for the **two-track singing representation research project**.

---

## 🧭 Where to Look

| Intent | Target Location / Command |
|---|---|
| **Compare approaches on a track** | [`results/summary.csv`](file:///home/bowen/bowen_lab/projects/singing_identity/results/summary.csv) (and companion [`results/summary_provenance.json`](file:///home/bowen/bowen_lab/projects/singing_identity/results/summary_provenance.json)) |
| **Inspect one specific run** | `runs/<experiment>/<approach>/<run_id>/` (contains `config.json`, `metadata.json`, `metrics.json`, `artifacts/`) |
| **Change how a model/approach works** | [`src/singing_identity/methods/`](file:///home/bowen/bowen_lab/projects/singing_identity/src/singing_identity/methods/) |
| **Change evaluation / metrics** | [`src/singing_identity/evaluation/`](file:///home/bowen/bowen_lab/projects/singing_identity/src/singing_identity/evaluation/) |
| **Add or change a dataset** | [`configs/datasets/`](file:///home/bowen/bowen_lab/projects/singing_identity/configs/datasets/) + [`src/singing_identity/data/`](file:///home/bowen/bowen_lab/projects/singing_identity/src/singing_identity/data/) |
| **Launch an experiment** | `uv run python scripts/run/run_stage1.py --synthetic --smoke-only --models acoustic` |
| **Regenerate feature cache** | `uv run python scripts/data/prepare_manifests.py --extract-features --approach configs/approaches/acoustic.json` (writes to `/localdisk/bowen/singing_identity/features/`) |

---

## 📁 Repository Organization & Directory Ownership

| Directory | Contents & Purpose | Written By |
|---|---|---|
| [`configs/`](file:///home/bowen/bowen_lab/projects/singing_identity/configs/) | Declarative JSON specifications for datasets, approaches, and experiments | **Humans** |
| [`data/`](file:///home/bowen/bowen_lab/projects/singing_identity/data/) | Canonical processed benchmark manifests (`data/processed/*.jsonl`). Raw audio is never stored here. | **Humans / Scripts** |
| [`src/singing_identity/`](file:///home/bowen/bowen_lab/projects/singing_identity/src/singing_identity/) | Core reusable scientific package (algorithms, metrics, data pipeline, provenance) | **Humans** |
| [`scripts/`](file:///home/bowen/bowen_lab/projects/singing_identity/scripts/) | One canonical CLI script per operation (`data/`, `run/`, `evaluate/`, `summarize/`, plus `wt_setup.sh`) | **Humans** |
| [`runs/`](file:///home/bowen/bowen_lab/projects/singing_identity/runs/) | Execution run folders (`config.json`, `metadata.json`, `command.txt`, `metrics.json`, `artifacts/`) | **Scripts** |
| [`results/`](file:///home/bowen/bowen_lab/projects/singing_identity/results/) | Aggregated benchmark tables (`summary.csv`) and milestone research reports | **Scripts & Humans** |
| [`docs/`](file:///home/bowen/bowen_lab/projects/singing_identity/docs/) | Research design specifications, notes (`blueprints/`, `advisor_reviews/`), and server workflows | **Humans** |
| [`tests/`](file:///home/bowen/bowen_lab/projects/singing_identity/tests/) | Automated unit tests for algorithms, manifests, runners, and provenance verification | **Humans** |

```text
singing_identity/
├── README.md                          # Benchmark introduction, dataset provenance, quickstart
├── pyproject.toml                     # Package dependencies & uv build definition
├── configs/
│   ├── datasets/                      # Manifest and raw data definitions (gtsinger, jvs_music)
│   ├── approaches/                    # Method and representation extractors (wavlm, mert, etc.)
│   └── experiments/                   # Experiment definitions (track1_timbre, track2_technique)
├── data/
│   ├── README.md                      # Documents dataset locations, mounts, and download links
│   └── processed/                     # Benchmark manifest files (lightweight jsonl)
│       ├── gtsinger_utterances.jsonl
│       ├── gtsinger_pairs.jsonl
│       ├── gtsinger_phone_examples.jsonl
│       └── gtsinger_phoneme_pairs.jsonl
├── src/singing_identity/              # Core reusable scientific package
│   ├── data/                          # Manifest reading/writing, synthetic generator & cache validation
│   ├── methods/                       # OAS dual fitting, residual projection, steering
│   ├── evaluation/                    # Verification metrics, ROC-DET, cosine scoring
│   ├── probing/                       # Probing pipelines (mode, retrieval, technique)
│   ├── intervention/                  # Intervention models & latent steering
│   ├── runner.py                      # Stage 1 benchmark execution engine
│   └── utils/                         # Research utilities, matrix math & provenance
├── scripts/
│   ├── data/                          # CLI entry points for manifest building & feature extraction
│   ├── run/                           # CLI entry points for experiment execution
│   ├── evaluate/                      # CLI entry points for metric evaluation
│   ├── summarize/                     # CLI entry points for report auditing
│   └── wt_setup.sh                    # Lab worktree environment setup
├── runs/                              # Run metadata, configs, and metric logs
│   └── README.md                      # Run standard and offloading guidelines
├── results/                           # Aggregated benchmark-level results
│   ├── README.md                      # Overview of benchmark findings
│   ├── index.md                       # Canonical historical run index
│   ├── summary.csv                    # Consolidated benchmark comparison table
│   └── summary_provenance.json        # Machine-readable companion provenance
├── docs/                              # Research documentation, guidelines, protocol
│   ├── workflows/                     # Server workflows and transfer instructions
│   ├── context/                       # Experiment context notes
│   ├── design/                        # Technical architecture specifications
│   ├── notes/                         # Conceptual blueprints (blueprints/, advisor_reviews/)
│   └── MIGRATION_RECORD.md            # Complete archive mapping from pre-reorganization Git history
└── tests/                             # Unit tests (all 51 tests passing)
```

---

## 🚦 Benchmark Tracks

| Track | Goal | Benchmark Split | Metrics |
|---|---|---|---|
| **Track 1: Timbre & Identity** | Cross-modal speaker verification (speech query vs. singing gallery) across acoustic and SSL models | 20 test speakers (JVS-MuSiC), duration & text controls | Top-1 Recall ($R@1$), $R@5$, MRR, Margin |
| **Track 2: Vocal Technique** | Technique subspace probing and latent steering | 200 singers (GTSinger), vowel-matched pairs | Linear Probe AUC, EER, Technique Separation |

---

## 🚀 Quickstart & Reproduction Workflow

### 1. Environment Setup & Package Installation
```bash
# Set up Python virtual environment via uv
uv sync

# Install singing_identity in editable development mode (the only supported installation method)
pip install -e .   # or: uv pip install -e .
```

### 2. Verify Datasets & Worktree Setup
```bash
# Verify scratch mounts and initialize worktree environment
bash scripts/wt_setup.sh
```

### 3. Run Benchmark Tests
```bash
# Run unit test suite
uv run pytest
```

### 4. Running an Experiment
```bash
# Run Stage 1 benchmark sweep
uv run python scripts/run/run_stage1.py --synthetic --smoke-only

# Run identity residual verification evaluation
uv run python scripts/evaluate/evaluate_verification.py --results-dir results/identity_residual_final_validation_2026-07-09
```

---

## 💾 Storage & Compute Guidelines

- **Compute Node:** Run on compute nodes (`valkyrie01`–`valkyrie08`), never on `athena` (jump host).
- **Heavy Data & Outputs:** Stored on node-local NVMe scratch (`/localdisk/bowen/singing_identity/data/` and `/localdisk/bowen/singing_identity/features/`).
- **NFS Policy:** Never store raw WAVs, large `.npz` feature caches, or heavy run outputs inside this Git repository.

---

## 🧬 Experiment Provenance & Reproducibility Layer

To ensure scientific integrity, every benchmark result is strictly traceable according to the provenance invariant:

> **The Invariant:** Every reported benchmark result must be traceable to one concrete run, and every run must identify the exact code, resolved configuration, data/features, execution command, and relevant environment that produced it.
>
> $$\text{RESULT} \longrightarrow \text{RUN ID} \longrightarrow \begin{cases} \text{Resolved Configuration } (\texttt{config.json}) \\ \text{Git State (40-char SHA + dirty flag + } \texttt{artifacts/git_patch.diff}) \\ \text{Data \& Feature Fingerprint (manifest SHA-256 + cache root)} \\ \text{Execution Context (command, seed, host, Python/CUDA, library versions)} \end{cases}$$

### Machine-Readable Run Layout

Every invocation of an experiment runner (e.g. `scripts/run/run_stage1.py`) automatically populates a standard run directory under `runs/<experiment>/<approach>/<run_id>/`:

```text
runs/<experiment>/<approach>/<run_id>/
├── config.json               # Complete, self-contained resolved hyperparameter snapshot
├── metadata.json             # Git SHA, dirty flag, data SHA-256, seed, host, CUDA & packages
├── command.txt              # Exact CLI command string executed
├── metrics.json              # Standardized metric payload (val_eer, r1, mrr, etc.)
├── logs/                     # Console stdout/stderr log files
└── artifacts/                # Generated artifacts
    └── git_patch.diff        # Captured working tree diff (if run on a dirty commit)
```

### Traceability Walkthrough: Tracing a Benchmark Row

In `results/summary.csv`, consider Row 1:

```csv
stage1_multimodel,acoustic,synthetic_gtsinger,20260905-155939_seed42,f1bde5ce3896ae88ef43b804e31c97902f1b246b,True,42,complete,...
```

From this single row:
1. **Run Directory:** Navigate directly to `runs/stage1_multimodel/acoustic/20260905-155939_seed42/`.
2. **Exact Command:** Read `command.txt`:
   ```bash
   python3 scripts/run/run_stage1.py --synthetic --smoke-only --models acoustic
   ```
3. **Exact Code State:** Inspect `metadata.json` (`git.commit: f1bde5ce3896ae88ef43b804e31c97902f1b246b`). Since `git.dirty: true`, view the full unified patch at `artifacts/git_patch.diff`.
4. **Data Fingerprint:** Inspect `metadata.json` (`data.manifest_sha256: 53a77785de992f568f41b9486c44f3b7bfd34a8b38d19bd18e8dd07041b7e752`), confirming exact match with `data/processed/gtsinger_utterances.jsonl`.
5. **Reproducing Feature Caches:** If features are missing on a fresh compute node, re-extract using:
   ```bash
   uv run python scripts/data/extract_features.py \
       --approach configs/approaches/acoustic.json \
       --manifest data/processed/gtsinger_utterances.jsonl \
       --feature-root /localdisk/bowen/singing_identity/features
   ```

### Provenance Enforcement & Auditing Tools

- **Enforce Clean Git State (Production Runs):**
  ```bash
  uv run python scripts/run/run_stage1.py --strict-reproducibility ...
  # Will immediately raise RuntimeError and halt if the working tree has uncommitted edits.
  ```

- **Validate Run Provenance:**
  ```bash
  uv run python scripts/summarize/validate_run_provenance.py runs/stage1_multimodel/acoustic/20260905-155939_seed42
  # [✓] Run: 20260905-155939_seed42 (COMPLETE)
  ```

- **Aggregate Benchmark Summary:**
  ```bash
  uv run python scripts/summarize/summarize_results.py
  # Scans runs/ and results/, generating results/summary.csv and results/summary_provenance.json
  ```

