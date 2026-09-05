# Benchmark Structure & Provenance Migration Record

**Date:** 2026-09-06 JST  
**Author:** Bowen / Antigravity  
**Branch:** `reorg/conventional-layout`  
**Goal:** Reorganize `singing_identity` into a conventional benchmark-style ML research repository with machine-readable experiment provenance.

---

## 1. Top-Level Path & Legacy Mappings

Every pre-existing file and directory was moved using `git mv` to ensure complete Git history preservation.

| Old Historical Path | New Semantic Path | Status & Notes |
|---|---|---|
| `experiments/track1_timbre/manifests/gtsinger_utterances.jsonl` | `data/processed/gtsinger_utterances.jsonl` | `git mv` (Tracked manifest) |
| `experiments/track1_timbre/manifests/gtsinger_pairs.jsonl` | `data/processed/gtsinger_pairs.jsonl` | `git mv` (Tracked manifest) |
| `experiments/track2_technique/manifests/gtsinger_phone_examples.jsonl` | `data/processed/gtsinger_phone_examples.jsonl` | `git mv` (Tracked manifest) |
| `experiments/track2_technique/manifests/gtsinger_phoneme_pairs.jsonl` | `data/processed/gtsinger_phoneme_pairs.jsonl` | `git mv` (Tracked manifest) |
| `experiments/track1_timbre/config.example.json` | `configs/experiments/track1_timbre.json` | `git mv` (Tracked declarative config) |
| `experiments/track2_technique/config.example.json` | `configs/experiments/track2_technique.json` | `git mv` (Tracked declarative config) |
| `experiments/track1_timbre/README.md` | `docs/notes/track1_timbre_spec.md` | `git mv` (Tracked specification) |
| `experiments/track2_technique/README.md` | `docs/notes/track2_technique_spec.md` | `git mv` (Tracked specification) |
| `experiments/track1_timbre/reports/*.json` | `results/track1_timbre/reports/` | `git mv` (Tracked evaluation reports) |
| `experiments/track1_timbre/results/*` | `results/track1_timbre/` | Moved (JSON metrics & JSONL predictions) |
| `experiments/track2_technique/results/*` | `results/track2_technique/` | Moved (JSON metrics & JSONL directions) |
| `experiments/shared_features/*` (1.1 GB `.npz`) | `/localdisk/bowen/singing_identity/features/shared_features/` | Offloaded to local NVMe scratch per lab policy |
| `free_recall/` | `docs/notes/blueprints/` | `git mv` (Renamed from conversational name) |
| `pro_suggestions/` | `docs/notes/advisor_reviews/` | `git mv` (Renamed from conversational name) |
| `docs/reports/` (Chowdhury 2022) | `docs/notes/literature_baseline_chowdhury2022/` | `git mv` (Removed overlapping reports dir) |
| `results/reports/` (temporary duplicate) | Removed | Consolidated directly into `results/` |
| `context/` | `docs/context/` | `git mv` |
| `design/` | `docs/design/` | `git mv` |
| `scripts/probing/` | `legacy/probing/` | `git mv` |
| `scripts/intervention/` | `legacy/intervention/` | `git mv` |
| `scripts/data_prep/` | `legacy/data_prep/` | `git mv` |
| `scripts/review/` | `legacy/review/` | `git mv` |
| `scripts/research_utils.py` | `legacy/research_utils.py` | `git mv` (canonical in `singing_identity.utils.research_utils`) |
| `scripts/run_stage1_overnight.py` | `legacy/run_stage1_overnight.py` | `git mv` |
| `scripts/run_targeted_repair_then_stage1.py` | `legacy/run_targeted_repair_then_stage1.py` | `git mv` |
| `scripts/build_demo.py` | `legacy/build_demo.py` | `git mv` |
| `scripts/mark_experiment.py` | `legacy/mark_experiment.py` | `git mv` |
| `scripts/prepare_raw_wavs.py` | `legacy/prepare_raw_wavs.py` | `git mv` |
| `scripts/audit_stage1_results.py` | `legacy/audit_stage1_results.py` | `git mv` |
| `scripts/*.sh` (ad-hoc wrappers) | `legacy/*.sh` | `git mv` |

---

## 2. Canonical Entry Points

Under `scripts/`, exactly one canonical script is maintained per operation:

1. **Prepare manifests & feature caches:** `scripts/data/prepare_manifests.py`
2. **Run experiments:** `scripts/run/run_stage1.py`
3. **Evaluate verification metrics:** `scripts/evaluate/evaluate_verification.py`
4. **Summarize benchmark results:** `scripts/summarize/summarize_results.py`
5. **Lab setup:** `scripts/wt_setup.sh` (required by lab worktree conventions)

---

## 3. Installation Standardization

- The machine-specific `.pth` file (`.venv/.../singing_identity.pth`) has been completely removed.
- `pyproject.toml` now defines a standard PEP-517/518 build system using `setuptools>=61.0`.
- Package installation is performed strictly via standard editable install:
  ```bash
  pip install -e .   # or: uv pip install -e .
  ```
- Verified: `import singing_identity` resolves cleanly to `src/singing_identity/__init__.py`.

---

## 4. Provenance Core Implementation

### 4.1 Provenance Invariant
> Every reported benchmark result must be traceable to one concrete run, and every run must identify the exact code, resolved configuration, data/features, execution command, and relevant environment that produced it.

$$\text{RESULT} \longrightarrow \text{RUN ID} \longrightarrow \begin{cases} \text{Resolved Configuration } (\texttt{config.json}) \\ \text{Git State (40-char SHA + dirty flag + } \texttt{artifacts/uncommitted.patch}) \\ \text{Data \& Feature Fingerprint (manifest SHA-256 + cache root)} \\ \text{Execution Context (command, seed, host, Python)} \end{cases}$$

### 4.2 Standard Run Directory Layout
```text
runs/<experiment>/<approach>/<run_id>/
├── config.json               # Self-contained snapshot of resolved hyperparameters
├── metadata.json             # Git SHA, branch, dirty flag, manifest SHA-256, seed, host
├── command.txt              # Exact CLI command string executed
├── metrics.json              # Standardized metric payload
├── logs/                     # Console stdout/stderr logs
└── artifacts/                # Run outputs
    └── uncommitted.patch     # Unified git diff when run on a dirty working tree
```

### 4.3 Summary Table & Provenance Mapping
- `results/summary.csv`: Aggregated table containing `run_id`, `git_commit`, `git_dirty`, `seed`, `provenance_status`, and benchmark metrics.
- `results/summary_provenance.json`: Companion machine-readable dictionary linking each row directly to its `config.json`, `metadata.json`, and run directory.

---

## 5. Test Suite Verification

All 54 test cases in `tests/` pass with zero regressions:
- 46 original algorithmic, data, and runner unit tests.
- 8 provenance and reproducibility unit tests in `tests/test_provenance.py`.
Execution time: ~12s under `uv run pytest`.
