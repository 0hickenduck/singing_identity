# 🎙️ Singing Representation Research

**Research question:** Where does singer identity survive when a person moves from speech to singing, and how can we leverage that to control technique and timbre in zero-shot singing voice conversion?

This directory contains all code, design documents, and results for the **two-track master's research project** on singing representation learning.

---

## 📁 Directory Structure

```
singing_representation/
│
├── design/                     # Design documents and context for Codex
│   ├── CODEX_INSTRUCTIONS.md   # ← START HERE (Codex entry point)
│   └── links/                  # Symlinks to pro_suggestions and free_recall docs
│
├── experiments/
│   ├── track1_timbre/          # Track 1: Timbre Mode Residual Mapper
│   │   └── README.md
│   └── track2_technique/       # Track 2: Technique Probing & Reconstruction
│       └── README.md
│
├── scripts/
│   ├── data_prep/              # Data extraction and manifest building
│   ├── probing/                # Linear probes, retrieval, residual control
│   └── intervention/           # Latent steering, LoRA, Seed-VC injection
│
├── notebooks/                  # Marimo exploratory analysis and visualization
│
├── results/                    # Compact reports, tables, and small final figures
│
└── README.md                   # This file
```

---

## 🚦 Track Overview

| Track | Goal | Primary Doc | Status |
|-------|------|-------------|--------|
| **Track 1: Timbre** | Map speech→singing mode residual; improve Seed-VC with speech-only reference | [pro3](pro_suggestions/pro3_voice_representation_idea_review.md) | 🟡 Ready to implement |
| **Track 2: Technique** | Find linear directions for vibrato/falsetto in frozen SSL; steer downstream synthesis | [pro5 §Backup](pro_suggestions/pro5_research_direction_ranking.md) | 🟡 Ready to implement |

---

## 📖 How to Read the Design Docs

1. **Server workflow:** [`LAB_SERVER_WORKFLOW.md`](LAB_SERVER_WORKFLOW.md) — where code, data, caches, runs, notebooks, and reports belong on the cluster
2. **Codex entry point:** [`design/CODEX_INSTRUCTIONS.md`](design/CODEX_INSTRUCTIONS.md)
3. **Detailed design:** [`pro_suggestions/pro3_voice_representation_idea_review.md`](pro_suggestions/pro3_voice_representation_idea_review.md) — full Stage A/B/C engineering spec
4. **30-day plan & go/no-go:** [`pro_suggestions/pro5_research_direction_ranking.md`](pro_suggestions/pro5_research_direction_ranking.md)
5. **Research methodology:** [`pro_suggestions/pro7_ml_audio_research_workflow.md`](pro_suggestions/pro7_ml_audio_research_workflow.md)

---

## 🖥️ Environment

- **Primary machine:** Lab server GPU node (`valkyrie*`), never the `athena` jump host
- **Active data/runs/caches:** `/localdisk/bowen`, verified with `findmnt`
- **Python manager:** `uv`
- **Notebook tool:** `marimo`
- **Current experiment scripts:** `scripts/data_prep/`, `scripts/probing/`, and `scripts/intervention/`
- **Branch:** `codex/research-system-architecture`

Run the lab preflight before long jobs:

```bash
bash scripts/check_lab_environment.sh
```

## Smoke Test

The repository includes a synthetic feature-cache fixture so the two experiment tracks can be tested before real datasets and model checkpoints are available.

```bash
uv run python -m unittest
```
