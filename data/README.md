# Benchmark Datasets & Manifests

This directory contains canonical benchmark dataset definitions and manifests for the **Singing Identity & Technique Representation Benchmark**.

---

## Storage Layout & Lab Conventions

To ensure high-throughput GPU I/O and prevent overloading shared NFS storage, audio datasets and large precomputed feature matrices are stored on node-local NVMe scratch rather than within the Git repository:

| Dataset | Localdisk Scratch Path | Format | Size | Description |
|---|---|---|---:|---|
| **GTSinger** | `/localdisk/bowen/singing_identity/data/GTSinger` | 48kHz WAV | ~53 GB | Multi-singer, multi-technique singing voice corpus |
| **JVS** | `/localdisk/bowen/singing_identity/data/jvs_ver1` | 24kHz WAV | ~5.5 GB | Japanese versatile speech dataset |
| **JVS-MuSiC** | `/localdisk/bowen/singing_identity/data/jvs_music` | 24kHz WAV | ~760 MB | Paired speech and singing recordings for JVS speakers |
| **Lab Archive** | `/home/bowen/bowen_lab/data/dataset_vtuber_20260524.tar.gz` | tar.gz | ~9.1 GB | Immutable master dataset backup |

---

## Processed Benchmark Manifests (`data/processed/`)

The benchmark tracks are defined by deterministic, lightweight JSONL manifest files stored in `data/processed/`:

- `gtsinger_utterances.jsonl`: Utterance-level metadata, acoustic statistics (F0 mean/std/voiced%, energy, RMS, SNR), phoneme alignments, and split groups.
- `gtsinger_pairs.jsonl`: Paired speech-vs-singing cross-modal verification pairs for Track 1 (Timbre & Identity).
- `gtsinger_phone_examples.jsonl`: Vowel-level segmented slices annotated with vocal technique labels for Track 2.
- `gtsinger_phoneme_pairs.jsonl`: Phoneme-matched contrastive pairs across vocal techniques.

---

## Data Preparation & Feature Extraction Pipeline

1. **Verify Raw Data on Node Scratch:**
   ```bash
   ls -ld /localdisk/bowen/singing_identity/data/*
   ```

2. **Generate Manifests:**
   ```bash
   uv run python scripts/data_prep/build_gtsinger_manifests.py
   uv run python scripts/data_prep/build_jvs_music_manifests.py
   ```

3. **Extract Model Features:**
   ```bash
   uv run python scripts/data_prep/extract_wavlm_features.py --config configs/experiments/track1_timbre.json
   uv run python scripts/data_prep/extract_hf_audio_features.py --model mert
   ```
