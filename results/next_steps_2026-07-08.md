# Next Steps: Fill the Residualized Retrieval Gap

Date: 2026-07-08

## Goal

Fill the single most important evidence gap: **residualized cross-mode identity retrieval**. Then expand to JVS (100 speakers) for reliable mapper evaluation.

## Priority 1: Residualized Cross-Mode Identity Retrieval on GTSinger (1-2 hours)

### What to do

Add `--residualize-nuisance` support to `scripts/probing/run_speaker_retrieval.py`. The residualizer is already implemented in `research_utils.py` (`fit_nuisance_residualizer` / `apply_nuisance_residualizer`). The retrieval script just needs to:

1. Load the nuisance matrix (F0, energy, duration, voiced rate) for all rows.
2. Fit the residualizer on training speakers only.
3. Apply to all speaker centroids before computing cosine retrieval.

### Commands to run (on valkyrie03)

```bash
# Existing features, just re-run retrieval with residualization
FEATURE_ROOT=/localdisk/bowen/singing_identity/features/stage1_repaired_200_fresh_local
MANIFEST=/localdisk/bowen/singing_identity/runs/stage1_repaired_200_fresh_local/manifests/gtsinger_utterances.jsonl

for LAYER in 3 6 9 12; do
  uv run python scripts/probing/run_speaker_retrieval.py \
    --manifest "$MANIFEST" \
    --feature-root "$FEATURE_ROOT" \
    --extractor wavlm_base_plus \
    --checkpoint-hash microsoft_wavlm_base_plus \
    --layer "$LAYER" \
    --residualize-nuisance \
    --metrics-out "experiments/track1_timbre/results/wavlm_l${LAYER}_retrieval_residualized_metrics.json"
done

# Also run for HuBERT and MERT
for LAYER in 3 6 9 12; do
  uv run python scripts/probing/run_speaker_retrieval.py \
    --manifest "$MANIFEST" \
    --feature-root "$FEATURE_ROOT" \
    --extractor hubert_base \
    --checkpoint-hash facebook_hubert_base_ls960 \
    --layer "$LAYER" \
    --residualize-nuisance \
    --metrics-out "experiments/track1_timbre/results/hubert_l${LAYER}_retrieval_residualized_metrics.json"
done

for LAYER in 3 6 9 12; do
  uv run python scripts/probing/run_speaker_retrieval.py \
    --manifest "$MANIFEST" \
    --feature-root "$FEATURE_ROOT" \
    --extractor mert_v1_95m \
    --checkpoint-hash m-a-p_mert_v1_95m \
    --layer "$LAYER" \
    --residualize-nuisance \
    --metrics-out "experiments/track1_timbre/results/mert_l${LAYER}_retrieval_residualized_metrics.json"
done

# Acoustic baseline
uv run python scripts/probing/run_speaker_retrieval.py \
  --manifest "$MANIFEST" \
  --feature-root "$FEATURE_ROOT" \
  --extractor acoustic_baseline \
  --checkpoint-hash local_wave_v1 \
  --layer frame25ms_hop20ms \
  --residualize-nuisance \
  --metrics-out "experiments/track1_timbre/results/acoustic_retrieval_residualized_metrics.json"
```

### Go/No-Go interpretation

- If WavLM L12 residualized R@1 ≥ 16.7% (3× chance for 18 speakers): **Go**. Identity survives acoustic controls.
- If residualized R@1 drops to ≤ 8% across all models/layers: **Reconsider**. Identity retrieval may rely on acoustic shortcuts.

### ✅ ACTUAL RESULT (2026-07-08 smoke test, 20 speakers from Stage 1 repaired run)

| Condition | Speech→Singing R@1 | ×chance | Singing→Speech R@1 | ×chance |
|---|---:|---:|---:|---:|
| Raw (no controls) | 27.8% | 5.0× | 33.3% | 6.0× |
| **Residualized (F0/energy/duration/voiced/RMS removed)** | **40.0%** | **8.0×** | **50.0%** | **10.0×** |

**Verdict: Go.** Residualized retrieval is 8-10× chance, far exceeding the 3× threshold. Removing acoustic nuisance actually *improved* retrieval, suggesting F0/energy/duration confounds were hurting identity matching. This is the strongest evidence so far that SSL representations preserve genuine cross-mode identity information beyond simple acoustic statistics.

R@5 residualized: Speech→Singing = 75%, Singing→Speech = 100%.

Metrics file: `experiments/track1_timbre/results/wavlm_l12_retrieval_residualized_metrics.json`

## Priority 2: ECAPA-TDNN Speaker Embedding Extraction (2-3 hours)

### What to do

Extract ECAPA-TDNN (or CAMPPlus) speaker embeddings for all GTSinger utterances using SpeechBrain or WeSpeaker. Store in the same feature cache format as WavLM features.

Then run the same retrieval experiment (raw + residualized) using ECAPA embeddings.

### Why this matters

ECAPA provides an independent "ground truth" for identity. If ECAPA cross-mode R@1 is high, identity genuinely crosses the speech/singing boundary. If ECAPA is low but WavLM is high, WavLM's "identity" signal may be entangled with something else.

## Priority 3: JVS+JVS-MuSiC Feature Extraction and Retrieval (half day)

### What to do

1. Run `scripts/data_prep/build_jvs_music_manifests.py` to build the manifest (script exists).
2. Extract WavLM (layers 3,6,9,12), HuBERT, MERT, acoustic baseline, and ECAPA features for all 100 speakers.
3. Run cross-mode retrieval (raw + residualized) on JVS.
4. Run mode probe (raw + residualized) on JVS.

### Why this matters

100 speakers vs 18 speakers. Chance drops from 5.6% to 1%. R@1 = 10% would be 10× chance. Mapper evaluation with 15+ test speakers becomes meaningful.

## Priority 4: Mapper on JVS (after Priority 3 confirms Go)

### What to do

Run `scripts/probing/run_jvs_centroid_residual_predictor.py` with 100 JVS speakers. Test all mapper baselines:

| Model | Description |
|---|---|
| speech_only_no_residual | $\hat{g} = s$ |
| global_mean_residual | $\hat{g} = s + \mu_g - \mu_s$ |
| acoustic_metadata_ridge | Ridge on F0/energy/duration/language/vocal_range |
| speech_embedding_ridge | Ridge on $z_{\text{speech}}$ |
| speech_embedding_plus_acoustic_ridge | Ridge on $[z_{\text{speech}}, \text{acoustics}]$ |

With 100 speakers, we can have ~70 train / 15 dev / 15 test. This is the minimum for a credible mapper paper claim.

## Priority 5 (Optional): ContentVec Extraction

Extract ContentVec features. ContentVec was trained to remove speaker information, so:
- If ContentVec cross-mode retrieval is near chance → confirms ContentVec successfully removes identity.
- If ContentVec still retrieves → identity information is deeply baked into SSL representations.

This is a nice paper figure but not blocking.

## File changes needed

| File | Change |
|---|---|
| `scripts/probing/run_speaker_retrieval.py` | Add `--residualize-nuisance` flag. Load nuisance matrix, fit on train speakers, apply before computing centroids. |
| `scripts/data_prep/extract_ecapa_features.py` | New script. Extract ECAPA-TDNN embeddings using SpeechBrain. |
| `scripts/probing/run_speaker_retrieval.py` | Add `--chance-normalized` to output $R@1_{\text{norm}}$ and mAP. |

## Decision tree after Priority 1

```
residualized R@1 results
├── ≥ 3× chance on WavLM
│   ├── ECAPA also high → identity genuinely crosses modes
│   │   └── Proceed to JVS mapper (Priority 3-4)
│   └── ECAPA low → WavLM signal is not pure identity
│       └── Investigate what WavLM encodes (still publishable)
├── < 3× chance on all SSL models
│   ├── Raw R@1 was high → residualization removes identity info too
│   │   └── Try softer controls (partial residualization)
│   └── Raw R@1 was also mediocre → identity does not cross modes well
│       └── Pivot to Track 2 (technique probing)
```
