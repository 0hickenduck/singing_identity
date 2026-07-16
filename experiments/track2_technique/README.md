# Track 2: Technique Probing & Reconstruction

**Goal:** Find linear directions for singing techniques (vibrato, falsetto, breathy, etc.) in frozen SSL representations, then steer downstream synthesis.

## Status
🟡 Ready to implement (activate if Track 1 go/no-go fails)

## Entry Point
See [`../design/CODEX_INSTRUCTIONS.md`](../design/CODEX_INSTRUCTIONS.md) Section 4 for full step-by-step.

## Outputs (will be created here)
- `manifests/phoneme_pairs.parquet` — same-phoneme, different-technique pairs from GTSinger
- `reports/` — compact metric summaries and final figures
- Large frame-level feature caches, direction dumps, audio samples, logs, and run roots belong under `/localdisk/bowen/singing_identity`

## First Runnable Loop

The first implemented probe estimates a same-phone technique direction and bootstraps angular stability.

```bash
FEATURE_ROOT=/localdisk/bowen/singing_identity/features/track2_first_loop
RUN_ROOT=/localdisk/bowen/singing_identity/runs/track2_first_loop
mkdir -p "$FEATURE_ROOT" "$RUN_ROOT"

uv run python scripts/probing/run_technique_directions.py \
  --pairs experiments/track2_technique/manifests/phoneme_pairs.jsonl \
  --feature-root "$FEATURE_ROOT" \
  --extractor wavlm_base_plus \
  --checkpoint-hash CHECKPOINT_HASH \
  --layer 6 \
  --metrics-out experiments/track2_technique/reports/vibrato_direction_layer6_metrics.json \
  --directions-out "$RUN_ROOT/vibrato_direction_layer6.jsonl" \
  --bootstrap 1000
```

## Latent Steering Loop From Completed Stage 1

Proceed narrowly with MERT layer 3 breathy. This script materializes the three causal listening-test conditions requested for each selected pair:

1. `baseline`: source content + target speaker ID.
2. `baseline_plus_source_vector`: baseline + original source vector.
3. `steered`: baseline + source vector + breathy direction delta Z.

Actual audio synthesis requires a frozen decoder adapter supplied through `--decoder-command`, using `{condition_npz}` and `{audio_out}` placeholders.

```bash
WORK_ROOT=/localdisk/bowen/singing_identity
RUN_ROOT=$WORK_ROOT/runs/stage1_repaired_200_fresh_local
FEATURE_ROOT=$WORK_ROOT/features/stage1_repaired_200_fresh_local
OUT_ROOT=$WORK_ROOT/runs/track2_latent_steering_mert_l3_breathy

uv run python scripts/intervention/run_latent_steering.py \
  --pairs "$RUN_ROOT/manifests/gtsinger_phoneme_pairs.jsonl" \
  --directions "$RUN_ROOT/track2/mert/3/technique_directions.jsonl" \
  --feature-root "$FEATURE_ROOT" \
  --extractor mert_v1_95m \
  --checkpoint-hash m_a_p_mert_v1_95m \
  --layer 3 \
  --phone "<AP>" \
  --technique breathy \
  --lambda-scale 1.0 \
  --out-root "$OUT_ROOT" \
  --manifest-out "$OUT_ROOT/conditions.jsonl" \
  --metrics-out "$OUT_ROOT/metrics.json" \
  --max-examples 20
```
