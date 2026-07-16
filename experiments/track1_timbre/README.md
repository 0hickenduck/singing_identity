# Track 1: Timbre Mode Residual

**Goal:** Map the vocal identity shift when a person moves from speech to singing ($\Delta = T_{\text{singing}} - T_{\text{speech}}$), then use it to improve zero-shot SVC when only target speech is available.

## Status
🟡 Ready to implement

## Entry Point
See [`../design/CODEX_INSTRUCTIONS.md`](../design/CODEX_INSTRUCTIONS.md) Section 3 for full step-by-step.

## Outputs (will be created here)
- `manifests/utterances.parquet` — the dataset manifest
- `reports/` — compact manifest summaries and decision notes
- Large feature caches, prediction dumps, logs, and run roots belong under `/localdisk/bowen/singing_identity`

## First Runnable Loop

Use JSONL until `pandas`/`pyarrow` are installed; the scripts also accept Parquet when those packages exist.

```bash
uv run python scripts/data_prep/build_manifest.py \
  --input raw_metadata.jsonl \
  --output experiments/track1_timbre/manifests/utterances.jsonl \
  --summary experiments/track1_timbre/reports/manifest_summary.json \
  --pairs-input raw_pairs.jsonl \
  --pairs-output experiments/track1_timbre/manifests/pairs.jsonl \
  --pairs-summary experiments/track1_timbre/reports/pair_summary.json

FEATURE_ROOT=/localdisk/bowen/singing_identity/features/track1_first_loop
RUN_ROOT=/localdisk/bowen/singing_identity/runs/track1_first_loop
mkdir -p "$FEATURE_ROOT" "$RUN_ROOT"

uv run python scripts/data_prep/validate_feature_cache.py \
  --manifest experiments/track1_timbre/manifests/utterances.jsonl \
  --feature-root "$FEATURE_ROOT" \
  --extractor wavlm_base_plus \
  --checkpoint-hash CHECKPOINT_HASH \
  --layer 6 \
  --report experiments/track1_timbre/reports/feature_cache_layer6.json

uv run python scripts/probing/run_mode_probe.py \
  --manifest experiments/track1_timbre/manifests/utterances.jsonl \
  --feature-root "$FEATURE_ROOT" \
  --extractor wavlm_base_plus \
  --checkpoint-hash CHECKPOINT_HASH \
  --layer 6 \
  --metrics-out experiments/track1_timbre/reports/mode_probe_layer6_metrics.json \
  --predictions-out "$RUN_ROOT/mode_probe_layer6_predictions.jsonl"

uv run python scripts/probing/run_residual_control.py \
  --manifest experiments/track1_timbre/manifests/utterances.jsonl \
  --feature-root "$FEATURE_ROOT" \
  --extractor wavlm_base_plus \
  --checkpoint-hash CHECKPOINT_HASH \
  --layer 6 \
  --metrics-out experiments/track1_timbre/reports/mode_probe_layer6_residualized_metrics.json \
  --predictions-out "$RUN_ROOT/mode_probe_layer6_residualized_predictions.jsonl"

uv run python scripts/probing/run_speaker_retrieval.py \
  --manifest experiments/track1_timbre/manifests/utterances.jsonl \
  --feature-root "$FEATURE_ROOT" \
  --extractor wavlm_base_plus \
  --checkpoint-hash CHECKPOINT_HASH \
  --layer 6 \
  --metrics-out experiments/track1_timbre/reports/retrieval_layer6_metrics.json
```

## Stage B/C Intervention Loop From Completed Stage 1

Primary candidates from the completed repaired Stage 1 run are WavLM layers 9 and 6. This writes decoder-ready condition vectors and a manifest under `/localdisk`. To synthesize audio, pass a real frozen decoder adapter through `--decoder-command`, using `{condition_npz}` and `{audio_out}` placeholders.

```bash
WORK_ROOT=/localdisk/bowen/singing_identity
RUN_ROOT=$WORK_ROOT/runs/stage1_repaired_200_fresh_local
FEATURE_ROOT=$WORK_ROOT/features/stage1_repaired_200_fresh_local
OUT_ROOT=$WORK_ROOT/runs/track1_seedvc_intervention_wavlm_l9

uv run python scripts/intervention/run_seedvc_inject.py \
  --manifest "$RUN_ROOT/manifests/gtsinger_utterances.jsonl" \
  --pairs "$RUN_ROOT/manifests/gtsinger_pairs.jsonl" \
  --feature-root "$FEATURE_ROOT" \
  --extractor wavlm_base_plus \
  --checkpoint-hash microsoft_wavlm_base_plus \
  --layer 9 \
  --mapper-model "$RUN_ROOT/track1/wavlm/9/mapper_model.npz" \
  --out-root "$OUT_ROOT" \
  --manifest-out "$OUT_ROOT/conditions.jsonl" \
  --metrics-out "$OUT_ROOT/metrics.json" \
  --max-pairs 20 \
  --pair-split test \
  --include-oracle
```

## Seed-VC Black-Box Prompt Baseline

The public Seed-VC CLI accepts source and reference audio, not arbitrary WavLM/MERT vectors. Use this baseline before deeper internal patching: convert the same different-speaker source singing clip twice, once with the target speech prompt and once with the target singing prompt.

```bash
WORK_ROOT=/localdisk/bowen/singing_identity
RUN_ROOT=$WORK_ROOT/runs/stage1_repaired_200_fresh_local
SEEDVC_ROOT=$WORK_ROOT/external/seed-vc
OUT_ROOT=$WORK_ROOT/runs/track1_seedvc_prompt_baseline_10pairs_rr

TMPDIR=/tmp /tmp/singing_identity_cuda_venv/bin/python \
  scripts/intervention/run_seedvc_prompt_baseline.py \
  --manifest "$RUN_ROOT/manifests/gtsinger_utterances.jsonl" \
  --pairs "$RUN_ROOT/manifests/gtsinger_pairs.jsonl" \
  --seedvc-root "$SEEDVC_ROOT" \
  --out-root "$OUT_ROOT" \
  --manifest-out "$OUT_ROOT/conditions.jsonl" \
  --metrics-out "$OUT_ROOT/metrics.json" \
  --max-pairs 10 \
  --pair-split test \
  --diffusion-steps 10 \
  --fp16 \
  --run

TMPDIR=/tmp /tmp/singing_identity_cuda_venv/bin/python \
  scripts/intervention/summarize_seedvc_prompt_outputs.py \
  --conditions "$OUT_ROOT/conditions.jsonl" \
  --manifest-out "$OUT_ROOT/listening_manifest.jsonl" \
  --summary-out "$OUT_ROOT/audio_summary.json"

TMPDIR=/tmp /tmp/singing_identity_cuda_venv/bin/python \
  scripts/intervention/evaluate_seedvc_prompt_outputs.py \
  --listening-manifest "$OUT_ROOT/listening_manifest.jsonl" \
  --scores-out "$OUT_ROOT/resemblyzer_scores.jsonl" \
  --summary-out "$OUT_ROOT/resemblyzer_summary.json" \
  --device cuda
```

Current balanced smoke result: 10 heldout target pairs, 20 WAVs, four target speakers. Resemblyzer triage found singing-prompt outputs moved toward target-singing references in 10/10 pairs and away from target-speech references in 10/10 pairs. Mean deltas: target singing `+0.1103`, target speech `-0.0750`, source singing `+0.0358`.
