#!/usr/bin/env bash
set -u

export PATH="$HOME/.local/bin:$PATH"
export XDG_CACHE_HOME=/localdisk/bowen/.cache
export HF_HOME=/localdisk/bowen/.cache/huggingface
export HF_DATASETS_CACHE=/localdisk/bowen/.cache/huggingface/datasets
export HUGGINGFACE_HUB_CACHE=/localdisk/bowen/.cache/huggingface/hub
export TRANSFORMERS_CACHE=/localdisk/bowen/.cache/huggingface/transformers
export TORCH_HOME=/localdisk/bowen/.cache/torch
export PIP_CACHE_DIR=/localdisk/bowen/.cache/pip
export UV_CACHE_DIR=/localdisk/bowen/.cache/uv
export UV_LINK_MODE=copy
export CUDA_CACHE_PATH=/localdisk/bowen/.cache/nv/ComputeCache
export TORCH_EXTENSIONS_DIR=/localdisk/bowen/.cache/torch_extensions
export TMPDIR=/localdisk/bowen/tmp

RUN_ROOT=/localdisk/bowen/singing_identity/runs/jvs_music_retrieval_2026-07-08
FEATURE_ROOT=/localdisk/bowen/singing_identity/features/jvs_music_2026-07-08
MANIFEST="$RUN_ROOT/manifests/jvs_music_utterances.jsonl"
RESULT_ROOT=experiments/track1_timbre/results/jvs_music_2026-07-08
PRED_ROOT="$RUN_ROOT/predictions"

mkdir -p "$RUN_ROOT/logs" "$FEATURE_ROOT" "$RESULT_ROOT" "$PRED_ROOT" "$TMPDIR"

{
  date -Is
  hostname
  printf "run_root=%s\n" "$RUN_ROOT"
  printf "feature_root=%s\n" "$FEATURE_ROOT"
  printf "cache_root=%s\n" "$XDG_CACHE_HOME"
  printf "manifest=%s\n" "$MANIFEST"
  printf "python=%s\n" "$(uv run python -c "import sys; print(sys.executable)")"
  findmnt -T /localdisk/bowen -o TARGET,SOURCE,FSTYPE,SIZE,AVAIL,USE%
  all_gpus
  all_cpus
} > "$RUN_ROOT/preflight_priority3.log" 2>&1

status=0

run_step() {
  local name="$1"
  shift
  echo "[$(date -Is)] START $name"
  "$@"
  local rc=$?
  echo "[$(date -Is)] END $name rc=$rc"
  if [ "$rc" -ne 0 ]; then
    status=$rc
  fi
  return "$rc"
}

if [ "$status" -eq 0 ]; then
  run_step acoustic_extract \
    uv run python -u scripts/data_prep/extract_acoustic_features.py \
      --manifest "$MANIFEST" \
      --feature-root "$FEATURE_ROOT" \
      --extractor acoustic_baseline \
      --checkpoint-hash local_wave_v1 \
      --layer frame25ms_hop20ms \
      --skip-existing \
      --metadata-out "$RUN_ROOT/acoustic_metadata.json"
fi

if [ "$status" -eq 0 ]; then
  run_step ecapa_extract \
    uv run --with speechbrain --with torch --with torchaudio --extra-index-url https://download.pytorch.org/whl/cu128 \
      python -u scripts/data_prep/extract_ecapa_features.py \
      --manifest "$MANIFEST" \
      --feature-root "$FEATURE_ROOT" \
      --extractor ecapa_tdnn \
      --checkpoint-hash speechbrain_spkrec_ecapa_voxceleb \
      --layer embedding \
      --device cuda \
      --skip-existing \
      --summary-out "$RUN_ROOT/ecapa_metadata.json"
fi

if [ "$status" -eq 0 ]; then
  run_step wavlm_extract \
    uv run --with torch --with torchaudio --with transformers --extra-index-url https://download.pytorch.org/whl/cu128 \
      python -u scripts/data_prep/extract_wavlm_features.py \
      --manifest "$MANIFEST" \
      --feature-root "$FEATURE_ROOT" \
      --layers 3 6 9 12 \
      --device cuda \
      --skip-existing
fi

if [ "$status" -eq 0 ]; then
  run_step hubert_extract \
    uv run --with torch --with torchaudio --with transformers --extra-index-url https://download.pytorch.org/whl/cu128 \
      python -u scripts/data_prep/extract_hf_audio_features.py \
      --manifest "$MANIFEST" \
      --feature-root "$FEATURE_ROOT" \
      --extractor hubert_base \
      --model-name facebook/hubert-base-ls960 \
      --checkpoint-hash facebook_hubert_base_ls960 \
      --layers 3 6 9 12 \
      --device cuda \
      --skip-existing
fi

if [ "$status" -eq 0 ]; then
  run_step mert_extract \
    uv run --with torch --with torchaudio --with transformers --with librosa --extra-index-url https://download.pytorch.org/whl/cu128 \
      python -u scripts/data_prep/extract_hf_audio_features.py \
      --manifest "$MANIFEST" \
      --feature-root "$FEATURE_ROOT" \
      --extractor mert_v1_95m \
      --model-name m-a-p/MERT-v1-95M \
      --checkpoint-hash m_a_p_mert_v1_95m \
      --layers 3 6 9 12 \
      --device cuda \
      --trust-remote-code \
      --skip-existing
fi

run_retrieval_pair() {
  local prefix="$1"
  local extractor="$2"
  local checkpoint="$3"
  local layer="$4"
  run_step "${prefix}_retrieval_raw" \
    uv run python -u scripts/probing/run_speaker_retrieval.py \
      --manifest "$MANIFEST" \
      --feature-root "$FEATURE_ROOT" \
      --extractor "$extractor" \
      --checkpoint-hash "$checkpoint" \
      --layer "$layer" \
      --metrics-out "$RESULT_ROOT/${prefix}_retrieval_metrics.json"
  run_step "${prefix}_retrieval_residualized" \
    uv run python -u scripts/probing/run_speaker_retrieval.py \
      --manifest "$MANIFEST" \
      --feature-root "$FEATURE_ROOT" \
      --extractor "$extractor" \
      --checkpoint-hash "$checkpoint" \
      --layer "$layer" \
      --residualize-nuisance \
      --metrics-out "$RESULT_ROOT/${prefix}_retrieval_residualized_metrics.json"
}

run_mode_pair() {
  local prefix="$1"
  local extractor="$2"
  local checkpoint="$3"
  local layer="$4"
  run_step "${prefix}_mode_raw" \
    uv run --with scikit-learn python -u scripts/probing/run_mode_probe.py \
      --manifest "$MANIFEST" \
      --feature-root "$FEATURE_ROOT" \
      --extractor "$extractor" \
      --checkpoint-hash "$checkpoint" \
      --layer "$layer" \
      --metrics-out "$RESULT_ROOT/${prefix}_mode_metrics.json" \
      --predictions-out "$PRED_ROOT/${prefix}_mode_predictions.jsonl"
  run_step "${prefix}_mode_residualized" \
    uv run --with scikit-learn python -u scripts/probing/run_mode_probe.py \
      --manifest "$MANIFEST" \
      --feature-root "$FEATURE_ROOT" \
      --extractor "$extractor" \
      --checkpoint-hash "$checkpoint" \
      --layer "$layer" \
      --residualize-nuisance \
      --metrics-out "$RESULT_ROOT/${prefix}_mode_residualized_metrics.json" \
      --predictions-out "$PRED_ROOT/${prefix}_mode_residualized_predictions.jsonl"
}

if [ "$status" -eq 0 ]; then
  for spec in \
    "acoustic acoustic_baseline local_wave_v1 frame25ms_hop20ms" \
    "ecapa ecapa_tdnn speechbrain_spkrec_ecapa_voxceleb embedding" \
    "wavlm_l3 wavlm_base_plus microsoft_wavlm_base_plus 3" \
    "wavlm_l6 wavlm_base_plus microsoft_wavlm_base_plus 6" \
    "wavlm_l9 wavlm_base_plus microsoft_wavlm_base_plus 9" \
    "wavlm_l12 wavlm_base_plus microsoft_wavlm_base_plus 12" \
    "hubert_l3 hubert_base facebook_hubert_base_ls960 3" \
    "hubert_l6 hubert_base facebook_hubert_base_ls960 6" \
    "hubert_l9 hubert_base facebook_hubert_base_ls960 9" \
    "hubert_l12 hubert_base facebook_hubert_base_ls960 12" \
    "mert_l3 mert_v1_95m m_a_p_mert_v1_95m 3" \
    "mert_l6 mert_v1_95m m_a_p_mert_v1_95m 6" \
    "mert_l9 mert_v1_95m m_a_p_mert_v1_95m 9" \
    "mert_l12 mert_v1_95m m_a_p_mert_v1_95m 12"; do
    set -- $spec
    run_retrieval_pair "$1" "$2" "$3" "$4"
    if [ "$status" -eq 0 ]; then
      run_mode_pair "$1" "$2" "$3" "$4"
    fi
  done
fi

printf "%s\n" "$status" > "$RUN_ROOT/priority3_exit_code.txt"
exit "$status"
