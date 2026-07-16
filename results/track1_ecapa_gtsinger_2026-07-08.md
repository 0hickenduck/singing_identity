# Track 1 ECAPA GTSinger Retrieval

Date: 2026-07-08

## Question

Does an independent speaker embedding model recover same-person identity across speech and singing on the repaired GTSinger subset?

## Run

- run_root: `/localdisk/bowen/singing_identity/runs/track1_ecapa_gtsinger_2026-07-08`
- feature_root: `/localdisk/bowen/singing_identity/features/stage1_repaired_200_fresh_local`
- manifest: `/localdisk/bowen/singing_identity/runs/stage1_repaired_200_fresh_local/manifests/gtsinger_utterances.jsonl`
- extractor: `ecapa_tdnn`
- checkpoint: `speechbrain/spkrec-ecapa-voxceleb`
- cache key: `speechbrain_spkrec_ecapa_voxceleb/embedding`
- script: `scripts/data_prep/extract_ecapa_features.py`
- speakers: 20
- chance R@1: 5.0%

Preflight was recorded in:

`/localdisk/bowen/singing_identity/runs/track1_ecapa_gtsinger_2026-07-08/preflight.log`

## Results

| Condition | Speech->Singing R@1 | x chance | Singing->Speech R@1 | x chance | S->G mAP | G->S mAP |
|---|---:|---:|---:|---:|---:|---:|
| Raw | 90.0% | 18.0x | 85.0% | 17.0x | 0.935 | 0.925 |
| Residualized | 90.0% | 18.0x | 85.0% | 17.0x | 0.942 | 0.925 |

The residualized condition removes F0 mean/std, voiced percentage, energy mean/std, duration, and RMS using a train-speaker-only residualizer.

## Verdict

ECAPA strongly retrieves same-person speech/singing pairs. This supports the Priority 1 Go decision: the cross-mode identity signal is not unique to WavLM and is visible in an independent speaker-recognition embedding.

## Outputs

- `experiments/track1_timbre/results/ecapa_retrieval_metrics.json`
- `experiments/track1_timbre/results/ecapa_retrieval_residualized_metrics.json`
- `/localdisk/bowen/singing_identity/runs/track1_ecapa_gtsinger_2026-07-08/ecapa_metadata.json`
