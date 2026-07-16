# Track 1 Residualized Cross-Mode Retrieval

Date: 2026-07-08

## Question

Does same-person speech-to-singing retrieval survive after removing low-dimensional acoustic nuisance variables?

## Run

- run_root: `/localdisk/bowen/singing_identity/runs/track1_residualized_retrieval_2026-07-08`
- feature_root: `/localdisk/bowen/singing_identity/features/stage1_repaired_200_fresh_local`
- manifest: `/localdisk/bowen/singing_identity/runs/stage1_repaired_200_fresh_local/manifests/gtsinger_utterances.jsonl`
- script: `scripts/probing/run_speaker_retrieval.py`
- controls: F0 mean/std, voiced percentage, energy mean/std, duration, RMS
- residualizer fit: deterministic train speakers only, then applied to all rows before centroid retrieval
- speakers: 20
- chance R@1: 5.0%

Preflight was recorded in:

`/localdisk/bowen/singing_identity/runs/track1_residualized_retrieval_2026-07-08/preflight.log`

The run was on `valkyrie03`; `/localdisk/bowen` was verified as local ext4 scratch.

## Results

| Condition | Speech->Singing R@1 | x chance | Singing->Speech R@1 | x chance | S->G mAP | G->S mAP |
|---|---:|---:|---:|---:|---:|---:|
| acoustic baseline | 40.0% | 8.0x | 40.0% | 8.0x | 0.514 | 0.530 |
| HuBERT L3 | 20.0% | 4.0x | 50.0% | 10.0x | 0.377 | 0.619 |
| HuBERT L6 | 20.0% | 4.0x | 55.0% | 11.0x | 0.395 | 0.690 |
| HuBERT L9 | 15.0% | 3.0x | 65.0% | 13.0x | 0.391 | 0.768 |
| HuBERT L12 | 20.0% | 4.0x | 50.0% | 10.0x | 0.407 | 0.659 |
| MERT L3 | 30.0% | 6.0x | 45.0% | 9.0x | 0.514 | 0.586 |
| MERT L6 | 25.0% | 5.0x | 40.0% | 8.0x | 0.457 | 0.570 |
| MERT L9 | 20.0% | 4.0x | 35.0% | 7.0x | 0.424 | 0.532 |
| MERT L12 | 15.0% | 3.0x | 25.0% | 5.0x | 0.352 | 0.425 |
| WavLM L3 | 25.0% | 5.0x | 40.0% | 8.0x | 0.405 | 0.568 |
| WavLM L6 | 10.0% | 2.0x | 30.0% | 6.0x | 0.268 | 0.463 |
| WavLM L9 | 10.0% | 2.0x | 30.0% | 6.0x | 0.315 | 0.495 |
| WavLM L12 | 40.0% | 8.0x | 60.0% | 12.0x | 0.571 | 0.722 |

## Verdict

Go for Priority 2/3 follow-up. WavLM L12 remains well above the 3x chance threshold after train-speaker-only nuisance residualization. Several HuBERT and MERT layers also exceed the threshold in at least one direction.

Important caveat: the acoustic baseline is also 8x chance in both directions after residualization, so this result is not sufficient by itself to claim that SSL-only identity information is isolated from all acoustic identity cues. ECAPA extraction and the larger JVS/JVS-MuSiC run remain the right next checks.

## Outputs

Metrics were written to:

- `experiments/track1_timbre/results/wavlm_l{3,6,9,12}_retrieval_residualized_metrics.json`
- `experiments/track1_timbre/results/hubert_l{3,6,9,12}_retrieval_residualized_metrics.json`
- `experiments/track1_timbre/results/mert_l{3,6,9,12}_retrieval_residualized_metrics.json`
- `experiments/track1_timbre/results/acoustic_retrieval_residualized_metrics.json`

Note: the existing MERT feature cache uses checkpoint hash `m_a_p_mert_v1_95m`, not `m-a-p_mert_v1_95m`.
