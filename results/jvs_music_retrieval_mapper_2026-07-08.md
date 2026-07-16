# JVS+JVS-MuSiC Retrieval and Mapper

Date: 2026-07-08

## Question

Do residualized cross-mode identity retrieval and centroid residual mappers hold up on a 100-speaker JVS+JVS-MuSiC setting?

## Run

- run_root: `/localdisk/bowen/singing_identity/runs/jvs_music_retrieval_2026-07-08`
- feature_root: `/localdisk/bowen/singing_identity/features/jvs_music_2026-07-08`
- manifest: `/localdisk/bowen/singing_identity/runs/jvs_music_retrieval_2026-07-08/manifests/jvs_music_utterances.jsonl`
- speakers: 100
- utterances: 10061 (`speech`: 9961, `singing`: 100)
- chance R@1: 1.0%
- script: `scripts/run_jvs_music_priority3_2026_07_08.sh`
- mapper script: `scripts/probing/run_jvs_centroid_residual_predictor.py`

`/localdisk/bowen/singing_identity/data/jvs_ver1` was completed from the lab NFS copy using `cpz` before feature extraction. Heavy reads/writes and caches stayed under `/localdisk`.

Preflight was recorded in:

`/localdisk/bowen/singing_identity/runs/jvs_music_retrieval_2026-07-08/preflight_priority3.log`

## Retrieval Results

| Condition | Raw S->G R@1 | Raw G->S R@1 | Resid S->G R@1 | Resid G->S R@1 | Resid S->G mAP | Resid G->S mAP |
|---|---:|---:|---:|---:|---:|---:|
| acoustic | 3% | 5% | 7% (7x) | 9% (9x) | 0.146 | 0.149 |
| ecapa | 96% | 96% | 97% (97x) | 95% (95x) | 0.983 | 0.971 |
| wavlm_l3 | 8% | 9% | 42% (42x) | 32% (32x) | 0.520 | 0.416 |
| wavlm_l6 | 5% | 7% | 17% (17x) | 13% (13x) | 0.310 | 0.255 |
| wavlm_l9 | 1% | 2% | 10% (10x) | 7% (7x) | 0.211 | 0.162 |
| wavlm_l12 | 4% | 4% | 16% (16x) | 13% (13x) | 0.277 | 0.233 |
| hubert_l3 | 4% | 13% | 43% (43x) | 34% (34x) | 0.538 | 0.447 |
| hubert_l6 | 5% | 10% | 43% (43x) | 17% (17x) | 0.503 | 0.314 |
| hubert_l9 | 4% | 10% | 28% (28x) | 15% (15x) | 0.391 | 0.256 |
| hubert_l12 | 2% | 6% | 24% (24x) | 13% (13x) | 0.344 | 0.214 |
| mert_l3 | 15% | 15% | 54% (54x) | 46% (46x) | 0.623 | 0.543 |
| mert_l6 | 10% | 9% | 52% (52x) | 31% (31x) | 0.602 | 0.430 |
| mert_l9 | 5% | 11% | 40% (40x) | 24% (24x) | 0.510 | 0.356 |
| mert_l12 | 6% | 10% | 32% (32x) | 19% (19x) | 0.431 | 0.304 |

Mode probes stayed highly separable after nuisance residualization: most SSL residualized mode probes reached separability AUC 1.000, while acoustic reached 0.984 and ECAPA reached 0.963. This means identity retrieval survives the nuisance controls, but speech-vs-singing domain information is still very detectable.

## Mapper Results

Each mapper run used the deterministic 70/15/15 speaker split from `deterministic_speaker_split`, so the test set has 15 speakers.

| Representation | Best delta MSE reduction | Best model | Best pred R@1 | Speech-only R@1 | Global R@1 |
|---|---:|---|---:|---:|---:|
| acoustic | 0.794 | speech_embedding_ridge | 27% | 33% | 27% |
| ecapa | 0.239 | speech_embedding_ridge | 100% | 100% | 100% |
| wavlm_l3 | 0.840 | speech_embedding_ridge | 80% | 13% | 60% |
| hubert_l3 | 0.840 | speech_embedding_ridge | 87% | 7% | 80% |
| hubert_l6 | 0.838 | speech_embedding_ridge | 80% | 7% | 73% |
| mert_l3 | 0.734 | speech_embedding_ridge | 80% | 53% | 87% |
| mert_l6 | 0.789 | speech_embedding_ridge | 73% | 40% | 80% |

## Verdict

The JVS expansion is a clear Go for mapper work. Retrieval remains far above chance on 100 speakers, especially MERT L3/L6, HuBERT L3/L6, WavLM L3, and ECAPA. The mapper baseline suite is now meaningful with 70 train / 15 dev / 15 test speakers.

The strongest mapper takeaway is that `speech_embedding_ridge` consistently reduces residual MSE for SSL representations. Retrieval R@1 after mapping is strongest for HuBERT L3 (87%) and WavLM L3 / HuBERT L6 / MERT L3 (80%), but global residual is also competitive for MERT and HuBERT. Acoustic metadata ridge is unstable and often worse than speech-only.

## Outputs

- Retrieval/mode metrics: `experiments/track1_timbre/results/jvs_music_2026-07-08/*.json`
- Prediction JSONL files: `/localdisk/bowen/singing_identity/runs/jvs_music_retrieval_2026-07-08/predictions`
- Mapper outputs: `/localdisk/bowen/singing_identity/runs/jvs_music_retrieval_2026-07-08/mapper`
- Feature caches: `/localdisk/bowen/singing_identity/features/jvs_music_2026-07-08`
