# Stage 1 Overnight Report

Generated: 2026-06-18T00:48:32

## Run Configuration

- Run root: `/home/bowen/bowen_lab/projects/singing_identity/results/stage1_overnight_robust_partial_run`
- Feature root: `/home/bowen/bowen_lab/projects/singing_identity/experiments/shared_features`
- Python: `/home/bowen/bowen_lab/projects/arti6_linearvc/.venv/bin/python`
- Device: `cuda`
- Models requested: `acoustic, wavlm, hubert, mert`
- Smoke only: `False`

## What We Wanted To Test

- Track 1: whether cross-mode identity survives speech-to-singing, and whether the residual is global, personalized, or mostly acoustic.
- Track 2: whether technique directions are stable enough across phones/singers to justify later steering.

## Data Used

```json
{
  "metadata_rows_selected": {
    "Chinese": 400,
    "English": 600,
    "French": 400,
    "German": 400,
    "Italian": 600,
    "Japanese": 400,
    "Korean": 600,
    "Russian": 200,
    "Spanish": 400
  },
  "pairs": {
    "hash": "1abe4320dca21713983f89e1db041b0362170a28254fa0a39ff69dc2c108ce40",
    "num_pairs": 238,
    "techniques": {
      "breathy": 128,
      "control": 106,
      "falsetto": 2,
      "glissando": 2
    }
  },
  "phone_examples": {
    "hash": "8fe79bdbc8d0650e754e8be34c630d4aa5c28a2c3216a05d2eea2935e9c3f735",
    "num_examples": 7673,
    "phones": 296,
    "techniques": {
      "breathy": 4136,
      "control": 3342,
      "falsetto": 99,
      "glissando": 96
    }
  },
  "root": "/home/bowen/bowen_lab/projects/arti6_linearvc/data/gtsinger_domain_eval",
  "skipped": {
    "missing_wav": 3762
  },
  "technique_pairs": {
    "hash": "307a4f301a032d00eac2d136b656b76fb10caed86198b88824f0a7a174d0fdc6",
    "num_pairs": 3213,
    "target_techniques": {
      "breathy": 3112,
      "falsetto": 76,
      "glissando": 25
    }
  },
  "utterances": {
    "languages": {
      "Chinese": 304,
      "English": 12,
      "French": 12,
      "German": 8,
      "Italian": 20,
      "Japanese": 2,
      "Korean": 10,
      "Russian": 4,
      "Spanish": 6
    },
    "manifest_hash": "ffd3ac8a7c7906a48a06d8f7831698794487de003cbc9154bb1dc20272a43931",
    "modes": {
      "singing_control": 106,
      "singing_technique": 132,
      "speech": 140
    },
    "num_speakers": 18,
    "num_utterances": 378,
    "techniques": {
      "breathy": 128,
      "control": 106,
      "falsetto": 2,
      "glissando": 2,
      "none": 140
    }
  }
}
```

## Job Status

- Successful jobs: 10
- Failed jobs: 3
- Skipped jobs: 21

Failed jobs:
- `smoke_extract_hubert` returncode=-15 log=`/home/bowen/bowen_lab/projects/singing_identity/results/stage1_overnight_robust_partial_run/jobs/smoke_extract_hubert/run.log`
- `smoke_extract_mert` returncode=-15 log=`/home/bowen/bowen_lab/projects/singing_identity/results/stage1_overnight_robust_partial_run/jobs/smoke_extract_mert/run.log`
- `smoke_extract_wavlm` returncode=-15 log=`/home/bowen/bowen_lab/projects/singing_identity/results/stage1_overnight_robust_partial_run/jobs/smoke_extract_wavlm/run.log`

## Track 1 Results

| Model | Layer | Controlled signed AUC | Separability AUC | R@1 | R@5 | Mapper cosine | Global residual cosine |
|---|---:|---:|---:|---:|---:|---:|---:|
| acoustic | frame25ms_hop20ms | 0.678 | 0.685 | 0.111 | 0.389 | 0.300 | 0.573 |

## Track 2 Results

| Model | Layer | Reliable groups | Passed groups | Best groups |
|---|---:|---:|---:|---|
| acoustic | frame25ms_hop20ms | 78 | 15 | q_zh/breathy angle=7.9 analogy=0.19, i_zh/breathy angle=9.6 analogy=0.00, o_it/glissando angle=9.7 analogy=0.33 |

## Human Check List

- Check failed optional model logs before interpreting missing model comparisons.
- Confirm any downloaded dataset/checkpoint licenses before thesis reporting.
- For Track 1, inspect whether global residual is consistently competitive across folds and datasets.
- For Track 2, inspect technique groups that pass both bootstrap and analogy tests before any steering.

## Decision

Use the table thresholds rather than raw AUC alone. Continue Track 1 if cross-mode retrieval and residual baselines hold across folds/models. Continue Track 2 only for techniques that pass both bootstrap stability and analogy retrieval.
