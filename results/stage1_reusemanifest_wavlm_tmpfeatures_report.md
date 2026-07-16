# Stage 1 Overnight Report

Generated: 2026-06-18T14:46:49

## Run Configuration

- Run root: `/home/bowen/bowen_lab/projects/singing_identity/results/stage1_reusemanifest_wavlm_tmpfeatures_run`
- Feature root: `/tmp/singing_identity_stage1_features/wavlm_reusemanifest`
- Python: `/tmp/singing_identity_ssl_venv/bin/python`
- Device: `cpu`
- Models requested: `wavlm`
- Smoke only: `False`

## What We Wanted To Test

- Track 1: whether cross-mode identity survives speech-to-singing, and whether the residual is global, personalized, or mostly acoustic.
- Track 2: whether technique directions are stable enough across phones/singers to justify later steering.

## Data Used

```json
{
  "fast_no_wav_stats": true,
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
    "manifest_hash": "6a8cb2efb1224f8f8fd427c89b32bb0687d7be2413a22ba6a82e50619efd9e21",
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

- Successful jobs: 31
- Failed jobs: 0
- Skipped jobs: 0

## Track 1 Results

| Model | Layer | Controlled signed AUC | Separability AUC | R@1 | R@5 | Mapper cosine | Global residual cosine |
|---|---:|---:|---:|---:|---:|---:|---:|
| wavlm | 12 | 1.000 | 1.000 | 0.333 | 0.778 | 0.690 | 0.676 |
| wavlm | 3 | 1.000 | 1.000 | 0.222 | 0.500 | 0.715 | 0.781 |
| wavlm | 6 | 1.000 | 1.000 | 0.056 | 0.611 | 0.799 | 0.804 |
| wavlm | 9 | 1.000 | 1.000 | 0.056 | 0.722 | 0.779 | 0.784 |

## Track 2 Results

| Model | Layer | Reliable groups | Passed groups | Best groups |
|---|---:|---:|---:|---|
| wavlm | 12 | 78 | 8 | d_zh/breathy angle=13.4 analogy=0.41, j_zh/breathy angle=19.6 analogy=0.50, c_zh/breathy angle=21.1 analogy=0.61 |
| wavlm | 3 | 78 | 7 | x_zh/breathy angle=18.1 analogy=0.70, d_zh/breathy angle=19.0 analogy=0.46, e_zh/breathy angle=19.0 analogy=0.09 |
| wavlm | 6 | 78 | 9 | d_zh/breathy angle=16.9 analogy=0.52, j_zh/breathy angle=21.1 analogy=0.66, x_zh/breathy angle=22.5 analogy=0.81 |
| wavlm | 9 | 78 | 5 | d_zh/breathy angle=15.5 analogy=0.57, j_zh/breathy angle=22.2 analogy=0.78, z_zh/breathy angle=24.0 analogy=0.70 |

## Human Check List

- Check failed optional model logs before interpreting missing model comparisons.
- Confirm any downloaded dataset/checkpoint licenses before thesis reporting.
- For Track 1, inspect whether global residual is consistently competitive across folds and datasets.
- For Track 2, inspect technique groups that pass both bootstrap and analogy tests before any steering.

## Decision

Use the table thresholds rather than raw AUC alone. Continue Track 1 if cross-mode retrieval and residual baselines hold across folds/models. Continue Track 2 only for techniques that pass both bootstrap stability and analogy retrieval.
