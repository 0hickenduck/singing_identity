# Stage 1 Overnight Report

Generated: 2026-06-18T17:10:49

## Run Configuration

- Run root: `/home/bowen/bowen_lab/projects/singing_identity/results/stage1_cuda_allmodels_smoke_probe_run`
- Feature root: `/home/bowen/bowen_lab/projects/singing_identity_stage1_features/cuda_allmodels_smoke_probe`
- Python: `/tmp/singing_identity_cuda_venv/bin/python`
- Device: `cuda`
- Models requested: `acoustic, wavlm, hubert, mert`
- Smoke only: `True`

## What We Wanted To Test

- Track 1: whether cross-mode identity survives speech-to-singing, and whether the residual is global, personalized, or mostly acoustic.
- Track 2: whether technique directions are stable enough across phones/singers to justify later steering.

## Data Used

```json
{
  "fast_no_wav_stats": true,
  "metadata_rows_selected": {
    "Chinese": 2,
    "English": 3,
    "French": 2,
    "German": 2,
    "Italian": 3,
    "Japanese": 2,
    "Korean": 3,
    "Russian": 1,
    "Spanish": 2
  },
  "pairs": {
    "hash": "5760803ceb361aff13f1420348982084aeed3db58811c58c665354ef769a5b36",
    "num_pairs": 10,
    "techniques": {
      "breathy": 10
    }
  },
  "phone_examples": {
    "hash": "4654057e5fdd96ba2f41f56d04a3d399866f600fe055076e6783703dbc4710e4",
    "num_examples": 316,
    "phones": 125,
    "techniques": {
      "breathy": 316
    }
  },
  "root": "/home/bowen/bowen_lab/projects/arti6_linearvc/data/gtsinger_domain_eval",
  "skipped": {
    "missing_wav": 10
  },
  "technique_pairs": {
    "hash": "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
    "num_pairs": 0,
    "target_techniques": {}
  },
  "utterances": {
    "languages": {
      "Chinese": 4,
      "English": 6,
      "French": 4,
      "German": 4,
      "Italian": 2
    },
    "manifest_hash": "a34d27f53fbc04124bae83295e7e12000bcc8ebf51f7f9e56a897c40c66f7ac3",
    "modes": {
      "singing_technique": 10,
      "speech": 10
    },
    "num_speakers": 10,
    "num_utterances": 20,
    "techniques": {
      "breathy": 10,
      "none": 10
    }
  }
}
```

## Job Status

- Successful jobs: 18
- Failed jobs: 0
- Skipped jobs: 0

## Track 1 Results

No Track 1 metrics were produced.

## Track 2 Results

No Track 2 metrics were produced.

## Human Check List

- Check failed optional model logs before interpreting missing model comparisons.
- Confirm any downloaded dataset/checkpoint licenses before thesis reporting.
- For Track 1, inspect whether global residual is consistently competitive across folds and datasets.
- For Track 2, inspect technique groups that pass both bootstrap and analogy tests before any steering.

## Decision

No decision: Stage 1 metrics did not complete.
