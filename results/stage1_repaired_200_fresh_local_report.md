# Stage 1 Overnight Report

Generated: 2026-06-23T02:09:04

## Run Configuration

- Run root: `/localdisk/bowen/singing_identity/runs/stage1_repaired_200_fresh_local`
- Feature root: `/localdisk/bowen/singing_identity/features/stage1_repaired_200_fresh_local`
- Python: `/tmp/singing_identity_cuda_venv/bin/python`
- Device: `cuda`
- Models requested: `acoustic, wavlm, hubert, mert`
- Smoke only: `False`

## What We Wanted To Test

- Track 1: whether cross-mode identity survives speech-to-singing, and whether the residual is global, personalized, or mostly acoustic.
- Track 2: whether technique directions are stable enough across phones/singers to justify later steering.

## Data Used

```json
{
  "fast_no_wav_stats": false,
  "metadata_rows_selected": {
    "All": 4000,
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
    "hash": "950dfa877d6fe8a7356c7041b2d052810f9acc754c66b88361dc6f3c83bbd926",
    "num_pairs": 4000,
    "techniques": {
      "breathy": 1699,
      "control": 1958,
      "falsetto": 66,
      "glissando": 190,
      "mixed_voice": 62,
      "pharyngeal": 25
    }
  },
  "phone_examples": {
    "hash": "cf0e5a8b4a234d278531b19d45da7de83b04ca51bcdac6b951069680f239017d",
    "num_examples": 336344,
    "phones": 456,
    "techniques": {
      "breathy": 136922,
      "control": 162796,
      "falsetto": 7046,
      "glissando": 20086,
      "mixed_voice": 6370,
      "pharyngeal": 3124
    }
  },
  "root": "/localdisk/bowen/singing_identity/data/gtsinger_domain_eval",
  "skipped": {},
  "technique_pairs": {
    "hash": "a405a320a678e30f1a1e206f3d8cea60d3672d9a7ad05e06390e48097d48295c",
    "num_pairs": 160960,
    "target_techniques": {
      "breathy": 124362,
      "falsetto": 7038,
      "glissando": 20080,
      "mixed_voice": 6358,
      "pharyngeal": 3122
    }
  },
  "utterances": {
    "languages": {
      "Chinese": 608,
      "English": 909,
      "French": 606,
      "German": 602,
      "Italian": 883,
      "Japanese": 607,
      "Korean": 896,
      "Russian": 303,
      "Spanish": 602
    },
    "manifest_hash": "5dcd0fb8ca413763e04fdb9e9464bfd3b6292e68b7451fff0a61024b6c5efce6",
    "modes": {
      "singing_control": 1958,
      "singing_technique": 2042,
      "speech": 2016
    },
    "num_speakers": 20,
    "num_utterances": 6016,
    "techniques": {
      "breathy": 1699,
      "control": 1958,
      "falsetto": 66,
      "glissando": 190,
      "mixed_voice": 62,
      "none": 2016,
      "pharyngeal": 25
    }
  }
}
```

## Job Status

- Successful jobs: 100
- Failed jobs: 0
- Skipped jobs: 0

## Track 1 Results

| Model | Layer | Raw signed AUC | Raw sep AUC | Controlled signed AUC | Controlled sep AUC | Shuffled sep AUC | R@1 | R@5 | Mapper cosine | Global residual cosine |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| acoustic | frame25ms_hop20ms | 0.948 | 0.948 | 0.739 | 0.739 | 0.514 | 0.350 | 0.600 | 0.617 | 0.629 |
| hubert | 12 | 0.999 | 0.999 | 0.792 | 0.792 | 0.512 | 0.300 | 0.650 | 0.729 | 0.637 |
| hubert | 3 | 0.999 | 0.999 | 0.787 | 0.787 | 0.517 | 0.300 | 0.400 | 0.735 | 0.666 |
| hubert | 6 | 1.000 | 1.000 | 0.834 | 0.834 | 0.512 | 0.300 | 0.550 | 0.730 | 0.664 |
| hubert | 9 | 0.999 | 0.999 | 0.796 | 0.796 | 0.508 | 0.200 | 0.550 | 0.736 | 0.660 |
| mert | 12 | 0.997 | 0.997 | 0.826 | 0.826 | 0.513 | 0.100 | 0.400 | 0.718 | 0.543 |
| mert | 3 | 0.999 | 0.999 | 0.820 | 0.820 | 0.519 | 0.200 | 0.500 | 0.649 | 0.537 |
| mert | 6 | 1.000 | 1.000 | 0.852 | 0.852 | 0.514 | 0.200 | 0.550 | 0.671 | 0.564 |
| mert | 9 | 0.999 | 0.999 | 0.802 | 0.802 | 0.519 | 0.250 | 0.450 | 0.699 | 0.574 |
| wavlm | 12 | 0.998 | 0.998 | 0.819 | 0.819 | 0.518 | 0.250 | 0.600 | 0.723 | 0.631 |
| wavlm | 3 | 0.999 | 0.999 | 0.828 | 0.828 | 0.512 | 0.150 | 0.400 | 0.782 | 0.723 |
| wavlm | 6 | 1.000 | 1.000 | 0.902 | 0.902 | 0.513 | 0.050 | 0.450 | 0.790 | 0.738 |
| wavlm | 9 | 0.999 | 0.999 | 0.839 | 0.839 | 0.524 | 0.100 | 0.550 | 0.797 | 0.731 |

## Track 2 Results

| Model | Layer | Reliable groups | Passed groups | Best groups |
|---|---:|---:|---:|---|
| acoustic | frame25ms_hop20ms | 802 | 116 | a_ja/breathy n=1446 spk=2 lang=1 angle=1.7 analogy=0.00 wrong=0.00 shuffled=0.00, UH2_en/breathy n=4 spk=1 lang=1 angle=2.0 analogy=0.50 wrong=0.25 shuffled=0.50, e_es/breathy n=2000 spk=2 lang=1 angle=2.3 analogy=0.00 wrong=0.00 shuffled=0.00 |
| hubert | 12 | 802 | 91 | dː_it/glissando n=4 spk=1 lang=1 angle=5.0 analogy=0.00 wrong=0.25 shuffled=0.25, dː_it/breathy n=4 spk=1 lang=1 angle=5.2 analogy=0.00 wrong=0.25 shuffled=0.25, <AP>/breathy n=2000 spk=7 lang=3 angle=9.2 analogy=0.01 wrong=0.01 shuffled=0.01 |
| hubert | 3 | 802 | 90 | <AP>/breathy n=2000 spk=7 lang=3 angle=8.3 analogy=0.01 wrong=0.01 shuffled=0.01, ɐ_ko/breathy n=2000 spk=3 lang=1 angle=8.6 analogy=0.01 wrong=0.01 shuffled=0.01, dː_it/glissando n=4 spk=1 lang=1 angle=10.2 analogy=0.00 wrong=0.25 shuffled=0.25 |
| hubert | 6 | 802 | 84 | dː_it/glissando n=4 spk=1 lang=1 angle=7.0 analogy=0.00 wrong=0.25 shuffled=0.25, dː_it/breathy n=4 spk=1 lang=1 angle=8.4 analogy=0.00 wrong=0.00 shuffled=0.00, <AP>/breathy n=2000 spk=7 lang=3 angle=8.9 analogy=0.01 wrong=0.01 shuffled=0.01 |
| hubert | 9 | 802 | 67 | dː_it/glissando n=4 spk=1 lang=1 angle=5.2 analogy=0.00 wrong=0.25 shuffled=0.25, dː_it/breathy n=4 spk=1 lang=1 angle=7.2 analogy=0.00 wrong=0.25 shuffled=0.25, <AP>/breathy n=2000 spk=7 lang=3 angle=9.4 analogy=0.01 wrong=0.01 shuffled=0.01 |
| mert | 12 | 802 | 94 | <AP>/breathy n=2000 spk=7 lang=3 angle=7.5 analogy=0.01 wrong=0.01 shuffled=0.01, n_ko/breathy n=1738 spk=3 lang=1 angle=10.2 analogy=0.01 wrong=0.02 shuffled=0.02, ɐ_ko/breathy n=2000 spk=3 lang=1 angle=10.9 analogy=0.01 wrong=0.01 shuffled=0.01 |
| mert | 3 | 802 | 114 | <AP>/breathy n=2000 spk=7 lang=3 angle=7.2 analogy=0.01 wrong=0.01 shuffled=0.01, ɐ_ko/breathy n=2000 spk=3 lang=1 angle=8.6 analogy=0.01 wrong=0.01 shuffled=0.01, n_ko/breathy n=1738 spk=3 lang=1 angle=9.8 analogy=0.01 wrong=0.02 shuffled=0.02 |
| mert | 6 | 802 | 100 | <AP>/breathy n=2000 spk=7 lang=3 angle=8.5 analogy=0.01 wrong=0.01 shuffled=0.01, ɐ_ko/breathy n=2000 spk=3 lang=1 angle=9.3 analogy=0.01 wrong=0.01 shuffled=0.01, dː_it/breathy n=4 spk=1 lang=1 angle=10.3 analogy=0.00 wrong=0.25 shuffled=0.50 |
| mert | 9 | 802 | 90 | <AP>/breathy n=2000 spk=7 lang=3 angle=7.8 analogy=0.01 wrong=0.01 shuffled=0.01, ɐ_ko/breathy n=2000 spk=3 lang=1 angle=10.1 analogy=0.01 wrong=0.01 shuffled=0.01, dː_it/breathy n=4 spk=1 lang=1 angle=11.0 analogy=0.50 wrong=0.50 shuffled=0.50 |
| wavlm | 12 | 802 | 76 | dː_it/breathy n=4 spk=1 lang=1 angle=4.7 analogy=0.00 wrong=0.25 shuffled=0.00, dː_it/glissando n=4 spk=1 lang=1 angle=6.4 analogy=0.00 wrong=0.25 shuffled=0.25, <AP>/breathy n=2000 spk=7 lang=3 angle=10.4 analogy=0.00 wrong=0.00 shuffled=0.01 |
| wavlm | 3 | 802 | 106 | ɐ_ko/breathy n=2000 spk=3 lang=1 angle=8.0 analogy=0.01 wrong=0.01 shuffled=0.01, <AP>/breathy n=2000 spk=7 lang=3 angle=8.5 analogy=0.01 wrong=0.01 shuffled=0.01, dː_it/glissando n=4 spk=1 lang=1 angle=8.8 analogy=0.00 wrong=0.00 shuffled=0.25 |
| wavlm | 6 | 802 | 95 | dː_it/breathy n=4 spk=1 lang=1 angle=7.8 analogy=0.00 wrong=0.25 shuffled=0.25, dː_it/glissando n=4 spk=1 lang=1 angle=8.4 analogy=0.00 wrong=0.25 shuffled=0.25, <AP>/breathy n=2000 spk=7 lang=3 angle=9.6 analogy=0.01 wrong=0.01 shuffled=0.01 |
| wavlm | 9 | 802 | 75 | dː_it/breathy n=4 spk=1 lang=1 angle=5.1 analogy=0.00 wrong=0.25 shuffled=0.25, dː_it/glissando n=4 spk=1 lang=1 angle=6.7 analogy=0.00 wrong=0.25 shuffled=0.25, <AP>/breathy n=2000 spk=7 lang=3 angle=10.0 analogy=0.01 wrong=0.01 shuffled=0.01 |

## Human Check List

- Check failed optional model logs before interpreting missing model comparisons.
- Confirm any downloaded dataset/checkpoint licenses before thesis reporting.
- For Track 1, inspect whether global residual is consistently competitive across folds and datasets.
- For Track 2, inspect technique groups that pass both bootstrap and analogy tests before any steering.

## Decision

Use the table thresholds rather than raw AUC alone. Continue Track 1 if cross-mode retrieval and residual baselines hold across folds/models. Continue Track 2 only for techniques that pass both bootstrap stability and analogy retrieval.
