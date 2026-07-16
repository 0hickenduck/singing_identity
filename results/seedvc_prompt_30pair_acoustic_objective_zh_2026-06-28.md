# SeedVC Objective Acoustic Evaluation

Manifest: `/localdisk/bowen/singing_identity/runs/track1_seedvc_prompt_baseline_30pairs/listening_manifest.jsonl`
Baseline condition: `target_speech_prompt`
Pairs: 30; conditions: 60

Negative distance delta means the condition is acoustically closer to that reference than baseline.

| condition | d dist target singing | d dist target speech | d dist source | d RMS dB | d F0 mean | d F0 std | d voiced pct | d centroid | d tilt low/high | d high-band |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| target_singing_prompt | -1.4452 | 0.0265 | -0.4280 | -0.3113 | -2.11 | 8.20 | 0.0000 | 74.08 | 3.0202 | -0.0004 |
| target_speech_prompt | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00 | 0.00 | 0.0000 | 0.00 | 0.0000 | 0.0000 |

Interpretation guardrail:

- These are acoustic proxies, not identity labels.
- F0/voicing/spectral tilt/high-band movement can support a phonation/content-delivery interpretation.
- If distance-to-target-singing improves while distance-to-target-speech worsens, the effect may still be domain/phonation rather than identity.
