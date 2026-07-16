# SeedVC Objective Acoustic Evaluation

Manifest: `/localdisk/bowen/singing_identity/runs/track1_seedvc_component_ablation_12pairs/listening_manifest.jsonl`
Baseline condition: `baseline_all_speech`
Pairs: 12; conditions: 96

Negative distance delta means the condition is acoustically closer to that reference than baseline.

| condition | d dist target singing | d dist target speech | d dist source | d RMS dB | d F0 mean | d F0 std | d voiced pct | d centroid | d tilt low/high | d high-band |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_all_speech | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00 | 0.00 | 0.0000 | 0.00 | 0.0000 | 0.0000 |
| oracle_all_singing | -1.3753 | 0.2744 | -0.1049 | -0.6148 | -3.05 | 2.30 | 0.0000 | 25.59 | 3.8002 | -0.0034 |
| singing_mel_only | -0.1763 | 0.3080 | 0.0435 | 1.8593 | -1.79 | -3.27 | 0.0000 | 134.54 | -0.4249 | 0.0071 |
| singing_mel_style | -1.2213 | 0.2823 | -0.1453 | -0.5610 | -2.06 | 0.54 | 0.0000 | 105.64 | 3.1187 | 0.0040 |
| singing_prompt_seq_mel | -0.4119 | 0.3785 | 0.0499 | 1.0811 | 0.36 | -0.19 | 0.0000 | 57.06 | 0.9982 | -0.0014 |
| singing_prompt_seq_only | 0.1584 | 0.0996 | 0.4908 | 2.0020 | -0.97 | 1.59 | 0.0000 | 81.86 | -0.8113 | 0.0053 |
| singing_prompt_seq_style | -0.6420 | -0.1177 | -0.2575 | -0.7771 | -3.63 | -3.22 | 0.0000 | 7.35 | 2.5906 | 0.0022 |
| singing_style_only | -0.6789 | 0.0232 | -0.6660 | -2.4035 | -8.28 | -13.95 | 0.0000 | -44.15 | 2.4167 | -0.0026 |

Interpretation guardrail:

- These are acoustic proxies, not identity labels.
- F0/voicing/spectral tilt/high-band movement can support a phonation/content-delivery interpretation.
- If distance-to-target-singing improves while distance-to-target-speech worsens, the effect may still be domain/phonation rather than identity.
