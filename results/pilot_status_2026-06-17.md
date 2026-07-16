# Two-Experiment Pilot Status - 2026-06-17

## Data Reached

- Dataset used: local GTSinger subset at `/home/bowen/bowen_lab/projects/arti6_linearvc/data/gtsinger_domain_eval`.
- Track 1 manifest: `experiments/track1_timbre/manifests/gtsinger_utterances.jsonl`.
- Track 1 pairs: 39 speech/singing pairs, 18 singers, 78 utterances.
- Track 2 phone examples: 1596 phone examples.
- Track 2 technique pairs: 234 same-singer/same-phone control-vs-technique pairs.
- Missing local wavs skipped: 3961 metadata rows. This is a data-subset limitation, not a code failure.

## Thresholds Used

Track 1 proceeds to Seed-VC only if:

- controlled mode AUC >= 0.75 after F0/energy/duration/voiced/rms residualization;
- cross-mode Recall@1 >= max(0.10, 2x chance);
- learned residual mapper beats global mean residual by at least 0.03 cosine.

Track 2 proceeds to latent steering only if:

- for a target technique, at least 2 reliable phone groups have >=3 pairs and bootstrap mean angle <= 30 degrees;
- median bootstrap angle for that technique/layer is <= 40 degrees;
- result is not only from one singer/language pocket.

## Track 1: Timbre Mode Residual

Stage reached: Stage A complete on WavLM Base+ and acoustic baseline; Stage B ridge residual mapper run as a pilot.

WavLM Base+ summary:

| Layer | Raw mode AUC | Controlled mode AUC | R@1 | R@5 | Mapper cosine | Global residual cosine |
|---|---:|---:|---:|---:|---:|---:|
| 3 | 1.00 | 0.20 | 0.222 | 0.500 | 0.692 | 0.690 |
| 6 | 1.00 | 0.20 | 0.056 | 0.611 | 0.746 | 0.741 |
| 9 | 1.00 | 0.20 | 0.056 | 0.778 | 0.748 | 0.739 |
| 12 | 1.00 | 0.20 | 0.333 | 0.833 | 0.626 | 0.620 |

Machine decision:

- Do not proceed to Seed-VC intervention yet.
- Raw speech/singing mode is trivially decodable, but controlled mode AUC collapses below threshold.
- Cross-mode identity retrieval is promising at layer 12, but the residual mapper only barely beats global residual and does not clear the margin.
- Next machine step should be stronger controls and a cleaner/larger paired subset, not audio generation.

Human check needed:

- Confirm whether this local GTSinger subset is acceptable for the first thesis pilot despite many missing wavs, or whether we should switch to JVS/JVS-MuSiC as originally planned.
- Inspect whether the residualization control is too strong for this small subset. Current result is a conservative stop, not a claim that timbre residual does not exist.

## Track 2: Technique Directions

Stage reached: Stage A direction stability complete on acoustic baseline, WavLM Base+, and MERT v1 95M.

Best reliable WavLM/MERT groups were near but mostly above the 30 degree stability threshold:

- WavLM layer 6: breathy stable 1/12 groups; falsetto stable 1/9; glissando stable 0/3.
- WavLM layer 12: falsetto stable 1/9; breathy and glissando stable 0 groups.
- MERT layers 6/9/12: no technique reached 2 stable phone groups.

Machine decision:

- Do not proceed to latent steering yet.
- Some phone-technique pockets are close, but no technique clears the stated stability gate.
- The current local subset is too sparse and language/singer-pocketed for a credible technique-direction claim.

Human check needed:

- Review top near-stable pockets before discarding the direction: examples include WavLM `o_it` breathy, WavLM `a_it` falsetto, and MERT `o_it`/`ɐ_ko` breathy.
- Decide whether to expand GTSinger download/cache for more control-vs-technique same-phone examples, especially vibrato, falsetto, and glissando.
- Security/reproducibility: MERT was loaded with Hugging Face remote code. Pin a model revision before using these results in a thesis run.

## Immediate Next Step

The highest-value next step is data expansion/cleanup:

1. Build or locate a less sparse JVS/JVS-MuSiC paired set for Track 1.
2. Expand GTSinger local wav availability for Track 2 so control-vs-technique pairs are not concentrated in a few singer/language pockets.
3. Rerun the same gates before any Seed-VC intervention or latent steering.
