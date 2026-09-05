# Track 1 Handoff: Prompt-Mode Mismatch Accounting

Date: 2026-06-28 JST

Purpose: handoff document for the next AI agent. This is the sharpened Track 1 plan after the direction reset discussion. It replaces the vague goal "disentangle singing voice" with a narrower experiment:

> Account for why a same-person speech reference and singing reference are not equivalent in frozen representations and in downstream Seed-VC behavior.

This is not a full disentanglement claim. It is a controlled residual accounting experiment with a downstream sanity check.

## 1. Why This Is Not Just Another Speaker-ID Drop Experiment

Existing work already shows several nearby facts:

- Human cross-modal recognition is harder across speech and singing than within one modality, and F0 differences matter for cross-modal voice identification.
- JVS-MuSiC already reports weak correlation between singing-voice similarity and speech similarity for the same population.
- Singer/speaker recognition papers already ask whether speaker-recognition models transfer from speech to singing.
- SVC/SVS systems already try to separate content, timbre, style, and technique architecturally.
- Recent speech-reference singing systems already use speech prompts for singing generation/conversion.

So our novelty is not:

- "speech and singing timbre are different";
- "speaker-ID accuracy drops on singing";
- "we can classify speech vs singing";
- "we disentangle voice into pure independent factors";
- "we are first to use speech as reference for singing conversion."

The defensible novelty is:

1. **Prompt-mode mismatch accounting**: quantify the representation gap between same-person speech and singing references under same-text paired data, instead of only reporting recognition drop.
2. **Incremental factor accounting**: estimate how much of the speech-to-singing residual is explained by F0/prosody, duration, energy/spectrum, phonation proxies, and technique labels.
3. **Downstream consequence**: test whether the unexplained residual or factor-specific residual predicts Seed-VC speech-prompt vs singing-prompt degradation.
4. **Technique as a moderator**: test whether strong singing technique/phonation compresses identity cues, rather than treating technique as only a style label.

The claim should be:

> Speech-prompted and singing-prompted references differ in frozen singing/speech representations. A large part of this mismatch is explainable by measurable acoustic/prosodic/technique factors, and the remaining residual is useful only if it predicts same-singer retrieval or Seed-VC prompt-mode failures.

## 2. Alignment With Existing Pro Advice And Current Results

This plan is consistent with:

- `pro2`: avoid training large SVC systems; use frozen encoders and low-compute probes.
- `pro3`: the same-person speech/singing identity residual is plausible but must control F0, duration, energy, phones, and technique.
- `pro5`: main direction is same-person speech-to-singing residual; technique is the strongest backup.
- `direction_reset_and_literature_scan_2026-06-25.md`: Stage 1 showed residuals are real, but Seed-VC native components did not give a clean controllable axis; next step should be residual factorization before more synthesis.

Current local evidence already available:

- Stage 1 data and features:
  - Run root: `/localdisk/bowen/singing_identity/runs/stage1_repaired_200_fresh_local`
  - Feature root: `/localdisk/bowen/singing_identity/features/stage1_repaired_200_fresh_local`
  - Pair manifest: `/localdisk/bowen/singing_identity/runs/stage1_repaired_200_fresh_local/manifests/gtsinger_pairs.jsonl`
  - Utterance manifest: `/localdisk/bowen/singing_identity/runs/stage1_repaired_200_fresh_local/manifests/gtsinger_utterances.jsonl`
  - Technique phone pairs: `/localdisk/bowen/singing_identity/runs/stage1_repaired_200_fresh_local/manifests/gtsinger_phoneme_pairs.jsonl`
- Existing features:
  - WavLM Base+ layers 3/6/9/12
  - HuBERT Base layers 3/6/9/12
  - MERT v1 95M layers 3/6/9/12
  - acoustic baseline
- Seed-VC prompt-mode run:
  - `/localdisk/bowen/singing_identity/runs/track1_seedvc_prompt_baseline_10pairs_rr`
  - existing result: singing prompt moved outputs toward target singing in all 10 pairs by Resemblyzer proxy.

## 3. Key Conceptual Distinction

Do not try to answer:

```text
What is singing voice made of?
```

Answer this instead:

```text
Why is target speech reference not equivalent to target singing reference?
```

Operational object:

```text
Delta_z(pair, representation) = z_singing_reference - z_speech_reference
```

The experiment asks:

1. How large is `Delta_z` across representations?
2. Can measurable variables predict `Delta_z` on held-out singers?
3. After subtracting predictable factors, does the remaining residual still encode same-singer identity?
4. Does residual magnitude predict Seed-VC speech-prompt vs singing-prompt gap?
5. Does stronger technique/phonation reduce identity recoverability?

## 4. Data Reality: What Can Be Done Now

The current GTSinger paired manifest has same-singer and same-text speech/singing pairs:

```text
pair_type = same_singer_same_phrase
same_text_flag = true
speech_utt_id
singing_utt_id
technique
speaker_id
song_id
language
```

The current `gtsinger_utterances.jsonl` has `phone_seq` for both speech and singing, but `alignment_path` is empty. Therefore:

- **Do now**: utterance-level same-text residual accounting. Content is controlled by same phrase/text, not by exact phone interval alignment.
- **Do next**: phone-level content-controlled accounting only after speech-side phone intervals are available.
- **Do not claim**: "phone-level speech/singing residual" unless both speech and singing interval boundaries are actually available.

For the near-term experiment, the paired same-text design is already cleaner than arbitrary speech vs singing comparison.

## 5. Experiment A: Utterance-Level Prompt-Mismatch Accounting

### 5.1 Unit

One same-text pair from `gtsinger_pairs.jsonl`:

```text
speech_utt_id -> singing_utt_id
same speaker
same phrase/text
same language
technique label on singing side
```

### 5.2 Representations

Start with four representation families:

```text
wavlm_base_plus / microsoft_wavlm_base_plus / layers 6, 9
hubert_base / facebook_hubert_base_ls960 / layer 12
mert_v1_95m / m_a_p_mert_v1_95m / layers 3, 12
acoustic_baseline / local_wave_v1 / frame25ms_hop20ms
```

Use existing feature-loading helpers in `scripts/research_utils.py`. Do not re-extract features unless validation fails.

### 5.3 Target

For each pair and representation:

```text
z_speech = pooled feature vector for speech_utt_id
z_sing = pooled feature vector for singing_utt_id
y = Delta_z = z_sing - z_speech
```

Pooling should match the existing Stage 1 convention unless there is a strong reason to change it. Use mean/std or current `load_feature_vector` behavior for consistency.

### 5.4 Covariate Groups

Construct pair-level delta covariates:

```text
content/text:
  same_text_flag
  phone_seq histogram or phone class histogram if easy
  text length / phone count

prosody/F0:
  delta_f0_mean
  delta_f0_std
  delta_f0_range
  delta_f0_voiced_pct

timing:
  duration_ratio = singing_duration / speech_duration
  delta_duration_sec
  phone_rate_proxy if phone_seq exists

energy:
  delta_energy_mean
  delta_energy_std
  delta_rms_db

spectrum/phonation proxy:
  use acoustic_baseline deltas first
  if available later: spectral centroid, rolloff, spectral tilt, HNR, CPP

technique:
  one-hot singing technique label
  breathy/control/vibrato/glissando/falsetto/mixed/pharyngeal
  technique strength proxy if available later

metadata controls:
  language
  vocal_range
  song_id, only for grouped split/checking, not as a free predictor unless justified
```

Important: technique is not a magic independent factor. It is a label/proxy. Report it as "label-associated explanation", not causal decomposition.

### 5.5 Models

Use nested multi-output ridge regression. Train only on train speakers; evaluate on held-out speakers.

```text
M0: global mean Delta_z only
M1: content/text proxies
M2: M1 + F0/prosody
M3: M2 + timing + energy
M4: M3 + spectrum/phonation proxy
M5: M4 + technique labels
```

For each model:

```text
pred_delta = model(X_pair)
z_pred_singing = z_speech + pred_delta
residual_delta = true_delta - pred_delta
```

### 5.6 Metrics

Report per representation/layer:

```text
delta_cosine = cos(pred_delta, true_delta)
delta_mse_reduction vs M0
residual_norm_reduction vs raw Delta_z
cos(z_speech + pred_delta, z_sing)
same-singer retrieval R@1/R@5 before and after predicted delta
```

Also report factor increments:

```text
M2 - M1: added value of F0/prosody
M3 - M2: added value of timing/energy
M4 - M3: added value of spectrum/phonation
M5 - M4: added value of technique labels
```

The main table is not "content is 90%". It is:

```text
Which factors explain prompt-mode mismatch on held-out singers,
and which representation spaces retain unexplained but stable residual?
```

### 5.7 Negative Controls

Mandatory:

```text
shuffle singing_utt_id within speaker
shuffle technique label
shuffle speaker_id across pairs
F0/duration/energy-only model
metadata-only model
acoustic baseline as shortcut detector
```

If the acoustic baseline explains the same amount as SSL representations, be cautious: the result may be mostly low-level acoustics, not identity/style.

### 5.8 Downstream Link To Seed-VC

Use existing Seed-VC prompt baseline:

```text
/localdisk/bowen/singing_identity/runs/track1_seedvc_prompt_baseline_10pairs_rr
```

For pairs with Seed-VC outputs:

```text
prompt_gap = sim(output_singing_prompt, target_singing)
           - sim(output_speech_prompt, target_singing)
```

Then correlate:

```text
raw_delta_norm vs prompt_gap
M5_residual_norm vs prompt_gap
technique-associated predicted_delta_norm vs prompt_gap
phonation/acoustic-associated predicted_delta_norm vs prompt_gap
```

This is the "so what" test. If residual accounting does not predict any downstream gap, the Track 1 story is weaker.

## 6. Experiment B: Technique-Induced Identity Contraction

This can be folded into Track 1 as a moderator analysis.

Hypothesis:

> Strong technique/style can pull voices toward a shared style manifold, reducing recoverable individual identity cues.

Operational tests:

```text
For each technique condition:
  compute singer retrieval R@1/R@5 within that condition
  compute between-singer distance
  compute within-singer variance
  compute technique classifier accuracy
```

Expected pattern if identity contraction exists:

```text
control singing:
  higher singer retrieval, higher between-singer separation

strong technique, e.g. breathy/vibrato/glissando/opera-like if data exists:
  lower singer retrieval, lower between-singer separation
  higher technique separability
```

Current available data supports testing this for GTSinger techniques, not true opera/professional classical singing. The opera idea should be framed as a future extension unless SVQTD/VocalSet/classical data are added.

## 7. Experiment C: Phone-Level Version Later

Once speech-side phone intervals are available, run a stricter version:

```text
mu[singer, mode, phone] = average latent over intervals for that phone
Delta[singer, phone] = mu[singer, singing, phone] - mu[singer, speech, phone]
```

Then repeat the accounting:

```text
Delta[singer, phone] ~ F0 + phone_duration + energy + phonation + technique
```

This is cleaner content control, but it is not the first runnable experiment because current speech-side interval boundaries are not guaranteed.

## 8. Immediate Run Queue

### Priority 0: For Tomorrow's Technique 1:1

Run or refresh the breathy technique-direction evidence first, because tomorrow's meeting is about technique.

Recommended immediate task for the next AI:

1. Build a filtered breathy manifest from:

```text
/localdisk/bowen/singing_identity/runs/stage1_repaired_200_fresh_local/manifests/gtsinger_phoneme_pairs.jsonl
```

Filter:

```text
target_technique == "breathy"
phone is vowel or voiced sonorant
exclude <AP>, <SP>, silence-like phones
minimum group size per phone >= 5 if possible
```

2. Run `scripts/probing/run_technique_directions.py` on:

```text
mert_v1_95m layer 3
wavlm_base_plus layer 6
wavlm_base_plus layer 9
```

3. Prepare a one-page readout:

```text
top phone-technique groups by bootstrap angle
analogy_top1 vs wrong_technique_top1 vs shuffled_direction_top1
whether breathy looks real or mostly sparse/noisy
```

Do not use `<AP>` or sparse phone groups as headline evidence.

### Priority 1: Track 1 Accounting Smoke

If there is time today, implement the first runnable smoke for Experiment A:

```text
scripts/probing/run_prompt_mismatch_accounting.py
```

Minimal version:

```text
input:
  --pairs gtsinger_pairs.jsonl
  --utterances gtsinger_utterances.jsonl
  --feature-root /localdisk/bowen/singing_identity/features/stage1_repaired_200_fresh_local
  --extractor wavlm_base_plus
  --checkpoint-hash microsoft_wavlm_base_plus
  --layer 6

output:
  metrics JSON
  pair-level predictions table
  short markdown summary
```

Run only WavLM L6 first. Then extend to HuBERT/MERT/acoustic after the script is validated.

## 9. Implementation Contract For Next AI

### New script to create

```text
/home/bowen/bowen_lab/projects/singing_identity/scripts/probing/run_prompt_mismatch_accounting.py
```

### Must reuse

```text
scripts/research_utils.py
read_table
write_json
write_table
load_feature_vector
```

### Must not do

```text
do not re-extract all features
do not train base models
do not use same speaker in train and test
do not claim disentanglement
do not report only random splits
do not headline technique labels without shuffle controls
```

### Split

Use speaker-disjoint folds:

```text
train speakers: 70%
validation speakers: 15%
test speakers: 15%
```

If singer count is too small in a subset, use leave-one-speaker-out or repeated grouped K-fold.

### Output root

Use:

```text
/localdisk/bowen/singing_identity/runs/track1_prompt_mismatch_accounting_2026-06-28
```

Compact report should go to:

```text
/home/bowen/bowen_lab/projects/singing_identity/results/track1_prompt_mismatch_accounting_2026-06-28.md
```

## 10. Interpretation Rules

Continue Track 1 if:

```text
held-out delta cosine improves over global mean baseline
predicted delta improves speech->singing retrieval or cosine
residual/factor magnitude correlates with Seed-VC prompt gap
```

Narrow Track 1 if:

```text
only acoustic/prosody factors explain the gap
technique labels add no value
downstream correlation is weak
```

Pivot toward Track 2 if:

```text
Track 1 residual accounting is mostly intuitive and does not predict downstream behavior
breathy/vector-arithmetic evidence is cleaner and more controllable
```

A negative but useful Track 1 result:

> The apparent same-person speech-to-singing identity residual in frozen representations is mostly explained by measurable prosody/acoustic/phonation factors and does not reliably predict downstream prompt-mode degradation.

This is publishable only if controls are strong and the conclusion is framed as a diagnostic warning against overinterpreting timbre embeddings.

## 11. Sources And Prior Anchors

- JVS-MuSiC reports weak correlation between singing-voice similarity and speech similarity for the same 100-speaker population: https://arxiv.org/abs/2001.07044
- Human voice recognition from spoken vs sung speech: cross-modal recognition is harder, and F0 differences matter: https://pubs.aip.org/asa/jel/article/4/6/065203/3299000/Who-is-singing-Voice-recognition-from-spoken
- SVCC 2025 moved from singer identity conversion to singing style conversion and found breathy/glissando/vibrato style modeling remains hard: https://arxiv.org/html/2509.15629v1
- Everyone-Can-Sing already uses speech reference for zero-shot singing generation/conversion, so do not claim speech-reference SVC novelty: https://arxiv.org/html/2501.13870v1
- S2Voice separates style conditioning and speaker/timbre conditioning in a winning SVCC 2025 system, supporting our motivation that global timbre is not enough: https://arxiv.org/html/2601.13629v1

