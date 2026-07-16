# Track 1 timbre experiment: Pro-style residual predictor design

Date: 2026-06-29

## 0. Current decision

Track 1 should return to the Pro design:

```text
target speech identity/timbre representation
    -> predicted singing identity/timbre representation
```

or equivalently:

```text
z_sing_hat = z_speech + delta_hat
delta_hat = A(speech-side information)
```

This is different from the recent GTSinger prompt-mismatch accounting experiment. That experiment asked which observable covariate blocks help predict `Delta_z = z_singing - z_speech`; it is useful as diagnostic accounting, but it is not yet the main deployable model.

The new main question is:

> Given only a target person's speech reference, can we estimate the identity/timbre representation that would be more appropriate for using that person as a singing voice target?

This is the version that connects directly to speech-reference SVC and to Pro's residual-adapter idea.

## 1. Important terminology correction

The phrase "use Speaker ID to predict Singer ID" should not mean using a categorical speaker label as the only input.

If the model input is just a discrete training speaker ID, then under held-out-singer evaluation the model cannot infer a new speaker. It would mostly memorize training speakers and would not be deployable.

The operational definition should be:

```text
input:
  continuous representation extracted from target speech audio
  + optional speech-side acoustic statistics
  + optional metadata available at deployment

output:
  predicted same-person singing representation
  or predicted speech-to-singing residual
```

So the real formulation is:

```text
A: (z_speech, speech_acoustic_stats, optional deployable priors)
   -> delta_hat

z_sing_hat = z_speech + delta_hat
```

The categorical speaker ID can be used only for grouping, split construction, centroid computation, and evaluation labels. It should not be the main deployable input.

## 2. Representation choice

The first representation space should match the downstream intervention target.

Priority order:

1. SeedVC native reference/timbre embedding, if we can cleanly extract and re-inject it.
2. A strong speaker-verification embedding such as ECAPA/CAMPPlus for analysis and objective identity metrics.
3. WavLM/HuBERT/ContentVec layer statistics for representation audit.
4. Acoustic statistics such as F0, energy, voiced ratio, duration, and simple mel statistics as control baselines, not as overclaimed "timbre" factors.

The cleanest paper logic is:

```text
analysis space:
  multiple frozen encoders show whether the residual is measurable

intervention space:
  SeedVC reference/timbre embedding tests whether the residual helps generation
```

If the analysis embedding and SeedVC embedding have different dimensions, do not force a single residual across spaces. Use separate adapters per representation space, or add an explicit projection head:

```text
delta_hat_seedvc = A_seedvc(z_speech_seedvc, stats)
delta_hat_ssl    = A_ssl(z_speech_ssl, stats)
```

Dimension matching should be treated as an implementation detail inside one representation family, not as evidence that the spaces are semantically identical.

## 3. Dataset decision

GTSinger is not ideal as the main Track 1 dataset because the number of same-person speech/singing identities is small for a held-out-singer residual predictor.

For timbre/identity residual, we do not require technique labels. Therefore JVS + JVS-MuSiC is a better first choice if available.

Local status checked:

```text
/work/smcintosh/data/jvs_ver1
```

exists and appears to contain JVS speech data:

```text
jvsXXX/parallel100/wav24kHz16bit
jvsXXX/nonpara30/wav24kHz16bit
jvsXXX/whisper10/wav24kHz16bit
jvsXXX/falset10/wav24kHz16bit
```

JVS-MuSiC status checked after user-provided path:

```text
/work/sora/ground_truth/DTW/jvs_music_ver1.zip
```

This zip exists and has been extracted locally for repeated feature extraction:

```text
/localdisk/bowen/singing_identity/data/jvs_music_ver1
```

This local extracted directory contains:

```text
jvs_music_ver1/jvsXXX/song_common/wav/raw.wav
jvs_music_ver1/jvsXXX/song_common/wav/modified.wav
jvs_music_ver1/jvsXXX/song_common/wav/modified_grouped.wav
jvs_music_ver1/jvsXXX/song_unique/wav/raw.wav
```

Audit result:

```text
JVS speech speakers:     100
JVS-MuSiC singers:       100
ID mismatch:             none
singing wavs per singer: 4
total singing wavs:      400
local extracted size:    726M
song_common/raw.wav:     100 files
```

The zip also contains:

```text
README.txt
singer_info.txt
similarity/*.csv
oneness/*.csv
paper.pdf
```

The README states that each singer's reading voice is in JVS, which matches the ID audit. `singer_info.txt` includes gender, key group, tempo, key, and unique song name. This metadata can be used for stratification and control baselines, but not as target-singing information in deployable claims unless it is available at inference.

Decisions after human feedback:

```text
extract/cache location:
  /localdisk/bowen/singing_identity/data/jvs_music_ver1

usage:
  research use; do not make license handling a blocker for running the experiment

first-pass singing condition:
  use song_common/wav/raw.wav only

later robustness conditions:
  song_common/wav/modified.wav
  song_common/wav/modified_grouped.wav
  song_unique/wav/raw.wav
```

JVS `falset10` can be used as a small mode-shift sanity check, but it should not replace true singing data in the main claim.

## 4. Core data construction

For each speaker `s`:

```text
speech clips:
  JVS parallel100 and/or nonpara30

singing clips:
  first pass: JVS-MuSiC song_common/wav/raw.wav
  later robustness: modified, modified_grouped, and song_unique
```

For each representation encoder `h`:

```text
z_speech_s = mean_i h(speech_clip_s,i)
z_sing_s   = mean_j h(singing_clip_s,j)
delta_s    = z_sing_s - z_speech_s
```

This centroid residual is the first version because it is simple, auditable, and less likely to overfit than phone-conditioned or contrastive decompositions.

If enough aligned or phone-level material is available later, a second version can compute phone-conditioned residuals:

```text
delta_s,p = mean h(singing phone p) - mean h(speech phone p)
```

but that is not the first implementation.

## 5. Split rule

The main split must be held-out singer:

```text
train speakers: used to learn global residual and adapter
dev speakers: tune model dimension, regularization, alpha
test speakers: never seen during model fitting
```

No clip from the same speaker should appear in both train and test for the deployable residual claim.

Use `song_common/wav/raw.wav` for the first pass because all singers share the same song and the condition is easiest to explain. Use `modified.wav`, `modified_grouped.wav`, and `song_unique/raw.wav` only after the first residual-predictor baseline is working.

## 6. Models to train

Start with low-capacity models before any high-dimensional neural adapter.

Recommended order:

1. No model:

```text
z_sing_hat = z_speech
```

2. Global residual:

```text
delta_global = mean_train_s(delta_s)
z_sing_hat = z_speech + delta_global
```

3. Acoustic-only predictor:

```text
delta_hat = Ridge(F0_stats, energy_stats, voiced_ratio, duration_stats)
```

This is a control for "we only predicted pitch/loudness/mode acoustics."

4. Linear / ridge residual predictor:

```text
delta_hat = Ridge(z_speech, speech_acoustic_stats)
```

5. Low-rank affine adapter:

```text
delta_hat = U V [z_speech; stats]
```

where rank is tuned on dev speakers.

6. Small MLP / MicroMapper:

```text
delta_hat = MLP([z_speech; stats])
```

Use only after the linear and low-rank versions are established. The MLP must be small because 100 speakers is still a small-sample regime.

## 7. F0 and singing/speech bias handling

F0 is a central confound. Singing differs from speech partly because it has different pitch range, pitch stability, duration, and energy. We should not overclaim that a residual is "timbre" if it is mostly F0/loudness/duration.

Deployable rule:

```text
Allowed:
  target speech F0/energy/duration statistics
  source singing melody if the downstream SVC task already provides source singing
  training-set population statistics

Not allowed for deployable claims:
  target singer's singing F0
  target singer's singing clips
  target singer-specific singing metadata unavailable at inference
```

Evaluation should report at least:

```text
main residual performance
residual performance after controlling or stratifying by speech F0 range
acoustic-only baseline performance
```

If acoustic-only features match the embedding-based adapter, the identity-residual story is weak. If the adapter beats acoustic-only and global residual under held-out-singer splits, the result is stronger.

## 8. Mandatory baselines

Representation-level baselines:

| Name | Definition | Purpose |
|---|---|---|
| speech only | `z_speech` | main no-adaptation baseline |
| global residual | `z_speech + mean_train(delta)` | simplest residual hypothesis |
| acoustic-only | predict residual from F0/energy/duration only | checks acoustic shortcut |
| wrong-speaker residual | add another speaker's residual | sanity check |
| random residual | magnitude-matched random direction | sanity check |
| oracle residual | `z_speech + (z_sing - z_speech)` | upper bound, not deployable |
| learned deployable residual | `z_speech + A(z_speech, stats)` | proposed method |

Downstream SeedVC baselines:

| Name | Reference conditioning | Claim type |
|---|---|---|
| speech reference | unmodified target speech reference | baseline |
| speech + global residual | adapted speech reference | simple residual baseline |
| speech + learned residual | adapted speech reference | proposed deployable method |
| speech + wrong/random residual | adapted with invalid residual | sanity check |
| singing reference | target singing reference | oracle upper bound |

## 9. Metrics

Representation-level metrics:

```text
MSE(delta_hat, delta_oracle)
cosine(delta_hat, delta_oracle)
cosine(z_sing_hat, z_sing_oracle)
same-person cross-mode retrieval Recall@1 / Recall@5
gap closed:
  (metric_deployable - metric_speech_baseline)
  / (metric_oracle - metric_speech_baseline)
```

Downstream audio metrics:

```text
target identity similarity proxy
source content / melody preservation proxy
naturalness or artifact proxy if available
objective metrics stratified by F0 range and gender if metadata is reliable
```

Human listening should come after objective sanity checks. The listening task should compare only meaningful conditions:

```text
speech reference baseline
learned residual
global residual
singing reference oracle
```

## 10. What this can and cannot claim

Safe claim if successful:

> A speech-derived identity representation can be adapted toward the same person's singing-reference representation under held-out-singer evaluation, and this adaptation improves over speech-only, global-residual, acoustic-only, and wrong/random residual baselines.

Stronger claim only if SeedVC intervention succeeds:

> The predicted residual is not only decodable in representation space; it improves speech-reference singing voice conversion when injected into a frozen downstream system.

Unsafe claims:

```text
We fully decompose timbre.
We recover causal singer identity.
The residual is pure timbre.
Speaker ID alone predicts singer ID for unseen people.
Oracle residual is a deployable method.
```

## 11. Relation to our previous work

Recent Track 1 prompt-mismatch accounting:

```text
useful for diagnostic accounting
not the main Pro-style residual predictor
does not by itself prove deployable speech-to-singing adaptation
```

Recent SeedVC prompt/component ablations:

```text
useful for understanding whether SeedVC reacts to speech vs singing prompts
not yet a proper residual-adapter downstream evaluation
```

Correct future SeedVC experiment:

```text
freeze SeedVC
change only reference/timbre conditioning
compare speech baseline, global residual, learned residual, wrong/random residual, singing oracle
```

Track 2 breathy work:

```text
separate from this timbre residual track
should not constrain Track 1 dataset choice unless technique labels become central again
```

## 12. Immediate next actions

1. Build a JVS/JVS-MuSiC manifest with speaker-disjoint train/dev/test splits, using:

```text
speech root:
  /work/smcintosh/data/jvs_ver1

singing root:
  /localdisk/bowen/singing_identity/data/jvs_music_ver1

first-pass singing file:
  song_common/wav/raw.wav
```

2. Extract a small first set of embeddings:
```text
SeedVC reference/timbre embedding if accessible
ECAPA or CAMPPlus
WavLM mid/high layer statistics
basic acoustic stats
```

3. Run representation-level residual baselines before any SeedVC audio generation:

```text
speech only
global residual
acoustic-only
linear/ridge adapter
low-rank adapter
oracle
wrong/random controls
```

4. Only if the representation-level result clears baselines, implement the frozen SeedVC reference-conditioning intervention.

## 13. Human check status before main experiment

Resolved:

1. JVS-MuSiC should be extracted/cached on localdisk if space is sufficient.
2. `/localdisk` has enough space, and the corpus has been extracted.
3. Generated examples are for research use, so license handling is not a blocker for running the experiment.
4. First pass should use `song_common/wav/raw.wav`.

Still useful to discuss with teacher:

1. Should the first teacher-facing claim be framed as:

```text
controlled same-person speech-to-singing residual prediction
```

or as:

```text
speech-reference SVC improvement via residual-adapted target conditioning
```

The first claim is easier to validate. The second is stronger but depends on the SeedVC intervention actually working.
