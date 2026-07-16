I am considering one specific paper/thesis idea. Please evaluate it as if you
were both a harsh reviewer and a practical advisor.

Context:

I am studying same-person speech-vs-singing timbre or vocal identity shift. The
intuition is that a speaker/singer does not have one static speaker identity
embedding. The same person may have a stable identity core plus a vocal-mode
specific residual: their speaking voice and singing voice are related, but not
identical. If we can model that residual, it may help singing voice conversion
when we only have target speech reference and no target singing reference.

Constraints:

I only have a few GPUs.
I cannot train a large SVC/codec/foundation model from scratch.
I prefer frozen-model analysis, small learned heads, small adapters, and
rigorous evaluation.
I want a paper-worthy research question, not just an engineering demo.
Candidate data:

GTSinger, because it has paired speech/singing, phoneme alignments, singing
technique labels, and 20 professional singers.
I know GTSinger is not enough for causal same-speaker language
disentanglement because singer and language are confounded.
Candidate models/features:

WavLM Base+ hidden layers, possibly layers 3/6/9/12;
ContentVec if useful;
ECAPA-TDNN as a speaker verification baseline/evaluator;
FACodec / NaturalSpeech 3 streams as a factorized-codec comparison;
Seed-VC as downstream zero-shot SVC baseline;
kNN-VC or FreeSVC as possible no-training or open baselines.
Candidate hypothesis:

Frozen SSL/audio-codec features contain separable evidence for:

stable identity core;
speech/singing vocal-mode residual;
content/phone information;
F0/prosody/energy;
singing technique/style.
Candidate pipeline:

Stage A: no synthesis. Cache frozen features and build a leakage map across
layers, temporal statistics, and codec streams.
Stage B: train small heads over frozen features to separate z_core and
z_mode.
Stage C: use the learned representation or residual to improve downstream
Seed-VC conditioning when only target speech reference is available.
I need you to produce a complete research review and experiment plan. Please
answer all of the following in one coherent response.

Part 1: Kill Or Improve The Idea

Is this idea good, mediocre, or bad?
What exactly is not novel? Discuss AdaIN-VC, LoIN, FACodec/NaturalSpeech 3,
FreeCodec, MSR-Codec, Seed-VC, GTSinger, SVCC 2025, and recent speaker
leakage probing work where relevant.
What is the strongest honest novelty framing?
What claims should I avoid?
What would Reviewer 2 attack?
What would make the idea publishable rather than just a probe study?
Part 2: Novelty Matrix

Build a novelty matrix comparing my idea against prior work. Include columns
for:

global mean/std or instance statistics;
local/multiscale temporal statistics;
factorized content/prosody/timbre/residual codecs;
speaker leakage probing in SSL features;
same-person speech-vs-singing identity shift;
phone/F0/duration-controlled analysis;
unseen-singer evaluation;
downstream SVC conditioning intervention;
public code/data feasibility.
Then identify the one or two defensible contribution claims.

Part 3: Stage A Experiment Design

Design the frozen-feature audit in engineering-ready detail:

exact dataset split strategy;
segment/example construction;
feature cache schema;
representation variants;
temporal statistics or multiscale bands;
probe targets;
metrics;
negative controls;
cheap acoustic shortcut baselines;
F0, phone, duration, song/take, SNR, singer, and language controls;
statistical analysis plan, including bootstrap or mixed-effects modeling;
stop/go criteria for moving to Stage B;
expected compute and storage.
Important: explain how to avoid confusing probe recoverability with true
disentanglement.

Part 4: Stage B And Stage C Design

Assume Stage A finds real signal. Design the intervention:

how to define z_core and z_mode;
losses that are justified and losses that are risky;
mandatory baselines;
how to distinguish oracle residual from deployable residual;
how to avoid requiring target singing at inference;
where, if anywhere, to intervene in Seed-VC conditioning;
objective metrics;
subjective evaluation design;
what counts as success;
what failure result could still be publishable.
Part 5: Baseline And Code Feasibility

Audit these candidates and decide whether each is core baseline, optional
baseline, or reference only:

WavLM Base+ / Large;
ContentVec;
ECAPA-TDNN;
FACodec / NaturalSpeech 3;
Seed-VC;
kNN-VC;
FreeSVC;
Serenade;
Vevo/Amphion;
HQ-SVC;
RVC / DDSP-SVC / so-vits-svc.
For each, state whether public code/checkpoints are actually usable, whether
target-speaker training is required, and what compute/environment risk it adds.

Part 6: AI-Coding And Engineering Checkpoints

I want to use AI coding agents to implement the experiments without hallucinated
or misaligned code. Define:

task decomposition;
file/artifact contracts;
manifest schema;
split invariants;
feature-cache schema;
smoke tests;
unit/integration tests;
result table format;
reproducibility requirements;
failure modes that must block progress.
Write this as a spec I could hand to a coding agent.

Part 7: Paper Strategy

Assume results are modest but real. Propose:

possible paper titles;
abstract structure;
contribution bullets;
related work organization;
method section outline;
experiment tables and figures;
ablations;
limitations;
strongest narrative if results are positive;
strongest narrative if results are mixed;
claims to avoid.
Part 8: Final Decision

End with:

go/no-go recommendation;
the first experiment I should run this week;
the result that would make you continue;
the result that would make you pivot;
a 30-day plan.
Please be blunt, concrete, and source-grounded. Distinguish verified facts from
speculation.

# pro
## Executive verdict

**Go, but only with a hard gate.** The idea is **good-but-fragile**, not automatically paper-worthy. As a thesis idea, I would approve **Stage A immediately** and make Stages B/C conditional on evidence that survives controls. As a paper idea, I would reject it if it is framed as “we discovered speaker identity disentanglement in SSL features” or “we improve SVC with a residual trick.” I would take it seriously if framed as:

> **A controlled study of same-person speech-to-singing identity shift in frozen SSL/codec representations, plus a small deployable conditioning intervention that estimates a singing-timbre residual from target speech only.**

The most important distinction is this: **probe recoverability is not disentanglement**. A classifier finding “speaker,” “mode,” or “technique” in WavLM features does not prove those factors are separable, causal, or usable. The paper becomes interesting only if you show that the residual is **predictable under held-out-singer or held-out-song conditions**, survives **phone/F0/duration/energy/language controls**, and produces a measurable downstream effect when target singing is unavailable.

Verified facts that matter: GTSinger is unusually suitable because it contains **80.59 hours of singing, 16.16 hours of paired speech, 20 professional singers, nine languages, six singing techniques, phoneme-level alignment, and style/technique labels**. However, each singer is associated with one language, so your concern about singer-language confounding is real. ([arXiv](https://arxiv.org/html/2409.13832v7 "https://arxiv.org/html/2409.13832v7")) SVCC 2025 also moved the field toward singing **style** conversion on GTSinger and found that style/naturalness, especially dynamic techniques such as breathy, glissando, and vibrato, remain difficult. ([arXiv](https://arxiv.org/html/2509.15629v1 "https://arxiv.org/html/2509.15629v1"))

---

# Part 1 — Kill or improve the idea

## Is the idea good, mediocre, or bad?

**As stated: mediocre-to-good.**  
**After reframing: good.**

The raw idea — “same person has stable identity core plus speech/singing residual” — is intuitive but dangerous because it sounds like a latent-factor claim you probably cannot prove with 20 singers. The publishable version is more modest:

> Frozen SSL and codec representations encode measurable, mode-dependent deviations between the same person’s speaking and singing timbre; some of these deviations can be estimated from speech-only references and used to improve speech-prompted zero-shot SVC conditioning.

That is a **measurement + intervention** paper, not a pure disentanglement paper.

The idea is **bad** only if you let it become one of these:

1. A layer-probing heatmap with no downstream intervention.
    
2. A “speaker identity disentanglement” claim based on linear probes.
    
3. A Seed-VC demo with cherry-picked examples.
    
4. A method that secretly uses target singing during adaptation and then claims speech-only SVC.
    

## What exactly is not novel?

### 1. Global mean/std and instance statistics are not novel

AdaIN-VC already used instance normalization ideas for one-shot voice conversion, separating speaker and content representations through adaptive instance-normalization-style statistics. ([arXiv](https://arxiv.org/pdf/1904.05742 "https://arxiv.org/pdf/1904.05742")) So “mean/std of hidden representations contains speaker/timbre information” is not new.

LoIN goes further by using **locality-based instance normalization** and local statistics computed on randomly selected frames, so even “local statistics can be useful for VC” is not untouched. ([ISCA Archive](https://www.isca-archive.org/interspeech_2023/gu23b_interspeech.pdf "https://www.isca-archive.org/interspeech_2023/gu23b_interspeech.pdf"))

**Implication:** you cannot claim novelty from using mean/std or local stats. You can use them as tools.

### 2. Factorized content/prosody/timbre/residual codecs are not novel

NaturalSpeech 3 / FACodec explicitly factorizes speech into content, prosody, timbre, and acoustic-detail streams. ([arXiv](https://arxiv.org/abs/2403.03100 "https://arxiv.org/abs/2403.03100")) FreeCodec similarly proposes a disentangled codec with timbre, prosody, and content components. ([arXiv](https://arxiv.org/abs/2412.01053 "https://arxiv.org/abs/2412.01053")) MSR-Codec proposes multiple streams including semantic, timbre, prosody, and residual streams. ([arXiv](https://arxiv.org/abs/2509.13068 "https://arxiv.org/abs/2509.13068"))

**Implication:** “we separate content/prosody/timbre/residual” is not novel. Your novelty would be testing whether these nominal streams actually preserve or suppress **same-person speech-vs-singing identity shift**.

### 3. Speaker leakage probing in SSL features is not novel

WavLM and similar SSL models are already known to contain speaker, content, and paralinguistic information. WavLM itself was designed for full-stack speech processing rather than pure content extraction. ([arXiv](https://arxiv.org/abs/2110.13900 "https://arxiv.org/abs/2110.13900")) Recent probing/removal work studies speaker leakage in SSL representations and asks how much speaker information remains in intermediate or final layers. ([arXiv](https://arxiv.org/html/2501.05310 "https://arxiv.org/html/2501.05310"))

**Implication:** a “leakage map” alone is weak. Your Stage A must be more specific: **same-speaker cross-mode residual geometry under singing controls**.

### 4. Speech-prompted singing conversion is not novel

This is the biggest reviewer trap. There is already direct work on using speech references for singing-related generation/conversion.

Everyone-Can-Sing proposes unified zero-shot SVS/SVC using a speech sample as voice-identity control and includes analysis of speech-reference versus singing-reference behavior. ([arXiv](https://arxiv.org/abs/2501.13870 "https://arxiv.org/abs/2501.13870")) SSANSVC explicitly tackles “bridging speech and singing” for SVC by aligning speech speaker embeddings and singing speaker embeddings. ([ISCA Archive](https://www.isca-archive.org/interspeech_2025/liu25h_interspeech.pdf "https://www.isca-archive.org/interspeech_2025/liu25h_interspeech.pdf"))

Seed-VC is also already a zero-shot VC/SVC system that supports reference speech and SVC inference; its repo documents zero-shot VC/SVC and 1–30 second reference speech usage. ([GitHub](https://github.com/Plachtaa/seed-vc "https://github.com/Plachtaa/seed-vc"))

**Implication:** you should not claim “first speech-reference SVC” or “first to bridge speech and singing identity.” That would be false.

### 5. GTSinger has already been used as a benchmark substrate

GTSinger was explicitly built with paired speech/singing, techniques, alignments, and benchmarks. ([arXiv](https://arxiv.org/html/2409.13832v7 "https://arxiv.org/html/2409.13832v7")) SVCC 2025 used a GTSinger subset and evaluated many systems on singing style conversion. ([arXiv](https://arxiv.org/html/2509.15629v1 "https://arxiv.org/html/2509.15629v1"))

**Implication:** using GTSinger is sensible, not novel. Your contribution must come from **the controlled analysis and intervention**, not dataset choice.

---

## Strongest honest novelty framing

The strongest framing is:

> **We study the same-person speech-to-singing timbre gap as a measurable, controllable residual in frozen SSL/codec representations. Unlike prior speech-prompted SVC systems, we isolate whether the residual survives phone/F0/duration/energy/technique controls, quantify oracle-vs-deployable residual gaps, and test whether a small frozen-model adapter improves target-speech-only SVC conditioning.**

That is defensible because it combines:

1. **Same-person paired speech/singing analysis**, not generic speaker leakage.
    
2. **Control-heavy residual modeling**, not just probing.
    
3. **Oracle versus deployable residual comparison**, which is methodologically important.
    
4. **Downstream conditioning intervention**, not just representation visualization.
    
5. **Small-model feasibility**, aligned with your GPU constraints.
    

## Claims to avoid

Avoid these claims:

- “We disentangle identity, content, prosody, and technique.”
    
- “There is a universal stable identity core.”
    
- “The speech-to-singing residual is causal.”
    
- “The residual is language-independent.”
    
- “FACodec / NaturalSpeech 3 streams are ground-truth factors.”
    
- “ECAPA similarity equals perceived singer identity.”
    
- “We solve speech-reference SVC.”
    
- “We outperform SOTA SVC.”
    
- “Probe accuracy proves usable information.”
    

Use weaker claims:

- “Evidence for a mode-dependent residual.”
    
- “Predictable under controlled conditions.”
    
- “Improves a frozen downstream baseline under a target-speech-only setting.”
    
- “Quantifies when frozen representations leak or suppress mode-specific identity cues.”
    

## What Reviewer 2 will attack

Reviewer 2’s strongest attacks will be:

1. **Only 20 singers.**  
    A residual learned across 20 professional singers may not generalize. You need cluster bootstrap and held-out-singer tests.
    
2. **Singer-language confound.**  
    In GTSinger, singers are tied to languages. You cannot make strong cross-lingual disentanglement claims. ([arXiv](https://arxiv.org/html/2409.13832v7 "https://arxiv.org/html/2409.13832v7"))
    
3. **F0 and duration shortcuts.**  
    Singing differs from speech in pitch range, phoneme duration, vibrato, energy, and articulation. Your residual may simply be prosody.
    
4. **Probe recoverability is not disentanglement.**  
    A classifier can exploit nuisance correlations. You need controls and intervention.
    
5. **Speech-reference SVC already exists.**  
    Everyone-Can-Sing, SSANSVC, Seed-VC, and SVCC systems reduce the novelty of a pure application story. ([arXiv](https://arxiv.org/abs/2501.13870 "https://arxiv.org/abs/2501.13870"))
    
6. **Seed-VC intervention may be brittle.**  
    Seed-VC is usable but its GitHub repository is archived read-only as of November 2025, so reproducibility requires version pinning and care. ([GitHub](https://github.com/Plachtaa/seed-vc "https://github.com/Plachtaa/seed-vc"))
    
7. **Objective identity metrics are biased.**  
    ECAPA is mostly speech-trained. Singing identity and speech identity are not identical; human studies show cross-mode voice recognition is difficult. ([AIP Publishing](https://pubs.aip.org/asa/jel/article/4/6/065203/3299000/Who-is-singing-Voice-recognition-from-spoken "https://pubs.aip.org/asa/jel/article/4/6/065203/3299000/Who-is-singing-Voice-recognition-from-spoken"))
    

## What makes it publishable rather than just a probe study?

It becomes publishable if you deliver all three:

1. **A controlled map**  
    Which layers/statistics/codec streams carry singer, mode, phone, F0, energy, technique, and language signals?
    
2. **A residual model with honest deployment constraints**  
    Separate:
    
    - oracle residual: target speech + target singing available;
        
    - deployable residual: target speech only;
        
    - population residual: learned from other singers.
        
3. **A downstream intervention**  
    Show that a frozen or lightly adapted Seed-VC-style system improves target identity similarity in SVC when only target speech is available, without sacrificing naturalness, lyrics/content, or melody.
    

Without #3, it is probably a workshop paper or thesis chapter. With #3, it can be a solid conference submission.

---

# Part 2 — Novelty matrix

Legend: **✓ central**, **~ partial / adjacent**, **— not central**.

|Work / idea|Global mean/std or instance stats|Local / multiscale temporal stats|Factorized content/prosody/timbre/residual codecs|Speaker leakage probing in SSL|Same-person speech-vs-singing identity shift|Phone/F0/duration-controlled analysis|Unseen-singer evaluation|Downstream SVC conditioning intervention|Public code/data feasibility|
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
|AdaIN-VC|✓|—|—|—|—|—|~|✓|✓|
|LoIN|✓|✓|—|—|—|—|~|✓|✓|
|FACodec / NaturalSpeech 3|~|~|✓|—|—|~|~|✓|~|
|FreeCodec|~|~|✓|—|—|—|~|~|~|
|MSR-Codec|~|~|✓|—|—|—|~|~|~|
|ContentVec / SSL speaker leakage probes|—|~|—|✓|—|~|~|—|✓|
|Everyone-Can-Sing|—|—|~|—|~|~|✓|✓|~|
|SSANSVC / speech-singing alignment|—|—|~|—|✓|~|✓|✓|~|
|GTSinger / SVCC 2025 setting|—|—|—|—|~|✓|✓|✓|✓|
|**Your proposed paper**|✓ as tool, not claim|✓ as tool, not claim|✓ as comparator|✓ as diagnostic|**✓ central**|**✓ central**|**✓ required**|**✓ required**|**✓ central constraint**|

The one or two defensible contribution claims are:

1. **Controlled representation contribution**  
    “We provide a controlled audit of same-person speech-vs-singing identity shift in frozen SSL and codec representations, separating mode-specific timbre residuals from phone, F0, duration, energy, technique, song/take, and language confounds.”
    
2. **Deployable intervention contribution**  
    “We introduce and evaluate a small target-speech-only residual adapter for zero-shot SVC conditioning, reporting the gap between oracle target-singing residuals and deployable speech-only estimates.”
    

Do not claim more than that unless results are unusually strong.

---

# Part 3 — Stage A experiment design: frozen-feature audit

## Goal

Stage A should answer:

> In frozen representations, is the same-person speech-to-singing shift structured, predictable, and not fully explained by phone/F0/duration/energy/technique/language shortcuts?

It should not claim disentanglement.

## Dataset split strategy

Use **multiple splits**, because no single GTSinger split can answer all questions.

### Split 1: paired within-singer, held-out song

Purpose: estimate same-person residuals without leaking exact song/phrase/take.

- Train/dev/test split by `song_id` or `song_group_id`.
    
- All utterances from the same song, lyric phrase, technique group, and paired speech item stay in the same split.
    
- Same singer can appear in train and test.
    
- This tests whether the residual is stable across material for known singers.
    

Use this for first residual maps and oracle/deployable gap estimates.

### Split 2: held-out singer

Purpose: test generalization to unseen target singers.

- Hold out entire singers for test.
    
- Because there are only 20 singers, use leave-one-singer-out or 5 folds of 4 held-out singers.
    
- Report language composition explicitly.
    
- Do not claim language-invariant generalization unless you have multiple singers per language in train and test.
    

This split is mandatory for publishability.

### Split 3: within-language held-out singer subset

Purpose: reduce singer-language confound.

- For languages with multiple singers, run held-out-singer tests within that language.
    
- For languages with only one singer, use them only for descriptive analysis, not language-controlled claims.
    
- If English/Chinese have enough singers, make them the primary controlled subset.
    

This is the split Reviewer 2 will trust most.

### Split 4: technique-held-out or technique-stratified

Purpose: avoid confusing technique/style with identity residual.

- Either hold out one technique during residual prediction, or stratify all probe/reporting metrics by technique.
    
- At minimum, report control-singing separately from technique singing.
    

GTSinger’s six technique labels are valuable here. ([arXiv](https://arxiv.org/html/2409.13832v7 "https://arxiv.org/html/2409.13832v7"))

## Segment and example construction

Use three levels of examples.

### Level 1: utterance-level examples

Each row is one speech or singing utterance.

Fields:

- `utt_id`
    
- `singer_id`
    
- `language`
    
- `mode`: `speech`, `singing_control`, `singing_technique`
    
- `technique`
    
- `song_id`
    
- `phrase_id`
    
- `take_id`
    
- `wav_path`
    
- `duration`
    
- `text`
    
- `phone_sequence`
    
- `alignment_path`
    
- `paired_utt_id`
    
- `f0_stats`
    
- `energy_stats`
    
- `snr_estimate`
    

Use for global stats, retrieval, and probe training.

### Level 2: phone-aligned examples

Each row is a phoneme occurrence.

Fields:

- `utt_id`
    
- `phone`
    
- `phone_start`
    
- `phone_end`
    
- `phone_duration`
    
- `mode`
    
- `singer_id`
    
- `song_id`
    
- `technique`
    
- frozen feature frames overlapping the phone
    

Use this for phone-controlled analysis.

### Level 3: pair examples

Each row is a speech/singing pair.

Types:

- same singer, same phrase;
    
- same singer, different phrase;
    
- different singer, same language;
    
- different singer, matched F0 range if possible;
    
- different singer, same song/technique if available.
    

Use this for cross-mode verification and residual geometry.

## Feature cache schema

Use a **manifest + feature store** design.

### `manifests/utterances.parquet`

Required columns:

```text
utt_id
singer_id
language
vocal_range
mode
technique
song_id
phrase_id
take_id
wav_path
start_sec
end_sec
duration_sec
sample_rate
num_samples
text
phone_seq
alignment_path
paired_utt_id
control_utt_id
split_group_key
snr_db
rms_db
f0_mean_hz
f0_std_hz
f0_min_hz
f0_max_hz
f0_voiced_pct
energy_mean
energy_std
alignment_quality_flag
```

### Feature files

Path pattern:

```text
features/{extractor}/{checkpoint_hash}/{layer_or_stream}/{utt_id}.npz
```

Each `.npz` contains:

```text
x: float16 or float32 array [T, D]
times_sec: float32 array [T]
voiced_mask: bool array [T]
phone_id: int array [T]
f0_hz: float32 array [T]
energy: float32 array [T]
```

Each extractor directory also needs:

```text
metadata.json
```

with:

```text
extractor_name
checkpoint_name
checkpoint_hash
git_commit
sample_rate
hop_sec
feature_dim
layer
normalization
created_by_command
input_manifest_hash
```

## Representation variants

Start small.

### Core

- **WavLM Base+**, layers 3, 6, 9, 12.
    
- Segment-level mean/std.
    
- Phone-level mean/std.
    
- Voiced-only and all-frame variants.
    

WavLM Base+ is a sensible default because public checkpoints are available and WavLM is designed for broad speech tasks rather than pure content extraction. ([GitHub](https://github.com/microsoft/unilm/blob/master/wavlm/README.md "https://github.com/microsoft/unilm/blob/master/wavlm/README.md"))

### Strong optional variants

- **ContentVec** final or 12th-layer features.  
    Useful as a content-oriented contrast, because ContentVec was designed to reduce speaker information relative to HuBERT-like content features. ([arXiv](https://arxiv.org/abs/2204.09224 "https://arxiv.org/abs/2204.09224"))
    
- **FACodec / NaturalSpeech 3 streams** if usable.  
    Use as a factorized-codec comparator, not as ground truth. FACodec explicitly targets content/prosody/timbre/acoustic-detail decomposition. ([arXiv](https://arxiv.org/abs/2403.03100 "https://arxiv.org/abs/2403.03100"))
    

### Evaluator / baseline embeddings

- **ECAPA-TDNN** speaker embeddings.  
    Use for speaker-verification-style metrics, but calibrate carefully because speech-trained speaker models can behave strangely on singing. SpeechBrain provides pretrained ECAPA-TDNN models for VoxCeleb-style speaker verification. ([SpeechBrain](https://speechbrain.readthedocs.io/en/latest/API/speechbrain.lobes.models.ECAPA_TDNN.html "https://speechbrain.readthedocs.io/en/latest/API/speechbrain.lobes.models.ECAPA_TDNN.html"))
    

### Cheap acoustic baselines

Mandatory:

- MFCC mean/std;
    
- log-mel mean/std;
    
- F0 mean/std/range;
    
- voiced percentage;
    
- vibrato rate/extent proxies;
    
- energy mean/std;
    
- duration and phone-duration statistics;
    
- spectral centroid/bandwidth/rolloff;
    
- harmonic-to-noise ratio or cepstral peak prominence if available;
    
- SNR estimate.
    

These are not glamorous, but they protect you from fooling yourself.

## Temporal statistics and multiscale bands

For each representation layer/stream, compute:

### Global utterance statistics

- mean;
    
- std;
    
- median;
    
- 10/25/75/90 percentiles;
    
- optional low-rank covariance or PCA projection.
    

### Local window statistics

Use windows:

- 200 ms;
    
- 500 ms;
    
- 1 s;
    
- whole phoneme;
    
- 3-phone context;
    
- full utterance.
    

For each window:

- mean;
    
- std;
    
- delta mean;
    
- delta std;
    
- voiced-only mean/std;
    
- normalized-by-phone mean/std.
    

### Multiscale comparison

Report which scale carries what:

- frame-level: phone/content and F0 leakage;
    
- phone-level: articulation and duration;
    
- utterance-level: identity, mode, language, technique;
    
- song/take-level: recording and style confounds.
    

## Probe targets

Use simple probes first. Complex probes should be ablations, not the main evidence.

### Classification targets

- `singer_id`
    
- `mode`: speech vs singing
    
- `technique`
    
- `language`
    
- `phone`
    
- `song_id` / `take_id` leakage
    
- `vocal_range` if available
    

### Regression targets

- F0 mean/range;
    
- frame-level F0;
    
- energy;
    
- phone duration;
    
- utterance duration;
    
- SNR estimate.
    

### Pairwise targets

- same singer vs different singer;
    
- same singer cross-mode vs same singer same-mode;
    
- same phrase vs different phrase;
    
- same technique vs different technique.
    

### Residual targets

For paired speech/singing features:

```text
delta_s,i = stat(singing_s,i) - stat(speech_s,i)
```

Probe whether `delta_s,i` is:

- stable within singer;
    
- predictable from speech;
    
- predictable across songs;
    
- predictable across unseen singers;
    
- mostly explained by F0/duration/energy/phone distribution.
    

## Metrics

### For classifiers

- balanced accuracy;
    
- macro F1;
    
- AUROC for binary probes;
    
- calibration error if used for decisions.
    

### For verification/retrieval

- EER;
    
- AUROC;
    
- Recall@1 / Recall@5;
    
- mean reciprocal rank;
    
- within-person cross-mode distance;
    
- between-person matched-mode distance;
    
- distance ratio:
    

```text
R = mean distance(same singer, speech-sing)
    / mean distance(different singer, same language, same mode)
```

Lower is better for cross-mode identity stability.

### For regression

- R²;
    
- MAE;
    
- Spearman correlation;
    
- residualized R² after nuisance controls.
    

### For residual prediction

- cosine similarity between predicted and oracle residual;
    
- MSE / normalized MSE;
    
- improvement in cross-mode retrieval after applying residual;
    
- gap closed relative to oracle residual:
    

```text
gap_closed =
  (metric_deployable - metric_speech_baseline)
  / (metric_oracle - metric_speech_baseline)
```

## Negative controls

Mandatory controls:

1. **Label permutation within singer/language**  
    If probe performance remains high after permutation, you have leakage.
    
2. **Random features with same shape**  
    Confirms the probe is not exploiting sample count imbalance.
    
3. **Duration-only and F0-only baselines**  
    If they match WavLM performance, the “identity residual” story collapses.
    
4. **Song/take classifier**  
    High song/take recoverability means recording or arrangement leakage.
    
5. **Pair shuffling**  
    Break true speech/singing pairs while preserving singer/language distributions.
    
6. **Same-language different-singer negatives**  
    Prevent language from solving identity.
    
7. **Same-technique negatives**  
    Prevent technique from solving mode residual.
    
8. **Silence / low-energy frame removal**  
    Prevent background/noise artifacts.
    

## Controls for nuisance variables

### F0

Compute per utterance and per phone:

- mean log-F0;
    
- F0 range;
    
- F0 slope;
    
- voiced percentage;
    
- vibrato proxy.
    

Then either:

- include F0 stats as covariates;
    
- residualize representation stats against F0;
    
- stratify comparisons by F0 bins.
    

### Phone content

Use GTSinger alignments.

- Compare same phone across speech/singing.
    
- Use phone-distribution matching.
    
- Compute phone-normalized statistics:
    

```text
feature_residual = feature_stat - mean_feature_for_phone
```

### Duration

Control for:

- utterance duration;
    
- phone duration;
    
- speech/singing duration ratio;
    
- local tempo.
    

### Energy

Control for:

- RMS;
    
- loudness;
    
- energy variance.
    

### Song/take

Keep all material from one song/take group within one split. Include song/take random effects.

### SNR / recording condition

Estimate SNR or use acoustic quality proxies. Include as covariates.

### Singer

Use singer as a random effect in statistical models. For held-out-singer tests, singer must not appear in train.

### Language

Because singer and language are confounded, do not claim full language disentanglement. Use:

- within-language subsets;
    
- language fixed effects;
    
- language-stratified reporting;
    
- explicit “language-confounded” labels for all-language analyses.
    

## Statistical analysis plan

Use cluster-level uncertainty, not frame-level uncertainty.

### Bootstrap

Use cluster bootstrap over:

- singer;
    
- song;
    
- phrase.
    

Do not bootstrap individual frames as independent samples.

Report:

- mean metric;
    
- 95% confidence interval;
    
- number of singers;
    
- number of songs;
    
- number of utterances.
    

### Mixed-effects modeling

For residual magnitude or similarity metrics, use models like:

```text
similarity ~ mode_pair
           + f0_distance
           + energy_distance
           + phone_distribution_distance
           + duration_distance
           + technique_match
           + language_match
           + snr_distance
           + (1 | singer_id)
           + (1 | song_id)
```

For mode effect in representation stats:

```text
feature_stat ~ mode
             + f0_mean
             + f0_range
             + energy
             + duration
             + phone_distribution
             + technique
             + language
             + snr
             + (1 | singer_id)
             + (1 | song_id)
```

If enough data exists, add random slopes:

```text
(1 + mode | singer_id)
```

### Multiple comparisons

You will test many layers, statistics, and probes. Use:

- Benjamini-Hochberg FDR correction across layer/stat/probe families;
    
- pre-register primary comparisons:
    
    - WavLM Base+ layers 3/6/9/12;
        
    - utterance mean/std and phone-level mean/std;
        
    - cross-mode same-singer retrieval;
        
    - residual prediction after controls.
        

## Stop/go criteria for Stage B

Continue to Stage B only if all are true:

1. Cross-mode same-person structure is detectable beyond cheap acoustic baselines.
    
2. Residual prediction beats:
    
    - global mean residual;
        
    - language/technique mean residual;
        
    - F0/duration/energy-only residual.
        
3. The effect survives held-out song tests.
    
4. At least some effect survives held-out singer or within-language held-out singer tests.
    
5. Confidence intervals exclude zero improvement under cluster bootstrap.
    
6. The residual is not merely a song/take or SNR artifact.
    

Pivot if:

- mode is highly decodable but identity does not survive cross-mode;
    
- residuals vanish after F0/duration/energy controls;
    
- held-out singer performance collapses;
    
- language explains most of the effect;
    
- cheap acoustic features match WavLM/FACodec.
    

## Expected compute and storage

Approximate total audio:

```text
80.59 h singing + 16.16 h speech = 96.75 h
```

At 50 frames/sec:

```text
96.75 * 3600 * 50 ≈ 17.4 million frames
```

For WavLM Base+ with 768-dimensional hidden states:

```text
17.4M frames * 768 dims * 2 bytes float16 ≈ 26.7 GB per layer
```

So:

- four selected layers: about **107 GB**;
    
- all 13 layers: about **347 GB**;
    
- stats-only tables: usually **a few GB or less**;
    
- adding ContentVec and codec streams may add **tens to hundreds of GB**, depending on cache format.
    

Practical recommendation:

1. First cache only stats for all data.
    
2. Cache full frame-level features only for selected layers and subsets.
    
3. Use float16 for frozen features.
    
4. Store manifests in Parquet.
    
5. Use Zarr/HDF5/NPZ shards, not one giant file.
    

## How to avoid confusing probe recoverability with true disentanglement

Say this explicitly in the paper:

> A successful probe shows that information is recoverable from a representation under a given probe family and dataset distribution. It does not imply that the representation factorizes the underlying generative causes, nor that changing the probed direction will selectively change only that factor.

Then enforce it experimentally:

- Use matched negatives.
    
- Use acoustic controls.
    
- Use pair shuffling.
    
- Use held-out singers.
    
- Use downstream intervention.
    
- Report when the intervention fails.
    
- Avoid the word “disentangled” unless you mean “operationally separated under our model and controls.”
    

---

# Part 4 — Stage B and Stage C design

## Stage B: defining `z_core` and `z_mode`

Let `h(x)` be a frozen representation statistic for utterance or phone-aligned segment `x`.

For singer `s`, mode `m`, item `i`:

```text
h_s,m,i = frozen statistic
```

A practical decomposition is:

```text
h_s,m,i = z_core_s + z_mode_s,m + z_content_i + z_prosody_i + noise
```

But you should not claim this is the true generative decomposition. Treat it as a **working linear factorization**.

### Option 1: centroid residual

For each training singer:

```text
z_speech_s = mean_i h_s,speech,i
z_sing_s   = mean_i h_s,singing,i

z_core_s = normalize((z_speech_s + z_sing_s) / 2)

delta_s = z_sing_s - z_speech_s
```

Then learn:

```text
delta_hat_s = A(z_speech_s, acoustic_stats_s, maybe language/technique priors)
```

where `A` is a small MLP, ridge regressor, or low-rank affine adapter.

### Option 2: contrastive core + residual

Learn small projection heads:

```text
z_core = P_core(h)
z_mode = P_mode(h)
```

Use:

- same-singer cross-mode contrastive loss for `z_core`;
    
- mode classification or reconstruction loss for `z_mode`;
    
- nuisance-control evaluation, not over-strong adversarial removal.
    

### Option 3: phone-conditioned residual

For phone `p`:

```text
delta_s,p = mean h_s,sing,p - mean h_s,speech,p
```

Then average across phones or learn a phone-conditioned adapter. This is more controlled but more complex.

My advice: start with **centroid residual + phone/F0/duration residualization**. It is harder to overfit and easier to explain.

## Losses that are justified

Use simple, auditable losses.

### Same-person cross-mode contrastive loss

Pull speech and singing from the same singer together in `z_core`; push matched-language different singers apart.

```text
L_core = InfoNCE(z_core_speech, z_core_singing)
```

Negatives should be matched by language, technique, vocal range, and F0 range when possible.

### Residual prediction loss

Predict the singing residual from speech-only features:

```text
L_delta = || delta_hat_s - delta_oracle_s ||_2^2
```

or cosine distance:

```text
L_delta = 1 - cos(delta_hat_s, delta_oracle_s)
```

### Reconstruction of representation statistics

Require:

```text
z_core + z_mode ≈ h
```

only in the projected feature-statistic space, not raw audio.

### Variance / collapse prevention

Use light regularization:

- L2;
    
- covariance regularization;
    
- variance floor.
    

## Losses that are risky

Be careful with:

### Strong adversarial removal of language, F0, or phone

With only 20 singers, adversarial losses can erase identity or create unstable artifacts. Use adversarial losses only as ablations.

### Orthogonality claims

A dot-product orthogonality constraint does not prove statistical independence or causal separation.

### ECAPA-only identity loss

ECAPA may reward speech-like similarity, not perceived singing identity.

### Losses using target singing at inference

You can use target singing for oracle analysis, never for deployable claims.

## Mandatory baselines

For Stage B, compare against:

1. **Speech centroid only**  
    Use target speech representation as-is.
    
2. **Global mean residual**
    

```text
delta_global = mean_train_singers(z_sing - z_speech)
```

3. **Language-conditioned mean residual**
    

```text
delta_language = mean_train_singers_same_language(...)
```

Only valid where language has enough singers.

4. **Technique-conditioned mean residual**
    

```text
delta_technique = mean_train_singers_same_technique(...)
```

5. **F0/duration/energy-only predictor**
    

A linear model using cheap acoustic features only.

6. **Random residual**
    

Magnitude-matched random direction.

7. **Oracle residual**
    

Uses target singing. Upper bound only.

8. **No residual**
    

Downstream baseline.

## Oracle residual versus deployable residual

This distinction is mandatory.

### Oracle residual

Uses:

```text
target speech + target singing
```

Purpose:

- estimate best possible residual effect;
    
- measure upper bound;
    
- show whether the intervention path can help at all.
    

Never present this as an inference-time method.

### Deployable residual

Uses:

```text
target speech only
```

Allowed inputs:

- target speech reference;
    
- source singing;
    
- training singers’ speech/singing pairs;
    
- population residual learned from training set.
    

Disallowed inputs:

- target singing;
    
- target singer fine-tuning on singing;
    
- test-set technique labels if unavailable at deployment, unless explicitly marked as oracle metadata.
    

## Stage C: where to intervene in Seed-VC

Seed-VC is a reasonable downstream target because it supports zero-shot VC/SVC with reference speech and has documented SVC inference paths. ([GitHub](https://github.com/Plachtaa/seed-vc "https://github.com/Plachtaa/seed-vc")) But treat it as a frozen black-box baseline unless you are confident patching internals.

### Preferred intervention: reference/timbre conditioning hook

Find the target reference encoder output used for timbre conditioning.

Then test:

```text
t_ref_adapted = t_ref + alpha * delta_hat
```

or:

```text
t_ref_adapted = W [t_ref ; delta_hat]
```

with `W` a small learned affine adapter.

Freeze Seed-VC. Train only the adapter.

### Safer intervention: statistics shift

If the conditioning is a sequence of hidden states rather than a pooled vector, apply an AdaIN-like shift:

```text
h_ref_adapted =
  sigma_target * normalize(h_ref) + mu_target
```

where:

```text
mu_target = mu_speech + alpha * delta_mu_hat
sigma_target = sigma_speech + alpha * delta_sigma_hat
```

This connects to prior instance-statistics VC but your contribution is the speech-to-singing residual under controls, not AdaIN itself.

### Weak but robust intervention: reference selection/reweighting

Select target speech segments whose predicted residual is closest to a desired singing-timbre prior. This avoids patching Seed-VC internals but is less likely to produce a strong result.

Use this as fallback.

## Objective metrics for Stage C

Use multiple metric families.

### Identity similarity

- ECAPA cosine similarity;
    
- another speaker/singer embedding if available;
    
- cross-mode calibrated EER;
    
- target speech similarity and, for evaluation only, target singing similarity.
    

Do not rely on ECAPA alone.

### Content / lyrics

- ASR WER/CER if lyrics are available;
    
- phoneme error rate if forced alignment is feasible;
    
- content embedding similarity to source.
    

### Melody and prosody

- F0 RMSE against source singing;
    
- voicing decision error;
    
- gross pitch error;
    
- melody contour correlation;
    
- vibrato rate/extent preservation.
    

### Audio quality

- DNSMOS or neural MOS predictor, if validated for singing with caution;
    
- spectral distortion;
    
- VERSA-style metrics if you follow SVCC practice. SVCC 2025 used both subjective and objective evaluation, including VERSA-based objective metrics. ([arXiv](https://arxiv.org/html/2509.15629v1 "https://arxiv.org/html/2509.15629v1"))
    

### Style / technique

- technique classifier agreement;
    
- F0 modulation features for vibrato/glissando;
    
- breathiness proxies if available.
    

## Subjective evaluation design

Minimum viable subjective test:

- 20–30 conversion items;
    
- 10–20 listeners per item;
    
- blind randomized order;
    
- systems:
    
    1. source singing;
        
    2. Seed-VC speech-reference baseline;
        
    3. your deployable residual adapter;
        
    4. global residual baseline;
        
    5. oracle residual upper bound, clearly marked only in analysis;
        
    6. ground-truth target singing if available.
        

### Rating dimensions

Use separate questions:

1. **Naturalness**  
    “How natural is the singing?”
    
2. **Target identity similarity**  
    Provide target speech reference. Optionally run a separate block with target singing reference for analysis.
    
3. **Lyrics/content preservation**  
    “Are the words preserved?”
    
4. **Melody preservation**  
    “Does it follow the source melody?”
    
5. **Singing-mode appropriateness**  
    “Does the converted voice sound like singing rather than speech-like vocalization?”
    

### Preferred test format

Use pairwise preference for the main claim:

```text
Baseline Seed-VC vs Residual-adapted Seed-VC
```

Ask:

> Which sample sounds more like the target person singing, while preserving the source melody and lyrics?

Pairwise tests are more sensitive than MOS for modest improvements.

Analyze with mixed-effects logistic or ordinal models:

```text
preference ~ system + (1 | listener) + (1 | item) + (1 | target_singer)
```

## What counts as success?

A convincing success looks like:

1. Deployable residual improves identity similarity over speech-reference baseline.
    
2. Naturalness does not significantly drop.
    
3. Melody/content metrics do not significantly drop.
    
4. The improvement is not matched by global residual or acoustic-only residual.
    
5. The deployable residual closes a nontrivial part of the oracle gap.
    
6. Human listeners prefer the residual-adapted output for target identity.
    

A modest but real success is enough. You do not need SOTA.

## Publishable failure results

A failure can still publish if it teaches something cleanly.

Examples:

1. **Residual is recoverable but not useful**  
    Frozen features contain a same-person mode residual, but Seed-VC conditioning ignores or overwrites it.
    
2. **Residual is mostly acoustic**  
    After phone/F0/duration/energy controls, the apparent identity shift disappears. This would be a valuable warning against naive speaker-identity claims.
    
3. **Oracle helps, deployable does not**  
    Target singing residuals improve conditioning, but speech-only prediction fails. This quantifies the missing information gap.
    
4. **Metric-human mismatch**  
    ECAPA improves while listeners do not, or vice versa. That is useful for the SVC evaluation community.
    

---

# Part 5 — Baseline and code feasibility audit

This is based on public documentation and repositories, not on actually running the code here.

|Candidate|Role|Public code/checkpoints usability|Target-speaker training required?|Compute / environment risk|Recommendation|
|---|---|---|---|---|---|
|**WavLM Base+**|Frozen feature extractor|Official checkpoints are publicly documented. ([GitHub](https://github.com/microsoft/unilm/blob/master/wavlm/README.md "https://github.com/microsoft/unilm/blob/master/wavlm/README.md"))|No|Low–medium|**Core**|
|**WavLM Large**|Stronger optional extractor|Public, but larger. ([GitHub](https://github.com/microsoft/unilm/blob/master/wavlm/README.md "https://github.com/microsoft/unilm/blob/master/wavlm/README.md"))|No|Medium storage/GPU|Optional|
|**ContentVec**|Content-oriented SSL comparison|Public paper/code exist. ([arXiv](https://arxiv.org/abs/2204.09224 "https://arxiv.org/abs/2204.09224"))|No|Medium; fairseq/version risk|Core optional|
|**ECAPA-TDNN**|Speaker-verification baseline/evaluator|SpeechBrain pretrained ECAPA models are available. ([SpeechBrain](https://speechbrain.readthedocs.io/en/latest/API/speechbrain.lobes.models.ECAPA_TDNN.html "https://speechbrain.readthedocs.io/en/latest/API/speechbrain.lobes.models.ECAPA_TDNN.html"))|No|Low|**Core evaluator**, but not sole metric|
|**FACodec / NaturalSpeech 3**|Factorized-codec comparison|Amphion/HF FACodec resources exist. ([Hugging Face](https://huggingface.co/amphion/naturalspeech3_facodec "https://huggingface.co/amphion/naturalspeech3_facodec"))|No for analysis|Medium–high dependency risk|Optional/reference|
|**Seed-VC**|Downstream zero-shot SVC baseline/intervention|Code/checkpoints documented; repo is archived read-only. ([GitHub](https://github.com/Plachtaa/seed-vc "https://github.com/Plachtaa/seed-vc"))|No for zero-shot; avoid fine-tuning|Medium; patching risk|**Core downstream**, version-pin|
|**kNN-VC**|No-training VC sanity baseline|Public code/checkpoints; WavLM + kNN + HiFi-GAN pipeline. ([GitHub](https://github.com/bshall/knn-vc "https://github.com/bshall/knn-vc"))|No|Low–medium; speech-VC not SVC|Optional|
|**FreeSVC**|Zero-shot SVC baseline|Public ICASSP 2025 code and pretrained weights documented; release notes mention quality limitations. ([GitHub](https://github.com/freds0/free-svc "https://github.com/freds0/free-svc"))|No for zero-shot usage|Medium; metadata/workflow risk|Optional|
|**Serenade**|Singing style conversion reference|Public GTSinger SSC recipe and pretrained models are documented. ([GitHub](https://github.com/lesterphillip/serenade "https://github.com/lesterphillip/serenade"))|Likely no for demo; training optional|Medium–high|Reference/optional|
|**Vevo / Amphion**|Strong modern zero-shot voice/style reference|Amphion provides Vevo inference; Vevo1.5 components are large. ([GitHub](https://github.com/open-mmlab/Amphion/blob/main/models/vc/vevo/README.md "https://github.com/open-mmlab/Amphion/blob/main/models/vc/vevo/README.md"))|No for zero-shot inference|High; large models/deps|Reference/optional|
|**HQ-SVC**|New zero-shot SVC reference|Public paper/repo exists, but usability should be verified. ([GitHub](https://github.com/ShawnPi233/HQ-SVC "https://github.com/ShawnPi233/HQ-SVC"))|No if zero-shot works|High; new-code risk|Reference/optional|
|**RVC**|Community target-trained SVC|Public and widely used; designed for small target datasets. ([GitHub](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/docs/en/README.en.md "https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/docs/en/README.en.md"))|Usually yes|Medium; ethical/license variability|Reference only|
|**DDSP-SVC**|Target-trained SVC|Public; lower training requirements than some VITS systems. ([GitHub](https://github.com/yxlllc/DDSP-SVC "https://github.com/yxlllc/DDSP-SVC"))|Usually yes|Medium|Reference only|
|**so-vits-svc**|Target-trained SVC framework|Public but archived/limited update; users train models independently. ([GitHub](https://github.com/svc-develop-team/so-vits-svc "https://github.com/svc-develop-team/so-vits-svc"))|Yes|Medium–high|Reference only|

Practical baseline stack:

- **Core Stage A:** WavLM Base+, ContentVec, ECAPA, acoustic baselines.
    
- **Optional Stage A:** FACodec streams.
    
- **Core Stage C:** Seed-VC baseline + your adapter.
    
- **Optional Stage C:** FreeSVC as external zero-shot SVC comparison.
    
- **Reference only:** Vevo, Serenade, HQ-SVC, RVC, DDSP-SVC, so-vits-svc.
    

Do not let baseline chasing consume the thesis. One strong downstream baseline plus rigorous controls beats six fragile demos.

---

# Part 6 — AI-coding and engineering checkpoints

Below is a spec you can hand to coding agents.

## Project rule

No experiment is valid unless it produces:

```text
config.yaml
resolved_config.yaml
manifest_hash.txt
split_hash.txt
checkpoint_hashes.json
metrics.json
predictions.parquet
run.log
```

No result table may be edited manually.

## Task decomposition

### Task 1: Dataset ingest

Input:

```text
GTSinger root directory
metadata files
alignment files
audio files
```

Output:

```text
manifests/utterances.parquet
manifests/pairs.parquet
reports/manifest_summary.json
```

Responsibilities:

- validate paths;
    
- validate sample rates;
    
- validate speech/singing pair mappings;
    
- compute durations;
    
- attach singer/language/mode/technique/song IDs;
    
- compute basic acoustic stats.
    

### Task 2: Split generation

Input:

```text
manifests/utterances.parquet
manifests/pairs.parquet
split_config.yaml
```

Output:

```text
splits/{split_name}/train.jsonl
splits/{split_name}/dev.jsonl
splits/{split_name}/test.jsonl
splits/{split_name}/split_report.json
```

Required split types:

```text
heldout_song
heldout_singer
within_language_heldout_singer
technique_stratified
```

### Task 3: Feature extraction

Input:

```text
utterances.parquet
extractor_config.yaml
```

Output:

```text
features/{extractor}/{checkpoint_hash}/{layer_or_stream}/{utt_id}.npz
features/{extractor}/{checkpoint_hash}/metadata.json
```

Extractors:

```text
wavlm_base_plus
contentvec
ecapa
facodec_optional
acoustic_baseline
```

### Task 4: Feature statistics

Input:

```text
feature cache
utterance manifest
phone alignments
```

Output:

```text
stats/{extractor}/{layer_or_stream}/utterance_stats.parquet
stats/{extractor}/{layer_or_stream}/phone_stats.parquet
stats/{extractor}/{layer_or_stream}/window_stats.parquet
```

Statistics:

```text
mean
std
median
quantiles
voiced_only_mean
voiced_only_std
phone_normalized_mean
phone_normalized_std
local_window_mean_std
```

### Task 5: Stage A probes

Input:

```text
stats parquet
splits
probe_config.yaml
```

Output:

```text
runs/{run_id}/metrics.json
runs/{run_id}/predictions.parquet
runs/{run_id}/probe_model.pkl or .pt
```

Probe families:

```text
logistic_regression
ridge_regression
linear_svm_optional
small_mlp_optional
cosine_retrieval
verification_scoring
```

### Task 6: Residual modeling

Input:

```text
paired stats
splits
residual_config.yaml
```

Output:

```text
runs/{run_id}/residual_model.pt
runs/{run_id}/residual_predictions.parquet
runs/{run_id}/oracle_vs_deployable_metrics.json
```

Models:

```text
global_mean_residual
language_mean_residual
technique_mean_residual
acoustic_only_linear
ridge_speech_to_delta
small_mlp_speech_to_delta
oracle_residual
```

### Task 7: Seed-VC intervention

Input:

```text
source singing wav
target speech wav
residual model
seedvc checkpoint
adapter_config.yaml
```

Output:

```text
generated/{run_id}/{item_id}_{system}.wav
runs/{run_id}/generation_manifest.parquet
runs/{run_id}/objective_metrics.json
```

Systems:

```text
seedvc_baseline
global_residual
deployable_residual
oracle_residual_upper_bound
random_residual_control
```

### Task 8: Reporting

Input:

```text
all metrics.json
all predictions.parquet
```

Output:

```text
reports/main_tables.xlsx or .csv
reports/figures/*.pdf
reports/figures/*.png
reports/reproducibility.md
```

## Manifest schema

### `utterances.parquet`

Required columns:

```text
utt_id: string
speaker_id: string
language: string
vocal_range: string
mode: enum[speech, singing_control, singing_technique]
technique: string
song_id: string
phrase_id: string
take_id: string
wav_path: string
start_sec: float
end_sec: float
duration_sec: float
sample_rate: int
num_samples: int
text: string
phone_seq: string
alignment_path: string
paired_utt_id: string
control_utt_id: string
split_group_key: string
snr_db: float
rms_db: float
f0_mean_hz: float
f0_std_hz: float
f0_min_hz: float
f0_max_hz: float
f0_voiced_pct: float
energy_mean: float
energy_std: float
alignment_quality_flag: enum[ok, missing, bad]
```

### `pairs.parquet`

Required columns:

```text
pair_id
speech_utt_id
singing_utt_id
speaker_id
language
song_id
phrase_id
technique
pair_type
same_text_flag
same_song_flag
```

## Split invariants

A split generator must fail if any invariant is violated.

### General invariants

```text
utt_id appears in exactly one split
pair_id appears in exactly one split
no missing wav_path
no missing speaker_id
no missing mode
```

### Held-out song split

```text
song_id must not cross train/dev/test
phrase_id must not cross train/dev/test
paired speech/singing items must remain in same split
```

### Held-out singer split

```text
speaker_id in test must not appear in train or dev
speaker_id in dev must not appear in train
```

### Deployable Stage C split

```text
target singer singing is forbidden as model input
target singer singing may appear only in evaluation metadata
oracle runs must be labeled oracle=true
deployable runs must be labeled oracle=false
```

### Language claim guard

The code must compute:

```text
num_singers_per_language_per_split
```

and block any result labeled `language_invariant=true` unless each evaluated language has at least two train singers and one held-out test singer.

## Feature-cache schema

Each feature file:

```text
{
  "x": [T, D],
  "times_sec": [T],
  "voiced_mask": [T],
  "phone_id": [T],
  "f0_hz": [T],
  "energy": [T]
}
```

Each cache metadata file:

```json
{
  "extractor_name": "wavlm_base_plus",
  "checkpoint_name": "...",
  "checkpoint_hash": "...",
  "git_commit": "...",
  "layer": 9,
  "sample_rate": 16000,
  "hop_sec": 0.02,
  "feature_dim": 768,
  "dtype": "float16",
  "normalization": "none",
  "input_manifest_hash": "...",
  "command": "..."
}
```

## Smoke tests

Run these before full extraction:

1. Extract features for 10 utterances.
    
2. Confirm no NaNs or infinities.
    
3. Confirm feature duration matches audio duration within one hop.
    
4. Confirm paired speech/singing rows share expected phrase/text metadata.
    
5. Confirm shuffled-label probe is near chance.
    
6. Confirm tiny probe can overfit 20 examples.
    
7. Confirm rerunning extraction gives identical cache paths and metadata hashes.
    
8. Confirm voiced mask is non-empty for voiced utterances.
    

## Unit tests

Required tests:

```text
test_manifest_required_columns
test_audio_paths_exist
test_duration_matches_num_samples
test_pair_mapping_valid
test_split_no_utt_overlap
test_split_no_song_leakage
test_split_no_singer_leakage_for_heldout_singer
test_deployable_no_target_singing_input
test_feature_cache_shapes
test_feature_cache_metadata_hash
test_phone_alignment_frame_mapping
test_probe_shuffled_labels_chance
test_bootstrap_groups_are_not_frames
test_results_schema
```

## Integration tests

Run one mini end-to-end experiment:

```text
10 singers or fewer
2 songs per singer
WavLM layer 6 only
utterance mean/std only
singer probe
mode probe
same-speaker verification
global residual baseline
```

Expected output:

```text
one metrics.json
one predictions.parquet
one result table row
one figure
```

## Result table format

Every result row must contain:

```text
run_id
stage
dataset_version
manifest_hash
split_name
split_hash
extractor
checkpoint_hash
layer_or_stream
representation
statistic_band
target
controls
probe_type
train_speakers
dev_speakers
test_speakers
train_items
test_items
metric_name
metric_value
ci_low
ci_high
p_value
seed
oracle_flag
deployable_flag
notes
```

## Prediction table format

```text
run_id
utt_id
pair_id
speaker_id
language
mode
technique
song_id
split
y_true
y_pred
score
baseline_name
system_name
oracle_flag
deployable_flag
```

## Reproducibility requirements

- Pin package versions.
    
- Record git commits.
    
- Record checkpoint hashes.
    
- Record manifest hash.
    
- Record split hash.
    
- Record random seeds.
    
- Record GPU model.
    
- Record command line.
    
- Save resolved config.
    
- Save all generated audio manifests.
    
- No result may depend on notebook state.
    
- All figures must be generated from saved result tables.
    

## Blocking failure modes

Stop immediately if any occurs:

1. Target singer appears in train for held-out-singer test.
    
2. Target singing is used in deployable inference.
    
3. Song/phrase leakage across splits.
    
4. More than 1% feature files contain NaNs.
    
5. Shuffled labels produce above-chance results.
    
6. F0/duration/energy baselines are missing from main comparisons.
    
7. Main confidence intervals are computed over frames instead of singers/songs.
    
8. ECAPA is the only identity metric.
    
9. Seed-VC checkpoint or code version is unpinned.
    
10. Sample-rate mismatch silently occurs.
    
11. Oracle results are mixed with deployable results.
    
12. Subjective evaluation is not blind or randomized.
    

---

# Part 7 — Paper strategy

## Possible titles

1. **A Core and a Residual? Controlled Analysis of Same-Person Speech-to-Singing Identity Shift**
    
2. **Speech References Are Not Singing References: Mode-Aware Timbre Residuals for Zero-Shot SVC**
    
3. **From Speaking Timbre to Singing Timbre: Auditing and Adapting Frozen Voice Representations**
    
4. **What Changes When a Speaker Sings? A Controlled Representation Study for Speech-Prompted SVC**
    
5. **Estimating Singing Timbre from Speech: A Frozen-Model Study of Same-Singer Mode Residuals**
    

## Abstract structure

Use five sentences:

1. **Problem:** zero-shot SVC often has target speech but not target singing.
    
2. **Hypothesis:** same-person speech and singing share identity cues but differ by a mode-specific residual.
    
3. **Method:** audit frozen SSL/codec representations on GTSinger with phone/F0/duration/energy/technique controls.
    
4. **Intervention:** train a small residual adapter and test it in frozen Seed-VC conditioning.
    
5. **Finding:** either modest improvement or a rigorous negative result showing which cues are not deployable.
    

## Contribution bullets

Strong version:

- A controlled audit of speech-vs-singing identity shift in frozen SSL and factorized-codec representations.
    
- A residual formulation separating oracle target-singing residuals from deployable target-speech-only estimates.
    
- A small frozen-model conditioning adapter for zero-shot SVC.
    
- An evaluation protocol with singer/song splits, acoustic controls, bootstrap uncertainty, and subjective identity tests.
    

Mixed-results version:

- A controlled leakage map showing where apparent identity residuals survive or collapse under nuisance controls.
    
- A quantified oracle-vs-deployable gap for speech-prompted SVC.
    
- Evidence that common speaker metrics and frozen features can overstate same-person speech/singing identity consistency.
    

## Related work organization

1. **Speech-prompted SVC and zero-shot singing conversion**  
    Seed-VC, Everyone-Can-Sing, SSANSVC, FreeSVC, SVCC 2025.
    
2. **Voice conversion via statistics and normalization**  
    AdaIN-VC, LoIN, instance statistics, local statistics.
    
3. **Factorized neural codecs**  
    FACodec / NaturalSpeech 3, FreeCodec, MSR-Codec.
    
4. **SSL representation probing and speaker leakage**  
    WavLM, ContentVec, speaker leakage/removal probes.
    
5. **Datasets and evaluation for singing voice**  
    GTSinger, SVCC 2025, singer identity metrics, human evaluation.
    

## Method section outline

1. Problem formulation  
    Define speech reference, singing source, target identity, mode residual.
    
2. Dataset and splits  
    GTSinger, pair construction, held-out song, held-out singer, language caveat.
    
3. Frozen representations  
    WavLM, ContentVec, ECAPA, optional FACodec streams, acoustic baselines.
    
4. Controlled residual audit  
    Temporal statistics, phone alignment, nuisance controls, probes, retrieval metrics.
    
5. Residual adapter  
    `z_core`, `z_mode`, oracle residual, deployable residual.
    
6. Downstream SVC intervention  
    Seed-VC baseline, conditioning hook, adapter variants.
    
7. Evaluation  
    Objective metrics, subjective tests, bootstrap and mixed-effects analysis.
    

## Tables and figures

### Table 1: Dataset and split summary

Rows:

- train/dev/test;
    
- singers;
    
- languages;
    
- songs;
    
- speech utterances;
    
- singing utterances;
    
- techniques.
    

### Figure 1: Hypothesis diagram

Show:

```text
speaker identity core
+ speech-mode residual
+ singing-mode residual
+ content/prosody/technique
```

Label it clearly as a modeling hypothesis.

### Figure 2: Leakage map

Heatmap:

```text
rows: WavLM layers / ContentVec / FACodec streams
columns: singer, mode, phone, F0, energy, technique, language, song/take
color: balanced accuracy or R²
```

### Figure 3: Controlled effect-size plot

Show residual effect before and after:

- no controls;
    
- F0 controls;
    
- F0 + duration;
    
- F0 + duration + phone;
    
- all controls.
    

### Figure 4: Cross-mode identity retrieval

Compare:

- speech embedding baseline;
    
- global residual;
    
- acoustic-only residual;
    
- deployable residual;
    
- oracle residual.
    

### Table 2: Stage A probe results

Include confidence intervals and shuffled-label controls.

### Table 3: Residual prediction

Columns:

- model;
    
- split;
    
- cosine with oracle residual;
    
- cross-mode retrieval improvement;
    
- gap closed;
    
- CI.
    

### Table 4: SVC objective evaluation

Rows:

- Seed-VC baseline;
    
- global residual;
    
- deployable residual;
    
- oracle residual;
    
- optional FreeSVC.
    

Columns:

- identity similarity;
    
- F0 correlation;
    
- content metric;
    
- quality metric;
    
- technique/style metric.
    

### Figure 5: Subjective evaluation

Pairwise preference or MOS with confidence intervals.

## Ablations

Required:

- WavLM layer 3/6/9/12;
    
- mean/std versus local statistics;
    
- voiced-only versus all frames;
    
- utterance-level versus phone-level;
    
- with/without F0 controls;
    
- with/without duration controls;
    
- with/without phone controls;
    
- WavLM versus ContentVec;
    
- WavLM versus FACodec streams if included;
    
- global residual versus language residual;
    
- oracle versus deployable residual;
    
- target speech reference duration;
    
- Seed-VC intervention strength `alpha`;
    
- held-out song versus held-out singer;
    
- technique-specific analysis.
    

## Limitations

State them aggressively:

- GTSinger has only 20 professional singers.
    
- Singer and language are confounded.
    
- Clean studio/professional singing may not generalize.
    
- Probe results do not prove disentanglement.
    
- ECAPA and similar metrics may be biased toward speech.
    
- Seed-VC intervention may be architecture-specific.
    
- Noncommercial dataset/model licenses may restrict release/use.
    
- Identity conversion has misuse risks; evaluation should use consented data only.
    

## Strongest narrative if results are positive

> Same-person speech and singing are neither identical nor unrelated in frozen audio representations. The difference is partly structured, survives key acoustic controls, and can be estimated well enough from target speech to improve speech-prompted SVC conditioning without retraining a large model.

## Strongest narrative if results are mixed

> Frozen models contain abundant recoverable mode and identity information, but much of the apparent speech-to-singing identity residual is entangled with F0, duration, technique, and language. Oracle residuals help, but deployable speech-only residuals expose a real information gap. This explains why speech-prompted SVC remains difficult and provides a controlled diagnostic protocol for future systems.

That mixed narrative is still respectable.

## Claims to avoid in the paper

- “Disentangled identity core.”
    
- “Causal residual.”
    
- “Language-independent.”
    
- “Universal across singers.”
    
- “SOTA SVC.”
    
- “Human-level singer similarity.”
    
- “Metric-proven identity preservation.”
    
- “Works without target singing” unless every inference path truly excludes target singing.
    
- “Codec stream X is content/prosody/timbre ground truth.”
    

---

# Part 8 — Final decision

## Go/no-go recommendation

**Go for Stage A. Do not commit yet to a full Stage C method paper.**

The idea deserves a serious first pass because GTSinger gives you rare paired speech/singing data with alignments and techniques. But the paper lives or dies on whether the residual survives controls and helps downstream conditioning. Without that, it is too close to existing probing and speech-prompted SVC work.

## First experiment to run this week

Run a **small but decisive Stage A pilot**:

1. Build the GTSinger manifest and pair table.
    
2. Create held-out-song and held-out-singer splits.
    
3. Extract WavLM Base+ layers 3/6/9/12 for a subset first.
    
4. Compute:
    
    - utterance mean/std;
        
    - voiced-only mean/std;
        
    - phone-level mean/std;
        
    - F0, energy, duration, MFCC baselines.
        
5. Evaluate:
    
    - same-singer speech-to-singing retrieval;
        
    - mode classification;
        
    - singer classification;
        
    - residual prediction from speech to singing;
        
    - same tests after F0/phone/duration/energy controls.
        
6. Include shuffled-label and acoustic-only baselines from day one.
    

Do this before touching Seed-VC.

## Result that would make me continue

Continue if you see:

- cross-mode same-singer retrieval above matched negative baselines;
    
- residual prediction beating global mean and acoustic-only residuals;
    
- improvement survives held-out songs;
    
- at least some effect survives held-out singers or within-language held-out-singer subsets;
    
- confidence intervals exclude zero under singer/song bootstrap;
    
- the result is not explained by F0, duration, phone distribution, energy, language, or song/take leakage.
    

A very good sign would be: WavLM mid/high layers or FACodec timbre/detail streams show a residual that improves cross-mode identity retrieval, while ContentVec suppresses it.

## Result that would make me pivot

Pivot if:

- F0/duration/energy baselines match the frozen-feature residual;
    
- the effect vanishes after phone-level controls;
    
- held-out-singer residual prediction collapses;
    
- language explains most of the signal;
    
- residual intervention helps ECAPA but hurts human identity/naturalness;
    
- oracle residual helps but deployable residual is noise.
    

A good pivot is still valuable:

> “Speech-to-singing identity shift in frozen representations is mostly prosodic and evaluation-metric-dependent.”

That could become a strong analysis paper.

## 30-day plan

### Days 1–7: manifest, splits, pilot cache

- Implement manifest and pair validation.
    
- Generate held-out-song and held-out-singer splits.
    
- Extract WavLM Base+ layers 3/6/9/12 on a subset.
    
- Compute acoustic baselines.
    
- Run smoke tests and shuffled-label controls.
    

Deliverable:

```text
manifest summary
split report
first leakage map
first cross-mode retrieval table
```

### Days 8–14: full Stage A

- Cache selected WavLM layers for all GTSinger.
    
- Add ContentVec.
    
- Add FACodec only if setup is painless.
    
- Run probe suite.
    
- Run nuisance-controlled analysis.
    
- Run cluster bootstrap.
    
- Decide Stage B go/no-go.
    

Deliverable:

```text
Stage A report with stop/go decision
```

### Days 15–21: Stage B residual modeling

- Implement oracle residual.
    
- Implement global/language/technique residual baselines.
    
- Implement acoustic-only predictor.
    
- Implement ridge or small-MLP deployable residual predictor.
    
- Evaluate oracle-vs-deployable gap in embedding/retrieval space.
    

Deliverable:

```text
residual prediction table
oracle/deployable gap figure
```

### Days 22–30: Stage C pilot

- Freeze Seed-VC.
    
- Locate the reference-conditioning hook.
    
- Implement the smallest possible residual injection.
    
- Generate a small test set:
    
    - baseline Seed-VC;
        
    - global residual;
        
    - deployable residual;
        
    - oracle residual upper bound.
        
- Run objective metrics.
    
- Run a tiny blinded subjective pilot if outputs are listenable.
    

Deliverable:

```text
Stage C pilot table
audio examples
paper/no-paper decision
```

Final blunt advice: **do not sell this as disentanglement. Sell it as controlled residual evidence plus a deployment-constrained intervention.** If Stage A survives controls and Stage C gives even modest listener-preferred gains, this is a real paper. If Stage A collapses under controls, you still have a useful negative result, but the method paper should be abandoned.