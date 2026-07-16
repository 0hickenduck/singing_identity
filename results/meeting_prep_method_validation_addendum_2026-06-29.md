# Meeting Prep Addendum: Method Validation After Re-reading Pro Suggestions

Date: 2026-06-29 JST

Purpose: correct and sharpen the meeting framing after re-reading `/home/bowen/bowen_lab/projects/singing_identity/pro_suggestions/`, especially `pro3_voice_representation_idea_review.md`, `pro5_research_direction_ranking.md`, and `pro7_ml_audio_research_workflow.md`.

## 1. The Residual Prediction Track Is Still Useful

The residual prediction experiment is not useless. The safer framing is:

> We are not decomposing the residual into literal percentages. We are doing incremental predictive accounting: if we add a covariate block, how much better can we predict the speech-to-singing representation residual under held-out-singer evaluation?

This matches the pro-model design. The original strong project was:

```text
Use frozen speech/audio embeddings as measuring instruments to quantify how a person's vocal identity representation changes between speech and singing, then test whether a lightweight speech-to-singing identity residual improves speech-reference SVC.
```

Operationally:

```text
Delta_z = z_singing - z_speech
```

Then compare residual predictors:

```text
M0: mean Delta_z
L1: F0/prosody + duration/energy
L2: L1 + language/vocal range metadata
L3: L2 + low-dimensional content/phone-class summaries
L4: L3 + technique labels
L5: L4 + acoustic proxy features
```

The interpretation should be:

```text
This covariate block contains predictive information about Delta_z.
```

Not:

```text
This covariate block is a clean causal component of timbre.
```

## 2. What We Must Name More Carefully

We should avoid vague factor names.

### F0 / Prosody

What we actually used:

```text
delta_f0_mean
delta_f0_std
delta_f0_range
delta_f0_voiced_pct
```

Better name:

> F0/prosody summary covariates.

Do not say:

> We decomposed out pitch.

### Rhythm / Timing

What we actually used:

```text
duration_ratio
delta_duration_sec
singing_phone_rate - speech_phone_rate
```

Better name:

> duration/timing proxies.

Do not say:

> We measured rhythm.

### Semantic / Content

What we actually used in v2:

```text
text_len_chars
phone_count
phone-class summaries: vowel/sonorant/consonant proportions
```

Better name:

> low-dimensional text/phone-distribution proxies.

Do not say:

> semantic information.

This is not semantic meaning. It is content/phonetic proxy information.

### Technique

What we actually used:

```text
GTSinger categorical technique label
```

Better name:

> technique-label covariate.

Do not say:

> technique dimension.

The label is a proxy. It does not guarantee the representation uses technique as a clean dimension.

### Acoustic Proxy

What we actually used:

```text
acoustic_baseline delta vector
```

Better name:

> acoustic/phonation proxy block.

Do not say:

> phonation factor.

It may include spectrum, F0, energy, duration, and other low-level acoustic structure.

## 3. The Correct Role Of SeedVC

The pro-model advice was clear: SeedVC should be a downstream intervention test for the residual hypothesis.

The right question is not:

> Does target singing prompt sound better than target speech prompt?

The better question is:

> If we predict a speech-to-singing residual from target speech, and inject or adapt SeedVC conditioning with that predicted residual, does it improve speech-reference SVC relative to unadapted speech reference, while staying below or approaching the target-singing oracle?

The comparison table should be:

| Condition | Meaning |
|---|---|
| speech reference | practical baseline |
| speech + global residual | simplest residual intervention |
| speech + learned/deployable residual | main proposed method |
| wrong-speaker residual | sanity control |
| random residual | sanity control |
| singing reference oracle | upper bound |

Our previous SeedVC prompt experiment is still useful, but only as a diagnostic:

```text
speech prompt and singing prompt produce different acoustic/audio outcomes
```

It does not yet evaluate the residual model. It mostly showed that prompt mode matters and that the change is confounded with singing-domain, phonation, and content delivery.

## 4. How To Turn Track 1 Into A More Complete Research Story

Current Track 1 result:

> Low-dimensional covariate blocks can predict part of Delta_z better than the mean residual baseline.

To make it research-complete, the next step should not be more decomposition language. It should run two linked branches:

```text
Branch 1: explicit predictive accounting
  Which named covariate blocks predict Delta_z?

Branch 2: deployable residual adapter
  Can we predict a useful singing-reference residual from target speech only?
```

These answer different questions. Branch 1 is explanatory. Branch 2 is closer to the pro-model downstream design.

### Step A: Clean Predictive Accounting

Use out-of-fold held-out-singer residual prediction and report:

```text
MSE improvement over M0
delta cosine
same-singer retrieval after adding predicted Delta_z
confidence intervals over speakers
per-block incremental gain
controls: shuffled technique, shuffled pairs, nuisance-only, metadata-only
```

Claim:

> These blocks provide predictive information about the speech-to-singing residual.

This branch can include covariates that are not deployable, as long as they are labeled correctly. For example:

```text
delta_f0_mean = singing_f0_mean - speech_f0_mean
```

uses the target singing side, so it is an analysis/oracle covariate, not something available when only target speech is given.

### Step B: Deployable Residual Adapter

Train a small predictor:

```text
delta_hat = A(z_speech, speech_acoustic_stats, speaker_embedding, optional priors)
z_sing_hat = z_speech + delta_hat
```

This follows the pro-model Stage B design:

```text
delta_hat_s = A(z_speech_s, acoustic_stats_s, maybe language/technique priors)
```

Allowed deployable inputs:

```text
target speech reference
speech-derived speaker embedding, e.g. ECAPA/CAMPPlus/SeedVC style embedding
speech F0/energy/duration/mel or representation statistics
training-set population residuals
metadata available at inference, if clearly declared
```

Disallowed for deployable claims:

```text
target singing audio
target singing F0/mel/acoustic stats
target singer fine-tuning on singing
test-set technique labels, unless explicitly marked as oracle metadata
```

Important distinction:

```text
speaker_id as a categorical ID
```

is only useful for training speakers or within-speaker analysis. It can memorize singer-specific residuals and will not generalize to unseen singers. For deployable held-out-singer work, use a continuous speech-derived speaker embedding or speech representation, not an arbitrary ID token.

Mel spectrogram or "mel structure" is also not a clean factor. It is a rich acoustic observation containing timbre, phonetic content, F0 traces, energy, recording/channel information, and maybe artifacts. If used, call it:

```text
speech-side acoustic/mel representation input
```

not:

```text
mel dimension of the residual
```

### Step C: Oracle Vs Deployable Residual

Define:

```text
oracle residual = z_target_singing - z_target_speech
deployable residual = predicted from target speech and population statistics, without target singing at inference
```

The pro-model explicitly emphasized this distinction.

What we have already done:

```text
partial residual prediction using explicit covariate blocks
comparison against mean Delta_z baseline
diagnostic SeedVC speech-prompt vs singing-prompt gap
```

What we have not yet fully done:

```text
train deployable adapter from target speech only
compare deployable residual against oracle residual
inject predicted residual into SeedVC reference conditioning
test speech + residual vs speech baseline vs singing oracle
```

### Step D: Downstream SeedVC Intervention

Freeze SeedVC. Modify only the reference-side conditioning if feasible.

Evaluate:

```text
speech reference
speech + predicted residual
wrong residual
random residual
singing reference oracle
```

Success is not "beats SOTA". Success is:

```text
closes a nontrivial part of the speech-reference to singing-oracle gap
without clear naturalness/content degradation
```

## 5. Track 2 Global-Vector Negative Result: What Was Actually Tested

The global-vector experiment used:

```text
paired phone intervals:
  base/control phone interval
  matching target technique phone interval

delta = z_target_technique_phone - z_control_phone
```

Filtering:

```text
removed silence/special phones
selected voiced phones
speaker-balanced round-robin sampling
breathy headline rows
non-breathy rows retained as wrong-technique controls
```

The speaker-balanced manifest has:

```text
3915 rows
3000 breathy rows
30 selected phone categories
19 speakers
8 languages
```

For each exact `phone + technique` group, the code:

1. computed mean delta direction from all but one pair in the group;
2. added that direction to the held-out control vector;
3. retrieved the correct target technique vector from the same group gallery;
4. compared against:
   - wrong-technique direction;
   - shuffled direction from other groups.

The key result:

| Representation | Breathy analogy top1 | Wrong-technique top1 | Shuffled top1 |
|---|---:|---:|---:|
| MERT L3 | 0.225 | 0.223 | 0.226 |
| WavLM L6 | 0.222 | 0.222 | 0.223 |
| WavLM L9 | 0.243 | 0.245 | 0.243 |

This supports the limited claim:

> Under exact-phone mean-delta vector analogy with speaker-balanced controls, breathy does not behave as one clean globally transferable direction.

It does not support the broader claim:

> No global breathy structure exists.

That broader claim would require stronger tests:

```text
global train/test direction across held-out singers
language-local vs global transfer matrix
phone-family rather than exact-phone grouping
projection/removal tests
fold-wise subspace stability
acoustic voice-quality grounding
```

## 6. Why The Track 2 Negative Claim Is More Trustworthy Than The Earlier False Negative

The earlier false negative in Track 1 came from a bad model setting:

```text
high-dimensional input
high-dimensional Delta_z target
small number of speakers
weak regularization / overfitting
```

The Track 2 global-vector negative result is different because it included:

```text
matched phone intervals
no training-heavy high-dimensional regression
speaker-balanced filtering
wrong-technique controls
shuffled-direction controls
leave-one-pair analogy within group
```

So it is a valid negative result for the method it tests.

But it is still not the final word, because the tested method is narrow:

```text
single mean vector per exact phone/technique group
mean/std interval pooling
nearest-target analogy top1
```

Thus the right conclusion is:

> The clean global-vector version failed. The next valid question is whether there is a partially shared global/local geometry.

## 7. Track 2 And Conditional Reconstruction / Adapter

The pro-model's Stage B/C design is not "just classify technique." It is:

```text
Stage B: learn small heads/adapters over frozen features
Stage C: test whether the learned residual/adaptation improves downstream conditioning
```

For technique, an analogous version would be:

```text
control representation + predicted technique residual -> technique representation
```

Or more cautiously:

```text
given control/breathy paired intervals, predict the breathy residual under held-out singer/phone/language conditions
```

Potential model:

```text
z_breathy_hat = z_control + g_theta(z_control, phone, language/singer-independent covariates)
```

Mandatory baselines:

```text
mean global breathy residual
phone-family residual
language-local residual
speaker-local residual, analysis-only
wrong-technique residual
random residual
oracle residual
```

This would make Track 2 more like the pro design: not just "is breathy decodable", but "can a deployable or semi-deployable adapter predict the technique residual?"

However, generation/intervention should only come after the representational adapter passes held-out tests. Otherwise it becomes another weak demo.

## 8. Recommended Next Step Before New Experiments

For the meeting, present the project as:

```text
Primary design:
  predict speech-to-singing residual from explicit covariate blocks;
  evaluate whether predicted residual improves downstream speech-reference conditioning.

Backup / linked design:
  for breathy, test whether technique residual is global/local and whether a small adapter can predict it.
```

The immediate next technical task after the meeting should be one of these, depending on the teacher's advice:

### If Teacher Likes Track 1

Implement a residual-intervention version of SeedVC:

```text
speech reference baseline
speech + predicted residual
speech + wrong residual
speech + random residual
singing reference oracle
```

This aligns with pro3/pro7.

### If Teacher Likes Track 2

Implement global/local technique residual prediction:

```text
control -> breathy residual prediction
held-out singer / held-out phone-family / held-out language tests
global vs language-local vs phone-local vs speaker-local comparison
projection/removal as non-generative validation
```

### If Teacher Likes Timbre Shrinkage

Reframe as identity-vs-technique tradeoff:

```text
speaker-ID separability before/after technique
technique-ID separability
variance explained by speaker vs technique
remove technique subspace and re-test speaker identity
```

This is conceptually attractive but needs careful data validation.

## 9. Updated Meeting Question To Ask

The most useful question for the teacher is:

> Should I keep the project centered on speech-to-singing residual prediction and downstream residual-adapted SVC, or pivot to the broader question of how singing technique reshapes speaker identity representation?

Then explain:

```text
The first path is closer to pro-model's original design and our current experiments.
The second path is conceptually richer but less validated and may need better data.
```
