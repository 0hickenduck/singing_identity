# Stage 1 Pilot Interpretation and Next Plan

Date: 2026-06-17

This note is a clearer interpretation of the first two experiment pilots. It corrects an important AUC interpretation issue from the earlier status memo.

## 1. What We Actually Ran

We ran two Stage 1 pilots.

### Track 1: Timbre / Speech-to-Singing Residual

Question:

> If we know a person's speech representation, can we predict how their representation shifts when they sing?

Data used:

- Local GTSinger subset.
- 39 speech/singing pairs.
- 18 singers.
- 78 utterances.

Representations tested:

- acoustic baseline;
- WavLM Base+ layers 3, 6, 9, 12.

Main probes:

- speech-vs-singing mode classifier;
- same-person speech/singing retrieval;
- speech-to-singing residual mapper;
- global mean residual baseline.

### Track 2: Technique Direction Discovery

Question:

> Can we find stable latent directions for singing techniques such as breathy, falsetto, and glissando?

Data used:

- 234 same-singer/same-phone control-vs-technique pairs.
- Reliable groups are phone-technique groups with at least 3 examples.

Representations tested:

- acoustic baseline;
- WavLM Base+ layers 3, 6, 9, 12;
- MERT v1 95M layers 3, 6, 9, 12.

## 2. How the Mode Classifier Was Trained

The classifier was trained by us. It is not a pretrained classifier.

For each utterance:

1. Load frozen feature sequence from a model layer, for example WavLM layer 12.
2. Compute utterance-level mean and standard deviation.
3. Concatenate them:

```text
x = [mean(feature_frames), std(feature_frames)]
```

For WavLM:

```text
feature_dim = 768
classifier input dim = 1536
```

Labels:

```text
speech = 0
singing = 1
```

Classifier:

```text
simple logistic regression implemented in NumPy
```

Training details:

- speaker-disjoint split;
- train rows: 58;
- test rows: 10;
- train speakers: 13;
- test speakers: 2;
- gradient descent: 1200 steps;
- learning rate: 0.05;
- small L2 penalty: `1e-4`.

Important limitation:

The current test split is tiny. It contains only two held-out singers:

- `IT-Bass-1`;
- `ZH-Tenor-1`.

So the AUC result is unstable and should be rerun with speaker-fold cross-validation.

## 3. How the Control Was Done

The current control is a mathematical linear residualization.

For each utterance, nuisance variables were:

```text
f0_mean_hz
f0_std_hz
f0_voiced_pct
energy_mean
energy_std
duration_sec
rms_db
```

Let:

```text
X = frozen representation statistics, shape [N, D]
Z = nuisance matrix, shape [N, 7]
```

The code adds an intercept:

```text
Z_aug = [1, Z]
```

Then fits:

```text
beta = pinv(Z_aug) @ X
```

and removes the nuisance-predictable component:

```text
X_residual = X - Z_aug @ beta
```

So yes: this is directly subtracting the linear subspace explained by F0, energy, duration, voiced ratio, and RMS.

This is fairly aggressive, but only linear. It does not remove nonlinear nuisance effects.

Important limitation:

The current implementation fits the residualization using all rows before splitting. For a final experiment, we should fit the nuisance projection on train only and apply it to dev/test. The current result is useful as a pilot but not a final claim.

## 4. Correct AUC Interpretation

Earlier interpretation was too simple.

AUC means:

```text
P(score_positive > score_negative)
```

Here:

```text
positive = singing
negative = speech
```

So:

- `AUC = 1.0`: singing scores are always higher than speech scores.
- `AUC = 0.5`: random ordering.
- `AUC = 0.0`: singing scores are always lower than speech scores.

Therefore, an AUC of `0.2` is not "no signal".

It means:

```text
the classifier's score is strongly ordered, but often in the wrong direction
```

If we only care about separability:

```text
separability_auc = max(AUC, 1 - AUC)
```

Then:

```text
AUC = 0.2 -> separability_auc = 0.8
```

This is above the original `0.75` threshold.

But it does not mean the original classifier generalized correctly, because the label orientation flipped on held-out singers.

Current better interpretation:

> After controls, WavLM still contains a speech/singing-separating direction, but the direction is not stable enough under the current tiny held-out-speaker split.

## 5. Why All Controlled AUCs Were 0.2

This is suspicious and should not be overinterpreted.

Likely reasons:

1. The test set is tiny: only 10 utterances.
2. The test set contains only 2 singers.
3. AUC resolution is coarse. With about 5 singing and 5 speech items, each pairwise ordering changes AUC by about `0.04`.
4. The same held-out singer dominates the errors across layers.
5. The residualized representations across WavLM layers may share a similar remaining nuisance or singer-pocket effect.

From saved predictions:

- `ZH-Tenor-1` mostly behaves correctly.
- `IT-Bass-1` singing examples mostly receive very low singing scores.

So this is not a clean "all layers prove the same thing" result. It is a small-split artifact plus a potentially real reversed direction.

Next fix:

Run leave-singer-out or K-fold speaker-disjoint evaluation and report:

```text
signed AUC
separability AUC = max(AUC, 1-AUC)
orientation stability across folds
per-singer AUC
```

## 6. Residual Mapper and Global Residual

For each speech/singing pair:

```text
h_speech = representation of speech utterance
h_singing = representation of singing utterance
delta_true = h_singing - h_speech
```

The mapper predicts:

```text
delta_pred = ridge_model(h_speech)
```

The mapper score is:

```text
cosine(delta_pred, delta_true)
```

The global residual baseline is:

```text
delta_global = mean(delta_true over training pairs)
```

Then the baseline score is:

```text
cosine(delta_global, delta_true_test)
```

So yes, the global residual is basically the average speech-to-singing shift in the training set.

This is not a bad result. If global residual performs similarly to the learned mapper, it means:

> a population-level speech-to-singing direction may exist.

What it does not show yet:

> a personalized mapper improves beyond the global direction.

Better interpretation:

The global residual should become a serious baseline and maybe even a simple method. We should not discard it.

## 7. Track 1 Current Result

What we gained:

1. Raw speech vs singing is perfectly separable in WavLM.
2. After linear nuisance control, separability may still exist, but orientation is unstable.
3. WavLM layer 12 has the best cross-mode identity retrieval:

```text
Recall@1 = 0.333
Recall@5 = 0.833
chance Recall@1 ~= 0.056
```

4. The global speech-to-singing residual is competitive with the learned mapper.

What this means:

Track 1 is not dead. The stronger direction is probably:

> study cross-mode identity survival and population-level speech-to-singing shifts before claiming personalized residual prediction.

## 8. How Technique Direction Was Tested

For Track 2, each pair is:

```text
same singer
same phone
control technique vs target technique
```

For each pair:

```text
delta = feature(target technique phone interval) - feature(control phone interval)
```

For each phone and technique, for example:

```text
phone = o_it
technique = breathy
```

we average deltas:

```text
direction = mean(delta)
```

Then we bootstrap:

1. sample 80% of examples with replacement;
2. recompute direction;
3. compute angle between bootstrap direction and full direction;
4. repeat 1000 times.

Small bootstrap angle means the direction is stable.

Pilot threshold:

```text
mean bootstrap angle <= 30 degrees
```

## 9. Track 2 Current Result

Examples:

- WavLM has a few near-stable pockets, such as Italian `o` breathy or Italian `a` falsetto.
- MERT did not clearly outperform WavLM on this small subset.
- Most median angles were around 37-42 degrees.

Current interpretation:

> There may be local technique directions, but the current subset is too sparse and too concentrated by singer/language/phone to justify latent steering yet.

This does not kill the technique direction idea. It says we need more examples and stronger validation.

## 10. Better Technique Validation To Add

The current bootstrap-angle test is only one test.

We should add analogy-style tests:

```text
control_A + delta_technique_from_B should become closer to technique_A
```

Possible tests:

1. Within-technique delta consistency:

```text
cos(delta_i, delta_j) within same technique
```

should be higher than:

```text
cos(delta_i, delta_k) across different techniques
```

2. Directional retrieval:

```text
feature(control_phone) + delta_technique
```

should retrieve the matching target-technique phone better than random or wrong-technique directions.

3. Permutation control:

Shuffle technique labels. Stable directions should disappear.

4. Singer-held-out direction transfer:

Estimate direction on training singers, test on held-out singers.

These are closer to the vector-analogy logic.

## 11. Revised Plan Before Proceeding

### Step 1: Fix reporting for Track 1

Add metrics:

- signed AUC;
- separability AUC;
- per-singer AUC;
- orientation stability across folds;
- nuisance-only baseline;
- train-only residualization.

Do not use the current single split as final evidence.

### Step 2: Rerun Track 1 with better split

Use:

- leave-one-singer-out or 5-fold singer-disjoint split;
- WavLM layers 3, 6, 9, 12;
- MERT;
- ContentVec or HuBERT;
- speaker/singer identity embedding if available.

Decision criteria:

- identity retrieval above chance with confidence intervals;
- separability survives controls;
- orientation stable or explicitly interpreted as reversed direction;
- global residual beats random/acoustic-only residual;
- mapper must beat global residual to claim personalization.

### Step 3: Treat global residual as a valid candidate

New candidate claim:

> a population-level speech-to-singing shift exists in frozen representations and may help as a simple conditioning prior.

This is weaker than personalized residual prediction but may be more robust.

### Step 4: Improve Track 2 data

Need more same-phone control-vs-technique examples, especially:

- vibrato;
- falsetto;
- glissando;
- breathy;
- control singing.

The current local GTSinger subset skipped many missing wavs. Data expansion is probably the highest-value next action.

### Step 5: Add analogy tests for Track 2

Before latent steering, require:

- stable bootstrap angle;
- within-technique delta similarity above between-technique similarity;
- directional retrieval above random/wrong-technique baselines;
- held-out-singer transfer.

### Step 6: Only then proceed to Stage 2/3

Proceed to Seed-VC or latent steering only if the corrected Stage 1 gates pass.

## 12. Current Research Decision

Do not desert the project.

Also do not claim success yet.

The project should be reframed as:

> controlled measurement of cross-mode identity and technique structure in frozen audio representations.

The most promising current finding is:

> WavLM layer 12 preserves cross-mode identity surprisingly well, while naive speech/singing mode detection is strongly affected by acoustic controls.

The most promising method direction is:

> population/global speech-to-singing residual first; personalized mapper only after more data.

