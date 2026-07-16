# Meeting Prep: What We Have Done And Where The Research Stands

Date: 2026-06-29 JST

Purpose: prepare for a one-on-one meeting. This memo intentionally slows down the project. It does not propose running more experiments immediately. It explains what we actually did, what the methods mean, what they do not prove, and what research directions are still coherent.

## 1. One-Sentence State Of The Project

We started with a broad idea about singing identity and timbre, but the experiments have narrowed it into a more precise problem:

> Singing changes frozen audio representations in ways that are partly acoustic/prosodic/phonation-related, and breathy phonation is encoded in those representations, but neither result currently supports a clean disentangled identity/technique decomposition or a single global technique vector.

So the project is not "failed", but the original claims were too broad. We are now deciding which narrower research question is worth keeping.

## 2. The Two Tracks We Actually Ran

### Track 1: Speech-To-Singing Representation Mismatch

Original intuition:

> Same-person speech and same-person singing should share identity, but they are not equivalent references. The residual between speaking and singing representations may contain timbre, pitch/prosody, rhythm, phonation, technique, and content-delivery differences.

Operational object:

```text
z_speech  = frozen representation of a same-person speech reference
z_singing = frozen representation of a same-person singing reference
Delta_z   = z_singing - z_speech
```

This `Delta_z` is what we called the speech-to-singing residual or prompt-mode mismatch.

Important clarification:

`Delta_z` is not pure timbre. It is a mixed representation movement caused by many factors:

```text
pitch / F0
duration / rhythm
energy / loudness
voicing
spectral shape / phonation
phone or content distribution
singing technique label
speaker-specific singing style
model-specific representation behavior
```

So Track 1 was not a true disentanglement experiment. It was a residual accounting experiment: can measurable factors predict part of the speech-to-singing representation gap?

### Track 2: Singing Technique, Mainly Breathy

Original intuition:

> Some singing techniques may appear as global directions in the latent or representation space, similar to concept directions in CV.

Because data was limited, we focused on breathy.

Operational object:

```text
z_control = frozen representation of a control/neutral phone interval
z_breathy = frozen representation of a matched breathy phone interval
delta     = z_breathy - z_control
```

The first question was whether `delta` has one clean global direction. The answer was no under strict controls.

The second question was whether breathy information is encoded at all. The answer was yes.

## 3. Track 1: What We Did Technically

Data and representation setup:

- Dataset: GTSinger same-singer/same-text speech/singing pairs.
- Representations: frozen WavLM, HuBERT, MERT, and acoustic baseline features.
- Main target: `Delta_z = z_singing - z_speech`.
- Split principle: speaker-disjoint evaluation, so the model must generalize to held-out singers.

We tested models that predict `Delta_z` from different groups of covariates.

The conceptual model ladder was:

```text
M0: average Delta_z only
L1: F0/prosody + duration/energy nuisance variables
L2: L1 + metadata such as language / vocal range
L3: L2 + low-dimensional content / phone-class summaries
L4: L3 + singing technique label
L5: L4 + acoustic proxy features, when target is non-acoustic
```

The first version failed in a misleading way:

```text
target: full high-dimensional Delta_z
input: high-dimensional exact phone histogram + metadata + technique
data: only around 20 speakers for speaker-disjoint generalization
```

This caused overfitting / model misspecification. That is why we saw the strange result where richer models could be worse than the average baseline.

The corrected v2 used:

```text
target: top-K PCA coordinates of Delta_z, K = 8/16/32/64
input: lower-dimensional covariates
model: multi-output ridge with strong regularization
split: still speaker-disjoint
```

The best v2 results were:

| Representation | Best K | Best model | MSE improvement over M0 | M0 R@1 | Best R@1 |
|---|---:|---|---:|---:|---:|
| WavLM L6 | 64 | L1 nuisance | +0.0725 | 0.5225 | 0.5863 |
| WavLM L9 | 16 | L5 acoustic proxy | +0.0865 | 0.5443 | 0.5970 |
| HuBERT L12 | 64 | L5 acoustic proxy | +0.0717 | 0.6302 | 0.6943 |
| MERT L3 | 64 | L5 acoustic proxy | +0.0621 | 0.5285 | 0.5647 |
| MERT L12 | 64 | L5 acoustic proxy | +0.0574 | 0.4258 | 0.5118 |

Interpretation:

1. The corrected model does predict part of the speech-to-singing representation movement.
2. The strongest easy explanation is low-dimensional acoustic/prosodic/phonation-related structure.
3. Technique labels add some information but are not the dominant explanatory factor.
4. This is still not a decomposition into exact ratios such as "40 percent identity, 30 percent rhythm, 30 percent technique."

## 4. What Track 1 Does Not Prove

Track 1 does not prove:

- a causal decomposition of singing identity;
- a clean separation of timbre, rhythm, phonation, and technique;
- that we can edit or control those factors downstream;
- that a specific ratio of identity-vs-technique dimensions has been measured;
- that SeedVC identity transfer is improved by the predicted residual.

The most honest Track 1 claim is:

> Frozen representations contain a predictable speech-to-singing mismatch. A stable part of this mismatch is explained by low-dimensional acoustic/prosodic/phonation-related covariates under held-out singer evaluation.

The weaker or unsafe claim would be:

> We decomposed singing timbre into semantic, rhythmic, prosodic, and technique components.

We have not done that.

## 5. SeedVC: What It Added And Why It Is Not The Main Proof

SeedVC was used as a downstream sanity check:

```text
source singing + target speech prompt
source singing + target singing prompt
```

The 30-pair prompt experiment showed:

- singing prompt moved outputs closer to target singing acoustic profile;
- singing prompt did not clearly move outputs closer to target speech profile;
- human listening heard a real difference, but mostly as content delivery / breathy phonation / acoustic style, not clean target identity.

Key acoustic result:

| Condition | Distance to target singing | Distance to target speech | F0 std | Spectral tilt low/high |
|---|---:|---:|---:|---:|
| target singing prompt vs speech prompt | -1.4452 | +0.0265 | +8.20 | +3.0202 |

The component ablation suggested:

```text
mel + style is close to oracle singing prompt
prompt_seq_only is weak or unreliable
```

But this is engineering diagnosis, not a core scientific proof.

Current SeedVC verdict:

> SeedVC confirms that prompt mode has audible and acoustic consequences, but the effect is closer to domain/phonation/content-delivery than clean identity.

This supports slowing down rather than doing more SeedVC ablations.

## 6. Track 2: What We Did Technically

We built breathy-control phone interval pairs and tested three levels of claim.

### 6.1 Global Vector / Direction

Question:

> Does breathy correspond to one transferable vector direction?

Method:

```text
delta = z_breathy - z_control
test vector analogy / nearest retrieval
compare against wrong-technique and shuffled-direction controls
```

Strict result:

| Representation | Breathy analogy top1 | Wrong-technique top1 | Shuffled top1 |
|---|---:|---:|---:|
| MERT L3 | 0.225 | 0.223 | 0.226 |
| WavLM L6 | 0.222 | 0.222 | 0.223 |
| WavLM L9 | 0.243 | 0.245 | 0.243 |

Verdict:

> A single global breathy vector is not supported.

This is a real negative result. It is useful because it prevents overclaiming.

### 6.2 Breathy Detection / Encoding

Question:

> Even if breathy is not one global vector, is breathy information encoded in the frozen representation?

Method:

```text
train simple linear classifier / direction
label: breathy vs control
split: held-out speakers
condition: raw and residualized against duration/F0/energy nuisance
control: shuffled labels
```

Results:

| Representation | Residualized AUC | Accuracy | Shuffled AUC |
|---|---:|---:|---:|
| WavLM L6 | 0.8238 | 0.7377 | 0.5173 |
| HuBERT L12 | 0.8198 | 0.7307 | 0.5115 |
| MERT L3 | 0.8093 | 0.7370 | 0.5167 |
| WavLM L9 | 0.7579 | 0.6837 | 0.5077 |
| MERT L12 | 0.7300 | 0.6587 | 0.5168 |
| Acoustic baseline | 0.6916 | 0.6354 | 0.5092 |

Interpretation:

1. Breathy is encoded in frozen representations.
2. It is not just F0/duration/energy, because residualization did not collapse AUC.
3. Acoustic baseline is weaker than WavLM/HuBERT/MERT, so SSL features contain extra useful information.
4. MERT is competitive but not uniquely better than speech SSL models.

### 6.3 Subspace Probe

Question:

> Is breathy captured by a broader subspace rather than a one-dimensional vector?

Method:

```text
learn top-K subspace from breathy-control deltas
measure held-out delta energy captured by the subspace
compare with wrong-technique and random subspaces
```

K=64 results:

| Representation | Breathy subspace capture | Wrong-technique capture | Random capture |
|---|---:|---:|---:|
| WavLM L6 | 0.6116 | 0.5644 | 0.0420 |
| HuBERT L12 | 0.7017 | 0.6547 | 0.0418 |
| MERT L3 | 0.6489 | 0.5877 | 0.0415 |

Interpretation:

Breathy subspace is much stronger than random, but only modestly stronger than wrong-technique subspace. That suggests:

```text
there is a broad technique / phonation / phone-realization subspace
but breathy-specific geometry is not yet clean
```

## 7. What Track 2 Does Not Prove

Track 2 does not prove:

- one global breathy vector;
- a controllable breathy steering direction;
- that breathy is novelly classifiable, because related work already uses SSL representations for singing phonation classification;
- that music-oriented models uniquely encode technique better than speech models.

The most honest Track 2 claim is:

> Breathy phonation is linearly detectable in frozen SSL representations under held-out speaker evaluation and remains detectable after simple nuisance residualization. However, this information does not appear as one clean globally transferable vector direction.

The next Track 2 question, if we continue, should be:

> Is breathy global, language-local, singer-local, phone-local, or partially shared?

## 8. What This Research Realm Is About

This project sits between four research areas.

### 8.1 Representation Probing

Question:

> What information is encoded in frozen representations?

Common methods:

```text
linear probes
speaker-held-out splits
shuffled-label controls
acoustic baselines
layer-wise comparison
```

Our Track 2 detection result belongs here.

Risk:

```text
high AUC only means decodable
it does not prove causality, editability, or a clean semantic vector
```

### 8.2 Concept Geometry / Latent Directions

Question:

> Does a concept correspond to a direction or subspace in representation space?

Common methods:

```text
concept activation vectors
mean-difference directions
PCA/SVD/cPCA
projection/removal tests
transfer across held-out groups
```

Our original "global breathy vector" idea belongs here.

Risk:

```text
PCA finds variance, not meaning
linear direction can pool many local effects
```

### 8.3 Voice Quality / Phonation

Question:

> How are breathiness, pressed voice, flow phonation, resonance, and other voice qualities represented acoustically and perceptually?

Relevant acoustic concepts:

```text
spectral tilt
HNR
CPP / CPPS
H1-H2
H1-A3
aperiodicity / noise
```

Our current acoustic baseline is still too simple. It has F0/duration/energy/spectrum summaries, but not enough classic voice-quality measures.

### 8.4 Singing Voice Conversion / Singing Style Conversion

Question:

> Can a model convert singer identity or singing style?

This is where SeedVC and SVCC-style work live.

Our current stance:

```text
SeedVC is useful as a sanity check and demo
but not a clean scientific proof of identity or technique decomposition
```

## 9. The "Timbre Shrinkage" Idea From The Teacher

Teacher's possible insight:

> Professional opera singers may have a more uniform way of expressing timbre or technique. Their personal timbre may shrink, while technique/genre conventions become stronger.

This could connect Track 1 and Track 2.

Possible formal hypothesis:

> Under singing technique, speaker identity separability decreases while technique/phonation separability increases.

Or:

> Professional technique compresses individual timbre variation into a smaller singer-identity subspace and aligns singers along shared technique/phonation dimensions.

How this could be measured, in principle:

```text
speaker separability in speech vs singing vs technique-specific singing
speaker variance / total variance ratio
technique variance / total variance ratio
within-speaker vs between-speaker distances
effective rank of speaker identity subspace
classification tradeoff: speaker-ID accuracy vs technique-ID accuracy
projection/removal: remove technique subspace and test speaker-ID recovery
```

But current limitations:

```text
we do not have a clean skill/professional-quality scale
we have limited singers
GTSinger technique labels are categorical, not technique proficiency scores
we cannot yet say stronger technique predicts stronger identity shrinkage
```

So this is a promising unifying idea, but not yet an established result.

## 10. Why "Ratio Decomposition" Is Tempting But Dangerous

The user idea:

> Maybe the speech-to-singing residual can be expressed as ratios: speaker-ID dimensions drop, technique dimensions rise, rhythm/prosody dimensions explain another part.

This is conceptually attractive, but it requires very careful definition.

Possible safer versions:

### Option A: Variance Decomposition

Fit a model or ANOVA-like decomposition:

```text
representation ~ speaker + language + phone + technique + acoustic variables
```

Then report variance explained by each factor.

Risk:

```text
factors are correlated
variance shares are not causal
small number of speakers makes estimates unstable
```

### Option B: Probe-Based Information Ratios

Measure how well different factors can be decoded:

```text
speaker-ID accuracy
technique-ID accuracy
phone accuracy
language accuracy
F0/energy prediction
```

Compare speech vs singing vs technique subsets.

Risk:

```text
probe accuracy depends on model capacity and split
not a direct dimension ratio
```

### Option C: Subspace Removal / Projection

Learn speaker, technique, and acoustic subspaces. Then test:

```text
remove technique subspace -> does speaker-ID improve or degrade?
remove speaker subspace -> does technique decoding remain?
project onto technique subspace -> does breathy remain but identity drop?
```

This is closer to the "dimension ratio" intuition, but still not a literal decomposition.

Current verdict:

> A ratio story may become a research direction, but we should not claim it yet. We first need a well-defined measurement of speaker identity subspace and technique/phonation subspace.

## 11. What We Should Probably Tell The Teacher

A concise meeting version:

> I started with two broad tracks. Track 1 asked whether same-person speech and singing references differ in frozen representations and whether that residual can be explained. We found a predictable speech-to-singing mismatch, mainly explained by low-dimensional acoustic/prosodic/phonation variables, but we have not achieved true disentanglement of timbre, rhythm, and technique.

> Track 2 asked whether singing technique, especially breathy, forms a global vector. The strict global-vector result is negative: breathy does not behave like one clean transferable direction. But breathy is still clearly encoded in WavLM/HuBERT/MERT under held-out speaker tests, even after residualizing simple F0/duration/energy variables.

> The most interesting new direction may be to connect these: singing technique may reshape identity representation. Instead of asking only "what is the residual made of", we could ask whether technique/phonation dimensions compete with or compress speaker identity dimensions in singing.

Then ask:

1. Is the identity-compression / timbre-shrinkage framing interesting to you?
2. Should I keep Track 1 as prompt-mode mismatch accounting, or pivot toward identity-vs-technique representation geometry?
3. Is a negative result about no global breathy vector acceptable if paired with a positive result about decodability and local/global structure?
4. Should SeedVC be kept only as a downstream sanity check rather than a main research claim?
5. Do we need a dataset with technique proficiency or professional/non-professional contrast to make the timbre-shrinkage hypothesis meaningful?

## 12. What Not To Do Before The Meeting

Do not run more experiments just to fill time.

Do not continue SeedVC component ablations as if they are the main research proof.

Do not claim:

```text
we disentangled singing timbre
we found a global breathy vector
MERT is clearly better for technique
SeedVC singing prompt improves identity
```

Do say:

```text
we found predictable speech-to-singing mismatch
we found breathy decodability but no clean global vector
we found that subjective identity judgment is hard and confounded with phonation/content delivery
we need to choose the next research question before running more
```

## 13. Recommended Next Direction Choices

### Choice 1: Conservative Track 1 Paper

Question:

> How do frozen representations differ between same-person speech and singing references?

Main evidence:

```text
Delta_z accounting
low-dimensional acoustic/prosodic covariates
SeedVC sanity check
```

Pros:

- closest to what we already did;
- grounded in existing results;
- less risky.

Cons:

- may feel descriptive;
- not enough novelty if framed as "many factors affect the residual."

### Choice 2: Track 2 Global/Local Technique Geometry

Question:

> Is breathy phonation represented globally or locally in frozen SSL models?

Main evidence:

```text
breathy detection
global vector failure
language/singer/phone local geometry
projection/removal tests
```

Pros:

- clearer technical question;
- negative and positive results can coexist;
- more methodologically crisp.

Cons:

- classification alone is not novel;
- needs careful controls to avoid another shallow probe paper.

### Choice 3: Identity-Technology Tradeoff / Timbre Shrinkage

Question:

> Does singing technique compress or reshape speaker identity representation?

Main evidence needed:

```text
speaker-ID separability across speech/singing/technique
technique-ID separability
variance/subspace analysis
possibly professional or skill-level metadata
```

Pros:

- best conceptual connection between teacher insight, Track 1, and Track 2;
- potentially more interesting than residual accounting.

Cons:

- currently least validated;
- may need better data or new labels.

## 14. My Current Recommendation

For the meeting, do not present this as a nearly finished experiment. Present it as a fork.

My recommended framing:

> The current evidence suggests that "decomposing singing timbre" is too broad. We have two grounded facts: speech-to-singing mismatch is partly predictable from acoustic/prosodic/phonation features, and breathy phonation is encoded but not as a clean global vector. The next research decision is whether to make this a paper about prompt-mode mismatch, global/local phonation geometry, or identity compression under singing technique.

My preference after reviewing the evidence:

1. Drop SeedVC as main proof; keep it as sanity check.
2. Do not claim full residual decomposition.
3. Keep Track 2's global/local breathy geometry as the cleanest methodological next step.
4. Discuss with the teacher whether the richer conceptual story should become identity-vs-technique tradeoff or timbre shrinkage.

## 15. Source Artifacts

Main local reports:

- `/home/bowen/bowen_lab/projects/singing_identity/results/experiment_debug_verdict_zh_2026-06-28.md`
- `/home/bowen/bowen_lab/projects/singing_identity/results/human_verdict_and_objective_followup_zh_2026-06-28.md`
- `/home/bowen/bowen_lab/projects/singing_identity/results/breathy_global_local_method_survey_2026-06-28.md`
- `/home/bowen/bowen_lab/projects/singing_identity/results/seedvc_30pair_prompt_gap_posthoc_zh_2026-06-28.md`

Main run roots:

- `/localdisk/bowen/singing_identity/runs/track1_prompt_mismatch_accounting_v2_2026-06-28`
- `/localdisk/bowen/singing_identity/runs/track2_breathy_detection_probe_2026-06-28`
- `/localdisk/bowen/singing_identity/runs/track2_breathy_subspace_probe_2026-06-28`
- `/localdisk/bowen/singing_identity/runs/track1_seedvc_prompt_baseline_30pairs`
- `/localdisk/bowen/singing_identity/runs/track1_seedvc_component_ablation_12pairs`

Relevant scripts:

- `/home/bowen/bowen_lab/projects/singing_identity/scripts/probing/run_prompt_mismatch_accounting_v2.py`
- `/home/bowen/bowen_lab/projects/singing_identity/scripts/probing/run_breathy_detection_probe.py`
- `/home/bowen/bowen_lab/projects/singing_identity/scripts/probing/run_technique_subspace_probe.py`
- `/home/bowen/bowen_lab/projects/singing_identity/scripts/intervention/run_seedvc_prompt_baseline.py`
- `/home/bowen/bowen_lab/projects/singing_identity/scripts/intervention/run_seedvc_component_ablation.py`

External reference anchors:

- TCAV concept vectors: https://proceedings.mlr.press/v80/kim18d.html
- Probe selectivity controls: https://arxiv.org/abs/1909.03368
- INLP / linear erasure: https://arxiv.org/abs/2004.07667
- GANSpace latent PCA directions: https://arxiv.org/abs/2004.02546
- InterFaceGAN supervised latent directions: https://openaccess.thecvf.com/content_CVPR_2020/papers/Shen_Interpreting_the_Latent_Space_of_GANs_for_Semantic_Face_Editing_CVPR_2020_paper.pdf
- Contrastive PCA: https://www.nature.com/articles/s41467-018-04608-8
- Aspiration probing in HuBERT: https://www.isca-archive.org/interspeech_2023/martin23_interspeech.html
- voice2mode singing phonation classification: https://arxiv.org/html/2602.13928v1
- SVCC 2025 singing style conversion: https://www.vc-challenge.org/

