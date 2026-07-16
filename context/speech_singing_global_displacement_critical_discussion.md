# Speech–Singing Global Displacement
## Critical Discussion, Conceptual Corrections, and Open Questions

### Document role

This document is a companion to the existing experiment report. It does **not** restate the experiment chronology, implementation details, run names, tables, or complete numerical results already recorded elsewhere.

Its purpose is to preserve the critical discussion that followed the experiments:

- what the current evidence actually establishes;
- which interpretations were initially too strong;
- which results are mainly diagnostic rather than substantive;
- what remains confounded or metric-dependent;
- what scientific value the result may or may not have;
- which next analyses would most efficiently distinguish the competing explanations.

The intended use is to provide this document together with the main experiment report to another researcher or AI system for continued analysis.

---

## 1. Revised conceptual picture

The current project should be understood through four distinct objects.

### 1.1 Utterance representation

A frozen SSL model produces a sequence of frame-level hidden states for each utterance. The current pooling compresses that sequence into one utterance vector using temporal mean and temporal standard deviation.

This pooling has a practical interpretation:

- temporal mean summarizes the average activation state;
- temporal standard deviation preserves a limited measure of how much each representation dimension varies over time.

However, mean-plus-standard-deviation pooling does not preserve full temporal structure. It discards ordering, contour shape, local transitions, rhythm, vibrato trajectory, and many other sequence properties. It is therefore a compact summary, not a neutral or complete representation of the utterance.

### 1.2 Speaker–mode centroid

For a particular speaker and mode, multiple utterance vectors are averaged to obtain a speech centroid or singing centroid.

Centroid averaging reduces utterance-level noise and makes stable speaker- or mode-associated differences easier to observe. It does **not** remove F0, duration, energy, phonation, content, language, channel, or style. Stable differences in those factors may become more visible after averaging.

### 1.3 Individual speech-to-singing displacement

For speaker \(i\),

\[
\Delta_i = C_i^{\mathrm{singing}} - C_i^{\mathrm{speech}}.
\]

This is a difference vector between two speaker–mode centroids. It should preferably be called an **individual displacement**, not a residual, because “residual” was also used for nuisance-regression outputs earlier in the project.

### 1.4 Shared mean displacement

Across training speakers,

\[
\mu_\Delta = \frac{1}{N}\sum_i \Delta_i.
\]

This is a speaker-balanced difference between the average singing centroid and the average speech centroid.

It is a first-order domain mean translation. It is not:

- an identity-conditioned estimator;
- a personalized mapping;
- a pure timbre vector;
- a complete singing representation;
- a proof of disentanglement.

The most useful decomposition is

\[
\Delta_i = \mu_\Delta + \epsilon_i,
\]

where \(\epsilon_i\) is the speaker-specific deviation from the shared mean.

---

## 2. Important terminology and interpretation corrections

Several phrases used earlier in the project should be weakened or replaced.

### 2.1 “Speech and singing are linearly perfectly separable”

The evidence is more limited:

> Under the evaluated speaker-centroid protocol, a simple linear probe achieves near-perfect held-out mode discrimination.

This does not establish perfect separation of all speech and singing frames or utterances in the population. It is centroid-level, dataset-specific, and dependent on the evaluated representation and aggregation procedure.

### 2.2 “The correction removes mode”

The symmetric correction cancels the train-estimated average speech–singing displacement. A safer interpretation is:

> It suppresses the dominant centroid-level, linearly decodable mean displacement associated with mode.

It does not demonstrate removal of all mode information, nonlinear mode information, frame-level singing structure, pitch trajectories, phonation, rhythm, or technique.

### 2.3 “The shared mean explains 65–83% of the variance”

The reported quantity is better understood as a fraction of **uncentered displacement energy** attributable to the shared mean.

It is not:

- identity variance explained;
- retrieval improvement explained;
- variance explained after centering;
- proof that the remaining structure is negligible.

The exact numerator and denominator should be explicitly stated in the final technical description.

### 2.4 “The residual is low-rank” or “the global vector has rank”

A single global vector has no meaningful multi-sample rank analysis. Rank refers to the matrix formed by stacking the centered individual displacements:

\[
\epsilon_i = \Delta_i - \mu_\Delta.
\]

The remaining speaker-specific deviations occupy a multidimensional subspace. A dominant mean component and a higher-rank centered remainder are fully compatible.

### 2.5 “The correction reveals identity”

A safer phrase is:

> The correction improves identity-relevant cross-mode correspondence under the evaluated retrieval and verification metrics.

The result does not establish pure identity, identity independent of style, or practical biometric recognition.

---

## 3. Evidence hierarchy after discussion

Not all results should receive equal narrative weight.

### 3.1 Strongest current evidence

The most informative empirical finding is:

> A displacement estimated only from training speakers substantially improves held-out cross-mode speaker matching while the identity decision rule itself remains unchanged.

The identity backend is not retrained between raw and corrected conditions. The same cosine rule is applied before and after translation. This makes the effect easier to attribute to the representation geometry rather than to increased classifier capacity.

The result is strengthened by the fact that:

- it transfers to held-out speakers;
- it appears across multiple frozen representation families;
- wrong-sign and arbitrary-vector controls do not reproduce it;
- retrieval and verification both improve;
- a limited-reference condition also improves;
- the effect is not restricted to one nominal gallery size.

### 3.2 Important structural evidence

The following results help characterize the phenomenon:

- held-out individual displacements align with the train-estimated mean direction;
- the shared mean accounts for a large fraction of uncentered displacement energy;
- the centered remainder is still multidimensional;
- most, but not all, speakers benefit.

These findings support a “large shared component plus substantial individual variation” picture.

### 3.3 Auxiliary mechanism evidence

The collapse of the centroid-level linear mode AUC after symmetric correction is mainly a diagnostic.

It is partly expected because the correction is explicitly designed to cancel the average class displacement. Its useful interpretation is limited to:

> After removing the dominant shared mean direction, no comparably strong alternative centroid-level linear mode direction generalizes across held-out speakers under the same probe.

This should not be presented as a central discovery.

### 3.4 Negative evidence

The failure of individualized mappers to robustly improve identity metrics beyond the global baseline is scientifically useful.

It indicates that:

- person-specific residual structure exists;
- but its existence does not imply that it is predictable from speech-side information alone;
- current evidence does not support a deployable personalized SSL-space residual mapper.

---

## 4. Main scientific concerns raised in discussion

### 4.1 Large relative gains coexist with weak absolute biometric performance

The corrected results are substantially better than the raw results, but the absolute verification performance remains poor by practical biometric standards.

Therefore, two statements must remain separate:

1. the global displacement has a large effect on the evaluated geometry;
2. the resulting system is not a strong biometric recognizer.

The project currently supports the first statement much more strongly than the second.

A large improvement from a weak baseline can be scientifically meaningful, but it must not be rhetorically converted into high recognition quality.

### 4.2 The current identity backend is deliberately simple

The retrieval system uses cosine similarity rather than a trained identity classifier.

This is a methodological strength because it isolates the geometry of the frozen representation. However, it is also a limitation because a stronger supervised backend may:

- learn to ignore the shared mode displacement;
- absorb the correction internally;
- use information unavailable to cosine;
- reduce or eliminate the observed gain.

The current conclusion is therefore about simple frozen-space matching, not about all possible speaker-recognition backends.

### 4.3 Cosine similarity may be central to the observed effect

Cosine measures angular similarity relative to the coordinate origin:

\[
\cos(x,y)=\frac{x^\top y}{\|x\|\|y\|}.
\]

It is not translation invariant:

\[
\cos(x,y) \neq \cos(x+c,y+c).
\]

A shared translation can substantially change cosine rankings even when pairwise relative geometry is otherwise similar.

This creates two competing interpretations.

#### Strong interpretation

The speech and singing representations contain a genuine cross-mode geometric displacement that obscures speaker correspondence under multiple reasonable metrics.

#### Weak interpretation

The global translation mainly repairs an origin-sensitive angular artifact of uncentered cosine geometry.

The current experiments do not yet distinguish these interpretations because retrieval, EER, and TMR all use cosine-derived scores.

This is now the most important unresolved methodological issue.

### 4.4 Centroid averaging may make the phenomenon appear cleaner

Speaker centroids reduce utterance-level variability. This is useful for discovering stable structure, but it creates an idealized condition.

The limited-reference experiment shows that the effect does not disappear when only one speech utterance is used. However:

- performance becomes lower;
- the singing side still uses a centroid;
- the experiment does not establish robust utterance-to-utterance matching;
- it does not show that the corrected speech vector reaches the actual singing vector.

The limited-reference result should be interpreted as robustness to reduced speech reference budget, not as proof of precise individual mapping.

### 4.5 The global displacement is not the individual displacement

For a held-out speaker,

\[
\Delta_i = \mu_\Delta + \epsilon_i.
\]

For a single speech utterance,

\[
z_{i,u}^{S} = C_i^S + \eta_{i,u},
\]

where \(\eta_{i,u}\) is utterance-specific deviation.

After global correction,

\[
z_{i,u}^{S}+\mu_\Delta
\]

still differs from the speaker’s singing centroid by a combination of:

- speaker-specific displacement error \(\epsilon_i\);
- utterance-specific deviation \(\eta_{i,u}\).

Therefore, corrected single-reference accuracy should not be expected to approach 100%. The experiment tests ranking improvement, not exact reconstruction.

### 4.6 Nuisance factors remain in the main result

The headline global correction does not first remove F0, duration, energy, phonation, or other acoustic factors.

This is appropriate if the scientific object is the broad speech-to-singing displacement. It also means the displacement cannot be interpreted as pure timbre or pure identity.

Some controls reduce specific confounds:

- language is controlled more cleanly in JVS than in multilingual GTSinger;
- lexical equality is tested in the same-text setting;
- matched-frame analysis addresses unequal frame counts.

However, these controls do not remove or causally isolate all nuisance variables.

### 4.7 Identity and personal style should not be treated as cleanly separable by default

Stable personal style can legitimately help identify a person:

- articulation habits;
- vibrato tendencies;
- phrasing;
- breathiness;
- register use;
- accent;
- habitual phonation.

Therefore, style is not automatically “contamination” for same-person matching.

The unresolved issue is generalization:

- across songs;
- across sessions;
- across microphones;
- across deliberate style changes;
- across languages;
- across recording conditions.

The present evidence supports dataset-level same-person correspondence more clearly than style-independent or session-independent identity.

### 4.8 Dataset-specific confounds remain central

GTSinger is strongly language-confounded and should not be treated as clean independent identity evidence.

JVS is the more credible identity-relevant dataset, but it still does not fully resolve:

- cross-song generalization;
- session/channel consistency;
- whether stable song or recording cues contribute;
- whether the result transfers to uncontrolled singing.

### 4.9 The current result lacks causal or generative semantics

The global vector is currently established as a useful translation for representation-space matching.

It has not been shown that varying

\[
z(\alpha)=z_{\mathrm{speech}}+\alpha\mu_\Delta
\]

causes an audible, monotonic transition from speech-like to singing-like output while preserving content and identity.

A linear change in mode-probe score under an \(lpha\)-sweep would be weak evidence because it is close to a consequence of the construction. Stronger evidence would require a decoder or downstream synthesis system and human evaluation.

Without such evidence, the vector should not be described as a controllable “singing direction.”

---

## 5. Scientific value: what may actually be interesting

The fact that the method differs from previous work is not, by itself, a meaningful novelty claim.

The possible scientific value is narrower:

> Poor cross-mode matching in frozen SSL space may reflect a large shared coordinate displacement rather than complete loss of speaker-related information.

This matters because it distinguishes two failure modes:

1. speaker correspondence is absent;
2. speaker correspondence exists but is poorly exposed by the chosen geometry.

The present evidence suggests that the second explanation accounts for a substantial part of the raw failure under cosine matching.

A second useful contribution is methodological:

> Any future individualized cross-mode mapper must beat a strong train-estimated global mean baseline under held-out identity metrics.

However, the current finding should not be elevated into a general “singing vector” claim without stronger metric, dataset, and intervention evidence.

---

## 6. Central unresolved fork

The project now faces a clear interpretive fork.

### Hypothesis A: general cross-mode geometry

The shared displacement is a genuine representation-space phenomenon that affects multiple sensible speaker-similarity metrics and remains useful even with stronger backends.

### Hypothesis B: cosine/origin artifact

The observed gain is primarily caused by the interaction between:

- an uncentered representation space;
- cosine’s dependence on the origin;
- a large speech–singing mean offset.

Under this explanation, a different centering, whitening, or speaker backend may remove the need for explicit translation.

Resolving this fork is more important than additional mode-probe experiments.

---

## 7. Highest-priority next analyses

### 7.1 Metric robustness ladder

Under the identical held-out speaker protocol, compare raw and globally corrected representations using:

1. raw cosine;
2. train-global-mean-centered cosine;
3. whitened cosine;
4. unnormalized Euclidean distance;
5. Mahalanobis distance;
6. LDA or PLDA;
7. a simple train-speaker-only linear speaker metric.

Important caution:

- Euclidean distance after L2 normalization is equivalent to cosine ranking and is not an independent test.

Interpretation:

- improvement across many metrics supports a general displacement;
- improvement only for raw cosine supports an origin-specific angular correction;
- no gain after PLDA suggests a strong backend can absorb the shift;
- additional gain after PLDA would increase practical relevance.

### 7.2 Explicit score-distribution analysis

Beyond aggregate R@1, EER, and TMR, inspect how correction changes:

- genuine score distribution;
- impostor score distribution;
- speaker-level score shifts;
- nearest-impostor identities;
- within-speaker versus between-speaker angular structure.

This may reveal whether improvement comes from:

- genuine pairs moving closer;
- impostor pairs moving apart;
- a few hard speakers;
- broad ranking reorganization;
- norm or origin effects.

### 7.3 Cross-song and cross-session evaluation

A stronger identity claim requires protocols in which:

- speech and singing content differ;
- singing references and queries come from different songs;
- sessions or recording conditions differ where possible;
- habitual style cues do not trivially identify the speaker.

### 7.4 Utterance-to-utterance evaluation

The current limited-reference test changes only the speech side. A stricter condition would test single speech utterance against single singing utterances, while carefully controlling content and duration.

This would determine how much of the current result depends on gallery centroid averaging.

### 7.5 Causal steering only after the geometric question is clarified

If a compatible decoder or synthesis model is available, an \(lpha\)-sweep may test whether the direction has audible semantics.

However, this should follow, not replace, the metric-robustness analysis. A representation-space score sweep alone is insufficient.

### 7.6 Do not expand the mode-probe branch unnecessarily

Searching for progressively weaker nonlinear mode signals after mean correction is a different research topic. It is not required to establish the current identity-geometry finding.

---

## 8. Claims that should currently be avoided

Avoid:

- “speech and singing are perfectly linearly separable”;
- “the correction removes all mode information”;
- “the global vector is a singing vector”;
- “the vector represents pure timbre”;
- “the method disentangles identity and mode”;
- “the shared mean explains most identity variance”;
- “single-reference correction predicts the speaker’s singing representation”;
- “the method achieves strong biometric verification”;
- “the finding is metric-independent”;
- “this is the first speech–singing subspace compensation method”;
- “GTSinger independently validates language-free identity transfer.”

---

## 9. Current conservative claim

A defensible current formulation is:

> Frozen SSL representations exhibit a large train-estimable mean displacement between speech and singing. Translating held-out speech representations by this displacement substantially improves cosine-based cross-mode speaker retrieval and verification without retraining the identity decision rule. The result indicates that a shared mode-associated offset obscures existing speaker correspondence under the evaluated geometry. However, absolute biometric performance remains weak, the displacement is not pure identity or timbre, substantial higher-rank speaker-specific variation remains, and current evidence does not yet distinguish a general cross-mode alignment phenomenon from a correction specific to cosine and the chosen coordinate origin.

---

## 10. Literature and positioning principles from the discussion

### 10.1 Novelty must be scientific, not merely procedural

A method being different from prior methods is insufficient. The project must identify what new understanding the result provides.

### 10.2 Prior tasks must be compared at the trial level

Speaker verification, closed-set identification, singer retrieval, diarization, clustering, and synthesis may all use speaker information, but they answer different questions.

Comparisons should focus on:

- what constitutes a trial;
- what is known at test time;
- whether identities are enrolled;
- whether labels are global or recording-local;
- whether a system identifies, verifies, clusters, or synthesizes.

### 10.3 Diarization is adjacent but not identical

Diarization depends on consistent speaker information, but typically assigns anonymous local speaker labels over time rather than identifying a known person.

It is relevant as a broader research trajectory involving speaker consistency under variability, but the present datasets and protocol do not directly constitute a diarization study.

### 10.4 Paper reading should not use a rigid checklist mechanically

The main goal is to identify each paper’s central problem, evidence, and actual claim. Split construction, aggregation, and training protocol should be examined when they materially affect interpretation, not as ritual questions detached from the paper’s contribution.

---

## 11. Questions for the next analysis stage

The next researcher or AI should prioritize the following questions:

1. Is the observed gain robust to centering, whitening, Euclidean, Mahalanobis, LDA, or PLDA scoring?
2. How much of the gain is explained specifically by cosine’s origin dependence?
3. Does correction improve genuine scores, reduce impostor scores, or mainly reorder nearest neighbors?
4. Does the effect survive cross-song and cross-session protocols?
5. How much performance remains when both query and gallery are single utterances?
6. Can a strong speaker backend learn the compensation without explicit translation?
7. Does explicit translation still add value after such a backend?
8. What semantic factors dominate the shared displacement: mode, duration, F0, phonation, segmentation, or recording domain?
9. Which components of the higher-rank centered remainder are stable across sessions or songs?
10. Is there any causal or generative use for the direction, or is its value primarily diagnostic?
11. What is the narrowest scientifically meaningful claim if the effect proves cosine-specific?
12. What additional experiment would most efficiently change the current interpretation?

---

## 12. Bottom-line discussion state

The discussion has shifted the project away from the strongest initial interpretation.

The current evidence does **not** justify a claim of pure identity residual, clean timbre disentanglement, high-performance biometrics, or a controllable singing direction.

The most credible result is narrower:

- frozen SSL speech and singing centroids exhibit a large shared mean displacement;
- a train-only translation improves held-out same-person correspondence under simple cosine geometry;
- the gain is large relative to the raw baseline;
- absolute recognition remains weak;
- individual deviations remain multidimensional;
- the most important unresolved question is whether this is a general geometric phenomenon or mainly a cosine/origin correction.

This unresolved distinction should guide the next stage of analysis.
