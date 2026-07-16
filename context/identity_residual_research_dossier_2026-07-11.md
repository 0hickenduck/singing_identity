# Speech–Singing Identity Residual: Research Dossier
## What we tried, what survived, what failed, and where the paper claim now stands

**Date:** 2026-07-11  
**Document role:** research map, paper-prototype narrative, evidence index, and negative-result archive  
**Canonical scope:** frozen representation analysis; not a new SVC model and not a claim of pure timbre isolation
**Project location:** `context/identity_residual_research_dossier_2026-07-11.md`

---

# 0. How to read this document

This document is meant to answer five questions in order:

1. What did this project originally try to do?
2. Why did the question change?
3. What data, representations, splits, and metrics were actually used?
4. Which conclusions are supported by which experiments?
5. Which failed, ambiguous, or side experiments must remain visible even if they do not enter the paper?

Evidence labels used throughout:

- **CORE:** suitable for the main paper claim.
- **SUPPORTING:** useful control or robustness evidence.
- **NEGATIVE:** a meaningful failed gate or result that narrows the claim.
- **EXPLORATORY:** informative but too small, confounded, or post hoc for a main claim.
- **SUPERSEDED:** an earlier result or interpretation replaced by a stricter analysis.
- **OPEN:** not resolved by the current experiments.

The short answer is:

> Frozen SSL audio representations contain a large speaker-shared speech-to-singing displacement. A speaker-balanced translation estimated only from training speakers substantially improves held-out cross-mode speaker correspondence across WavLM, HuBERT, and MERT. JVS/JVS-MuSiC supplies the primary identity-relevant evidence; GTSinger supplies a same-text mechanism replication but is strongly language-confounded. The effect survives single-reference and matched-frame controls. However, the displacement is not pure timbre, the remaining residual is not one-dimensional, and the evaluated individualized SSL mappers do not reliably improve identity metrics beyond the global correction.

Current project decision:

- write the main story as a **representation geometry analysis**;
- treat the global translation as a diagnostic and baseline, not an algorithmic novelty claim;
- stop individualized mapper tuning unless stronger independent singing data becomes available;
- keep SeedVC and breathy-technique work as internal/side evidence, not main proof;
- do not start new representation experiments merely to enlarge the story.

---

# 1. How the research question evolved

## 1.1 Original two-track plan

The project began with two connected but distinct tracks.

### Track 1: speech-to-singing timbre or identity residual

The initial intuition was:

> A person's singing representation may equal their speech representation plus a learnable speech-to-singing residual.

The hoped-for downstream use was to adapt a speech reference toward a singing reference for zero-shot singing voice conversion. The early language used terms such as *timbre residual*, *MicroMapper*, and *speech-to-singing mapping*.

### Track 2: singing-technique directions

The second intuition was:

> Techniques such as breathy singing may form stable directions in frozen representation spaces and may be usable for steering.

Both tracks began as inexpensive representation audits before any large generative-model investment.

## 1.2 What Stage 1 established

The repaired Stage 1 run used 6,016 GTSinger utterances from 20 singers, including 4,000 speech/singing pairs, with cached acoustic, WavLM, HuBERT, and MERT features.

Stage 1 established that:

- speech and singing are strongly separable in multiple frozen representations;
- some layers retain above-chance cross-mode identity information;
- a global speech-to-singing residual is already a strong baseline;
- the observed residual cannot safely be called pure timbre because it may include F0, duration, energy, phonation, content, segmentation, language, and recording-domain effects;
- broad technique-vector claims are weakened by technique imbalance and limited speaker/language coverage.

This was not a failed stage. It exposed that the original scientific object—“singing timbre”—was underspecified.

## 1.3 Why SeedVC was investigated

SeedVC was used to ask whether the representation mismatch had an audible downstream counterpart.

The sequence was:

1. compare SeedVC outputs conditioned on target speech versus target singing prompts;
2. extract SeedVC-native prompt-side components;
3. predict or replace selected components;
4. listen for movement from speech-prompt output toward singing-prompt output.

The black-box prompt baseline showed a measurable and audible prompt-mode effect. However, the effect sounded more like changes in content delivery, phonation, and acoustic style than a clean target-identity improvement.

The native latent audit found that semantic/prompt statistics were mathematically predictable, but they were also correlated with duration, F0 voiced proportion, and RMS. A semantic-only intervention produced valid audio but received the human verdict `semantic weak`.

Component ablation suggested that mel context plus style embedding could approach the all-singing oracle more closely than prompt-sequence-only changes. No single native component cleanly explained the audible effect.

Decision:

> Pause SeedVC component mapping. SeedVC is a useful engineering diagnostic but not a clean scientific instrument for isolating identity, timbre, mode, or technique.

## 1.4 Why the project pivoted to residual accounting

The failed attempt to isolate a SeedVC component sharpened the question:

> Before synthesizing anything, what portion of the same-person speech-to-singing displacement is shared across speakers, what portion is associated with broad mode/acoustic differences, and what person-specific structure remains?

This reframed the project from generative mapping to representation analysis.

The next experiments therefore focused on:

- train-speaker-only residualization;
- speaker-disjoint identity retrieval;
- acoustic/mode/random controls;
- a mandatory global-mean baseline;
- reliability and mapper gates;
- larger-speaker JVS/JVS-MuSiC validation;
- same-text and matched-frame controls;
- verification, reference-budget, and residual-spectrum analyses.

## 1.5 The experiment ladder and its decisions

| Phase | Question | Main outcome | Status | Consequence |
|---|---|---|---|---|
| Stage 1 frozen audit | Is there a speech/singing mismatch and cross-mode identity signal? | Yes, but “timbre” is too broad | SUPPORTING | Narrow the scientific object |
| SeedVC prompt baseline | Does prompt mode audibly affect SVC output? | Yes, but not clearly as identity | EXPLORATORY | Inspect native components |
| SeedVC native audit/intervention | Can one predictable component carry the effect? | Semantic-only intervention weak; effect distributed/confounded | NEGATIVE | Pause SeedVC as main line |
| Track 2 breathy direction | Is breathy one transferable global direction? | Decodable, but not a clean global vector | NEGATIVE/SIDE | Do not use as paper headline |
| Residualized retrieval | Does identity survive nuisance removal? | Yes, but acoustic baseline can also be strong | SUPPORTING | Add ECAPA and larger JVS data |
| JVS/JVS-MuSiC expansion | Does the phenomenon hold with 100 speakers? | Strong global correction gains in SSL spaces | CORE | Run strict mechanism gates |
| Individualized mapper gate | Can test speech predict speaker-specific residual beyond global mean? | Not reliably for SSL identity metrics | NEGATIVE | Stop mapper tuning |
| Same-text validation | Is the result only a text-distribution artifact? | No; all three SSL models pass strongly | CORE | Paper-readiness gate passes |
| Verification/reference robustness | Is it only centroid R@1 or small galleries? | No; EER, TMR, one-reference, and gallery tests improve | CORE | Promote verification evidence |
| Matched-frame reevaluation | Is unequal frame count a complete explanation? | No; large gains remain | CORE | Retain matched-frame control |
| Residual spectrum | Is the residual effectively one-dimensional? | Shared mean is dominant, centered residual remains higher-rank | CORE qualification | Reject a rank-one claim |

---

# 2. Current research question and claim boundary

## 2.1 Fixed research question

The paper is not about proposing a new singing voice conversion model, learning a deployable individualized mapper, isolating pure timbre, or removing pitch.

The fixed question is:

> How are same-person speech and singing references organized in frozen SSL representation spaces, and can a speaker-shared train-estimated displacement recover identity-relevant cross-mode speaker correspondence for held-out speakers?

## 2.2 Current supported claim

> Frozen SSL audio representations exhibit a large speaker-shared mean displacement between speech and singing. A speaker-balanced translation estimated exclusively from training speakers substantially improves held-out cross-mode speaker retrieval and verification across multiple SSL families. The effect survives a same-text mechanism replication and matched-frame controls, while substantial higher-rank speaker-specific residual variation remains. JVS provides the primary identity-relevant evidence; multilingual GTSinger is not treated as language-independent identity validation.

A shorter paper-ready version is:

> A train-estimable global speech-to-singing displacement obscures identity-relevant geometry in frozen SSL representations; correcting it improves held-out cross-mode retrieval and verification.

## 2.3 What the claim does and does not mean

The supported interpretation is:

> A large part of the apparent cross-mode identity mismatch is attributable to a shared displacement of the speech and singing clouds. Correcting this displacement reveals identity correspondence already present in the frozen representation.

It does not mean:

- the SSL representation lacked identity before correction;
- the entire residual is one-dimensional;
- mode information is completely removed;
- pitch, duration, prosody, content, or recording domain has been causally removed;
- pure vocal timbre has been isolated;
- the operation transforms a waveform;
- a personalized residual can be predicted reliably from test speech;
- the result solves speech-reference singing voice conversion.

---

# 3. Data, representations, and common protocol

## 3.1 Base representation unit

For WavLM, HuBERT, and MERT, the extractors store a hidden-state sequence

\[
X_u \in \mathbb{R}^{T_u \times D}
\]

for utterance $u$. Unless a matched-frame experiment states otherwise, the utterance vector is the concatenation of temporal mean and standard deviation:

\[
z_u = [\operatorname{mean}_t X_u;\operatorname{std}_t X_u].
\]

For each speaker $i$ and mode $m\in\{\mathrm{speech},\mathrm{singing}\}$, utterance vectors are averaged to obtain a speaker-mode centroid:

\[
C_i^m = \frac{1}{|U_i^m|}\sum_{u\in U_i^m} z_u.
\]

Most headline retrieval and verification results use these speaker-mode centroids. The single-reference experiment replaces the full speech centroid with one sampled speech utterance vector while retaining a singing gallery centroid.

## 3.2 Global displacement

For training speakers only:

\[
\mu_\Delta
=
\frac{1}{N_{\mathrm{train}}}
\sum_{i\in\mathrm{train}}
\left(C_i^{\mathrm{sing}}-C_i^{\mathrm{speech}}\right).
\]

Because each training speaker contributes exactly one speech centroid and one singing centroid, this is also:

\[
\mu_\Delta
=
\mu_{\mathrm{sing}}-\mu_{\mathrm{speech}}.
\]

The main speech-to-singing correction is:

\[
C_{i,\mathrm{corr}}^{\mathrm{speech}}
=
C_i^{\mathrm{speech}}+\mu_\Delta.
\]

The reverse correction subtracts the vector from singing. For the mode-probe analysis, a symmetric correction moves speech by $+\frac{1}{2}\mu_\Delta$ and singing by $-\frac{1}{2}\mu_\Delta$.

This is mathematically a speaker-balanced domain-mean translation. It is not presented as a novel optimization algorithm.

## 3.3 Datasets and their roles

| Dataset | Speakers | Speech/singing relation | Role in this project | Main limitation |
|---|---:|---|---|---|
| JVS + JVS-MuSiC | 100 | same person; `parallel100` speech versus common song `katatsumuri` | main statistical dataset | not same-text; one common singing item limits independent singing reliability |
| GTSinger clean subset | 20 | lexical-equal speech/singing pairs, `technique == control` | same-text mechanism/control dataset | small; language-only expected R@1 is 70% in ten-speaker galleries |
| GTSinger Stage 1 pool | 20 | broader paired material and techniques | early audit, SeedVC, technique side track | technique imbalance and language/singer confounding |

The clean GTSinger subset was not selected using the manifest's `same_text_flag`, because that flag was hard-coded by the builder. Instead, equality was independently checked after NFKC normalization, removal of `<AP>/<SP>` boundaries, and whitespace normalization.

The resulting subset contains:

- 1,953 lexical-equal pairs;
- 3,906 distinct utterances;
- all 20 singers;
- nine languages;
- unique speech and singing utterance IDs.

An exhaustive metadata audit over all \(\binom{20}{10}=184{,}756\) possible ten-speaker galleries gives a **70% mean expected R@1** for an oracle that knows only the query language and guesses uniformly among same-language gallery singers (range 40–90%; 2.5th–97.5th percentiles 50–90%). On average, 43.2% of queries are the only test singer of their language. This is numerically comparable to corrected SSL R@1 and makes language a direct threat to the identity interpretation. Same-text pairing removes lexical mismatch between paired modes; it does not separate language from singer.

## 3.4 Speaker-disjoint splits

- JVS/JVS-MuSiC: repeated 60-train/20-dev/20-test speaker-disjoint splits; 20 principal seeds.
- GTSinger same-text: repeated 10-train/10-test speaker-disjoint splits; 50 seeds.
- No test speaker contributes to the fitted global vector, nuisance model, mode probe, mapper, or verification threshold.
- The global vector uses one speaker, one vote rather than allowing speakers with more utterances to dominate.

Repeated split intervals are useful for protocol stability but the splits reuse speakers. Therefore the robustness suite also aggregates by unique test speaker and reports speaker bootstrap and sign-permutation results.

## 3.5 Retrieval and verification

Retrieval uses cosine similarity between L2-normalized query and gallery centroids.

- S→G: speech or corrected-speech query against held-out singing centroids.
- G→S: singing or corrected-singing query against held-out speech centroids.
- R@1 measures whether the correct identity ranks first.
- Gallery-size experiments sample smaller held-out galleries and compute query-specific chance.

Verification uses:

- genuine trials: same-speaker cross-mode centroid pairs;
- impostor trials: all off-diagonal cross-speaker pairs;
- EER: descriptive ROC crossing on held-out test trials;
- TMR@FMR=1%: operating threshold selected using training-speaker trials only, then applied to test trials.

## 3.6 Primary models, layers, and comparators

The final paper-facing set is:

- WavLM Base+ layer 12;
- MERT v1 95M layer 3;
- HuBERT base layer 6.

These settings were retained after a broader layer sweep and represent three different SSL families. The full sweep remains part of the evidence archive and should be reported in an appendix or supplement so the primary-layer selection is not mistaken for a preregistered choice.

Comparators:

- **ECAPA-TDNN:** supervised speaker embedding sanity check. It is often near ceiling, showing that cross-mode identity is not universally absent across representation types.
- **Acoustic features:** low-level baseline. It can be strong in small GTSinger evaluations, so SSL gains cannot be interpreted automatically as pure high-level identity.
- **Wrong-sign vector:** tests directional specificity.
- **Same-norm random vectors:** tests whether arbitrary perturbations help.
- **Random/shuffled/low-rank controls:** test pipeline leakage and arbitrary geometry changes.
- **Global mean:** mandatory baseline for every individualized mapper.

---

# 4. Main evidence

## 4.1 JVS/JVS-MuSiC main result

| Model | Raw S→G R@1 | Corrected R@1 | Gain | Raw EER → corrected | Raw TMR@1% → corrected |
|---|---:|---:|---:|---:|---:|
| WavLM L12 | 13.5% | 36.3% | +22.8 pp | 42.1% → 26.6% | 3.0% → 14.5% |
| MERT L3 | 36.5% | 66.0% | +29.5 pp | 28.5% → 16.1% | 14.3% → 38.8% |
| HuBERT L6 | 19.5% | 60.5% | +41.0 pp | 37.1% → 18.6% | 2.0% → 30.0% |

Wrong-sign correction degrades performance. Same-norm random directions remain near raw.

Interpretation:

- the gain is not only a small-gallery rank crossing;
- the train-estimated direction has the correct sign and identity relevance;
- JVS remains content-unmatched, so this result alone cannot identify mode separately from content, segmentation, and recording-domain effects.

## 4.2 GTSinger same-text result

| Model | Raw S→G R@1 | Corrected R@1 | Gain | Raw EER → corrected | Raw TMR@1% → corrected |
|---|---:|---:|---:|---:|---:|
| WavLM L12 | 19.8% | 73.8% | +54.0 pp | 37.5% → 20.9% | 5.4% → 33.4% |
| MERT L3 | 27.4% | 60.4% | +33.0 pp | 35.4% → 23.7% | 8.4% → 20.2% |
| HuBERT L6 | 19.8% | 76.2% | +56.4 pp | 38.2% → 18.5% | 4.0% → 28.2% |

The reverse G→S direction also improves strongly. All three SSL models pass the preregistered content-control gate: at least +5 pp, paired interval excluding zero, and no reproduction by wrong-sign or same-norm random controls.

Interpretation:

- lexical mismatch is not a complete explanation;
- the result replicates the first-order mode/correction mechanism under lexical equality;
- it is **not** clean independent evidence of abstract speaker identity because language alone has 70% expected R@1 in the evaluated gallery design;
- all-off-diagonal verification also contains many easier cross-language impostors;
- the result does not prove that temporal, prosodic, style, session, or recording differences are irrelevant.

## 4.3 Mode mechanism

Train-only L2-penalized logistic probes on speaker-mode centroids show:

- raw mode AUC approximately 1.0;
- corrected mode AUC approximately 0.47–0.54.

This supports the interpretation that a dominant shared direction makes speech and singing nearly perfectly linearly separable and obscures same-person correspondence.

It does not prove that all nonlinear, local, or temporal mode information has been removed.

## 4.4 Single-reference robustness

| Dataset/model | Raw R@1 | Corrected R@1 |
|---|---:|---:|
| JVS WavLM | 13.0% | 28.6% |
| JVS MERT | 35.0% | 59.8% |
| JVS HuBERT | 18.2% | 50.8% |
| GTSinger WavLM | 21.3% | 61.1% |
| GTSinger MERT | 28.6% | 52.1% |
| GTSinger HuBERT | 20.5% | 66.8% |

The effect is therefore not solely an artifact of using a full speech centroid.

## 4.5 Matched-frame reevaluation

The strict GTSinger reevaluation uses exactly 100 representation frames on both sides and deterministic crops selected to minimize voiced-frame-ratio mismatch.

| Model | Raw R@1 | Corrected R@1 | Gain | EER delta |
|---|---:|---:|---:|---:|
| WavLM L12 | 29.2% | 76.8% | +47.6 pp | −11.4 pp |
| MERT L3 | 38.6% | 71.0% | +32.4 pp | −5.0 pp |
| HuBERT L6 | 31.8% | 72.2% | +40.4 pp | −12.9 pp |

The matched-frame global vector remains strongly aligned with the full-utterance vector; WavLM splitwise cosine is roughly 0.94–0.96. Mode AUC again falls from approximately 1.0 to chance.

This rules out unequal frame count and gross voiced-duration mismatch as complete explanations. It is not causal duration removal.

## 4.6 Held-out residual structure and uncertainty

For held-out speakers:

- individual residual cosine with the train global vector is approximately 0.78–0.92 on average;
- 71%–85% of speakers have positive rank improvement;
- speaker-bootstrap intervals exclude zero;
- paired sign-permutation p-values are at most 0.0005;
- the shared mean explains approximately 0.65–0.83 of uncentered residual energy.

However, the centered residual is not one-dimensional:

- effective rank is about 6 on GTSinger;
- effective rank is about 23–28 on JVS.

The structural description is therefore:

\[
\Delta_i = \mu_{\mathrm{global}} + \epsilon_i,
\]

where the mean is dominant but (\epsilon_i) remains higher-rank.

Energy accounting and identity performance are reported separately. High explained energy is not assumed to imply high retrieval improvement.

## 4.7 Individualized mapper gate

The evaluated predictors do not establish a deployable individualized SSL mapper beyond the global correction.

- ECAPA shows mapper-over-global evidence, but it is a supervised speaker space and often near ceiling.
- MERT L3 shows weak or unstable gains; important MSE intervals include zero.
- WavLM L12 learned mappers do not reliably beat the global baseline on retrieval.
- lower residual reconstruction MSE does not consistently improve R@1, EER, TMR, or speaker rank.
- JVS singing material does not provide a strong independent repeated-singing reliability ceiling.

The correct negative claim is:

> Speaker-specific residual variation exists, but the evaluated test-speech-only predictors did not demonstrate reliable held-out SSL identity benefit beyond the global correction.

---

# 5. Claim–evidence–artifact map

| Claim or boundary | Evidence | Level | Main artifact |
|---|---|---|---|
| Train-only global translation improves held-out identity | JVS R@1, EER, TMR across three SSL families | CORE | `results/identity_residual_next_stage_report_2026-07-10.md` |
| Result is not reproduced by arbitrary vectors | wrong-sign and same-norm random controls | CORE | `results/identity_residual_paper_readiness_2026-07-10.md` |
| Text distribution is not a complete explanation | lexical-equal GTSinger control subset | CORE | `results/identity_residual_paper_readiness_2026-07-10.md` |
| Unequal frame count is not a complete explanation | equal-100-frame, voiced-ratio-matched evaluation | CORE | `results/identity_residual_next_stage_report_2026-07-10.md` |
| Result is not only centroid averaging | one-speech-reference evaluation | SUPPORTING | `results/identity_residual_next_stage_report_2026-07-10.md` |
| Result is not only R@1 on a small gallery | held-out EER and train-calibrated TMR@1% | CORE | `results/identity_residual_next_stage_report_2026-07-10.md` |
| Shared direction is mode-dominant | train-only logistic mode AUC falls to chance | CORE | `results/identity_residual_paper_readiness_2026-07-10.md` |
| Most speakers, not only a few, benefit | speaker aggregate, bootstrap, sign permutation | CORE | `results/identity_residual_next_stage_report_2026-07-10.md` |
| Residual is not rank one | centered effective rank on both datasets | CORE qualification | `results/identity_residual_next_stage_report_2026-07-10.md` |
| Personalized SSL mapper is not established | B1/B2 gates against global baseline | NEGATIVE | `results/identity_residual_response_to_original_plan_2026-07-09.md` |
| SeedVC prompt effect is not clean identity proof | human listening and acoustic post hoc | EXPLORATORY | `results/human_verdict_and_objective_followup_zh_2026-06-28.md` |
| Breathy is encoded but not one clean global vector | detection versus global-direction controls | NEGATIVE/SIDE | `results/meeting_prep_research_state_2026-06-29.md` |

Exact run roots and commands are recorded in `context/EXPERIMENT_REGISTRY.md`.

---

# 6. Alternative explanations: reduced, unresolved, or outside scope

## 6.1 Explanations substantially weakened by controls

| Possible explanation | Evidence against it | Precise conclusion |
|---|---|---|
| Small-gallery-only ranking artifact | EER and train-calibrated TMR improve | gain is not only R@1 crossing |
| Full-centroid-only artifact | one-reference evaluation improves | averaging is not necessary for the effect |
| Text-distribution-only artifact | lexical-equal GTSinger pairs | text mismatch is not a complete explanation of the mode/correction effect; language-independent identity is not established |
| Unequal frame-count-only artifact | fixed 100-frame evaluation | frame count is not a complete explanation |
| Arbitrary perturbation | same-norm random directions stay near raw | learned direction matters |
| Unsigned centering | wrong sign degrades | direction sign matters |
| A few influential speakers | speaker-level bootstrap and sign permutation | effect is broadly distributed |
| Strict rank-one residual | centered effective rank | explicitly rejected |

The wording “not a complete explanation” is intentional. These factors can still contribute.

## 6.2 Unresolved within the current paper

- exact causal roles of pitch, prosody, duration, phonation, segmentation, and recording domain;
- separation of abstract speaker identity from speaker-correlated style and session/channel cues;
- language-matched identity performance in GTSinger;
- cross-song generalization in JVS/JVS-MuSiC, whose main evaluation uses one common song;
- whether the exact vector transfers across datasets without refitting;
- nonlinear and temporal mode information after centroid correction;
- demographic generalization beyond the available speakers and languages;
- whether higher-rank residual coefficients can be predicted from test speech without oracle singing;
- waveform-domain and SVC consequences.

## 6.3 Outside the current claim

- pure timbre isolation;
- pitch removal;
- controllable singing-technique editing;
- SeedVC mechanism explanation;
- a new speaker-recognition or domain-adaptation architecture;
- a deployable personalized speech-to-singing mapper.

---

# 7. Negative, ambiguous, and side experiments that must remain visible

## 7.1 SeedVC prompt-mode baseline

What was done:

- generated paired outputs using target speech versus target singing prompts;
- expanded to a 30-pair review;
- ran objective acoustic comparisons and post hoc joins to Track 1 features;
- conducted human listening in Marimo review notebooks.

What was learned:

- prompt mode changes output audibly and measurably;
- singing prompts often sound more compatible with target singing;
- the dominant perceptual changes include content delivery, phonation, and acoustic style;
- the experiment does not cleanly prove improved target identity.

Status: **EXPLORATORY; not main-paper evidence.**

## 7.2 SeedVC native latent audit and semantic intervention

The native audit examined:

- CAMPPlus style;
- semantic statistics;
- prompt-condition statistics;
- mel statistics;
- pooled diagnostic vectors.

Semantic statistics were predictable but correlated with duration, voiced proportion, and RMS. A semantic-only affine intervention produced technically valid outputs but no obvious reliable movement toward the all-singing oracle.

Human verdict: `semantic weak`.

Status: **NEGATIVE as a causal single-component path.**

## 7.3 SeedVC component ablation

The ablation swapped speech versus singing versions of:

- prompt sequence;
- prompt mel context;
- CAMPPlus style;
- two-component combinations;
- all-singing oracle.

Mel plus style was often closest to the oracle, but individual components were difficult to distinguish and the output differences were small. This suggests a distributed or interaction-dependent prompt effect rather than a single clean residual component.

Status: **engineering diagnostic only.**

## 7.4 Track 2 breathy technique

Strict controls separate two questions:

1. Is breathy information decodable from frozen representations? **Yes.**
2. Does breathy form one stable speaker-balanced global direction that transfers by analogy? **Not under the tested setup.**

This is a useful negative result because decodability does not imply a clean, transferable concept vector.

Status: **side-track negative result; not part of the current paper package.**

## 7.5 ECAPA near-ceiling behavior

ECAPA retrieves speech/singing identity very strongly before correction and often reaches near-ceiling performance after it.

These are two distinct readouts, not a contradiction:

- raw ECAPA S→G R@1 of 92.8% means strong same-person speech-to-singing matching in the controlled ten-speaker centroid gallery;
- raw mode AUC near 1.0 means the same embeddings still make speech versus singing almost perfectly linearly decodable;
- corrected R@1 of 98.6% together with corrected mode AUC of 0.506 shows that removing the dominant shared mode component need not destroy speaker correspondence.

“Near ceiling” is local to this controlled gallery and does not imply perfect open-set or in-the-wild verification.

This matters because:

- the speech–singing identity gap is representation-dependent rather than universal;
- the contribution is specifically about geometry in selected frozen general-purpose SSL spaces;
- ECAPA is a sanity comparison, not evidence for the novelty of the global vector.

## 7.6 Acoustic baseline warning

The acoustic baseline was strong in some small GTSinger residualized-retrieval settings. This prevents the claim that every above-chance or corrected result necessarily reflects abstract speaker identity.

The larger all-Japanese JVS results, ECAPA comparison, directional controls, and verification metrics justify the narrower identity-relevant interpretation. GTSinger's same-text result supports the mode mechanism but cannot independently justify identity because language is strongly confounded with singer.

## 7.7 Individualized mapper failure

The mapper line is the most important negative result for the current paper:

- residual structure can be stable or higher-rank;
- reconstruction MSE can improve;
- neither fact guarantees held-out identity benefit;
- a more expressive model containing a bias term does not automatically identify an identity-preserving solution.

Future mapper work must beat the global baseline on identity metrics, not merely reconstruct residual vectors.

---

# 8. Superseded results and interpretation corrections

| Earlier result or assumption | Problem | Current replacement |
|---|---|---|
| Signed AUC read directly as separability | AUC below 0.5 can mean reversed direction, not weak separability | corrected separability interpretation and train-only logistic mode probe |
| Old Track 2 analogy result | query could contaminate its own direction; controls incomplete | leave-one-out direction plus wrong-technique and shuffled-direction controls |
| GTSinger manifest `same_text_flag` | hard-coded by builder | independent NFKC lexical equality audit |
| Blanket restricted-gallery chance | gallery sizes vary by query | mean query-specific `1/gallery_size` |
| Difference-of-means mode probe | too close to the intervention being tested | train-only L2-penalized logistic probe |
| Mapper success judged by residual MSE | MSE does not imply retrieval or verification benefit | mandatory global baseline plus identity metrics |
| Early small-split mapper result | too few speakers and weak baseline accounting | repeated speaker-disjoint B1/B2 gates |
| “Pure timbre residual” language | residual contains mode, acoustic, temporal, and domain factors | global speech-to-singing mode-associated displacement |
| Duration treated as causal explanation | duration is highly mode-predictive but within-mode removal does not reproduce the effect | duration described as mode/segmentation proxy |

Earlier tables should not be copied into a paper without checking whether they have been superseded by corrected final-validation outputs.

---

# 9. Scientific contribution and novelty boundary

## 9.1 Low algorithmic novelty

The following are not novel in isolation:

- mean subtraction or mean translation;
- difference-of-means directions;
- CORAL-style statistical alignment;
- linear analysis of frozen SSL features;
- speaker/domain disentanglement;
- speech references for singing identity;
- parameter-free activation steering.

The global vector must not be presented as a new adapter architecture.

## 9.2 Potential empirical novelty

The potentially novel contribution is the validated combination of:

1. same-person speech and singing;
2. speaker-disjoint train-only estimation;
3. frozen general-purpose speech/music SSL representations;
4. one-speaker-one-vote global displacement;
5. held-out retrieval and verification;
6. correct-sign, wrong-sign, and same-norm-random controls;
7. same-text mechanism replication, with explicit language-confound qualification;
8. matched-frame and voiced-ratio control;
9. single-reference robustness;
10. mode-separability reduction;
11. held-out residual alignment and speaker-level uncertainty;
12. dominant mean plus higher-rank centered residual;
13. negative mapper conclusion under a mandatory global baseline.

## 9.3 Strongest contribution statement

> We conduct a controlled geometric audit of same-person speech and singing references in frozen SSL representations. Across three model families, we find that a speaker-shared mean displacement accounts for a large fraction of cross-mode residual energy and obscures existing speaker correspondence. A translation estimated only from training speakers improves held-out retrieval and verification, survives directional and matched-frame controls, and reduces linear mode separability to chance, while the remaining residual remains higher-rank. JVS provides the primary identity-relevant evidence, while language-confounded GTSinger supplies a same-text mechanism replication.

## 9.4 Claims the paper may make

- a large speaker-shared mean displacement;
- a global speech-to-singing displacement or mode-associated residual;
- a train-estimable speaker-balanced translation;
- improvement of held-out identity-relevant geometry;
- robustness to same-text, single-reference, and matched-frame controls;
- substantial higher-rank residual variation remains;
- more expressive residual predictors should beat the global baseline on identity metrics.

## 9.5 Claims the paper must avoid

- “first discovery of the speech–singing gap”;
- “first speech–singing domain adapter”;
- “the residual is one-dimensional”;
- “we removed pitch/duration/prosody”;
- “we isolated pure timbre”;
- “we learned individualized singing identity residuals”;
- “the global mapping is theoretically optimal”;
- “the correction transforms waveforms”;
- “the method solves speech-reference SVC”;
- “prior work never used speech identity for singing.”

---

# 10. Literature map

## 10.1 What prior work already establishes

| Literature family | Established before this project | What remains different here |
|---|---|---|
| Singing speaker recognition | speech-trained speaker systems degrade on singing | geometry of frozen SSL and held-out mean translation |
| Speech/singing domain adaptation | domain and covariance alignment can improve robustness | speaker-balanced first-order characterization and identity-relevant correspondence in frozen SSL |
| Singer identity learning | singer-specialized encoders can be trained | audit of information already present in frozen generic representations |
| Speech-reference singing generation | speech can condition target singing identity | upstream representation explanation rather than generation feasibility |
| Singing-oriented SSL adaptation | speech SSL has a singing-domain gap | no retraining; same-person identity geometry |
| General speaker DA | statistical alignment is mature | specific speech–singing identity phenomenon and control package |
| Linear SSL geometry/steering | means and subspaces can carry strong factors | shared speech–singing displacement validated for held-out identity |

## 10.2 Closest direct speech–singing identity work

### JukeBox: A Multilingual Singer Recognition Dataset

Chowdhury, Cozzo, Ross, 2020.

- Establishes severe degradation of speech-trained recognition on singing.
- Its 2020 experiment transfers models from VoxCeleb speech to disjoint JukeBox singing identities; it is not same-person speech-enrollment/singing-test verification.
- For the 1D-Triplet-CNN, VoxCeleb→JukeBox gives TMR@1% 24.72% and EER 26.48%, versus 91.23% and 4.09% on VoxCeleb→VoxCeleb; singing fine-tuning reaches 29.71% and 24.36%.
- Does not use systematic same-person paired speech/singing training identities.
- Does not analyze a global frozen-SSL mean displacement.
- Paper role: motivation and prior evidence for the domain gap.

### Domain Adaptation for Speaker Recognition in Singing and Spoken Voice

Chowdhury, Cozzo, Ross, ICASSP 2022.

- Studies **same-speaker cross-modal verification**, not speech-versus-singing mode classification.
- Extends JukeBox with corresponding spoken voice and applies domain adaptation to speaker recognition.
- The verification decision is “same person or different person” for cross-mode samples, conceptually closer to this project's EER/TMR than to its mode probe.
- Its in-the-wild multilingual music setting is substantially harsher than controlled JVS/GTSinger, so its scores cannot be used to claim that the current project proves the gap is globally “shallower.”
- Exact table values, trial aggregation, and the detailed DeepCORAL/CORAL+ protocol require rechecking against the closed five-page PDF; no number is inferred here.
- Removes any claim of being the first speech–singing domain adaptation study.

### Earlier Mehrabani and Hansen work

Fully reviewed titles:

- *Speaker clustering for a mixture of singing and reading*;
- *Singing speaker clustering based on subspace learning in the GMM mean supervector space*.

The 2012/2013 studies use UT-Sing: 33 native-English speakers reading and singing lyrics from five self-selected songs, with 15 train and 18 disjoint test speakers. Their task is unsupervised two-speaker clustering of mixed ten-second reading/singing segments, not cross-modal verification. MFCC GMM mean supervectors plus LPP/LDA/PLDA improve mixed-condition clustering from 80.6–82.9% baselines to about 90% with PLDA and up to 94.5% with LDA+PLDA.

Novelty consequence: historical work already demonstrates held-out speech/singing style compensation in a speaker subspace. The current project cannot claim first held-out cross-mode subspace compensation. Its remaining distinction is the frozen generic SSL audit, paired first-order displacement, retrieval/verification package, directional/content/frame controls, mode probe, residual-rank analysis, and negative mapper gate.

## 10.3 Singer representation and speech-reference generation

### Torres et al., 2024

*Singer Identity Representation Learning using Self-Supervised Techniques* trains a singer-specialized encoder with contrastive/VICReg/BYOL-like objectives and pitch/content augmentations.

It establishes singer identity representation learning and nuisance-aware training. It does not study train-speaker global displacement in frozen generic SSL spaces.

### Learning Singing From Speech, DurIAN-SC, Learn2Sing, Learn2Sing 2.0, Everyone-Can-Sing

These systems show that speech-derived identity can condition singing synthesis or conversion. Therefore this project cannot claim first speech-to-singing identity transfer.

Their shared distinction from the current work is that they train a generative or factorized system to impose transfer. They do not explain the geometry already present in frozen upstream representations.

## 10.4 Singing SSL and dataset references

- **SingOMD** and **SVPT:** singing-oriented adaptation or pretraining for generation; relevant evidence that direct speech SSL has a singing-domain gap.
- **JVS-MuSiC:** defines the 100-singer common-song setting and reports weak correlation between singing and speech perceptual similarity.
- **GTSinger:** supplies paired speech, multilingual singing, technique labels, and the same-text control pool.
- **NHSS:** parallel same-singer spoken and sung lyrics; full paper still needs checking for simple mean/affine baselines and identity evaluation.

## 10.5 General adaptation and simple frozen-representation geometry

- **CORAL, DeepCORAL, CORAL+, CORAL++:** establish that statistical domain alignment is mature.
- **van Niekerk et al., 2021:** utterance means in frozen SSL carry strong speaker information.
- **Liu, Tang, Goldwater, 2023:** speaker and phonetic information can occupy near-orthogonal SSL subspaces and generalize to unseen speakers.
- **Mohamed et al., 2024:** systematic SSL geometry analysis across speaker/phone subspaces.
- **Eta-WavLM, 2025:** simple linear operations can separate speaker-specific and speaker-independent components.
- **Activation Steering for Accent Adaptation, 2026:** difference-of-means steering in a speech foundation model; makes clear that the formula itself is not novel.

## 10.6 Cross-literature comparison

| Work/category | Same-person speech+singing | Paired train identities | Frozen generic SSL | Global mean translation | Held-out identity | Retrieval + verification | Same-text/matched-frame | Mode probe |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| JukeBox 2020 | No paired speech | No | No | No | Singer test IDs | Singing-only | No | No |
| Chowdhury 2022 | Yes; exact protocol PDF-required | PDF-required | No | Domain adaptation; exact first-moment handling PDF-required | Yes | Cross-mode verification | PDF-required | Embedding analysis; exact probe PDF-required |
| Mehrabani & Hansen 2013 | Yes, mixed clusters | Yes, 15 train speakers | No | No explicit global translation; LPP/LDA/PLDA | Yes, 18 disjoint test speakers | Two-speaker clustering | Same lyrics available; no matched-frame package | No |
| Torres et al. | Mixed evaluation datasets | No targeted cross-mode setup | Trains singer encoder | No | OOD singer tasks | Mainly same-mode | No | No |
| Speech-reference singing systems | Identity conditions singing | Generative training | No | No | Often zero/one-shot | Generation metrics | Task-dependent | No |
| Frozen SSL geometry work | Speech only | N/A | Yes | mean/subspace operations | Often unseen speakers | probes/tasks | N/A | factor probes |
| Current project | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes |

No located work currently matches the full final row. This is not proof of absence.

Recommended novelty wording:

> Prior work has established singing-domain degradation, speech/singing subspace compensation, and speech-conditioned singing generation. We are not aware of prior work that systematically characterizes a speaker-balanced first-order speech-to-singing displacement in frozen generic SSL representations under held-out-speaker retrieval/verification, directional controls, lexical replication, and matched-frame evaluation.

---

# 11. Reviewer objections and defensible answers

| Objection | Defensible answer |
|---|---|
| “This is only a trivial mean shift.” | The operation is intentionally simple and not claimed as an algorithmic contribution. The finding is the magnitude, held-out alignment, identity relevance, and robustness of the shared displacement. |
| “A complex mapper contains a bias and should do better.” | Capacity does not guarantee an identity-preserving solution. MSE improvement did not reliably yield better retrieval or verification beyond the global baseline. |
| “Explained energy is not identity performance.” | Correct. Energy and identity metrics are reported separately; their relationship is empirical, not assumed. |
| “It is caused by text or duration.” | Same-text and equal-frame/voiced-ratio controls retain large gains. These factors are not complete explanations, but causal removal is not claimed. |
| “Ten-person R@1 is weak.” | JVS uses 20-speaker held-out galleries, and the paper also reports EER, train-calibrated TMR@1%, gallery-size tests, and speaker-level uncertainty. |
| “GTSinger can identify speakers from language.” | Correct. A language-only oracle has 70% expected R@1 in the ten-speaker galleries. GTSinger is therefore used as a same-text mechanism replication, not language-independent identity evidence; JVS is the primary identity-relevant dataset. |
| “Identity may be style or session.” | The paper claims identity-relevant speaker correspondence, not pure identity. Shared song/language in JVS blocks those two cues from naming a gallery speaker, but style and session/channel contributions remain unresolved. |
| “One JVS song is not singer identity.” | The result establishes correspondence in a controlled common-song setting, not cross-song singer generalization. That boundary is stated explicitly. |
| “Mode AUC at chance means all singing information is gone.” | It means a specified linear centroid-level probe no longer separates modes after correction. Nonlinear, local, and temporal information may remain. |
| “ECAPA already works.” | Yes. Raw identity retrieval and raw mode classification can both be near ceiling because identity and mode are simultaneously encoded. The contribution concerns frozen general-purpose SSL geometry, not universal impossibility or a better recognition tool. |

---

# 12. Current decisions and open questions

## 12.1 Completed enough for the paper package

- main JVS held-out retrieval and verification;
- same-text GTSinger validation;
- wrong-sign and same-norm random controls;
- train-only mode probing;
- single-reference and gallery robustness;
- matched-frame reevaluation;
- speaker-level uncertainty;
- residual energy and effective-rank description;
- negative individualized mapper gate;
- literature boundary sufficient to begin writing, with historical caveats.

## 12.2 Lines intentionally stopped

- further individualized SSL mapper tuning;
- low-rank adapter sweep without a pre-specified test-speech-only coefficient rule;
- SeedVC as a main causal or identity claim;
- broad singing-technique steering;
- cross-dataset transfer as a prerequisite for writing.

## 12.3 Useful optional work, not current blockers

- obtain the Chowdhury 2022 PDF and verify its exact trial construction and table values;
- compute within-language GTSinger retrieval and language-matched impostor verification only if needed for a reviewer response;
- fully audit NHSS mapping baselines;
- complete exact citation metadata and code availability for every important paper;
- report the full layer sweep in a supplement;
- cross-dataset vector transfer only if needed for a specific reviewer concern;
- revisit individualized mapping only with stronger independent repeated-singing material.

## 12.4 Questions this document should help answer in discussion

1. Is the strongest paper a representation-analysis paper rather than a model paper? Current answer: yes.
2. Is “global mode-associated displacement” the right term? Current answer: yes, with explicit representation-level caveats.
3. Is residual higher-rank structure itself enough to justify an adapter? Current answer: no.
4. Does ECAPA weaken the paper? It narrows the claim but also shows the analysis is representation-specific and meaningful.
5. Should SeedVC appear? At most as motivation or internal exploratory context, not as main evidence.
6. Should breathy technique appear? Only as a separate side result or future-work lesson.

---

# 13. Artifact and reproducibility index

## 13.1 Master indexes

- Experiment registry: `/Users/bowen/research/project/context/EXPERIMENT_REGISTRY.md`
- Literature closure and terminology audit: `/Users/bowen/research/project/context/identity_residual_literature_closure_2026-07-11.md`
- Original-plan response: `/Users/bowen/research/project/results/identity_residual_response_to_original_plan_2026-07-09.md`
- Paper-readiness report: `/Users/bowen/research/project/results/identity_residual_paper_readiness_2026-07-10.md`
- Protocol/matched-frame summary: `/Users/bowen/research/project/results/identity_residual_next_stage_report_2026-07-10.md`
- Independent validation notes: `/Users/bowen/research/project/results/identity_residual_next_stage_validation_2026-07-10.md`
- Earlier direction reset: `/Users/bowen/research/project/results/direction_reset_and_literature_scan_2026-06-25.md`
- Human listening verdict: `/Users/bowen/research/project/results/human_verdict_and_objective_followup_zh_2026-06-28.md`
- Two-track meeting state: `/Users/bowen/research/project/results/meeting_prep_research_state_2026-06-29.md`

## 13.2 Core scripts

- `scripts/probing/run_identity_residual_suite.py`
- `scripts/probing/run_identity_residual_synthetic.py`
- `scripts/probing/run_identity_residual_mapper_eval.py`
- `scripts/probing/run_identity_residual_final_validation.py`
- `scripts/probing/run_identity_residual_same_text.py`
- `scripts/probing/run_identity_residual_robustness.py`
- `scripts/probing/run_identity_residual_matched_frames.py`
- `scripts/probing/run_identity_residual_spectrum.py`

## 13.3 Core run roots

- `/localdisk/bowen/singing_identity/runs/identity_residual_final_validation_2026-07-09`
- `/localdisk/bowen/singing_identity/runs/identity_residual_same_text_2026-07-10`
- `/localdisk/bowen/singing_identity/runs/identity_residual_protocol_robustness_2026-07-10`
- `/localdisk/bowen/singing_identity/runs/identity_residual_matched_frames_2026-07-10`

## 13.4 Side/negative experiment artifacts

- SeedVC prompt review: `notebooks/track1_seedvc_prompt_review.py`
- SeedVC semantic intervention review: `notebooks/track1_seedvc_semantic_intervention_review.py`
- SeedVC component ablation review: `notebooks/track1_seedvc_component_ablation_review.py`
- Prompt-mode objective report: `results/seedvc_prompt_30pair_acoustic_objective_zh_2026-06-28.md`
- Prompt-gap post hoc report: `results/seedvc_30pair_prompt_gap_posthoc_zh_2026-06-28.md`
- Component objective report: `results/seedvc_component_12pair_acoustic_objective_zh_2026-06-28.md`
- Breathy method/experiment context: `results/breathy_global_local_method_survey_2026-06-28.md`

## 13.5 Reproducibility rule

The compact reports and code live in the repository. Heavy features, cached representations, prediction tables, and run roots remain under verified local scratch. Final paper claims should cite the compact report and exact run entry, not rely on memory or an early exploratory table.

---

# 14. Core references and review status

## Direct and closest work

- Chowdhury, A., Cozzo, A., Ross, A. *JukeBox: A Multilingual Singer Recognition Dataset.* 2020. arXiv:2008.03507. **Reviewed.**
- Chowdhury, A., Cozzo, A., Ross, A. *Domain Adaptation for Speaker Recognition in Singing and Spoken Voice.* ICASSP 2022. DOI: 10.1109/ICASSP43922.2022.9746111. **Abstract/task verified; exact protocol and tables pending an accessible PDF. A previous “fully read” label was not sufficiently traceable and has been withdrawn.**
- Mehrabani, M., Hansen, J. H. L. *Speaker clustering for a mixture of singing and reading.* **Full paper reviewed.**
- Mehrabani, M., Hansen, J. H. L. *Singing speaker clustering based on subspace learning in the GMM mean supervector space.* **Full paper reviewed, including rendered table inspection.**
- Hansen, J. H. L., Bokshi, M., Khorram, S. *Speech variability: A cross-language study on acoustic variations of speaking versus untrained singing.* JASA 2020. **Full-paper review pending.**

## Singer identity and generation

- Torres, B., Lattner, S., Richard, G. *Singer Identity Representation Learning using Self-Supervised Techniques.* 2024. arXiv:2401.05064. **Task/protocol reviewed.**
- Zhang, L. et al. *Learning Singing From Speech.* 2019. arXiv:1912.10128.
- Zhang, L. et al. *DurIAN-SC.* 2020. arXiv:2008.03009.
- Xue, H. et al. *Learn2Sing.* 2020. arXiv:2011.08467.
- Xue, H. et al. *Learn2Sing 2.0.* 2022. arXiv:2203.16408.
- Dai, S. et al. *Everyone-Can-Sing.* 2025. arXiv:2501.13870.

## Singing SSL and datasets

- Tang, Y. et al. *SingOMD.* 2024. arXiv:2406.08905.
- Li, R. et al. *Self-Supervised Singing Voice Pre-Training towards Speech-to-Singing Conversion.* 2024. arXiv:2406.02429. **Abstract reviewed; full review pending.**
- Tamaru, H. et al. *JVS-MuSiC.* 2020. arXiv:2001.07044.
- Zhang, Y. et al. *GTSinger.* 2024. arXiv:2409.13832.
- Sharma, B. et al. *NHSS: A Speech and Singing Parallel Database.* 2020. arXiv:2012.00337. **Full baseline audit pending.**

## Domain adaptation and frozen representation geometry

- Lee, K. A. et al. *The CORAL+ Algorithm for Unsupervised Domain Adaptation of PLDA.* arXiv:1812.10260.
- Li, Z., Zhang, S., Chen, B. *CORAL++.* 2022. arXiv:2202.01092.
- van Niekerk, B. et al. *Analyzing Speaker Information in Self-Supervised Models to Improve Zero-Resource Speech Processing.* 2021. arXiv:2108.00917.
- Liu, O. D., Tang, H., Goldwater, S. *Self-supervised Predictive Coding Models Encode Speaker and Phonetic Information in Orthogonal Subspaces.* 2023. arXiv:2305.12464.
- Mohamed, M. et al. *Orthogonality and Isotropy of Speaker and Phonetic Information in Self-Supervised Speech Representations.* 2024. arXiv:2406.09200.
- Ruggiero, G. et al. *Eta-WavLM.* 2025. arXiv:2505.19273.
- Sun, J. et al. *Activation Steering for Accent Adaptation in Speech Foundation Models.* 2026. arXiv:2603.05813.

---

# 15. Final writing guidance

The paper should read as a representation analysis paper, not a model paper.

The narrative is:

1. Speech–singing speaker mismatch and speech-reference singing generation are already known.
2. Existing systems either train a recognizer/generator or align domains; they do not explain the same-person geometry already present in frozen generic SSL spaces.
3. In the tested SSL spaces, speech and singing are almost perfectly linearly mode-separable.
4. A large portion of this separation is a speaker-shared train-estimable displacement.
5. Correcting it improves held-out identity-relevant speaker retrieval and verification, with JVS as the primary evidence.
6. The improvement survives single-reference and matched-frame controls; multilingual GTSinger supplies a same-text mechanism replication but is language-confounded.
7. The mean is dominant, but the centered residual remains higher-rank.
8. Higher-rank structure and lower reconstruction MSE do not by themselves justify an individualized mapper.
9. ECAPA, acoustic baselines, SeedVC ambiguity, and the breathy negative result define the boundary of the interpretation rather than being hidden.

The value of the work is therefore the combination of:

- a strong empirical representation finding;
- a conservative mechanism interpretation;
- a robust speaker-disjoint evaluation package;
- explicit negative and superseded-result accounting;
- a mandatory global baseline for future personalized mapping work.

The paper should not claim that the real-world gap is globally “shallower” than Chowdhury 2022, nor that the recovered factor is pure identity. The strongest defensible phrasing is a controlled first-order geometric finding in selected frozen SSL representations.
