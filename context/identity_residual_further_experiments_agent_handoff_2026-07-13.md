# Speech--singing identity residual: further-experiments agent handoff

**Date:** 2026-07-13 JST

**Purpose:** executable handoff for the next agent who will implement the remaining experiments, run them on the lab cluster, and return paper-facing results.

**Scope:** close the current representation-geometry paper first. Frame-level synthesis and professional/amateur work are gated follow-ups, not reasons to enlarge the current paper indefinitely.

---

## 0. The instruction to the executing agent

Complete the following in order:

```text
R0  reproduce the existing headline baseline
 |
 +-- G1  held-out direction + magnitude
 +-- G2  layerwise curves over cached layers
 |
 +-- M1  six-cell cosine origin/alignment audit
 +-- M2  OAS-whitened cosine
 +-- M3  unnormalized Euclidean
 +-- M4  OAS Mahalanobis
 +-- M5  regularized LDA backend
 +-- A1  analogy-style ordering and offset consistency
 +-- C1  closest-work protocol boundary (Chowdhury 2022)
 |
 +-- write the paper-facing gate report and stop

Only after that report is reviewed:

S0 decoder reconstruction gate -> S1 frame-level alpha sweep -> S2 prosody grid

Separate project branch:

K0 professional/amateur dataset audit -> K1 factorized skill analysis
```

Do **not** compensate for a failed gate by adding models, datasets, neural metric learners, or mapper tuning. A failed robustness condition is a result.

Before editing code, read:

1. [AGENTS.md](../AGENTS.md) and [LAB_SERVER_WORKFLOW.md](../LAB_SERVER_WORKFLOW.md);
2. [one-on-one experiment plan](identity_residual_one_on_one_experiment_plan_2026-07-13.md);
3. [research dossier](identity_residual_research_dossier_2026-07-11.md);
4. [critical discussion](speech_singing_global_displacement_critical_discussion.md);
5. [metric/origin robustness audit](identity_residual_metric_origin_robustness_audit_2026-07-13.md);
6. the recent entries in the [experiment registry](EXPERIMENT_REGISTRY.md).

The fifth file contains expanded proofs and statistical details for the metric audit. This handoff is self-contained at the experiment level; when an algebraic or calibration detail is unclear, the audit is authoritative.

### Non-negotiable execution rules

- Reproduce the saved baseline before claiming a new result.
- Estimate every mean, displacement, covariance, whitening transform, LDA/PLDA transform, hyperparameter, and operating threshold without test-speaker information.
- Preserve the existing speaker splits and seed lists.
- Compare raw and corrected conditions on the same split, gallery, score backend, and fitted transform.
- Save per-split and per-speaker results, not only aggregate tables.
- Do not overwrite existing results.
- Do not scan or copy the 281 GB raw frame-feature roots unnecessarily; use compact caches where available.
- Run heavy work on a `valkyrie` compute node with verified `/localdisk`, never on `athena`.
- Add completed runs to `context/EXPERIMENT_REGISTRY.md` before ending.

---

## 1. Fixed scientific object and terminology

The speech/singing labels are known. This experiment does **not** perform unsupervised clustering.

For utterance \(u\), a frozen SSL model produces frame representations \(X_u\in\mathbb{R}^{T\times F}\). The current utterance vector is

\[
z_u=
\left[
\operatorname{mean}_t X_u;
\operatorname{std}_t X_u
\right].
\]

For speaker \(i\), average utterance vectors within mode:

\[
C_i^S=\operatorname{mean}_{u\in S_i}z_u,
\qquad
C_i^G=\operatorname{mean}_{u\in G_i}z_u.
\]

Here \(S\) means speech and \(G\) means singing. The individual displacement is

\[
\Delta_i=C_i^G-C_i^S.
\]

Using training speakers only,

\[
d=\mu_{\Delta,\mathrm{train}}
=
\frac{1}{N_{\mathrm{train}}}
\sum_{i\in\mathrm{train}}\Delta_i
=\mu_G-\mu_S,
\]

where

\[
\mu_S=\operatorname{mean}_{i\in\mathrm{train}}C_i^S,
\qquad
\mu_G=\operatorname{mean}_{i\in\mathrm{train}}C_i^G.
\]

The direct held-out prediction is

\[
\widehat C_i^G=C_i^S+d.
\]

### Current defensible claim

> Frozen SSL speech and singing centroids exhibit a speaker-shared, train-estimable displacement. Applying this displacement improves identity-relevant cross-mode correspondence for unseen speakers under the evaluated retrieval and verification protocols.

Do not convert this into any of the following without new evidence:

- a pure timbre vector;
- removal of all singing information;
- an exact individual speech-to-singing transformation;
- a complete or universal “singing direction”;
- a practical biometric system;
- a deployable speech-to-singing generator;
- a novel difference-of-means algorithm.

---

## 2. Current result that R0 must reproduce

The primary JVS/JVS-MuSiC evaluation uses 100 speakers and repeated 60 train / 20 dev / 20 test speaker-disjoint splits. The test gallery contains 20 unseen speakers; random R@1 is 5%.

| Model/layer | Held-out \(\cos(\Delta_i,d)\) | R@1 raw -> corrected | EER raw -> corrected | TMR@FMR=1% raw -> corrected |
|---|---:|---:|---:|---:|
| WavLM Base+ L12 | 0.917 | 13.5% -> 36.3% | 42.1% -> 26.6% | 3.0% -> 14.5% |
| HuBERT Base L6 | 0.891 | 19.5% -> 60.5% | 37.1% -> 18.6% | 2.0% -> 30.0% |
| MERT-95M L3 | 0.806 | 36.5% -> 66.0% | 28.5% -> 16.1% | 14.3% -> 38.8% |

Wrong-sign and same-norm random directions do not reproduce the gain. The effect survives one-speech-reference and matched-frame conditions. GTSinger supplies same-text mechanism evidence but is not clean language-independent identity evidence.

### Important correction to the saved literature narrative

The \(0.78\text{--}0.92\) direction alignment is held out. The previously quoted \(65\%\text{--}83\%\) uncentered energy was computed on **training residuals used to estimate the mean**. It must not be described as held-out explained energy. G1 below fixes this.

### R0 pass condition

For all three JVS headline configurations, reproduce the saved mean R@1 and EER within numerical tolerance under the same seeds. A difference greater than 0.5 percentage point requires an audit before continuing.

Audit, at minimum:

- ordered speaker list;
- split membership;
- centroid input rows and speech-reference cap;
- feature-cache hash/path;
- vector dtype;
- L2 normalization placement;
- query and gallery ordering.

---

## 3. Fixed data, split, and model protocol

| Dataset | Speaker protocol | Repetitions | Role | Main limitation |
|---|---|---:|---|---|
| JVS + JVS-MuSiC | 60 train / 20 dev / 20 test | 20 fixed seeds | primary identity evidence | unmatched speech/song content; one common singing item limits independent singing reliability |
| GTSinger clean same-text control | 10 train / 10 test | 50 fixed seeds | same-text mechanism and analogy audit | 20 singers; language strongly confounded with singer; no dev split |

### Headline configurations

- `wavlm_l12`
- `hubert_l6`
- `mert_l3`

Run the complete metric/backend audit on these six dataset-by-model settings. Use additional layers for G2 only; do not cross every backend with every layer.

### Seed lists

- JVS: preserve the 20 seeds in `run_identity_residual_robustness.py::JVS_SEEDS`.
- GTSinger: preserve the 50 seeds in `run_identity_residual_same_text.py::GTSINGER_SEEDS`.

### Existing server paths

Run from the canonical server repository:

```text
/home/bowen/bowen_lab/projects/singing_identity
```

JVS:

```text
manifest:
/localdisk/bowen/singing_identity/runs/jvs_music_retrieval_2026-07-08/manifests/jvs_music_utterances.jsonl

feature root:
/localdisk/bowen/singing_identity/features/jvs_music_2026-07-08

compact full-sweep caches:
/localdisk/bowen/singing_identity/runs/identity_residual_jvs_fullsweep_2026-07-09/cache
```

GTSinger:

```text
pair manifest:
/localdisk/bowen/singing_identity/runs/stage1_repaired_200_fresh_local/manifests/gtsinger_pairs.jsonl

utterance manifest:
/localdisk/bowen/singing_identity/runs/stage1_repaired_200_fresh_local/manifests/gtsinger_utterances.jsonl

feature root:
/localdisk/bowen/singing_identity/features/stage1_repaired_200_fresh_local

clean same-text pair caches:
/localdisk/bowen/singing_identity/runs/identity_residual_same_text_2026-07-10/cache
```

The large raw feature roots and most compact result CSVs exist on `valkyrie03`; they are intentionally excluded from normal laptop sync. Work on the server or transfer only small tables. Do not copy the full feature roots.

### Existing code to reuse

- `scripts/probing/run_identity_residual_suite.py`
  - `MODEL_SPECS`
  - `load_feature_cache`
  - `make_speaker_split`
  - `centroids`
  - `retrieval`
- `scripts/probing/run_identity_residual_robustness.py`
  - `load_jvs`
  - `load_gtsinger`
  - EER/TMR utilities
  - current geometry and speaker-level uncertainty utilities
- `scripts/probing/run_identity_residual_final_validation.py`
  - current direct global, wrong-sign, random, reverse, and mode-centered variants
- `scripts/probing/run_identity_residual_same_text.py`
  - independently verified same-text pair selection
  - GTSinger pair-vector loader and speaker centroids
- `scripts/probing/run_identity_residual_suite.py::split_half_vectors`
  - starting point for GTSinger independent halves
- `scripts/probing/run_technique_directions.py::analogy_top1`
  - structural precedent only; the new analogy definition differs

Create a new runner instead of mutating or overwriting completed results:

```text
scripts/probing/run_identity_residual_metric_robustness.py
tests/test_identity_residual_metric_robustness.py

run root:
/localdisk/bowen/singing_identity/runs/identity_residual_metric_robustness_2026-07-13

small result root:
results/identity_residual_metric_robustness_2026-07-13
```

Use:

```bash
uv run --extra probe python scripts/probing/run_identity_residual_metric_robustness.py \
  --run-root /localdisk/bowen/singing_identity/runs/identity_residual_metric_robustness_2026-07-13 \
  --results-dir results/identity_residual_metric_robustness_2026-07-13
```

The exact CLI may be extended, but it must record all resolved arguments in `experiment_card.yaml`.

---

## 4. G1 -- held-out direction and magnitude closure

### Question

Does the train mean predict both the **direction and length** of unseen speakers' speech-to-singing displacements?

### Required quantities

For every held-out speaker:

\[
a_i=\cos(\Delta_i,d),
\]

\[
\rho_i=
\frac{\|\Delta_i-d\|_2}
{\|\Delta_i\|_2},
\]

\[
\kappa_i=
\frac{\|d\|_2}
{\|\Delta_i\|_2},
\]

and, as a useful length/projection diagnostic,

\[
\beta_i=
\frac{\Delta_i^\top d}{\|d\|_2^2}.
\]

For every split:

\[
E_{\mathrm{test}}
=
1-
\frac{\sum_{i\in\mathrm{test}}\|\Delta_i-d\|_2^2}
{\sum_{i\in\mathrm{test}}\|\Delta_i\|_2^2}.
\]

### Interpretation

- \(E_{\mathrm{test}}>0\): the train mean predicts held-out residuals better than predicting zero displacement.
- \(\rho_i<1\): the global vector improves residual reconstruction for speaker \(i\).
- \(\rho_i=0\): exact residual prediction.
- \(\rho_i>1\): the global vector makes residual reconstruction worse for that speaker.
- \(\beta_i\approx1\): the projection length along \(d\) is close to the train mean's length.

### Implementation shortcut already present

`run_identity_residual_suite.py::run_b0` already computes `test_delta`, `mu`, and `residual = test_delta - mu`; the residual is currently unused. `run_identity_residual_robustness.py::add_geometry_rows` already saves held-out alignment and norms, but its explained-energy table is explicitly train-only. Add new held-out fields or a new CSV; do not relabel the old training-energy output.

### Outputs

- `heldout_magnitude_per_speaker_split.csv`
- `heldout_magnitude_summary.csv`
- per dataset/model: distribution of \(a_i,\rho_i,\kappa_i,\beta_i\)
- fraction of unique speakers with aggregate \(\rho_i<1\)
- paired speaker bootstrap interval for \(E_{\mathrm{test}}\) or its numerator/denominator recomputation

### Decision gate

- **Direction + magnitude supported:** mean \(E_{\mathrm{test}}>0\), speaker-level 95% interval above zero, and a majority of unique speakers have \(\rho_i<1\).
- **Direction only:** cosine alignment remains positive but \(E_{\mathrm{test}}\) is mixed, near zero, or negative. Retain the shared-direction and functional retrieval claims; do not say that \(d\) accurately reconstructs individual displacements.

---

## 5. G2 -- layerwise curve

### Question

Are the three headline layers representative, and where in model depth do direction, magnitude, and identity utility appear?

### Available layer reality

Raw frame archives exist remotely at layers 3/6/9/12 for all three SSL families. `MODEL_SPECS` currently omits `wavlm_l3` and `hubert_l3`; add them only after validating the expected archive paths. Compact JVS caches already exist for 10 of 12 SSL configurations. Build only the missing compact caches. GTSinger non-headline layers will need compact pair caches created from existing frame archives.

### Required configurations

- WavLM Base+: L3, L6, L9, L12
- HuBERT Base: L3, L6, L9, L12
- MERT-95M: L3, L6, L9, L12

### Metrics per layer

- mean held-out \(\cos(\Delta_i,d)\);
- held-out \(E_{\mathrm{test}}\);
- median \(\rho_i\);
- raw and corrected R@1;
- raw and corrected EER;
- TMR@FMR=1% only where calibration has adequate resolution.

### Figure

Produce one four-panel figure per dataset or one compact faceted figure:

```text
panel 1: held-out direction cosine vs layer
panel 2: held-out E_test vs layer
panel 3: corrected - raw R@1 vs layer
panel 4: corrected - raw EER vs layer
```

Do not run whitening/LDA/PLDA at every layer. The layer curve is a diagnostic against cherry-picking, not a new exhaustive benchmark.

---

## 6. What the coordinate-origin audit can and cannot invalidate

Cosine between centroids is origin sensitive:

\[
\cos(x+c,y+c)\neq\cos(x,y).
\]

However, a common translation cancels in every displacement:

\[
(C_i^G+c)-(C_i^S+c)=C_i^G-C_i^S.
\]

Therefore the following are origin invariant:

- \(d\);
- held-out \(\cos(\Delta_i,d)\);
- \(\|\Delta_i-d\|\), \(E_{\mathrm{test}}\), and \(\rho_i\).

The origin audit primarily challenges the **retrieval/verification interpretation**, not the existence of a shared residual direction. Keep those claims separate in the report.

Define

\[
a=\mu_S,
\qquad
b=\mu_G,
\qquad
d=b-a,
\qquad
m=\frac{a+b}{2}.
\]

### Exact identities that must appear in code assertions and reporting

Mode-centered cosine is not a pure origin control:

\[
\cos(C_i^S-a,C_j^G-b)
=
\cos((C_i^S+d)-b,C_j^G-b),
\]

and

\[
\cos(C_i^S-a,C_j^G-b)
=
\cos((C_i^S+d/2)-m,(C_j^G-d/2)-m).
\]

Under Euclidean or a fixed Mahalanobis metric,

\[
\|(C_i^S+d)-C_j^G\|
=
\|(C_i^S-a)-(C_j^G-b)\|.
\]

Thus query-only correction, symmetric correction, and mode centering yield the same corrected **distance** ranking. Do not present them as independent distance experiments.

Also,

\[
\left\|
\frac{x}{\|x\|}-
\frac{y}{\|y\|}
\right\|^2
=2-2\cos(x,y).
\]

Euclidean distance after L2 normalization is exactly redundant with cosine.

---

## 7. M1 -- six-cell cosine origin/alignment factorial

For each split, fit \(a,b,d,m\) from train speakers. Define alignment placement:

\[
\begin{aligned}
\texttt{none}:&\quad q=C^S,\quad g=C^G,\\
\texttt{query}:&\quad q=C^S+d,\quad g=C^G,\\
\texttt{symmetric}:&\quad q=C^S+d/2,\quad g=C^G-d/2.
\end{aligned}
\]

Run exactly:

| ID | Common origin | Alignment | Score | Scientific role |
|---|---|---|---|---|
| `cos_o0_none` | 0 | none | \(\cos(C^S,C^G)\) | exact raw baseline |
| `cos_o0_query` | 0 | query | \(\cos(C^S+d,C^G)\) | exact existing correction |
| `cos_o0_sym` | 0 | symmetric | \(\cos(C^S+d/2,C^G-d/2)\) | correction-placement diagnostic |
| `cos_om_none` | \(m\) | none | \(\cos(C^S-m,C^G-m)\) | pure shared-origin test |
| `cos_om_query` | \(m\) | query | \(\cos(C^S+d-m,C^G-m)\) | correction under the same new origin |
| `cos_om_sym` | \(m\) | symmetric | \(\cos(C^S-a,C^G-b)\) | exact mode-centered condition |

This factorial answers two distinct questions:

1. Holding alignment fixed, how much does moving the common origin change performance?
2. Holding the origin fixed, how much does explicit train-only alignment change performance?

### Origin-dominance diagnostic

For a metric converted to higher-is-better utility \(u\), define

\[
G_{\mathrm{current}}
=u(\texttt{cos\_o0\_query})-u(\texttt{cos\_o0\_none}),
\]

\[
G_{\mathrm{origin}}
=u(\texttt{cos\_om\_none})-u(\texttt{cos\_o0\_none}),
\]

\[
G_{\mathrm{conditional}}
=u(\texttt{cos\_om\_query})-u(\texttt{cos\_om\_none}).
\]

Report \(G_{\mathrm{origin}}/G_{\mathrm{current}}\) without clipping.

Use `origin-dominated` only as a declared descriptive label when:

- pooled centering recovers at least 80% of the current gain; and
- the additional centered-origin correction has a paired interval including zero; and
- this holds consistently for at least two of the three JVS SSL models.

The 80% rule is a predeclared interpretation threshold, not a natural constant.

---

## 8. M2--M4 -- whitening and origin-invariant distances

The pooled embedding dimension is about 1536, while there are only 120 train speaker-mode centroids in JVS and 20 in GTSinger. The empirical covariance is singular. Unregularized whitening and a raw pseudoinverse are not acceptable primary analyses.

Construct a speaker-balanced train matrix with exactly one speech and one singing centroid per train speaker:

\[
Z_{\mathrm{train}}
=
[C_1^S,C_1^G,\ldots,C_N^S,C_N^G]^\top.
\]

Fit a train-only Oracle Approximating Shrinkage (OAS) covariance as the primary estimator:

\[
\Sigma_\lambda
=(1-\lambda)\widehat\Sigma
+\lambda\frac{\operatorname{tr}(\widehat\Sigma)}{D}I.
\]

Record:

- fitted shrinkage;
- covariance trace;
- minimum effective eigenvalue;
- condition number;
- fit speaker IDs and a transform hash.

Use float64 and only a small eigenvalue floor as a numerical guard.

### M2: OAS-whitened cosine

With \(W=\Sigma_\lambda^{-1/2}\), evaluate both conditions using the **same** \(m,W\):

\[
\cos(W(C^S-m),W(C^G-m)),
\]

\[
\cos(W(C^S+d-m),W(C^G-m)).
\]

Also compute whitened residual alignment and \(E_{\mathrm{test}}\):

\[
\cos(W\Delta_i,Wd),
\qquad
E_{\mathrm{test}}^{W}
=1-
\frac{\sum_i\|W(\Delta_i-d)\|^2}
{\sum_i\|W\Delta_i\|^2}.
\]

### M3: unnormalized Euclidean

Use higher-is-better score

\[
s_E(x,y)=-\|x-y\|_2^2.
\]

Run raw and query-corrected. Do not L2 normalize first.

### M4: OAS Mahalanobis

Use

\[
s_M(x,y)
=-(x-y)^\top\Sigma_\lambda^{-1}(x-y).
\]

Run raw and query-corrected under the same fitted covariance.

### Shrinkage sensitivity

OAS is the primary result. In an appendix, report the fixed grid

\[
\lambda\in
\{0.1,0.25,0.5,0.75,0.9,0.99,1.0\}
\]

without choosing the best test value.

At \(\lambda=1\):

- whitened-cosine rankings must equal pooled-centered-cosine rankings;
- Mahalanobis rankings must equal Euclidean rankings.

### Interpretation

- Correction helps whitened cosine: the effect survives scale/correlation normalization.
- Correction helps Euclidean and Mahalanobis: the result is not only an angular/origin artifact.
- Whitening alone is strong and correction still helps: metric normalization and translation are complementary.
- A strong raw backend removes the incremental correction gain: the functional result is largely absorbed by that backend; the origin-invariant residual-direction result remains.

---

## 9. M5 -- regularized LDA; PLDA is conditional

### Required LDA

Train a speaker LDA using one raw speech centroid and one raw singing centroid per **training** speaker, both assigned the same speaker label. This balances speakers and modes and gives the backend direct access to cross-mode within-speaker variation.

Fit the raw transform once. Apply that same transform to raw and corrected test vectors; do not refit a separate LDA after correction for the primary contrast.

Use shrinkage LDA (`solver=eigen`, automatic/analytic shrinkage).

JVS candidate output dimensions:

\[
k\in\{8,16,32,59\}.
\]

Select \(k\) using raw dev EER only; break ties by higher raw dev MRR and then smaller \(k\). Lock the transform and \(k\) before test scoring.

GTSinger has ten train speakers and no dev split. Treat LDA as exploratory:

- fixed primary \(k=8\), subject to the valid maximum;
- report \(k\in\{2,4,8\}\) as a no-selection sensitivity grid;
- never select the best value on test.

Report both LDA-cosine and, if useful, LDA-Euclidean, but do not multiply backend variants unnecessarily.

### Optional PLDA

There is no PLDA dependency or implementation in this repository. Do not write an unvalidated ad hoc PLDA merely to fill a table.

PLDA may be added only if:

- a trusted implementation is available;
- synthetic tests verify score orientation and same-speaker behavior;
- it is run on JVS only;
- dev speakers select configuration;
- raw and corrected use the same fitted backend;
- instability across reasonable ranks is reported.

With 60 train identities and two centroid observations per identity, PLDA is marginal; GTSinger PLDA is not credible. A missing PLDA row is preferable to an unreliable one.

### Backend interpretation

- Raw LDA matches/exceeds corrected cosine and correction adds no stable gain: classic supervised compensation absorbs the practical value; position the contribution as simple frozen-space geometry.
- A strong raw LDA still improves after correction: the correction is backend-robust and the practical representation claim strengthens.
- Raw LDA is weak or unstable: disappearance of the gain is inconclusive.

---

## 10. Common identity evaluation protocol

Every compatible condition must produce one higher-is-better score matrix. For distances, negate squared distance.

The primary test matrix is strictly cross-modal:

- row \(i\): held-out speaker \(i\)'s speech query;
- column \(j\): held-out speaker \(j\)'s singing gallery vector;
- genuine verification trials: diagonal \((i=i)\);
- impostor verification trials: off-diagonal \((i\neq j)\).

Do not mix speech candidates into the primary singing gallery. That would answer the different and easier question “which vector shares the query's mode?” rather than “which singer is the same person as this speech query?” Same-mode speech--speech or singing--singing results may be shown only as independently sampled reference ceilings; never compare a centroid with itself.

### Required metrics

Primary:

- R@1;
- descriptive held-out EER;
- JVS TMR at dev-calibrated FMR=1%.

Secondary:

- R@5;
- MRR;
- mean and median rank;
- ROC-AUC;
- genuine-minus-impostor margin within the same backend;
- mean/SD genuine and impostor scores;
- optional \(d'\).

Do not numerically compare raw margins across cosine, Euclidean, Mahalanobis, and PLDA because score scales differ. Compare margins only within a backend.

### Calibration

JVS:

- fit representation/backend on 60 train speakers;
- use 20 dev speakers to select LDA/PLDA hyperparameters and condition-specific operating thresholds;
- choose the lowest dev threshold with empirical FMR no greater than 1%;
- record threshold, achieved dev FMR, and impostor count;
- apply the locked threshold once to 20 test speakers.

The new suite's dev-calibrated TMR is not numerically identical to the legacy train-calibrated TMR. Preserve a `legacy_train_calibrated` column if direct historical comparison is required, but label the calibration source.

GTSinger:

- has only 90 ordered train impostor centroid trials and no dev set;
- one false match already equals 1.11%;
- do not treat TMR@FMR=1% as a primary stable operating point;
- primary metrics are R@1/MRR, ROC-AUC, and descriptive EER;
- retain legacy 1% TMR only with `insufficient_resolution/non_independent_calibration` status;
- an exploratory train-cross-fitted TMR@FMR=10% may be reported.

### Pairing and uncertainty

Every correction claim is a paired comparison within the same backend and split:

\[
\Delta\mathrm{R@1}
=\mathrm{R@1}_{\mathrm{corrected}}
-\mathrm{R@1}_{\mathrm{raw}},
\]

\[
\Delta\mathrm{EER}
=\mathrm{EER}_{\mathrm{corrected}}
-\mathrm{EER}_{\mathrm{raw}}.
\]

Repeated split seeds reuse speakers and are not independent subjects.

For query-decomposable metrics:

1. aggregate each unique speaker's paired difference over all test appearances;
2. bootstrap unique speakers 10,000 times;
3. report a paired percentile interval;
4. run a two-sided speaker-level sign-flip test;
5. apply Holm correction across the three predeclared SSL models within each primary backend/metric family.

For EER/AUC/TMR:

- always report paired per-split deltas and sign consistency;
- for paper-facing intervals, use a speaker-cluster bootstrap/Bayesian bootstrap over saved score matrices as specified in `identity_residual_metric_origin_robustness_audit_2026-07-13.md`;
- state that intervals condition on fitted train transforms, while the repeated-split distribution describes train-set sensitivity.

### Practical “absorbed” label

A backend may be called `absorbing` only when:

1. its raw performance is at least as good as the current corrected-cosine baseline; and
2. adding \(d\) produces no stable paired improvement; and
3. the residual mean change is below a predeclared practical tolerance, suggested as 2 percentage points for R@1 and 1 percentage point for EER.

These tolerances are analysis conventions, not universal standards. Always show the continuous deltas and intervals.

---

## 11. A1 -- analogy-style evaluation inspired by phonological arithmetic

This is **analogy-style**, not an exact reproduction. The phonological paper contains multiple feature relations across phones; speech-to-singing supplies one broad relation type.

Do not compare our cosine numerically with its 92%/94% headline numbers. Those are task-specific fractions of successful analogy orderings, not cosine values.

### GTSinger lower--arithmetic--upper ordering

Use only GTSinger, where multiple same-text observations allow utterance-disjoint halves. Do not use JVS's within-file frame halves as an independent singing upper bound.

For each held-out speaker, deterministically form disjoint A/B observations. Fit \(d\) from training speakers only. Define:

\[
s_i^{\mathrm{arith}}
=
\cos(C_{i,B}^G,C_{i,A}^S+d),
\]

\[
s_i^+
=
\cos(C_{i,B}^G,C_{i,A}^G),
\]

\[
s_{ij}^-
=
\cos(C_{i,B}^G,C_{j,B}^G),
\qquad j\neq i.
\]

Report:

- raw same-person score \(\cos(C_{i,B}^G,C_{i,A}^S)\);
- arithmetic score;
- same-person singing-repeat upper score;
- all or repeated different-person lower scores;
- fraction satisfying

\[
s_{ij}^-<s_i^{\mathrm{arith}}<s_i^+;
\]

- 99% speaker-clustered/bootstrap interval;
- corrected retrieval R@1 in the same split-half gallery.

Because the “upper” can be noisy, also report the two inequalities separately. Arithmetic exceeding a noisy upper reference is not automatically scientific failure.

Run this ordering under:

- raw origin cosine;
- train pooled-centered cosine.

Do not repeat the full metric grid.

### Offset-consistency AUC

For held-out speakers, construct true offsets

\[
\Delta_i=C_i^G-C_i^S
\]

and endpoint-mismatched offsets

\[
\Delta_{i\to j}^{-}=C_j^G-C_i^S,
\qquad j\neq i.
\]

L2-normalize offsets and score their alignment with the train direction:

\[
r(v)=\cos(v,d).
\]

Compute ROC-AUC distinguishing true from mismatched offsets and compare it with a shuffled-identity permutation distribution.

Call this `offset-consistency AUC` or `PCS-inspired AUC`, not exact PCS unless the paper's exact pairing construction is reproduced.

### Analogy gate

- Ordering exceeds the identity-permutation null and offset AUC has a speaker-level interval above 0.5: `analogy-like shared relation` is supported.
- Ordering/AUC fail: do not use “vector arithmetic.” The direction, magnitude, retrieval, and verification results remain separate and may still stand.

Primary references:

- paper: <https://arxiv.org/abs/2602.18899>
- official implementation: <https://github.com/juice500ml/phonetic-arithmetic>
- PCS implementation: <https://github.com/juice500ml/phonetic-arithmetic/blob/main/pcs.py>

---

## 12. C1 -- closest-work protocol boundary is required

Chowdhury, Cozzo, and Ross (ICASSP 2022), *Domain Adaptation for Speaker Recognition in Singing and Spoken Voice*, is the closest direct same-speaker cross-modal verification work. It is not safe to use its headline numbers as if they were on the same benchmark: JukeBox-V2 is in-the-wild and multilingual, while the current primary result uses controlled JVS/JVS-MuSiC centroids and a 20-speaker gallery.

Before writing the novelty comparison, obtain and inspect the five-page paper through institutional access or an author copy. If the PDF remains unavailable, mark unknown fields `PDF_REQUIRED`; do not reconstruct them from citations or abstracts.

Extract exactly:

- enrollment and test modalities;
- whether train and evaluation identities are disjoint;
- utterance, segment, or speaker-centroid trial unit;
- genuine/impostor construction and trial counts;
- backbone and embedding dimension;
- CORAL, DeepCORAL, triplet, or other adaptation objective and fitting data;
- whether first-moment centering is part of the method;
- score backend and threshold-calibration source;
- EER/TMR definitions and every comparable table value.

Then create two outputs:

1. `closest_work_protocol_matrix.md`: a row-by-row protocol comparison among Chowdhury 2022, Mehrabani/Hansen's LDA/PLDA work, and this project;
2. `closest_work_metric_mapping.csv`: each reported metric, its direction, calibration source, trial unit, gallery size if applicable, and a `directly_comparable` boolean with justification.

If the paper's trial construction can be matched without changing the fixed speaker splits, re-express the saved JVS score matrices under that construction as a labeled supplementary analysis. Do not tune a new model merely to imitate the earlier system. If the construction cannot be matched, stop at the protocol matrix.

The comparison must answer:

- What was already shown by trained speech--singing domain adaptation?
- What is specific here to frozen generic SSL, a speaker-balanced train-only mean displacement, and held-out geometric analysis?
- Does M5 show that classic supervised compensation absorbs the current gain?
- Which numerical comparisons are invalid because dataset, gallery, trials, or calibration differ?

Allowed conclusion:

> Prior work established cross-modal degradation and trained style/domain compensation. This project tests whether selected frozen SSL spaces contain a held-out speaker-shared first-order displacement and audits when that simple geometry helps.

Forbidden conclusion without a protocol-matched experiment:

> The speech--singing gap is globally shallower than Chowdhury 2022.

Primary references:

- paper record/DOI: <https://doi.org/10.1109/ICASSP43922.2022.9746111>
- JukeBox dataset paper: <https://www.cse.msu.edu/~rossarun/pubs/ChowdhuryCozzoRossJukeBox_INTERSPEECH2020.pdf>

---

## 13. N1 -- nonlinear mode probe is conditional

The current result shows that a train-only **linear centroid-level** mode probe falls from AUC near 1.0 to approximately chance after symmetric correction. It does not show that all mode information disappears.

Run one nonlinear probe only if the intended paper claim remains stronger than “dominant linear centroid-level mode separability falls to chance.”

Allowed choice:

- one regularized RBF-SVM or a very small MLP;
- train-only fitting;
- dev-selected hyperparameters for JVS;
- fixed/inner-train protocol for GTSinger;
- untouched test speakers;
- raw vs corrected AUC.

Do not search many nonlinear classifiers. If nonlinear mode AUC remains above chance, report:

> The correction removes the dominant linear centroid-level mode separation, while nonlinear mode information remains recoverable.

If the paper already uses the narrower linear wording, skip N1.

---

## 14. S0--S2 -- conditional downstream synthesis

This branch is not paper-blocking and must not begin before the representation gate report is written.

### Why the current pooled vector cannot be decoded

The current \(z=[\operatorname{mean};\operatorname{std}]\) has discarded temporal order, F0 contour, duration/rhythm trajectory, energy trajectory, and phase. It cannot be passed directly to a vocoder.

The phonological-arithmetic paper instead estimates a mean-difference direction, returns to a WavLM-large final-layer frame sequence, adds the direction to selected frames, and decodes with a matching WavLM-conditioned Vocos.

### S0: decoder feasibility gate

Use one compatible configuration only:

- WavLM-large final layer;
- matching WavLM-conditioned Vocos from the phonological paper;
- a newly estimated train-only \(F\)-dimensional **frame-mean** speech-to-singing direction, not the current \(2F\) pooled vector.

First perform \(\alpha=0\) copy reconstruction for held-out real speech and real singing.

Stop if:

- singing reconstruction is materially worse than speech;
- content is not intelligible;
- speaker identity is not preserved well enough to interpret an intervention.

If the decoder is out of domain, report `S0 failed/blocked`. Do not begin open-ended decoder training automatically.

### S1: frame-level dose response

Estimate

\[
d_{\mathrm{frame}}
=
\operatorname{mean}H^G_{\mathrm{train}}
-
\operatorname{mean}H^S_{\mathrm{train}},
\]

with one-speaker-one-vote balancing. For held-out speech frames:

\[
H_\alpha=H+\alpha d_{\mathrm{frame}},
\qquad
\alpha\in\{-1,0,0.25,0.5,0.75,1,1.5\}.
\]

Controls:

- \(\alpha=0\) decoder reconstruction;
- wrong sign;
- same-norm random direction;
- optional oracle individual direction as a labeled upper bound only.

Measure:

- independent speech/singing probability;
- F0 range, stability, and trajectory;
- vibrato rate and extent;
- HNR and spectral tilt;
- ASR/CER or phone preservation;
- independent speaker similarity;
- naturalness;
- blinded pairwise “more singing-like?” judgments.

Proceed only if positive \(\alpha\) produces a monotonic, control-beating change in prespecified singing-related properties while intelligibility, identity, and naturalness remain acceptable.

Interpret S1 as singing-related phonation/mode steering at fixed timing. A constant frame shift cannot invent melody, new note durations, or time warping.

### S2: separate mode from melody/rhythm

Only if S1 passes, cross residual strength with prosody:

| Prosody supplied to the generator | \(\alpha=0\) | 0.25 | 0.5 | 0.75 | 1.0 |
|---|---:|---:|---:|---:|---:|
| original speech F0/duration | | | | | |
| target singing F0/duration | | | | | |

Interpretation:

- target prosody alone works: melody/rhythm dominate;
- \(\alpha\) helps under speech prosody: the direction has causal acoustic semantics;
- target prosody and \(\alpha\) interact: the direction supplies complementary phonation/style;
- no monotonic effect: the centroid direction is descriptive, not a useful synthesis control.

GTSinger supplies paired speech, scores, alignments, and an official AlignSTS benchmark:

- dataset/paper: <https://arxiv.org/abs/2409.13832>
- code: <https://github.com/AaronZ345/GTSinger>

---

## 15. K0--K1 -- professional/amateur is a separate branch

Do not subtract the centroid of a professional dataset such as GTSinger/VocalSet from the centroid of an unrelated amateur dataset such as DAMP. The resulting vector would combine skill with speaker, microphone, genre, repertoire, language, mastering, and dataset pipeline.

### K0: dataset feasibility audit

For every candidate dataset, record:

- operational definition of professional/amateur;
- number of singers and recordings per singer;
- same-song/same-score coverage;
- whether the same singer supplies both conditions;
- longitudinal or before/after-training recordings;
- language and genre;
- microphone/session domain;
- pitch/timing/technique annotations;
- independent teacher ratings;
- license and actual access.

Minimum candidates:

- Neural Singing Voice Beautifier / PopBuTFy: exact amateur-to-professional beautification precedent with parallel versions and open code, but the “amateur” condition is simulated by trained performers rather than a longitudinal novice-to-expert trajectory.
  Paper: <https://aclanthology.org/2022.acl-long.549/>
  Code/data instructions: <https://github.com/MoonInTheRiver/NeuralSVB>
- Jingju a cappella corpus: professional and amateur singers in one cultural/recording program; useful for representation audit, but skill remains speaker-confounded.
  <https://arxiv.org/abs/1708.03986>
- Isophonics Singing Voice Audio Dataset: professional, semi-professional, and amateur singers; small and mostly nonparallel.
  <https://isophonics.net/SingingVoiceDataset.html>

### K1: factorized skill analysis

Treat proficiency as a multidimensional object:

- pitch accuracy;
- onset/rhythm accuracy;
- breath control;
- register and phonation control;
- vibrato stability;
- dynamics;
- diction;
- expression.

Prefer, in order:

1. the same singer over training time;
2. matched singers performing the same material in the same recording domain;
3. within-dataset speaker-disjoint analysis;
4. cross-dataset analysis only as explicitly domain-confounded exploration.

Begin with prediction/correlation against independent skill measurements. Attempt steering only if a direction or low-dimensional subspace:

- replicates across unseen singers;
- survives song/session/domain controls;
- correlates monotonically with external skill measurements;
- preferably appears within the same singer longitudinally.

The preferred question is:

> Which independently measurable singing skills form reproducible directions or subspaces, and which require temporal or nonlinear modeling?

Do not begin with “Is professionalism one direction?”

---

## 16. Required implementation assertions

The new metric runner must fail loudly unless all of these hold within numerical tolerance:

1. `d == mean(G_train - S_train) == mean(G_train) - mean(S_train)`.
2. `cos_om_sym` input vectors equal explicit mode-centered vectors.
3. Euclidean query-corrected scores/ranks equal symmetric-corrected and mode-centered distances.
4. Mahalanobis query-corrected scores/ranks equal symmetric-corrected distances.
5. L2-normalized Euclidean rankings equal cosine rankings.
6. OAS minimum eigenvalue is positive and all transformed scores are finite.
7. At \(\lambda=1\), whitened-cosine rankings equal pooled-centered-cosine rankings.
8. At \(\lambda=1\), Mahalanobis rankings equal Euclidean rankings.
9. Train, dev, and test speaker-ID sets are disjoint for every seed.
10. No test vector contributes to origin, covariance, backend, hyperparameter, or threshold fitting.
11. Raw and corrected rows within a backend have identical gallery IDs and transform hashes.
12. Higher-is-better score orientation is verified by a synthetic same-speaker test.

Add synthetic tests with a known shared domain offset and known speaker signal. The tests should verify that:

- common centering can change cosine but not Euclidean distances;
- the correct offset improves the intended cross-mode matching;
- wrong-sign and random directions do not systematically reproduce the improvement;
- held-out \(E_{\mathrm{test}}\) is positive when the train offset generalizes and negative when it is deliberately misspecified.

---

## 17. Required artifact contract

The runner must produce compact, reproducible artifacts:

```text
results/identity_residual_metric_robustness_2026-07-13/
  README_results.md
  experiment_card.yaml
  fit_audit.csv
  baseline_reproduction.csv
  heldout_magnitude_per_speaker_split.csv
  heldout_magnitude_summary.csv
  layerwise_metrics_per_split.csv
  layerwise_summary.csv
  metric_results_per_split.csv
  metric_paired_deltas.csv
  metric_speaker_aggregate.csv
  analogy_results_per_speaker_split.csv
  analogy_summary.csv
  closest_work_protocol_matrix.md
  closest_work_metric_mapping.csv
  score_matrices/
    <dataset>__<model>__<seed>__<condition>.npz
  figures/
    layerwise_geometry.png
    origin_alignment_factorial.png
    metric_backend_robustness.png
    analogy_ordering.png
  gate_report.md
```

Each metric row must include at least:

```text
dataset, model, layer, split_seed,
condition_id, score_family, origin, alignment,
covariance_method, shrinkage, lda_dim,
train_speakers, dev_speakers, test_speakers,
transform_hash, calibration_source,
calibration_threshold, achieved_calibration_FMR,
n_genuine, n_impostor,
R1, R5, MRR, mean_rank, median_rank,
ROC_AUC, EER, TMR_FMR1,
genuine_mean, impostor_mean, margin
```

Every score-matrix archive must contain:

- score matrix;
- ordered query speaker IDs;
- ordered gallery speaker IDs;
- condition ID;
- split seed;
- transform hash;
- score orientation (`higher_is_better=true`).

`gate_report.md` must label every planned experiment:

- `PASS`;
- `FAIL`;
- `BLOCKED`, with exact missing prerequisite;
- `NOT RUN`, with a predeclared reason.

Maximum four main figures. Put sensitivity sweeps in appendix files rather than multiplying main-paper figures.

---

## 18. Final interpretation table the agent must complete

| Observed result | Supported wording | Wording not supported |
|---|---|---|
| Positive held-out cosine only | shared displacement direction | correct individual transformation |
| Positive held-out \(E_{\mathrm{test}}\) | direction and magnitude jointly generalize | entire residual is explained |
| Pooled centering recovers most cosine gain | original coordinate contributes strongly to raw cosine failure | geometric direction is nonexistent |
| Correction helps Euclidean/Mahalanobis | not only a cosine-origin artifact | solved cross-modal recognition |
| Strong LDA absorbs correction gain | simple frozen-cosine geometry is the main functional setting | global direction was meaningless |
| Strong LDA still improves after correction | correction is robust to a classic supervised backend | universally backend-independent |
| Analogy ordering/AUC passes | analogy-like shared relation | identical phonological arithmetic mechanism |
| Linear mode AUC falls to chance | dominant linear centroid-level mode separation is suppressed | all mode information disappears |
| Monotonic S1 dose response | causal steering of measured singing-related attributes | complete speech-to-singing conversion |
| Pro/amateur category direction only | dataset-level proficiency-associated separation | a universal skill axis |

### Required final prose decision

The result report must end by choosing exactly one of these positions:

1. **Origin-dominated functional effect:** shared residual geometry exists, but simple re-origining or a classic backend absorbs most retrieval benefit.
2. **Metric-robust translation:** direction/magnitude generalize and explicit correction improves multiple origin-invariant or supervised backends.
3. **Direction-only finding:** residuals align, but magnitude or functional robustness is weak.
4. **Mixed/model-dependent result:** no single claim applies across the three SSL families; report the boundary instead of averaging it away.

---

## 19. Stop rule

Complete R0, G1, G2, M1--M5, A1, and C1 first. Write the compact paper-facing results package and update the experiment registry.

Then stop.

Do not automatically begin synthesis, PLDA reimplementation, nonlinear probe search, a new conversion model, or professional/amateur steering. Those branches require a separate decision after the representation evidence is visible.
