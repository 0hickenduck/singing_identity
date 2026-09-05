# Metric / coordinate-origin robustness: executable audit

This note audits the proposed robustness conditions and turns them into an executable, speaker-disjoint protocol. It is intended to be integrated into the larger further-experiments handoff.

## 1. What the origin problem can and cannot invalidate

For train speakers, define

\[
a=\mu_S=\operatorname{mean}_{i\in\mathrm{train}}C_i^S,
\qquad
b=\mu_G=\operatorname{mean}_{i\in\mathrm{train}}C_i^G,
\]

\[
d=b-a=\operatorname{mean}_{i\in\mathrm{train}}(C_i^G-C_i^S),
\qquad
m=\frac{a+b}{2}.
\]

A common translation by any vector \(c\) leaves all residuals unchanged:

\[
(C_i^G+c)-(C_i^S+c)=C_i^G-C_i^S.
\]

Therefore the following existing or planned geometric quantities are **origin invariant**:

- the train global residual \(d\);
- held-out \(\cos(\Delta_i,d)\), where \(\Delta_i=C_i^G-C_i^S\);
- held-out residual error \(\|\Delta_i-d\|\), \(E_{\mathrm{test}}\), and \(\rho_i\).

The origin risk applies specifically to cosine-based retrieval and verification scores. A failure of raw cosine robustness would narrow the functional claim; it would not by itself erase the shared-direction result.

## 2. Exact equivalences that must be stated before running anything

### 2.1 Mode centering is not a pure origin test

The proposed mode-centered score is

\[
\cos(C_i^S-a,C_j^G-b).
\]

It has two exact alternative forms:

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

Thus mode centering combines mode alignment with a choice of common origin. It is not an independent test of whether the origin alone explains the result. The pure origin comparison must apply the **same common transform to both modes** and compare raw versus corrected within that transform.

### 2.2 Distance backends collapse several proposed conditions

For Euclidean distance,

\[
\|(C_i^S+d)-C_j^G\|
=
\|(C_i^S-a)-(C_j^G-b)\|.
\]

The same is true for Mahalanobis distance. Query-only correction, symmetric correction, and mode centering give identical corrected distances because they differ only by a common translation.

Also,

\[
\left\|\frac{x}{\|x\|}-\frac{y}{\|y\|}\right\|^2
=2-2\cos(x,y).
\]

Consequently, Euclidean distance after L2 normalization is exactly the same ranking as cosine and must not be reported as independent evidence.

### 2.3 Whitened cosine and Mahalanobis are related but not identical

Let \(P=\Sigma_\lambda^{-1}\). Whitened cosine is

\[
s_{\mathrm{wcos}}(x,y)
=
\frac{x^\top P y}{\sqrt{x^\top Px}\sqrt{y^\top Py}},
\]

whereas the Mahalanobis score is

\[
s_M(x,y)=-(x-y)^\top P(x-y).
\]

The first discards radius in the whitened space; the second retains it. They are valid separate tests.

## 3. Required experiment matrix

All rows use identical speaker centroids, split seeds, test galleries, and train-fitted \(a,b,d,m\). `none`, `query`, and `symmetric` mean

\[
\begin{aligned}
\texttt{none}:&\quad q=C^S,\quad g=C^G,\\
\texttt{query}:&\quad q=C^S+d,\quad g=C^G,\\
\texttt{symmetric}:&\quad q=C^S+d/2,\quad g=C^G-d/2.
\end{aligned}
\]

### 3.1 Required cosine factorial

| ID | Common origin | Alignment | Score | Purpose |
|---|---|---|---|---|
| `cos_o0_none` | \(0\) | none | \(\cos(C^S,C^G)\) | exact current raw baseline |
| `cos_o0_query` | \(0\) | query | \(\cos(C^S+d,C^G)\) | exact current correction |
| `cos_o0_sym` | \(0\) | symmetric | \(\cos(C^S+d/2,C^G-d/2)\) | correction-placement diagnostic |
| `cos_om_none` | \(m\) | none | \(\cos(C^S-m,C^G-m)\) | pure train-global-origin test |
| `cos_om_query` | \(m\) | query | \(\cos(C^S+d-m,C^G-m)\) | corrected result under the same new origin |
| `cos_om_sym` | \(m\) | symmetric | \(\cos(C^S-a,C^G-b)\) | exact mode-centered condition |

This \(2\times3\) matrix separates two questions that the original list confounded:

1. Holding alignment fixed, does changing the common origin alter performance?
2. Holding the origin fixed, does train-only mode alignment improve performance?

Do not infer that mode centering is an independent alternative to \(d\); it uses exactly the same train quantities.

### 3.2 Required covariance and distance backends

| ID | Train-fitted backend | Alignment rows | Notes |
|---|---|---|---|
| `wcos_oas` | pooled-centroid OAS whitening, centered at \(m\), then cosine | none, query | primary scale/correlation robustness |
| `euclid` | negative squared Euclidean distance, no L2 normalization | none, query | origin invariant; tests whether angle is necessary |
| `mahal_oas` | negative squared OAS Mahalanobis distance | none, query | origin invariant; tests scale/correlation without radial normalization |
| `lda_cos` | regularized speaker LDA, then cosine | none, query | required supervised backend |
| `plda` | validated PLDA implementation | none, query | optional, JVS only; see limitations below |

For every backend, the raw and corrected rows must share the **same fitted transform**. Do not refit whitening/LDA/PLDA after correction. Otherwise the comparison changes both the representation and the backend.

## 4. Covariance fitting in the high-dimensional, small-sample regime

Fit covariance from the speaker-balanced raw train-centroid matrix

\[
Z_{\mathrm{train}}
=
[C_1^S,C_1^G,\ldots,C_N^S,C_N^G]^\top,
\]

with one speech and one singing centroid per train speaker. Do not use unequal numbers of utterances, and never use dev/test centroids to estimate the mean or covariance.

The empirical covariance rank is at most \(2N_{\mathrm{train}}-1\): at most 119 for JVS and 19 for GTSinger, while the pooled embedding dimension is typically much larger. Therefore:

- unregularized full whitening (\(\lambda=0\)) is invalid;
- pseudoinverse whitening is not the primary analysis because it deletes the large empirical nullspace and is highly sample dependent;
- use train-only Oracle Approximating Shrinkage (OAS) as the primary, parameter-free estimator;
- record the fitted shrinkage coefficient, trace, minimum effective eigenvalue, and condition number;
- compute in float64 and floor eigenvalues at \(10^{-8}\operatorname{tr}(\Sigma)/D\) only as a numerical guard.

The covariance definition must stay fixed across raw and corrected rows:

\[
\Sigma_\lambda=(1-\lambda)\widehat\Sigma+\lambda\tau I,
\qquad
\tau=\frac{\operatorname{tr}(\widehat\Sigma)}{D}.
\]

As a sensitivity appendix, evaluate the predeclared grid

\[
\lambda\in\{0.1,0.25,0.5,0.75,0.9,0.99,1.0\}
\]

without selecting the best test value. At \(\lambda=1\), whitened cosine must equal pooled-centered cosine, and Mahalanobis ranking must equal Euclidean ranking; implement these as unit tests.

For runtime, it is unnecessary to materialize \(\Sigma^{-1/2}\). Whitened cosine and Mahalanobis require only bilinear forms with \(P=\Sigma_\lambda^{-1}\), which can be computed through a Woodbury/dual solve using the much smaller \(2N\times2N\) train Gram matrix.

## 5. Supervised backend

### 5.1 Required: regularized LDA

Fit speaker LDA from the same two train centroids per speaker, with the two modes assigned the same speaker label. Fit on raw representations once and apply the same transform to raw and corrected test vectors. Use shrinkage LDA (`solver=eigen`, analytic Ledoit--Wolf/automatic shrinkage).

For JVS, choose the LDA output dimension from

\[
k\in\{8,16,32,\min(59,64)\}
\]

using **raw-cosine dev EER only**, with ties broken by higher raw dev MRR and then smaller \(k\). Lock that \(k\) and transform for both raw and corrected test scoring. Do not refit on train+dev; this keeps the train population identical to the existing global-vector experiment.

GTSinger has no dev split and only ten train speakers. Use fixed \(k=8\) as an explicitly exploratory result and report \(k\in\{2,4,8\}\) as a no-selection sensitivity grid. Do not select the best GTSinger value on test.

### 5.2 Optional: PLDA

PLDA should be run only with an already validated implementation and synthetic unit tests. Sixty JVS training identities with two centroid observations each are marginal; ten GTSinger identities are not credible for a PLDA conclusion. Therefore:

- PLDA is JVS-only and secondary;
- use PCA dimensions \(\{16,32,64,96\}\) and PLDA speaker ranks \(\{8,16,32\}\), restricted to valid ranks;
- select the pair using raw dev EER, then lock it for raw/corrected test scoring;
- report the full dev-selected configuration and sensitivity neighborhood;
- if utterance-level samples are used instead of two centroids, label PLDA as an additional-data supervised baseline rather than a data-matched comparison.

A negative PLDA result is inconclusive if performance is unstable across reasonable ranks or the raw PLDA backend is itself weak.

## 6. Leakage and calibration protocol

### 6.1 Fit boundaries

For every split seed, fit the following from train speakers only:

- \(a,b,d,m\);
- covariance/shrinkage;
- LDA/PLDA/PCA parameters;
- any score calibration model.

JVS uses the existing 60 train / 20 dev / 20 test split. Dev may select backend hyperparameters and operating thresholds; test may only be scored once. No selected hyperparameter may be shared across split seeds if it was selected using another seed's test results.

GTSinger uses the existing 10 train / 10 test split. Because there is no dev set, use analytic/fixed hyperparameters. Do not tune shrinkage, LDA dimension, or PLDA configuration using the ten test speakers.

### 6.2 Verification thresholds

EER and verification ROC-AUC are threshold-free test-set summaries; label EER as a descriptive held-out ROC crossing.

For JVS TMR@FMR=1%, choose a separate threshold for each condition from the 20 dev speakers after fitting the transform on the 60 train speakers. There are 380 ordered dev impostor scores. Choose the lowest threshold whose **empirical** dev FMR is no greater than 1%, record the actual achieved FMR, threshold, and number of impostor trials, then apply it unchanged to test.

Use condition-specific dev thresholds because cosine, whitened cosine, distances, and PLDA scores have different scales. Applying the raw-cosine threshold to corrected or Mahalanobis scores is not a fair system comparison.

GTSinger has only 90 ordered train impostor centroid trials and no dev set. Since one error already corresponds to 1.11%, a claimed 1% calibration effectively requires zero observed false matches and is unstable. Primary GTSinger reporting should therefore use retrieval, ROC-AUC, and descriptive EER. If an operating point is required, report train-cross-fitted TMR@FMR=10% as exploratory and retain the existing train-calibrated TMR@1% only as a legacy-comparison column clearly marked `insufficient_resolution/non-independent_calibration`.

## 7. Metrics and saved outputs

For each dataset/model/layer/split/condition save the complete dev and test score matrices, then compute:

**Retrieval**

- R@1 and R@5;
- MRR;
- mean and median rank;
- per-query genuine-minus-mean-impostor score margin.

**Verification**

- ROC-AUC;
- descriptive test EER;
- JVS TMR@dev-calibrated FMR=1%;
- mean/SD of genuine and impostor scores;
- optional \(d'=(\mu_g-\mu_i)/\sqrt{(\sigma_g^2+\sigma_i^2)/2}\).

Do not compare raw score margins numerically across cosine, Euclidean, Mahalanobis, and PLDA because their scales differ. Compare margins only within a backend; ranks, AUC, EER, and calibrated operating points can be compared across backends.

Required output columns include:

`dataset, model, layer, split_seed, condition_id, score_family, origin, alignment, covariance_method, shrinkage, lda_dim, train_speakers, dev_speakers, test_speakers, calibration_source, calibration_threshold, achieved_calibration_FMR, n_genuine, n_impostor`, followed by all metrics.

## 8. Statistical comparison

Every claim about correction must be a paired, within-backend contrast on the same split and gallery:

\[
\Delta R@1=R@1_{\mathrm{corrected}}-R@1_{\mathrm{raw}},
\qquad
\Delta EER=EER_{\mathrm{corrected}}-EER_{\mathrm{raw}}.
\]

Repeated split seeds reuse speakers and are not independent samples. Therefore do not present a naïve CI over 20 or 50 seed values as a population CI.

For query-decomposable metrics (hit@1, reciprocal rank, rank, and per-query margin):

1. aggregate each speaker's paired condition difference over all test appearances;
2. bootstrap unique speakers 10,000 times (100 JVS speakers; 20 GTSinger speakers);
3. report the paired percentile CI and a two-sided speaker-level sign-flip permutation p-value;
4. use Holm correction across the three predeclared SSL models for each primary backend/metric family.

For verification-distribution metrics (AUC, EER, TMR), report paired split deltas and sign consistency, plus a speaker-cluster Bayesian bootstrap of saved score matrices. In each of 10,000 draws, sample independent \(w_i\sim\operatorname{Exponential}(1)\) for every unique dataset speaker; genuine trial \(i\) receives weight \(w_i\), impostor trial \((i,j)\) receives \(w_iw_j\), and weighted ROC/EER/TMR are recomputed before averaging across the predeclared split seeds. Use the same weights for the paired raw and corrected conditions. This preserves speaker as the sampling unit despite repeated test appearances. State that these intervals condition on the fitted train transforms; the repeated-split distribution separately describes train-set sensitivity.

Primary robustness gate for JVS:

- in at least two of the three SSL models, corrected versus raw has a 95% speaker-cluster CI above zero for R@1 and below zero for EER under each required backend family being claimed;
- TMR and MRR must change in the expected direction, but are supporting metrics;
- no conclusion should depend on one selected shrinkage value or LDA dimension.

## 9. Interpretation rules

### Origin-dominated result

Let \(u(M)=M\) for higher-is-better metrics and \(u(M)=-M\) for lower-is-better metrics. Define

\[
G_{\mathrm{current}}=u(\texttt{cos\_o0\_query})-u(\texttt{cos\_o0\_none}),
\]

\[
G_{\mathrm{origin}}=u(\texttt{cos\_om\_none})-u(\texttt{cos\_o0\_none}),
\]

\[
G_{\mathrm{conditional}}=u(\texttt{cos\_om\_query})-u(\texttt{cos\_om\_none}).
\]

Report \(G_{\mathrm{origin}}/G_{\mathrm{current}}\) with no clipping. Call the retrieval gain `origin-dominated` only if pooled centering recovers at least 80% of the current gain and the additional corrected-versus-raw contrast under the centered origin includes zero, consistently in at least two SSL models. Otherwise describe the result as mixed or correction-robust. The 80% criterion is a declared decision rule, not a natural constant.

Do not use `cos_om_sym` alone for this decision: it already contains the full mode alignment.

### Whitening / distance result

- If corrected improves within whitened cosine, the effect survives train-estimated scale and correlation normalization.
- If corrected improves within Euclidean and Mahalanobis scoring, the effect is not only a cosine-angle artifact.
- If whitening alone makes raw performance strong but correction still adds, say that metric learning and translation are complementary.
- If a backend's raw score absorbs the gain, check whether that raw backend is at least as good as current corrected cosine before claiming that it `absorbs` the shift.

### LDA / PLDA result

- If raw LDA/PLDA matches or exceeds current corrected cosine and explicit correction adds no stable gain, the practical finding is mainly about simple frozen cosine geometry; a classic supervised speaker backend already compensates it.
- If explicit correction still improves a strong supervised backend, the representational and practical claim is stronger.
- If raw LDA/PLDA is weak or unstable, disappearance of the correction gain is inconclusive rather than evidence of successful compensation.

## 10. Mandatory implementation assertions

The runner should fail loudly unless all of the following hold within numerical tolerance:

1. `d == mean(G_train - S_train) == mean(G_train) - mean(S_train)`.
2. `cos_om_sym` vectors equal explicit mode-centered vectors.
3. Euclidean query-only corrected scores/ranks equal symmetric and mode-centered corrected distances.
4. Mahalanobis query-only corrected scores/ranks equal symmetric corrected distances.
5. L2-normalized Euclidean rankings equal cosine rankings.
6. OAS minimum eigenvalue is positive and all scores are finite.
7. At \(\lambda=1\), whitened-cosine rankings equal pooled-centered-cosine rankings and Mahalanobis rankings equal Euclidean rankings.
8. Train, dev, and test speaker-ID sets are disjoint for every seed.
9. No test vector participates in origin, covariance, supervised-backend, hyperparameter, or threshold fitting.
10. Raw and corrected rows within a backend have identical gallery IDs and transform hashes.

## 11. Recommended execution order

1. Implement and unit-test the six-cell cosine factorial. This is the fastest and most diagnostic origin audit.
2. Add OAS whitened cosine, Euclidean, and OAS Mahalanobis with the equivalence assertions.
3. Add regularized LDA on JVS; retain GTSinger as exploratory.
4. Add PLDA only if a validated implementation is already available.
5. Produce one summary table of paired raw-to-corrected deltas and one origin/alignment interaction plot per model; keep all shrinkage and LDA sensitivity curves in the appendix.
