# Speech--singing identity residual metric/origin gate report

## Technical summary

R0 baseline reproduction: **PASS**. G1 held-out direction-and-magnitude gate: **PASS**.
The final paper position is assigned from the continuous backend results below; no synthesis, PLDA reimplementation, nonlinear-probe search, or professional/amateur branch was started.

## Gate status

| Gate | Status | Evidence |
|---|---|---|
| R0 headline baseline | PASS | `baseline_reproduction.csv` |
| G1 held-out direction + magnitude | PASS | `heldout_magnitude_summary.csv` |
| G2 layerwise diagnostic | PASS | All 12 requested SSL layers on both datasets; `layerwise_summary.csv`. |
| M1 six-cell cosine origin audit | PASS | Exact six cells and algebraic assertions; `metric_results_per_split.csv`. |
| M2 OAS-whitened cosine | PASS | Train-only speaker-balanced dual OAS; fit hashes in `fit_audit.csv`. |
| M3 unnormalized Euclidean | PASS | Origin-invariant raw/query comparison. |
| M4 OAS Mahalanobis | PASS | Same fitted OAS transform for raw/corrected. |
| M5 regularized LDA | PASS | JVS dev-selected dimension; GTSinger fixed/sensitivity grid. |
| PLDA | NOT RUN | Predeclared: no trusted validated repository dependency and marginal two-observation regime. |
| A1 analogy-style audit | PASS | Utterance-disjoint GTSinger halves, 99% unique-speaker bootstrap, and identity-permutation null; `analogy_summary.csv`. |
| C1 closest-work boundary | BLOCKED | IEEE five-page PDF unavailable; PDF-dependent fields are `PDF_REQUIRED` in the companion matrix. |
| N1 nonlinear mode probe | NOT RUN | Conditional gate not triggered; paper wording remains linear centroid-level. |

## Held-out magnitude generalization

The train-speaker mean is evaluated against predicting zero displacement for unseen speakers. Positive E_test with a speaker-bootstrap interval above zero and a majority of speakers with rho < 1 is the predeclared joint direction-and-magnitude gate.

| Dataset | Model | E_test | 95% speaker CI | rho<1 speakers | Gate |
|---|---|---:|---:|---:|---|
| GTSinger_same_text_control | hubert_l6 | 0.747 | [0.697, 0.787] | 100.0% | PASS |
| GTSinger_same_text_control | mert_l3 | 0.583 | [0.522, 0.630] | 95.0% | PASS |
| GTSinger_same_text_control | wavlm_l12 | 0.767 | [0.731, 0.803] | 100.0% | PASS |
| JVS_JVSMuSiC | hubert_l6 | 0.786 | [0.769, 0.800] | 100.0% | PASS |
| JVS_JVSMuSiC | mert_l3 | 0.641 | [0.623, 0.658] | 100.0% | PASS |
| JVS_JVSMuSiC | wavlm_l12 | 0.830 | [0.812, 0.846] | 100.0% | PASS |

## Metric and backend robustness

Mean paired test-split changes are shown below. Negative delta EER is favorable. Score margins are not compared across backend scales.

| Dataset | Model | Backend | Delta R@1 | Delta EER |
|---|---|---|---:|---:|
| JVS_JVSMuSiC | wavlm_l12 | cosine/original origin | +0.228 | -0.155 |
| JVS_JVSMuSiC | wavlm_l12 | cosine/pooled origin | +0.290 | -0.046 |
| JVS_JVSMuSiC | wavlm_l12 | OAS-whitened cosine | -0.003 | +0.000 |
| JVS_JVSMuSiC | wavlm_l12 | Euclidean | +0.240 | -0.153 |
| JVS_JVSMuSiC | wavlm_l12 | OAS Mahalanobis | +0.000 | -0.000 |
| JVS_JVSMuSiC | wavlm_l12 | regularized LDA-cosine | -0.002 | +0.001 |
| JVS_JVSMuSiC | hubert_l6 | cosine/original origin | +0.410 | -0.185 |
| JVS_JVSMuSiC | hubert_l6 | cosine/pooled origin | +0.348 | -0.073 |
| JVS_JVSMuSiC | hubert_l6 | OAS-whitened cosine | +0.000 | +0.000 |
| JVS_JVSMuSiC | hubert_l6 | Euclidean | +0.410 | -0.197 |
| JVS_JVSMuSiC | hubert_l6 | OAS Mahalanobis | +0.000 | +0.000 |
| JVS_JVSMuSiC | hubert_l6 | regularized LDA-cosine | +0.002 | +0.003 |
| JVS_JVSMuSiC | mert_l3 | cosine/original origin | +0.295 | -0.124 |
| JVS_JVSMuSiC | mert_l3 | cosine/pooled origin | +0.335 | -0.078 |
| JVS_JVSMuSiC | mert_l3 | OAS-whitened cosine | +0.000 | +0.000 |
| JVS_JVSMuSiC | mert_l3 | Euclidean | +0.298 | -0.122 |
| JVS_JVSMuSiC | mert_l3 | OAS Mahalanobis | +0.000 | -0.000 |
| JVS_JVSMuSiC | mert_l3 | regularized LDA-cosine | +0.003 | +0.000 |
| GTSinger_same_text_control | wavlm_l12 | cosine/original origin | +0.540 | -0.166 |
| GTSinger_same_text_control | wavlm_l12 | cosine/pooled origin | +0.380 | -0.085 |
| GTSinger_same_text_control | wavlm_l12 | OAS-whitened cosine | +0.000 | +0.000 |
| GTSinger_same_text_control | wavlm_l12 | Euclidean | +0.532 | -0.165 |
| GTSinger_same_text_control | wavlm_l12 | OAS Mahalanobis | -0.002 | +0.000 |
| GTSinger_same_text_control | wavlm_l12 | regularized LDA-cosine | -0.002 | +0.002 |
| GTSinger_same_text_control | hubert_l6 | cosine/original origin | +0.564 | -0.197 |
| GTSinger_same_text_control | hubert_l6 | cosine/pooled origin | +0.272 | -0.089 |
| GTSinger_same_text_control | hubert_l6 | OAS-whitened cosine | +0.000 | +0.000 |
| GTSinger_same_text_control | hubert_l6 | Euclidean | +0.574 | -0.197 |
| GTSinger_same_text_control | hubert_l6 | OAS Mahalanobis | +0.000 | -0.001 |
| GTSinger_same_text_control | hubert_l6 | regularized LDA-cosine | +0.008 | +0.005 |
| GTSinger_same_text_control | mert_l3 | cosine/original origin | +0.330 | -0.117 |
| GTSinger_same_text_control | mert_l3 | cosine/pooled origin | +0.092 | -0.052 |
| GTSinger_same_text_control | mert_l3 | OAS-whitened cosine | +0.000 | -0.000 |
| GTSinger_same_text_control | mert_l3 | Euclidean | +0.318 | -0.118 |
| GTSinger_same_text_control | mert_l3 | OAS Mahalanobis | +0.000 | -0.000 |
| GTSinger_same_text_control | mert_l3 | regularized LDA-cosine | -0.000 | -0.003 |

## Layerwise and analogy evidence

The layerwise curve is a diagnostic against cherry-picking, not a backend-by-layer benchmark. A1 is labeled analogy-style/PCS-inspired and is not numerically compared with phonological-arithmetic headline percentages.

See `figures/layerwise_geometry.png` and `figures/analogy_ordering.png` alongside the exact CSV summaries.

## Scope and limitations

- JVS singing content is unmatched to speech and includes one common singing item; GTSinger is small and singer-language confounded.
- GTSinger TMR@FMR=1% is not treated as primary because 90 ordered train impostors cannot resolve 1% independently.
- Repeated split seeds reuse speakers. Speaker-bootstrap intervals aggregate unique held-out speakers; split distributions describe train-set sensitivity.
- The missing handoff-linked metric-audit document prevents claiming that this implementation reproduces any additional unpublished Bayesian-bootstrap convention; all implemented uncertainty is named explicitly in the CSVs.
- LDA is fit in the exact centered span of the speaker-balanced training vectors before shrinkage LDA, avoiding null dimensions without test information.

## Final paper position

**1. Origin-dominated functional effect (backend-absorbed arm).**

The shared residual direction and magnitude generalize, and explicit translation strongly improves unnormalized Euclidean matching, so the finding is not merely a cosine-origin artifact. However, pooled centering recovers only a minority of the original cosine gain and therefore does **not** meet the predeclared `origin-dominated` descriptor. Train-only OAS-whitened cosine already exceeds corrected original-cosine performance for all three JVS models, and adding the displacement changes R@1/EER by approximately zero; regularized LDA likewise shows no stable incremental gain. The functional benefit is therefore absorbed by a strong classic metric/backend even though the origin-invariant displacement geometry remains.

## Recommended next step

Review this representation gate package. Do not start S0--S2 or K0--K1 until the paper claim boundary is accepted.
