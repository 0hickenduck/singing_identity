# One-on-one brief: speech–singing identity residual

## Current claim

Frozen SSL representations contain a **speaker-shared speech-to-singing displacement**. A single vector estimated from training speakers improves speech-to-singing matching for unseen speakers:

\[
\Delta_i=C_i^{\mathrm{sing}}-C_i^{\mathrm{speech}},\qquad
\mu_{\mathrm{train}}=\operatorname{mean}_{i\in\mathrm{train}}\Delta_i,
\qquad
\widehat C_i^{\mathrm{sing}}=C_i^{\mathrm{speech}}+\mu_{\mathrm{train}}.
\]

```text
shared direction  →  correct direction + length  →  better same-person matching
held-out cosine      held-out residual error         retrieval / verification
already measured     missing                         already measured
```

## Main JVS evidence

100 speakers; repeated 60/20/20 train/dev/test speaker-disjoint splits.

| Model | Held-out direction cosine | R@1: raw → corrected | EER: raw → corrected | TMR@1%: raw → corrected |
|---|---:|---:|---:|---:|
| WavLM L12 | 0.917 | 13.5% → 36.3% | 42.1% → 26.6% | 3.0% → 14.5% |
| HuBERT L6 | 0.891 | 19.5% → 60.5% | 37.1% → 18.6% | 2.0% → 30.0% |
| MERT L3 | 0.806 | 36.5% → 66.0% | 28.5% → 16.1% | 14.3% → 38.8% |

Wrong-sign and same-norm random controls do not reproduce the gains. The result also survives one-reference and matched-frame tests. GTSinger provides a same-text mechanism replication, but language is confounded with singer and therefore is not the main identity evidence.

## Minimum remaining experiments

### 1. Must run: held-out magnitude accounting

\[
E_{\mathrm{test}}=1-
\frac{\sum_{i\in\mathrm{test}}\|\Delta_i-\mu_{\mathrm{train}}\|^2}
{\sum_{i\in\mathrm{test}}\|\Delta_i\|^2},
\qquad
\rho_i=\frac{\|\Delta_i-\mu_{\mathrm{train}}\|}{\|\Delta_i\|}.
\]

**Why:** cosine tests angle only. The previously quoted 65–83% energy figure was computed on training residuals, so it cannot yet support held-out magnitude generalization. This analysis uses cached centroids and should be very fast.

### 2. Strongly recommended: layerwise curve

For every cached layer, plot held-out cosine, $E_{\mathrm{test}}$, R@1 gain, and EER change.

**Why:** shows whether the effect is stable across depth and prevents the three selected layers from looking cherry-picked. No feature extraction is needed if all hidden layers are already cached.

### 3. Only if we want a stronger claim: nonlinear mode probe

The current linear probe falls from AUC ≈ 1.0 to 0.47–0.54 after correction. This supports only:

> Dominant **linear, centroid-level** mode separability falls to chance.

It does not prove that all mode information disappears. Run an RBF-SVM/small MLP only if we want to investigate nonlinear recoverability; otherwise keep the narrower wording.

## Paper connection

[*Self-supervised Speech Models Discover Phonological Vector Arithmetic*](https://arxiv.org/abs/2602.18899) also separates direction from scale. Its main analogy cosine and normalized-offset PCS test direction; its 92%/94% figures are analogy success rates, **not cosine values**. It studies scale separately through a controlled $\lambda$ intervention. Our parallel is:

- direction: held-out residual cosine;
- magnitude: held-out $E_{\mathrm{test}}$ and $\rho_i$;
- functional consequence: retrieval, EER, and TMR improvement.

## Decisions needed today

1. Is the core claim “shared direction improves identity correspondence,” or the stronger “mode information is removed”?
2. Are held-out magnitude accounting and the layerwise curve enough to close the paper?
3. Should GTSinger remain only a same-text mechanism control?

**Recommendation:** discuss these decisions now; run the two cached analyses after the one-on-one. Do not restart mapper tuning or new feature extraction before the claim is fixed.

[Detailed experiment plan](./identity_residual_one_on_one_experiment_plan_2026-07-13.md) · [Full dossier](./identity_residual_research_dossier_2026-07-11.md)
