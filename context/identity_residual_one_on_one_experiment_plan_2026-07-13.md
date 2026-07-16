# One-on-one: the shortest experiment plan for the identity-residual paper

**Meeting goal:** agree on the exact claim and the minimum remaining experiments.  
**Recommendation:** do **not** start a new run before the meeting. The existing result is already sufficient for a useful discussion; the main missing analysis uses cached representations and should be run after the claim is agreed.

## 1. The whole experiment in one picture

For speaker $i$, the frozen SSL model produces frame representations. We summarize each utterance by its temporal mean and standard deviation, then average utterances to obtain speech and singing centroids:

\[
z_u=[\operatorname{mean}_t X_u;\operatorname{std}_t X_u],\qquad
C_i^{\mathrm{speech}},\ C_i^{\mathrm{sing}}.
\]

The speaker's observed speech-to-singing displacement is

\[
\Delta_i=C_i^{\mathrm{sing}}-C_i^{\mathrm{speech}}.
\]

Using **training speakers only**, we estimate one shared displacement

\[
\mu_{\mathrm{train}}=\frac{1}{N_{\mathrm{train}}}\sum_{i\in\mathrm{train}}\Delta_i,
\]

and translate a held-out speaker's speech representation:

\[
\widehat C_i^{\mathrm{sing}}=C_i^{\mathrm{speech}}+\mu_{\mathrm{train}}.
\]

The logic has three separate levels:

```text
Direction                 Magnitude                    Identity consequence
Is Δ_i parallel to μ?  →  Does μ reach the target?  →  Is the same singer easier to find?
held-out cosine            held-out residual error      retrieval + verification
```

A high direction cosine alone does **not** guarantee the correct vector length, and neither geometric result alone guarantees improved speaker matching.

## 2. What we already know

JVS/JVS-MuSiC is the main identity-relevant evaluation: 100 speakers, repeated 60/20/20 train/dev/test speaker-disjoint splits. The test gallery has 20 unseen speakers.

| Model/layer | Mean held-out $\cos(\Delta_i,\mu_{\mathrm{train}})$ | Raw → corrected R@1 | Raw → corrected EER | Raw → corrected TMR@FMR=1% |
|---|---:|---:|---:|---:|
| WavLM Base+ L12 | 0.917 | 13.5% → 36.3% | 42.1% → 26.6% | 3.0% → 14.5% |
| HuBERT Base L6 | 0.891 | 19.5% → 60.5% | 37.1% → 18.6% | 2.0% → 30.0% |
| MERT 95M L3 | 0.806 | 36.5% → 66.0% | 28.5% → 16.1% | 14.3% → 38.8% |

Additional evidence already completed:

- Wrong-sign correction hurts, while same-norm random directions remain near the raw baseline.
- The effect remains with one speech reference and with equal-100-frame GTSinger crops.
- GTSinger lexical-equal pairs replicate the first-order mechanism, but language is confounded with singer, so GTSinger should not be the main independent identity claim.
- A train-only **linear** mode probe falls from AUC ≈ 1.0 to 0.47–0.54 after symmetric correction.
- Individualized mappers do not reliably beat the global correction on held-out SSL identity metrics.

### Required wording correction

The mode result supports:

> The dominant mode information at the speaker-centroid level is no longer **linearly separable** after correction.

It does **not** show that all speech/singing information disappears. A nonlinear classifier, local geometry, or temporal representation may still recover mode.

There is also an important energy-accounting correction. The $0.78\text{–}0.92$ direction result is held out, but the previously quoted $65\%\text{–}83\%$ uncentered energy figure was computed on training residuals used to estimate the mean. We should not call it held-out explained energy until the following experiment is run.

## 3. Minimum experiments to run after the meeting

| Priority | Experiment | What ambiguity it resolves | Required output | Expected cost |
|---|---|---|---|---|
| **P1 — must** | **Held-out magnitude generalization** | A cosine of 0.8–0.92 may still point in the right direction but have the wrong length. | Test $E_{\mathrm{test}}$, per-speaker normalized remaining error $\rho_i$, bootstrap interval; all six dataset/model settings. | Very short; cached centroids only. |
| **P2 — strongly recommended** | **Layerwise curve** | Are the selected layers representative, or are we showing favorable layers? How does the effect vary across model depth? | For every cached layer: held-out cosine, $E_{\mathrm{test}}$, R@1 gain, and EER change. One compact figure per model. | Short; no feature extraction if all layers are cached. |
| **P3 — conditional** | **Nonlinear mode probe** | Did correction remove only a linear mean offset, while nonlinear mode information remains? | Train/dev-selected RBF-SVM or small MLP; untouched test-speaker AUC before/after correction. | Short CPU run, but only needed if we want a claim stronger than “linear separability falls.” |
| **P4 — optional but conceptually useful** | **Split-half residual ceiling** | Is the global vector close to the reproducible within-person residual, or merely better than an impostor? | On GTSinger, compare same-person split-half residual similarity, global-vector similarity, and different-person residual similarity. | Short, but JVS cannot provide a strong version because it has one common singing item. |
| **P5 — conditional** | **Language-matched GTSinger trials** | Are GTSinger identity scores driven by language-mismatched impostors? | Same-language impostor EER/TMR and feasible within-language retrieval. | Run only if GTSinger is presented as identity evidence rather than a mechanism control. |

### P1: exact computation

Using only held-out speakers and the train-fitted vector:

\[
E_{\mathrm{test}}
=
1-
\frac{\sum_{i\in\mathrm{test}}\|\Delta_i-\mu_{\mathrm{train}}\|_2^2}
{\sum_{i\in\mathrm{test}}\|\Delta_i\|_2^2},
\qquad
\rho_i=
\frac{\|\Delta_i-\mu_{\mathrm{train}}\|_2}{\|\Delta_i\|_2}.
\]

- $E_{\mathrm{test}}>0$: the train mean predicts held-out residuals better than predicting zero displacement.
- Larger $E_{\mathrm{test}}$: direction **and length together** generalize better.
- $\rho_i=0$: perfect residual prediction; $\rho_i=1$: no improvement over zero displacement; $\rho_i>1$: the global vector makes residual reconstruction worse for that speaker.

This is the shortest experiment that directly tests our strongest geometric statement.

## 4. Relation to the phonological vector-arithmetic paper

The main item-level analogy test in [*Self-supervised Speech Models Discover Phonological Vector Arithmetic*](https://arxiv.org/abs/2602.18899) evaluates

\[
\cos\!\left(r_{p_1},\ r_{p_2}+r_{p_3}-r_{p_4}\right),
\]

so its primary analogy evidence concerns **direction**. Its PCS analysis also L2-normalizes relation offsets before cosine/AUC comparison ([implementation](https://github.com/juice500ml/phonetic-arithmetic/blob/main/pcs.py)). The paper's headline 92%/94% values are analogy **success rates**, not cosine values, and therefore should not be compared numerically with our 0.917 cosine.

That paper studies scale separately by applying $R+\lambda v$, resynthesizing audio, and measuring how acoustic properties change with $\lambda$. It shows that scale can be functionally meaningful; it does not establish that two naturally occurring analogy offsets have equal norms. This is exactly why our held-out magnitude experiment should be separate from direction cosine.

A useful parallel is:

| Question | Phonological arithmetic paper | Our paper |
|---|---|---|
| Direction | Item cosine and normalized-offset PCS | Held-out $\cos(\Delta_i,\mu_{\mathrm{train}})$ |
| Magnitude | Controlled $\lambda$ intervention | **Missing:** held-out $E_{\mathrm{test}}$ and $\rho_i$ |
| Functional consequence | Resynthesized acoustic change | Retrieval and verification improvement |

## 5. What not to start now

- No more individualized mapper tuning unless a new hypothesis predicts identity gains beyond the global baseline.
- No new feature extraction before checking which layers are already cached.
- No claim that mode information “disappears”; say that dominant **linear centroid-level** separability falls to chance.
- No claim that GTSinger proves language-independent identity.
- No cross-paper numerical comparison with [Chowdhury et al. (ICASSP 2022)](https://doi.org/10.1109/ICASSP43922.2022.9746111) without emphasizing that its in-the-wild multilingual verification setting is substantially harder and not directly matched to JVS/GTSinger.

## 6. Decisions to ask the supervisor

1. Is the paper's main claim **a shared direction that improves identity correspondence**, or do we want the stronger and riskier claim that mode information is removed?
2. Is GTSinger only a same-text mechanism replication, or must it support identity? The latter requires language-matched evaluation.
3. Is P1 + P2 enough to close the experimental package, with P3 only if we retain stronger “mode removal” wording?
4. Should the paper foreground the representation finding and negative mapper result, rather than algorithmic novelty?

## 7. A 30-second meeting version

> We found that frozen SSL speech and singing centroids differ by a highly consistent train-estimable direction. Adding that train-speaker mean displacement substantially improves retrieval and verification for unseen JVS speakers across WavLM, HuBERT, and MERT. The result is not explained by random directions, sign, centroid averaging, or unequal frame count. However, our probe only shows that linear mode separability falls, not that all mode information disappears. The main missing analysis is held-out magnitude accounting: the current cosine tests direction, while the existing energy number was training-side. I propose we run that cached analysis and a layerwise curve after agreeing today on the exact claim; nonlinear mode probing and language-matched GTSinger trials should be conditional on stronger claims.

## Project evidence

- [Full research dossier](./identity_residual_research_dossier_2026-07-11.md)
- [Main robustness report](../results/identity_residual_next_stage_report_2026-07-10.md)
- [Independent validation notes](../results/identity_residual_next_stage_validation_2026-07-10.md)
- [Paper-readiness report](../results/identity_residual_paper_readiness_2026-07-10.md)
