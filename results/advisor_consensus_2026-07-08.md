# Advisor Consensus: Two Independent Reviews Agree

Date: 2026-07-08

Two independent advisor sessions reviewed this project. Their recommendations converge on several points. Where we have experimental evidence, it is listed.

## 1. Points where both advisors agree

### 1.1 "Pure timbre residual" is too strong a claim

Both advisors independently warned against framing the project as decomposing "pure timbre."

Advisor 1 (methodology):
> "pure timbre residual" is too strong and hard to prove. Say "identity-related information not explained by measured acoustic covariates."

Advisor 2 (execution):
> Delta_z is not pure timbre. It is a mixed representation movement caused by many factors.

Our own evidence confirms this. The prompt-mismatch accounting v2 showed the speech-to-singing residual is partly explained by F0, duration, energy, and phonation covariates. We never achieved a clean separation of timbre from other factors.

### 1.2 Do not make the project depend on SVC synthesis

Both advisors said the same thing using different words.

Advisor 1:
> Do not make the whole project depend on Seed-VC synthesis working.

Advisor 2:
> SeedVC is useful as a sanity check and demo, but not a clean scientific proof.

Our SeedVC experiments confirmed: singing prompt changes output acoustics, but the effect is closer to phonation/content-delivery than clean identity improvement. Human listeners could not reliably judge which output had better target identity.

### 1.3 Start with linear models, not MLPs

Advisor 1:
> Start with a linear or low-rank residual mapper, not an MLP. With small data, a full MLP can overfit badly.

Advisor 2 (our own experience):
> The v1 high-dimensional accounting overfitted because it used exact phone histograms to predict full 1536-d Delta_z with only 20 speakers.

We validated this the hard way: v1 showed M5 performing worse than the M0 average baseline. After switching to PCA-reduced targets and ridge regression with strong regularization (v2), all models consistently beat M0.

### 1.4 Speaker-disjoint splits are mandatory

Both advisors emphasized held-out singer evaluation. All our experiments used speaker-disjoint splits. This is not a gap.

### 1.5 Track 1 has better near-term paper potential than Track 2

Advisor 1:
> Make Track 1 the main project. Keep Track 2 as backup.

Advisor 2:
> Keep Track 2's global/local breathy geometry as the cleanest methodological next step. [But Track 1 is closer to a publishable result.]

Track 2's global breathy vector failed strict controls (analogy ≈ wrong-technique ≈ shuffled). Track 2's detection result (AUC 0.82 residualized) is positive but not sufficient alone for a paper without deeper geometry analysis.

### 1.6 Acoustic-only baselines must always be included

Both advisors require F0/energy/duration-only baselines as a "cheating detector."

Our acoustic baseline results:
- Mode classification: AUC 0.948 raw, 0.739 residualized
- Cross-mode retrieval: R@1 = 11.1% (2× chance, weaker than SSL models)
- Breathy detection: AUC 0.69 (weaker than SSL models at 0.82)

The acoustic baseline is consistently weaker than SSL representations, confirming SSL features encode information beyond simple acoustic statistics.

### 1.7 The three-level success framing is realistic

| Level | What it requires | Current status |
|---|---|---|
| Minimum: cross-mode identity analysis | Retrieval + mode probe + residualized retrieval across models/layers | 80% done; residualized retrieval is the gap |
| Medium: mapper improves retrieval | Ridge/linear mapper on ≥50 speakers, identity-disjoint | Mapper exists but only tested on 3 test speakers |
| Bonus: mapper improves SVC output | SeedVC prompt-latent mapper | SeedVC sensitivity confirmed, but identity improvement unclear |

## 2. Points where the advisors differ

### 2.1 Dataset choice

Advisor 1 recommends JVS+JVS-MuSiC (100 speakers, Japanese) or NUS-48E (12 speakers, English) for Track 1, with GTSinger for Track 2.

Advisor 2 recommends NUS-48E as primary, NHSS as backup.

We used GTSinger for everything. JVS manifests are built but features not extracted. NUS-48E not acquired.

Decision needed: JVS has 100 speakers (best for mapper generalization). NUS-48E has 12 (small but cleanly paired). GTSinger has 20 with technique labels. For Track 1 as main line, JVS is the strongest candidate.

### 2.2 Which SSL models to include

Advisor 1 includes wav2vec 2.0 and ContentVec. Advisor 2 does not emphasize these.

We extracted WavLM, HuBERT, MERT. ContentVec and wav2vec 2.0 are missing. ContentVec is particularly important because it was trained to remove speaker information, making it a useful control.

### 2.3 ECAPA-TDNN

Advisor 1 explicitly requires ECAPA. Advisor 2 does not mention it but it is implied by "speaker verification baseline."

We have not extracted ECAPA. This is a gap both advisors would flag.

## 3. What this means for the Go/No-Go decision

The core Go/No-Go criterion from Advisor 1:
> Continue Track 1 if residualized cross-mode retrieval is at least 3× chance, stable across splits, and at least one SSL model/layer clearly beats acoustic-only controls.

Current evidence:

| Criterion | Status |
|---|---|
| Raw cross-mode retrieval ≥ 3× chance | ✅ WavLM L12 R@1=27.8% ≈ 5× chance (18 speakers) |
| **Residualized** cross-mode retrieval ≥ 3× chance | ❓ **Not tested** |
| SSL beats acoustic-only controls | ✅ WavLM L12 R@1=27.8% vs acoustic 11.1% |
| SVC decoder sensitive to prompt mode | ✅ SeedVC 30-pair confirmed |

The single missing piece for a Go decision is the residualized identity retrieval experiment.
