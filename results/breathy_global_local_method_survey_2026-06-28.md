# Breathy Global/Local Geometry: Method Survey And Experiment Design

Date: 2026-06-28 JST

Purpose: decide the next Track 2 experiment before running more probes. The goal is not to repeat "low-rank residual" by habit. The goal is to test whether breathy phonation in frozen audio representations is global, local, partially shared, or only decodable without stable geometry.

## Executive Verdict

The strongest next question is:

> Breathy singing is linearly detectable in frozen WavLM/HuBERT/MERT representations. Is the underlying geometry a transferable global direction, a low-rank shared subspace, or mostly local to language, singer, phone, pitch range, and model layer?

This is a defensible research question. It is stronger than simple breathy classification, because phonation classification from SSL features is already close to existing work.

The experiment should not start from "PCA finds a low-rank breathy subspace." It should start from competing hypotheses:

1. One global breathy vector.
2. A partially shared global component plus local language/singer/phone deviations.
3. No stable geometry: breathy is decodable only through local or confounded cues.

The most publishable outcome is likely the second one, not the first one.

## What The Literature Suggests

| Area | Representative sources | Standard method | What transfers to our case |
|---|---|---|---|
| Concept vectors | TCAV: https://proceedings.mlr.press/v80/kim18d.html | Learn a linear concept direction from positive vs reference activations; test stability/significance. | Use matched control intervals as negatives. A CAV/logistic direction is a baseline for "breathy is linearly decodable", not proof of a global controllable breathy vector. |
| Probe controls | Hewitt and Liang selectivity: https://arxiv.org/abs/1909.03368 | Compare real-label probe to control/random-label probes. | AUC alone is weak. We need shuffled-within-singer/language/phone controls, plus confound-only probes. |
| Representation erasure | INLP: https://arxiv.org/abs/2004.07667; amnesic probing: https://arxiv.org/abs/2006.00995; LEACE: https://papers.nips.cc/paper_files/paper/2023/hash/d066d21c619d0a78c5b557fa3291a8f4-Abstract-Conference.html | Remove linearly decodable concept directions and re-probe. | Best non-generative test for whether a direction/subspace contains the breathy information. Also test collateral damage to singer, language, phone, and acoustic variables. |
| Latent directions in CV | GANSpace: https://arxiv.org/abs/2004.02546; InterFaceGAN: https://openaccess.thecvf.com/content_CVPR_2020/papers/Shen_Interpreting_the_Latent_Space_of_GANs_for_Semantic_Face_Editing_CVPR_2020_paper.pdf | PCA or supervised hyperplane directions, validated by edits. | PCA/SVD is acceptable only as an exploratory candidate. Its credibility comes from transfer/intervention/control, not explained variance. |
| Contrastive / differential subspaces | Contrastive PCA: https://www.nature.com/articles/s41467-018-04608-8 | Find directions enriched in target data relative to background data. | Difference-SVD and cPCA are natural for breathy-control pairs, but must beat paired/shuffled/wrong-technique controls. |
| Speech SSL probing | SUPERB: https://sls.csail.mit.edu/publications/2021/JeffLai_Interspeech_2021.pdf; large-scale evaluation: https://arxiv.org/html/2404.09385v2 | Frozen upstream model plus lightweight heads, layer-wise evaluation, acoustic baselines. | Our WavLM/HuBERT/MERT frozen-probe setup is standard. We should report layer-wise and include acoustic/log-mel style baselines. |
| Phonetic subspace probing | Aspiration in HuBERT: https://www.isca-archive.org/interspeech_2023/martin23_interspeech.html and https://arxiv.org/html/2306.06232v1 | Phone-level probes, confound classes, positive/negative controls, constrained PCA dimensionality selected by controls. | This is the closest methodological precedent. For us, reduced dimensionality should be selected by positive/negative controls, not by variance or best test AUC. |
| CCA/PWCCA/CKA layer analysis | Word-level S3M CCA: https://arxiv.org/html/2307.00162v1 | Compare model layers with external linguistic/acoustic property vectors. | Useful for layer/model comparison and acoustic alignment. Not enough alone to claim a breathy concept. |
| Singing phonation classification | voice2mode: https://arxiv.org/html/2602.13928v1 | HuBERT/wav2vec layer-wise embeddings plus SVM/XGBoost; compare to spectrogram/mel/MFCC. | Direct precedent for "SSL representations classify singing phonation." Therefore our novelty cannot be classification AUC alone. |
| Singing style conversion | SVCC 2025: https://www.vc-challenge.org/ and https://arxiv.org/html/2509.15629v2 | Style conversion with naturalness, singer similarity, style similarity evaluation. | Generation is a downstream demonstration, not the main proof. SVCC already makes label-conditioned style conversion a crowded area. |
| Voice editing directions | User-driven voice latent navigation: https://arxiv.org/html/2408.17068v1 | Analyze generator Jacobians to find voice editing directions; validate with user studies and acoustic/visualized changes. | If we later do intervention, decoder-specific confounds must be separated from representation geometry. |

## Why Low-Rank/PCA Is Not Enough

Residualized difference-PCA can answer a narrow question:

```text
After removing measured covariates, what high-variance directions remain in breathy-control differences?
```

It cannot by itself answer:

```text
Is breathy a semantic/global/causal subspace?
```

Main risks:

1. PCA finds variance, not semantic purity.
2. Residualization can remove real breathiness if breathiness genuinely changes F0, energy, duration, voicing stability, or spectral tilt.
3. Linear residualization can leave nonlinear nuisance structure.
4. A low-rank component can be induced by mean pooling, layer normalization, speaker imbalance, phone imbalance, or preprocessing.
5. High AUC can come from aggregate local effects without a transferable global direction.

Therefore, low-rank methods should be treated as candidate discovery, not as evidence by themselves.

## Existing Assets: Do Not Recreate

Use these existing artifacts first:

- Full feature cache:
  `/localdisk/bowen/singing_identity/features/stage1_repaired_200_fresh_local/`
  - `acoustic_baseline/local_wave_v1`
  - `wavlm_base_plus/microsoft_wavlm_base_plus/{3,6,9,12}`
  - `hubert_base/facebook_hubert_base_ls960/{3,6,9,12}`
  - `mert_v1_95m/m_a_p_mert_v1_95m/{3,6,9,12}`

- Breathy/control and wrong-technique manifests:
  `/localdisk/bowen/singing_identity/runs/track2_breathy_technique_directions_2026-06-28/`
  - `voiced_phone_pairs_breathy_headline_with_controls_speaker_balanced.jsonl`
  - `directions.csv`
  - `metrics.json`

- Prior detection results:
  `/localdisk/bowen/singing_identity/runs/track2_breathy_detection_probe_2026-06-28/`

- Prior subspace results:
  `/localdisk/bowen/singing_identity/runs/track2_breathy_subspace_probe_2026-06-28/`

- Existing probe scripts:
  `/home/bowen/bowen_lab/projects/singing_identity/scripts/probing/run_breathy_detection_probe.py`
  `/home/bowen/bowen_lab/projects/singing_identity/scripts/probing/run_technique_subspace_probe.py`
  `/home/bowen/bowen_lab/projects/singing_identity/scripts/probing/run_technique_directions.py`

Missing or worth adding:

- A Track 2-only marimo review notebook.
- A stricter global/local geometry script with fold-wise residualization, transfer matrix, principal angles, and erasure/removal tests.
- Acoustic voice-quality features beyond the current simple F0/duration/energy/spectrum summaries: CPPS/CPP, HNR, spectral slope/tilt, H1-H2, H1-A3, aperiodicity/high-frequency noise if feasible.

## Data Reality

Current speaker-balanced manifest:

```text
3915 total interval pairs
3000 breathy-control pairs
19 singers
8 languages
30 phone categories
```

Language distribution:

```text
Korean   975
Italian  840
English  720
Spanish  570
Japanese 330
German   240
French   160
Chinese   80
```

Interpretation:

- Enough for representation probing and global/local tests.
- Enough for high-resource language-local tests in Korean, Italian, English, Spanish.
- Weak for Chinese/French standalone claims.
- Not enough to train a serious generative breathy converter from scratch.

## Recommended Experiment Stack

### E0. Acoustic And Confound Sanity Baselines

Purpose: verify the pipeline and avoid false representation claims.

Baselines:

```text
F0 mean/std/range
duration
energy/RMS
voiced ratio
spectral centroid / tilt / high-band
phone / phone family
language
singer
```

Add if feasible:

```text
CPPS / CPP
HNR
H1-H2
H1-A3
aperiodicity / noise ratio
```

Required outputs:

- acoustic-only breathy classifier
- nuisance-only classifier
- known F0/energy direction positive control

If the method cannot recover obvious F0/energy directions, a negative breathy result is not trustworthy.

### E1. Cross-Fitted Global Direction

Methods:

```text
mean difference direction
logistic / linear SVM / LDA direction
CAV-style positive-vs-control direction
```

Splits:

```text
leave-singer-out
leave-language-out
leave-phone-family-out
leave-singer-and-phone-out if enough data
```

Metrics:

```text
paired sign accuracy: score(breathy) > score(control)
ROC-AUC
PR-AUC
balanced accuracy
macro-F1
group-stratified metrics by language/singer/phone
bootstrap CI over singers, not only intervals
```

Verdict:

- If it transfers across held-out singers and phones, there is a global component.
- If it only works within seen groups, it is local or confounded.

### E2. Global Vs Local Transfer Matrix

Train directions/subspaces separately on:

```text
all data
each high-resource language
each speaker, when enough examples
each phone family
```

Evaluate every trained direction on every test group.

Report:

```text
train group x test group performance matrix
cosine similarity between direction vectors
principal angles / Grassmann distances for subspaces
cluster structure of local directions
```

This directly answers the user's question: is breathy global, language-local, singer-local, or phone-local?

### E3. Low-Rank Subspace, But With Controls

Candidate methods:

```text
difference-SVD on paired deltas
contrastive PCA: breathy vs matched control background
PLS/CCA: representation deltas aligned to acoustic breathiness proxies
```

Rank selection:

```text
K = 1, 2, 4, 8, 16, 32, 64
nested CV only
parallel/permutation analysis
positive/negative control criterion inspired by aspiration probing
```

Do not select K by test-set AUC or explained variance alone.

Required controls:

```text
random same-dimensional subspace
shuffled labels within speaker/language/phone strata
wrong-technique subspace
pseudo-control deltas among control intervals
pseudo-breathy deltas among breathy intervals
phone-only subspace
singer-only subspace
```

Required metrics:

```text
held-out delta energy capture
held-out paired ranking
fold-to-fold subspace stability
projection/removal effect on breathy probe
collateral damage to singer/language/phone probes
acoustic alignment with breathiness proxies
```

### E4. Projection / Removal Test

This is the strongest non-generative test.

Procedure:

1. Learn a direction/subspace on train folds.
2. Project held-out embeddings onto it and evaluate breathy decoding.
3. Remove it from held-out embeddings and re-probe.
4. Compare to random, wrong-technique, and nuisance subspaces.

Expected pattern for a meaningful breathy subspace:

```text
projection retains breathy information
removal reduces breathy decodability
removal does not heavily destroy singer/language/phone information
random/wrong-technique removal has smaller effect
```

### E5. Optional Intervention, Not Main Proof Yet

Generation should be delayed unless E1-E4 show stable geometry.

Reason:

- If generated audio changes, the decoder may be responsible.
- If generated audio does not change, the representation geometry may still be real but not causally accessible through that decoder.
- SVCC 2025 already makes label-conditioned singing style conversion a crowded baseline, so generation is not automatically novel.

If used later, generation should be framed as a downstream demo:

```text
global direction steering
language-local direction steering
singer-local direction steering
compare acoustic breathiness movement and human listening
```

## Publishable Vs Weak Evidence

Publishable evidence:

- strict train/test separation
- residualization fitted inside train folds only
- held-out singer and held-out phone evaluation
- global and local models compared directly
- stable subspace geometry across folds
- negative controls fail as expected
- acoustic grounding beyond F0/duration/energy
- erasure/removal selectively reduces breathy information
- uncertainty reported over singers/languages, not only intervals

Weak evidence:

- PCA plots
- explained variance alone
- AUC alone
- residualization before splitting
- cherry-picked model layers
- no wrong-technique/shuffled/random controls
- no group-stratified metrics
- no acoustic voice-quality grounding

## Revised Main Claim Template

Strong possible claim:

> Breathy phonation is linearly detectable in frozen speech/music representations, but it is not well described by one universal vector. Instead, a small shared component transfers across singers and phones, while most variation is language-, singer-, phone-, or pitch-conditioned.

If global transfer fails:

> Breathy phonation is decodable but geometrically local: global probes overstate the result because they pool language/singer/phone-specific cues.

If global transfer succeeds:

> A stable breathy component exists across singers/languages/phones, and removal/projection tests show that this component carries breathy information beyond measured acoustic nuisance variables.

## Immediate Next Implementation

Do not rebuild features or manifests.

Implement one new strict script:

```text
scripts/probing/run_breathy_global_local_geometry.py
```

The script should load:

```text
/localdisk/bowen/singing_identity/runs/track2_breathy_technique_directions_2026-06-28/voiced_phone_pairs_breathy_headline_with_controls_speaker_balanced.jsonl
/localdisk/bowen/singing_identity/features/stage1_repaired_200_fresh_local/
```

It should produce:

```text
global_direction_metrics.json
local_transfer_matrix.csv
subspace_stability.csv
projection_removal_metrics.json
group_stratified_metrics.csv
method_config.json
```

Then add a marimo review notebook:

```text
notebooks/track2_breathy_global_local_review.py
```

This notebook should show:

- data coverage by language/singer/phone
- global vs local transfer heatmap
- rank curve with controls
- principal angle/stability plots
- projection/removal results
- concise verdict text in Chinese for human check

