# Identity residual paper closure: final experiment design and analysis plan

**Date:** 2026-07-15 JST
**Status:** ready for an executing agent
**Predecessor documents (read before coding):**

1. [further-experiments agent handoff 2026-07-13](identity_residual_further_experiments_agent_handoff_2026-07-13.md) — data paths, splits, seeds, code reuse map, implementation assertions. This plan **inherits all of its non-negotiable execution rules, §3 fixed data/split/model protocol, §16 assertions, and §17 artifact contract** unless explicitly overridden below.
2. [metric/origin robustness audit 2026-07-13](identity_residual_metric_origin_robustness_audit_2026-07-13.md) — algebraic identities and statistical protocol; authoritative for any calibration/covariance detail.
3. [critical discussion](speech_singing_global_displacement_critical_discussion.md) — allowed and forbidden claims.
4. [EXPERIMENT_REGISTRY.md](EXPERIMENT_REGISTRY.md) — especially `identity_residual_metric_robustness_2026-07-13`.

---

## 0. Where the project stands and what this plan is for

The 2026-07-13 metric robustness run established:

- **Not a pure cosine/origin artifact:** pooled centering recovers only 9–24% of the cosine gain; unnormalized Euclidean retains large translation gains.
- **Backend-absorbed:** train-only OAS-whitened cosine makes *raw* (untranslated) JVS R@1 reach 72.8% (WavLM L12), 93.3% (HuBERT L6), 89.5% (MERT L3), with incremental translation gain ≈ 0. Regularized LDA shows no stable increment either.
- **Geometry intact:** held-out displacement generalization (`E_test` 0.583–0.830, ρ<1 in 95–100% of splits) and GTSinger analogy gates (AUC 0.754–0.869) pass.

Consequence: the paper is **not** a method paper about a translation vector. It is an **analysis paper** whose headline is:

> Cross-mode (speech↔singing) speaker identity is largely preserved in frozen SSL representations — a train-only second-order normalization brings frozen HuBERT within a few points of a dedicated speaker encoder — but it is masked under naive cosine geometry by a shared, train-estimable mode displacement that is equivalently neutralized by explicit translation or by classical covariance normalization.

This plan closes the remaining evidence gaps for that narrative and defines, for every experiment, (a) the exact procedure, (b) every plausible outcome, and (c) which paper wording each outcome licenses. **A failed gate is a result.** Do not add models, datasets, neural metric learners, or mapper tuning to rescue any outcome.

### Execution order

```text
W1  diagonal vs full whitening decomposition        (blocking; cheapest)
W2  displacement–spectrum alignment + top-k PC removal (blocking)
G2x layerwise curves incl. diagonal whitening        (blocking; extends 07-13 G2 spec)
U1  utterance-to-utterance protocol                  (blocking)
U2  independent-content / cross-song audit + eval    (blocking if feasible, else documented as infeasible)
C1  Chowdhury 2022 protocol closure                  (blocking for related-work; unchanged from 07-13 §12)
N1  nonlinear mode probe                             (conditional; unchanged from 07-13 §13)
--- write gate_report + paper narrative selection ---
S0–S2, K0–K1                                         (gated; NOT part of this plan)
```

W1, W2 and G2x share loaders and can be one runner. U1/U2 need utterance-level score matrices and should be a second runner. Do not start S/K branches under any outcome of this plan.

### Runner, run root, results root

```text
scripts/probing/run_identity_residual_paper_closure.py
tests/test_identity_residual_paper_closure.py

run root:
/localdisk/bowen/singing_identity/runs/identity_residual_paper_closure_2026-07-15

small results root:
results/identity_residual_paper_closure_2026-07-15
```

Reuse (do not mutate): `run_identity_residual_metric_robustness.py` loaders, OAS fitting, split/seed lists, score-matrix writer, speaker-bootstrap utilities. Reproduce the 07-13 whitened-cosine headline rows exactly (R0-style check, tolerance 0.5 pp) before running anything new; a mismatch requires an audit, not a rerun with new code.

Headline configurations remain `wavlm_l12`, `hubert_l6`, `mert_l3` on JVS (60/20/20 × 20 seeds) and clean same-text GTSinger (10/10 × 50 seeds). All transforms train-only; all comparisons paired within split/backend; speaker-cluster bootstrap + sign-flip permutation + Holm across the three SSL models as in 07-13 §10.

---

## 1. W1 — diagonal vs full whitening decomposition

### Question

Is the backend absorption explained by **per-dimension scale differences alone** (a diagonal variance re-weighting), or does it require **cross-dimension correlation structure** (full covariance)?

This directly tests the working hypothesis that "the retrieval gain was caused by mismatched per-dimension scalar variances." That hypothesis is currently *unproven*: the 07-13 run only fitted full OAS covariance.

### Procedure

Fit from the same speaker-balanced train matrix `Z_train` (one speech + one singing centroid per train speaker), the same origin `m`, all train-only:

| ID | Transform | Score |
|---|---|---|
| `wcos_diag_none` | `W = diag(var(Z_train) + eps)^(-1/2)` | `cos(W(C^S − m), W(C^G − m))` |
| `wcos_diag_query` | same `W` | `cos(W(C^S + d − m), W(C^G − m))` |
| `wcos_oas_none` | full OAS (reproduce 07-13) | as 07-13 M2 |
| `wcos_oas_query` | full OAS | as 07-13 M2 |
| `cos_om_none` / `cos_o0_query` | reference rows from 07-13 | copied, not recomputed |

`eps = 1e-8 · tr(Σ̂)/D` as the numerical floor. Record per-dimension variance spectrum summary (min/median/max, ratio max/min) in `fit_audit.csv`.

Also compute for both whitenings: whitened held-out alignment `cos(WΔ_i, Wd)` and whitened `E_test` — this shows whether the shared direction survives inside the normalized space or is flattened by it.

### Predeclared decision rule

Let `R1(b)` be mean corrected-free raw R@1 under backend `b`. Define the absorption fraction

```text
A(diag) = [R1(wcos_diag_none) − R1(cos_om_none)] / [R1(wcos_oas_none) − R1(cos_om_none)]
```

computed per model/dataset, no clipping.

### Outcomes and what each licenses

| Outcome | Criterion | Paper wording licensed | Next step |
|---|---|---|---|
| **W1-a: diagonal suffices** | `A(diag) ≥ 0.8` in ≥2/3 SSL models, and diag translation increment CI includes zero | "The masking is dominated by per-dimension variance imbalance; a train-only diagonal standardization recovers cross-mode identity." Simplest narrative; per-dimension scale becomes the headline mechanism. | W2 becomes a mechanism illustration, not load-bearing. |
| **W1-b: correlations required** | `A(diag) ≤ 0.5` in ≥2/3 SSL models | "Per-dimension rescaling is insufficient; the masking lives in a correlated subspace." The displacement-direction geometry (W2) becomes the central mechanism figure. | W2 is load-bearing; run with priority. |
| **W1-c: intermediate / model-dependent** | otherwise | Report the boundary per model family; do not average it away. Frame as "partially scale, partially correlational," with per-model A(diag) table. | Both W2 views shown; narrative selects per-model. |
| **W1-x: diag whitening *hurts* raw** | `R1(wcos_diag_none) < R1(cos_om_none)` | Report as-is; strengthens the claim that structure (not scale) is what matters. Treat as W1-b for narrative. | — |

---

## 2. W2 — displacement–covariance spectrum alignment and top-k PC removal

### Question

Do translation (07-10) and whitening (07-13) neutralize the **same** subspace? Concretely: does `d` lie in the top-variance principal components of the pooled train centroid covariance, and does removing a few top PCs recover the retrieval gain?

This is the figure that unifies the two result sets. It is analogous to the "All-but-the-Top" post-processing result for word embeddings — cite that literature; do not claim the technique as novel.

### Procedure

Per split (train-only):

1. Eigendecompose the same speaker-balanced train covariance `Σ̂` used by OAS (before shrinkage): eigenvectors `v_1..v_D` sorted by descending eigenvalue.
2. **Alignment curve:** cumulative energy of the unit displacement in the top-k eigenspace, `E_d(k) = Σ_{j≤k} (v_jᵀ d̂)²` for k = 1..64, with a null band from 1,000 random unit vectors (same dimension). Also record which single PC has max |v_jᵀ d̂| and its variance rank.
3. **Top-k PC removal (`abtt_k`):** project both query and gallery centroids off the top-k eigenvectors (after centering at `m`), then score with cosine. k ∈ {1, 2, 4, 8, 16}. Conditions: `none` and `query` alignment (to check whether translation still adds anything after removal).
4. **Variance-rank control:** repeat removal with k random eigenvectors drawn from ranks 65–512 (5 draws) — removal of arbitrary directions must not reproduce the gain.

### Predeclared decision rule

Define `G_abtt(k) = R1(abtt_k_none) − R1(cos_om_none)` and compare with the whitening gain `G_w = R1(wcos_oas_none) − R1(cos_om_none)`.

### Outcomes

| Outcome | Criterion | Paper wording licensed | Consequence |
|---|---|---|---|
| **W2-a: concentrated + recoverable** | `E_d(8) ≥ 0.6` (vs null ≪) and `G_abtt(k*) ≥ 0.8·G_w` for some k* ≤ 8, random-eigenvector control near zero | "The mode displacement is aligned with the dominant variance directions; translation, top-k PC removal, and whitening are three repairs of the same low-dimensional masking subspace." This is the strongest, most unified mechanism claim. | Main mechanism figure = alignment curve + `R1 vs k` curve. |
| **W2-b: concentrated but not recoverable** | `E_d(8) ≥ 0.6` but `G_abtt ≤ 0.5·G_w` for all k ≤ 16 | "d lies in high-variance directions, but whitening's benefit is not reducible to removing them; fine-grained re-weighting matters." | Report honestly; mechanism figure shows both curves; narrative shifts toward W1's decomposition. |
| **W2-c: not concentrated** | `E_d(16)` inside null band | "The displacement is not preferentially aligned with dominant variance directions; whitening absorbs it through broad re-weighting." Translation and whitening are then *distinct* mechanisms that happen to both help. | Do not claim the unified story. This does not endanger the paper; it changes the mechanism section. |
| **W2-x: translation still adds after removal** | `abtt_k_query − abtt_k_none` CI > 0 at the best k | Report as a residual first-moment effect beyond second-moment repair — a genuinely interesting wrinkle; add to discussion, not headline. | — |

---

## 3. G2x — layerwise curves (extends 07-13 §5)

### Question

Where in model depth do (a) identity survival, (b) displacement direction strength, and (c) whitening recoverability appear? Guards against cherry-picking L12/L6/L3.

### Procedure

As specified in 07-13 §5 (layers 3/6/9/12 × three families, JVS primary), with **one change**: at every layer also run `wcos_diag_none` (diagonal whitening raw). Full OAS whitening stays headline-layers-only (it is the expensive, singularity-prone fit; the diagonal fit is cheap and stable at every layer).

Metrics per layer: raw R@1, +d R@1, diag-whitened R@1, held-out `cos(Δ_i, d)`, held-out `E_test`, raw/corrected EER.

### Outcomes

| Outcome | Wording licensed |
|---|---|
| Whitened identity peaks at late layers for WavLM/HuBERT, early for MERT (expected from existing evidence) | "Identity survival and mode masking are layer-structured and differ between speech-trained and music-trained SSL models" — standard analysis-paper figure. |
| Some layer shows high raw R@1 without any correction | Report prominently: the masking is layer-specific, and layer selection is an alternative repair. Add that layer to the headline table. |
| Headline layers are not the best layers | State it; move the best layer into the main table and keep the old ones for continuity. Not a failure. |

No gate here — G2x is descriptive. It cannot fail, only inform.

---

## 4. U1 — utterance-to-utterance protocol (de-centroiding)

### Question

The entire current evidence base is centroid-vs-centroid. Does the phenomenon (masked-then-recoverable identity) survive when **both** query and gallery are single utterances? This is the biggest remaining scope question (critical discussion §7.4) and the most likely reviewer attack.

### Procedure

JVS primary. Per split and seed:

- Query: one randomly drawn speech utterance per test speaker.
- Gallery: one randomly drawn singing utterance per test speaker.
- 25 draws per split; report mean and spread across draws; speaker-cluster bootstrap on per-speaker aggregates.
- Conditions (headline models only): `cos_om_none`, `cos_o0_query` (+d), `wcos_oas_none`, `wcos_diag_none`. All transforms remain fitted on **train centroids** (do not refit on utterances — the claim is that a centroid-fitted repair generalizes down).
- Also run the intermediate condition already partially covered on 07-10 (single speech query vs singing centroid gallery) for continuity.
- GTSinger same-text version as supporting evidence (utterance pairs already exist in the clean subset).

Chance R@1 = 5% (20-speaker gallery).

### Outcomes

| Outcome | Criterion | Wording licensed | Consequence |
|---|---|---|---|
| **U1-a: survives** | whitened utterance-level R@1 ≥ 3× chance in ≥2/3 models, CI > chance | "The recovery is not an artifact of centroid averaging; single-utterance cross-mode matching is well above chance under the centroid-fitted repair." Major strengthening — promote to headline table. | — |
| **U1-b: degrades but above chance** | between 1.5× and 3× chance | Report the centroid→utterance degradation curve honestly; claim scope = "centroid-level strongly, utterance-level moderately." | Add per-utterance-count curve (1, 2, 5, 10 references) reusing 07-10 machinery. |
| **U1-c: collapses** | ≤ 1.5× chance | "The phenomenon is a centroid-level regularity; utterance-level variability dominates single-shot matching." This is a *scope statement*, not a paper-killer — the mechanism story (W1/W2) is unaffected because it is defined on centroids. | State prominently in limitations; do not attempt utterance-level metric learning to fix it. |

---

## 5. U2 — independent-content / cross-song evaluation

### Question

Could stable *content or recording* cues (same song, same session) be doing the matching? The JVS main result is content-unmatched but every speaker sings the same song (`katatsumuri`), so song identity cannot be tested there; GTSinger has multiple songs/techniques per singer.

### Procedure

**U2.0 (audit, half a day, must precede any run):** enumerate per-singer material in GTSinger clean subset and JVS-MuSiC caches: distinct songs, sessions, technique groups per singer, utterance counts. Write `u2_material_audit.csv`. If no dataset supports a song-disjoint split with ≥10 test singers and ≥2 songs per singer, mark U2 `BLOCKED (material)` in the gate report and stop U2 — do **not** substitute a weaker improvised protocol.

**U2.1 (if feasible, GTSinger):** song-disjoint cross-mode retrieval — singing gallery built from song set A, speech queries as usual, then singing gallery from song set B; same speakers, same transforms (train-only as ever). Compare within-protocol raw / +d / whitened. Supporting: singing→singing same-person retrieval across disjoint songs (does identity in singing survive a song change at all, independent of speech).

### Outcomes

| Outcome | Wording licensed |
|---|---|
| **U2-a: stable across song sets** | "Matching is not carried by song-specific cues within the evaluated corpus." Strengthens identity interpretation. |
| **U2-b: notable drop** | "Part of the correspondence is tied to song/session-stable cues" — cite as the honest boundary; aligns with critical discussion §4.7 (style vs identity is not cleanly separable). Keep both numbers. |
| **U2-x: blocked** | Limitation paragraph: "cross-song generalization could not be evaluated with the available material" + the audit table in appendix. Acceptable for an analysis paper; flag as future work. |

---

## 6. C1 and N1 — unchanged

- **C1 (Chowdhury et al. 2022 protocol closure)** is executed exactly as specified in 07-13 §12, including the `PDF_REQUIRED` discipline and the two required artifacts (`closest_work_protocol_matrix.md`, `closest_work_metric_mapping.csv`). It remains blocking for the related-work section. One addition: the protocol matrix must now also position **whitening-as-baseline** — i.e., record whether Chowdhury's adaptation (CORAL-family) is itself a second-moment method, because if so, the 07-13 backend-absorption result is *convergent with* rather than contradicted by their findings. CORAL matches second moments; this must be stated precisely from the PDF, not from memory.
- **N1 (nonlinear mode probe)** stays conditional per 07-13 §13: run only if the paper keeps any wording stronger than "dominant linear centroid-level mode separability falls to chance."

---

## 7. Statistical and implementation requirements (delta over 07-13)

All of 07-13 §16 assertions apply. Additional assertions for this plan:

1. Diagonal whitening at `Σ = I` limit: `wcos_diag` rankings equal pooled-centered cosine rankings when all train variances are equal (synthetic test).
2. `abtt_k` with k = 0 must reproduce `cos_om_none` exactly.
3. `abtt` projection must be idempotent (applying twice = once) and remove exactly k dimensions of variance (checked on train data).
4. `E_d(D) = 1` within tolerance for the full eigenbasis.
5. U1 utterance draws are seeded and logged; the same draw indices are used across conditions within a (split, draw) pair.
6. No utterance used in a U1/U2 test trial may enter any transform fit (transforms are centroid-train-only by construction; assert speaker disjointness anyway).
7. Every new condition writes into the same per-row schema as 07-13 §17 with `condition_id` values: `wcos_diag_*`, `abtt_{k}_*`, `utt_{backend}_*`, `xsong_{backend}_*`.

### Artifact contract

```text
results/identity_residual_paper_closure_2026-07-15/
  README_results.md
  experiment_card.yaml
  baseline_reproduction.csv          # 07-13 whitened headline rows re-check
  w1_whitening_decomposition.csv     # per split/model rows + A(diag)
  w2_spectrum_alignment.csv          # E_d(k) curves + null bands
  w2_abtt_results.csv                # R1/EER vs k + random-eigvec control
  g2x_layerwise.csv
  u1_utterance_results.csv           # per draw + per-speaker aggregate
  u2_material_audit.csv
  u2_crosssong_results.csv           # if feasible
  figures/
    w1_decomposition_bars.png
    w2_alignment_and_abtt.png        # the unification figure
    g2x_layer_curves.png
    u1_centroid_to_utterance.png
  score_matrices/<dataset>__<model>__<seed>__<condition>.npz
  gate_report.md                     # PASS/FAIL/BLOCKED/NOT-RUN per experiment
```

Maximum four main figures; sensitivity to appendix. Update `context/EXPERIMENT_REGISTRY.md` with one compact entry before ending the session.

---

## 8. Final narrative selection — the writing gate

`gate_report.md` must end by selecting exactly one paper framing. The mapping from outcomes to framings:

| Framing | Trigger | Headline sentence template |
|---|---|---|
| **F1 — Scale-anisotropy diagnosis** | W1-a (diagonal suffices), any W2 | "Cross-mode identity in frozen SSL space is masked chiefly by per-dimension variance imbalance; a train-only standardization restores it to near dedicated-encoder level." |
| **F2 — Dominant-direction masking** (default expectation) | W1-b or W1-c, plus W2-a | "A shared, train-estimable mode displacement aligned with the dominant variance directions masks cross-mode identity; translation, top-PC removal, and whitening are equivalent repairs of the same subspace." |
| **F3 — Backend-absorbed geometry with open mechanism** | W2-b or W2-c | "Cross-mode identity survives frozen SSL representations and is recoverable by classical second-order normalization; the masking is not reducible to a single low-dimensional direction." |

Orthogonal scope modifiers appended to whichever framing wins:

- U1-a → add "including single-utterance matching"; U1-c → add explicit centroid-scope limitation.
- U2-a → add "robust to song identity within the evaluated corpus"; U2-b/x → limitation sentence.
- C1 → one positioning paragraph, wording constrained by 07-13 §12's allowed/forbidden conclusions.

All three framings are publishable analysis papers. **No outcome of this plan justifies reopening the mapper, SeedVC, or steering lines, and no outcome justifies delaying the paper for new representation experiments.** If any result seems to demand a fourth framing, stop and escalate to the human rather than improvising.

### Target venue and length calibration

Interspeech/ICASSP 4-page analysis paper, or SLT/ASRU if the cycle timing fits better. The thesis chapter version can additionally absorb: Track 2 breathy detection (registry `track2_breathy_detection_probe_2026-06-28`), the mapper negative result, and the SeedVC audit as secondary chapters.

---

## 9. Stop rule

Complete, in order: baseline reproduction → W1 → W2 → G2x → U1 → U2 (or BLOCKED) → C1 → (N1 if triggered) → `gate_report.md` + registry entry.

Then stop. S0–S2 (decoder/steering) and K0–K1 (professional/amateur) remain gated behind a human decision after the paper-facing report is reviewed, exactly as in the 07-13 handoff.
