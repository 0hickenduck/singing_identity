# Identity residual paper supplement: OAS-at-best-layer and W2-x statistical closure

**Date:** 2026-07-16 JST
**Status:** ready for an executing agent
**Relation to prior plans:** this is a **strictly additive supplement** to the completed
[paper-closure plan 2026-07-15](identity_residual_paper_closure_experiment_plan_2026-07-15.md).
It inherits, unless explicitly overridden below, all non-negotiable execution rules, the §3 fixed
data/split/model protocol, the §10 statistical protocol, the §16 assertions, and the §17 artifact
contract of the [further-experiments agent handoff 2026-07-13](identity_residual_further_experiments_agent_handoff_2026-07-13.md),
plus the additional assertions in §7 of the 07-15 plan.

**Predecessor documents (read before coding):**

1. [paper-closure experiment plan 2026-07-15](identity_residual_paper_closure_experiment_plan_2026-07-15.md)
2. [further-experiments agent handoff 2026-07-13](identity_residual_further_experiments_agent_handoff_2026-07-13.md)
3. `results/identity_residual_paper_closure_2026-07-15/gate_report.md` and the CSVs beside it
4. [EXPERIMENT_REGISTRY.md](EXPERIMENT_REGISTRY.md) — entry `identity_residual_paper_closure_2026-07-15`

---

## 0. Why this supplement exists and what it may not do

The 2026-07-15 closure run selected framing **F2 — dominant-direction masking** with U1-a and
U2-a modifiers. All gates passed. Two loose ends were found during the 2026-07-16 laptop-side
audit of the closure artifacts:

1. **G2x found that layer 3 is the best layer for +d-corrected JVS R@1 in all three families**
   (WavLM L3 0.583 vs headline L12 0.362; HuBERT L3 0.642 vs headline L6 0.605; MERT headline
   already is L3). But full OAS whitening — the transform that produces the paper's strongest raw
   numbers — was fitted at headline layers only. The main table therefore cannot answer the
   predictable reviewer question: *"is OAS at the best layer even stronger than your headline?"*
   → **X1** closes this with one cheap, predeclared run.
2. **The W2-x check is statistically unfinished for WavLM.** `w2_summary.csv` records a
   `translation_increment_at_best_k_R1` of **+3.75 pp for WavLM at its selected k\*=1**, and a
   split-level paired normal interval computed from `w2_abtt_results.csv`
   (`top_variance_pcs` rows only) is **[+0.8, +6.7] pp, 13/20 splits positive** (k=2 also positive;
   k≥8 compatible with zero). The gate report states "translation has no positive speaker-cluster
   interval after the selected removal", but no speaker-cluster bootstrap output for the abtt
   contrast is present in the compact results directory. The claim must be backed by the
   predeclared speaker-cluster protocol or amended.
   → **X2** computes the authoritative interval from the saved score matrices and writes a gate
   addendum either way.

Neither item can change the selected framing. **F2, U1-a, U2-a stay fixed under every outcome
below.** This supplement only (a) completes the main table against the layer-choice attack and
(b) makes one sentence of the gate report statistically watertight. If any result appears to
demand a framing change, stop and escalate to the human — do not improvise a new narrative.

Explicitly out of scope, regardless of outcome: new models, new datasets, new layers beyond L3,
GTSinger layer sweeps, mapper/SeedVC/steering (S0–S2), professional/amateur (K0–K1), PLDA,
nonlinear probes (N1 conditions unchanged), and any utterance-level transform refitting.

### Execution order

```text
R0'  closure-baseline reproduction (blocking; must be exact-tolerance)
X1   full OAS whitening at JVS layer 3 for WavLM and HuBERT   (blocking)
X2   speaker-cluster bootstrap closure of the W2-x contrast    (blocking)
X3   log-scale alignment figure variant                        (non-blocking, cosmetic)
---  gate_report_addendum.md + registry entry, then stop ---
```

### Runner, run root, results root

```text
scripts/probing/run_identity_residual_paper_supplement.py
tests/test_identity_residual_paper_supplement.py

run root:
/localdisk/bowen/singing_identity/runs/identity_residual_paper_supplement_2026-07-16

small results root:
results/identity_residual_paper_supplement_2026-07-16
```

Run on a `valkyrie` compute node with verified `/localdisk`, from
`/home/bowen/bowen_lab/projects/singing_identity`, via `uv run --extra probe`. Record all resolved
arguments in `experiment_card.yaml`. Do not overwrite anything under
`results/identity_residual_paper_closure_2026-07-15/` or its run root — the addendum is a new file.

### Code to reuse (do not mutate)

- `run_identity_residual_paper_closure.py::reproduce_whitened_baseline` and
  `::expected_whitened_baseline` — R0' reuses these against the 07-15 `baseline_reproduction.csv`.
- `run_identity_residual_paper_closure.py::run_g2x` — the JVS layer-3 compact-cache loaders for
  `wavlm_l3` / `hubert_l3` already exist there (G2x ran on them); lift the loading path, not a copy.
- The 07-13 metric-robustness runner's OAS fit (same estimator, eps floor
  `1e-8 · tr(Σ̂)/D`, float64, fit-audit fields, transform hashing) — X1 must go through the **same
  code path** used for the headline OAS rows, only pointed at L3 features.
- `run_identity_residual_paper_closure.py::save_score_matrix`, `::speaker_paired_interval`,
  `::write_csv` — X1/X2 outputs use the identical row schema and bundle format.

---

## 1. R0' — closure-baseline reproduction

Reproduce the six OAS-whitened JVS headline rows of
`results/identity_residual_paper_closure_2026-07-15/baseline_reproduction.csv` (which themselves
reproduced 07-13 at 0.00 pp), tolerance 0.5 pp. A mismatch requires an audit (ordered speaker
list, split membership, cache hash, dtype, transform hash), not a rerun with new code.

Additionally: re-read `w2_summary.csv` and assert the runner's recomputation of
`translation_increment_at_best_k_R1` from `w2_abtt_results.csv` reproduces the stored values
exactly (float equality within 1e-12) for all three JVS models. This proves X2 is analyzing the
same numbers the gate report summarized.

---

## 2. X1 — full OAS whitening at JVS layer 3 (WavLM, HuBERT)

### Question

Does the paper's strongest backend (train-only OAS-whitened cosine) get **better, equal, or worse**
when moved from the headline layers (WavLM L12, HuBERT L6) to the G2x-optimal layer 3? This
guards the main table against the "your headline layers were not the best layers" attack with a
number instead of a sentence.

MERT is excluded from new computation: its headline layer already is L3 — copy its closure rows
as reference, do not recompute. GTSinger is excluded: no non-headline-layer GTSinger caches exist,
the layer finding is a JVS result, and building new caches would violate the additive-only scope.

### Procedure

JVS only, 20 fixed seeds, 60/20/20 splits, identical seed list and speaker splits as always.
Per split, at layer 3 features for `wavlm_l3` and `hubert_l3`:

1. Build the speaker-balanced train matrix `Z_train` (one speech + one singing centroid per train
   speaker) **at L3**, origin `m` = pooled train mean at L3, train-only — the same construction as
   07-13 M2, same code path, new features.
2. Fit full OAS covariance; record shrinkage, trace, minimum effective eigenvalue, condition
   number, fit speaker IDs, transform hash into `fit_audit.csv`.
3. Score the four cells: `wcos_oas_none`, `wcos_oas_query` at L3, with the L3-fitted `d`.
   (Raw `cos_om_none` / `cos_o0_query` at L3 already exist in `g2x_layerwise.csv`; copy them in
   as context columns, do not recompute.)
4. Also record whitened held-out alignment `cos(WΔ_i, Wd)` and whitened `E_test` at L3.
5. Write per-row results in the 07-13 §17 schema with `condition_id` values
   `oasl3_wcos_none` / `oasl3_wcos_query`, plus score-matrix bundles.

### Predeclared decision rule

Primary contrast, per family f ∈ {WavLM, HuBERT}:

```text
D(f) = R1(wcos_oas_none @ L3) − R1(wcos_oas_none @ headline layer)
```

paired within split seed (same speaker splits on both sides), aggregated per unique test speaker,
speaker bootstrap 10,000×, two-sided sign-flip test, Holm across the two families. The headline
side comes from the reproduced R0' rows, not from re-fitted transforms.

### Outcomes and what each licenses

| Outcome | Criterion | Paper consequence |
|---|---|---|
| **X1-a: L3 stronger** | CI(D) > 0 and point D ≥ +3 pp in ≥1 family | Add an "OAS @ L3" row to the main table for that family; one sentence: "layer selection compounds with second-order normalization." Headline abstract numbers change **only** if D ≥ +5 pp with CI > 0 in **both** families — in that case stop and show the human before editing the abstract. |
| **X1-b: no stable difference** | CI(D) includes 0 in both families | One sentence in the layer paragraph: "the L3 advantage under translation correction does not transfer to the whitened backend; the headline layers remain representative." Main table unchanged. This outcome fully answers the reviewer question. |
| **X1-c: L3 weaker** | CI(D) < 0 in ≥1 family | Headline table already optimal; report D in the appendix layer table. Strengthens the current table choice. |
| **X1-x: translation adds after OAS @ L3** | `oasl3_wcos_query − oasl3_wcos_none` speaker-cluster CI > 0 | Report beside the X2 result as the same first-moment-residual phenomenon at another layer; discussion only. |

No outcome reopens G2x, adds layers 6/9, or touches GTSinger.

---

## 3. X2 — speaker-cluster closure of the W2-x contrast

### Question

Under the predeclared 07-13 §10 protocol (per-speaker aggregation → 10,000 speaker bootstrap →
percentile interval → two-sided sign-flip → Holm across the three SSL models), is
`abtt_k*_query − abtt_k*_none` positive at each model's selected k\*? This either confirms or
amends the gate-report sentence "translation has no positive speaker-cluster interval after the
selected removal."

The selected k\* per model is fixed by the closure run's `w2_summary.csv` `best_abtt_k` column:
**WavLM k\*=1, HuBERT k\*=8, MERT k\*=1.** Do not reselect k.

### Procedure

1. Load the saved closure score-matrix bundles
   `score_matrices/JVS_JVSMuSiC__<model>__<seed>__abtt_<k>_{none|query}.npz` from the 07-15 run
   root (main `top_variance_pcs` conditions only — assert the stored transform hashes match the
   `w2_abtt_results.csv` rows being closed). If any bundle is missing, recompute that condition
   from the cached centroids **only if** the recomputed transform hash equals the logged hash;
   otherwise mark X2 `BLOCKED (material)` and stop X2 — do not substitute a fresh transform fit.
2. Primary: per model at its k\*, per-speaker paired R@1 difference (query − none) aggregated over
   all test appearances of each unique speaker; 10,000-sample speaker bootstrap percentile
   interval; two-sided sign-flip p; Holm across the three models.
3. Secondary (descriptive, no gate): the same interval at every k ∈ {1, 2, 4, 8, 16}, and the
   split-level paired normal intervals — the table must reproduce the laptop-side audit values
   (WavLM k=1 mean +3.75 pp) as a cross-check.
4. Write `x2_w2x_speaker_bootstrap.csv` with one row per (model, k): mean, CI low/high, sign-flip
   p, Holm-adjusted decision, n unique speakers, bundle hashes.

### Outcomes and what each licenses

| Outcome | Criterion | Consequence |
|---|---|---|
| **X2-a: W2-x confirmed for WavLM** | Holm-corrected speaker-cluster CI > 0 at k\*=1 for WavLM | Adopt the predeclared W2-x wording in the discussion: "for WavLM a small (≈2–4 pp) first-moment residual persists after removing 1–2 dominant PCs and vanishes by k=8; second-moment repair does not fully subsume translation." Amend the gate-report sentence via `gate_report_addendum.md`. Headline and F2 unchanged — W2-a's criterion never referenced this contrast. |
| **X2-b: no positive interval** | CI includes 0 (or Holm-rejected) for all models at k\* | Keep the gate-report sentence; the addendum records the split-level trend and states why the speaker-cluster interval is authoritative (speakers, not splits, are the exchangeable unit; the 20 splits reuse the same 100 speakers). |
| **X2-c: positive for >1 model** | CI > 0 at k\* in ≥2 models | Same W2-x wording but generalized; additionally re-check that W2-a's recovery fractions were computed on `none` rows only (they were — assert it). Escalate to the human only if this coincides with X1 producing a table change, to review the mechanism paragraph once, together. |

Never edit `results/identity_residual_paper_closure_2026-07-15/gate_report.md` in place. The
addendum file is the single source of amendment, and the paper cites the pair.

---

## 4. X3 — log-scale alignment figure variant (non-blocking)

The closure figure `w2_alignment_and_abtt.png` left panel plots `E_d(k)` which saturates at ~0.999
by k=2 and reads as a flat line. Produce `figures/w2_alignment_log_inset.png` plotting
`1 − E_d(k)` on a log y-axis with the random-vector null band, same data
(`w2_spectrum_alignment.csv`), no recomputation. Keep the original figure untouched; the paper
picks one. Cosmetic; failure of this step must not block the addendum.

---

## 5. Statistical and implementation requirements (delta over 07-13 §16 and 07-15 §7)

1. X1's OAS fit at L3 must execute the identical estimator code path as the headline fit
   (same function, same eps floor, float64); the only permitted difference is the feature layer.
   Assert transform hashes differ from headline hashes and are logged.
2. Copied rows (MERT L3 reference, raw-L3 context columns, W2 summary values) must equal their
   source CSVs exactly; assert, do not re-derive.
3. X2's reconstruction of each `w2_abtt_results.csv` R@1 value from its score bundle must match
   within 1e-12 before any bootstrap is computed.
4. X1 headline-vs-L3 pairing must use identical split seeds and identical test-speaker orderings
   on both sides of every pair; assert gallery ID equality per seed.
5. All new conditions write the 07-13 §17 row schema; `condition_id` prefixes: `oasl3_*`, `x2_*`.
6. Synthetic test for X1: on data with a planted shared offset and equal per-dimension variances,
   OAS-whitened rankings at two different "layers" of identical features must coincide.
7. Synthetic test for X2: a constructed score matrix with a known per-speaker advantage must
   produce a positive speaker-cluster interval, and its speaker-permuted version must not.

### Artifact contract

```text
results/identity_residual_paper_supplement_2026-07-16/
  README_results.md
  experiment_card.yaml
  baseline_reproduction.csv            # R0' — six closure rows re-check
  fit_audit.csv                        # L3 OAS fits
  x1_oas_l3_results.csv                # per split rows + paired D(f) summary
  x1_oas_l3_summary.csv
  x2_w2x_speaker_bootstrap.csv         # primary + all-k secondary + split-level cross-check
  figures/
    x1_oas_layer_bars.png
    w2_alignment_log_inset.png
  score_matrices/<dataset>__<model>__<seed>__<condition>.npz   # symlink to run root
  gate_report_addendum.md              # X1/X2/X3 status + amended or reaffirmed sentences
```

`gate_report_addendum.md` must (a) restate the unchanged framing line (F2 + U1-a + U2-a),
(b) give each of X1/X2/X3 a PASS/FAIL/BLOCKED/NOT-RUN label with the licensed wording chosen from
the outcome tables above, and (c) end with the final main-table row list for the paper (which
rows exist, which were added by X1, which layer each uses). Update
`context/EXPERIMENT_REGISTRY.md` with one compact entry before ending the session.

---

## 6. Stop rule

Complete, in order: R0' → X1 → X2 → X3 (or skip X3 on any plotting obstacle) →
`gate_report_addendum.md` + registry entry.

Then stop. Paper writing proceeds from the closure results plus this addendum. No further
representation experiments are authorized by any outcome of this plan; S0–S2 and K0–K1 remain
gated behind a human decision exactly as before. If the executing agent believes a result
requires anything beyond the outcome tables above, it must stop and escalate rather than extend
the run.
