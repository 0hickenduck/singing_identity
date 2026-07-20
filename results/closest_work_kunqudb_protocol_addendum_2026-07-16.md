# Closest-work addendum: KunquDB (Zhou, Lin, Liu & Li, ICPR 2024)

**Date:** 2026-07-16 JST
**Source:** user-supplied PDF at
`docs/pdf/2024_Zhou_ICPR_2024_KunquDB_An_Attempt_for_Speaker_Verification_in_the_Chinese_Opera_Scenario.pdf`
(read in full; facts below are from the PDF, not the abstract).
**Role:** extends `results/identity_residual_paper_closure_2026-07-15/closest_work_protocol_matrix.md`
with a third contemporary row. The closure artifact itself is not modified. The citation-facing
summary of this paper lives in `paper_related_work_survey_2026-07-16.md` §2 (updated in parallel);
this file holds the protocol-delta table and the wording constraints.

## What the paper does (PDF facts)

- **Corpus:** KunquDB — 339 speakers, 128 h, audio-visual, from Kunqu Opera Art Canon videos;
  utterances labeled by vocal manner: **stage speech (ST, nianbai)** vs **singing (S)**;
  60,066 ST / 17,902 S utterances. Train split: 200 speakers; held-out test split with
  ST 4,177 / S 961 trial utterances (Table 2; per-speaker counts partly garbled in extraction —
  re-check against the PDF table before quoting speaker-level test counts).
- **Four trials:** Undifferentiated; ST-domain; S-domain; **Cross-domain = enroll singing,
  test stage speech**.
- **System:** ResNet34-GSP + ArcFace (256-d), pretrained on VoxBlink2 (~16k h, 110k speakers),
  fine-tuned on KunquDB train. Scoring: cosine. Metrics: EER, mDCF(P_target=0.01).
- **Adaptation methods (all trained):** DDAL — attention-based split of the feature map into
  `f_id` + `f_domain` with gradient-reversal adversarial domain classifiers; BCST — Siamese
  cosine pair loss pulling together same-speaker cross-manner utterance embeddings. Grid M0–M6.
- **Headline cross-domain EERs:** M0 (pretrained only) **28.52%**; fine-tuned M1 9.84%; best
  M6 (DDAL+BCST) **8.25%** (vs 6.2–7.5% same-manner). Authors: performance "notably declines in
  cross-domain scenarios"; DA mitigates but "room for further improvement" remains.
- **Geometry content:** one t-SNE figure (Fig. 5). For M0, same-speaker utterances do **not**
  converge across manners; instead "S utterances predominantly in the left upper quadrant and
  ST utterances in the right lower quadrant" — i.e., the embedding space clusters by vocal
  manner before identity. No quantitative geometry: no displacement vector, no PCA/eigen
  analysis, no covariance normalization, no whitening, no first-moment translation baseline,
  no frozen-SSL analysis anywhere in the paper.

## Protocol matrix row (delta vs this project)

| Field | KunquDB 2024 | This project |
|---|---|---|
| "Speech" side | **Stage speech (nianbai)** — stylized, elevated pitch (their Fig. 1 shows ST pitch well above regular speech) | Ordinary read speech (JVS parallel corpus; GTSinger speech control) |
| Cross trial | Enroll singing → test stage speech, utterance-level (2 s crops) | Speech query → singing gallery, centroid-level primary + seeded single-utterance U1 |
| Identity supervision | ArcFace on 110k VoxBlink2 speakers + 200 in-domain fine-tune speakers | None — frozen generic SSL, no speaker labels anywhere |
| Repair type | Trained: adversarial disentangling (DDAL), contrastive pairs (BCST), full fine-tuning | Closed-form train-only statistics: mean translation, top-k PC removal, OAS whitening |
| Geometry analysis | Qualitative t-SNE only | Quantified: held-out Δ_i alignment, E_d(k), A(diag), repair-equivalence, layer curves |
| Comparable numbers? | No — dataset, language/genre, trial unit, gallery, calibration all differ | — |

## Positioning for the paper

1. **Convergent phenomenon, different vocal register.** Their M0 t-SNE — same-speaker
   utterances separated by vocal manner in a *supervised, VoxBlink2-trained* embedding space —
   is the qualitative counterpart of our quantified mode displacement in frozen SSL space. Cite
   it as evidence the masking phenomenon is not an artifact of our SSL choice or pooling.
2. **Trained vs closed-form repair.** DDAL/BCST are the third trained-compensation datapoint
   (after Chowdhury's DeepCORAL/CORAL+ and their fine-tuning): trained adaptation helps
   (28.5→8.3% EER) but does not close the cross-manner gap. Our contribution is orthogonal:
   in frozen SSL space the centroid-level gap closes with train-only classical statistics, and
   we identify *which* subspace carries it.
3. **Register caveat cuts both ways.** Their "speech" is already operatic stage speech —
   acoustically closer to singing than ordinary speech (elevated pitch), yet cross-manner
   verification still degrades sharply. Do not claim their task is easier or harder than ours;
   the registers, languages, and units differ. No numerical cross-paper comparison is licensed.

**Allowed wording:**

> Zhou et al. (2024) observe on Chinese-opera data that a strong supervised speaker embedding
> clusters by vocal manner rather than identity before adaptation, and that trained adversarial
> and contrastive adaptation substantially reduces but does not eliminate the cross-manner gap.

**Forbidden without new experiments:** any EER comparison against their Table 5; any claim that
our repairs would outperform DDAL/BCST on KunquDB; any claim about ordinary-speech↔opera
transfer.

## Follow-up hooks (not for this paper)

KunquDB is public and has same-speaker ST/S material at scale (339 speakers) — a candidate
external replication corpus for the displacement/whitening analysis in future work (thesis
chapter or journal extension), subject to the stage-speech register caveat.
