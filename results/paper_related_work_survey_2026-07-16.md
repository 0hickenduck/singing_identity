# Paper related-work survey and positioning — identity residual analysis paper

**Date:** 2026-07-16 JST
**Scope:** literature support for the F2-framed analysis paper selected by
`results/identity_residual_paper_closure_2026-07-15/gate_report.md`.
**Method:** arXiv API sweeps (`singing × speaker verification`, `singer identity`,
`anisotropy/isotropy × speech representations`, `speaker information × self-supervised × layers`),
Semantic Scholar citation graph of Chowdhury et al. 2022, targeted abstract fetches, plus the
in-repo C1 protocol matrix. Web queries executed 2026-07-16. Repo's earlier scans
(`direction_reset_and_literature_scan_2026-06-25.md`, `survey_report.md`) cover the SVC/mapper
line and general SSL background; they feed the thesis chapters, not this paper's related work.

---

## 1. What this paper is actually introducing

One sentence (F2 + modifiers, from the gate report):

> Cross-mode (speech↔singing) speaker identity is largely preserved in frozen SSL
> representations, but masked under naive cosine scoring by a shared, train-estimable mode
> displacement that lies almost entirely inside the top-8 principal components of the centroid
> covariance; translation, top-PC removal, and train-only whitening are three equivalent repairs
> of that one subspace — and the recovery survives single-utterance matching and disjoint song
> sets.

Three pillars, each with a distinct literature conversation it enters:

| Pillar | Claim | Conversation it joins |
|---|---|---|
| **P1 Finding** | Frozen generic SSL (WavLM/HuBERT/MERT) retains cross-mode identity: train-only OAS-whitened cosine reaches 72.8/93.3/89.5% centroid R@1 on JVS — near dedicated-encoder territory without any speaker training. | Cross-mode speaker verification literature says the speech↔singing gap is severe (Chowdhury 2022 cross-modal EER 42–49%; JukeBox). We show the gap is largely a *scoring-geometry* problem in frozen SSL space, not missing information. |
| **P2 Mechanism** | The masking is a low-dimensional, shared, train-estimable displacement aligned with dominant variance directions (E_d(8)≈0.999); diagonal rescaling is insufficient (A(diag) 0.06–0.23) — correlation structure is required; removal of k≤8 PCs recovers ≥82% of the whitening gain. | NLP embedding post-processing (all-but-the-top, whitening, rogue dimensions, anisotropy) — we are the speech/cross-mode instantiation and we *diagnose* rather than propose the technique. |
| **P3 Scope** | Layer-structured (L3 best under +d in all families); survives utterance-to-utterance (3.9–12.8× chance); stable across disjoint GTSinger song halves; converges with classical second-order backend theory (CORAL-family = second-moment matching). | SSL layer-wise probing + classical speaker-recognition backend compensation (NAP/WCCN/IDVC/CORAL+). |

The honest framing for the intro: **an analysis/diagnosis paper**, not a method paper. The
"method" (mean translation, PC removal, whitening) is deliberately classical; the contribution is
showing these are *the same repair* for a newly characterized cross-mode masking structure, with
predeclared gates and held-out geometry. No claim of protocol-matched superiority over any prior
system (C1 boundary), no pure-timbre-vector claim, no biometric-deployment claim.

---

## 2. Closest-work differentiation matrix

Verified against abstracts/PDFs on 2026-07-16. ★ = must hold the PDF before final wording (§5).

| Work | What it shows | How we differ | Role in paper |
|---|---|---|---|
| **Chowdhury, Cozzo & Ross 2022**, ICASSP (PDF in repo, C1 closed) | Trained DA (DeepCORAL/CORAL+ on 1D-CNN / i-x-vector PLDA) for spoken↔singing verification on JukeBox; cross-modal EER stays 42–45%; DA "does not significantly improve" cross-domain results. | We use *frozen generic* SSL, no training; controlled 20-speaker centroid protocol; we localize the gap to a shared displacement subspace and show classical second-order normalization suffices. Their CORAL-family choice is *convergent* second-moment evidence, not a competitor baseline. | Related-work anchor; the C1 protocol matrix already fixes allowed/forbidden sentences. |
| **KunquDB — Zhou, Lin, Liu & Li 2024**, ICPR (arXiv 2403.13356; **PDF read 2026-07-16**, `docs/pdf/2024_Zhou_ICPR_2024_KunquDB_*.pdf`) | 339-speaker Kunqu-opera AV corpus (67.5 h stage speech `nianbai` / 60.9 h singing `changci`; 288 speakers have both); four cosine-scored trial sets, incl. **cross-domain = singing enrollment vs stage-speech test**; ResNet34-GSP pretrained on VoxBlink2, fine-tuned on 200 KunquDB speakers with **DDAL** (attention feature-map split + gradient-reversal domain adversary) and **BCST** (same-speaker cross-manner Siamese cosine loss); cross-domain EER 28.52% (pretrained-only M0) → 8.25% (M6). | Trained adversarial/contrastive compensation — **not CORAL-family**, so it belongs to the trained-compensation sentence, not the second-moment convergence sentence. No frozen-SSL analysis, no covariance/whitening backend, no displacement geometry. Two hooks for us: their Eq. (1) `f = f_id + f_domain` *assumes* exactly the additive decomposition whose shared first moment we estimate train-only; their M0 t-SNE (Fig. 5) shows embeddings clustering by vocal manner before adaptation — a qualitative preview of our quantified mode displacement. Caveat when citing: stage speech is heightened theatrical declamation (higher Leq/F0), not conversational speech. | Related work ¶1 (trained-compensation line, newest evidence) + one mechanism-¶ echo (t-SNE mode clustering). |
| **Torres, Lattner & Richard 2023**, ISMIR (arXiv 2401.05064) | Trains dedicated **singer identity encoders** with contrastive SSL on 44.1 kHz vocals; beats speaker-verification and wav2vec2 baselines on singing; out-of-domain tests on 4 corpora. | They *train new encoders for singing*; we *analyze frozen generic* models and speech↔singing correspondence of the same person, which they do not evaluate. Their result strengthens our motivation: dedicated encoders exist, yet frozen SSL already carries the identity across modes once de-masked. | Related work; also the "dedicated speaker/singer encoder" reference point alongside ECAPA. |
| **JukeBox — Chowdhury et al. 2020**, Interspeech (arXiv 2008.03507) | In-the-wild multilingual singing speaker-recognition dataset; speech-trained models degrade heavily on singing. | Dataset/benchmark evidence of the gap; no mechanism. | Intro motivation citation. |
| **Mehrabani & Hansen 2012**, Interspeech | PLDA/LDA speaker clustering across read vs sung speech, 33 speakers. | Pre-deep-learning; clustering not retrieval; already in protocol matrix. | Related work, historical anchor. |
| **Vocal92 2023**, IEEE Access ★ | A cappella solo singing + speech dataset (cites Chowdhury 2022). | Potential additional corpus citation for the limitation/future-work sentence (independent speech+singing material). Not an analysis competitor. | Datasets/limitations footnote; check speaker counts + parallelism in PDF. |
| **Mohamed et al. 2024**, Interspeech (arXiv 2406.09200) | Cumulative Residual Variance measure; speaker and phonetic subspaces in six SSL models are largely orthogonal; isotropy correlates with probing accuracy. | Geometry of *within-mode* speaker vs phone structure; no cross-mode displacement, no retrieval masking, no repair equivalence. Closest speech-side geometry paper; cite in mechanism discussion. | Mechanism related work. |
| **van Rensburg, van Niekerk & Kamper 2026**, subm. IEEE SPL (arXiv 2603.03096) ★ | PCA on utterance-averaged SSL features: top PC encodes pitch/gender; others intensity/F2/noise; dimensions manipulable via synthesis. | Interpretability of top PCs, no verification/retrieval, no cross-mode setting, no masking claim (verified from abstract). **Supports** our mechanism: dominant variance directions carry pitch-adjacent factors — exactly where a speech→singing displacement should live. | Mechanism discussion support; verify full PDF has no hidden retrieval experiment. |
| **Wisniewski et al. 2025** (arXiv 2506.11096) | wav2vec2/HuBERT embeddings are strongly anisotropic ("high similarity between random embeddings"); DTW keyword search robust anyway. | Documents speech-SSL anisotropy but treats it as benign for their task; we show a concrete task where the anisotropic dominant subspace actively masks identity and quantify the repair. | Mechanism related work. |
| **Choi et al. 2026**, ACL Findings (arXiv 2602.18899) | Phonological vector arithmetic in S3Ms across 96 languages; mean-difference feature directions compose. | Same "shared linear direction" spirit at phone level; our A1 analogy gates were explicitly inspired by it (07-13 §11) and our wording stays "analogy-like", never "same mechanism". | Cite where the analogy/offset-consistency gates are introduced. |
| **Mu & Viswanath 2018** (1702.01417); **Su et al. 2021** (2103.15316); **Timkey & van Schijndel 2021** (2109.04404); **Ethayarajh 2019** (1909.00512); **Gao et al. 2019** (1907.12009) | ABTT top-PC removal, whitening for sentence retrieval, rogue dimensions dominating cosine, anisotropy of contextual embeddings, degeneration cone. | NLP-domain; token/sentence semantics. We import the *diagnosis vocabulary* and predeclared the parallel in the 07-15 plan ("cite, do not claim the technique as novel"). W1's diagonal-vs-full result maps onto Timkey's per-dimension story and *refutes* the pure per-dimension account for our case. | Mechanism related work, 2–3 sentences. |

**Bottom line of the matrix:** the intersection "frozen generic SSL × same-person speech↔singing ×
geometric diagnosis (displacement/PC/whitening equivalence) × predeclared held-out gates" is
empty in every sweep we ran. Nobody else occupies it as of 2026-07-16.

## 3. Scoop-risk assessment

- Chowdhury 2022 has **7 citing papers** (Semantic Scholar, 2026-07-16); none does frozen-SSL
  geometric analysis; the only speech↔singing follow-ups are datasets (Vocal92, KunquDB-adjacent)
  and style-factorization ASV.
- **KunquDB PDF verified 2026-07-16:** methods are DDAL + BCST (adversarial/contrastive,
  trained); zero overlap with frozen-SSL geometric diagnosis; confirms our lane is empty.
- arXiv sweeps for `singing + speaker verification` (25 most recent) surface only KunquDB and
  Torres as adjacent; neither analyzes frozen-SSL cross-mode geometry.
- Speech-anisotropy papers (2024–2026) stop at within-mode geometry or interpretability.
- Residual risk: a 2026 preprint could appear before submission — re-run the two arXiv sweeps
  (queries in the header) the week before submission; both take minutes via the arXiv API.

Risk level: **low**. The main defensive asset is the mechanism equivalence result (W2-a) plus the
predeclared-gate methodology, which a rushed competitor would not replicate.

## 4. Citation plan by paper section

Verified IDs unless marked (verify). ~28 references — fits a 4-page ICASSP/Interspeech layout.

**Intro (problem + finding):** Chowdhury 2022 (DOI 10.1109/ICASSP43922.2022.9746111); JukeBox
2008.03507; Mehrabani & Hansen 2012 (ISCA archive); KunquDB 2403.13356; Torres 2401.05064;
WavLM 2110.13900; HuBERT 2106.07447; MERT 2306.00107.

**Related work — SSL speaker structure & layers:** Pasad layer-wise 2107.04734 and comparative
2211.03929; SUPERB 2105.01051 (verify ID); Mohamed 2406.09200; Riera 2302.14055; Gubian
2506.10855; Lin FFN-speaker 2506.21712 (pick 2–3 of the last four by fit); Yamamoto SSL×singing
2306.12714.

**Related work — geometry & post-processing:** Mu & Viswanath 1702.01417; Su 2103.15316;
Timkey 2109.04404; Ethayarajh 1909.00512; Gao 1907.12009 (optional); Wisniewski 2506.11096;
van Rensburg 2603.03096; Choi 2602.18899 (at the analogy gates).

**Related work — classical backends/DA (the convergence paragraph):** CORAL — Sun, Feng &
Saenko AAAI 2016 (arXiv 1511.05547, verified); Deep CORAL
1607.01719; CORAL+ — Lee, Wang & Koshinaka, ICASSP 2019 (arXiv 1812.10260, verified);
NAP — Solomonoff, Campbell & Boardman 2005/2007 (verify); WCCN — Hatch, Kajarekar & Stolcke,
Interspeech 2006 (verify); IDVC — Aronowitz, ICASSP 2014 (verify); PLDA — Prince & Elder,
ICCV 2007 (verified via the KunquDB reference list, which also confirms x-vector Snyder et al.
ICASSP 2018, i-vector Dehak et al. 2010, and GMM-UBM Reynolds et al. 2000 if needed);
SUPERB 2105.01051 (verified).

**Method:** OAS shrinkage — Chen, Wiesel, Eldar & Hero (arXiv 0907.4698, IEEE TSP 2010);
ECAPA-TDNN 2005.07143 (the "dedicated speaker encoder" comparator sentence).

**Datasets:** JVS 1908.06248; JVS-MuSiC 2001.07044; GTSinger 2409.13832.

**Discussion/limitations:** NHSS 2012.00337 and NUS-48E (Duan et al. 2013, verify) as the
parallel-corpus future-work sentence; Vocal92 (IEEE Access 2023) likewise; SVCC-2025 analysis
2509.15629 only if a singer-similarity-metric sentence survives editing.

## 5. Must-obtain PDFs before final wording (`PDF_REQUIRED` discipline)

| PDF | What to extract before citing specifics |
|---|---|
| ~~KunquDB (ICPR 2024)~~ | **DONE 2026-07-16** — facts extracted into §2 row; DDAL+BCST, not CORAL-family; no geometry analysis. |
| Mohamed 2024 (Interspeech) | Whether their isotropy analysis touches style/domain shift at all; exact CRV definition for one comparative sentence. |
| van Rensburg 2026 | Confirm no retrieval/verification experiment hides in the 5 pages; get the exact top-PC/pitch phrasing we echo. |
| Choi 2026 v3 (ACL Findings) | Confirm the WavLM+Vocos decoding detail (cited in our S-branch plan) and the mean-difference construction, before any sentence that parallels their arithmetic. |
| IDVC (Aronowitz 2014), NAP (Solomonoff/Campbell), WCCN (Hatch 2006) | Exact formulations for the one-paragraph classical-backend convergence claim (CORAL+ now verified via arXiv 1812.10260; Semantic Scholar was rate-limited on 2026-07-16 — re-check these three). |
| Vocal92 (IEEE Access, open) | Speaker count, parallel speech/singing structure — one dataset sentence. |
| NUS-48E (Duan et al. 2013) | Exact reference + parallel-corpus details for the future-work sentence. |

## 6. Related-work section skeleton (4-page layout, 3 paragraphs)

1. **Cross-mode speaker identity.** Severity of the speech↔singing gap (JukeBox; Chowdhury 2022;
   KunquDB 2024); trained compensation: DA/fine-tuning helps within-mode, cross-modal stays hard
   (Chowdhury Table 1; KunquDB cross-domain EER 28.5%→8.3% only after fine-tuning plus
   adversarial DDAL and Siamese BCST — and their Eq. (1) `f = f_id + f_domain` assumes the
   additive decomposition we estimate explicitly, while their pre-adaptation t-SNE shows the
   mode-clustered geometry we quantify); dedicated singer encoders sidestep via in-domain
   training (Torres 2023; ECAPA as speech-side reference). *Slot our claim:* the gap in frozen
   generic SSL is largely scoring geometry, diagnosable and repairable train-only.
2. **Geometry of SSL/embedding spaces.** Anisotropy and dominant dimensions distort cosine (NLP
   cluster; Wisniewski speech-side; Timkey per-dimension account); post-processing repairs (ABTT,
   whitening); SSL speaker/phone subspace structure and layers (Mohamed; Pasad; van Rensburg's
   top-PC=pitch/gender). *Slot:* we connect these to a concrete cross-mode identity task, show
   the per-dimension account is insufficient (W1) and the subspace account exact (W2), and that
   the displacement, not identity, occupies the top PCs.
3. **Classical backend compensation.** NAP/WCCN/LDA projections and PLDA covariance modeling;
   IDVC's removal of dataset-shift eigendirections; CORAL/CORAL+ second-moment alignment.
   *Slot:* our translation ≡ top-PC removal ≡ whitening equivalence is the frozen-SSL,
   held-out-geometry restatement of that tradition — convergent with Chowdhury's method choice,
   which is why we treat their result as corroboration under a different protocol, never as a
   beaten baseline.

## 7. Claim boundaries the writing must keep (from gate report + critical discussion)

No pure timbre/identity/singing-vector claim; no "all mode information removed" (linear
centroid-level only); no cross-paper numerical superiority (C1); GTSinger cross-song support is
language-confounded with singer; centroid-fitted transforms, JVS single-song gallery caveat
stated wherever U1 numbers appear; the W2-x WavLM residual (+3.75 pp at k=1, X2 pending) goes in
the discussion per the supplement plan's outcome table, not the headline.
