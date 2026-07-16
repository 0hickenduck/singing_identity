# Closest-work protocol boundary

## C1 status

**PASS.** The user supplied the five-page IEEE PDF at `docs/pdf/2022_Chowdhury_ICASSP_2022_2022_IEEE_Internat_Domain_Adaptation_for_Speaker_Recognition_in_Singing_and_Spoken_Voice.pdf` (SHA-256 `5ae8f30e687072f2cc3eaef73c2c1e0cf613e75729c67d866ceeb9bee707888e`). The matrix below distinguishes facts stated in the PDF from details the PDF does not report.

## Protocol matrix

| Field | Mehrabani & Hansen 2012 | Chowdhury, Cozzo & Ross 2022 | This project |
|---|---|---|---|
| Task | Unsupervised two-speaker clustering, number of clusters known | Spoken, singing, and cross-modal speaker verification | Cross-modal retrieval/verification plus held-out representation geometry |
| Data | UT-Sing English subset; 33 speakers | VoxCeleb2 subset plus JukeBox-V1/V2 | JVS/JVS-MuSiC (100 speakers) and clean-control GTSinger (20 singers) |
| Training material | 15 speakers with read+sung samples for PLDA | 5,994 VoxCeleb2 videos/celebrities; JukeBox-V1 training singing data for fine-tuning/DA | Speaker-balanced train centroids only; no evaluation identity enters a transform fit |
| Cross-modal identities | Not an enrollment/query protocol | 92 of 98 JukeBox-V1 test singers have spoken interviews; four 5-second spoken samples each (368 total) | JVS 20 held-out speakers per split; GTSinger 10 held-out singers per split |
| Enrollment/test modality | Mixed read/sung 10-second segments | Spoken-versus-singing; direction/enrollment convention is `NOT_REPORTED_IN_PDF` | Speech query; singing gallery |
| Identity split | 15 train, 18 disjoint test speakers | Uses named VoxCeleb2 and JukeBox train/test sets; explicit identity-overlap audit is `NOT_REPORTED_IN_PDF` | JVS 60/20/20 over 20 seeds; GTSinger 10/10 over 50 seeds |
| Trial/evaluation unit | 10-second segments from all 153 pairs of test speakers | Spoken side: 5-second samples; singing test unit and genuine/impostor counts are `NOT_REPORTED_IN_PDF` | Centroids and seeded single utterances; diagonal genuine and all off-diagonal impostor scores |
| Backbones | MFCC/GMM mean supervector + PLDA | 1D-Triplet-CNN using MFCC-LPC features (89K parameters); iVector-PLDA; xVector-PLDA (4.2M parameters) | Frozen WavLM Base+, HuBERT Base, and MERT-95M; mean+SD pooling (~1,536-D) |
| Embedding dimension | 1,216-D supervector | `NOT_REPORTED_IN_PDF` | About 1,536-D pooled SSL vectors |
| Adaptation | PLDA trained on mixed read+sung data | 1D-CNN: DeepCORAL plus separate singing/spoken cosine-triplet losses; PLDA systems: CORAL+; described as unsupervised DA | Mean translation, diagonal scaling, OAS whitening/Mahalanobis, ABTT, shrinkage LDA audit |
| Second-moment mechanism | PLDA models within/between covariance | **Yes.** DeepCORAL minimizes squared Frobenius distance between singing/spoken embedding covariances; CORAL+ adapts PLDA | OAS whitening is a fixed train-only second-order backend; ABTT removes train covariance PCs |
| First-moment centering | PLDA has a global mean parameter | No separately evaluated first-moment speech-to-singing translation is described | Explicit origin/alignment factorial and train mean displacement |
| Score backend | Euclidean/Ward clustering and PLDA LLR refinements | CNN embeddings matched by cosine; PLDA classifiers used for i/x-vector systems; further score details `NOT_REPORTED_IN_PDF` | Cosine, Euclidean, OAS-whitened cosine, OAS Mahalanobis, shrinkage LDA-cosine |
| Metrics/calibration | Clustering accuracy; no verification threshold | TMR@FMR=1% and EER; threshold/calibration source `NOT_REPORTED_IN_PDF` | R@1/MRR/EER; JVS dev-speaker calibration in the 2026-07-13 audit |
| Directly comparable? | No | No: dataset, unit, score construction, gallery, and calibration differ or are incompletely reported | Reference protocol |

## Cross-modal Table 1 values from Chowdhury et al. 2022

| Category | Model | TMR@FMR=1% | EER |
|---|---|---:|---:|
| Baseline | 1D-Triplet-CNN | 1.39% | 48.17% |
| Baseline | iVector-PLDA | 2.37% | 49.11% |
| Baseline | xVector-PLDA | 0.00% | 48.14% |
| Fine-tuned | 1D-Triplet-CNN | 0.60% | 43.02% |
| Fine-tuned | iVector-PLDA | 1.36% | 43.54% |
| Fine-tuned | xVector-PLDA | 1.20% | 44.82% |
| DA-based | 1D-Triplet-CNN | 1.67% | 42.11% |
| DA-based | iVector-PLDA | 1.73% | 44.64% |
| DA-based | xVector-PLDA | 1.58% | 42.78% |

The authors conclude that DA improves same-mode generalizability but does not significantly improve their cross-domain Table 1 results; they attribute this to insufficient cross-domain training data for learning an individual spoken-to-singing mapping.

## Positioning

Chowdhury et al. already established speech–singing cross-modal degradation and tested trained covariance-alignment compensation. Their use of DeepCORAL/CORAL+ is **convergent with the present finding that second-order normalization absorbs the naive-cosine gap**. The contributions are nevertheless different: they train/adapt speaker-recognition systems to match domain covariances, whereas this project analyzes frozen generic SSL geometry, estimates a held-out shared first-order displacement, and shows that translation, dominant-PC removal, and a fixed classical covariance backend become functionally redundant under a controlled centroid protocol.

Allowed wording:

> Prior work established severe cross-modal degradation and trained covariance-alignment compensation. This project shows that selected frozen SSL spaces retain held-out cross-mode identity, while a shared dominant covariance subspace masks that identity under naive cosine scoring.

No cross-paper numerical superiority claim is licensed.

## Sources

- User-supplied IEEE PDF (local path and hash above)
- [DOI record](https://doi.org/10.1109/ICASSP43922.2022.9746111)
- [Mehrabani & Hansen 2012](https://www.isca-archive.org/interspeech_2012/mehrabani12b_interspeech.pdf)
