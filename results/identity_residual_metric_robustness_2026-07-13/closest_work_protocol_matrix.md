# Closest-work protocol boundary

## Bottom line

The numerical results are not directly comparable across these studies. Mehrabani and Hansen (2012) evaluated two-speaker clustering of 10-second read/sung segments. Chowdhury, Cozzo, and Ross (2022) is the closest stated speech--singing verification work, but its five-page IEEE PDF was not accessible in this execution environment; protocol fields that require that PDF are therefore marked `PDF_REQUIRED`. The present project evaluates unseen-speaker speech-centroid queries against singing-centroid galleries in frozen SSL spaces.

## Protocol matrix

| Field | Mehrabani & Hansen 2012 | Chowdhury, Cozzo & Ross 2022 | This project |
|---|---|---|---|
| Study | *Speaker Clustering for a Mixture of Singing and Reading* | *Domain Adaptation for Speaker Recognition in Singing and Spoken Voice* | Speech--singing identity residual metric/origin audit |
| Task | Unsupervised two-speaker clustering, with the number of clusters known | `PDF_REQUIRED` | Cross-modal retrieval and verification with known speech/singing labels |
| Data | UT-Sing English subset: 33 speakers, close-talk a-cappella reading and singing of the same selected song lyrics | The paper record identifies JukeBox-V2, but exact evaluation construction is `PDF_REQUIRED` | JVS + JVS-MuSiC (100 speakers) and clean same-text GTSinger controls (20 singers) |
| Enrollment/query modality | Mixed read and sung 10-second segments; not an enrollment/query protocol | `PDF_REQUIRED` | One speech centroid per held-out speaker (20-speech-reference cap in JVS; same-text control-pair centroid in GTSinger) |
| Test/gallery modality | Mixed read and sung 10-second segments | `PDF_REQUIRED` | One singing centroid per held-out speaker; strictly cross-modal gallery |
| Identity split | 15 train speakers and 18 disjoint test speakers | `PDF_REQUIRED` | JVS: 60 train / 20 dev / 20 test, 20 fixed seeds; GTSinger: 10 train / 10 test, 50 fixed seeds |
| Trial/evaluation unit | 10-second segment; all segments from each unique pair of test speakers clustered, 153 speaker pairs | `PDF_REQUIRED` | Speaker-mode centroid; diagonal cross-modal trials genuine and all off-diagonal trials impostor |
| Backbone / embedding | 19 MFCCs; 64-mixture MAP-adapted GMM; 1,216-dimensional mean supervector | `PDF_REQUIRED` | Frozen WavLM Base+, HuBERT Base, and MERT-95M; mean+standard-deviation pooling (about 1,536 dimensions) |
| Adaptation / compensation | PLDA trained on read+sung samples of 15 speakers; PLDA LLR used to refine k-means or hierarchical clusters | CORAL/DeepCORAL/triplet/other exact objective: `PDF_REQUIRED` | Speaker-balanced train-only mean displacement; OAS whitening/Mahalanobis and shrinkage LDA audited as fixed backends |
| First-moment centering | PLDA includes a global mean parameter; no separately evaluated train mean speech-to-singing translation | `PDF_REQUIRED` | Explicitly audited: original origin, train pooled midpoint, query translation, symmetric translation, and mode centering |
| Score/backend | Euclidean k-means, Ward hierarchical clustering, and PLDA LLR cluster refinement | `PDF_REQUIRED` | Cosine, unnormalized Euclidean, OAS-whitened cosine, OAS Mahalanobis, and regularized LDA-cosine |
| Calibration | Not a threshold-calibrated verification task | `PDF_REQUIRED` | JVS operating threshold selected on dev-speaker impostors; test EER descriptive. GTSinger 1% TMR not primary due resolution |
| Main reported values | Mixture accuracy: k-means 80.6%, hierarchical 82.9%, one-pass k-means+PLDA 87.2%, one-pass hierarchical+PLDA 86.3%, hierarchical with LLR distance 85.7%, 3-pass k-means+PLDA 89.9%, 3-pass hierarchical+PLDA 89.2% | Every comparable table value: `PDF_REQUIRED` | See `baseline_reproduction.csv`, `metric_results_per_split.csv`, and `metric_paired_deltas.csv` |
| Directly numerically comparable? | No: clustering accuracy, two-speaker mixtures, segment unit, and supervised PLDA differ | No pending PDF inspection; dataset, trial unit, gallery, and calibration are already known not to be safely assumed equal | Reference protocol |

## What was already shown

Mehrabani and Hansen showed that mixing reading and singing reduced conventional clustering accuracy and that a PLDA speaker/style model trained on disjoint identities improved two-speaker mixed-style clustering. The JukeBox dataset paper separately established a large, multilingual, in-the-wild singing corpus (7,000 recordings from 936 singers, 18 languages, studio-to-live recording variation) and evaluated singing verification systems trained or pre-trained on speech.

The inaccessible Chowdhury 2022 PDF is required before stating its exact enrollment/test modalities, trial counts, feature dimensions, adaptation fitting data, centering operations, score backend, calibration source, or table values. None of those fields are reconstructed from its title, abstract record, citations, or the earlier JukeBox dataset paper.

## What is specific here

This project asks whether selected frozen generic SSL spaces contain a speaker-shared, train-estimable first-order speech-to-singing displacement. It evaluates that displacement on held-out identities, separates direction/magnitude generalization from origin-sensitive cosine retrieval, and tests whether explicit translation retains value under origin-invariant distances and a classic supervised LDA backend.

## Allowed novelty wording

> Prior work established cross-modal degradation and trained style/domain compensation. This project tests whether selected frozen SSL spaces contain a held-out speaker-shared first-order displacement and audits when that simple geometry helps.

## Invalid numerical comparisons

- Do not compare this project's 20-speaker-gallery R@1 with two-speaker clustering accuracy.
- Do not compare EER/TMR values until enrollment/test modalities, trial construction, calibration source, and identity splits are verified from the Chowdhury 2022 PDF.
- Do not treat JukeBox 2020 dataset/protocol details as a substitute for the 2022 domain-adaptation protocol.
- Do not claim the speech--singing gap is globally shallower than Chowdhury 2022 without a protocol-matched experiment.

## Primary sources inspected

- [Mehrabani & Hansen, Interspeech 2012 paper](https://www.isca-archive.org/interspeech_2012/mehrabani12b_interspeech.pdf)
- [Chowdhury, Cozzo & Ross, ICASSP 2022 DOI record](https://doi.org/10.1109/ICASSP43922.2022.9746111) — full PDF inaccessible here
- [Chowdhury, Cozzo & Ross, JukeBox dataset paper](https://www.cse.msu.edu/~rossarun/pubs/ChowdhuryCozzoRossJukeBox_INTERSPEECH2020.pdf)

