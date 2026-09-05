# Identity-residual literature closure and terminology audit

**Date:** 2026-07-11  
**Scope:** Chowdhury direct comparison, ECAPA interpretation, speech-to-singing generation prior art, Mehrabani/Hansen historical review, and identity/style/language wording.  
**Companion dossier:** `context/identity_residual_research_dossier_2026-07-11.md`

## 1. Bottom line

The paper can move to writing, but the sharpest defensible story is narrower than “the speech–singing identity gap is shallow.”

1. **Chowdhury 2022 is a same-speaker cross-modal verification study, not a speech-versus-singing mode classifier.** A trial asks whether a spoken and a sung sample belong to the same person. Its speaker encoder/matcher is therefore conceptually closer to this project's EER/TMR evaluation than to this project's mode probe.
2. **The two studies are not numerically comparable enough to claim that this project disproves a previously deep gap.** JukeBox is multilingual, in-the-wild music audio with accompaniment, chorus, live/studio and channel variation. This project uses controlled a-cappella/studio data and mostly speaker-mode centroids. A simple correction working here does not show that the same correction would solve JukeBox-V2.
3. **The mean direction is simple, but its empirical role was not guaranteed.** The finding is not the algebra. It is that a speaker-balanced, train-identity-only first moment has the correct sign, transfers to held-out speakers, improves both retrieval and verification, defeats wrong-sign/random controls, and exposes a dominant mode-associated component in frozen SSL spaces.
4. **ECAPA can encode identity and mode at the same time.** Raw ECAPA R@1 near ceiling means same-person cross-mode matching is already strong; raw mode AUC near 1 means speech and singing remain separately decodable. These are not contradictory.
5. **End-to-end speech-to-singing work already transfers speech-derived identity, often far more effectively as a synthesis system.** It learns or imposes that transfer using task-specific encoders, scores/F0/content, and generative training. It does not answer the frozen-representation geometry question.
6. **Mehrabani and Hansen materially narrow the historical novelty boundary.** Their 2012/2013 work already treats reading and singing as style variation, evaluates held-out speakers, and uses PLDA/LDA subspaces to recover speaker clustering. The current paper cannot claim the first held-out speech/singing subspace compensation. Its remaining difference is the frozen generic SSL audit, speaker-balanced first-order displacement, and the modern control/evaluation package.
7. **GTSinger language confounding is a direct threat to the word “identity,” not a minor limitation.** A language-only oracle has 70% expected R@1 in the ten-person galleries—comparable to the corrected 60–76% results. GTSinger should be presented as a same-text mechanism replication, not independent clean identity evidence. JVS should anchor the identity-relevant claim.

## 2. What Chowdhury actually verifies

Three tasks must be kept separate:

| Task | Input/output | Corresponding evidence here |
|---|---|---|
| Mode classification | One embedding; predict `speech` versus `singing` | Linear mode probe and AUC |
| Closed-set identity retrieval | One query; rank a finite gallery of known speaker centroids | S→G/G→S R@1 |
| Speaker verification | Two samples or embeddings; score “same person” versus “different people” | Cross-mode EER and TMR@FMR=1% |

Chowdhury 2022 is the third task. It is not “put audio into a foundation model and train a small predictor to say speak/sing,” and it is not necessarily a closed-set “which of these people is it?” classifier. The model produces speaker-discriminative embeddings/scores; cross-modal genuine trials pair speech and singing from the same person, while impostor trials pair different people.

This project's verification is similar in decision semantics but not identical in protocol. Here:

- genuine trials are same-speaker speech/singing **centroid** pairs;
- impostors are all off-diagonal speaker-centroid pairs;
- EER is descriptive on held-out identities;
- the TMR@1% threshold is calibrated on training speakers and frozen for test speakers.

The exact Chowdhury 2022 enrollment/test aggregation, split composition and table values could not be independently rechecked in this closure pass because the [IEEE full text](https://doi.org/10.1109/ICASSP43922.2022.9746111) is closed and neither the authors' publication page nor an open repository currently exposes the PDF. The abstract confirms JukeBox-V2, corresponding spoken voice and domain-adapted speaker recognition. Any more detailed 2022 protocol row below is therefore marked **PDF-required**, not inferred.

### Do not conflate the 2020 and 2022 JukeBox experiments

The [2020 JukeBox paper](https://www.cse.msu.edu/~rossarun/pubs/ChowdhuryCozzoRossJukeBox_INTERSPEECH2020.pdf) does **not** contain paired speech and singing from the same test identity. It trains speaker-recognition systems on a subset of VoxCeleb2 speech and evaluates/fine-tunes them on singing identities in JukeBox. It establishes severe domain-transfer difficulty, but it is not the same-person speech-enrollment/singing-test trial used in 2022.

Exact 2020 results for the 1D-Triplet-CNN are:

| Training/evaluation domain | TMR@FMR=1% | minDCF | EER |
|---|---:|---:|---:|
| VoxCeleb speech → VoxCeleb speech | 91.23% | 1.82 | 4.09% |
| VoxCeleb speech → JukeBox singing | 24.72% | 8.35 | 26.48% |
| VoxCeleb + JukeBox fine-tuning → JukeBox singing | 29.71% | 7.91 | 24.36% |

Those values quantify speech-trained model transfer to a disjoint singing dataset; they are not a direct numeric baseline for this project's cross-modal centroid verification.

## 3. Why could Chowdhury remain poor if a global direction is so simple?

There is no paradox. At least four distinctions matter.

### 3.1 Marginal alignment is not identity-conditional alignment

A domain method can make the overall speech and singing distributions overlap while still pairing singer A's speech most closely with singer B's singing. CORAL/DeepCORAL-style objectives primarily target domain statistics; triplet learning targets speaker separation under its sampled training conditions. Neither objective guarantees that a single held-out speaker's speech-to-singing vector has the correct class-conditional direction.

The current estimator is deliberately simpler:

\[
\mu_\Delta = \frac{1}{N}\sum_i(C_{i,\mathrm{sing}}-C_{i,\mathrm{speech}}),
\]

with one speaker, one vote over the same training-identity set. Algebraically this is just the difference of the speaker-balanced singing and speech means; pairing does **not** make the global mean an identity-conditional estimator. Pairing matters for balancing the two mode centroids, evaluating same-person correspondence, and analyzing the per-speaker remainder.

The empirical result is therefore sharper and more modest: this marginal first moment happens to have the correct sign and improves class-conditional held-out correspondence. The formula is trivial; that behavioral fact is not an automatic consequence of covariance alignment.

### 3.2 First moments may be hidden or discarded by the recognition pipeline

Speaker pipelines commonly include centering, length normalization, metric learning and PLDA-like scoring. Depending on where adaptation occurs, a global offset can be partly removed, distorted or made irrelevant to the final score. A paper can therefore perform sophisticated domain adaptation without ever testing the exact frozen-space intervention used here.

The safe statement is not “Chowdhury forgot to subtract the mean.” Without the full paper, that would be speculation. The safe statement is: **their reported adaptation objective does not establish the held-out, speaker-balanced first-moment phenomenon isolated here.**

### 3.3 JukeBox is a much harsher observation model

JukeBox contains 936 singers, about 7,000 songs and 467 hours across 18 languages, with studio/live recordings, accompaniment, duets and chorus. Those factors add source separation, channel, language, genre and session mismatch on top of vocal-mode mismatch. A clean global vocal displacement can be swamped by these other axes.

JVS/JVS-MuSiC and the selected GTSinger control subset are intentionally cleaner. Their result answers: “Is a large first-order component visible under controlled conditions?” It does not answer: “Is first-order translation sufficient for real-world music verification?”

### 3.4 Centroid aggregation makes a different statistical problem

This project's primary evaluation averages many utterances/frames into speaker-mode centroids. Averaging suppresses utterance noise and makes a stable shared offset easier to observe. If Chowdhury 2022 evaluates shorter or less-aggregated samples, that is intrinsically harder; however, the exact 2022 trial aggregation remains **PDF-required** and should not be asserted until checked.

### Novelty consequence

Do **not** write:

> Prior work treated the gap as deep, but we show it is actually shallow.

Write instead:

> In controlled paired datasets and selected frozen SSL layers, a surprisingly large component of cross-mode mismatch is first-order and speaker-shared. This complements prior in-the-wild recognition work; it does not imply that first-order correction is sufficient under JukeBox-like conditions.

## 4. What ECAPA near ceiling means

For the ten-speaker same-text GTSinger sanity evaluation:

- raw ECAPA S→G R@1 is 92.8%;
- corrected R@1 is 98.6%;
- raw mode AUC is approximately 1.0;
- corrected mode AUC is approximately 0.506.

So ECAPA simultaneously supports two almost orthogonal readouts:

1. **Who is it?** Same-person speech and singing already land close enough for strong cross-mode matching.
2. **Which mode is it?** A linear probe can still almost perfectly distinguish speech from singing.

An embedding can preserve singer identity in one subspace and encode vocal mode in another—or encode them in partially shared geometry—without contradiction. “Near ceiling” also must not be generalized beyond this controlled, ten-person, centroid-based gallery. It is a sanity check that the gap is representation-dependent, not proof of universal open-set speaker verification.

The useful scientific contrast is therefore:

- ECAPA: a speaker-specialized representation already organizes cross-mode identity well;
- WavLM/HuBERT/MERT: generic frozen layers contain useful identity correspondence, but a dominant shared mode displacement obscures it under cosine matching.

## 5. Why prior end-to-end speech-to-singing systems do not subsume this result

They often solve the engineering task better, but they solve a different task.

| Work | What it learns/uses | Why it does not answer this paper's question |
|---|---|---|
| [Learning Singing From Speech](https://arxiv.org/abs/1912.10128) | Unified speech/singing synthesis with learned speaker embeddings; target speaker speech plus source singing controls | Trains a waveform generator and identity pathway; does not audit a frozen generic SSL space |
| [DurIAN-SC](https://arxiv.org/abs/2008.03009) | About 20 s target speech enrollment, a speaker-recognition d-vector, source F0/energy/alignment | Uses a task-specific speaker module and explicit source-singing controls |
| [Learn2Sing](https://arxiv.org/abs/2011.08467) | Target speech, singing-teacher data, score/lyrics, style tags and adversarial domain learning | Imposes speaker/style separation during training |
| [Learn2Sing 2.0](https://arxiv.org/abs/2203.16408) | Diffusion synthesis and mutual-information-based speaker/style separation | Evaluates generation quality, not pre-existing frozen-space geometry |

These papers remove any claim of “first speech-reference singing identity transfer.” They do not remove the narrower contribution:

> auditing what same-person speech/singing geometry already exists in frozen generic SSL representations, and showing that a train-only speaker-balanced first moment exposes that correspondence.

This is an insight paper, not a better speech-to-singing tool. ECAPA and the generative literature make that positioning clearer rather than fatal.

## 6. Full review of Mehrabani and Hansen

### 6.1 Data and task

The [2012 precursor](https://www.isca-archive.org/interspeech_2012/mehrabani12b_interspeech.pdf) and [2013 full study](https://personal.utdallas.edu/~jxh052100/Publications/JP-90-SpeechComm-Mehrabani-Hansen-SigningSpeakerCluster-Mar2013.pdf) use UT-Sing:

- 33 native-English speakers;
- each person reads and sings lyrics from five self-selected popular songs;
- a cappella recording in a sound booth with a close-talk microphone;
- disjoint 15-speaker train and 18-speaker test groups;
- ten-second segments;
- unsupervised clustering of a two-speaker mixture containing reading and singing segments.

This is not cross-modal verification and not a speech-query/singing-gallery retrieval task. It asks whether reading and singing segments from the same unseen person cluster together.

### 6.2 Representation and subspace model

The 2013 system uses 19-dimensional MFCCs, a 64-component GMM-UBM and 1,216-dimensional GMM mean supervectors. PLDA models speaker and style/session variability, and an LDA+PLDA system further improves clustering.

Across all 153 pairs of the 18 held-out test speakers, reported clustering accuracy is:

| Method | Reading only | Singing only | Mixed reading + singing |
|---|---:|---:|---:|
| K-means baseline | 99.7% | 91.3% | 80.6% |
| Hierarchical baseline | 99.9% | 92.8% | 82.9% |
| LPP | — | — | 84.7% / 88.0% |
| Iterative PLDA | — | — | 90.2% / 90.5% |
| LDA + PLDA | — | — | up to 94.5% |

The 2012 precursor reports the same core pattern, with mixed K-means/hierarchical baselines of 80.6%/82.9% improving to about 89.9%/89.2% after PLDA refinement.

### 6.3 Novelty verdict

This prior art already establishes:

- singing can be treated as a speaking-style/domain nuisance for speaker organization;
- train-speaker subspaces can compensate reading/singing variation;
- the compensation transfers to held-out speakers;
- mixed speech/singing identity organization can improve without a modern end-to-end singer encoder.

Therefore avoid:

- “first held-out speech/singing subspace compensation”;
- “prior work only trained recognizers or generators”;
- “first demonstration that mode removal improves unseen-speaker organization.”

The current project remains different in the conjunction of:

- frozen generic speech/music SSL representations;
- explicit same-person mode centroids and one-speaker-one-vote displacement;
- a parameter-free first-moment intervention;
- retrieval plus held-out verification;
- wrong-sign and same-norm-random controls;
- lexical-equal, matched-frame and single-reference checks;
- train-only mode probing;
- dominant mean versus higher-rank residual analysis;
- a negative gate showing individualized predictors do not beat the global baseline.

Mehrabani/Hansen narrows the novelty, but does not collapse it.

## 7. Identity, language, style and session audit

### 7.1 GTSinger: language is enough to mimic much of R@1

The 20 speakers in the evaluated manifest are partitioned by language as follows:

| Language | Speakers |
|---|---:|
| Chinese | 2 |
| English | 3 |
| French | 2 |
| German | 2 |
| Italian | 3 |
| Japanese | 2 |
| Korean | 3 |
| Russian | 1 |
| Spanish | 2 |

For each of all \(\binom{20}{10}=184{,}756\) possible ten-speaker test galleries, consider an oracle that knows only the query language and guesses uniformly among gallery singers with that language. If a gallery contains \(K\) represented languages, its expected accuracy is \(K/10\).

The exhaustive metadata result is:

- mean expected R@1: **70%**;
- median: **70%**;
- minimum/maximum: **40% / 90%**;
- 2.5th/97.5th percentiles: **50% / 90%**;
- mean fraction of queries that are the only test singer of their language: **43.2%**.

This is the same numerical scale as corrected SSL R@1 (60.4–76.2%). It does not prove that the models use only language, but it proves that speaker-disjoint splitting does not remove language leakage. Same-text pairing removes lexical mismatch **within a pair**; it does not separate language from singer identity.

The all-off-diagonal GTSinger impostor set also contains many cross-language negatives, so EER/TMR can be helped by language. The clean follow-up, if ever required, is within-language retrieval and language-matched impostor verification—not another mapper sweep.

Paper role:

> GTSinger is a same-text replication of the mode-associated displacement and correction mechanism, not standalone evidence of abstract speaker identity.

### 7.2 JVS: stronger identity relevance, but no cross-song claim

[JVS](https://arxiv.org/abs/1908.06248) and [JVS-MuSiC](https://arxiv.org/abs/2001.07044) use the same 100 native-Japanese professional speakers and controlled studio settings. All singers perform the common song `katatsumuri`.

This has two opposite consequences:

- **Strength:** language and song identity are shared across the gallery, so neither can by itself tell the system which person is correct. This makes JVS the stronger evidence for speaker-relevant cross-mode correspondence.
- **Limit:** because the evaluation uses one common song per singer, it cannot establish that the recovered representation generalizes to a different song by the same singer. It measures “this person's speech corresponds to this person's rendition in the controlled common-song setting,” not general singer identity across repertoire.

The papers describe consistent studio settings, which reduces recording-domain variation. It does not fully exclude speaker-specific session/channel cues, especially when a speaker's material is recorded together. Thus “recording is controlled” is defensible; “recording cannot contribute” is not.

JVS-MuSiC also contains a second singer-specific song, but that song is confounded with speaker and is not a clean cross-song control by itself.

### 7.3 Technique control is not style removal

Selecting `technique == control` removes labeled special techniques such as breathy singing. It does not remove ordinary singer-specific phrasing, habitual phonation, register choice, accent, session or channel. After subtracting the global mode mean, the remainder may still mix identity, style, language, pitch, timing, phonation and recording factors.

Therefore the safest object name is:

- **speaker-shared mode-associated displacement** for the global vector;
- **identity-relevant cross-mode correspondence** or **speaker correspondence** for the recovered matching geometry;
- **speaker-specific residual variation** for the centered remainder.

Avoid “pure identity,” “pure timbre,” “identity residual,” and “mode removed” as literal causal claims.

## 8. Direct comparison table

| Study | Scientific task | Representation/training | Evaluation identities | Data difficulty | What the result establishes |
|---|---|---|---|---|---|
| JukeBox 2020 | Speech-trained speaker model transferred to singer recognition; no same-person cross-modal pair | MFCC/LPC 1D Triplet-CNN, i-vector, x-vector; PLDA/fine-tuning | Disjoint JukeBox singer IDs | In-the-wild multilingual songs with music/chorus | Speech-trained recognition degrades severely on singing |
| Chowdhury 2022 | Same-person speech–singing speaker verification | JukeBox-V2 plus domain adaptation; full method/table recheck **PDF-required** | Held-out verification is reported; exact split/pairing details **PDF-required** | JukeBox-style multilingual and recording variability | Domain adaptation is relevant but cross-modal verification remains challenging |
| Mehrabani & Hansen 2013 | Two-speaker clustering of mixed reading/singing segments | MFCC GMM supervectors, LPP/LDA/PLDA | 18 test speakers disjoint from 15 train speakers | Clean a-cappella booth recordings, multiple songs | Subspace compensation can recover unseen-speaker grouping across style |
| Current project | Frozen-space geometric audit; retrieval and centroid verification | No encoder training; train-speaker one-vote global translation | Repeated held-out speakers | Controlled JVS/GTSinger; centroid aggregation; known confounds | A dominant first-order mode-associated component obscures existing cross-mode speaker correspondence in selected SSL layers |

The only defensible direct numeric contrast currently available is JukeBox 2020 versus this project, and even that is motivational rather than apples-to-apples. The 2022 numeric row must wait for the actual PDF.

## 9. Recommended paper positioning

### One-sentence contribution

> We audit same-person speech and singing in frozen generic SSL representations and find that, under controlled held-out-speaker evaluation, a speaker-balanced train-only mean displacement removes a dominant linearly decodable mode component and improves identity-relevant cross-mode correspondence, while substantial higher-rank residual variation remains.

### Novelty sentence

> Prior work has established singing-domain degradation, speech/singing subspace compensation and speech-conditioned singing generation; our contribution is a controlled first-order geometric characterization in frozen generic SSL spaces, together with directional, content, duration, verification and speaker-level controls.

### Dataset-role sentence

> JVS/JVS-MuSiC provides the primary identity-relevant evidence because language and song are shared across the gallery; multilingual GTSinger provides a same-text mechanism replication but is not treated as a language-independent identity benchmark.

### “So what?” sentence

> The value is explanatory rather than a replacement for ECAPA or a singing generator: it identifies why generic SSL cosine geometry can understate cross-mode speaker correspondence and supplies a mandatory first-order baseline for any future personalized mapper.

## 10. Closure status

Completed:

- separated mode classification, identity retrieval and speaker verification;
- extracted exact JukeBox 2020 values;
- fully reviewed the 2012 and 2013 Mehrabani/Hansen papers;
- checked the main speech-to-singing generation line;
- quantified the GTSinger language-only gallery baseline;
- defined a defensible identity/style/song/session wording boundary.

One literature blocker remains:

- **Chowdhury 2022 exact tables and trial construction require the five-page PDF.** The DOI page is closed, the authors' publication page links only to the DOI, and no legitimate open manuscript was located. If the PDF is supplied, the exact direct-comparison row can be completed without running any new experiment.

No new representation or generation experiment is required to begin writing. Within-language GTSinger scoring is a targeted reviewer-response analysis if the underlying embeddings/trials already exist; it should not be allowed to reopen the mapper line.
