Act as my research advisor. I have limited compute, a few GPUs, and I want a
master's-level project in speech/singing voice research. I care about
representation learning, timbre shift, probing/steering, singing voice
conversion, and possibly singing skill evaluation. I need a direction that is
novel, feasible, and educational.

Candidate directions:

Same-person speech-vs-singing timbre shift:
frozen SSL/audio-codec leakage map, stable identity core, mode residual, and
downstream Seed-VC intervention.
Singing style or technique direction discovery:
find and possibly steer directions for vibrato, breathiness, falsetto,
glissando, etc.
Objective evaluation metrics for singing style conversion:
correlate automatic metrics/probes with SVCC-style subjective goals.
Cross-lingual timbre leakage or stress test:
test speaker/language entanglement, but only if data supports the claim.
Singing quality/skill/technique evaluator:
MOS, reference-conditioned karaoke scoring, technique recognition, or
multi-dimensional feedback.
Robustness to separated-vocal artifacts and noisy web singing data:
source-separation/noise/F0 perturbation stress tests for SVC or evaluators.
Please:

Rank these directions by novelty, feasibility, data availability, code
availability, publishability, and fit to the 2024-2026 field zeitgeist.
Identify which direction you would force me to pursue first and why.
Identify one backup direction.
Give the single experiment I should run this week.
Give the result that would convince you to continue.
Give the result that would make you pivot.
Give the one paper or benchmark I should imitate structurally.
Give the one overclaim you would forbid.
Give a 30-day plan with weekly deliverables.
Give a list of questions I should ask another Pro model to double-check
your advice.
Be blunt and concrete. I prefer honest negative feedback over vague optimism.

# 
## My blunt recommendation

I would **force you to start with Direction 1: same-person speech-vs-singing timbre shift**, framed as:

**“Where does singer identity survive when a person moves from speech to singing, and does that residual explain failures or prompt sensitivity in zero-shot SVC?”**

Do **not** start by training a new SVC model. Start with **frozen SSL/audio-codec/speaker embeddings, probing, residual analysis, and a small Seed-VC intervention**. That is feasible on a few GPUs, educational, and more defensible than “we made another converter.”

The key field facts behind this recommendation: SVCC 2023 explicitly showed that cross-domain SVC is harder than in-domain SVC, especially for similarity, and that only a few objective metrics correlated significantly with perception. ([arXiv](https://arxiv.org/abs/2306.14422 "[2306.14422] The Singing Voice Conversion Challenge 2023")) Seed-VC directly targets timbre leakage and training-inference mismatch, releases code/models, and already supports zero-shot singing voice conversion. You have same-person speech/singing data options: JVS-MuSiC uses the same 100 people as the JVS speech corpus, NHSS has 10 singers with sung and spoken versions of 100 English songs, and GTSinger adds 80.59 hours of singing, 20 singers, nine languages, six technique labels, and 16.16 hours of paired speech. ([arXiv](https://arxiv.org/abs/2001.07044 "[2001.07044] JVS-MuSiC: Japanese multispeaker singing-voice corpus")) For the backup, GTSinger and VocalSet make technique probing realistic rather than wishful: GTSinger has phoneme-level technique labels for mixed voice, falsetto, breathy, pharyngeal, vibrato, and glissando; VocalSet has 10.1 hours from 20 singers demonstrating standard and extended techniques. ([GitHub](https://github.com/AaronZ345/GTSinger "GitHub - AaronZ345/GTSinger: Dataset and code of GTSinger(NeurIPS 2024 Spotlight): A Global Multi-Technique Singing Corpus with Realistic Music Scores for All Singing Tasks · GitHub"))

## Risk-adjusted ranking

Scores are 1–5. **Overall rank is not a pure average; it is my master’s-project priority under limited compute.**

|Overall|Direction|Novelty|Feasibility|Data|Code|Publishability|2024–2026 zeitgeist|Blunt verdict|
|--:|---|--:|--:|--:|--:|--:|--:|---|
|**1**|**Same-person speech-vs-singing timbre shift**|4.5|4.0|4.5|4.5|4.2|5.0|Best intersection of your interests. Real research question, low-compute path, and a clear downstream Seed-VC hook.|
|**2**|**Singing style/technique direction discovery**|4.0|4.5|5.0|4.0|3.8|5.0|Best backup. Very feasible now because GTSinger exists. But “linear probe finds vibrato” alone is too shallow. You need transfer or steering.|
|**3**|**Objective evaluation metrics for singing style conversion**|3.8|3.0|3.5|4.5|4.0|4.8|Important, but dangerous: without subjective style/similarity ratings, it becomes a weak correlation exercise.|
|**4**|**Robustness to separated-vocal artifacts/noisy web singing**|3.5|4.5|4.0|4.0|3.5|4.5|Good as an evaluation axis, not your thesis core. Recent work is already moving here; R2-SVC explicitly uses F0 perturbation and separation-artifact simulation. ([arXiv](https://arxiv.org/html/2510.20677v1 "R2-SVC: TOWARDS REAL-WORLD ROBUST AND EXPRESSIVE ZERO-SHOT SINGING VOICE CONVERSION"))|
|**5**|**Singing quality/skill/technique evaluator**|3.0|3.0|3.5|4.0|3.2|4.5|MOS prediction is feasible; “skill evaluation” is not honest unless you have human skill labels. SingMOS/VoiceMOS already make generic singing MOS crowded. ([Hugging Face](https://huggingface.co/datasets/TangRain/SingMOS-Pro "TangRain/SingMOS-Pro · Datasets at Hugging Face"))|
|**6**|**Cross-lingual timbre leakage/stress test**|3.5|2.5|2.5|3.5|2.8|4.0|Do not start here. Multilingual singing data is not automatically cross-lingual identity data. You will likely confound singer, language, recording, and phonetics.|

## Direction I would force you to pursue first

**Primary project: same-person speech-vs-singing timbre shift.**

Your thesis should not be “speech and singing are different.” That is obvious. The thesis should be:

> **Frozen content and codec representations used by modern VC/SVC systems retain a measurable mixture of singer identity and speech/singing mode. A stable identity component and a mode residual can be mapped, and the residual partly predicts or controls prompt-mode failures in zero-shot singing voice conversion.**

This is good because it has three layers of contribution:

1. **Representation learning:** layer-wise leakage map across HuBERT/WavLM/Whisper/codec/Seed-VC features.
    
2. **Interpretability:** stable identity core versus speech/singing residual, with F0/loudness/content controls.
    
3. **Downstream relevance:** Seed-VC prompt intervention or prompt-mode stress test.
    

It is also educational: you will learn SSL feature extraction, speaker/singer verification, linear probing, residualization, SVC inference, objective metrics, and small listening-test design.

The biggest risk is that your “mode residual” is actually just **pitch range, vowel duration, vibrato, loudness, or dataset channel**. If you do not control those, the project becomes pseudo-interpretability.

## Backup direction

**Backup: singing technique direction discovery on GTSinger/VocalSet.**

Frame it as:

> **Which frozen audio representations linearly encode singing techniques, which directions generalize across singer/language, and can those directions steer or at least predict technique preservation in SVC?**

This is safer than quality/skill evaluation because the labels exist. GTSinger gives you realistic technique labels and paired speech; VocalSet gives an older but clean technique dataset. ([GitHub](https://github.com/AaronZ345/GTSinger "GitHub - AaronZ345/GTSinger: Dataset and code of GTSinger(NeurIPS 2024 Spotlight): A Global Multi-Technique Singing Corpus with Realistic Music Scores for All Singing Tasks · GitHub"))

But again: **do not publish “we classify vibrato.”** That is a class project. You need at least one of:

- cross-singer and cross-dataset generalization,
    
- layer-wise comparison across SSL/codecs,
    
- control for F0 statistics,
    
- steering/intervention in a downstream model,
    
- or a failure analysis showing which techniques are entangled with identity.
    

## The single experiment to run this week

Run a **speech-to-singing identity leakage triage**.

Use **JVS + JVS-MuSiC first**, because it gives you many same-person speech/singing identities and is small enough to move fast. Add NHSS only if setup is painless.

### Dataset

Use 50–100 same people.

For each person:

- speech: 5–10 clean JVS utterances,
    
- singing: 5–10 crops from JVS-MuSiC,
    
- crop length: 3–5 seconds,
    
- normalize loudness,
    
- remove silence,
    
- store mean F0, F0 std, voiced ratio, energy, duration.
    

### Frozen representations

Extract utterance-level pooled embeddings from:

- HuBERT-base or WavLM-base layers,
    
- Whisper-small encoder layers,
    
- Seed-VC content encoder if easy,
    
- one speaker/singer identity model, preferably ECAPA/CAM++ plus the ISMIR singer-identity SSL model if setup is easy. The singer-identity SSL work explicitly releases code and pretrained models. ([arXiv](https://arxiv.org/abs/2401.05064 "[2401.05064] Singer Identity Representation Learning using Self-Supervised Techniques"))
    

### Probes

For every model/layer:

1. **Mode probe:** speech vs singing, speaker-disjoint split.  
    This tells you where “singingness” lives.
    
2. **Speaker leakage probe:** train singer/speaker ID on speech, test on singing; then train on singing, test on speech.  
    This tells you whether the identity representation transfers across mode.
    
3. **Same-person retrieval:** for each singing crop, retrieve nearest speech speaker centroid.  
    Report Recall@1 and Recall@5. With 100 speakers, chance Recall@1 is 1%.
    
4. **Residual control:** regress out mean F0, F0 std, energy, duration, voiced ratio, and repeat the probes.  
    This is non-negotiable.
    
5. **Tiny Seed-VC smoke test:** for 5–10 target speakers, convert the same source singing clip twice:
    
    - target reference = target’s speech,
        
    - target reference = target’s singing.
        
    
    Measure speaker/singer similarity to the target singing reference, F0 correlation, CER/lyrics if usable, and SingMOS as a weak naturalness proxy. SingMOS/SingMOS-Pro have ready-to-use predictors, but treat them as MOS proxies, not truth. ([GitHub](https://github.com/South-Twilight/SingMOS "GitHub - South-Twilight/SingMOS: Officail repo for SingMOS-Pro (ICASSP 2026) · GitHub"))
    

### Deliverable by the end of the week

One figure:

> **Layer on x-axis; y-axis has mode AUC, cross-mode speaker Recall@1, and same-speaker speech↔singing cosine gap.**

One audio folder:

> **10 Seed-VC pairs: same source singing, same target identity, speech prompt vs singing prompt.**

That is enough to decide whether this project has a pulse.

## Result that would convince me to continue

Continue if you see **all three**:

1. **A nontrivial layer-wise tradeoff:** some layers preserve cross-mode identity while others strongly encode speech/singing mode.
    
2. **The effect survives controls:** after regressing out F0, energy, voiced ratio, and duration, mode classification is still clearly above chance, e.g. AUC ≥ 0.75, and cross-mode identity retrieval is still meaningfully above chance, e.g. Recall@1 ≥ 10% with 100 speakers.
    
3. **Seed-VC cares about prompt mode:** speech prompts and singing prompts for the same target produce consistent differences in target-singing similarity or listener preference. A small but consistent effect is enough: for example, singing-prompt outputs beat speech-prompt outputs in ≥70% of same-target comparisons, or objective singer similarity shifts by a consistent margin.
    

A beautiful result would be: **a residual vector estimated from training speakers moves held-out speech embeddings closer to their singing embeddings and predicts which Seed-VC prompts fail.**

## Result that would make me pivot

Pivot if any of these happen:

- The speech/singing residual vanishes after F0/loudness/duration controls.
    
- Cross-mode speaker retrieval is near chance for every useful representation.
    
- Results disagree wildly across HuBERT/WavLM/Whisper/Seed-VC features.
    
- Seed-VC output is insensitive to whether the target prompt is speech or singing.
    
- The only strong signal is gender/pitch range.
    
- Dataset artifacts dominate: recording condition, song identity, or phonetic content explains the effect better than mode.
    

If that happens, pivot to **GTSinger technique direction discovery**, because the labels are cleaner for the questions you care about.

## The one paper/benchmark to imitate structurally

Imitate **SVCC 2023**, not a huge model paper.

Why: it gives you the right structure—public data, in-domain versus cross-domain conversion, baseline systems, naturalness/similarity evaluation, objective metrics, and an explicit admission that objective metrics only partially predict perception. ([arXiv](https://arxiv.org/abs/2306.14422 "[2306.14422] The Singing Voice Conversion Challenge 2023"))

Your mini-version:

- **Task A:** target prompt is singing.
    
- **Task B:** target prompt is speech.
    
- **Analysis:** representation leakage predicts Task B degradation.
    
- **Evaluation:** speaker/singer similarity, F0 metrics, intelligibility proxy, SingMOS proxy, and a small ABX/MUSHRA-style listening test.
    

Do **not** imitate a paper that trains a giant SVC model. That is a compute trap.

## The one overclaim I forbid

Forbidden claim:

> “We disentangle singer identity from singing style.”

You probably will not.

Allowed claim:

> “In these datasets and frozen representations, we find a reproducible speech/singing mode residual that is not fully explained by F0/energy controls, and this residual predicts or partially mitigates prompt-mode sensitivity in zero-shot SVC.”

Also forbidden:

> “Our model evaluates singing skill.”

Unless you collect real human skill/technique labels from singers or teachers, MOS is not skill.

## 30-day plan

### Week 1 — Triage and kill test

Deliverables:

- working data manifest for JVS/JVS-MuSiC, optionally NHSS;
    
- embedding extraction scripts;
    
- layer-wise plots for mode AUC, speaker leakage, and cross-mode retrieval;
    
- 10 Seed-VC speech-prompt vs singing-prompt audio pairs;
    
- one-page decision memo: continue, narrow, or pivot.
    

Decision point: if there is no robust signal after controls, pivot immediately.

### Week 2 — Make the measurement credible

Deliverables:

- repeat the core probes on a second dataset or at least a second split;
    
- add F0/loudness/duration/voiced-ratio residualization;
    
- compare HuBERT/WavLM/Whisper/Seed-VC content features, plus speaker/singer embeddings;
    
- define “stable identity core” operationally, not philosophically;
    
- produce the first draft of the main result figure.
    

Target figure: **representation layer × metric heatmap** showing where identity, mode, and leakage live.

### Week 3 — Downstream intervention

Deliverables:

- Seed-VC prompt-mode stress test on 20–40 conversions;
    
- one simple intervention:
    
    - singing-prompt replacement,
        
    - average speech→singing residual shift,
        
    - projection away from mode direction,
        
    - or prompt/reference selection based on representation distance;
        
- objective metrics: singer similarity, F0 correlation/RMSE, CER if lyrics are usable, SingMOS proxy;
    
- small listening test design, even if only 8–15 listeners.
    

The intervention can be tiny. It just needs to show that the representation analysis is not decorative.

### Week 4 — Package as a paper-shaped project

Deliverables:

- 4–6 page workshop-style paper draft;
    
- clean repo with cached metadata and reproducible scripts;
    
- audio demo page;
    
- limitations section;
    
- ethics statement: public/consented datasets only, no celebrity/web voice cloning demos;
    
- thesis outline with three chapters:
    
    1. representation leakage map,
        
    2. speech/singing residual analysis,
        
    3. Seed-VC prompt-mode intervention.
        

By day 30, you should know whether this is a thesis or a dead end.

## Questions to ask another Pro model to double-check this advice

Ask it these, verbatim if useful:

1. **What prior work already studies same-person speech-vs-singing timbre residuals in SSL or codec representations?**
    
2. **Is JVS + JVS-MuSiC a valid dataset pair for speech/singing identity analysis, or are recording/content confounds too severe?**
    
3. **Which same-person speech/singing datasets are best for this project under open research licenses?**
    
4. **Which frozen encoders should I probe: HuBERT, WavLM, Whisper, EnCodec, DAC, SpeechTokenizer, Seed-VC internals, CAM++/ECAPA, or singer-identity SSL models?**
    
5. **How should I control for F0, loudness, vibrato, duration, phonetic content, and recording channel without removing actual timbre?**
    
6. **What is the cleanest operational definition of “timbre leakage” for a frozen content representation?**
    
7. **How would you intervene in Seed-VC without retraining the whole model?**
    
8. **What objective metrics are least misleading for singing voice conversion similarity and style preservation?**
    
9. **What minimum listening test would make the claims credible for a master’s thesis?**
    
10. **Is the GTSinger technique-direction backup already too obvious because of TechSinger/GTSinger benchmarks?**
    
11. **What result would make the speech-vs-singing project publishable even if the intervention fails?**
    
12. **What ethical constraints should I impose on voice-conversion demos and released audio?**


ok answer the 12 questions you provide 
and any other you think need to go further or refine

## Revised bottom line

I would **still force you to start with same-person speech-vs-singing timbre shift**, but I would sharpen the project:

> **Main project:** map where frozen speech/audio representations leak singer identity and speech/singing mode, then test whether that map predicts or improves speech-prompted zero-shot SVC behavior.

The backup should be **GTSinger technique-direction discovery**, but only if you make it more than “classify vibrato.” SVCC 2025 has already moved the field toward **singing style conversion**, and it explicitly reports that breathy, glissando, and vibrato remain hard dynamic styles. That makes your backup timely, but also more crowded. ([arXiv](https://arxiv.org/html/2509.15629v1 "The Singing Voice Conversion Challenge 2025: From Singer Identity Conversion To Singing Style Conversion"))

---

# Answers to the 12 questions

## 1. What prior work already studies same-person speech-vs-singing timbre residuals in SSL or codec representations?

**Exact answer:** I do **not** see a clearly dominant prior paper that does exactly your proposed project: **same-person speech-vs-singing residuals inside frozen SSL/audio-codec representations plus downstream SVC intervention**.

But there is a lot of adjacent work, so your novelty must be narrow and disciplined.

The strongest adjacent dataset paper is **JVS-MuSiC**. It uses the same 100 people as the JVS speech corpus, explicitly says this enables studying the relationship between speech and singing, and reports that speech similarity and singing similarity are only weakly correlated. That is basically the empirical opening for your thesis.

The strongest adjacent benchmark is **SVCC 2023**, which showed that cross-domain SVC is harder than in-domain SVC, especially in similarity, and that only a few objective metrics significantly correlate with perception. The strongest modern model-adjacent hook is **Seed-VC**, which explicitly targets timbre leakage and training-inference mismatch, and includes zero-shot singing voice conversion with F0 conditioning. ([arXiv](https://arxiv.org/abs/2411.09943?utm_source=chatgpt.com "Zero-shot Voice Conversion with Diffusion Transformers"))

The strongest adjacent “speech prompt to singing timbre” work is the 2025 speech-prompted SVC paper that introduces **Singing Speech Alignment Network** to align speaker and singer embeddings. ([ISCA Archive](https://www.isca-archive.org/interspeech_2025/liu25h_interspeech.pdf?utm_source=chatgpt.com "Bridging Speech and Singing: Multi-stage Speech-Prompted Singing Voice ...")) That paper makes your topic more obviously relevant, but also means you should avoid claiming nobody has thought about speech-prompted SVC.

**Blunt framing:** your novelty is not “speech and singing differ.” Your novelty is:

> **A layer-wise leakage and residual map of frozen representations, validated against prompt-mode failures in zero-shot SVC.**

That is plausible. “We discovered speech and singing timbre shift” is not.

---

## 2. Is JVS + JVS-MuSiC valid, or are confounds too severe?

**Valid for triage and representation analysis. Not sufficient alone for a thesis claim.**

JVS-MuSiC is excellent for a first experiment because it has **100 singers who are the same people as in JVS**, and the corpus was explicitly designed to study singing voice personality and speech/singing relationships. It also gives a common song, which reduces song-content variance across singers.

But it has confounds you must respect:

First, the singing and speech are not truly parallel. Speech utterances and sung lyrics differ. Second, the singing data includes key/tempo variation, and the corpus also contains Melodyne-modified versions; the paper itself notes key/tempo manipulation and possible timbre changes from pitch shifting. Third, the common song is short. Fourth, the Japanese nursery-song setup may not generalize to pop/expressive singing.

**Use JVS/JVS-MuSiC as your Week 1 kill test.** Then replicate the core finding on **NHSS** or **GTSinger**. NHSS is smaller but cleaner for parallel speech/singing because it contains sung songs and spoken lyric counterparts from the same 10 singers. ([arXiv](https://arxiv.org/abs/2012.00337 "[2012.00337] NHSS: A Speech and Singing Parallel Database")) GTSinger is broader and more modern, with 80.59 hours of singing, 20 professional singers, nine languages, six technique labels, and 16.16 hours of paired speech. ([arXiv](https://arxiv.org/abs/2409.13832 "[2409.13832] GTSinger: A Global Multi-Technique Singing Corpus with Realistic Music Scores for All Singing Tasks"))

---

## 3. Which same-person speech/singing datasets are best under open research constraints?

Use this order:

|Dataset|Use it for|Why|Main weakness|
|---|---|---|---|
|**JVS + JVS-MuSiC**|Week 1 identity shift triage|100 same people, easy size, explicit speech/singing relationship|Not parallel; Japanese only; limited singing material|
|**NHSS**|Parallel speech/singing validation|10 singers, 100 songs sung and spoken, 7 hours total, utterance/word annotations|Only 10 singers; English pop-song domain|
|**GTSinger**|Modern thesis-scale validation and technique backup|80.59 hours, 20 singers, 9 languages, technique labels, paired speech, benchmarks/code|Fewer identities than JVS; professional singers only|
|**VocalSet**|Technique backup only|10.1 hours, 20 professional singers, standard and extended techniques, CC-BY|Not speech/singing paired; mostly vowels/scales/exercises|

JVS-MuSiC and JVS give you **identity count**. NHSS gives you **parallel content**. GTSinger gives you **modern style/technique relevance**. VocalSet gives you **clean technique labels**.

Do **not** make cross-lingual claims from JVS + GTSinger unless your split is designed for that. Multilingual data is not automatically a multilingual identity benchmark.

---

## 4. Which frozen encoders should you probe?

Start with **six**, not twenty.

### Must probe

1. **WavLM-base or WavLM-large**  
    WavLM is designed for full-stack speech processing and explicitly targets multifaceted information including speaker identity, content, and paralinguistics. ([arXiv](https://arxiv.org/abs/2110.13900?utm_source=chatgpt.com "WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech ..."))
    
2. **HuBERT-base**  
    This is the canonical SSL content model, and many VC/SVC systems are built around HuBERT-like or ContentVec-like representations. ([arXiv](https://arxiv.org/abs/2106.07447?utm_source=chatgpt.com "HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units"))
    
3. **ContentVec**  
    ContentVec is directly relevant because it was designed to reduce speaker variation in SSL content representations. If it still leaks speech/singing identity, that is interesting. ([arXiv](https://arxiv.org/abs/2204.09224?utm_source=chatgpt.com "ContentVec: An Improved Self-Supervised Speech Representation by Disentangling Speakers"))
    
4. **Whisper-small or Whisper-medium encoder**  
    Whisper is trained on large-scale multilingual weak supervision and is widely used in recent speech/audio systems; it may encode language/content more strongly than identity, which makes it a useful contrast. ([arXiv](https://arxiv.org/abs/2212.04356?utm_source=chatgpt.com "Robust Speech Recognition via Large-Scale Weak Supervision"))
    
5. **Seed-VC internal content/timbre features, if accessible**  
    This matters because you want a downstream Seed-VC hook, and the repository supports zero-shot VC and zero-shot SVC from 1–30 second references. ([GitHub](https://github.com/Plachtaa/seed-vc?utm_source=chatgpt.com "GitHub - Plachtaa/seed-vc: zero-shot voice conversion & singing voice ..."))
    
6. **Singer-identity SSL embedding model**  
    Use it as a target-side similarity metric, not as your main representation. The ISMIR singer-identity work releases code and pretrained models and explicitly focuses on singer identity representations. ([GitHub](https://github.com/SonyCSLParis/ssl-singer-identity?utm_source=chatgpt.com "GitHub - SonyCSLParis/ssl-singer-identity"))
    

### Nice but not first week

Probe **EnCodec/DAC/SpeechTokenizer/WavTokenizer** only after your SSL experiment works. EnCodec, SpeechTokenizer, and WavTokenizer are interesting because they expose quantized or hierarchical token spaces, but they will multiply your analysis complexity. ([AudioCraft](https://audiocraft.metademolab.com/encodec.html?utm_source=chatgpt.com "EnCodec: High-fidelity Neural Audio Compression"))

**Blunt choice:** Week 1 = WavLM, HuBERT, ContentVec, Whisper, ECAPA/CAM++ or singer-ID SSL. Add codecs later.

---

## 5. How should you control for F0, loudness, vibrato, duration, phonetic content, and recording channel without removing actual timbre?

Use **three levels of control**.

### Level 1: nuisance-feature residualization

For each 3–5 second crop, compute:

- mean log-F0,
    
- F0 standard deviation,
    
- voiced ratio,
    
- energy mean/std,
    
- duration,
    
- spectral centroid,
    
- spectral rolloff,
    
- speaking/singing rate proxy,
    
- maybe vibrato rate/depth for sung clips.
    

Then fit probes both **before and after residualizing embeddings against these features**. If your speech/singing mode probe collapses after this, your “mode residual” was mostly pitch/energy/duration.

### Level 2: split controls

Use speaker-disjoint and song-disjoint splits where possible. For JVS-MuSiC, compare common-song-only results against unique-song results. For NHSS, use song-held-out splits. For GTSinger, use singer-held-out and language-held-out splits.

### Level 3: matched-pair analysis

For NHSS, exploit spoken and sung versions of the same lyrics. For JVS-MuSiC, use same-person speech centroid versus singing centroid, but avoid pretending the content is parallel. NHSS is explicitly parallel at the speech/singing lyric level, with spoken counterparts and annotations. ([arXiv](https://arxiv.org/abs/2012.00337 "[2012.00337] NHSS: A Speech and Singing Parallel Database"))

**Do not over-control.** If you regress out every spectral feature, you may remove real timbre. The key is to report two results:

> “Raw representation contains X.”  
> “After pitch/energy/duration controls, Y remains.”

That is honest.

---

## 6. What is the cleanest operational definition of “timbre leakage”?

Use this:

> **Timbre leakage is the amount of singer/speaker identity or speech/singing mode information recoverable from a representation that is intended to encode content, independent of explicit pitch, energy, duration, and phonetic controls.**

Measure it four ways:

1. **Linear speaker/singer ID probe**  
    Train on speech, test on singing; train on singing, test on speech.
    
2. **Same-person retrieval**  
    Given a singing crop, retrieve the matching speech speaker centroid.
    
3. **Mode probe**  
    Predict speech versus singing with speaker-disjoint splits.
    
4. **Conversion consequence**  
    If a representation says two same-person speech/singing samples are far apart, does Seed-VC also perform worse with speech prompts than singing prompts?
    

Your strongest thesis figure should show:

> **Layer × representation: mode AUC, cross-mode identity retrieval, and downstream prompt-mode degradation.**

Do not define leakage only as “speaker classifier accuracy.” That misses the downstream relevance.

---

## 7. How would you intervene in Seed-VC without retraining the whole model?

Use interventions in this order.

### Intervention 1: reference prompt selection

For each target speaker/singer, choose the reference clip whose embedding is closest to the target singer centroid or farthest from the speech-mode direction. This requires no model surgery.

Compare:

- random speech prompt,
    
- best speech prompt by representation distance,
    
- singing prompt,
    
- speech + singing concatenated prompt if available.
    

This alone may be publishable as “representation-guided prompt selection.”

### Intervention 2: reference concatenation and balancing

Seed-VC accepts short reference speech/audio clips; the repo says the current model can clone from a 1–30 second reference and supports zero-shot SVC. ([GitHub](https://github.com/Plachtaa/seed-vc?utm_source=chatgpt.com "GitHub - Plachtaa/seed-vc: zero-shot voice conversion & singing voice ...")) So test:

- 5s speech prompt,
    
- 15s speech prompt,
    
- 5s singing prompt,
    
- 5s speech + 5s singing,
    
- speech prompt with high-pitch/loudness-normalized segments removed.
    

### Intervention 3: feature-space projection

If you can hook internal features:

[  
h' = h - \alpha w_{\text{mode}}  
]

where (w_{\text{mode}}) is the linear speech/singing classifier direction. This tests whether removing mode information from the content path improves target-timbre similarity.

### Intervention 4: residual shift

Estimate:

[  
r = \mathbb{E}[z_{\text{singing}} - z_{\text{speech}}]  
]

for same-person pairs, then apply:

[  
z'_{\text{speech-ref}} = z_{\text{speech-ref}} + \alpha r  
]

This is riskier and more speculative, but it is the cleanest “mode residual steering” experiment.

**Blunt advice:** start with prompt selection and prompt-mode stress tests. Only do feature-space steering if the Week 1 leakage map is strong.

---

## 8. What objective metrics are least misleading for SVC similarity and style preservation?

Use a **basket**, not a single metric.

For **naturalness/quality**:

- SingMOS or SingMOS-Pro predictor as a singing-specific MOS proxy,
    
- UTMOS/NISQA only as secondary speech-quality proxies,
    
- VERSA if you want a reproducible metric toolkit.
    

SingMOS-Pro contains 7,981 singing clips generated by 41 models across 12 datasets, with professional ratings for overall/lyrics/melody quality; it is more relevant to singing than generic speech MOS. ([arXiv](https://arxiv.org/abs/2510.01812?utm_source=chatgpt.com "SingMOS-Pro: An Comprehensive Benchmark for Singing Quality Assessment")) VERSA is attractive because it provides a standardized toolkit with many speech/audio/music metrics. ([ACL Anthology](https://aclanthology.org/2025.naacl-demo.19/?utm_source=chatgpt.com "VERSA: A Versatile Evaluation Toolkit for Speech, Audio, and Music"))

For **identity similarity**:

- singer-identity SSL embedding cosine,
    
- ECAPA/CAM++/speaker embedding cosine as secondary,
    
- ABX human similarity if possible.
    

For **style/technique preservation**:

- technique classifier/probe accuracy,
    
- vibrato rate/depth error,
    
- glissando slope error,
    
- breathiness/noise ratio proxy,
    
- F0 RMSE/correlation,
    
- loudness correlation.
    

SVCC 2025 found that neural MOS systems like SHEET-SSQA and SingMOS correlated well with subjective quality, while speaker and singer embeddings had high rank correlation with subjective identity/style similarity; but the paper still says objective metrics need improvement. ([arXiv](https://arxiv.org/html/2509.15629v1 "The Singing Voice Conversion Challenge 2025: From Singer Identity Conversion To Singing Style Conversion"))

**Forbidden behavior:** reporting SingMOS as “human-perceived singing skill.” It is not that.

---

## 9. What minimum listening test would make the claims credible for a master’s thesis?

Minimum viable listening test:

### For prompt-mode SVC

Use **ABX or paired preference**, not just MOS.

For each item:

- A = output using speech prompt,
    
- B = output using singing prompt,
    
- X/reference = target singer singing sample,
    
- ask: “Which output sounds more like the target singer?”
    
- ask separately: “Which output sounds more natural?”
    

Use:

- 20–30 conversion pairs,
    
- 10–20 listeners,
    
- each pair rated by at least 5 listeners,
    
- headphone check,
    
- randomized order,
    
- no celebrity voices,
    
- no model names shown.
    

SVCC 2025 used MOS for naturalness, XAB for style similarity, and a 4-point singer identity similarity test with multiple target references. ([arXiv](https://arxiv.org/html/2509.15629v1 "The Singing Voice Conversion Challenge 2025: From Singer Identity Conversion To Singing Style Conversion")) Your master’s-scale version should imitate that structure, not its scale.

**Minimum acceptable:** 100–200 total judgments.  
**Good master’s version:** 300–500 judgments.  
**Do not claim SOTA from this.** Claim “consistent perceptual trend in a small controlled listening test.”

---

## 10. Is the GTSinger technique-direction backup already too obvious because of TechSinger/GTSinger benchmarks?

**Yes, if you do naïve classification. No, if you use it as a controlled interpretability/steering benchmark.**

GTSinger already includes phoneme-level technique annotations and benchmarks for technique-controllable SVS, technique recognition, style transfer, and speech-to-singing conversion. ([arXiv](https://arxiv.org/abs/2409.13832 "[2409.13832] GTSinger: A Global Multi-Technique Singing Corpus with Realistic Music Scores for All Singing Tasks")) SVCC 2025 also uses a GTSinger subset and evaluates seven styles: breathy, falsetto, mixed voice, pharyngeal, glissando, vibrato, and control. ([arXiv](https://arxiv.org/html/2509.15629v1 "The Singing Voice Conversion Challenge 2025: From Singer Identity Conversion To Singing Style Conversion"))

So these are weak projects:

- “We classify vibrato.”
    
- “We train a technique recognizer.”
    
- “We visualize technique embeddings.”
    

These are stronger:

- “Which frozen encoders encode technique independent of singer identity?”
    
- “Do technique directions transfer from GTSinger to VocalSet?”
    
- “Are dynamic styles like breathy/glissando/vibrato encoded differently from static styles like falsetto/mixed/pharyngeal?”
    
- “Can technique directions predict failure cases in SVCC-style conversion?”
    
- “Can subtracting an identity direction improve technique classification?”
    

SVCC 2025 explicitly reports breathy, glissando, and vibrato as difficult styles because they involve dynamic time-varying components. ([arXiv](https://arxiv.org/html/2509.15629v1 "The Singing Voice Conversion Challenge 2025: From Singer Identity Conversion To Singing Style Conversion")) That is your opening.

---

## 11. What result would make the speech-vs-singing project publishable even if the intervention fails?

Any one of these could be enough for a workshop or strong master’s thesis:

### Publishable result A: representation map

You show a robust layer-wise pattern across JVS/JVS-MuSiC and NHSS/GTSinger:

- lower/middle layers preserve cross-mode identity,
    
- upper layers encode speech/singing mode,
    
- ContentVec reduces but does not eliminate identity leakage,
    
- speaker embeddings fail asymmetrically from speech to singing,
    
- singer embeddings outperform speech speaker embeddings for singing similarity.
    

### Publishable result B: residual survives controls

You show that the speech/singing residual remains after controlling F0, energy, duration, and voiced ratio.

### Publishable result C: negative downstream result

You show that frozen representation leakage **does not** predict Seed-VC prompt-mode behavior, but provide a clean benchmark and analysis explaining why. Negative results can be publishable if the benchmark is clean and the failure is informative.

### Publishable result D: prompt-mode benchmark

You release a small reproducible stress test:

- same target identity,
    
- speech prompt versus singing prompt,
    
- fixed source singing,
    
- objective metrics,
    
- listening-test template,
    
- audio demo page.
    

This would imitate the useful parts of SVCC while staying master’s-scale. SVCC 2023 and 2025 are benchmark-style papers with shared data, baselines, objective metrics, and listening tests; your project can be a focused mini-benchmark.

---

## 12. What ethical constraints should you impose on voice-conversion demos and released audio?

Use these rules:

1. **Only public research datasets or consented voices.**
    
2. **No celebrity, influencer, classmate, professor, or “random YouTube singer” demos.**
    
3. **No deceptive before/after demos that imply real endorsement.**
    
4. **Watermark or clearly label all converted audio as synthetic.**
    
5. **Release scripts and metrics, not a turnkey impersonation app.**
    
6. **Do not optimize for making a target person maximally cloneable.**
    
7. **Do not publish converted samples of non-consenting private individuals.**
    
8. **In the paper, call it identity/style similarity, not “cloning accuracy.”**
    

Seed-VC’s public repo makes zero-shot voice and singing conversion easy from short references, which is exactly why you should be conservative with demos. ([GitHub](https://github.com/Plachtaa/seed-vc?utm_source=chatgpt.com "GitHub - Plachtaa/seed-vc: zero-shot voice conversion & singing voice ..."))

---

# Additional questions I think you should ask — with answers

## 13. What should the exact thesis title be?

Use something like:

> **“Mapping Speech-to-Singing Timbre Shift in Frozen Audio Representations for Zero-Shot Singing Voice Conversion”**

Avoid:

> “Disentangling Singer Identity and Style”

That title overclaims.

---

## 14. What is the actual research hypothesis?

Use three hypotheses:

**H1:** Frozen speech/audio representations contain both a stable identity component and a speech/singing mode component.

**H2:** The strength and location of identity/mode information differ by encoder and layer.

**H3:** The distance between a target’s speech and singing representations predicts prompt-mode degradation in zero-shot SVC.

This is testable. It can fail. That is good science.

---

## 15. What is the smallest publishable unit?

A 4–6 page workshop paper with:

1. Dataset protocol: JVS/JVS-MuSiC + NHSS or GTSinger.
    
2. Layer-wise leakage map.
    
3. Controls for F0/energy/duration.
    
4. Seed-VC prompt-mode stress test.
    
5. Small listening test or at least audio demo + objective metrics.
    
6. Clear limitations.
    

Do **not** wait until you have a new model. Your contribution is measurement + intervention, not architecture.

---

## 16. What is the biggest technical risk?

The biggest risk is that your “mode residual” is just **F0 range and phonation dynamics**.

Singing differs from speech in pitch, duration, energy, vibrato, and phonation. SVCC 2023 explicitly notes that SVC is harder because singing involves wider variation in pitch, energy, expression, and style. If your probe only learns “high sustained pitch = singing,” your project is dead.

That is why F0/energy/duration controls are not optional.

---

## 17. What is the biggest novelty risk?

The biggest novelty risk is **speech-prompted SVC** already being an active topic. The 2025 speech-prompted SVC paper directly investigates converting speech timbre into singing and proposes speaker/singer embedding alignment. ([ISCA Archive](https://www.isca-archive.org/interspeech_2025/liu25h_interspeech.pdf?utm_source=chatgpt.com "Bridging Speech and Singing: Multi-stage Speech-Prompted Singing Voice ..."))

Your defense:

> “We are not proposing a new speech-prompted SVC model. We are analyzing frozen representations and explaining/predicting prompt-mode sensitivity.”

That distinction matters.

---

## 18. What baseline should you implement first?

Implement **no-training baselines** first:

1. ECAPA/CAM++ speech speaker embedding.
    
2. Singer-ID SSL embedding.
    
3. WavLM pooled layer embeddings.
    
4. HuBERT/ContentVec pooled layer embeddings.
    
5. Whisper pooled encoder embeddings.
    
6. Seed-VC speech-prompt vs singing-prompt inference.
    

If you are training a neural model in Week 1, you are probably wasting time.

---

## 19. What exact figure should go in the paper?

Main figure:

|x-axis|y-axis / color|
|---|---|
|Encoder layer|mode AUC|
|Encoder layer|speech→singing identity Recall@1|
|Encoder layer|singing→speech identity Recall@1|
|Encoder layer|same-person speech/singing cosine gap|
|Encoder layer|residualized mode AUC|

Then a second figure:

> **Seed-VC prompt-mode degradation versus representation speech/singing gap.**

If that scatter plot has a trend, you have a thesis.

---

## 20. What would make me kill the project immediately?

Kill or pivot if:

- after controls, mode AUC falls near chance;
    
- cross-mode identity retrieval is near chance for all representations;
    
- results only separate male/female or pitch range;
    
- JVS result does not replicate on NHSS or GTSinger;
    
- Seed-VC speech-prompt and singing-prompt outputs are perceptually indistinguishable;
    
- the only “improvement” comes from cherry-picked audio examples.
    

Your pivot is then GTSinger technique-direction discovery.

---

## 21. What should the GTSinger backup look like?

Backup project title:

> **“Static and Dynamic Singing Technique Directions in Frozen Audio Representations”**

Core idea:

- static-ish techniques: falsetto, mixed voice, pharyngeal;
    
- dynamic techniques: vibrato, glissando, breathy;
    
- compare which encoders capture each;
    
- test cross-singer generalization;
    
- test GTSinger → VocalSet transfer;
    
- test whether directions predict SVCC-style conversion failures.
    

This is aligned with SVCC 2025 because the benchmark reports breathy, glissando, and vibrato as especially difficult. ([arXiv](https://arxiv.org/html/2509.15629v1 "The Singing Voice Conversion Challenge 2025: From Singer Identity Conversion To Singing Style Conversion"))

---

## 22. Should you do singing skill evaluation?

Not as the main project.

There is emerging data for listener-rated amateur singing and narration: a 2025 AES paper reports 4,300 ratings of 940 recordings with dimensions including skill, likability, passion, sincerity, and intelligibility. That makes skill evaluation more feasible than it used to be.

But for your interests, skill evaluation is a trap unless you want to become an evaluation/dataset person. “Singing skill” is socially and musically loaded. You will need human labels, rater reliability, genre controls, accompaniment/noise controls, and careful claims.

Use skill evaluation only as a later side branch.

---

## 23. Should you include source-separation artifacts?

As an auxiliary stress test, yes. As the thesis core, no.

R2-SVC explicitly targets real-world robustness using F0 perturbation and source-separation artifact simulation, including reverberation/echo-like artifacts. ([arXiv](https://arxiv.org/abs/2510.20677?utm_source=chatgpt.com "R2-SVC: Towards Real-World Robust and Expressive Zero-shot Singing Voice Conversion")) That means robustness is timely, but it is also becoming its own full project.

For you, add a small appendix experiment:

- clean vocal,
    
- separated vocal,
    
- separated + mild reverb/echo,
    
- F0 perturbed,
    
- compare identity/mode probes and Seed-VC output degradation.
    

Do this only after the core speech/singing representation result works.

---

## 24. Should you use cross-lingual singing?

Not first.

GTSinger has nine languages, but multilingual singing data does not automatically let you claim language-independent timbre leakage. ([arXiv](https://arxiv.org/abs/2409.13832 "[2409.13832] GTSinger: A Global Multi-Technique Singing Corpus with Realistic Music Scores for All Singing Tasks")) You would need singer-balanced language coverage. If singer and language are confounded, your “language effect” is just singer identity or recording style.

Possible safe claim:

> “We include language-held-out stress tests where dataset support allows.”

Unsafe claim:

> “We disentangle language and singer timbre in singing.”

---

## 25. What is the strongest final project scope?

Here is the exact scope I would approve:

> **A master’s thesis measuring speech/singing mode and identity leakage in frozen SSL/audio-codec representations, validating the effect across JVS/JVS-MuSiC and NHSS or GTSinger, and testing whether representation distance predicts prompt-mode sensitivity in Seed-VC zero-shot SVC.**

One sentence contribution:

> **We show where same-person speech and singing separate in modern audio representations, which parts survive pitch/energy controls, and whether those separations matter for zero-shot SVC prompt selection.**

That is concrete, feasible, and not fake-grandiose.

---

# Final narrowed plan

Your next concrete move is unchanged:

**This week, run the leakage-map experiment.**

Minimum result table:

|Representation|Best layer|Mode AUC|Residualized mode AUC|Speech→singing R@1|Singing→speech R@1|
|---|--:|--:|--:|--:|--:|

Minimum Seed-VC result:

|Target|Source|Speech prompt similarity|Singing prompt similarity|Which sounds more target-like?|
|---|---|--:|--:|---|

Continue only if you see a robust residual and some downstream prompt-mode effect. Pivot to GTSinger technique directions if the residual collapses after controls.