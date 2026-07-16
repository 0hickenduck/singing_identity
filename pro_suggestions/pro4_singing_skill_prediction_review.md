I am considering a second research direction: automatic singing skill or singing
quality evaluation.

Original vague idea:

Can we predict how skilled a singer is from audio? Can we predict what a person
would sound like after becoming more professional at singing?

I suspect the "future professional voice" target may not be valid because there
is no obvious longitudinal ground truth. I want you to reformulate this into
researchable tasks if possible.

Constraints:

I only have a few GPUs.
I prefer small models, frozen embeddings, interpretable features, or
benchmark/evaluation work over training a large audio model.
I care about useful singing feedback, not just a number.
I want to know whether this can be a real paper or only a product idea.
Please give a complete review and experiment plan.

Part 1: Reality Check

Is "predict future professional voice" a valid supervised learning target?
What kind of ground truth would be required to make it valid?
What public datasets actually support instead: current quality, MOS,
technique labels, reference-match correctness, pairwise improvement, or
multi-dimensional feedback?
What claims should I avoid?
Part 2: Dataset Survey

Compare these datasets/sources where relevant:

SingMOS-Pro;
SingMOS-v1;
GTSinger;
VocalSet;
SVQTD;
Lyra-SA;
PESnQ / SingEval / AME430;
VocalVerse / QwenFeat-Vocal-Score;
TG-Critic;
10KSinging;
Learn-by-Referencing / ranking-based singing assessment;
MFFMOS / GOSMOS;
classic Nakano / MiruSinger work;
any Waseda/Nakano/Goto/Hiraga karaoke or singing assessment work that is
relevant.
For each, state:

labels;
access;
code availability;
best use;
leakage risks;
whether it is suitable for a small-GPU project.
Part 3: Task Formulations

Design rigorous alternatives to "future professional potential":

no-reference current singing quality;
reference-conditioned karaoke pitch/rhythm/lyric scoring;
singing technique or vocal-attribute recognition;
multi-dimensional feedback for pitch, rhythm, timbre, breath, emotion, and
technique;
pairwise improvement ranking across takes.
For each, give:

why it matters;
dataset;
baselines;
metrics;
split strategy;
expected compute;
scientific risk;
product/usefulness risk.
Part 4: Karaoke Scoring Traps

Explain how a karaoke scoring model can be wrong or misleading. Include:

punishment of expressive deviation;
accompaniment/source-separation leakage;
popularity and platform bias;
song difficulty;
singer identity leakage;
generated-vocal versus real-amateur domain shift;
mismatch between pitch accuracy and musical quality.
Part 5: Three Small-GPU Experiments

Design three concrete experiments I could run:

one safest benchmark experiment;
one more novel technique/feedback experiment;
one risky but interesting experiment.
For each, include data, features, model, metrics, controls, expected compute,
minimum publishable result, and failure interpretation.

Part 6: Final Recommendation

Should I pursue singing skill evaluation as my main thesis direction, a backup
direction, or not at all? Compare it against the speech/singing timbre-shift
idea. End with a go/no-go recommendation and a 30-day plan.

Please be strict and source-grounded.

# pro
## Bottom line

**“Predict what this singer would sound like after becoming professional” is not a valid supervised learning target with today’s public singing datasets.** It is a causal, longitudinal, counterfactual target. Current public data can support **current perceived quality**, **reference-match correctness**, **technique/attribute recognition**, **multi-dimensional feedback**, and **pairwise take ranking**. It cannot support a defensible claim about a person’s future professional voice or latent potential.

A real paper is possible, but not as “future professional voice prediction.” The strongest paper shape is:

**small-model, leakage-robust, interpretable singing feedback evaluation**, with carefully separated claims: “we predict human ratings / sub-ratings under a defined protocol,” not “we objectively measure talent.”

---

# Part 1: Reality check

## Is “predict future professional voice” a valid supervised target?

**Not with current public data.** To supervise that task, the label would need to be the **same singer’s future recording after a defined training process**, not a different professional singer, not a generated “improved” version, and not a cross-sectional amateur/professional contrast.

The valid target would require something like:

1. **Longitudinal identity tracking**: the same singer recorded at time (t_0), (t_1), (t_2), etc.
    
2. **Standardized recordings**: same or controlled microphone, room, key, repertoire, accompaniment, vocal health state, warmup, and coaching context.
    
3. **Training-exposure metadata**: hours practiced, teacher type, lesson content, technique drills, genre, language, age, puberty/adult status, health events.
    
4. **Expert and listener ratings at each time point**: ideally multi-dimensional, not just a global score.
    
5. **Matched repertoire and difficulty**: otherwise the model learns song difficulty or repertoire selection rather than improvement.
    
6. **Counterfactual framing**: “what would this person sound like after professional training” depends on unobserved future behavior, biology, coaching, and motivation. That is closer to causal inference than normal supervised prediction.
    

No dataset in your list gives that. Some sources contain multiple recordings by singers or repeated performances, but they are not structured longitudinal training data. For example, SVQTD even allows the same classical singer to appear at different career stages, but it is YouTube-derived and was built for attribute labeling, not controlled future-outcome prediction. ([Yanze Xu](https://yanzexu.xyz/SVQTD/ "SVQTD | Singing Voice Quality and Technique Database (SVQTD) is a classical male singing dataset for describing classical tenor singing voices from vocal pedagogy point of view."))

## What public data actually supports

Public or requestable datasets support these safer targets:

|Target|Supported?|Representative sources|What you can claim|
|---|--:|---|---|
|**Current MOS / perceived quality**|Yes|SingMOS, SingMOS-Pro, SingEval, TG-Critic, Lyra-SA, maybe 10KSinging if obtained|“Predicts human ratings on this corpus/protocol.”|
|**Generated singing MOS**|Yes|SingMOS-v1, SingMOS-Pro|“Assesses generated or converted singing quality,” not amateur skill. SingMOS-Pro is built around SVS/SVC/SVR systems plus real vocals. ([arXiv](https://arxiv.org/html/2510.01812v4 "SingMOS-Pro: An Comprehensive Benchmark for Singing Quality Assessment"))|
|**Reference-match pitch/rhythm/lyrics**|Yes|PESnQ, Lyra-SA, Learn-by-Referencing-style tasks|“Scores similarity/correctness relative to a reference/score.”|
|**Technique / vocal attribute recognition**|Yes|GTSinger, VocalSet, SVQTD|“Recognizes labeled technique/attribute categories under split constraints.”|
|**Multi-dimensional feedback**|Partly|PESnQ, VocalVerse/QwenFeat-Vocal-Score, SVQTD, Acapella Evaluation|“Predicts subscore-like dimensions,” with annotation-noise caveats.|
|**Pairwise better/worse take ranking**|Partly|Learn-by-Referencing formulation, DAMP/SingEval, 10KSinging if accessible, custom small collection|“Ranks takes by annotated/current quality,” not true future improvement.|
|**Future professional potential**|No|None public|Do not claim.|

## Claims you should avoid

Avoid these unless you collect the right data:

- “We predict how professional this singer will become.”
    
- “We predict the singer’s future professional voice.”
    
- “We evaluate innate talent.”
    
- “We objectively measure singing skill.”
    
- “The model gives coach-level feedback.”
    
- “The score is independent of genre, song, language, recording device, gender, age, or culture.”
    
- “High pitch/rhythm accuracy means high musical quality.”
    
- “Generated-vocal MOS transfers to amateur karaoke skill.”
    
- “The model measures the singer, not the recording.”
    

The safe claim is narrower:

> “We study small-model singing quality and feedback prediction under leakage-controlled splits, and quantify which aspects generalize across singers, songs, datasets, and recording conditions.”

That can be a real paper.

---

# Part 2: Dataset survey

## Dataset/source comparison

|Source|Labels|Access / code|Best use|Leakage / validity risks|Small-GPU fit|
|---|---|---|---|---|---|
|**SingMOS-Pro**|Utterance- and system-level MOS; extended lyrics, melody, and overall ratings. It contains 7,981 Chinese/Japanese vocal clips, 11.15 hours, 44,247 ratings from 78 annotators, generated by 41 models across 12 datasets. ([arXiv](https://arxiv.org/html/2510.01812v4 "SingMOS-Pro: An Comprehensive Benchmark for Singing Quality Assessment"))|Hugging Face dataset; official benchmark, pretrained predictor, and GitHub code are available. ([Hugging Face](https://huggingface.co/datasets/TangRain/SingMOS-Pro "TangRain/SingMOS-Pro · Datasets at Hugging Face"))|Safest benchmark for MOS prediction, especially generated singing quality.|Strong risk of **system/model/dataset leakage**. Random utterance splits can overstate generalization. Also mostly generated-vocal domain, not ordinary human skill.|**Yes.** Frozen SSL embeddings + ridge/MLP are feasible.|
|**SingMOS-v1**|Overall MOS only; 3,421 clips, 4.25 hours, 16 kHz, Chinese/Japanese. ([Hugging Face](https://huggingface.co/datasets/TangRain/SingMOS-v1 "TangRain/SingMOS-v1 · Datasets at Hugging Face"))|Hugging Face, CC BY-NC 4.0; SingMOS GitHub has pretrained predictors. ([Hugging Face](https://huggingface.co/datasets/TangRain/SingMOS-v1 "TangRain/SingMOS-v1 · Datasets at Hugging Face"))|Small benchmark, replication, ablation, VoiceMOS-style MOS prediction.|Same generated-vocal and system leakage issues; fewer dimensions than Pro.|**Excellent.**|
|**GTSinger**|Professional studio singing; 80.59 hours, 20 singers, 9 languages; six technique labels: mixed voice, falsetto, breathy, pharyngeal, vibrato, glissando; phoneme alignments, global style labels, realistic scores, paired speech. ([GitHub](https://github.com/AaronZ345/GTSinger "GitHub - AaronZ345/GTSinger: Dataset and code of GTSinger(NeurIPS 2024 Spotlight): A Global Multi-Technique Singing Corpus with Realistic Music Scores for All Singing Tasks · GitHub"))|Hugging Face/Google Drive dataset and GitHub code; includes technique-recognition benchmark scripts. ([GitHub](https://github.com/AaronZ345/GTSinger "GitHub - AaronZ345/GTSinger: Dataset and code of GTSinger(NeurIPS 2024 Spotlight): A Global Multi-Technique Singing Corpus with Realistic Music Scores for All Singing Tasks · GitHub"))|Technique recognition, speech-singing style comparison, interpretable attribute work.|Not a general skill dataset: all are professional/studio. Technique may leak through singer, language, folder, or repertoire.|**Yes** for classifiers or frozen embeddings; not for training a full SVS model.|
|**VocalSet**|10.1 hours of monophonic professional singers demonstrating standard and extended techniques across vowels, scales, arpeggios, long tones, and excerpts; 20 singers, voice types included. ([Zenodo](https://zenodo.org/records/1193957 "VocalSet: A Singing Voice Dataset"))|Zenodo, CC BY 4.0. ([Zenodo](https://zenodo.org/records/1193957 "VocalSet: A Singing Voice Dataset"))|Technique/vowel/voice-type classification; interpretable feature studies.|Technique, vowel, singer, register, and exercise context are confounded. Not skill quality.|**Excellent.**|
|**SVQTD**|Nearly 4,000 classical tenor segments, 10.7 hours, labeled on seven pedagogical attributes: chest/head resonance, front/back placement, throat openness, roughness, vibrato quality. ([Yanze Xu](https://yanzexu.xyz/SVQTD/ "SVQTD \| Singing Voice Quality and Technique Database (SVQTD) is a classical male singing dataset for describing classical tenor singing voices from vocal pedagogy point of view."))|Request by signed agreement; public GitHub code includes preprocessing, openSMILE features, SVM/neural baselines. ([Yanze Xu](https://yanzexu.xyz/SVQTD/ "SVQTD \| Singing Voice Quality and Technique Database (SVQTD) is a classical male singing dataset for describing classical tenor singing voices from vocal pedagogy point of view."))|Most relevant public-ish source for **interpretable vocal feedback**.|Classical male tenor only; YouTube audio; source-separation artifacts; aria/singer leakage; possible repeated singer at different stages but not controlled longitudinal data. ([Yanze Xu](https://yanzexu.xyz/SVQTD/ "SVQTD \| Singing Voice Quality and Technique Database (SVQTD) is a classical male singing dataset for describing classical tenor singing voices from vocal pedagogy point of view."))|**Yes.** Very good for small models.|
|**Lyra-SA**|Real karaoke-style singing assessment: 10 songs, 100 complete singing samples per song, total 1,000; includes singing audio, MIDI, lyrics, and listener ratings/metadata. ([Lyracobar](https://lyracobar.y.qq.com/singvoicedataset_en.html "天琴实验室 - QQ音乐"))|Application/request; CC BY-NC 4.0. Code availability appears minimal from the public repo. ([Lyracobar](https://lyracobar.y.qq.com/singvoicedataset_en.html "天琴实验室 - QQ音乐"))|Reference-conditioned karaoke scoring; song-level split experiments.|WeSing platform bias; ordinary-listener rating bias; phone-mic conditions; accompaniment/original-song leakage is explicitly noted. ([Lyracobar](https://lyracobar.y.qq.com/singvoicedataset_en.html "天琴实验室 - QQ音乐"))|**Yes.**|
|**PESnQ**|Expert ratings for singing quality parameters: intonation, rhythm, vibrato/expression, timbre/voice quality, articulation, dynamics/volume. ([Cambridge University Press & Assessment](https://www.cambridge.org/core/journals/apsipa-transactions-on-signal-and-information-processing/article/technical-framework-for-automatic-perceptual-evaluation-of-singing-quality/5F6AECB907FE842481D070850EDF1EFA "A technical framework for automatic perceptual evaluation of singing quality \| APSIPA Transactions on Signal and Information Processing \| Cambridge Core"))|GitHub includes audio dataset folder, subjective ground truths, Python scripts, ARFF files; designed for short monophonic clips with reference/test pairs. ([GitHub](https://github.com/chitralekha18/PESnQ_APSIPA2017 "GitHub - chitralekha18/PESnQ_APSIPA2017 · GitHub"))|Best old-school interpretable reference-based baseline.|Tiny scale; reference dependence; monophonic clean assumption; can reward reference conformity over artistry.|**Excellent. No GPU needed.**|
|**SingEval**|400 DAMP renditions: 4 songs × 100 singers, crowdsourced singing quality annotations. ([GitHub](https://github.com/chitralekha18/SingEval "GitHub - chitralekha18/SingEval: Curated and human annotated (singing quality score) subset of DAMP singing vocals dataset · GitHub"))|Annotation subset public; audio obtained separately through DAMP/Smule. ([GitHub](https://github.com/chitralekha18/SingEval "GitHub - chitralekha18/SingEval: Curated and human annotated (singing quality score) subset of DAMP singing vocals dataset · GitHub"))|Current singing-quality benchmark; no-reference and reference-conditioned scoring.|Only 4 songs; DAMP/Smule metadata includes singer IDs and social “love” counts, so identity/platform leakage is easy. ([OpenAIRE - Explore](https://explore.openaire.eu/search/dataset?pid=10.5281%2Fzenodo.2533417 "DAMP-1k: Digital Archive of Mobile Performances - Smule 100x10"))|**Yes.**|
|**AME430 / augmented explainable SQA**|Augmented positive/negative singing quality data; targets include overall quality and explainable components such as pitch correctness/rhythm correctness. The paper augments professional singing with negative samples due to lack of annotated bad singing. ([APSIPA](https://www.apsipa.org/proceedings/2021/pdfs/0000904.pdf?utm_source=chatgpt.com "Training Explainable Singing Quality Assessment Network with Augmented Data"))|GitHub code exists for “Towards Training Explainable Singing Quality Assessment Network with Augmented Data.” ([GitHub](https://github.com/AME430/Towards-Training-Explainable-Singing-Quality-Assessment-Network-with-Augmented-Data/tree/master?utm_source=chatgpt.com "AME430/Towards-Training-Explainable-Singing-Quality-Assessment ... - GitHub"))|Useful as a baseline or augmentation cautionary tale.|Synthetic “bad singing” can create artifacts; model may learn augmentation corruption rather than real novice errors.|**Yes.**|
|**VocalVerse / QwenFeat-Vocal-Score**|Open subset of about 1,000 high-proficiency singing examples from a larger 100k/10k screening pipeline; expert scores and text comments for timbre, breath, emotion, and technique. ([GitHub](https://github.com/CarlWangChina/QwenFeat-Vocal-Score "GitHub - CarlWangChina/QwenFeat-Vocal-Score: VocalVerse: A powerful vocal evaluation framework powered by the Qwen LLMs · GitHub"))|Hugging Face data/model resources and GitHub. ([Hugging Face](https://huggingface.co/karl-wang/QwenFeat-Vocal-Score "karl-wang/QwenFeat-Vocal-Score · Hugging Face"))|Multi-dimensional feedback and text-critique modeling.|Strong selection bias: released subset is top-end, not representative of beginners. QwenAudio-based scoring is not small-GPU if you run the full model.|**Dataset yes; full Qwen model no.** Use frozen smaller embeddings.|
|**TG-Critic**|Three-level labels: Awesome, Mediocre, Inferior; subjective ground truths for NUS48E and PESnQ-DS; algorithm outputs included. ([GitHub](https://github.com/YuejieGao/TG-CRITIC/blob/main/README.md "TG-CRITIC/README.md at main · YuejieGao/TG-CRITIC · GitHub"))|GitHub available. ([GitHub](https://github.com/YuejieGao/TG-CRITIC/blob/main/README.md "TG-CRITIC/README.md at main · YuejieGao/TG-CRITIC · GitHub"))|Reference-independent quality classification; baseline comparison.|Three classes collapse many musical dimensions; label construction and corpus size limit generality.|**Yes.**|
|**10KSinging**|Reported as about 9,756 songs / 190 singers with overall quality ratings; recent work reports results on 10KSinging for solo and accompanied singing. ([ISMIR 2024](https://ismir2024program.ismir.net/poster_89.html?utm_source=chatgpt.com "ISMIR 2024"))|Access appears request/unclear from public sources; I would not make it a dependency until you confirm access.|Potentially strong current-skill dataset if obtained.|Platform/rater bias, singer identity leakage, accompaniment, popularity, song difficulty.|**Yes if obtained; risky dependency.**|
|**Learn-by-Referencing / ranking-based ASA**|Ranking/metric-learning formulation: map more pleasant singing closer to a reference track using spectrogram/chroma and triplet-style learning. ([Zenodo](https://zenodo.org/records/5624579 "Learn by Referencing: Towards Deep Metric Learning for Singing Assessment"))|Paper open; code/data availability must be checked separately.|Pairwise/reference-conditioned ranking; useful alternative to absolute scores.|Reference similarity may punish stylistic individuality; anchor/reference choice matters.|**Yes.**|
|**MFFMOS / GOSMOS**|I could not verify a clearly public singing-specific dataset or benchmark under these exact names. Search results point more generally to multi-feature-fusion MOS ideas, not a stable public singing dataset. ([SSRN](https://api.ssrn.com/content/v1/authors/7565690/papers/rss?utm_source=chatgpt.com "SSRN RSS for author with id 7565690"))|Treat as unverified until you identify the exact paper/repo.|Possible baseline family, not a core data source.|High risk of acronym confusion.|Unknown.|
|**Nakano / MiruSinger / Waseda-AIST-Goto-Hiraga line**|Classic work on singing skill visualization/evaluation, including score-independent evaluation for unknown melodies and pitch-interval/vibrato-based criteria. ([AIST Staff](https://staff.aist.go.jp/t.nakano/SingingSkill/?utm_source=chatgpt.com "An Automatic Singing Skill Evaluation Method for Unknown Melodies - AIST"))|Mostly papers/demos, not modern public benchmark datasets.|Conceptual foundation for interpretable feedback and visual explanation.|Small/private legacy data; not directly comparable to modern SSL benchmarks.|**Yes conceptually; little/no training needed.**|

## What the survey says

The dataset landscape strongly favors **evaluation and feedback**, not **future prediction**. The best small-GPU research path is not to train a new audio foundation model, but to ask:

> Which small, interpretable, frozen-embedding systems predict singing ratings or feedback dimensions under leakage-controlled singer/song/system splits?

That is a paper-shaped question.

---

# Part 3: Researchable task formulations

## 1. No-reference current singing quality

**Task.** Given a singing recording, predict current perceived quality or MOS without a reference score.

**Why it matters.** This is the closest reformulation of “how skilled is this singer?” It is useful for search, ranking, self-practice, and dataset filtering.

**Datasets.** SingEval/DAMP, TG-Critic, Lyra-SA, 10KSinging if obtained, VocalVerse for high-end ratings. SingMOS/SingMOS-Pro are valid for generated-vocal MOS, but should not be framed as amateur-singer skill. ([arXiv](https://arxiv.org/html/2510.01812v4 "SingMOS-Pro: An Comprehensive Benchmark for Singing Quality Assessment"))

**Baselines.**

- Mean-by-song baseline.
    
- OpenSMILE/eGeMAPS + ridge/SVR/XGBoost.
    
- Pitch/vibrato/rhythm summary features + ridge.
    
- Frozen wav2vec2/HuBERT/MuQ/CLAP embeddings + linear/MLP head.
    
- SingMOS predictor zero-shot for MOS-like corpora. ([GitHub](https://github.com/South-Twilight/SingMOS "GitHub - South-Twilight/SingMOS: Officail repo for SingMOS-Pro (ICASSP 2026) · GitHub"))
    

**Metrics.**

- Pearson/Spearman/Kendall correlation.
    
- MAE/RMSE.
    
- Quadratic weighted kappa for ordinal labels.
    
- Calibration by song, singer, gender, language, and recording condition.
    

**Split strategy.**

- Leave-singer-out.
    
- Leave-song-out.
    
- If generated systems are involved, leave-system/model/dataset-out.
    
- Never report only random clip splits.
    

**Compute.** One GPU for embedding extraction; CPU training for ridge/SVR/XGBoost. Very feasible.

**Scientific risk.** Quality labels are noisy and multidimensional; high correlations may come from dataset artifacts.

**Product risk.** A single number is not actionable and can be unfair.

**Paper viability.** Good if you emphasize leakage-controlled evaluation, calibration, and error analysis.

---

## 2. Reference-conditioned karaoke pitch/rhythm/lyric scoring

**Task.** Given a singer recording plus reference audio, MIDI, score, or lyrics, predict pitch/rhythm/lyric correctness.

**Why it matters.** This is much more defensible than “skill.” It answers: “Did the singer match the intended melody and timing?”

**Datasets.** Lyra-SA, PESnQ, SingEval/DAMP with references, Learn-by-Referencing-style ranking. Lyra-SA includes MIDI and lyrics, and PESnQ is explicitly designed for reference/test singing comparisons. ([Lyracobar](https://lyracobar.y.qq.com/singvoicedataset_en.html "天琴实验室 - QQ音乐"))

**Baselines.**

- F0 extraction + dynamic time warping.
    
- Median-subtracted pitch contour and pitch-derivative features, following PESnQ’s motivation to avoid punishing key transposition too harshly. ([Cambridge University Press & Assessment](https://www.cambridge.org/core/journals/apsipa-transactions-on-signal-and-information-processing/article/technical-framework-for-automatic-perceptual-evaluation-of-singing-quality/5F6AECB907FE842481D070850EDF1EFA "A technical framework for automatic perceptual evaluation of singing quality | APSIPA Transactions on Signal and Information Processing | Cambridge Core"))
    
- Onset/beat deviation.
    
- Lyric/phoneme alignment score.
    
- Feature fusion with ridge/XGBoost.
    

**Metrics.**

- Correlation with human pitch/rhythm/overall scores.
    
- F0 error in cents.
    
- Voicing F1.
    
- Onset alignment error.
    
- Lyric alignment F1 or word/phoneme timing error.
    
- Song-normalized score correlation.
    

**Split strategy.**

- Leave-song-out is mandatory.
    
- Also report within-song ranking.
    
- Evaluate raw audio vs separated vocals to expose accompaniment leakage.
    

**Compute.** Mostly CPU; optional small GPU for separation or embeddings.

**Scientific risk.** Reference matching is not the same as musical quality.

**Product risk.** Can punish expressive deviation, riffs, transposition, rubato, and stylistic choices.

**Paper viability.** Good if framed as **reference-match assessment**, not holistic singing quality.

---

## 3. Singing technique or vocal-attribute recognition

**Task.** Detect or rate technique/attribute labels such as vibrato, breathy voice, falsetto, glissando, resonance, roughness, or throat openness.

**Why it matters.** This is more useful feedback than a global score. “Your vibrato is unstable” or “this segment is breathy” is more actionable than “72/100.”

**Datasets.** GTSinger, VocalSet, SVQTD. GTSinger provides six technique labels; VocalSet provides professional technique demonstrations; SVQTD provides pedagogy-like attribute labels for classical tenor singing. ([GitHub](https://github.com/AaronZ345/GTSinger "GitHub - AaronZ345/GTSinger: Dataset and code of GTSinger(NeurIPS 2024 Spotlight): A Global Multi-Technique Singing Corpus with Realistic Music Scores for All Singing Tasks · GitHub"))

**Baselines.**

- Acoustic features: F0 stability, vibrato rate/extent, spectral tilt, HNR, jitter/shimmer proxies, energy envelope.
    
- openSMILE + SVM/XGBoost.
    
- Frozen SSL embeddings + linear classifier.
    
- Small CQT CNN/CRNN.
    

**Metrics.**

- Macro-F1.
    
- Balanced accuracy.
    
- AUROC.
    
- Ordinal MAE for ordered SVQTD labels.
    
- Leave-singer-out and cross-corpus transfer.
    

**Split strategy.**

- Leave-singer-out.
    
- Leave-language-out for GTSinger.
    
- Leave-aria/song-out for SVQTD.
    
- Cross-dataset tests where labels overlap, e.g., breathy/vibrato.
    

**Compute.** Very small.

**Scientific risk.** Technique labels are genre- and pedagogy-dependent.

**Product risk.** A detected technique is not automatically “good” or “bad.” Avoid prescriptive vocal-health claims.

**Paper viability.** Strong if you focus on interpretability and cross-corpus generalization.

---

## 4. Multi-dimensional singing feedback

**Task.** Predict separate feedback dimensions: pitch, rhythm, timbre, breath, emotion, technique, vibrato, pronunciation, dynamics, etc.

**Why it matters.** This aligns best with your stated goal: useful feedback, not just a number.

**Datasets.**

- PESnQ: intonation, rhythm, vibrato/expression, voice quality, articulation, dynamics. ([Cambridge University Press & Assessment](https://www.cambridge.org/core/journals/apsipa-transactions-on-signal-and-information-processing/article/technical-framework-for-automatic-perceptual-evaluation-of-singing-quality/5F6AECB907FE842481D070850EDF1EFA "A technical framework for automatic perceptual evaluation of singing quality | APSIPA Transactions on Signal and Information Processing | Cambridge Core"))
    
- VocalVerse/QwenFeat-Vocal-Score: timbre, breath, emotion, technique, plus text comments. ([GitHub](https://github.com/CarlWangChina/QwenFeat-Vocal-Score "GitHub - CarlWangChina/QwenFeat-Vocal-Score: VocalVerse: A powerful vocal evaluation framework powered by the Qwen LLMs · GitHub"))
    
- SVQTD: resonance, placement, openness, roughness, vibrato. ([Yanze Xu](https://yanzexu.xyz/SVQTD/ "SVQTD | Singing Voice Quality and Technique Database (SVQTD) is a classical male singing dataset for describing classical tenor singing voices from vocal pedagogy point of view."))
    
- Acapella Evaluation: pitch, rhythm, vocal range, timbre, pronunciation, vibrato, dynamics, breath control, overall. ([Hugging Face](https://huggingface.co/datasets/ccmusic-database/acapella_evaluation "ccmusic-database/acapella · Datasets at Hugging Face"))
    

**Baselines.**

- Two-stage interpretable model: acoustic features → dimension scores → overall score.
    
- Multi-task frozen-embedding model.
    
- Feature-ablation model: pitch-only, rhythm-only, timbre-only, full.
    
- SHAP/permutation importance for feedback explanation.
    

**Metrics.**

- Per-dimension Pearson/Spearman.
    
- Per-dimension MAE.
    
- Rank correlation for “what to improve first.”
    
- Reliability by label dimension.
    
- Human usefulness study if you can afford a small annotation round.
    

**Split strategy.**

- Leave-singer-out and leave-song-out.
    
- Do not mix comments or expert text into audio-only prediction unless you explicitly study multimodal supervision.
    
- Report per-dimension failure, not just average.
    

**Compute.** Feasible with frozen embeddings and small heads.

**Scientific risk.** Dimensions are correlated and inconsistently defined across datasets.

**Product risk.** Bad feedback can be worse than no feedback. You need uncertainty estimates and cautious language.

**Paper viability.** Best overall fit for your preferences.

---

## 5. Pairwise improvement ranking across takes

**Task.** Given two takes, predict which one is better or more improved.

**Why it matters.** Ranking is often more reliable than absolute scoring, and it matches real coaching practice: “Take B is better than Take A because pitch stability improved.”

**Datasets.** Learn-by-Referencing formulation; DAMP/SingEval for pairwise within-song ranking; 10KSinging if accessible. True longitudinal improvement would still require new data. ([Zenodo](https://zenodo.org/records/5624579 "Learn by Referencing: Towards Deep Metric Learning for Singing Assessment"))

**Baselines.**

- Pairwise logistic regression on feature differences.
    
- Bradley–Terry model.
    
- RankNet/LightGBM ranker.
    
- Reference-match delta: pitch/rhythm/lyric error differences.
    
- Triplet metric learning with reference anchor.
    

**Metrics.**

- Pairwise accuracy.
    
- AUC.
    
- Kendall tau.
    
- NDCG.
    
- “Best take selected” accuracy.
    

**Split strategy.**

- Leave-singer-song-out.
    
- Pair only comparable songs or normalize by song difficulty.
    
- Do not call it “improvement” unless the pair has real temporal order and controlled practice context.
    

**Compute.** Very small.

**Scientific risk.** Public data usually supports “better/worse,” not “improved over time.”

**Product risk.** Users may interpret rank feedback as personal progress; avoid that unless you have longitudinal evidence.

**Paper viability.** Interesting but riskier. Best as a second experiment, not the first.

---

# Part 4: Karaoke scoring traps

## 1. Punishing expressive deviation

Karaoke scoring can penalize exactly what good singers do: rubato, swing timing, scoops, bends, riffs, delayed consonants, altered phrasing, ornamentation, key transposition, or intentional vibrato. PESnQ explicitly notes that simple pitch-contour distance can punish key transposition and therefore uses median-subtracted and derivative pitch representations to reduce that problem. ([Cambridge University Press & Assessment](https://www.cambridge.org/core/journals/apsipa-transactions-on-signal-and-information-processing/article/technical-framework-for-automatic-perceptual-evaluation-of-singing-quality/5F6AECB907FE842481D070850EDF1EFA "A technical framework for automatic perceptual evaluation of singing quality | APSIPA Transactions on Signal and Information Processing | Cambridge Core"))

**Control:** report separate “reference adherence” and “expressive deviation” features. Do not collapse them into “quality.”

## 2. Accompaniment and source-separation leakage

Lyra-SA states that real phone recordings may include accompaniment or original-song leakage. SVQTD also used source separation and explicitly mentions separation artifacts during annotation filtering. ([Lyracobar](https://lyracobar.y.qq.com/singvoicedataset_en.html "天琴实验室 - QQ音乐"))

A model may learn:

- accompaniment loudness,
    
- backing-vocal contamination,
    
- source-separation artifacts,
    
- platform compression,
    
- recording device quality,
    

rather than singing skill.

**Control:** evaluate raw audio, separated vocal, and accompaniment-only baselines. If accompaniment-only predicts score, your task is contaminated.

## 3. Popularity and platform bias

Lyra-SA songs were selected from high-volume WeSing usage, and DAMP/Smule metadata contains singer IDs and social “love” counts. ([Lyracobar](https://lyracobar.y.qq.com/singvoicedataset_en.html "天琴实验室 - QQ音乐"))

A model can learn platform taste, song popularity, gender/age perception, or fan behavior.

**Control:** remove social metadata; split by song and singer; report subgroup calibration.

## 4. Song difficulty

A raw score across different songs is not fair. A simple song with narrow range and slow tempo is easier than a high-range, ornamented song.

**Control:** use song-normalized scores, difficulty offsets, leave-song-out tests, and within-song rankings.

## 5. Singer identity leakage

Many singing datasets have many clips per singer but few total singers. GTSinger has 20 professional singers; VocalSet has 20 singers; DAMP contains singer IDs. ([GitHub](https://github.com/AaronZ345/GTSinger "GitHub - AaronZ345/GTSinger: Dataset and code of GTSinger(NeurIPS 2024 Spotlight): A Global Multi-Technique Singing Corpus with Realistic Music Scores for All Singing Tasks · GitHub"))

Random splits can let the model recognize the singer and memorize their average score.

**Control:** leave-singer-out, singer-adversarial checks, and a singer-ID classifier audit.

## 6. Generated-vocal versus real-amateur domain shift

SingMOS/SingMOS-Pro are valuable, but they focus heavily on generated singing from SVS/SVC/SVR systems plus real vocals, not ordinary amateurs recording into phones. ([arXiv](https://arxiv.org/html/2510.01812v4 "SingMOS-Pro: An Comprehensive Benchmark for Singing Quality Assessment"))

A model trained there may detect vocoder artifacts, synthesis errors, or system identity rather than human skill.

**Control:** cross-test on real karaoke/amateur data and report the domain gap.

## 7. Pitch accuracy is not musical quality

PESnQ’s own framing includes intonation, rhythm, vibrato, timbre/voice quality, articulation, dynamics, and vocal clarity. ([Cambridge University Press & Assessment](https://www.cambridge.org/core/journals/apsipa-transactions-on-signal-and-information-processing/article/technical-framework-for-automatic-perceptual-evaluation-of-singing-quality/5F6AECB907FE842481D070850EDF1EFA "A technical framework for automatic perceptual evaluation of singing quality | APSIPA Transactions on Signal and Information Processing | Cambridge Core"))

A singer can be pitch-accurate and boring, tense, nasal, breathy, rhythmically stiff, emotionally flat, or stylistically inappropriate. Another singer can deviate from the score and sound better.

**Control:** never present pitch/rhythm match as holistic quality. Use multi-dimensional outputs.

---

# Part 5: Three small-GPU experiments

## Experiment 1 — Safest benchmark paper

### Title

**Leakage-Robust Small-Model Benchmarking for Singing MOS Prediction**

### Data

- SingMOS-v1.
    
- SingMOS-Pro.
    
- Optional: cross-evaluate on TG-Critic or SingEval if licenses/audio access allow.
    

SingMOS-Pro is ideal because it has official splits, score files, system metadata, and pretrained baselines. ([Hugging Face](https://huggingface.co/datasets/TangRain/SingMOS-Pro "TangRain/SingMOS-Pro · Datasets at Hugging Face"))

### Features

- Frozen wav2vec2-base / HuBERT / MuQ-style embeddings.
    
- openSMILE/eGeMAPS.
    
- Pitch histogram, F0 stability, voicing ratio, vibrato stats.
    
- Duration and loudness features as diagnostic controls, not as final trusted skill features.
    

### Model

- Ridge regression.
    
- SVR.
    
- XGBoost/LightGBM.
    
- Small 2-layer MLP.
    
- Compare to official SingMOS predictor.
    

### Metrics

- Utterance-level Pearson/Spearman.
    
- System-level Pearson/Spearman.
    
- MAE/RMSE.
    
- Calibration by dataset, system type, language, model family.
    
- Performance drop from official/random split to held-out-system split.
    

### Controls

- Official split.
    
- Held-out generating system/model split.
    
- Held-out source dataset split.
    
- Generated-vocal vs real-vocal subset if metadata allows.
    
- “System ID only” or “dataset ID only” baseline to quantify leakage.
    

### Expected compute

One GPU for embedding extraction. Small heads train on CPU. This is the safest experiment for a few GPUs.

### Minimum publishable result

A publishable result does **not** require beating a large model. A good paper could show:

- small frozen embeddings are competitive under official splits;
    
- performance collapses under held-out-system splits;
    
- pitch/timbre/interpretable features explain certain failure modes;
    
- existing MOS benchmarks overestimate real generalization.
    

That is a real benchmark/evaluation paper.

### Failure interpretation

If performance is high only on official/random splits and low on held-out-system splits, the conclusion is still valuable: **current singing MOS benchmarks are vulnerable to system/domain leakage.**

---

## Experiment 2 — More novel feedback experiment

### Title

**Interpretable Singing Technique and Vocal-Attribute Feedback with Cross-Corpus Generalization**

### Data

- GTSinger for six professional technique labels.
    
- VocalSet for technique/vowel demonstrations.
    
- SVQTD for classical pedagogical attributes.
    
- Optional: VocalVerse for timbre/breath/emotion/technique score probing.
    

GTSinger and VocalSet provide technique labels; SVQTD provides more pedagogy-like attributes such as resonance, placement, openness, roughness, and vibrato quality. ([GitHub](https://github.com/AaronZ345/GTSinger "GitHub - AaronZ345/GTSinger: Dataset and code of GTSinger(NeurIPS 2024 Spotlight): A Global Multi-Technique Singing Corpus with Realistic Music Scores for All Singing Tasks · GitHub"))

### Features

Interpretable features:

- F0 stability.
    
- Vibrato rate and extent.
    
- Spectral tilt.
    
- Harmonic-to-noise ratio.
    
- Energy-envelope smoothness.
    
- Breathiness proxies.
    
- Jitter/shimmer-like perturbation proxies.
    
- Formant/resonance proxies where reliable.
    

Frozen embeddings:

- wav2vec2/HuBERT/MuQ.
    
- Segment-level mean/std pooling.
    

### Model

- Logistic regression / ordinal logistic regression.
    
- XGBoost or random forest.
    
- Small multi-task MLP.
    
- SHAP/permutation importance for explanation.
    

### Metrics

- Macro-F1.
    
- Balanced accuracy.
    
- AUROC.
    
- Ordinal MAE for SVQTD attributes.
    
- Leave-singer-out performance.
    
- Cross-corpus transfer where labels overlap: vibrato, breathy, falsetto/head-like production.
    

### Controls

- Singer-ID classifier audit.
    
- Leave-singer-out split.
    
- Leave-language-out split for GTSinger.
    
- Equal segment duration.
    
- Compare interpretable-only vs embedding-only vs combined.
    
- Report when embeddings improve accuracy but reduce interpretability/generalization.
    

### Expected compute

Very feasible. Embedding extraction may take hours; model training is lightweight.

### Minimum publishable result

A strong result would be:

- a unified taxonomy mapping overlapping vocal technique labels across datasets;
    
- evidence that interpretable acoustic features generalize better across corpora than frozen embeddings for some attributes;
    
- reliable uncertainty/error analysis showing where feedback should be withheld.
    

This is the best match to your preference for **useful feedback**.

### Failure interpretation

If labels do not transfer, that is still useful: it means vocal-technique labels are corpus-, genre-, and pedagogy-specific. The paper becomes a **cross-dataset negative result and taxonomy paper**, which can still be valuable.

---

## Experiment 3 — Risky but interesting experiment

### Title

**Pairwise Best-Take Ranking for Singing Practice Without Longitudinal Claims**

### Data

- SingEval/DAMP for within-song comparisons.
    
- Lyra-SA if obtained.
    
- 10KSinging if access is confirmed.
    
- Optional small self-collected pilot with repeated takes and expert pairwise judgments.
    

DAMP/SingEval gives multiple singers and songs, but it does not by itself prove training improvement. Learn-by-Referencing gives a precedent for ranking/metric learning against a reference. ([GitHub](https://github.com/chitralekha18/SingEval "GitHub - chitralekha18/SingEval: Curated and human annotated (singing quality score) subset of DAMP singing vocals dataset · GitHub"))

### Features

- Reference-match pitch error.
    
- Rhythm/onset deviation.
    
- Lyric/phoneme alignment if available.
    
- F0 stability and vibrato features.
    
- Frozen SSL embeddings.
    
- Predicted sub-dimensions from Experiment 2.
    

### Model

- Bradley–Terry model.
    
- Pairwise logistic regression.
    
- LightGBM ranker.
    
- Small RankNet.
    
- Triplet loss with reference anchor only if you have enough data.
    

### Metrics

- Pairwise accuracy.
    
- AUC.
    
- Kendall tau.
    
- NDCG.
    
- Best-take top-1 accuracy.
    
- Correlation between predicted improvement reason and human subscore difference.
    

### Controls

- Compare only within the same song unless using difficulty normalization.
    
- Leave-singer-out.
    
- Leave-song-out.
    
- Remove social/popularity metadata.
    
- Pitch-only and rhythm-only baselines.
    
- Shuffle pair order to ensure no accidental temporal leakage.
    

### Expected compute

Small. Most of the work is data cleaning and evaluation design.

### Minimum publishable result

The minimum publishable claim is:

> “Pairwise ranking is more stable and actionable than absolute score prediction for singing practice, but public data supports best-take ranking rather than true longitudinal improvement.”

That is defensible.

### Failure interpretation

If ranking fails across songs or singers, the conclusion is: **public karaoke data does not support general improvement prediction without new longitudinal annotation.** That would directly answer your original question and justify not pursuing the future-voice idea.

---

# Part 6: Final recommendation

## Should this be your main thesis direction?

**Not as originally stated.** “Future professional voice” is a no-go as a supervised learning target.

**As a reformulated direction, singing skill/quality evaluation is viable but should be a backup or secondary thesis direction**, unless you focus it very tightly around:

> **interpretable, leakage-controlled, multi-dimensional singing feedback using small models.**

That version can become a real paper. A generic “singing score predictor” is closer to a product prototype.

## Real-paper version versus product-only version

|Direction|Paper potential|Product potential|Risk|
|---|--:|--:|---|
|Future professional voice prediction|Very low|Medium as speculative voice-conversion demo|Invalid target; no ground truth|
|Single singing-quality score|Medium-low|High|Too product-like; unfair/opaque|
|Karaoke pitch/rhythm scoring|Medium|High|Easy to misrepresent as quality|
|Technique/attribute feedback|High|Medium-high|Label taxonomy and domain transfer are hard|
|Multi-dimensional feedback benchmark|High|High|Needs careful uncertainty and split design|
|Pairwise best-take ranking|Medium-high|High|Public data may not be truly longitudinal|

## Comparison with the speech/singing timbre-shift idea

Based on your constraints, the **speech/singing timbre-shift idea is probably stronger as a main thesis direction** if it has a clean research question. It is likely more scientifically novel and less entangled with subjective labels. GTSinger’s paired speech and singing data could even connect the two directions, because it includes 16.16 hours of paired speech along with professional singing and technique labels. ([GitHub](https://github.com/AaronZ345/GTSinger "GitHub - AaronZ345/GTSinger: Dataset and code of GTSinger(NeurIPS 2024 Spotlight): A Global Multi-Technique Singing Corpus with Realistic Music Scores for All Singing Tasks · GitHub"))

Singing quality evaluation is more immediately useful, but messier:

- labels are subjective;
    
- datasets are small or biased;
    
- leakage is severe;
    
- claims are easy to overstate;
    
- “feedback” requires more than a score.
    

So my recommendation is:

**Main thesis:** speech/singing timbre-shift, if you can define a precise signal-processing or representation-learning question.  
**Backup / side paper:** singing feedback evaluation.  
**Do not pursue:** future professional voice prediction.

## Go / no-go

**Go** for:

> “Small-model, interpretable, leakage-robust singing quality and feedback assessment.”

**No-go** for:

> “Predict future professional voice” or “predict latent singing potential.”

---

# 30-day plan

## Days 1–5: Lock the task and data

- Download or request SingMOS-v1, SingMOS-Pro, VocalSet, GTSinger, SVQTD, Lyra-SA, SingEval/DAMP as licenses allow.
    
- Confirm 10KSinging access; do not depend on it.
    
- Create a claim sheet with forbidden claims and allowed claims.
    
- Decide whether the first paper target is MOS benchmark, technique feedback, or pairwise ranking.
    

**Milestone:** one-page task definition: “current rating/feedback prediction, not future potential.”

## Days 6–10: Build the safest baseline

- Extract frozen embeddings from SingMOS-v1/Pro.
    
- Train ridge/SVR/XGBoost/MLP MOS predictors.
    
- Reproduce official split results.
    
- Add openSMILE and pitch/vibrato feature baselines.
    
- Compare to SingMOS pretrained predictor.
    

**Milestone:** first table of MOS prediction results.

## Days 11–15: Run leakage stress tests

- Held-out system split.
    
- Held-out source dataset split.
    
- Held-out language split if possible.
    
- System-ID-only and dataset-ID-only baselines.
    
- Error analysis by generated-vocal system type.
    

**Milestone:** answer whether the benchmark is measuring general singing quality or system/domain cues.

## Days 16–20: Build feedback/technique prototype

- Train leave-singer-out classifiers on GTSinger and VocalSet.
    
- Train SVQTD attribute models if access is approved.
    
- Compare interpretable acoustic features vs frozen embeddings.
    
- Generate explanations for vibrato, breathiness, resonance-like attributes.
    

**Milestone:** first actionable-feedback model, even if imperfect.

## Days 21–25: Cross-dataset and usefulness checks

- Test technique transfer where labels overlap.
    
- Evaluate whether feedback dimensions predict overall quality better than opaque embeddings.
    
- Identify failure modes where feedback should be withheld.
    
- Write examples of good and bad feedback language.
    

**Milestone:** decide whether the paper is a benchmark paper, feedback paper, or negative-result leakage paper.

## Days 26–30: Paper decision

Write a 4-page internal draft with:

1. task definition;
    
2. dataset audit;
    
3. split protocol;
    
4. baseline results;
    
5. leakage findings;
    
6. feedback examples;
    
7. limitations.
    

At day 30, use this decision rule:

- **Proceed as paper** if you have either competitive small-model results under strict splits, or a clear leakage/domain-shift finding that changes how these datasets should be evaluated.
    
- **Keep as product prototype** if performance is useful only under random splits or within one platform/song set.
    
- **Drop the direction** if neither MOS nor feedback dimensions generalize under leave-singer/leave-song/system splits.
    

My strict recommendation: **use this as a backup/side-paper direction, not the main thesis direction, unless the first 30 days reveal a strong leakage-robust feedback result.**