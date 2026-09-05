I want to understand the literature around speech/singing voice representation,
conversion, disentanglement, and evaluation. Please do citation chaining from
these anchors:

GTSinger;
Seed-VC;
FACodec / NaturalSpeech 3;
SVCC 2025;
LoIN;
AdaIN-VC;
FreeCodec;
MSR-Codec;
speaker leakage probing in SSL speech representations;
SingMOS-Pro or other singing quality assessment papers.
My constraints:

I only have a few GPUs.
I need papers that help design a feasible experiment, not just impressively
large models.
I care about public code/data and experimental patterns.
For each anchor:

summarize the anchor paper/system;
list the most relevant cited works;
list the most relevant citing works;
identify the experiment-design pattern it teaches;
identify what it already solves;
identify what gap remains;
say whether it is experiment-critical, baseline candidate, related-work
context, or not useful.
Then synthesize:

the top 20 papers I should read;
the top 8 experiment patterns I should imitate;
the strongest novelty gaps left;
the papers with usable public code/data;
the papers that are important but too expensive or unreproducible for me.
Please cite sources and distinguish verified facts from speculation.


# pro
## How I’m separating evidence from judgement

**Verified facts** below are source-backed. **Experiment pattern, remaining gap, and usefulness classification** are my research-design judgements from those facts, not claims the papers necessarily make.

For a few-GPU project, the practical center of gravity is **SVCC 2025 + GTSinger/SVCC data + Seed-VC + FACodec/MSR-style stream probing + SingMOS/Versa + speaker-leakage probes**. NaturalSpeech 3, Vevo-like systems, and large challenge submissions are valuable mainly as **design signals**, not as models to retrain.

---

# Anchor-by-anchor citation chain

## 1. GTSinger

**Verified summary.** GTSinger is a NeurIPS 2024 Spotlight singing corpus designed for multiple singing tasks. It contains **80.59 hours** of studio-quality singing, **20 professional singers**, **9 languages**, phoneme-level annotations for **six singing techniques**, realistic scores, manual phoneme-to-audio alignments, global style labels, and **16.16 hours of paired speech**. The paper reports four benchmark tasks: technique-controllable SVS, technique recognition, style transfer, and speech-to-singing conversion. The authors also provide dataset and benchmark code. ([arXiv](https://arxiv.org/abs/2409.13832 "https://arxiv.org/abs/2409.13832")) The GitHub repo says the full dataset, code, and processed data were released. ([GitHub](https://github.com/AaronZ345/GTSinger "https://github.com/AaronZ345/GTSinger"))

**Relevant cited/upstream works to chain.** The most useful upstream dataset/model works are **M4Singer** for multi-style Mandarin singing with musical scores, **SingStyle111** for multilingual style-transfer data, and **DiffSinger** for score-conditioned diffusion SVS. M4Singer’s repo describes a multi-style, multi-singer, score-provided Mandarin corpus and says it is available for free research use. ([GitHub](https://github.com/M4Singer/M4Singer "https://github.com/M4Singer/M4Singer")) SingStyle111 has **111 songs**, **8 professional singers**, **12.8 hours**, English/Chinese/Italian coverage, phrase segmentation, lyrics, MIDI, scores, and phoneme alignments. ([Zenodo](https://zenodo.org/records/10265401 "https://zenodo.org/records/10265401")) DiffSinger has official code for shallow-diffusion singing voice synthesis. ([GitHub](https://github.com/MoonInTheRiver/DiffSinger "https://github.com/MoonInTheRiver/DiffSinger"))

**Relevant citing/downstream works.** The most important verified downstream use is **SVCC 2025**, which used a GTSinger subset for singing style conversion, with seven styles including breathy, falsetto, mixed voice, pharyngeal, glissando, vibrato, and control. ([arXiv](https://arxiv.org/html/2509.15629v1 "https://arxiv.org/html/2509.15629v1")) The SVCC dataset card states that the challenge subset is subject to GTSinger license terms and that full GTSinger was not allowed for challenge training to avoid leakage. ([Hugging Face](https://huggingface.co/datasets/lestervioleta/svcc2025 "https://huggingface.co/datasets/lestervioleta/svcc2025"))

**Experiment-design pattern it teaches.** Build a benchmark around **controlled paired phrases**: same or comparable phrase, multiple styles/techniques, held-out singer, aligned score/phoneme information, and task-specific labels. For your project, this means the cleanest experiment is not “train on all singing”; it is “hold out singer/style and test disentanglement.”

**What it already solves.** It solves the “where do I get annotated singing technique/style data?” problem better than older singing corpora.

**Gap that remains.** GTSinger supplies the data; it does not by itself solve **source singer leakage**, **target style transfer**, or **dynamic style modeling** such as vibrato/glissando/breathy transitions. SVCC 2025’s results show those remain open.

**Usefulness.** **Experiment-critical** as a data/design anchor. For strict SVCC comparability, use the SVCC subset/rules rather than training on full GTSinger.

---

## 2. Seed-VC

**Verified summary.** Seed-VC is a zero-shot VC framework using a diffusion transformer. Its core ideas are an **external timbre shifter during training** to perturb source timbre and reduce leakage, plus use of the **entire reference speech context** rather than a single target-speaker vector. It extends to zero-shot singing voice conversion with **F0 conditioning**, and the paper says code and pretrained models were released. ([arXiv](https://arxiv.org/html/2411.09943v1 "https://arxiv.org/html/2411.09943v1")) The paper explicitly frames the problem as timbre leakage in SSL/ASR/PPG content features, where stronger bottlenecks such as k-means can reduce leakage but hurt intelligibility/WER. ([arXiv](https://arxiv.org/html/2411.09943v1 "https://arxiv.org/html/2411.09943v1"))

**Relevant cited/upstream works.** Follow **HuBERT**, **wav2vec 2.0**, ASR/PPG-based VC, bottleneck/k-means methods, FreeVC/ContentVec-style SSL VC, and OpenVoice/CosyVoice as practical zero-shot VC comparators. Seed-VC itself names SSL models such as HuBERT/wav2vec and PPG/ASR extraction as common content representations that still leak timbre. ([arXiv](https://arxiv.org/html/2411.09943v1 "https://arxiv.org/html/2411.09943v1"))

**Relevant citing/downstream works.** SVCC 2025 system **S5** was based on open-sourced SeedVC and added a **Residual Style Adaptor** to capture articulation and vocal techniques; it used about **10k hours of speech** and **500 hours of singing**. ([arXiv](https://arxiv.org/html/2509.15629v1 "https://arxiv.org/html/2509.15629v1")) The Seed-VC repo supports zero-shot VC, real-time VC, and zero-shot SVC, and says it can clone from a **1–30 second** reference without training; it also advertises very low-data fine-tuning, with a minimum of one utterance and fast T4 training. ([GitHub](https://github.com/Plachtaa/seed-vc "https://github.com/Plachtaa/seed-vc"))

**Experiment-design pattern it teaches.** Use **training/inference mismatch attacks**: perturb source timbre during training so the model cannot rely on leaked source identity, then condition on richer reference context. For a few-GPU experiment, imitate this with a **small adaptor or augmentation**, not full retraining.

**What it already solves.** It is a strong public baseline for zero-shot speech VC and a usable singing conversion baseline.

**Gap that remains.** It mainly targets timbre conversion. SVCC shows style conversion, especially dynamic singing style, remains hard even for SeedVC-derived systems.

**Usefulness.** **Baseline candidate / experiment-critical baseline**, especially because pretrained inference and small fine-tuning are feasible.

---

## 3. FACodec / NaturalSpeech 3

**Verified summary.** NaturalSpeech 3 uses **FACodec**, a factorized neural codec with vector quantization subspaces for **content, prosody, timbre, and acoustic details**, then generates each factor with a factorized diffusion model. ([arXiv](https://arxiv.org/html/2403.03100v3 "https://arxiv.org/html/2403.03100v3")) The paper reports scaling experiments up to **1B parameters** and **200k hours** of training data, with improved WER and speaker similarity as scale increases. ([arXiv](https://arxiv.org/html/2403.03100v3 "https://arxiv.org/html/2403.03100v3")) The FACodec repo is usable as a standalone codec: it provides install instructions, a test script, and pretrained checkpoints from Hugging Face; the repo says FACodec decomposes speech into content, prosody, timbre, and acoustic-detail subspaces. ([GitHub](https://github.com/lifeiteng/naturalspeech3_facodec "https://github.com/lifeiteng/naturalspeech3_facodec"))

**Relevant cited/upstream works.** Follow **SoundStream**, **EnCodec**, **DAC**, **VALL-E**, **NaturalSpeech 2**, **SpeechTokenizer**, **AutoVC**, **NANSY-like disentanglement**, HuBERT/wav2vec-style SSL, and codec language models.

**Relevant citing/downstream works.** FreeCodec explicitly compares itself against supervised FACodec and notes that FACodec uses phone/F0/speaker labels and operates at a higher bitrate. “Exploring Disentangled Neural Speech Codecs” also lists FACodec as a major disentangled codec reference. ([arXiv](https://arxiv.org/html/2508.08399v1 "Exploring Disentangled Neural Speech Codecs from Self-Supervised Representations"))

**Experiment-design pattern it teaches.** Treat representation learning as **stream factorization**: content stream, prosody stream, timbre stream, residual/detail stream. Then evaluate by reconstruction, stream swapping, leakage probing, and downstream conversion.

**What it already solves.** It gives a clean, reusable implementation of factorized speech coding.

**Gap that remains.** Full NaturalSpeech 3 is not few-GPU reproducible; the scale result depends on 500M–1B models and up to 200k hours. The public FACodec is useful, but singing-specific disentanglement is not solved by the paper.

**Usefulness.** **FACodec: baseline candidate / experiment-critical representation.** **NaturalSpeech 3 full TTS system: related-work context, too expensive to reproduce.**

---

## 4. SVCC 2025

**Verified summary.** SVCC 2025 shifted the challenge from singer-identity conversion toward **singing style conversion**, introduced two tasks, released baselines and a dedicated dataset, and evaluated **26 systems** with large-scale listening tests and objective metrics. Top systems reached ground-truth-like singer identity in some cases, but style and naturalness remained below ground truth, especially for breathy, glissando, and vibrato styles. ([arXiv](https://arxiv.org/html/2509.15629v1 "https://arxiv.org/html/2509.15629v1")) Task 1 is in-domain style conversion for a seen singer; Task 2 is zero-shot style conversion for an unseen singer, explicitly testing disentanglement of singer identity from singing style. ([arXiv](https://arxiv.org/html/2509.15629v1 "https://arxiv.org/html/2509.15629v1"))

**Relevant cited works.** The references include the VCC challenge series, SVCC 2023, GTSinger, SingStyle111, Serenade, SeedVC, Vevo2/Vevo1.5, VERSA, ContentVec, and SVC systems from prior challenges. ([arXiv](https://arxiv.org/html/2509.15629v1 "https://arxiv.org/html/2509.15629v1"))

**Relevant citing/downstream works.** It is too recent for a mature citation graph, but the dataset page releases baseline/system samples and says papers from other systems will be added as they are released. ([Hugging Face](https://huggingface.co/datasets/lestervioleta/svcc2025 "https://huggingface.co/datasets/lestervioleta/svcc2025"))

**Experiment-design pattern it teaches.** Copy the **two-task split** and evaluation stack: in-domain seen-singer style conversion; zero-shot unseen-singer style conversion; MOS for naturalness; XAB for style similarity; singer identity similarity with multiple references; objective metrics with correlation analysis. The challenge used MOS, XAB, singer-identity similarity, 480 ratings/system, and VERSA-based objective tests. ([arXiv](https://arxiv.org/html/2509.15629v1 "https://arxiv.org/html/2509.15629v1"))

**What it already solves.** It gives the cleanest public task definition for singing **style** conversion rather than just singer conversion.

**Gap that remains.** No submitted system reached ground-truth naturalness/style similarity; breathy, glissando, and vibrato had especially low style accuracy, which the paper attributes to dynamic components such as time-varying noise and F0 modulation/drift. ([arXiv](https://arxiv.org/html/2509.15629v1 "https://arxiv.org/html/2509.15629v1"))

**Usefulness.** **Experiment-critical.** This is the benchmark I would organize your project around.

---

## 5. LoIN

**Verified summary.** LoIN is “Locality-Based Instance Normalization” for voice conversion. The paper argues that global instance-normalization statistics may not be locally consistent, causing incomplete feature decoupling. LoIN computes normalization statistics from randomly selected local frames, is lightweight, less computationally intensive, and transferable to IN-driven VC methods. ([ISCA Archive](https://www.isca-archive.org/interspeech_2023/gu23b_interspeech.html "https://www.isca-archive.org/interspeech_2023/gu23b_interspeech.html")) Its GitHub repo is public under MIT license, but it is small: 5 commits, no releases, and minimal project surface. ([GitHub](https://github.com/BrightGu/LoINVC "GitHub - BrightGu/LoINVC: Robust Feature Decoupling in Voice Conversion by using Locality-Based Instance Normalization · GitHub"))

**Relevant cited/upstream works.** The key upstream line is **AdaIN/IN-based VC**, plus AutoVC-style bottleneck VC and older feature-decoupling systems.

**Relevant citing/downstream works.** I did not verify a strong downstream citation trail. Treat it as a method idea rather than a field-defining anchor.

**Experiment-design pattern it teaches.** A cheap ablation: compare **global IN vs local/random-frame IN** inside a fixed VC architecture, then measure singer leakage, content preservation, and conversion quality.

**What it already solves.** It gives a low-cost normalization intervention for better feature decoupling.

**Gap that remains.** It is speech-VC-oriented, not singing-style-oriented, and the public repo is too minimal to be a strong modern baseline alone.

**Usefulness.** **Baseline candidate / ablation idea**, not a main baseline.

---

## 6. AdaIN-VC

**Verified summary.** AdaIN-VC is a one-shot voice conversion method that separates speaker and content representations with instance normalization. It can convert between unseen speakers using one source and one target utterance, and the paper reports objective and subjective target-similarity results. ([ISCA Archive](https://www.isca-archive.org/interspeech_2019/chou19_interspeech.html "https://www.isca-archive.org/interspeech_2019/chou19_interspeech.html")) The official repo provides code, pretrained model links, normalization parameters, and preprocessing support for VCTK/LibriTTS; the paper experiments were done on VCTK. ([GitHub](https://github.com/wangxiii/One-Shot-VC "GitHub - wangxiii/One-Shot-VC · GitHub"))

**Relevant cited/upstream works.** AutoVC, nonparallel VC, style transfer via instance normalization, speaker/content bottlenecks, and unsupervised speaker representation learning.

**Relevant citing/downstream works.** AGAIN-VC directly extends the idea with activation guidance and adaptive instance normalization to improve the quality/speaker-similarity tradeoff. ([arXiv](https://arxiv.org/abs/2011.00316 "https://arxiv.org/abs/2011.00316")) LoIN is another IN-family descendant. ([ISCA Archive](https://www.isca-archive.org/interspeech_2023/gu23b_interspeech.html "https://www.isca-archive.org/interspeech_2023/gu23b_interspeech.html"))

**Experiment-design pattern it teaches.** Use **normalization as a content bottleneck** and condition a decoder on target speaker/style. This is useful as a small, interpretable baseline where failure modes are easy to inspect.

**What it already solves.** It gives a clean one-shot VC baseline requiring much less compute than diffusion/codec LMs.

**Gap that remains.** Audio quality and generality are below modern pretrained systems; it does not address singing style, melody preservation, or dynamic vocal techniques.

**Usefulness.** **Baseline candidate / related-work context.** Good for ablations, not as your strongest system.

---

## 7. FreeCodec

**Verified summary.** FreeCodec is a self-supervised disentangled neural speech codec for reconstruction and zero-shot TTS/VC-like disentanglement. It uses distinct frame-level encoders and dedicated quantizers to separate intrinsic speech properties, aiming for better coding efficiency with **57 tokens**. It explicitly positions itself against supervised FACodec and claims stronger ultra-low-bitrate disentanglement; in a VC-style evaluation, FreeCodec-v3 at **0.45 kbps** reports better speaker similarity than compared baselines and lower WER than several alternatives.

**Relevant cited/upstream works.** FACodec, TiCodec, SingleCodec, EnCodec, DAC, WavTokenizer, semantic codecs, VQ-VAE/RVQ codecs, SSL-factorized representations.

**Relevant citing/downstream works.** Recent codec work such as DisCo-Speech frames standard codec timbre/prosody entanglement as a bottleneck and proposes a disentangled codec with content/prosody/timbre subspaces. ([arXiv](https://arxiv.org/abs/2512.13251 "[2512.13251] DisCo-Speech: Controllable Zero-Shot Speech Generation with A Disentangled Speech Codec")) Spark-TTS and other codec-LM systems are adjacent, although Spark-TTS itself is more of an LLM-TTS inference system than a FreeCodec reproduction. ([GitHub](https://github.com/sparkaudio/spark-tts "GitHub - SparkAudio/Spark-TTS: Spark-TTS Inference Code · GitHub"))

**Experiment-design pattern it teaches.** Evaluate codecs not only by reconstruction but by **attribute disentanglement under stream replacement**: source content + target speaker + source/target prosody, with WER/CER/F0/speaker similarity.

**What it already solves.** Conceptually, it shows a compact codec can be designed around disentangled speech factors.

**Gap that remains.** The public repo did not yet contain released code/checkpoints in the source I verified; it says code and pretrained models would be released “soon,” and GitHub shows no releases. ([GitHub](https://github.com/exercise-book-yq/freecodec "GitHub - exercise-book-yq/FreeCodec: FREECODEC: A DISENTANGLED NEURAL SPEECH CODEC WITH FEWER TOKENS · GitHub")) Also, the paper is speech-focused, not singing-focused.

**Usefulness.** **Related-work context / experiment pattern.** Not a reliable baseline unless the code/checkpoints become usable.

---

## 8. MSR-Codec

**Verified summary.** MSR-Codec is a low-bitrate multi-scale residual codec that encodes speech into four streams: **semantic, timbre, prosody, and residual**. The paper claims high-fidelity reconstruction, information disentanglement, two-stage lightweight TTS, and voice conversion via independent manipulation of timbre/prosody. ([arXiv](https://arxiv.org/abs/2509.13068 "[2509.13068] MSR-Codec: A Low-Bitrate Multi-Stream Residual Codec for High-Fidelity Speech Generation with Information Disentanglement")) The repo provides official PyTorch **inference** code, Apache-2.0 license, and demo/checkpoint structure. ([GitHub](https://github.com/herbertLJY/MSRCodec "GitHub - herbertLJY/MSRCodec: MSRCodec demo · GitHub"))

**Relevant cited/upstream works.** FACodec, codec-LM tokenizers, SSL semantic tokens, speaker encoders, prosody representations, residual codecs.

**Relevant citing/downstream works.** Too recent for a stable citation trail. Treat DisCo-Speech and “Exploring Disentangled Neural Speech Codecs” as adjacent codec-disentanglement work rather than confirmed descendants. ([arXiv](https://arxiv.org/abs/2512.13251 "[2512.13251] DisCo-Speech: Controllable Zero-Shot Speech Generation with A Disentangled Speech Codec"))

**Experiment-design pattern it teaches.** Use **pretrained stream extraction and stream swapping** rather than retraining. For a few-GPU project, MSR-Codec is useful if you can run inference and probe streams for singer/style leakage.

**What it already solves.** It gives a concrete multi-stream representation for speech and an inference pathway.

**Gap that remains.** It is not validated for singing style conversion, and the public repo is inference-oriented rather than a full reproducible training recipe.

**Usefulness.** **Baseline candidate for inference/probing; related-work context for training.**

---

## 9. Speaker leakage probing in SSL speech representations

**Verified summary.** A large-scale probing study analyzes 11 SSL models and decomposes speaker identity into acoustic, prosodic, and paralinguistic attributes. It reports a layer hierarchy: early layers encode acoustics, middle layers encode more abstract traits, final layers are not purely linguistic, and larger models can recover speaker identity in deep layers. It also reports that intermediate SSL layers can capture dynamic prosody better than specialized speaker embeddings. ([arXiv](https://arxiv.org/html/2501.05310 "A Large-Scale Probing Analysis of Speaker-Specific Attributes in Self-Supervised Speech Representations")) Eta-WavLM proposes a simple linear decomposition of SSL representations into speaker-specific and speaker-independent components, motivated by the fact that removing speaker identity often degrades content. ([arXiv](https://arxiv.org/html/2505.19273v1 "Eta-WavLM: Efficient Speaker Identity Removal in Self-Supervised Speech Representations Using a Simple Linear Equation")) Another Interspeech 2023 paper finds speaker and phonetic information in nearly orthogonal subspaces and uses PCA-based speaker normalization without transcriptions. ([ISCA Archive](https://www.isca-archive.org/interspeech_2023/liu23j_interspeech.html "ISCA Archive - Self-supervised Predictive Coding Models Encode Speaker and Phonetic Information in Orthogonal Subspaces"))

**Relevant cited/upstream works.** wav2vec 2.0, HuBERT, WavLM, UniSpeech-SAT, ContentVec, k-means pseudo-labeling, speaker encoders, phone-discrimination probes.

**Relevant citing/downstream works.** Eta-WavLM and orthogonal-subspace work are the most relevant method descendants for your purposes. ContentVec is also central because it explicitly regularizes HuBERT-style representations for speaker disentanglement while preserving content. ([arXiv](https://arxiv.org/abs/2204.09224 "https://arxiv.org/abs/2204.09224"))

**Experiment-design pattern it teaches.** Freeze representations, extract layers/streams, and train cheap probes for **singer identity, source style, target style, F0 statistics, phoneme/lyrics, and dynamics**. Do this before training a large generator.

**What it already solves.** It gives a low-compute diagnostic toolkit for representation leakage.

**Gap that remains.** Most probing work is speech-centric. Singing style needs probes for **vibrato rate/depth, glissando drift, breathiness/noise, falsetto/mixed/pharyngeal style**, and singer/style separation.

**Usefulness.** **Experiment-critical evaluation method.**

---

## 10. SingMOS-Pro and singing quality assessment

**Verified summary.** SingMOS-Pro is a public singing quality assessment benchmark. It contains **7,981 clips** generated by **41 models** across **12 datasets**, with at least five experienced annotators per clip; it extends earlier SingMOS overall ratings with lyrics, melody, and overall annotations. ([arXiv](https://arxiv.org/html/2510.01812v4 "SingMOS-Pro: An Comprehensive Benchmark for Singing Quality Assessment")) The paper reports **11.15 hours**, **44,247 ratings**, and 78 annotators; it includes SVS, SVC, SVR, and ground-truth clips. ([arXiv](https://arxiv.org/html/2510.01812v4 "SingMOS-Pro: An Comprehensive Benchmark for Singing Quality Assessment")) The GitHub repo provides easy-to-use SingMOS predictors, including `singmos_pro` and `singmos_v1`, trained at 16 kHz. ([GitHub](https://github.com/South-Twilight/SingMOS "GitHub - South-Twilight/SingMOS: Officail repo for SingMOS-Pro (ICASSP 2026) · GitHub")) The Hugging Face dataset card shows 7.98k rows and CC-BY-4.0 licensing. ([Hugging Face](https://huggingface.co/datasets/TangRain/SingMOS-Pro "TangRain/SingMOS-Pro · Datasets at Hugging Face"))

**Relevant cited/upstream works.** SingMOS, VoiceMOS 2024, MOS-Bench/SHEET, DNSMOS, UTMOS, pitch-and-spectrum-aware SQA, OpenCpop, DiffSinger, M4Singer, GTSinger, and common vocoder/codec baselines appear in the paper’s references and data construction. ([arXiv](https://arxiv.org/html/2510.01812v4 "SingMOS-Pro: An Comprehensive Benchmark for Singing Quality Assessment"))

**Relevant citing/downstream works.** SVCC 2025 found neural MOS systems such as SHEET-SSQA and SingMOS had the highest correlation with subjective naturalness among tested objective metrics, with SRCC above 0.6; speaker/singer embeddings correlated better for identity/style similarity, above 0.75. ([arXiv](https://arxiv.org/html/2509.15629v1 "https://arxiv.org/html/2509.15629v1"))

**Experiment-design pattern it teaches.** Use automatic singing MOS for **screening and ablation triage**, but keep a small human MOS/XAB/identity test for the final claims.

**What it already solves.** It gives public data and a usable model for singing MOS prediction.

**Gap that remains.** Automatic MOS is not style similarity. The paper itself notes future value in incorporating melody and lyric scores more effectively. ([arXiv](https://arxiv.org/html/2510.01812v4 "SingMOS-Pro: An Comprehensive Benchmark for Singing Quality Assessment"))

**Usefulness.** **Experiment-critical evaluation support**, but not a final arbiter.

---

# Top 20 papers/systems to read, in order

1. **SVCC 2025** — task definition, split design, metrics, and failure modes.
    
2. **GTSinger** — the main data/design anchor for technique/style labels.
    
3. **Seed-VC** — strongest practical public zero-shot VC/SVC baseline.
    
4. **NaturalSpeech 3 / FACodec** — factorized representation design; use FACodec, do not reproduce the full TTS scale.
    
5. **SingMOS-Pro** — singing quality assessment and MOS-prediction baseline.
    
6. **VERSA** — metric-battery infrastructure; its repo provides 90+ metrics and optional installs. ([GitHub](https://github.com/wavlab-speech/versa "https://github.com/wavlab-speech/versa"))
    
7. **ContentVec** — speaker-disentangled SSL representation adapted from HuBERT. ([arXiv](https://arxiv.org/abs/2204.09224 "https://arxiv.org/abs/2204.09224"))
    
8. **Large-Scale Probing Analysis of Speaker-Specific Attributes in SSL Speech Representations** — layer-wise leakage/prosody probes.
    
9. **Eta-WavLM** — simple linear speaker-removal baseline.
    
10. **Self-supervised Predictive Coding Models Encode Speaker and Phonetic Information in Orthogonal Subspaces** — PCA/subspace speaker normalization.
    
11. **AdaIN-VC** — small, interpretable IN-based one-shot VC baseline.
    
12. **LoIN** — lightweight IN ablation.
    
13. **AGAIN-VC** — activation bottleneck + AdaIN, with public pretrained model. ([arXiv](https://arxiv.org/abs/2011.00316 "https://arxiv.org/abs/2011.00316"))
    
14. **AutoVC** — canonical autoencoder-only zero-shot/nonparallel VC bottleneck baseline. ([arXiv](https://arxiv.org/abs/1905.05879 "https://arxiv.org/abs/1905.05879"))
    
15. **FreeVC** — WavLM bottleneck + VITS, text-free one-shot VC with code/pretrained models. ([arXiv](https://arxiv.org/abs/2210.15418 "https://arxiv.org/abs/2210.15418"))
    
16. **VQMIVC** — vector quantization + mutual-information disentanglement with code/pretrained models. ([GitHub](https://github.com/Wendison/VQMIVC "https://github.com/Wendison/VQMIVC"))
    
17. **FreeCodec** — low-token disentangled codec idea; read for design, not reproducibility.
    
18. **MSR-Codec** — multi-stream residual codec; read and try inference/probing.
    
19. **Serenade** — singing style conversion via audio infilling, cyclic training, and F0 post-processing. ([arXiv](https://arxiv.org/abs/2503.12388 "https://arxiv.org/abs/2503.12388"))
    
20. **SingStyle111** — small, multilingual, style-transfer-oriented singing dataset. ([Zenodo](https://zenodo.org/records/10265401 "https://zenodo.org/records/10265401"))
    

Honorable mentions: **M4Singer** if you need score-conditioned Mandarin singing data, and **DiffSinger** if you need a public SVS generation baseline. ([GitHub](https://github.com/M4Singer/M4Singer "https://github.com/M4Singer/M4Singer"))

---

# Top 8 experiment patterns to imitate

1. **SVCC-style split:** seen-singer in-domain style conversion plus unseen-singer zero-shot style conversion. This directly tests whether singer identity and singing style are disentangled. ([arXiv](https://arxiv.org/html/2509.15629v1 "https://arxiv.org/html/2509.15629v1"))
    
2. **Probe before generation:** for every SSL/codec layer or stream, train cheap probes for singer ID, source style, F0 statistics, lyrics/phoneme content, and dynamic style. This follows the SSL leakage probing literature and can be run on a few GPUs or even one GPU. ([arXiv](https://arxiv.org/html/2501.05310 "A Large-Scale Probing Analysis of Speaker-Specific Attributes in Self-Supervised Speech Representations"))
    
3. **Stream swapping:** use source content, target timbre, source/target prosody, and residual/detail streams in all combinations. FACodec, FreeCodec, and MSR-Codec all motivate this design. ([GitHub](https://github.com/lifeiteng/naturalspeech3_facodec "https://github.com/lifeiteng/naturalspeech3_facodec"))
    
4. **Normalization bottleneck ablations:** compare no bottleneck, global IN, AdaIN, LoIN, activation bottleneck, VQ, and k-means. This is cheap and interpretable. ([ISCA Archive](https://www.isca-archive.org/interspeech_2019/chou19_interspeech.html "https://www.isca-archive.org/interspeech_2019/chou19_interspeech.html"))
    
5. **Timbre perturbation / anti-leakage training:** imitate Seed-VC’s timbre-shifter logic with small augmentations or a frozen-source perturbation module rather than retraining a giant DiT. ([arXiv](https://arxiv.org/html/2411.09943v1 "https://arxiv.org/html/2411.09943v1"))
    
6. **Dynamic-style ablations:** separately evaluate vibrato, glissando, and breathiness. SVCC 2025 identifies these as the hard styles and links them to dynamic F0/noise behavior. ([arXiv](https://arxiv.org/html/2509.15629v1 "https://arxiv.org/html/2509.15629v1"))
    
7. **Metric battery plus correlation sanity check:** run VERSA/SingMOS/SHEET-style metrics, but report which correlate with your human tests. SVCC did this and found MOS predictors useful for naturalness, while embeddings were stronger for identity/style similarity. ([arXiv](https://arxiv.org/html/2509.15629v1 "https://arxiv.org/html/2509.15629v1"))
    
8. **Freeze-large, train-small:** use pretrained Seed-VC, FACodec, MSR-Codec, ContentVec/WavLM, and train only probes/adapters/residual style modules. This matches your few-GPU constraint much better than reproducing NaturalSpeech 3 or Vevo-scale systems.
    

---

# Strongest novelty gaps left

1. **Singing-style leakage probes.** Verified problem: SSL representations leak speaker traits; SVCC’s Task 2 requires singer/style disentanglement. My inference: a focused benchmark that measures **source singer leakage vs source style leakage vs target style uptake** for singing would be publishable and feasible. ([arXiv](https://arxiv.org/html/2501.05310 "A Large-Scale Probing Analysis of Speaker-Specific Attributes in Self-Supervised Speech Representations"))
    
2. **Dynamic style representation.** Verified problem: breathy, glissando, and vibrato are hard, and SVCC attributes this to dynamic information. My inference: a model that explicitly represents **time-varying breath noise, vibrato rate/depth, and glissando F0 drift** could make a strong contribution. ([arXiv](https://arxiv.org/html/2509.15629v1 "https://arxiv.org/html/2509.15629v1"))
    
3. **Singing-specific factorized codec.** Verified fact: FACodec, FreeCodec, and MSR-Codec factorize speech attributes; singing style conversion remains unsolved. My inference: adapting factorized codecs to singing, then probing each stream for singer/style/melody leakage, is a strong few-GPU direction if you freeze most components. ([GitHub](https://github.com/lifeiteng/naturalspeech3_facodec "https://github.com/lifeiteng/naturalspeech3_facodec"))
    
4. **Objective style metrics.** Verified problem: SVCC found objective metrics still need improvement; naturalness MOS predictors and identity/style embeddings correlate only partially with subjective scores. My inference: a style-specific metric for vibrato/glissando/breathy similarity would be valuable. ([arXiv](https://arxiv.org/html/2509.15629v1 "https://arxiv.org/html/2509.15629v1"))
    
5. **Small-data reproducibility.** Verified fact: top SVCC systems used data scales like 10k speech + 500h singing or 100k speech + 7k singing. My inference: a reproducible system trained on the SVCC 75h-style setting plus public pretrained encoders would be more useful to the community than another huge private-data model. ([arXiv](https://arxiv.org/html/2509.15629v1 "https://arxiv.org/html/2509.15629v1"))
    
6. **Lyrics/melody-aware singing MOS.** Verified gap: SingMOS-Pro notes that incorporating melody and lyric scores more effectively remains future work. My inference: combining MOS prediction with explicit pitch/lyrics/style probes could produce a stronger evaluation package. ([arXiv](https://arxiv.org/html/2510.01812v4 "SingMOS-Pro: An Comprehensive Benchmark for Singing Quality Assessment"))
    

---

# Papers/resources with usable public code or data

**Most usable immediately:**

|Resource|Use in your experiment|Caveat|
|---|---|---|
|**SVCC 2025 dataset**|Main benchmark split and test design|Dataset is a GTSinger subset; full GTSinger is disallowed for challenge-comparable training. ([Hugging Face](https://huggingface.co/datasets/lestervioleta/svcc2025 "https://huggingface.co/datasets/lestervioleta/svcc2025"))|
|**GTSinger**|Broader data analysis, technique labels, non-challenge experiments|License/rules matter; do not mix into SVCC training if claiming challenge comparability. ([GitHub](https://github.com/AaronZ345/GTSinger "https://github.com/AaronZ345/GTSinger"))|
|**Seed-VC**|Strong zero-shot VC/SVC baseline; pretrained inference and small fine-tuning|GPL-3.0 repo; full challenge-level training data is much larger than few-GPU. ([GitHub](https://github.com/Plachtaa/seed-vc "https://github.com/Plachtaa/seed-vc"))|
|**FACodec**|Factorized codec streams and stream-swap probes|Use codec/checkpoints, not full NaturalSpeech 3 training. ([GitHub](https://github.com/lifeiteng/naturalspeech3_facodec "https://github.com/lifeiteng/naturalspeech3_facodec"))|
|**MSR-Codec**|Multi-stream inference/probing baseline|Inference code, not full training recipe. ([GitHub](https://github.com/herbertLJY/MSRCodec "GitHub - herbertLJY/MSRCodec: MSRCodec demo · GitHub"))|
|**SingMOS-Pro + SingMOS repo**|Automatic singing MOS screening|Not a replacement for style-similarity listening tests. ([GitHub](https://github.com/South-Twilight/SingMOS "GitHub - South-Twilight/SingMOS: Officail repo for SingMOS-Pro (ICASSP 2026) · GitHub"))|
|**VERSA**|Metric battery and correlation analysis|Many optional metric dependencies; start with a small subset. ([GitHub](https://github.com/wavlab-speech/versa "https://github.com/wavlab-speech/versa"))|
|**AdaIN-VC / AGAIN-VC / LoIN**|Small VC baselines and bottleneck ablations|Older/smaller systems; useful for mechanisms, not SOTA. ([GitHub](https://github.com/wangxiii/One-Shot-VC "GitHub - wangxiii/One-Shot-VC · GitHub"))|
|**FreeVC / VQMIVC / ContentVec**|Modern public speech-VC/disentanglement baselines|Mostly speech, not singing; still useful for leakage comparisons. ([GitHub](https://github.com/OlaWod/FreeVC "https://github.com/OlaWod/FreeVC"))|
|**SingStyle111**|Small multilingual style-transfer data|Smaller than GTSinger, but very useful for few-GPU style experiments. ([Zenodo](https://zenodo.org/records/10265401 "https://zenodo.org/records/10265401"))|
|**M4Singer / DiffSinger**|Score-conditioned singing data/model context|More SVS-oriented than conversion-oriented. ([GitHub](https://github.com/M4Singer/M4Singer "https://github.com/M4Singer/M4Singer"))|

**Not reliably usable as-is:** **FreeCodec** has a repo and demo, but the verified repo still says code/pretrained models will be released “soon” and shows no releases. ([GitHub](https://github.com/exercise-book-yq/freecodec "GitHub - exercise-book-yq/FreeCodec: FREECODEC: A DISENTANGLED NEURAL SPEECH CODEC WITH FEWER TOKENS · GitHub")) **DisCo-Speech** is relevant but its repo says source code is still being prepared. ([GitHub](https://github.com/disco-speech/DisCo-Speech "GitHub - disco-speech/DisCo-Speech · GitHub"))

---

# Important but too expensive or unreproducible for you

1. **NaturalSpeech 3 full system.** The design is important, but the reported system scales to 500M–1B parameters and up to 200k hours of training data. Use FACodec, not the full system. ([arXiv](https://arxiv.org/html/2403.03100v3 "https://arxiv.org/html/2403.03100v3"))
    
2. **SVCC top large-data systems.** S5 used about 10k hours speech + 500h singing; S6/S7 used 100k hours speech plus about 7k–7.5k hours singing. Treat these as upper-bound context, not feasible baselines. ([arXiv](https://arxiv.org/html/2509.15629v1 "https://arxiv.org/html/2509.15629v1"))
    
3. **Full SVCC listening-test scale.** The challenge used 480 ratings/system. You can imitate the design with a smaller listening study, but not the full scale. ([arXiv](https://arxiv.org/html/2509.15629v1 "https://arxiv.org/html/2509.15629v1"))
    
4. **FreeCodec reproduction.** The paper is useful, but the verified public repo did not yet provide code/checkpoints. ([GitHub](https://github.com/exercise-book-yq/freecodec "GitHub - exercise-book-yq/FreeCodec: FREECODEC: A DISENTANGLED NEURAL SPEECH CODEC WITH FEWER TOKENS · GitHub"))
    
5. **MSR-Codec training.** Inference/probing is feasible; retraining the codec from scratch is not supported by the public repo I verified. ([GitHub](https://github.com/herbertLJY/MSRCodec "GitHub - herbertLJY/MSRCodec: MSRCodec demo · GitHub"))
    
6. **DisCo-Speech full pipeline.** It is highly relevant to disentangled codec control, but verified source-code release was still pending. ([GitHub](https://github.com/disco-speech/DisCo-Speech "GitHub - disco-speech/DisCo-Speech · GitHub"))
    

---

# A feasible experiment I would actually run

Use **SVCC 2025** as the main benchmark. Compare **Seed-VC zero-shot**, **FACodec stream-swap**, **MSR-Codec stream-swap**, and one or two small normalization/bottleneck baselines such as **AdaIN-VC/LoIN/FreeVC/VQMIVC**. Freeze the large encoders/codecs, train only probes and a small style adaptor. Evaluate with: singer/style/content probes, WER/CER or lyrics intelligibility, F0/vibrato/glissando/breathiness descriptors, SingMOS/Versa/SHEET-style automatic metrics, and a small SVCC-style human MOS/XAB/identity test. This is much more likely to produce a credible few-GPU paper than trying to scale a new generator.