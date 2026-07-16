I am a master's student exploring speech/singing voice research. I only have a
few GPUs, so I cannot train a new foundation model or do scaling-heavy singing
voice conversion. I want a research direction that is novel enough, interesting,
and feasible.

My broad interests:

same-person timbre or vocal identity shift between speaking and singing;
same-person timbre shift across languages, if data makes this possible;
speaker/timbre/content/prosody disentanglement in speech or singing;
probing and steering directions in frozen SSL/audio-codec representations;
singing style or technique control, such as vibrato, breathiness, falsetto,
glissando, belting, mixed voice, and other singing techniques;
automatic singing quality or skill evaluation, if it can be formulated
rigorously.
Please act as a senior researcher and give me a 2024-2026 field map for:

singing voice conversion;
singing style conversion;
speech-to-singing or singing-to-speech conversion;
cross-lingual speaker/timbre disentanglement;
SSL/audio-codec representation factorization;
singing quality, skill, or technique evaluation.
I need you to be critical and source-grounded. Please use primary sources where
possible: papers, official challenge pages, datasets, code repositories, and
proceedings.

Please answer:

What is the current zeitgeist? What are people excited about now?
What claims are saturated or no longer novel?
What problems remain clearly unsolved?
Which labs, groups, datasets, benchmarks, and challenges are central?
Which 15-25 papers/systems should I read first?
Which systems have usable code/checkpoints and which are paper-only?
What research questions fit a small-GPU master's project?
Which directions are likely too risky, too incremental, or too engineering-only?
Then produce a ranked list of 8-12 possible research directions. For each
direction, include:

one-sentence research question;
why it matters now;
expected novelty;
key related work;
dataset needed;
baseline needed;
minimum experiment;
expected compute;
likely failure mode;
whether I should pursue, park, or reject it.
Output should include:

a concise executive summary;
a table of key papers/systems;
a table ranking research directions;
a short reading list grouped by topic;
explicit warnings about overclaims.


# answer
## Executive summary

The most promising small-GPU thesis space is **not** “train a better singing voice converter.” The field has moved fast toward large, unified speech/singing models and challenge-winning systems that use large corpora, big autoregressive/flow/diffusion models, and sometimes private or web-harvested data. Vevo1.5, for example, exposes open checkpoints but its public model card describes 101k hours of speech, 438.9 hours of open singing, and up to 7k hours of internal singing data for some variants; a June 2026 unified speech/singing preprint reports training on 30k hours of speech plus 35k hours of singing. ([Hugging Face](https://huggingface.co/amphion/Vevo1.5 "https://huggingface.co/amphion/Vevo1.5"))

The best master’s-level wedge is: **probe, evaluate, and lightly steer frozen SSL/audio-codec representations for singing-specific factors**: singer identity across speech/singing, style/technique leakage, vibrato/glissando/breathy control, and objective quality/technique evaluation. This uses open datasets such as GTSinger, M4Singer, VocalSet, NHSS, NUS-48E, SingMOS-Pro, and challenge baselines rather than training a foundation model. GTSinger is especially central because it gives 80.59 hours of singing, 20 professional singers, 9 languages, six technique annotations, realistic scores, phoneme alignments, global style labels, and 16.16 hours of paired speech. ([NeurIPS Proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/023d2c1a17cf35b11a0cbb43a0677c91-Abstract-Datasets_and_Benchmarks_Track.html "https://proceedings.neurips.cc/paper_files/paper/2024/hash/023d2c1a17cf35b11a0cbb43a0677c91-Abstract-Datasets_and_Benchmarks_Track.html"))

The 2025–2026 zeitgeist is **singing style conversion**, not just singer identity conversion. SVCC 2025 explicitly shifted from singer identity conversion to **singing style conversion**, including breathy, falsetto, mixed voice, pharyngeal, glissando, vibrato, and control styles; the 2026 analysis says top systems reached strong singer-identity scores, but dynamic style modeling and naturalness remain difficult, especially for breathy, glissando, and vibrato. ([Singing Voice Conversion Challenge](https://vc-challenge.org/ "https://vc-challenge.org/"))

My strongest recommendations are:

1. **Pursue representation/evaluation projects first.** They are novel, publishable, and feasible on one or a few GPUs.
    
2. **Park true same-person cross-lingual singing identity shift** until you verify metadata supports the same singer across languages.
    
3. **Reject “new SVC system with HuBERT + F0 + diffusion/VITS”** unless it tests a sharply defined representation or evaluation hypothesis.
    
4. **Be very careful with “disentanglement” claims.** In this field, many papers say disentanglement, but few test leakage, invariance, intervention, and cross-domain generalization rigorously.
    

---

## 1. Current zeitgeist, 2024–2026

### 1.1 From singer identity conversion to style/technique conversion

SVCC 2023 showed that singing voice conversion had reached high naturalness in top systems, but target-speaker similarity remained below real target recordings, and objective metrics correlated only weakly with perception. ([arXiv](https://arxiv.org/abs/2306.14422 "https://arxiv.org/abs/2306.14422")) By SVCC 2025, the benchmark moved to **singing style conversion**, where identity and linguistic content should remain fixed while “how the singer sings” changes. The official task includes in-domain and zero-shot style conversion, with style references sometimes coming from a different singer. ([Singing Voice Conversion Challenge](https://vc-challenge.org/ "https://vc-challenge.org/"))

This shift matters because it exposes the core unsolved problem: **style is not a static embedding**. Vibrato, glissando, breathiness, falsetto, belting/mixed voice, and phrasing are time-varying, partly physiological, partly musical, and entangled with F0, intensity, articulation, vowel shape, and singer identity. The SVCC 2025 analysis says style modeling and naturalness remain challenging, especially for dynamic information in breathy, glissando, and vibrato styles. ([arXiv](https://arxiv.org/abs/2509.15629 "https://arxiv.org/abs/2509.15629"))

### 1.2 Big unified speech/singing models are exciting but not thesis-friendly to train

Vevo proposes controllable zero-shot voice imitation by separating content-style modeling from acoustic modeling and using prompted style/timbre references. ([arXiv](https://arxiv.org/abs/2502.07243 "https://arxiv.org/abs/2502.07243")) Vevo1.5 extends this to unified speech and singing modeling with controllability over text, prosody, style, and timbre, and releases substantial pretrained components in Amphion. ([GitHub](https://github.com/open-mmlab/Amphion/blob/main/models/svc/vevosing/README.md "https://github.com/open-mmlab/Amphion/blob/main/models/svc/vevosing/README.md")) The 2026 trend continues toward unified models such as UniVoice, which factorizes content, melody, and timbre but reports training on tens of thousands of hours. ([arXiv](https://arxiv.org/abs/2606.05852 "https://arxiv.org/abs/2606.05852"))

For you, this suggests a strategy: **use these systems as frozen objects of study**, not as systems to reproduce from scratch.

### 1.3 Technique-controllable singing is becoming a real benchmark topic

GTSinger made technique labels and paired speech/singing data much more accessible. ([NeurIPS Proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/023d2c1a17cf35b11a0cbb43a0677c91-Abstract-Datasets_and_Benchmarks_Track.html "https://proceedings.neurips.cc/paper_files/paper/2024/hash/023d2c1a17cf35b11a0cbb43a0677c91-Abstract-Datasets_and_Benchmarks_Track.html")) TCSinger and TCSinger2 target zero-shot singing synthesis with style transfer/control, including cross-lingual speech and singing style prompts; their public repos provide code, and TCSinger has released checkpoints. ([GitHub](https://github.com/AaronZ345/TCSinger "https://github.com/AaronZ345/TCSinger")) TechSinger focuses directly on vocal technique control, supporting five languages and seven vocal techniques, with code and checkpoints released; its public checkpoint is limited to Chinese and English. ([arXiv](https://arxiv.org/abs/2502.12572 "https://arxiv.org/abs/2502.12572"))

This makes technique control a good thesis area, but the best contribution is probably **measurement, probing, and controllable intervention**, not “we add another technique token.”

### 1.4 Objective evaluation is hot because MOS and speaker similarity are bottlenecks

VoiceMOS 2024 added a singing voice track for SVS/SVC MOS prediction, and the challenge paper reports that successful systems used retrieval-based methods and non-SSL features such as spectrograms and pitch histograms, not just SSL embeddings. ([Google Sites](https://sites.google.com/view/voicemos-challenge/past-challenges/voicemos-challenge-2024 "https://sites.google.com/view/voicemos-challenge/past-challenges/voicemos-challenge-2024")) SingMOS-Pro then extended singing quality assessment with 7,981 clips from 41 models across 12 datasets, with lyrics, melody, and overall ratings for part of the corpus. ([arXiv](https://arxiv.org/html/2510.01812 "https://arxiv.org/html/2510.01812"))

The important negative result: objective metrics are still not replacements for listeners. SVCC 2025 found chroma-alignment and speaker embeddings correlated most with subjective scores, but still not enough to replace subjective evaluation. ([arXiv](https://arxiv.org/abs/2509.15629 "https://arxiv.org/abs/2509.15629"))

---

## 2. What is saturated or no longer novel

These claims are weak unless you add a rigorous new evaluation, dataset split, or intervention:

|Saturated claim|Why it is weak now|What would make it publishable|
|---|---|---|
|“We disentangle content, singer, pitch, and style.”|Many systems already use HuBERT/Whisper/ContentVec/F0/speaker embeddings and call the result disentanglement.|Show leakage tests, counterfactual edits, cross-domain retrieval, and failure prediction.|
|“We propose a new SVC model with better MOS.”|SVC naturalness is already high in strong systems; subjective tests are expensive and often underpowered.|A controlled benchmark on hard style/identity splits, with open outputs and analysis.|
|“Zero-shot singing conversion from any reference.”|Top systems now use large data and strong pretrained models; demos are not enough.|Held-out singer, held-out language, held-out technique, and robust identity/style evaluation.|
|“Style control by adding a global style token.”|SVCC 2025 shows dynamic styles remain difficult; global style labels often miss frame-level timing. ([arXiv](https://arxiv.org/abs/2509.15629 "https://arxiv.org/abs/2509.15629"))|Demonstrate temporal control over vibrato rate/extent, glissando slope, breathiness, falsetto register, etc.|
|“Automatic singing quality score.”|Generic MOS predictors often fail out of domain; SingMOS-Pro explicitly motivates singing-specific quality assessment because existing objective metrics capture limited perceptual aspects. ([arXiv](https://arxiv.org/html/2510.01812 "https://arxiv.org/html/2510.01812"))|Define the construct: overall MOS, lyrics accuracy, melody accuracy, timbre naturalness, technique correctness, or skill rubric.|
|“Cross-lingual singer/timbre disentanglement.”|Speaker/timbre is coupled with pronunciation, language, pitch range, recording, and singer distribution; RefXVC explicitly notes pronunciation-dependent timbre variation in cross-lingual VC. ([arXiv](https://arxiv.org/abs/2406.16326 "https://arxiv.org/abs/2406.16326"))|Same-speaker multilingual data or a careful confound analysis showing what cannot be concluded.|

---

## 3. Problems that remain clearly unsolved

**Same-person speech ↔ singing identity.** We do not yet have a robust answer to: “What part of a voice identity persists from speech to singing?” Paired speech/singing datasets exist, but they are small or uneven. NHSS has 100 songs sung and spoken by 10 singers, totaling 7 hours, while NUS-48E has 169 minutes from 12 subjects. ([HltNUS](https://hltnus.github.io/NHSSDatabase/ "https://hltnus.github.io/NHSSDatabase/")) GTSinger’s 16.16 hours of paired speech is currently the most useful modern source. ([NeurIPS Proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/023d2c1a17cf35b11a0cbb43a0677c91-Abstract-Datasets_and_Benchmarks_Track.html "https://proceedings.neurips.cc/paper_files/paper/2024/hash/023d2c1a17cf35b11a0cbb43a0677c91-Abstract-Datasets_and_Benchmarks_Track.html"))

**Dynamic style control.** Vibrato, glissando, breathiness, falsetto, mixed voice, belting, and register transitions are not static categories. They require time-local analysis and sometimes physiological interpretation.

**Objective metrics for style similarity and identity.** Speaker embeddings, chroma alignment, F0 metrics, and MOS predictors are useful diagnostics, but SVCC 2025 explicitly says they are not listener replacements. ([arXiv](https://arxiv.org/abs/2509.15629 "https://arxiv.org/abs/2509.15629"))

**Representation factorization that survives singing.** FACodec and NaturalSpeech 3 factorize speech into content, prosody, timbre, and acoustic detail, but the core question for you is whether those factors remain clean for singing, technique, and cross-domain speech↔singing use. NaturalSpeech 3 itself is speech-centered and reports large-scale training up to 1B parameters and 200k hours; FACodec is the usable open component. ([arXiv](https://arxiv.org/abs/2403.03100 "https://arxiv.org/abs/2403.03100"))

**Cross-lingual timbre identity.** This is attractive but data-limited. GTSinger has nine languages, but you must verify whether the same singer appears across multiple languages before claiming same-person cross-lingual timbre shift. Otherwise, you can only study language/singer confounding.

**Rigorous singing skill evaluation.** ASSE is emerging, but “skill” is a construct problem. A 2024 ISMIR paper addresses solo and accompaniment singing skill evaluation with 10KSinging and reports Pearson correlations of 0.769 for solo and 0.709 with accompaniment, but this is still narrower than a general singing-skill evaluator. ([ISMIR 2024](https://ismir2024program.ismir.net/poster_89.html "https://ismir2024program.ismir.net/poster_89.html"))

---

## 4. Central groups, datasets, benchmarks, and challenges

### Groups and ecosystems to watch

|Ecosystem|Why central|
|---|---|
|**Nagoya / Toda Lab / VCC-SVCC / VoiceMOS orbit**|Organizes/anchors SVCC, VoiceMOS/AudioMOS-style evaluation, and Serenade; SVCC 2025 organizers include Nagoya, CUHK-Shenzhen, CMU, and NII. ([Singing Voice Conversion Challenge](https://vc-challenge.org/ "https://vc-challenge.org/"))|
|**CUHK-Shenzhen / OpenMMLab Amphion / Zhizheng Wu group**|Vevo, Vevo1.5, Vevo2/Amphion-style unified speech/singing models and released checkpoints. ([GitHub](https://github.com/open-mmlab/Amphion/blob/main/models/svc/vevosing/README.md "https://github.com/open-mmlab/Amphion/blob/main/models/svc/vevosing/README.md"))|
|**Zhejiang University / Zhou Zhao group**|GTSinger, TCSinger, TCSinger2, TechSinger, SVPT; very central for technique, multilingual SVS, and speech-to-singing. ([GitHub](https://github.com/AaronZ345/TCSinger "https://github.com/AaronZ345/TCSinger"))|
|**NUS HLT / NUS singing corpora**|NHSS and NUS-48E are still important for same-person speech/singing comparisons. ([HltNUS](https://hltnus.github.io/NHSSDatabase/ "https://hltnus.github.io/NHSSDatabase/"))|
|**Microsoft Research / NaturalSpeech line**|Factorized codecs and large-scale speech synthesis; FACodec is useful for frozen representation studies. ([arXiv](https://arxiv.org/abs/2403.03100 "https://arxiv.org/abs/2403.03100"))|
|**MIR / singing evaluation groups**|VocalSet, 10KSinging/ASSE, SingMOS-Pro, VoiceMOS singing track, SVDD/SingFake-style evaluation and forensics. ([Zenodo](https://zenodo.org/records/1193957 "https://zenodo.org/records/1193957"))|

### Datasets and benchmarks

|Dataset / benchmark|Use it for|Notes|
|---|---|---|
|**GTSinger**|Technique recognition/control, style transfer, speech-to-singing, paired speech/singing identity|80.59h, 20 singers, 9 languages, six techniques, aligned phonemes, realistic scores, 16.16h paired speech. ([NeurIPS Proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/023d2c1a17cf35b11a0cbb43a0677c91-Abstract-Datasets_and_Benchmarks_Track.html "https://proceedings.neurips.cc/paper_files/paper/2024/hash/023d2c1a17cf35b11a0cbb43a0677c91-Abstract-Datasets_and_Benchmarks_Track.html"))|
|**SVCC 2025**|Singing style conversion benchmark|In-domain and zero-shot SSC; styles include breathy, falsetto, mixed voice, pharyngeal, glissando, vibrato, control; baselines Serenade and Vevo1.5 are open. ([Singing Voice Conversion Challenge](https://vc-challenge.org/ "https://vc-challenge.org/"))|
|**SVCC 2023**|SVC identity baseline context|Shows naturalness high but target similarity and objective metrics still problematic. ([arXiv](https://arxiv.org/abs/2306.14422 "https://arxiv.org/abs/2306.14422"))|
|**M4Singer**|Mandarin SVS/SVC, score-based baselines|20 professional singers, 700 Chinese pop songs, SATB coverage, score and alignment. ([M4Singer](https://m4singer.github.io/ "https://m4singer.github.io/"))|
|**VocalSet**|Technique classification, vowels, controlled vocal exercises|10.1h, 20 professional singers, standard and extended vocal techniques; not lyric-level pop singing. ([Zenodo](https://zenodo.org/records/1193957 "https://zenodo.org/records/1193957"))|
|**NHSS**|Same-person speech/singing paired study|10 singers, 100 songs, 7h, English pop lyrics sung and spoken. ([HltNUS](https://hltnus.github.io/NHSSDatabase/ "https://hltnus.github.io/NHSSDatabase/"))|
|**NUS-48E**|Speech/singing acoustic comparison|169 minutes, 12 subjects, 48 English songs, phone-level annotations for singing. ([Readkong](https://www.readkong.com/page/the-nus-sung-and-spoken-lyrics-corpus-a-quantitative-8954066 "https://www.readkong.com/page/the-nus-sung-and-spoken-lyrics-corpus-a-quantitative-8954066"))|
|**SingMOS-Pro**|Automatic singing quality assessment|7,981 clips, 41 models, 12 datasets, ratings for overall plus lyrics/melody on a subset; public HF dataset. ([arXiv](https://arxiv.org/html/2510.01812 "https://arxiv.org/html/2510.01812"))|
|**VoiceMOS 2024 singing track**|MOS prediction for SVS/SVC|Singing track covers SVS/SVC systems and larger variety of systems/listeners/languages. ([Google Sites](https://sites.google.com/view/voicemos-challenge/past-challenges/voicemos-challenge-2024 "https://sites.google.com/view/voicemos-challenge/past-challenges/voicemos-challenge-2024"))|
|**VoiceMOS/AudioMOS 2025–2026**|Broader automatic quality assessment|2025 expanded toward singing/music/general synthetic audio; 2026 returned to speech-focused tracks, signaling that even speech MOS generalization remains unsolved. ([Google Sites](https://sites.google.com/view/voicemos-challenge/voicemos-challenge-2026 "https://sites.google.com/view/voicemos-challenge/voicemos-challenge-2026"))|
|**SVDD 2024 / SingFake / CtrSVDD**|Singing deepfake detection and ethics-adjacent evaluation|SVDD 2024 introduced controlled and in-the-wild singing deepfake tracks, with 47 controlled-track submissions. ([arXiv](https://arxiv.org/abs/2408.16132 "https://arxiv.org/abs/2408.16132"))|

---

## 5. Key papers/systems to read first

Legend: **usable** = code and/or checkpoints/data appear usable from official repo or official page; **limited** = some assets but not enough to reproduce full training; **paper/demo** = no confirmed official full code/checkpoints in the sources I checked.

|Priority|Paper / system|Area|Why read it|Open assets status|Critical note|
|--:|---|---|---|---|---|
|1|**SVCC 2025 official + analysis**|SSC benchmark|Defines the 2025 shift from singer identity to singing style conversion; reports 33 systems and hard failure modes. ([Singing Voice Conversion Challenge](https://vc-challenge.org/ "https://vc-challenge.org/"))|**Usable/benchmark**: baseline code for Serenade and Vevo1.5 announced. ([Singing Voice Conversion Challenge](https://vc-challenge.org/ "https://vc-challenge.org/"))|Read before choosing any SSC topic.|
|2|**SVCC 2023**|SVC benchmark|Establishes that naturalness is easier than target similarity, and objective metrics are weak. ([arXiv](https://arxiv.org/abs/2306.14422 "https://arxiv.org/abs/2306.14422"))|Challenge outputs/baselines vary|Important for “what is saturated.”|
|3|**Serenade**|Singing style conversion|Audio-infilling approach; explicitly targets style modeling, source style disentanglement, and melody preservation. ([arXiv](https://arxiv.org/abs/2503.12388 "https://arxiv.org/abs/2503.12388"))|**Usable**: repo + pretrained models; noncommercial and anti-impersonation conditions. ([GitHub](https://github.com/lesterphillip/serenade "https://github.com/lesterphillip/serenade"))|Good baseline for SSC, not a full thesis by itself.|
|4|**Vevo / Vevo1.5 / Amphion**|Frozen controllable speech/singing model|Two-stage content-style and acoustic modeling; Vevo1.5 unifies speech and singing with style/timbre/prosody control. ([arXiv](https://arxiv.org/abs/2502.07243 "https://arxiv.org/abs/2502.07243"))|**Usable inference/checkpoints** in Amphion/HF. ([Hugging Face](https://huggingface.co/amphion/Vevo1.5 "https://huggingface.co/amphion/Vevo1.5"))|Use as frozen model; training scale is huge.|
|5|**S²Voice**|SVCC 2025 winning SSC system|2026 ICASSP accepted; builds on Vevo with style conditioning, global speaker embedding, web-harvested singing corpus, SFT/DPO. ([arXiv](https://arxiv.org/abs/2601.13629 "https://arxiv.org/abs/2601.13629"))|**Paper/demo** from sources checked|Useful to understand top-system direction; likely not reproducible on small GPUs.|
|6|**GTSinger**|Dataset/benchmarks|Core dataset for multilingual, multi-technique singing and paired speech/singing. ([NeurIPS Proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/023d2c1a17cf35b11a0cbb43a0677c91-Abstract-Datasets_and_Benchmarks_Track.html "https://proceedings.neurips.cc/paper_files/paper/2024/hash/023d2c1a17cf35b11a0cbb43a0677c91-Abstract-Datasets_and_Benchmarks_Track.html"))|**Usable dataset/code**. ([GitHub](https://github.com/AaronZ345/GTSinger "https://github.com/AaronZ345/GTSinger"))|Best thesis substrate.|
|7|**TCSinger / TCSinger2**|Zero-shot SVS, style transfer/control|Style transfer across speech and singing prompts; TCSinger2 addresses boundary robustness and multi-level prompt control. ([GitHub](https://github.com/AaronZ345/TCSinger "https://github.com/AaronZ345/TCSinger"))|**Usable code; TCSinger checkpoints**; TCSinger2 code released, no releases package. ([GitHub](https://github.com/AaronZ345/TCSinger "https://github.com/AaronZ345/TCSinger"))|Good for baselines, but full training may still be heavy.|
|8|**TechSinger**|Technique-controllable SVS|Directly targets vocal techniques such as mixed voice, falsetto, breathy, etc. ([arXiv](https://arxiv.org/abs/2502.12572 "https://arxiv.org/abs/2502.12572"))|**Usable code/checkpoints**; public checkpoint only Chinese/English. ([GitHub](https://github.com/gwx314/TechSinger/blob/main/README.md "https://github.com/gwx314/TechSinger/blob/main/README.md"))|Excellent for technique-evaluation experiments.|
|9|**SVPT**|Speech-to-singing|Self-supervised singing pretraining for STS; tackles paired-data scarcity with unpaired singing data. ([arXiv](https://arxiv.org/abs/2406.02429 "https://arxiv.org/abs/2406.02429"))|**Paper/samples**; full official code not confirmed from source checked|Strong conceptual reading.|
|10|**Everyone-Can-Sing**|Speech-reference SVS/SVC|Unified zero-shot SVS/SVC using speech reference for identity; relevant to same-person speech→singing timbre. ([arXiv](https://arxiv.org/abs/2501.13870 "https://arxiv.org/abs/2501.13870"))|**Paper/demo** from sources checked|Good motivation, but avoid overclaiming if not reproducible.|
|11|**Singing-to-Speech with Generative Flow**|Singing-to-speech|One of the few explicit S2S systems; repo includes training/inference and checkpoints. ([GitHub](https://github.com/jhuang448/singing-to-speech "https://github.com/jhuang448/singing-to-speech"))|**Usable code/checkpoints**|Niche but clean small project if framed as evaluation.|
|12|**NaturalSpeech 3 + FACodec**|Factorized codec|Factorizes content/prosody/timbre/acoustic detail; FACodec is released. ([arXiv](https://arxiv.org/abs/2403.03100 "https://arxiv.org/abs/2403.03100"))|**FACodec usable**, full NaturalSpeech 3 not small-scale|Great for probing; do not claim singing disentanglement without tests.|
|13|**ContentVec**|SSL disentanglement baseline|Classic speaker-disentangled SSL representation; official code and pretrained models. ([Proceedings of Machine Learning Research](https://proceedings.mlr.press/v162/qian22b.html "https://proceedings.mlr.press/v162/qian22b.html"))|**Usable code/checkpoints**|Good baseline for content/singer leakage studies.|
|14|**RefXVC**|Cross-lingual VC|Notes pronunciation-dependent timbre shifts and uses global/local reference embeddings. ([arXiv](https://arxiv.org/abs/2406.16326 "https://arxiv.org/abs/2406.16326"))|**Paper/samples** from source checked|Useful for framing cross-lingual timbre confounds.|
|15|**CoMoSVC**|Efficient SVC|Consistency model for fast SVC; single GTX4090 experiments; code/samples announced. ([arXiv](https://arxiv.org/abs/2401.01792 "https://arxiv.org/abs/2401.01792"))|**Code available** per paper/project|More engineering than science unless used as baseline.|
|16|**SaMoye**|Zero-shot SVC|Open-source zero-shot SVC claim with large 1,815h/6,367-speaker dataset; repo has weights. ([arXiv](https://arxiv.org/abs/2407.07728 "https://arxiv.org/abs/2407.07728"))|**Usable inference/weights**|Arxiv page says withdrawn; treat claims cautiously. ([arXiv](https://arxiv.org/abs/2407.07728 "https://arxiv.org/abs/2407.07728"))|
|17|**HQ-SVC**|Low-resource zero-shot SVC|AAAI 2026 repo claims single consumer GPU and <80h data. ([GitHub](https://github.com/ShawnPi233/HQ-SVC "https://github.com/ShawnPi233/HQ-SVC"))|**Inference/pretrained released; training code listed in release plan**|Promising but verify reproducibility before relying on it.|
|18|**SoulX-Singer / SoulX-Singer-SVC**|Large zero-shot SVS/SVC|Official inference code; raw-audio SVC without lyrics/MIDI. ([GitHub](https://github.com/Soul-AILab/SoulX-Singer "https://github.com/Soul-AILab/SoulX-Singer"))|**Usable inference/model**|Strong black-box baseline; training is not small-GPU.|
|19|**M4Singer**|Mandarin SVS/SVC dataset|Multi-style, multi-singer Mandarin corpus with score/alignment and SVC benchmark. ([M4Singer](https://m4singer.github.io/ "https://m4singer.github.io/"))|**Usable code/data**|Great for Mandarin baselines; less useful for cross-lingual same-person identity.|
|20|**VocalSet**|Vocal technique dataset|20 singers, vowels, scales/arpeggios/long tones, standard and extended techniques. ([Zenodo](https://zenodo.org/records/1193957 "https://zenodo.org/records/1193957"))|**Usable dataset**|Technique labels are controlled but less naturalistic than pop singing.|
|21|**NHSS / NUS-48E**|Speech/singing paired corpora|Useful for same-person speech↔singing analyses. ([HltNUS](https://hltnus.github.io/NHSSDatabase/ "https://hltnus.github.io/NHSSDatabase/"))|**Usable datasets**|Small, English-only, but clean for probing.|
|22|**SingMOS-Pro / SingMOS predictor**|Singing quality assessment|Public benchmark and ready-to-use MOS predictors. ([arXiv](https://arxiv.org/html/2510.01812 "https://arxiv.org/html/2510.01812"))|**Usable data/model**|Do OOD/system/language splits, not random splits only.|
|23|**VoiceMOS 2024**|MOS challenge|Singing voice track with SVS/SVC samples; challenge paper gives useful baseline lessons. ([Google Sites](https://sites.google.com/view/voicemos-challenge/past-challenges/voicemos-challenge-2024 "https://sites.google.com/view/voicemos-challenge/past-challenges/voicemos-challenge-2024"))|**Usable challenge data via CodaBench registration**|Evaluation research is highly feasible.|
|24|**ASSE ISMIR 2024**|Singing skill evaluation|Addresses solo and accompaniment singing skill evaluation. ([ISMIR 2024](https://ismir2024program.ismir.net/poster_89.html "https://ismir2024program.ismir.net/poster_89.html"))|Paper; dataset access needs checking|Good cautionary reference for “skill” formulation.|
|25|**SVDD 2024**|Singing deepfake detection|Important adjacent benchmark for responsible SVC/SVS evaluation. ([arXiv](https://arxiv.org/abs/2408.16132 "https://arxiv.org/abs/2408.16132"))|Challenge/data assets|Not your main topic, but useful ethics/evaluation context.|

---

## 6. What fits a small-GPU master’s project

Your best project shape is:

**Dataset + frozen representations + small probes/interventions + rigorous evaluation.**

Concretely, use frozen WavLM/HuBERT/ContentVec/Whisper/FACodec/Vevo tokens, extract features for GTSinger/NHSS/NUS/VocalSet/SingMOS-Pro, and train only linear probes, shallow MLPs, projection layers, or small adapters. Then make claims about **what information is where**, **what leaks**, **what can be steered**, and **which metrics predict listener judgments**.

Feasible experiments include:

- speaker/singer retrieval across speech↔singing;
    
- technique leakage classifiers on “content” and “timbre” representations;
    
- vibrato/glissando/breathiness detectors;
    
- representation ablations by layer/codebook;
    
- small orthogonal projections or adversarial-removal heads;
    
- style/identity metric correlation with SVCC or self-generated baselines;
    
- OOD singing MOS prediction using SingMOS-Pro.
    

Avoid spending your thesis budget on:

- training a new foundation model;
    
- competing with S²Voice/Vevo/SoulX via more architecture engineering;
    
- training a full SVC model from scratch unless it is only a baseline;
    
- subjective MOS collection as the main contribution unless you have budget, IRB/ethics clarity, and a strong protocol.
    

---

## 7. Ranked research directions

|Rank|Direction|One-sentence research question|Why it matters now|Expected novelty|Key related work|Dataset needed|Baseline needed|Minimum experiment|Expected compute|Likely failure mode|Verdict|
|--:|---|---|---|---|---|---|---|---|---|---|---|
|1|**Same-person speech↔singing identity map in frozen representations**|Which SSL/codec layers and codebooks preserve same-person identity across speaking and singing, and which encode modality/style instead?|Same-person timbre shift is central, and GTSinger/NHSS/NUS now make it testable without training a generator.|**High**, if you use controlled cross-domain retrieval and leakage tests.|GTSinger, NHSS, NUS-48E, FACodec, ContentVec. ([NeurIPS Proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/023d2c1a17cf35b11a0cbb43a0677c91-Abstract-Datasets_and_Benchmarks_Track.html "https://proceedings.neurips.cc/paper_files/paper/2024/hash/023d2c1a17cf35b11a0cbb43a0677c91-Abstract-Datasets_and_Benchmarks_Track.html"))|GTSinger paired speech; NHSS/NUS for validation.|ECAPA/x-vector/WavLM speaker embeddings, ContentVec, HuBERT, FACodec subspaces.|Extract frozen features; evaluate same-singer retrieval/EER across speech→singing, singing→speech, style-held-out, singer-held-out; train linear probes for singer, modality, technique.|1 GPU for feature extraction; probes run in hours.|Too few singers; embeddings may cluster by language/recording rather than identity.|**Pursue first.**|
|2|**Technique/style leakage audit for “disentangled” representations**|Do content, timbre, prosody, and codec subspaces actually leak vibrato, breathiness, falsetto, glissando, or singer identity?|Many systems claim disentanglement, but SVCC 2025 shows style remains hard.|**High**, because negative results are publishable if rigorous.|SVCC 2025, GTSinger, TechSinger, FACodec, ContentVec. ([arXiv](https://arxiv.org/abs/2509.15629 "https://arxiv.org/abs/2509.15629"))|GTSinger technique labels; VocalSet for external technique check.|Linear probes on HuBERT/ContentVec/Whisper/WavLM/FACodec/Vevo token streams.|Train technique and singer classifiers on each representation; test layer/codebook leakage; correlate leakage with conversion failures from Serenade/Vevo/TechSinger outputs.|<1 GPU-day after feature extraction.|Technique labels are noisy or taxonomy mismatch across datasets.|**Pursue.**|
|3|**Objective metric suite for singing style conversion**|Can dynamic style metrics for vibrato, glissando, breathiness, and register improve correlation with perceived SSC style similarity?|SVCC 2025 says objective metrics correlate but are not listener replacements. ([arXiv](https://arxiv.org/abs/2509.15629 "https://arxiv.org/abs/2509.15629"))|**Medium-high**, especially if open-source and challenge-compatible.|SVCC 2025, Serenade, VoiceMOS/SingMOS-Pro. ([Singing Voice Conversion Challenge](https://vc-challenge.org/ "https://vc-challenge.org/"))|SVCC 2025 samples if accessible; GTSinger-generated baseline outputs; SingMOS-Pro for quality side.|Speaker embedding similarity, F0/chroma alignment, MOS predictor, style classifier.|Build metrics: vibrato rate/extent, F0 modulation, glissando slope, breathiness proxy, style posterior; compare with subjective style/naturalness if available.|Mostly CPU + feature extraction GPU.|No public system-level subjective labels; metric overfits acoustic proxies.|**Pursue.**|
|4|**Frozen codec/prosody latent steering for vibrato or glissando**|Can a simple direction in F0/prosody/codebook space change vibrato or glissando while preserving lyrics, melody contour, and singer identity?|Style conversion needs controllability without retraining huge models.|**Medium-high** if intervention is causal and measured.|FACodec, Vevo1.5, Serenade, TechSinger. ([GitHub](https://github.com/lifeiteng/naturalspeech3_facodec "https://github.com/lifeiteng/naturalspeech3_facodec"))|GTSinger technique pairs; VocalSet long tones/scales.|Signal-processing F0 vibrato injection; TechSinger/Serenade if runnable.|Learn technique directions from paired/control examples; apply to held-out singers; measure content ASR, pitch error, speaker similarity, style classifier, small listening test.|1–3 GPU-days if decoding many samples; small if only feature-space analysis.|Decoder may not accept out-of-distribution latent edits; artifacts mistaken for style.|**Pursue cautiously.**|
|5|**Multitask singing quality / skill evaluator with explicit sub-scores**|Can a singing evaluator predict overall quality while separately modeling lyrics, melody, timbre naturalness, and technique correctness?|SingMOS-Pro makes singing-specific SQA feasible, and generic speech MOS does not solve singing. ([arXiv](https://arxiv.org/html/2510.01812 "https://arxiv.org/html/2510.01812"))|**Medium-high** if you emphasize OOD generalization and interpretability.|SingMOS-Pro, VoiceMOS 2024, ASSE 2024. ([arXiv](https://arxiv.org/html/2510.01812 "https://arxiv.org/html/2510.01812"))|SingMOS-Pro; optional GTSinger technique labels.|SingMOS predictor, SSL-MOS, pitch/spectrogram baselines.|Freeze SSL/audio features; train small multi-head model; evaluate leave-system-out, leave-language-out, leave-task-out; report calibration.|1 GPU, hours to a day.|MOS labels reflect system artifacts rather than singing skill; random splits inflate performance.|**Pursue.**|
|6|**Speech-prompt-to-singing timbre alignment adapter**|Can a small mapping convert a speaker embedding from speech into the corresponding singing-timbre prompt space?|Speech-reference singing is attractive, but same-person speech→singing identity remains underdefined.|**Medium**; good if paired-data analysis is strong.|GTSinger paired speech, NHSS, Everyone-Can-Sing, Vevo/SoulX. ([NeurIPS Proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/023d2c1a17cf35b11a0cbb43a0677c91-Abstract-Datasets_and_Benchmarks_Track.html "https://proceedings.neurips.cc/paper_files/paper/2024/hash/023d2c1a17cf35b11a0cbb43a0677c91-Abstract-Datasets_and_Benchmarks_Track.html"))|GTSinger paired speech; NHSS for external test.|Direct speech embedding as prompt; direct singing embedding; Procrustes/MLP adapter.|Learn speech→singing embedding mapping on singers; evaluate retrieval and frozen SVC prompt identity.|<1 GPU-day for embeddings/adapters; more if generating samples.|Too few speakers for supervised mapping; generation quality dominated by black-box model.|**Pursue if paired metadata is clean.**|
|7|**Low-resource singing technique recognition/segmentation**|Can technique detectors trained on GTSinger generalize to VocalSet and unseen languages/singers with calibrated uncertainty?|Technique labels are increasingly used for control, but detectors and taxonomies are fragile.|**Medium**, but useful and feasible.|GTSinger, VocalSet, TechSinger. ([NeurIPS Proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/023d2c1a17cf35b11a0cbb43a0677c91-Abstract-Datasets_and_Benchmarks_Track.html "https://proceedings.neurips.cc/paper_files/paper/2024/hash/023d2c1a17cf35b11a0cbb43a0677c91-Abstract-Datasets_and_Benchmarks_Track.html"))|GTSinger phoneme-level techniques; VocalSet technique clips.|WavLM/HuBERT/FACodec features + pitch/spectral features.|Train singer-held-out and language-held-out classifiers; test on VocalSet; report uncertainty/OOD rejection.|<1 GPU-day.|Labels differ: “breathy” in lyric singing vs isolated vowel exercises may not match.|**Pursue as a solid fallback.**|
|8|**Same-person cross-lingual singing timbre shift**|Does a singer’s timbre representation shift systematically with language, and can representations separate language from identity?|Cross-lingual timbre is interesting, and RefXVC highlights language/pronunciation-timbre coupling. ([arXiv](https://arxiv.org/abs/2406.16326 "https://arxiv.org/abs/2406.16326"))|**High if true same-singer multilingual data exists; low otherwise.**|RefXVC, TCSinger, GTSinger. ([arXiv](https://arxiv.org/abs/2406.16326 "https://arxiv.org/abs/2406.16326"))|Same singer across multiple languages; verify GTSinger metadata first.|Multilingual speaker embeddings; language classifiers; adversarial projection.|Retrieval/probing across same singer/different language; compare speech vs singing if possible.|Small.|Dataset may not contain enough same-person multilingual singing; then claims collapse.|**Park until metadata verified.**|
|9|**Singing-to-speech conversion as evaluation, not generation**|Can we quantify how well S2S removes melody/rhythm while preserving linguistic content and identity?|S2S is underexplored and has open code, but the use case is narrower.|**Medium** if framed as a benchmark/evaluation paper.|Singing-to-Speech generative flow, NHSS/NUS. ([GitHub](https://github.com/jhuang448/singing-to-speech "https://github.com/jhuang448/singing-to-speech"))|DSing if using the repo; NHSS/NUS for speech/singing pairs.|Published S2S model; signal-processing duration/F0 flattening baseline.|Run existing S2S; evaluate ASR, speaker similarity, F0/rhythm normalization, human naturalness on a small set.|1 GPU, maybe 1–3 days if training.|The task may feel niche unless tied to representation/identity theory.|**Park or pursue as a side project.**|
|10|**Fine-tune a full SVC/SSC model to beat challenge baselines**|Can a small model outperform Serenade/Vevo-style baselines on SSC?|Tempting, but top systems now use large data and heavy engineering.|**Low** unless you have a very sharp hypothesis.|Serenade, Vevo1.5, S²Voice, SoulX, HQ-SVC. ([arXiv](https://arxiv.org/abs/2503.12388 "https://arxiv.org/abs/2503.12388"))|SVCC/GTSinger.|Challenge baselines.|Fine-tune one open system and report MOS/style metrics.|Multi-GPU days to weeks.|Becomes engineering-only; subjective eval insufficient.|**Reject as main thesis.**|
|11|**Train a new unified speech/singing foundation model**|Can we build a new foundation model for speech and singing with disentangled content/melody/timbre?|This is where the field is going, but not feasible for a few GPUs.|**Low for you**, because scale dominates.|Vevo1.5, UniVoice, NaturalSpeech 3. ([Hugging Face](https://huggingface.co/amphion/Vevo1.5 "https://huggingface.co/amphion/Vevo1.5"))|Tens of thousands of hours.|Large baselines.|Impossible to evaluate fairly at small scale.|Massive.|Underpowered model; no credible comparison.|**Reject.**|

---

## 8. Recommended thesis framing

The strongest title shape would be something like:

**“What Does Vocal Identity Mean Across Speech and Singing? A Frozen-Representation Study of Timbre, Technique, and Style Leakage.”**

A strong thesis could have three papers/chapters:

1. **Representation map:** Which frozen layers/codebooks preserve identity across speech↔singing?
    
2. **Leakage study:** Which representations leak technique/style/language?
    
3. **Light steering/evaluation:** Can removing or adding a learned technique direction improve style control or predict conversion failure?
    

This is feasible, novel, and defensible. It gives you experimental depth without competing with industry-scale generation.

---

## 9. Short reading list by topic

### Singing voice conversion and style conversion

Read first: **SVCC 2023**, **SVCC 2025 official page + analysis**, **Serenade**, **Vevo1.5**, **S²Voice**. These tell you what the benchmark thinks is solved and unsolved. ([arXiv](https://arxiv.org/abs/2306.14422 "https://arxiv.org/abs/2306.14422"))

Then skim as systems/baselines: **CoMoSVC**, **SaMoye**, **HQ-SVC**, **SoulX-Singer-SVC**. ([arXiv](https://arxiv.org/abs/2401.01792 "https://arxiv.org/abs/2401.01792"))

### Singing style, technique, and controllable SVS

Read: **GTSinger**, **TCSinger**, **TCSinger2**, **TechSinger**, **VocalSet**. ([NeurIPS Proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/023d2c1a17cf35b11a0cbb43a0677c91-Abstract-Datasets_and_Benchmarks_Track.html "https://proceedings.neurips.cc/paper_files/paper/2024/hash/023d2c1a17cf35b11a0cbb43a0677c91-Abstract-Datasets_and_Benchmarks_Track.html"))

### Speech-to-singing and singing-to-speech

Read: **SVPT**, **Everyone-Can-Sing**, **Singing-to-Speech with Generative Flow**, plus NHSS/NUS-48E dataset papers/pages. ([arXiv](https://arxiv.org/abs/2406.02429 "https://arxiv.org/abs/2406.02429"))

### SSL/audio-codec factorization and disentanglement

Read: **ContentVec**, **NaturalSpeech 3**, **FACodec**, **Vevo**, **RefXVC**. ([Proceedings of Machine Learning Research](https://proceedings.mlr.press/v162/qian22b.html "https://proceedings.mlr.press/v162/qian22b.html"))

### Quality, skill, and technique evaluation

Read: **SingMOS-Pro**, **SingMOS predictor repo**, **VoiceMOS 2024**, **ASSE ISMIR 2024**, and **SVDD 2024** for responsible evaluation context. ([arXiv](https://arxiv.org/html/2510.01812 "https://arxiv.org/html/2510.01812"))

---

## 10. Explicit warnings about overclaims

**Do not claim “disentanglement” from architecture diagrams.** Claim only what you test: invariance, leakage, separability, controllability, or causal intervention.

**Do not treat speaker embeddings as ground truth identity.** Speech speaker verifiers can fail on singing, pitch range, language, vibrato, register, breathiness, and recording conditions.

**Do not claim same-person cross-lingual timbre shift unless the same singer actually sings multiple languages.** GTSinger is multilingual, but you need to inspect singer-language metadata before making same-person claims. ([NeurIPS Proceedings](https://proceedings.neurips.cc/paper_files/paper/2024/hash/023d2c1a17cf35b11a0cbb43a0677c91-Abstract-Datasets_and_Benchmarks_Track.html "https://proceedings.neurips.cc/paper_files/paper/2024/hash/023d2c1a17cf35b11a0cbb43a0677c91-Abstract-Datasets_and_Benchmarks_Track.html"))

**Do not claim MOS prediction replaces listening tests.** SVCC 2025 explicitly says objective metrics are not yet true replacements for subjective scores. ([arXiv](https://arxiv.org/abs/2509.15629 "https://arxiv.org/abs/2509.15629"))

**Do not claim “style conversion” from a style classifier alone.** A classifier can reward artifacts. Use multiple evidence streams: F0 dynamics, spectral/aperiodicity features, content preservation, singer similarity, and listening tests.

**Do not compare your small model directly against private-data systems as if it is fair.** Vevo1.5/S²Voice/SoulX-style systems may use huge public/private corpora and large models. Use them as frozen baselines or analysis targets, not as fair training comparisons. ([Hugging Face](https://huggingface.co/amphion/Vevo1.5 "https://huggingface.co/amphion/Vevo1.5"))

**Do not ignore consent and licensing.** Serenade’s repo explicitly prohibits impersonating singers without consent and disallows commercial use under its license. ([GitHub](https://github.com/lesterphillip/serenade "https://github.com/lesterphillip/serenade"))

**Do not make “singing skill” a single scalar unless your rubric is explicit.** Skill can mean pitch accuracy, rhythm accuracy, tone quality, breath control, diction, vibrato control, technique appropriateness, or stylistic authenticity. A publishable evaluator needs a declared construct and validation split.

My ranked recommendation: start with **Direction 1 + Direction 2**, then add either **Direction 3** or **Direction 5** as the thesis’s applied payoff.