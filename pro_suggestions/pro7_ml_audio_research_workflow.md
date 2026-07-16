I want to learn how to do this kind of ML/audio research properly. Use my
speech/singing timbre-shift idea as the running example.

Idea:

Use frozen speech/audio representations to study same-person speech-vs-singing
identity shift, then maybe improve speech-reference singing voice conversion by
modeling a stable identity core plus a mode-specific residual.

Constraints:

few GPUs;
need novelty;
need careful experiments;
want to use AI coding agents without letting them hallucinate;
want to learn the whole research process from idea to paper.
Please teach the entire workflow:

how to turn a vague intuition into a research question;
how to scan literature;
how to build a novelty matrix;
how to choose data;
how to choose baselines;
how to write hypotheses;
how to design a pilot;
how to detect confounds;
how to design the main experiment;
how to plan ablations;
how to design downstream intervention;
how to decide whether subjective evaluation is needed;
how to interpret mixed results;
how to write honest claims;
how to decide whether to pivot.
For each step, include:

concrete deliverable;
example for my case;
bad version;
good version;
what an AI coding/research assistant can help with;
what I must verify myself.
End with a template checklist I can reuse for future research ideas.

## Your idea, sharpened

A strong version of your project is:

**Use frozen speech/audio embeddings as measuring instruments to quantify how a person’s vocal identity representation changes between speech and singing, then test whether a lightweight “speech-to-singing identity residual” improves speech-reference singing voice conversion.**

The key is not to claim “identity is a vector” too early. Treat that as a **testable modeling assumption**:

[  
z_{i,m,u}^{(\ell)} = f_\ell(x_{i,m,u})  
]

where (i) is person, (m \in {\text{speech}, \text{singing}}), (u) is utterance/song segment, and (f_\ell) is a frozen model layer. Your proposed decomposition is:

[  
z_{i,m} \approx c_i + r_m + \epsilon_{i,m}  
]

where (c_i) is a stable person-specific vocal identity core, (r_m) is a mode-specific speech/singing residual, and (\epsilon) captures content, pitch, recording, performance, and noise.

This is timely but not empty space. Frozen speech representations such as wav2vec 2.0, HuBERT, and WavLM are well-established; WavLM explicitly targets “full-stack” speech processing and notes that speech contains speaker identity, content, and paralinguistic information. Music/audio representations such as MERT are also relevant because singing has musical pitch and timbral structure that speech SSL models may not handle well. ([arXiv](https://arxiv.org/abs/2006.11477 "[2006.11477] wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations"))

For data, there are useful paired or same-speaker speech/singing resources: NUS-48E has sung and spoken lyrics from 12 subjects; NHSS has 100 songs sung and spoken by 10 singers, totaling about 7 hours; JVS-MuSiC has 100 singers whose reading voices are in the JVS corpus, plus a common song and pitch/tempo-normalized variants. ([Sound & Music Computing Lab](https://smcnus.comp.nus.edu.sg/archive/pdf/2012-2013/2013_05-Pub-NUS-48E.pdf "PaperManuscript_Revision"))

For downstream conversion, the space is already active. SVCC 2023 included a cross-domain task where source singing is converted toward a target **speech** speaker, and the challenge reported that cross-domain SVC is harder than in-domain SVC, especially for similarity. Recent speech-reference SVC/SVS papers are close to your idea: _Everyone-Can-Sing_ uses speech samples to control voice identity in zero-shot singing synthesis/conversion, and a 2025 Interspeech paper specifically studies speech-prompted SVC via speaker–singer embedding adaptation. ([Singing Voice Conversion Challenge](https://vc-challenge.org/svcc2023/index.html "Singing Voice Conversion Challenge 2023"))

So the novelty should **not** be “speech-reference singing voice conversion exists.” A safer novelty target is:

**A controlled, model-agnostic study of same-person speech-vs-singing identity displacement in frozen representations, with a lightweight residual intervention tested against strong speech-reference SVC baselines.**

Also, because voice cloning touches identity, consent, and misuse risks, keep an ethics/data-rights section from the beginning, not as an afterthought. Recent work on voice actors and synthetic voice datasets emphasizes privacy, reputation, accountability, consent, credit, and compensation risks around vocal identity reuse. ([arXiv](https://arxiv.org/abs/2507.16247 "[2507.16247] PRAC3 (Privacy, Reputation, Accountability, Consent, Credit, Compensation): Long Tailed Risks of Voice Actors in AI Data-Economy"))

---

# The full workflow

## 1. Turn a vague intuition into a research question

**Concrete deliverable:** a one-page problem formulation with variables, measurable quantities, and 2–3 research questions.

**Example for your case:**

Vague intuition:

> A person sounds like themselves when speaking and singing, but not exactly. Maybe frozen embeddings can reveal the shift, and maybe that shift can help speech-reference SVC.

Research questions:

1. **Representation question:** In frozen speech/audio embedding spaces, is a same-person speech-to-singing shift systematic, layer-dependent, and smaller than different-person variation?
    
2. **Decomposition question:** Can embeddings be decomposed into a stable person core plus a speech/singing residual better than naive speaker-only or mode-only models?
    
3. **Intervention question:** Does applying a learned speech-to-singing residual to a speech reference improve speech-reference SVC similarity without hurting naturalness or musicality?
    

**Bad version:**

> I will use wav2vec to show that speech and singing identity are different, then improve SVC.

Why bad: it assumes the answer, does not define “identity,” does not say how it will be measured, and mixes analysis with system-building.

**Good version:**

> Given paired or same-speaker speech/singing recordings, I will measure whether frozen embedding spaces preserve same-person similarity across mode, quantify the direction and variance of speech-to-singing displacement, and test whether a learned residual adaptation improves speech-reference SVC relative to unadapted speech-reference and singing-reference oracle baselines.

**What an AI assistant can help with:** generate candidate formulations, turn them into variables, propose alternative hypotheses, and draft a one-page project memo.

**What you must verify yourself:** whether the question is actually answerable with available data, whether “identity” is operationalized honestly, and whether the novelty survives close prior work.

---

## 2. Scan the literature without drowning

**Concrete deliverable:** an annotated bibliography grouped by role, not by chronology.

For this project, use four buckets:

1. **Frozen representation papers:** wav2vec 2.0, HuBERT, WavLM, Whisper-style encoders, ECAPA-TDNN/x-vectors, MERT/BEATs/CLAP.
    
2. **Speech-vs-singing datasets and analysis:** NUS-48E, NHSS, JVS/JVS-MuSiC.
    
3. **Singing voice conversion and speech-reference SVC:** SVCC 2023, Everyone-Can-Sing, SSANSVC, Seed-VC-style systems, diffusion/recognition-synthesis SVC systems.
    
4. **Evaluation and ethics:** speaker similarity, naturalness, MOS/ABX/MUSHRA-style tests, consent and vocal identity risks.
    

**Example for your case:**

A good first search plan:

> “speech singing same speaker corpus”;  
> “speech-prompted singing voice conversion”;  
> “speaker singer embedding adaptation”;  
> “self-supervised speech representation speaker identity singing”;  
> “singing voice conversion challenge cross-domain speech target”;  
> “speaker verification singing voice cross-domain”.

**Bad version:**

> Read 40 arXiv papers and keep notes like “interesting,” “maybe useful,” “good baseline.”

**Good version:**

For every paper, extract only:

|Field|Question|
|---|---|
|Task|What problem does it solve?|
|Data|Does it use same people in speech and singing?|
|Representation|What is treated as identity/timbre?|
|Confounds|Does it control content, pitch, tempo, recording?|
|Evaluation|Objective, subjective, or both?|
|Relevance|Dataset, baseline, method, or warning?|
|Gap|What exactly does it not answer?|

**What an AI assistant can help with:** summarize abstracts, build BibTeX, create paper tables, suggest search terms, cluster related work.

**What you must verify yourself:** paper claims, dataset access, whether code actually runs, whether the paper’s “speech reference” setting matches yours, and whether AI summaries invented claims. Always read the abstract, method, experiments, and limitations yourself.

---

## 3. Build a novelty matrix

**Concrete deliverable:** a novelty matrix with rows as prior work and columns as dimensions of your contribution.

**Example novelty matrix for your project:**

|Prior area / paper type|What it gives you|What it may not settle|Your possible novelty|
|---|---|---|---|
|NUS-48E / NHSS|Paired sung/spoken lyrics, useful for controlled comparison|Small speaker counts; mostly acoustic or corpus-focused analysis|Apply modern frozen embeddings and identity probes|
|JVS + JVS-MuSiC|100 same speakers with speech and singing; common song and pitch/tempo variants|Japanese-only; limited songs per singer|Stronger same-person cross-mode identity analysis|
|wav2vec 2.0 / HuBERT / WavLM|Frozen speech representations|Not designed specifically for singing identity shift|Layer-wise speech-vs-singing identity geometry|
|MERT / BEATs / CLAP|Audio/music representations|May encode musical attributes more than person identity|Compare speech-trained vs music/audio-trained spaces|
|SVCC 2023|Public SVC benchmark and cross-domain speech-target task|Challenge systems optimize conversion, not necessarily representation analysis|Use challenge-style evaluation but ask why speech reference fails|
|Everyone-Can-Sing / SSANSVC|Very close: speech-reference singing synthesis/conversion|May not provide a controlled representation study of same-person identity displacement|Analysis-first residual model, then lightweight intervention|
|SpSiVC-style unified speech/singing VC|Shows speech/singing unified conversion and F0 constraints matter|Focuses on conversion/pitch constraints, not stable identity core|Separate identity core from mode residual and confounds|

**Bad version:**

> Nobody has studied this before.

This is almost always false or too broad.

**Good version:**

> Prior work has studied speech/singing acoustic differences, speech-reference SVC, and speaker/singer adaptation. The open gap is a controlled, layer-wise study of same-person speech-to-singing identity displacement in frozen representations, tied to a minimal residual intervention and evaluated against speech-reference SVC baselines.

**What an AI assistant can help with:** create the initial matrix, find missing columns, detect “too similar” papers, and produce BibTeX.

**What you must verify yourself:** the novelty claim. This is your responsibility. Read the closest papers fully, especially Everyone-Can-Sing and SSANSVC, because they are near your proposed downstream direction.

---

## 4. Choose data

**Concrete deliverable:** a data card and split plan.

Your data card should include:

- dataset name and citation;
    
- number of speakers/singers;
    
- language;
    
- speech/singing pairing;
    
- whether lyrics/content match;
    
- recording conditions;
    
- license and allowed use;
    
- train/dev/test split;
    
- known confounds;
    
- whether audio can be redistributed;
    
- whether synthetic outputs are ethically allowed.
    

**Example for your case:**

A practical few-GPU plan:

**Pilot data:** NHSS or NUS-48E. NHSS is attractive because it has sung and spoken versions of English pop-song lyrics by the same singers; NUS-48E is smaller but phonetically annotated and explicitly designed for sung/spoken lyric comparison. ([HltNUS](https://hltnus.github.io/NHSSDatabase/ "NHSS: A Speech and Singing Parallel Database"))

**Main representation data:** JVS + JVS-MuSiC. JVS-MuSiC has 100 singers whose reading voices are in JVS, and it includes a common song plus modified pitch/tempo variants, which is unusually useful for separating person identity from song/pitch/tempo confounds. ([Google Sites](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_music "Shinnosuke Takamichi (高道 慎之介) - jvs_music"))

**Downstream data:** SVCC 2023-style cross-domain setting or a small subset of same-speaker data where you can test speech-reference vs singing-reference conversion. SVCC 2023 explicitly included cross-domain SVC with target speech speakers and listening tests using natural audio references. ([Singing Voice Conversion Challenge](https://vc-challenge.org/svcc2023/index.html "Singing Voice Conversion Challenge 2023"))

**Bad version:**

> Use any singing dataset and any speech dataset, then compare embeddings.

Why bad: speaker identity, language, microphone, content, and recording environment become hopelessly confounded.

**Good version:**

> Use same-person speech/singing data first. Prefer matched lyrics when possible. Add a larger same-speaker dataset second. Keep a held-out speaker split. Never let the same speaker appear in both train and test for downstream residual learning unless the experiment is explicitly within-speaker adaptation.

**What an AI assistant can help with:** write download scripts, generate manifests, compute durations, detect sample rates, create speaker splits, flag missing files.

**What you must verify yourself:** license, consent, whether speaker IDs truly correspond across speech and singing, whether recordings are clean, whether there is accompaniment leakage, and whether your usage is ethically defensible.

---

## 5. Choose baselines

**Concrete deliverable:** a baseline table with “cheap,” “standard,” “strong,” and “oracle” baselines.

**Example for your case:**

For the **representation study**:

|Baseline|Purpose|
|---|---|
|MFCC + statistics|Old-school acoustic sanity check|
|x-vector / ECAPA-TDNN|speaker-verification baseline|
|wav2vec 2.0 / HuBERT|speech SSL baselines|
|WavLM|speech SSL with speaker/paralinguistic relevance|
|MERT / BEATs|music/audio representation comparison|
|random speaker labels|leakage sanity check|
|same-mode speaker verification|upper-bound-ish comparison|
|cross-mode speaker verification|main stress test|

ECAPA-TDNN is a reasonable speaker embedding baseline because it was designed for speaker verification and improved over TDNN/x-vector-style systems. ([arXiv](https://arxiv.org/abs/2005.07143 "[2005.07143] ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification"))

For the **downstream intervention**:

|Baseline|Purpose|
|---|---|
|speech reference, no adaptation|main baseline|
|singing reference oracle|upper bound for reference mode|
|global residual: (z_\text{speech} + \Delta)|simplest version of your idea|
|speaker-conditioned residual|stronger but still lightweight|
|random residual / wrong-speaker residual|sanity check|
|SSAN/Everyone-Can-Sing/Seed-VC-style system if runnable|strong recent baseline|

**Bad version:**

> Compare my method only to a naive SVC model.

**Good version:**

> Include cheap baselines, frozen-representation baselines, strong recent systems, and oracle references. If your method only beats the weak baseline, say that.

**What an AI assistant can help with:** identify runnable repositories, write wrapper scripts, standardize inputs/outputs, extract features, produce tables.

**What you must verify yourself:** that baselines are fairly tuned, that checkpoints are legitimate, that licenses permit use, that all systems use the same test set, and that no baseline is accidentally handicapped.

---

## 6. Write hypotheses before experiments

**Concrete deliverable:** a hypothesis sheet with directional predictions, metrics, and failure criteria.

**Example hypotheses:**

**H1: Cross-mode identity preservation.**  
For a good identity-preserving embedding layer, same-person speech–singing distance should be lower than different-person speech–singing distance:

[  
d(z_i^\text{speech}, z_i^\text{sing}) < d(z_i^\text{speech}, z_j^\text{sing})  
]

for (i \neq j).

**H2: Identity shift is layer-dependent.**  
Lower or middle layers may encode acoustics and pitch differently from higher layers; speaker identity may peak in different layers depending on model family.

**H3: Mode residual is systematic but not purely identity.**  
The average displacement (\Delta = E_i[z_i^\text{sing} - z_i^\text{speech}]) should explain some cross-mode shift, but residual variance will correlate with pitch range, vowel distribution, tempo, and singer style.

**H4: Residual adaptation helps only under controlled conditions.**  
A speech-to-singing residual should improve target similarity for speech-reference SVC, but may hurt naturalness if it injects pitch/style artifacts into the reference embedding.

**Bad version:**

> I hypothesize that my model will sound better.

**Good version:**

> I hypothesize that residual-adapted speech references will improve speaker/singer similarity over unadapted speech references, while naturalness will be statistically indistinguishable from baseline or degrade by less than a predeclared margin.

**What an AI assistant can help with:** rewrite vague hypotheses into falsifiable ones, suggest metrics, generate null hypotheses.

**What you must verify yourself:** whether the hypotheses are meaningful, whether they can be falsified, and whether the metrics actually correspond to your claims.

---

## 7. Design a pilot

**Concrete deliverable:** a pilot report with plots, metrics, failure notes, and a go/no-go decision.

**Pilot goal:** learn whether the idea has signal before building a system.

**Example pilot:**

Use 5–10 speakers from NHSS or JVS/JVS-MuSiC.

Run:

1. preprocess audio to common sample rate;
    
2. extract frozen embeddings from several layers;
    
3. pool embeddings per utterance and per speaker;
    
4. compute same-speaker cross-mode distances;
    
5. compute different-speaker cross-mode distances;
    
6. train linear probes for speaker and mode;
    
7. compute speaker verification AUC/EER for:
    
    - speech → speech;
        
    - singing → singing;
        
    - speech → singing;
        
8. plot distances with confidence intervals;
    
9. manually listen to outliers.
    

**Bad version:**

> Train a full SVC model immediately.

**Good version:**

> Spend one week testing whether frozen embeddings show cross-mode identity structure. If no representation preserves same-person identity above chance, do not build the downstream model yet.

**What an AI assistant can help with:** write feature extraction scripts, make a reproducible notebook, implement metrics, generate plots, write unit tests using synthetic embeddings.

**What you must verify yourself:** audio quality, speaker labels, whether plots match the actual files, whether outliers are mislabeled, and whether the pilot result is robust enough to justify the main experiment.

---

## 8. Detect confounds

**Concrete deliverable:** a confound ledger.

For your project, the main confounds are:

|Confound|Why it matters|Control|
|---|---|---|
|Lyrics/content|embeddings may cluster by phonetic content|matched lyrics, content-balanced splits|
|Language|speech and singing may differ by dataset/language|avoid mixing languages in core analysis|
|F0/pitch range|singing has wider and more stable pitch|F0-normalized controls, pitch/tempo variants|
|Tempo/duration|singing stretches vowels|duration-matched segments, tempo-normalized data|
|Recording condition|mic/room may identify dataset, not person|same corpus first, loudness normalization|
|Accompaniment|music leakage can dominate embeddings|vocals-only data, source separation audit|
|Gender/register|pitch may masquerade as identity|stratify by gender/register|
|Singer skill|trained singers may shift differently|record metadata when available|
|Same song leakage|model may learn song identity|held-out song and held-out speaker splits|
|Segment length|short clips may lose identity signal|test multiple durations|

**Bad version:**

> Normalize loudness and assume it is fine.

**Good version:**

> For every claimed effect, identify at least one control that could make the effect disappear. Then run that control.

**What an AI assistant can help with:** compute F0 distributions, duration histograms, phoneme/lyric overlap, loudness, SNR proxies, and train confound classifiers.

**What you must verify yourself:** whether the confound is conceptually fatal, whether the automatic estimates are reliable, and whether the control actually tests the confound.

---

## 9. Design the main experiment

**Concrete deliverable:** a frozen experimental protocol before running final tests.

A good main experiment has three parts.

### A. Representation geometry

For each model (f), layer (\ell), and pooling method:

- compute (z_i^\text{speech});
    
- compute (z_i^\text{sing});
    
- compute same-person cross-mode distance;
    
- compute different-person cross-mode distance;
    
- compute within-speech and within-singing distances;
    
- report effect sizes and confidence intervals.
    

Main test:

[  
d_\text{same cross-mode} < d_\text{different cross-mode}  
]

and

[  
d_\text{same cross-mode} > d_\text{same within-mode}  
]

The first says identity survives mode shift. The second says mode shift is real.

### B. Decomposition test

Compare models:

1. **speaker-only:** (z \approx c_i)
    
2. **mode-only:** (z \approx r_m)
    
3. **speaker + mode:** (z \approx c_i + r_m)
    
4. **speaker + speaker-specific mode residual:** (z \approx c_i + r_{i,m})
    

Ask whether the additive residual explains variance beyond speaker-only and mode-only models.

### C. Confound controls

Run the same tests under:

- matched lyrics where available;
    
- pitch/tempo-normalized variants where available;
    
- fixed segment length;
    
- speaker-held-out splits;
    
- gender-stratified analysis;
    
- dataset-held-out analysis if combining corpora.
    

**Bad version:**

> Show a UMAP where speech and singing separate.

UMAP can be useful exploration, but it is not proof.

**Good version:**

> Use UMAP only as a diagnostic. Main claims come from predeclared distance tests, verification metrics, linear probes, mixed-effects models, and confidence intervals.

**What an AI assistant can help with:** implement the pipeline, generate config files, run batch extraction, make reproducible figures, and produce metric tables.

**What you must verify yourself:** split correctness, metric definitions, statistical interpretation, and whether conclusions survive controls.

---

## 10. Plan ablations

**Concrete deliverable:** an ablation table with priority and compute cost.

**Example ablation plan:**

|Ablation|Question|Priority|
|---|---|---|
|model family|speech SSL vs speaker encoder vs music/audio SSL|high|
|layer|where is identity most stable?|high|
|pooling|mean, attentive pooling, voiced-only pooling|high|
|segment length|how much audio is needed?|high|
|content matching|is the effect just lyrics/phones?|high|
|F0 normalization|is the effect just pitch?|high|
|tempo normalization|is the effect just duration?|medium|
|gender stratification|does pitch/register dominate?|high|
|residual type|global vs speaker-conditioned vs learned adapter|high|
|dataset|NHSS/NUS vs JVS-MuSiC|high|
|downstream generator|does the effect generalize across SVC systems?|medium/high, compute permitting|

**Bad version:**

> Try every model and every parameter until something works.

**Good version:**

> Choose ablations that can falsify the core claim. Anything that only improves numbers but does not test the claim is secondary.

**What an AI assistant can help with:** run configuration sweeps, summarize logs, check missing results, and generate ablation tables.

**What you must verify yourself:** which ablations are scientifically necessary, whether the grid is fair, and whether you are overfitting your story to ablation outcomes.

---

## 11. Design the downstream intervention

**Concrete deliverable:** a minimal intervention system with a clean comparison table.

The intervention should be small. With few GPUs, avoid training a full diffusion SVC model from scratch.

### Minimal version

Estimate a global residual:

[  
\Delta = E_i[z_i^\text{sing} - z_i^\text{speech}]  
]

Then adapt a speech reference:

[  
\hat{z}_i^\text{sing-ref} = z_i^\text{speech-ref} + \alpha \Delta  
]

Tune (\alpha) on dev speakers only.

### Stronger version

Learn a tiny adapter:

[  
\hat{z}_i^\text{sing-ref} = g_\theta(z_i^\text{speech-ref}, \text{F0 stats}, \text{duration stats})  
]

where (g_\theta) is a small MLP, linear map, affine transform, or LoRA-style adapter. Keep the SVC generator frozen.

### Comparisons

|Condition|Meaning|
|---|---|
|speech reference|current practical use case|
|speech + global residual|simplest version of your idea|
|speech + learned residual|stronger intervention|
|wrong-speaker residual|sanity check|
|random residual|sanity check|
|singing reference oracle|upper bound|
|target natural singing|reference for subjective tests|

**Bad version:**

> Train a new SVC model and claim the whole architecture proves the identity-core hypothesis.

**Good version:**

> Keep the generator fixed. Change only the reference representation. Then any improvement is more plausibly connected to the residual hypothesis.

**What an AI assistant can help with:** wrap an existing SVC system, build data loaders, implement residual transforms, run inference jobs, and organize generated audio.

**What you must verify yourself:** whether the SVC model actually uses the embedding the way you think, whether audio outputs are free from artifacts, whether the residual is not leaking test speakers, and whether the intervention is ethically acceptable.

---

## 12. Decide whether subjective evaluation is needed

**Concrete deliverable:** an evaluation decision memo.

Use this rule:

**Representation-only paper:** subjective evaluation is optional.  
**Generated-audio paper:** subjective evaluation is usually needed if you claim naturalness, musicality, or perceived identity similarity.

SVCC 2023 is a warning: it used large-scale crowd-sourced listening tests and found that only a few objective measurements significantly correlated with perceptual performance, while cross-domain SVC remained especially difficult for similarity. That means objective speaker-embedding similarity alone is not enough for a strong perceptual claim.

**Example subjective design:**

- **Similarity ABX:** Given target reference and two converted samples, which sounds more like the target person?
    
- **Naturalness MOS or preference:** Which sample sounds more natural as singing?
    
- **Musicality/preference:** Which better preserves melody, pitch stability, and singing style?
    
- **Identity-vs-quality separation:** Ask separate questions; do not collapse everything into “which is better?”
    

**Bad version:**

> ECAPA cosine improved, so people will hear better identity.

**Good version:**

> Objective metrics suggest better similarity; a listener test is needed before claiming perceptual improvement.

**What an AI assistant can help with:** create a web listening-test interface, randomize trials, generate anonymized sample IDs, compute confidence intervals, and flag inattentive raters.

**What you must verify yourself:** IRB/ethics requirements, consent, rater instructions, whether samples are safe to release, whether the listening test answers the actual claim, and whether the statistics are valid.

---

## 13. Interpret mixed results

**Concrete deliverable:** a results interpretation table with “what happened,” “possible explanation,” “test,” and “claim change.”

**Examples for your case:**

|Mixed result|Possible interpretation|Follow-up|
|---|---|---|
|WavLM preserves cross-mode identity, MERT does not|speech SSL may encode speaker traits better; music SSL may emphasize pitch/music attributes|compare layer-wise probes and F0 controls|
|Same-person speech–singing distance is lower than different-person, but only for long clips|identity needs enough phonetic coverage|report duration sensitivity|
|Residual improves objective similarity but hurts naturalness|residual may push embeddings out of generator’s training manifold|reduce (\alpha), add adapter regularization|
|Residual helps men but not women, or vice versa|pitch/register confound|gender/register-stratified analysis|
|Subjective similarity does not follow ECAPA cosine|speaker verifier may be biased toward speech-like cues|report metric disagreement honestly|
|Only matched-lyrics data works|content is a major confound|narrow claim to matched content|

**Bad version:**

> The result is mixed, so choose the best metric and write the paper around it.

**Good version:**

> The result supports only part of the hypothesis. Separate “identity survives mode shift” from “the residual improves SVC.” One can be true while the other fails.

**What an AI assistant can help with:** cluster failures, find outliers, produce per-speaker reports, generate possible explanations.

**What you must verify yourself:** which explanation is actually supported, whether a result is statistically and practically meaningful, and whether your story changed after seeing the data.

---

## 14. Write honest claims

**Concrete deliverable:** a claim ledger.

Before writing the paper, make a table:

|Claim|Evidence|Figure/table|Limitation|
|---|---|---|---|
|frozen WavLM layers preserve cross-mode identity better than chance|cross-mode verification AUC / distance test|Table 2|dataset/language limited|
|singing induces systematic embedding displacement|same-person within-mode vs cross-mode distance|Fig. 3|pitch/content confounds partially controlled|
|global residual improves objective similarity|SVC metric comparison|Table 5|subjective test needed or limited|
|residual adaptation improves perceived similarity|listener ABX|Table 6|rater pool/sample size limits|

**Bad version:**

> We solve speech-reference singing voice conversion by modeling identity.

**Good version:**

> On same-speaker speech/singing corpora, frozen speech representations exhibit a measurable cross-mode identity shift. A simple residual adaptation improves [specific metric] over an unadapted speech-reference baseline, but remains below singing-reference oracle performance and is sensitive to [specific confound].

**Paper structure:**

1. **Introduction:** practical problem and scientific question.
    
2. **Related work:** speech SSL, speech/singing corpora, SVC, speech-reference SVC, evaluation.
    
3. **Problem formulation:** identity core and mode residual as testable model.
    
4. **Datasets and ethics:** data rights, splits, consent, limitations.
    
5. **Representation experiments:** frozen models, layers, probes, confounds.
    
6. **Residual intervention:** minimal adapter, baselines, ablations.
    
7. **Evaluation:** objective and subjective if applicable.
    
8. **Discussion:** what worked, what failed, what not to conclude.
    
9. **Limitations:** language, data size, embedding bias, cloning risk.
    
10. **Reproducibility:** code, configs, splits, checkpoints, generated sample policy.
    

**What an AI assistant can help with:** draft sections, enforce consistent terminology, check that each claim points to a table, produce LaTeX tables, format references.

**What you must verify yourself:** every factual claim, every citation, every number, every limitation, and every ethical statement.

---

## 15. Decide whether to pivot

**Concrete deliverable:** a go/no-go/pivot memo after pilot and after main experiments.

Use predeclared pivot rules.

### After pilot

Pivot if:

- same-person cross-mode verification is near chance for all representations;
    
- results vanish under content or pitch controls;
    
- datasets are too small or licensing is unusable;
    
- a recent paper already did the same thing with stronger evidence.
    

Possible pivots:

1. **Benchmark paper:** “How well do frozen speech/audio representations preserve identity across speech and singing?”
    
2. **Dataset/evaluation paper:** “A controlled protocol for same-speaker speech/singing identity evaluation.”
    
3. **Negative result paper:** “Speaker embeddings fail under speech-to-singing mode shift.”
    
4. **Confound paper:** “Pitch and content dominate apparent speaker identity in speech/singing embeddings.”
    
5. **Downstream-only paper:** if analysis is unsurprising but residual adaptation improves SVC.
    

### After downstream experiments

Pivot if:

- residual helps embeddings but not audio;
    
- audio improves objectively but fails subjective tests;
    
- improvements are only on one dataset;
    
- gains are smaller than a simpler baseline;
    
- the best result requires too much compute or too much hidden engineering.
    

**Bad version:**

> Keep adding tricks until there is a positive result.

**Good version:**

> Decide whether the contribution is representation analysis, evaluation methodology, a lightweight intervention, or a negative result. Do not force all three into one paper if only one is strong.

**What an AI assistant can help with:** summarize evidence, draft decision memos, compare options, list missing experiments.

**What you must verify yourself:** strategic judgment, novelty, community fit, ethical risk, and whether the remaining work is worth your time.

---

# A few-GPU execution strategy

Your compute-friendly plan should be:

1. **Phase 1: no training.** Precompute frozen embeddings. Run representation analysis, probes, and confound controls.
    
2. **Phase 2: tiny models only.** Train linear probes, small adapters, or residual maps. Keep large encoders and generators frozen.
    
3. **Phase 3: downstream intervention.** Use an existing SVC/SVS model if legally and technically usable. Change only the reference representation.
    
4. **Phase 4: subjective evaluation only when needed.** Do not pay for listening tests until objective experiments and ablations identify the best few systems.
    
5. **Phase 5: paper.** Lead with the strongest verified contribution. Do not promise a full new SVC system if the real contribution is analysis.
    

---

# How to use AI coding/research assistants safely

Use AI agents as **junior implementers and auditors**, not as sources of truth.

A good agent task prompt:

> Implement `extract_embeddings.py` for the datasets listed in `manifest.csv`. Inputs: audio path, speaker ID, mode, dataset, split. Outputs: one `.npy` file per utterance and a `features_manifest.csv`. Use only the models listed in `config.yaml`. Add a synthetic-data unit test. Do not change split files. Do not make literature claims. Report skipped files and errors.

Require agents to produce:

- exact command used;
    
- environment file;
    
- input/output schema;
    
- checksums or file counts;
    
- tests;
    
- metric definitions;
    
- assumptions;
    
- failure cases.
    

Never let agents silently decide:

- train/test splits;
    
- which paper is “closest”;
    
- which samples to exclude;
    
- which metric is primary;
    
- whether a result is significant;
    
- whether generated audio is ethically acceptable;
    
- whether a claim is novel.
    

Use a second “red-team” agent to ask:

> Find leakage, confounds, unfair baselines, incorrect citations, and unsupported claims in this experiment.

Then verify the red-team output yourself.

---

# Reusable research checklist template

Copy this for future ideas.

## 0. Idea capture

- Intuition:
    
- Practical motivation:
    
- Scientific question:
    
- Why now:
    
- Why me / why feasible:
    
- Few-GPU strategy:
    

## 1. Research question

- Main RQ:
    
- Secondary RQs:
    
- Operational definitions:
    
- Unit of analysis:
    
- Primary metric:
    
- Failure criterion:
    

## 2. Literature scan

- Seed papers:
    
- Closest papers:
    
- Dataset papers:
    
- Baseline papers:
    
- Evaluation papers:
    
- Ethics/legal papers:
    
- What prior work already solves:
    
- What prior work does not solve:
    

## 3. Novelty matrix

|Prior work|Task|Data|Method|Evaluation|Gap|Threat to novelty|
|---|---|---|---|---|---|---|

Final novelty claim:

## 4. Data plan

- Dataset:
    
- License:
    
- Consent constraints:
    
- Speakers:
    
- Modes/conditions:
    
- Matching variables:
    
- Splits:
    
- Exclusions:
    
- Confounds:
    
- Data card complete: yes/no
    

## 5. Baselines

|Baseline|Why included|Implementation|Compute|Expected strength|
|---|---|---|---|---|

Weak baseline:  
Strong baseline:  
Oracle:  
Sanity-check baseline:

## 6. Hypotheses

|Hypothesis|Direction|Metric|Test|What would falsify it|
|---|---|---|---|---|

## 7. Pilot

- Minimal dataset:
    
- Minimal models:
    
- Primary plot:
    
- Primary metric:
    
- Time budget:
    
- Go/no-go rule:
    
- Pilot result:
    
- Decision:
    

## 8. Confound ledger

|Confound|Risk|Diagnostic|Control|Result|
|---|---|---|---|---|

## 9. Main experiment

- Frozen protocol:
    
- Splits locked:
    
- Models locked:
    
- Metrics locked:
    
- Statistical test:
    
- Confidence intervals:
    
- Multiple-comparison plan:
    
- Reproducibility artifacts:
    

## 10. Ablations

|Ablation|Claim tested|Priority|Compute|Result|
|---|---|---|---|---|

## 11. Downstream intervention

- Intervention:
    
- What is frozen:
    
- What is trained:
    
- Baselines:
    
- Oracle:
    
- Objective metrics:
    
- Subjective metrics:
    
- Failure modes:
    

## 12. Subjective evaluation decision

- Are generated samples involved?
    
- Are perceptual claims made?
    
- Listener task:
    
- Rater criteria:
    
- Sample selection:
    
- Ethics/consent:
    
- Statistical plan:
    

## 13. Mixed-result interpretation

|Result|Supports|Contradicts|Alternative explanation|Follow-up|
|---|---|---|---|---|

## 14. Claim ledger

|Paper claim|Evidence|Figure/table|Limitation|
|---|---|---|---|

Forbidden claims:

## 15. Pivot decision

- Continue because:
    
- Pivot because:
    
- Strongest remaining contribution:
    
- Weakest link:
    
- Next 2-week milestone:
    
- Stop condition: