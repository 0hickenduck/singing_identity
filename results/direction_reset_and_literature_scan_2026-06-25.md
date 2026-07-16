# Direction Reset and Literature Scan

Generated: 2026-06-25 JST

## Executive Summary

We have completed the original Stage 1 frozen-representation audit and several Seed-VC intervention probes. The project is not stuck because the early experiments failed; it is stuck because the initial story, "predict singing timbre from speech," became scientifically underspecified after we reached the decoder/listening stage.

The clearest current conclusion is:

> Speech-to-singing residual is real in multiple representations, but in Seed-VC it does not appear to be controlled by one clean, audible native component. We should stop scaling Seed-VC component mapping for now and pivot to residual factorization: quantify how much of the speech-to-singing shift is explained by F0/prosody/acoustics, speaker-style embeddings, and singing technique labels before doing more audio synthesis.

This report has three purposes:

1. State where we are now.
2. Reconstruct the prior plan and what went wrong or became unclear.
3. Summarize recent literature and propose new, cleaner research directions.

## 1. Where We Are Now

### Completed Infrastructure

The repaired Stage 1 run completed successfully.

- Run root: `/localdisk/bowen/singing_identity/runs/stage1_repaired_200_fresh_local`
- Feature root: `/localdisk/bowen/singing_identity/features/stage1_repaired_200_fresh_local`
- Data: 6016 utterances, 20 speakers, 4000 speech/singing pairs
- Models extracted: acoustic baseline, WavLM Base+, HuBERT base, MERT v1 95M
- Stage 1 report: `results/stage1_repaired_200_fresh_local_report.md`
- Job status: 100 successful jobs, 0 failed jobs

The repaired data run removed the earlier missing-wav blocker. All heavy features and generated runs are under `/localdisk/bowen/singing_identity`, while code/docs/notebooks live under `/home/bowen/bowen_lab/projects/singing_identity`.

### Stage 1 Track 1: Speech-to-Singing Residual

Stage 1 showed that speech and singing are strongly separable in frozen representations, even after nuisance residualization.

Key Track 1 table excerpts:

| Model | Layer | Controlled sep AUC | Shuffled sep AUC | R@1 | R@5 | Mapper cosine | Global residual cosine |
|---|---:|---:|---:|---:|---:|---:|---:|
| acoustic | frame25ms | 0.739 | 0.514 | 0.350 | 0.600 | 0.617 | 0.629 |
| wavlm | 6 | 0.902 | 0.513 | 0.050 | 0.450 | 0.790 | 0.738 |
| wavlm | 9 | 0.839 | 0.524 | 0.100 | 0.550 | 0.797 | 0.731 |
| hubert | 12 | 0.792 | 0.512 | 0.300 | 0.650 | 0.729 | 0.637 |
| mert | 12 | 0.826 | 0.513 | 0.100 | 0.400 | 0.718 | 0.543 |

Interpretation:

- Speech vs. singing separability is real, but it is not enough by itself because acoustic-only features also separate modes.
- WavLM L6/L9 have strong residual mapper cosine, but weak cross-mode retrieval.
- HuBERT and acoustic baselines sometimes preserve identity better than WavLM.
- The Stage 1 "mode residual" is not pure timbre. It likely includes F0 range, voicing, duration, energy, phonation, and technique.

### Stage 1 Track 2: Technique Directions

Track 2 found many reliable technique groups, but the clean evidence is concentrated in imbalanced parts of GTSinger.

Important facts:

- GTSinger technique examples are dominated by `breathy` and `control`.
- Many "passed groups" have tiny speaker/language coverage or near-zero analogy scores.
- MERT L3 breathy remained the strongest narrow candidate, but broad technique steering was not justified.

Interpretation:

Track 2 is promising only if narrowed to a specific technique with enough examples, likely breathy first. Broad technique steering claims would currently be too weak.

## 2. What The Previous Exploration Plan Was

The original project had two parallel tracks.

### Track 1: Timbre / Mode Residual

Original goal:

> Estimate a speech-to-singing residual, `Delta = T_singing - T_speech`, and use it to improve speech-reference zero-shot SVC.

Planned steps:

1. Build paired same-person speech/singing manifests.
2. Extract frozen features from WavLM, HuBERT, MERT, acoustic baseline.
3. Run mode probes, cross-mode speaker retrieval, residual controls, and MicroMapper.
4. Feed mapped embeddings into a frozen decoder or SVC system.
5. Ask whether speech references can be adapted toward singing references.

### Track 2: Singing Technique

Original goal:

> Find stable directions for singing techniques in frozen SSL representations and steer downstream synthesis.

Planned steps:

1. Use GTSinger phone-level technique pairs.
2. Compute same-phone technique deltas.
3. Bootstrap direction stability.
4. Run wrong-technique and shuffled-direction controls.
5. If stable, inject or steer a frozen decoder.

### Why The Original Plan Was Reasonable

The plan was reasonable as a first-stage audit because it asked low-cost questions:

- Is there a measurable speech/singing shift?
- Does identity survive across modes?
- Are residuals more than F0/duration/energy?
- Are technique directions stable enough to try synthesis?

These questions were answered partially positively.

## 3. What Actually Happened After Stage 1

### Seed-VC Black-Box Prompt Baseline

We cloned Seed-VC under:

`/localdisk/bowen/singing_identity/external/seed-vc`

We generated a 10-pair balanced prompt-mode batch:

`/localdisk/bowen/singing_identity/runs/track1_seedvc_prompt_baseline_10pairs_rr`

Conditions:

- same source singing
- target speech prompt
- target singing prompt

Automatic Resemblyzer proxy found:

- singing-prompt output moved toward target singing in 10/10 pairs
- singing-prompt output moved away from target speech in 10/10 pairs
- mean delta to target singing: `+0.1103`
- mean delta to target speech: `-0.0750`
- mean delta to source: `+0.0358`

Human listening: singing prompts often sounded more singing-like and generally useful.

Interpretation:

Seed-VC responds to prompt mode. That supported deeper investigation.

### Seed-VC Native Latent Audit

We extracted Seed-VC prompt-side native latents for 200 speech/singing pairs:

`/localdisk/bowen/singing_identity/runs/seedvc_native_latents_all200`

Components:

- `style`: CAMPPlus embedding
- `semantic_stats`: Whisper/semantic features mean/std
- `prompt_stats`: length-regulated prompt condition mean/std
- `mel_stats`: prompt mel mean/std
- `pooled`: all components concatenated, diagnostic only

Heldout audit results:

| Component | Speech-sing cosine | Ridge mapped-to-sing cosine | Ridge delta-to-true-delta cosine | Caveat |
|---|---:|---:|---:|---|
| style | 0.451 | 0.501 | 0.519 | speech/singing strongly separated; direct mapping risky |
| semantic_stats | 0.944 | 0.976 | 0.705 | most predictable, but correlated with duration/F0/RMS |
| prompt_stats | 0.944 | 0.959 | 0.575 | secondary candidate |
| mel_stats | 0.993 | 0.990 | 0.476 | too close at baseline; likely nuisance-heavy |
| pooled | 0.986 | 0.990 | 0.866 | not clean; mixes all factors |

Important nuisance correlations for `semantic_stats` delta norm:

- duration delta: `0.608`
- F0 voiced-pct delta: `0.426`
- RMS delta: `0.406`

Interpretation:

`semantic_stats` looked mathematically promising, but not necessarily timbre-specific.

### Semantic Intervention

We synthesized 4 heldout pairs with:

- baseline speech prompt
- mapped semantic stats
- oracle singing prompt

Run:

`/localdisk/bowen/singing_identity/runs/track1_seedvc_semantic_intervention_4pairs`

Human result:

> `semantic weak`

Interpretation:

Semantic-only affine shifting produced valid audio, but did not create an obvious audible movement toward oracle singing prompt. This means predictable latent residual is not automatically a useful causal control.

### Seed-VC Component Ablation

We then ran oracle component swaps:

- `prompt_seq`: length-regulated semantic/F0 prompt condition
- `mel`: prompt mel context
- `style`: CAMPPlus style embedding

Run:

`/localdisk/bowen/singing_identity/runs/track1_seedvc_component_ablation_4pairs`

Generated conditions:

- `baseline_all_speech`
- `oracle_all_singing`
- `singing_prompt_seq_only`
- `singing_mel_only`
- `singing_style_only`
- `singing_prompt_seq_mel`
- `singing_prompt_seq_style`
- `singing_mel_style`

Human result:

> Differences were generally small. Oracle singing was still best. Among single-component swaps, style seemed largest, but only one of four examples made it clearly audible.

Interpretation:

Seed-VC's prompt-mode effect does not decompose cleanly into one obvious native component in this small batch. The effect may be distributed across prompt sequence, mel context, and style; or the audible effect may be too weak/noisy for this setup.

## 4. Problems Encountered

### Problem 1: The Scientific Question Drifted

Original phrasing:

> Can we predict singing voice from speech?

This became uncomfortable because it sounds like a generic SVC engineering task, and speech-prompted singing voice conversion already exists as a research problem.

Better phrasing:

> What factors constitute the same-person speech-to-singing identity/style residual, and which factors remain after controlling for F0, duration, energy, spectral envelope, and technique?

This is a representation-analysis question, not just a synthesis task.

### Problem 2: "Timbre" Was Too Broad

In our runs, the residual can include:

- F0 range and voiced ratio
- duration and rhythm
- RMS/energy
- spectral envelope
- phonation style, e.g. breathy
- singer identity/style
- decoder prompt effects

Calling all of this "timbre" is scientifically unsafe.

### Problem 3: Seed-VC Is Not A Clean Scientific Instrument

Seed-VC uses multiple prompt pathways:

- prompt mel context
- CAMPPlus style embedding
- length-regulated prompt condition
- source semantic/F0 condition
- diffusion transformer in-context behavior

Changing one component does not necessarily isolate a perceptual factor. The weak ablation result suggests the effect is distributed or hidden in decoder interactions.

### Problem 4: Listening Evaluation Became Hard

The user correctly noted that many differences are hard to judge. This is not a user failure; it is an experimental design issue.

For future listening checks, we need:

- fewer conditions per screen
- explicit ABX or preference tasks
- clearer effect-size targets
- better selected examples
- objective proxies used only as triage, not proof

### Problem 5: Technique Data Is Imbalanced

GTSinger is rich, but for our current slice:

- breathy dominates
- falsetto, mixed voice, pharyngeal are sparse
- many technique groups are single-language or few-speaker

Track 2 should be narrow until the dataset is expanded or balanced.

## 5. Latest Literature Scan

This scan used recent primary sources available as of 2026-06-25.

### 5.1 SVCC 2025: The Field Has Shifted Toward Singing Style Conversion

The most important field-level signal is the Singing Voice Conversion Challenge 2025 analysis.

Source:

- [An Extensive Analysis of the Singing Voice Conversion Challenge 2025 Evaluation Results](https://arxiv.org/html/2509.15629v2)

Key points:

- The challenge moved beyond singer identity conversion and added singing style conversion.
- It evaluated 33 systems with large-scale crowd-sourced listening.
- Top systems reached strong singer identity scores, but singing style and naturalness remained hard.
- The paper explicitly identifies dynamic information in breathy, glissando, and vibrato styles as a main difficulty.

Relevance to us:

This supports pivoting away from "speech predicts singing timbre" and toward "identity/style/technique residual decomposition." It also validates our discomfort: singing style is not a single static embedding.

### 5.2 S2Voice: Strong Systems Separate Style Conditioning And Timbre Conditioning

Source:

- [S2Voice: Style-Aware Autoregressive Modeling with Enhanced Conditioning for Singing Style Conversion](https://arxiv.org/html/2601.13629v1)

Key points:

- Winning SVCC 2025 system.
- Adds style embeddings to the autoregressive model through FiLM-style layer-norm conditioning and style-aware cross-attention.
- Adds a global speaker embedding into the flow-matching transformer to improve timbre similarity.
- The paper frames incomplete disentanglement between style and timbre as a core problem.

Relevance to us:

Our Seed-VC ablation is consistent with this: style and timbre are not cleanly isolated by a single native component. The stronger modern design is explicit multi-condition modeling, not one-vector replacement.

### 5.3 Speech-Prompted SVC Exists, And It Uses Speaker-Singer Embedding Adaptation

Source:

- [Bridging Speech and Singing: Multi-stage Speech-Prompted Singing Voice Conversion with Speaker Embedding Adaptation](https://www.isca-archive.org/interspeech_2025/liu25h_interspeech.html)

Key points:

- This paper directly studies conversion from speech timbre into singing.
- It proposes a Singing Speech Alignment Network (SSAN) to align speech and singing timbre embeddings.
- The paper explicitly states that it emphasizes timbre conversion between speech and singing, rather than speech-to-singing generation.
- It uses cycle training and reconstruction losses to prevent the embedding adapter from drifting.

Relevance to us:

If we want to continue Track 1 as a synthesis story, this is the closest paper. But it also means a naive "we predict singing from speech" framing is not novel enough. Our novel angle would need to be residual analysis or a smaller, better-controlled adapter/probe, not just speech-prompted SVC.

### 5.4 VibE-SVC2: Singing Style Is Coupled Across Pitch, Energy, And Timbre

Source:

- [Vibrato Expression Control for Singing Voice Conversion with Improving Independent Control](https://arxiv.org/html/2606.17126v1)

Key points:

- The paper argues that singing style attributes are dynamically coupled across pitch, energy, and timbre.
- It separates pitch-related and timbre-related components for vibrato/phonation control.
- It proposes vibrato rate scaling and subharmonic correction for difficult phonation styles.

Relevance to us:

This is highly relevant to why our residual is hard to interpret. If style is synchronized across F0, loudness, and spectral timbre, then subtracting `T_singing - T_speech` in one embedding space will mix all of them. This supports residual factorization before synthesis.

### 5.5 TechSinger: Technique Control Is Becoming Label/Detector-Based

Source:

- [TechSinger: Technique Controllable Multilingual Singing Voice Synthesis via Flow Matching](https://arxiv.org/abs/2502.12572)

Key points:

- Supports multiple languages and seven vocal techniques.
- Uses flow matching.
- Builds a technique detection model to annotate datasets with phoneme-level technique labels.
- Supports prompt-based technique specification.

Relevance to us:

This supports a Track 2 pivot: rather than treating technique as an incidental residual, explicitly model/detect technique labels and measure their directions. It also suggests that our GTSinger labels are useful, but we need stronger label balance and detection/verification.

### 5.6 InvoxSVC: Local Temporal Prompt Information Matters

Source:

- [InvoxSVC: Any-to-any Zero-shot Singing Voice Conversion with In-Context Learning in Latent Flow Matching](https://sap.ist.i.kyoto-u.ac.jp/EN/bib/intl/ZHO-ICME25.pdf)

Key points:

- Uses content, pitch, and global singer embedding as conditioning.
- Adds in-context learning by concatenating target singer sequential features with source singing features during inference.
- The authors report that timbre gains came from certain articulations and temporal information, not only from a global speaker embedding.
- They discuss the tradeoff between detailed articulation and timbre leakage.

Relevance to us:

This may explain why our global/static component ablation was weak. The singing prompt advantage may live in local temporal articulatory information, not in a single global vector or mean/std shift.

### 5.7 SSL Content Encoders Still Struggle With Timbre Leakage

Source:

- [Simple and Effective Content Encoder for Singing Voice Conversion via SSL-Embedding Dimension Reduction](https://www.isca-archive.org/interspeech_2025/zhou25e_interspeech.html)

Key points:

- Token methods can hurt content reconstruction.
- Embedding methods can leak timbre.
- The paper proposes selecting/fixing a subset of SSL channels to reduce timbre leakage while preserving content.

Relevance to us:

This supports treating WavLM/HuBERT/MERT layers and channels as mixtures, not clean content/timbre/style axes. It also suggests a concrete new method: channel subset or dimension selection for residual analysis.

### 5.8 Seed-VC: Full Reference Context Matters

Source:

- [Zero-shot Voice Conversion with Diffusion Transformers](https://arxiv.org/abs/2411.09943)

Key points:

- Seed-VC uses a diffusion transformer and entire reference context for fine-grained timbre features.
- It explicitly targets timbre leakage and train/inference mismatch.
- It extends to zero-shot SVC with F0 conditioning.

Relevance to us:

Seed-VC is good as a black-box sanity check, but because it relies on full reference context, a one-component mean/std intervention is probably too crude for clean mechanistic claims.

## 6. What New Directions Are Possible

### Direction A: Residual Factorization / Residual Accounting

This is my recommended next main direction.

Question:

> How much of `Delta = singing - speech` is explained by acoustic/prosodic factors, and how much remains as identity/style/technique residual?

Method:

1. For each paired speech/singing item, compute deltas in several spaces:
   - ECAPA / Resemblyzer / CAMPPlus speaker embeddings
   - WavLM L6/L9 and HuBERT L12
   - MERT L3 for technique-sensitive space
   - acoustic spectral stats
2. Build explanatory variables:
   - F0 mean/std/range/voiced pct
   - duration ratio
   - RMS/energy
   - spectral centroid/rolloff/MFCC/formant-like stats
   - language, phone coverage, technique label
3. Fit train-speaker-only models:
   - ridge regression
   - random forest / gradient boosting for explanatory upper bound
   - nuisance-only baseline
4. Evaluate on heldout speakers:
   - delta cosine
   - R2 / partial R2
   - residual norm after removing each factor group
   - whether speaker identity remains decodable from the residual

Output:

- A table showing the residual budget:
  - how much is F0/prosody
  - how much is energy/spectral envelope
  - how much is technique
  - how much remains unexplained but speaker-consistent

Why this is better:

It directly answers the scientific question before asking a decoder to synthesize.

### Direction B: Speaker-Singer Embedding Adapter

Question:

> Can a small adapter align speech speaker embeddings and singing speaker embeddings without synthesizing audio?

This follows the SSAN idea from Liu & Shi 2025, but we can frame it analytically:

- Inputs: speech and singing embeddings for the same person.
- Train: small alignment network with matched/mismatched cosine embedding loss and reconstruction losses.
- Evaluate:
  - same-speaker cross-mode retrieval
  - heldout speaker generalization
  - residual factorization after nuisance controls

Why this is useful:

It creates a clearer Track 1 story than Seed-VC component swapping:

> We measure and align speaker identity across vocal mode, while quantifying how much non-identity residual remains.

Risk:

It is close to existing speech-prompted SVC work, so the novelty must be in the analysis and controls, not the adapter itself.

### Direction C: Technique-First Track 2

Question:

> Are specific singing techniques linearly or locally controllable after controlling for phone/F0/energy?

Recommended target:

- breathy first, because data is available
- then vibrato, because latest literature treats vibrato as a central dynamic style problem

Method:

1. Build balanced phone/speaker/language groups.
2. Estimate technique directions in MERT, WavLM, and acoustic spaces.
3. Explicitly regress out F0 and energy.
4. Report whether the direction survives controls.
5. Only then attempt synthesis.

Why this is attractive:

SVCC 2025 and TechSinger both show that singing style/technique is a live research topic. It is more novel than generic speech-prompted SVC.

Risk:

GTSinger imbalance can create false positives. Need strict split and balanced subsets.

### Direction D: Local Temporal Prompt Analysis

Question:

> Is the singing prompt advantage carried by local temporal/articulatory information rather than global embeddings?

Method:

1. Do not pool prompt sequences to mean/std first.
2. Analyze frame/segment-level prompt condition deltas.
3. Align speech and singing prompts at phrase/phone level if possible.
4. Test whether local deltas correlate with phone class, onset/offset, high F0, vibrato regions, or breathy segments.

Why this matters:

InvoxSVC suggests in-context learning gains can come from temporal articulation detail, not global speaker embedding.

Risk:

Requires better alignment and more complex visualization. It is probably a second-phase analysis after residual accounting.

### Direction E: Evaluation Redesign

Question:

> How do we make human listening useful instead of confusing?

Plan:

- Use AB or ABX notebooks.
- Limit each screen to one comparison.
- Ask one question at a time:
  - identity closer?
  - singing-mode closer?
  - technique changed?
  - artifacts worse?
- Preselect examples where oracle difference is strong.
- Keep objective proxies next to audio, but do not ask the human to browse files manually.

This is an engineering/process direction, but it matters because the recent listening checks became cognitively overloaded.

## 7. Recommended Next Plan

### Immediate Decision

Pause Seed-VC component mapping.

Reason:

- Semantic mapping was weak by human listening.
- Component ablation did not isolate a strong single component.
- Latest literature suggests style/timbre/prosody are coupled and often require explicit multi-condition modeling.

### Next Experiment: Residual Accounting

Implement a script:

`scripts/probing/run_residual_factor_accounting.py`

Inputs:

- Stage 1 paired manifest.
- Cached features for acoustic, WavLM, HuBERT, MERT.
- Optional speaker embeddings from CAMPPlus/ECAPA/Resemblyzer.

Outputs:

- `results/residual_factor_accounting_2026-06-25.md`
- JSON metrics under `/localdisk/bowen/singing_identity/runs/residual_factor_accounting_<date>`

Core table:

| Target representation | Raw delta norm | Explained by F0/prosody | Explained by energy/spectrum | Explained by technique | Remaining residual | Heldout delta cosine |
|---|---:|---:|---:|---:|---:|---:|

Stop/go:

- If most residual is explained by F0/prosody/acoustics, the thesis should pivot toward "speech-singing domain mismatch is mostly prosodic/acoustic, not identity timbre."
- If a stable speaker-consistent residual remains, continue Track 1 with a small speaker-singer adapter.
- If technique explains a large portion, move Track 2 to the main story.

### Optional Next Experiment After Accounting

If style residual remains:

- Train a speaker-singer embedding adapter, not a Seed-VC prompt component mapper.

If technique residual is stronger:

- Narrow to breathy/vibrato with phone-level controls.

If no stable residual remains:

- Reframe the project as a negative/diagnostic result about how current frozen representations confound singing mode with prosody/acoustic factors.

## 8. Working Thesis After Reset

A cleaner thesis direction could be:

> Frozen speech/audio representations encode a strong speech-to-singing shift, but this shift is not a pure timbre vector. By factorizing the residual into prosody, acoustic envelope, singer/style, and technique components, we can identify which parts are stable across speakers and which parts are decoder- or dataset-specific.

This is more defensible than:

> We can predict singing voice from speech.

And it is closer to the latest literature, which treats singing style conversion as a dynamic, coupled, and partially unresolved problem.

## 9. Artifact Index

Important local artifacts:

- Stage 1 report: `results/stage1_repaired_200_fresh_local_report.md`
- Current research log: `results/stage1_current_research_log.md`
- Seed-VC prompt baseline: `/localdisk/bowen/singing_identity/runs/track1_seedvc_prompt_baseline_10pairs_rr`
- Seed-VC native latent audit: `/localdisk/bowen/singing_identity/runs/seedvc_native_latents_all200`
- Semantic intervention: `/localdisk/bowen/singing_identity/runs/track1_seedvc_semantic_intervention_4pairs`
- Component ablation: `/localdisk/bowen/singing_identity/runs/track1_seedvc_component_ablation_4pairs`
- Human review protocol: `design/HUMAN_REVIEW_PROTOCOL.md`
- Component ablation notebook: `notebooks/track1_seedvc_component_ablation_review.py`

## 10. Sources

- [An Extensive Analysis of the Singing Voice Conversion Challenge 2025 Evaluation Results](https://arxiv.org/html/2509.15629v2)
- [S2Voice: Style-Aware Autoregressive Modeling with Enhanced Conditioning for Singing Style Conversion](https://arxiv.org/html/2601.13629v1)
- [Bridging Speech and Singing: Multi-stage Speech-Prompted Singing Voice Conversion with Speaker Embedding Adaptation](https://www.isca-archive.org/interspeech_2025/liu25h_interspeech.html)
- [Vibrato Expression Control for Singing Voice Conversion with Improving Independent Control](https://arxiv.org/html/2606.17126v1)
- [TechSinger: Technique Controllable Multilingual Singing Voice Synthesis via Flow Matching](https://arxiv.org/abs/2502.12572)
- [InvoxSVC: Any-to-any Zero-shot Singing Voice Conversion with In-Context Learning in Latent Flow Matching](https://sap.ist.i.kyoto-u.ac.jp/EN/bib/intl/ZHO-ICME25.pdf)
- [Simple and Effective Content Encoder for Singing Voice Conversion via SSL-Embedding Dimension Reduction](https://www.isca-archive.org/interspeech_2025/zhou25e_interspeech.html)
- [Zero-shot Voice Conversion with Diffusion Transformers](https://arxiv.org/abs/2411.09943)
