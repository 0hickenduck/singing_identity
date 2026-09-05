# Project Blueprint: Two-Track Research Architecture

**Timestamp:** 2026-06-17 13:26:00 JST
**Status:** Validated & Ready for Execution

This document serves as the master logical blueprint for the two parallel research tracks in the speech/singing representation project.

---

## Track 1: Timbre Shift (Speech $\to$ Singing Mode Residual)
**Goal:** Map the vocal identity shift when a person transitions from speaking to singing, and use it to improve zero-shot cross-domain Singing Voice Conversion (SVC).

1. **Data Prep:** Use a paired dataset (e.g., JVS-MuSiC) containing both speaking and singing audio from the same person.
2. **Extraction:** Use a frozen speaker encoder (specifically the one natively used by the downstream generator, e.g., Seed-VC's speaker encoder) to extract disentangled speaker embeddings: $T_{\text{speech}}$ and $T_{\text{singing}}$.
3. **Adapter Training:** Train a lightweight model (MLP/Linear) to predict the singing embedding from the speech embedding: $T_{\text{singing\_est}} = \operatorname{Adapter}(T_{\text{speech}})$.
   * *Loss function:* Cosine similarity or MSE between $T_{\text{singing\_est}}$ and the ground truth $T_{\text{singing}}$.
4. **Downstream Task / Evaluation:** 
   * Input an unseen speaker's *speech* recording.
   * Predict their *singing* embedding $T_{\text{singing\_est}}$.
   * Inject $T_{\text{singing\_est}}$ as the reference timbre prompt into the frozen Seed-VC generator (along with a source singing track for melody).
   * **Success Metric:** Does the generated audio sound more like the target speaker's true singing voice than if we had just used their raw speech embedding?

---

## Track 2: Technique Probing & Reconstruction
**Goal:** Discover linear directions for singing techniques (vibrato, falsetto, etc.) in frozen SSL models, and use them to explicitly control singing style synthesis.

### Stage A: Latent Space Probing
1. **Analogy Logic:** Similar to word vectors ($[b] - [p] = [d] - [t]$), we extract frames for the *same phoneme* sung with *different techniques* (e.g., Normal vs. Vibrato) using GTSinger's precise phone-level technique labels.
2. **Cross-Model Comparison:** Perform bootstrapping on the difference vector $\Delta Z = Z_{\text{vibrato}} - Z_{\text{normal}}$ across different foundation models (WavLM, HuBERT, ContentVec, MERT).
3. **Outcome:** Rank the models/layers based on how strongly and linearly they encode these technique directions. A music-trained model (MERT) is hypothesized to perform better than speech-trained ones.

### Stage B/C: Reconstruction & Bottleneck Injection
1. **Bottleneck Mechanism:** To reliably alter the technique, the base content representation must be stripped of technique information. This is achieved via architectural bottlenecks (like FiLM conditioning) or explicit training constraints (Gradient Reversal Layer - GRL).
2. **Architectural Intervention (Where does the extra matrix go?):**
   * **Route A (Latent Steering):** The "Matrix" is an adapter applied directly to the extracted latent features. 
     $Z_{\text{target}} = Z_{\text{original}} + \Delta Z$. We then feed $Z_{\text{target}}$ to a frozen decoder.
   * **Route B (LoRA / Internal Weight Tuning):** The extra matrices ($A$ and $B$) are inserted *inside* the layers of the base model. The forward pass becomes $h = W_{\text{frozen}}x + BAx$. This is used if the decoder needs to be fine-tuned to accept the new technique directions without catastrophic forgetting.
