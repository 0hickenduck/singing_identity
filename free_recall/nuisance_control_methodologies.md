# Research Note: Controlling Nuisance Variables in Speech-to-Singing Representation Analysis

**Timestamp:** 2026-06-17 09:05:00 JST

This note documents the design choices, mathematical details, and alternative methodologies for isolating true speaker identity residuals from basic acoustic shortcuts (pitch/F0, energy, and duration) during cross-mode (speech vs. singing) representation analysis.

---

## 1. GTSinger Phoneme-Level Technique Annotation Validation
* **Verification:** Yes, **GTSinger annotations are explicitly phoneme-level**. 
* **Implication for Analogy Tests:** Because the labels (e.g., *Vibrato, Falsetto, Breathy, Pharyngeal, Glissando*) are bound to specific phoneme boundaries, we can precisely slice the overlapping frames of a frozen SSL representation (e.g., WavLM, HuBERT) and average-pool them to get a clean, isolated token representation of a specific phone under a specific technique. This makes token-level analogy testing ($cos(r[b], r[p] + r[d] - r[t])$) highly feasible and structured.

---

## 2. Nuisance Regression: Mathematical Details & Strength Control

"Nuisance Regression" (specifically **Linear Regression Residualization**) is a mathematical projection method to strip out linear correlations.

### The Mathematics
Let $h \in \mathbb{R}^D$ be a frozen representation vector (e.g., a pooled segment or frame representation from WavLM).
Let $c \in \mathbb{R}^k$ be a vector of $k$ acoustic nuisance variables (e.g., $c = [\log(F0_{\text{mean}}), F0_{\text{std}}, \text{Energy}, \text{Duration}]$).

1. We fit a linear regression model across our dataset to predict each dimension $d$ of $h$ using the covariates $c$:
   $$\hat{h}_d = w_d^T c + b_d$$
   In matrix form for all dimensions:
   $$\hat{h} = W c + b \quad \text{where } W \in \mathbb{R}^{D \times k}, b \in \mathbb{R}^D$$
2. The **residualized representation** $h_{\text{residual}}$ is:
   $$h_{\text{residual}} = h - (W c + b)$$

By construction, $h_{\text{residual}}$ is mathematically orthogonal to the nuisance vector $c$ on the training set.

### How Strong is it?
* **Strength:** It is **extremely strong** in removing *linear* dependencies. Any downstream linear probe (classifier) will find exactly $0\%$ correlation between the residual representation and the control variables on the training domain.
* **Limitations:**
  1. **Non-linearity:** It only removes linear relationships. If $F0$ has a non-linear effect on the hidden states, some $F0$ information will still leak through.
  2. **Over-cleansing (Aggressiveness):** If speaker identity and pitch are naturally correlated (e.g., a speaker with a higher default pitch), regressing out $F0$ will partially erase the speaker's true timbre.

### Controlling the Strength
To prevent "over-cleansing" or adjust how aggressively we strip these features, we can implement:

1. **Partial Projection (Scaling Parameter $\lambda$):**
   Inject a scalar controller $\lambda \in [0, 1]$:
   $$h_{\text{adapted}} = h - \lambda (W c + b)$$
   * $\lambda = 0$: No modification (raw features).
   * $\lambda = 1$: Complete mathematical stripping.
   * $\lambda \in (0, 1)$: Soft control, allowing us to find a sweet spot where we reduce the shortcut without destroying the timbre.
2. **Selective Covariates:**
   Only include specific parameters in $c$ (e.g., only include $F0$ mean, but keep $F0$ standard deviation / dynamics).
3. **Kernel/MLP Control:**
   Use non-linear predictors (Kernel Ridge Regression or a small MLP) to predict $\hat{h}$ and compute the residual, but this increases the risk of overfitting and over-cleansing.

---

## 3. Alternative Methodologies for Nuisance Control

Instead of latent-space mathematical residualization, several alternative approaches exist in literature or signal processing:

### Method A: DSP-level Signal Normalization (Source-level)
* **How it works:** Instead of correcting vectors, we normalize the raw audio waveforms *before* extracting representations.
  * Use a vocoder (e.g., WORLD, Praat) or pitch-shifting tool (e.g., Rubberband) to shift the singing voice's $F0$ contour down to the target singer's average speaking pitch range.
  * Use time-stretching to make the singing phoneme durations match the average duration of the speaking phonemes.
  * Extract representations (WavLM/HuBERT) from this "F0-and-duration-normalized" audio.
* **Pros:** Physically eliminates the shortcut at the source; no risk of mathematical over-cleansing in the embedding space.
* **Cons:** DSP artifacts (pitch shifting can distort formants and introduce phase issues, which in turn alters the representation's timbre).

### Method B: Stratified Matching (Matching Cohorts)
* **How it works:** A purely statistical data-selection control (common in epidemiology).
  * We do not modify the audio or the vectors.
  * Instead, we construct pairs of speech and singing frames that *already* share the same phone, and whose F0 and energy values naturally fall into the exact same overlapping bins.
  * We evaluate the probes or retrieval models *only* on these matched test sets.
* **Pros:** Zero distortion to the audio or representations; highly rigorous.
* **Cons:** Drastically reduces the amount of usable data, as most singing frames have much higher pitch than speech frames.

### Method C: Adversarial Disentanglement (GRL / Adversarial Probing)
* **How it works:** 
  * Train a small projection network $P(h)$ using a Gradient Reversal Layer (GRL).
  * The network is trained with two objectives: (1) Predict speaker identity. (2) Fail to predict F0, energy, and duration.
* **Pros:** Learns non-linear disentanglement.
* **Cons:** Requires active training (non-frozen); requires larger compute; susceptible to training instability.

### Method D: Quantization / Information Bottleneck
* **How it works:** Squeeze representation frames through a Vector Quantization (VQ) codebook or an information bottleneck (like the content stream of FACodec).
* **Pros:** Highly robust; standard in modern audio language modeling.
* **Cons:** Quantization is lossy and might throw away subtle speaker traits needed for zero-shot speaker estimation.

---

## 4. Geometric Interpretation of Nuisance Regression
When we regress out covariates $c \in \mathbb{R}^k$ from a representation $h \in \mathbb{R}^D$, we are performing an orthogonal projection in vector space:
* **Dimensionality Contrast:** The covariates vector $c$ has a very small dimension ($k \ll D$, e.g., $k=4$ parameters like mean F0, std F0, energy, duration, compared to $D=768$ hidden dimensions for WavLM).
* **Subspace Rank:** The regression matrix $W \in \mathbb{R}^{D \times k}$ has a maximum rank of $k$. Therefore, the space spanned by the predicted features $\hat{h} = Wc + b$ is at most a $k$-dimensional subspace.
* **Orthogonal Complement:** By subtracting $Wc$, we project $h$ onto the orthogonal complement of this $k$-dimensional subspace. The remaining $D - k$ dimensions (e.g., $768 - 4 = 764$ dimensions) remain completely intact. 
* **Conclusion:** It is **not** a full-rank removal that collapses the entire representation space. It is a surgical projection that removes only the $k$ directions linearly aligned with the nuisance variables, preserving the vast majority of the latent space.

---

## 5. Fine-Grained Strength Control
Using a single scalar $\lambda$ applies an isotropic scaling factor to all dimensions. We can design finer-grained control mechanisms:

1. **Dimension-Wise Diagonal Vector ($\Lambda \in \mathbb{R}^D$):**
   Instead of a scalar, apply a Hadamard product with a vector $\Lambda$:
   $$h_{\text{adapted}} = h - \Lambda \odot (W c + b)$$
   This allows different hidden dimensions to have different levels of residualization, since some dimensions may encode F0 much more strongly than others. We can optimize $\Lambda$ to balance speaker classification and F0 suppression.
2. **Layer-Wise Calibration:**
   Apply a separate scalar $\lambda_l$ for each layer $l$. Lower layers (which retain more raw acoustics) may require a larger $\lambda$, whereas higher semantic layers may require a much smaller $\lambda$ to prevent identity degradation.
3. **Gated Adaptive Residualization:**
   Define $\lambda(h)$ dynamically as a function of the embedding:
   $$\lambda(h) = \sigma(V h + d)$$
   where $V, d$ are tiny trainable parameters, allowing the model to adaptively decide how much to regress out based on the local phonetic context.

---

## 6. Parameter-Efficient Adversarial Disentanglement (LoRA + GRL)
Can we use Gradient Reversal Layers (GRL) on adapters (LoRA) while keeping the base foundation model frozen?

* **Yes, absolutely.** This is a highly elegant and standard approach in Parameter-Efficient Fine-Tuning (PEFT).
* **How it works:**
  1. The base model (e.g., WavLM content encoder) is **frozen**.
  2. We insert trainable LoRA matrices $A$ and $B$ into selected attention or MLP layers.
  3. The final output representation $h_{\text{out}}$ (which includes the LoRA contribution) is routed to:
     * **Target Task Head:** Computes standard target loss (e.g., speaker similarity).
     * **Nuisance Predictor Head:** Predicts F0/duration/energy. Crucially, a **Gradient Reversal Layer (GRL)** is placed *before* this predictor.
  4. **Backward Pass:** The GRL multiplies the gradient from the Nuisance Predictor by a negative scaling factor $-\alpha$. 
  5. **Parameter Updates:** Because the base model parameters are frozen, these reversed gradients *cannot* modify the foundation model. Instead, they flow directly into the trainable LoRA parameters $A$ and $B$.
* **Outcome:** The LoRA matrices are forced to learn a low-rank weight adjustment that actively suppresses or cancels out the F0/nuisance information in the network's forward activations, without needing to touch or retrain the massive base model. This makes adversarial disentanglement highly feasible on limited GPU compute.

---

## 7. Comparative Analysis: FiLM vs. Cross-Attention for 1D/Scalar Conditioning
If the conditioning label $c$ is only 1-dimensional (e.g., a binary technique tag $\{0, 1\}$, a single scalar F0 value, or a scalar technique intensity), the mathematical behaviors of **FiLM** and **Cross-Attention** diverge fundamentally, especially when the condition sequence length is 1 (a single global value per sequence).

### 1. Mathematical Collapse of Cross-Attention ($N_c = 1$)
Let the hidden states be $H \in \mathbb{R}^{T \times D}$ (where $T$ is sequence length).
Let the condition be a single scalar $c \in \mathbb{R}$ ($N_c = 1$). Even if we project it to a $D_c$-dimensional vector $C \in \mathbb{R}^{1 \times D_c}$, the sequence length remains $1$.

In Cross-Attention:
* **Query:** $Q = H W_Q \in \mathbb{R}^{T \times D_{\text{attn}}}$
* **Key:** $K = C W_K \in \mathbb{R}^{1 \times D_{\text{attn}}}$
* **Value:** $V = C W_V \in \mathbb{R}^{1 \times D_{\text{attn}}}$

The attention weight matrix is computed using softmax over the Key sequence dimension ($N_c = 1$):
$$\text{AttentionWeights}_t = \operatorname{softmax}\left( \frac{Q_t K^T}{\sqrt{D_{\text{attn}}}} \right)$$

Because there is **only one Key** in the key sequence ($N_c = 1$), the softmax denominator is a sum of a single exponent:
$$\operatorname{softmax}(x)_1 = \frac{e^x}{\sum_{j=1}^1 e^x} = \frac{e^x}{e^x} = 1$$

Therefore:
* The attention weight is **exactly $1$ for all time steps $t$**, regardless of the Query $Q_t$.
* The query-dependent filtering mechanism of attention **completely collapses**.
* The output of the attention block is simply the Value vector projected back to the hidden space:
  $$\text{Output}_t = 1 \cdot V = C W_V W_O \in \mathbb{R}^D$$
* After the residual connection:
  $$h_{\text{new}, t} = h_t + \beta(c) \quad \text{where } \beta(c) = C W_V W_O + b_{\text{attn}}$$

**Conclusion:** For a 1D global label, Cross-Attention mathematically collapses into a **simple dynamic bias addition**. It is equivalent to adding a constant vector (which is a linear projection of the label) to all frames. It cannot perform multiplicative scaling.

### 2. FiLM (Feature-wise Linear Modulation)
FiLM modulates the hidden states $h_t$ using both multiplicative scaling $\gamma(c)$ and additive bias $\beta(c)$:
$$\operatorname{FiLM}(h_t, c) = \gamma(c) \odot h_t + \beta(c)$$
where $\gamma(c) = w_\gamma c + b_\gamma \in \mathbb{R}^D$ and $\beta(c) = w_\beta c + b_\beta \in \mathbb{R}^D$.

* **Feature Scaling:** FiLM can actively **suppress** (scale near 0) or **amplify** (scale > 1) specific features in the hidden state $h_t$ based on the scalar $c$.
* **Expression:** It is significantly more expressive than Cross-Attention because it allows interactive modulation (multiplication) rather than just static shifting (addition).

### 3. Comparison Summary

| Axis | FiLM (1D Condition) | Cross-Attention (1D Global Condition) |
| :--- | :--- | :--- |
| **Mathematical Operation** | Multiplicative Scaling + Additive Bias ($\gamma(c) \odot h + \beta(c)$) | Additive Bias only ($h + \beta(c)$) due to Softmax collapse |
| **Parameter Complexity** | $O(D)$ ($2D$ parameters per layer) | $O(D^2)$ ($4D^2$ parameters for $W_Q, W_K, W_V, W_O$) |
| **Query-Dependency** | Highly interactive (interacts directly with the hidden features $h$) | Collapses; output is independent of the Query vector |
| **Efficiency** | Extremely lightweight and fast | Heavily over-parameterized and redundant |

For 1D global labels, **FiLM is mathematically superior, significantly more parameter-efficient, and does not suffer from softmax collapse.**

---

## 8. Aligned vs. Unaligned Conditioning: The Decision Boundary
When designing style/technique conditioning for frame-level audio generation (such as GTSinger), the choice between **FiLM** and **Cross-Attention** depends on whether the conditioning sequence is **temporally aligned** with the hidden states.

### 1. Frame-Wise Aligned Conditioning (GTSinger Case)
In GTSinger, we have frame-wise aligned labels (e.g., a sequence of technique indicators $c_t$ matched 1-to-1 with each audio frame $h_t$).

* **Should we use Cross-Attention here?**
  * **No.** Cross-Attention calculates long-range relationships ($Q_t$ attending to $K_{t'}$). 
  * Because the technique labels are already aligned in time with the phonemes/frames, a frame at time $t$ only needs to know the technique label *at that exact time $t$* (local conditioning).
  * Using Cross-Attention introduces:
    1. **$O(T^2)$ computational overhead** (which scales quadratically with song length).
    2. **Temporal Blurring:** Attention weights can diffuse style information across time boundaries. For example, a vibrato style label at time $t$ might leak into a normal singing segment at time $t'$ because of soft attention weights, degrading the precision of style control.
* **FiLM / AdaLN is the correct choice:** It applies strictly local, frame-by-frame scaling and bias shifting ($O(T)$ complexity) without any risk of temporal leaking or blurring.

### 2. NaturalSpeech 3 and FiLM / AdaLN
In **NaturalSpeech 3 / FACodec**, they use FiLM (specifically **AdaLN - Adaptive Layer Normalization**, which is mathematically identical to FiLM but applied inside the normalization layers) to inject style/timbre vectors.
* **Why did they use FiLM?**
  1. **Global Timbre has no temporal dimension:** The speaker/timbre embedding is a single, global sequence-length 1 vector. As shown in Section 7, Cross-Attention collapses on $N_c=1$. AdaLN/FiLM allows this global vector to scale and shift the frame activations dynamically.
  2. **Preserving Strict Alignment:** For aligned prosody/pitch contours, they inject them locally. Using attention would risk blurring the alignment between phonemes, pitch, and the output waveform.
  3. **Diffusion/Flow Matching standard:** NaturalSpeech 3 uses a diffusion-based generator (specifically DiT-style). DiT standardizes on AdaLN/FiLM because it is extremely stable, lightweight, and allows continuous scale-shifting of denoising states.

### 3. The Golden Rule of Conditioning

* **Use Cross-Attention when the condition is UNALIGNED in time:**
  * *Example:* You are feeding a reference audio clip of a different length to guide the style of a target sentence. The model must learn *where* to look in the reference audio to style the current frame.
* **Use FiLM / AdaLN when the condition is ALIGNED in time or GLOBAL:**
  * *Example:* Frame-wise aligned pitch curves, frame-wise technique labels, global speaker ID vectors, or global style category tags. This enforces local constraints, prevents temporal blurring, and runs in $O(T)$ time.



