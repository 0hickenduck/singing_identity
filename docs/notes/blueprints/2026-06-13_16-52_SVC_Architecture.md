# Free Recall: SVC Architecture, Conditioning, and LoRA
**Timestamp**: 2026-06-13 16:52:35 JST

## 1. Core Model Architectures
- **Autoregressive (AR) Model**: Predicts the acoustic token at time step $t$ using the sequence from $0$ to $t-1$. Execution is sequential and iterative, capturing macro-level dependencies.
- **Autoencoder**: The target is to reconstruct the input. The architecture enforces an information bottleneck (compression) to learn generative features before decoding back to the original space.
- **Acoustic Generation**: Shifted from GANs (Generative Adversarial Networks) to Diffusion and Flow Matching. Flow matching optimizes the transport from a noise distribution to the data distribution via a vector field, rendering continuous acoustic features.

## 2. Signal Processing & Representations
- **Mel-Spectrogram Generation**: Splits the 1D audio waveform into time chunks (frames) using STFT. Extracts the magnitude (intensity) of frequency bins and applies Mel-scale logarithmic compression to align with human auditory perception.
- **The Phase Problem**: The STFT yields complex numbers (magnitude + phase). The Mel-spectrogram discards the phase angle. Because exact time alignment is lost, direct inverse transformation is impossible.
- **Phase Reconstruction**: Requires a Neural Vocoder (a specialized decoder, e.g., Vocos or HiFi-GAN) to predict and reconstruct the missing phase information, yielding the final time-domain audio wave.

## 3. Implementation Ideas: Technique Conditioning (e.g., Vibrato)
Based on `GTSinger` dataset technique labels, there are two primary methods to inject control signals into the AR model:
1. **Prompting (Discrete Tokens)**: Prepending the technique label (e.g., `[VIBRATO]`) directly into the input token sequence.
2. **Conditioning / Cross-Attention**: Extracting a style embedding from the label and using continuous conditioning mechanisms (as seen in NaturalSpeech 3) or Cross-Attention to continuously modulate the AR model's hidden states across layers. 
   - *Next Step Actionable*: Implement the continuous conditioning method and compare its efficiency and leakage against discrete prompting.

## 4. Unresolved Question: LoRA Matrix Complexity
**Problem Statement**: How specifically do we design the complexity (size) of the Low-Rank Adaptation (LoRA) matrices? What defines "big" vs "small"?
