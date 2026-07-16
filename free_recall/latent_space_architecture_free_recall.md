# Free Recall: Latent Space, Tags, and Modern Architecture

**Timestamp:** 2026-06-15 17:22:50 JST

## 1. Finding Directions in Latent Space (Insights from the Phonological Analogies Paper)
There is a crucial distinction between testing if a direction *exists* (Analogy) and actually *using* a direction (Synthesis), as demonstrated in the paper: **[[b]=[d]-[t]+[p]: Self-supervised Speech Models Discover Phonological Vector Arithmetic](https://arxiv.org/abs/2602.18899)**.

* **Experiment 1: Analogy Testing (Token-Level)**
  * To test phonological relations like `[b]:[p] = [d]:[t]` (voicing), we **do not** use global means. 
  * Instead, we get a single representation for *one specific occurrence* of a phone by average-pooling the frames inside that exact segment. 
  * We test the vector arithmetic using cosine similarity: $cos(r[b], r[p]+r[d]-r[t])$.
  * **Statistical Bootstrapping:** Because we only have a finite pool of these tokens, we repeatedly and randomly sample different combinations of `[b], [p], [d], [t]` from our dataset to calculate an average cosine score. This proves the analogy holds generally, not just for one "lucky" sample.

* **Experiment 2: Controllable Synthesis (Global Mean)**
  * To actually *apply* a style (like adding voice/vibrato to a frame), we calculate the global direction: $Mean(Voiced) - Mean(Unvoiced)$.
  * We then add a scaled version of this global direction vector to our target representation to control the synthesis.

## 2. Terminology: "Bootstrapping"
* **Linguistic Origin:** Derives from the physically impossible phrase "pull yourself up by your own bootstraps." Metaphorically, it means to accomplish or start something using *only the resources you already have* (e.g., "bootstrapping a startup" means funding it yourself without outside investors).
* **Statistical Meaning:** Simulating "new" datasets by repeatedly resampling (often with replacement) from your *existing* observed dataset. It is used to estimate how stable or uncertain a measurement (like a mean or cosine similarity) is, without needing to go out and collect entirely new real-world data.

## 3. GTSinger Tags & FiLM Application
* **Phoneme-Level Detail:** GTSinger provides tags (Vibrato, Breathy, Falsetto, etc.) at the phoneme level, not the sentence level.
* **Why detail matters:** Because the tags are highly granular, it dictates how we apply styling. If applying FiLM, we can calculate modifiers continuously based on these detailed tags, allowing highly dynamic style changes within a single song.
* **Conditioning Choice (FiLM vs. Cross-Attention):** 
  * Because GTSinger tags are **temporally aligned** with the frames, we should use **FiLM** rather than Cross-Attention. 
  * Cross-Attention would introduce $O(T^2)$ computational overhead and cause **temporal blurring/leakage** of styles across frame boundaries.
  * *Golden Rule:* Use Cross-Attention for *unaligned* conditioning (e.g., arbitrary-length style reference audio). Use FiLM/AdaLN for *aligned* (e.g., frame-wise tags, pitch curves) or *global* (e.g., speaker ID) conditioning.
  * For full details, see the dedicated research note: [nuisance_control_methodologies.md](file:///Users/bowen/research/project/free_recall/nuisance_control_methodologies.md).


## 4. Frame vs. Phone & Information Bottlenecks
* **1 Token = 1 Frame.** 
* **The Trade-off:** Modern models process audio per-frame rather than per-phone because forcing phone boundaries in singing is highly inaccurate and destroys intra-phone transitions (like glissando or vibrato).
* **The Bottleneck Necessity:** While frames preserve these beautiful dynamic details, they also leak a lot of unwanted information (like the source speaker's identity/timbre). Therefore, staying at the frame-level *requires* adding a harsh **Information Bottleneck** (like Vector Quantization or adversarial learning) to squeeze out the identity while keeping the dynamics.

## 5. Machine Learning Terminology Clarification
* **KNN (K-Nearest Neighbors):** A *supervised* learning algorithm used for classification. You have a dataset with known labels, and you want to predict the "label" for a new, unlabelled point by checking the labels of its $K$ closest neighbors (majority vote).
* **K-Means:** An *unsupervised* learning algorithm used for clustering. You have data with *no labels*, and you want to find hidden patterns by grouping them into $K$ clusters. It does this by finding $K$ central points (centroids).
