# Free Recall: Pro3 - Voice Representation Idea Review (Experiment Design)

**Timestamp:** 2026-06-15 01:04:05 JST

## 1. Core Motivation & The Big Picture
* **Observation:** When specialized Speaker ID recognizers (like ECAPA-TDNN) are used to process singing voice audio, they fail or behave strangely.
* **Core Question:** Is this because the model lacks generalization, or because a person's timbre genuinely changes when they sing?
* **Core Hypothesis:** Even though the timbre changes, there must be a **Stable Identity Core** combined with a predictable **Mode Residual** (the deviation from speech to singing).
* **Ultimate Goal:** If we can isolate and model this residual, we can accurately **estimate a person's singing voice using only their speech audio**.

## 2. The Three-Stage Experiment Design

### Stage A: Frozen Feature Audit & Cut-off
* **Step 1: Nuisance Regression (Cut-off):** We must prove that the difference between speech and singing is **not** just because singing has a higher pitch or longer duration. We use math (linear regression residualization) to forcibly "cut off" or strip away the F0 (pitch), energy, and duration information from the frozen representation.
* **Step 2: Probing & Retrieval:** If the remaining "clean" features can still be used to train a classifier to accurately distinguish speech from singing, or to retrieve the correct singer's identity across modes, it proves we have found a genuine, mode-specific timbre residual.

### Interlude: Unsupervised Clustering (Visualization)
* Alongside supervised probing, we use **K-means** or **t-SNE** to cluster the features.
* We expect to see: (1) Speech and singing vectors forming two distinct, separable blobs. (2) Within the singing blob, different singing techniques (like vibrato or breathy) naturally forming their own sub-clusters. This provides strong visual intuition.

### Stage C: Downstream Intervention
* If Stage A is successful, we introduce a lightweight **Adapter**.
* **Workflow:** We feed the speech feature into the Adapter → The Adapter predicts the timbre gap/residual ($\Delta$) → $V_{estimated\_singing} = V_{speech} + \Delta$.
* **Validation:** We pass this modified, "estimated singing" feature into a downstream zero-shot generative model (like Seed-VC). We then use various evaluation metrics to verify if the generated singing sounds more like the target singer.

---

## 💡 Mistakes & Clarifications
* **Mistake 1: Using Adversarial Learning to remove timbre?** 
  * *Correction:* Not here. Adversarial learning requires end-to-end network updates. Since our constraint is "low compute", we are doing "frozen representation analysis." To cut off F0, we simply use **Linear Residualization**.
* **Mistake 2: ECAPA is useless?** 
  * *Correction:* While ECAPA is weak at recognizing singing, this "weakness" is exactly the baseline we use to quantify *how much* cross-mode identity loss exists.

---

## 🧠 Active Retrieval Prompts
*(Try to recall these key concepts we discussed, which were skipped in your recall)*

1. **Classifier Data Split:** To convince reviewers that our residual pattern is universal to all humans (and not just memorized), what extremely strict **Data Split** method must we use when training the simple linear probe in Stage A?
2. **The Core Contribution Table:** You mentioned estimating the singing sound in Stage C. In the paper, we need to compare three types of residuals. Try to recall the English terms for them:
   * The "upper bound" residual (when we have both speech and singing data).
   * The "real-world" residual (when we only have speech and must estimate the gap).
   * The "average" residual learned from other singers.
