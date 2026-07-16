# 实验 Debug Verdict：Track 1 Accounting v2 与 Breathy Encoding

日期：2026-06-28 JST

## 结论更新

上一版 “M5 弱于 average” 不应该被解释为科学负结果。新的低维输入 + 低秩目标实验显示，Track 1 accounting 在所有主表示上都能超过 M0 mean-delta baseline。旧结果主要是模型设定问题：高维 exact phone histogram + 直接预测 1536 维 Delta_z，在只有 20 个 speaker 的 speaker-disjoint split 下容易泛化失败。

Breathy 也没有断掉。更准确的 verdict 是：

- 单个全局 breathy direction / vector analogy：未通过 wrong-technique 与 shuffled-direction 控制。
- Breathy 是否被 frozen representations 编码：通过。speaker-disjoint breathy detection 在 WavLM L6、HuBERT L12、MERT L3 上 AUC 约 0.81-0.82，且 residualize duration/F0/energy 后几乎不掉。
- “音乐模型一定更强”：当前不支持。MERT L3 很强，但 WavLM L6 与 HuBERT L12 同样或更强。

## Track 1 v2：修正了什么

旧版问题：

```text
target: full 1536-d Delta_z
input: 400+ exact phone histogram + nuisance + metadata + technique
split: speaker-disjoint, only 20 speakers
```

新版改成：

```text
target: train speakers 上 Delta_z 的 top-K PCA residual subspace, K=8/16/32/64
input L1: F0/prosody + duration/energy
input L2: L1 + language/vocal range
input L3: L2 + low-dimensional content/phone-class summaries
input L4: L3 + technique label
input L5: L4 + acoustic proxy, only for non-acoustic target
model: multi-output ridge, alpha grid includes very large alpha so bad models can shrink back toward M0
```

## Track 1 v2 结果

| 表示 | 最佳 K | 最佳模型 | MSE 改善 vs M0 | M0 R@1 | 最佳 R@1 | L4 technique MSE 改善 |
|---|---:|---|---:|---:|---:|---:|
| WavLM Base+ L6 | 64 | L1 nuisance | 0.0725 | 0.5225 | 0.5863 | 0.0600 |
| WavLM Base+ L9 | 16/32 | L5 或 L1 | 0.0865 | 0.5443 | 0.6245 | 0.0754 |
| HuBERT Base L12 | 64 | L5 acoustic proxy | 0.0717 | 0.6302 | 0.6943 | 0.0509 |
| MERT 95M L3 | 64 | L5 acoustic proxy | 0.0621 | 0.5285 | 0.5647 | 0.0422 |
| MERT 95M L12 | 64 | L5 acoustic proxy | 0.0574 | 0.4258 | 0.5118 | 0.0545 |

Interpretation:

1. 低维/低秩版本稳定超过 M0，所以旧版 “M5 < average” 是 overfitting / misspecification signal，不是研究方向失败。
2. L1 nuisance 已经很强，说明 speech-to-singing prompt mismatch 的主要可预测部分仍然是 F0、duration、energy、voicing 相关。
3. Technique label 在 L4 有正贡献，但不是主导解释项。它有用，但还不能作为核心 claim。
4. Acoustic proxy 在 HuBERT/MERT 上有额外增益，说明 phonation/spectrum 类因素可能比离散 technique label 更接近真实机制。

## Breathy：从 Direction 改成 Encoding Verdict

### Direction verdict

Speaker-balanced vector analogy 的结果：

| 表示 | analogy top1 | wrong-technique top1 | shuffled-direction top1 |
|---|---:|---:|---:|
| MERT L3 | 0.225 | 0.223 | 0.226 |
| WavLM L6 | 0.222 | 0.222 | 0.223 |
| WavLM L9 | 0.243 | 0.245 | 0.243 |

Verdict: 单个 global breathy vector 不成立，至少当前 GTSinger + exact phone + mean/std pooling 下不成立。

### Detection verdict

新跑的 speaker-disjoint breathy detection：

| 表示 | raw AUC | residualized AUC | shuffled AUC | residualized accuracy |
|---|---:|---:|---:|---:|
| WavLM Base+ L6 | 0.8237 | 0.8238 | 0.5173 | 0.7377 |
| HuBERT Base L12 | 0.8206 | 0.8198 | 0.5115 | 0.7307 |
| MERT 95M L3 | 0.8100 | 0.8093 | 0.5167 | 0.7370 |
| WavLM Base+ L9 | 0.7559 | 0.7579 | 0.5077 | 0.6837 |
| MERT 95M L12 | 0.7308 | 0.7300 | 0.5168 | 0.6587 |
| Acoustic baseline | 0.6871 | 0.6916 | 0.5092 | 0.6354 |

Verdict:

1. Breathy 是被编码的，不是 dead end。
2. Residualize interval duration/F0/energy 后 AUC 几乎不掉，说明不是简单 pitch/energy/length shortcut。
3. Acoustic baseline 也能检测，但弱于 WavLM L6/HuBERT/MERT L3，所以 SSL representation 里有超过当前低层 acoustic baseline 的信息。
4. MERT 没有明显统治 WavLM/HuBERT。当前更准确的说法是：music-oriented MERT L3 is competitive, but not uniquely superior.

## 为什么之前是正向、这次一开始像负向

旧 breathy 正向证据主要说明“breathy 是最有希望的候选”。但当时有几个问题：

1. 数据被 breathy 主导，最容易先看到结构。
2. 很多强 group 集中在单语言、少数 phone、少数 speaker。
3. `<AP>`、sparse phones、query leakage 或 insufficient controls 会让 vector analogy 显得过强。
4. 之前 latent steering run 用的是 `<AP>`，而且没有 decoder/audio success，不应当作为最终正证据。

现在的新结论不是反转，而是细化：

```text
breathy is encoded
but not as one clean global linear direction
```

## 当前可写的研究描述

Track 1:

> Speech and singing prompts differ by a strong shared mode shift in frozen representations. Low-dimensional acoustic/prosodic covariates explain a stable component of this mismatch on held-out singers. Full high-dimensional utterance-level accounting overfits unless the target Delta is restricted to a low-rank subspace.

Track 2:

> Breathy phonation is linearly detectable in frozen SSL representations under speaker-disjoint evaluation and remains detectable after residualizing simple duration/F0/energy covariates. However, this information does not appear as a single globally transferable vector direction under strict wrong-technique and shuffled-direction controls.

## 下一步实验

1. Track 1 v2 加 confidence intervals / permutation tests，确认 L1/L4/L5 的增益稳定。
2. Track 1 下游 Seed-VC link 扩到 30-50 pairs。这属于 Track 1，因为它验证 representation mismatch 是否预测 speech-prompt vs singing-prompt degradation。
3. Track 2 从 single vector 转向 subspace：PCA/PLS on breathy-control deltas，测试 top-K subspace 是否比 single direction 过控制。
4. Track 2 加 phone-family level，而不是 exact phone level，降低 language/phone sparsity。
5. 如果要比较 music-oriented model，指标应从 steering direction 改成 detection/residualized detection/subspace stability。

## 文件索引

- Track 1 v2 outputs: `/localdisk/bowen/singing_identity/runs/track1_prompt_mismatch_accounting_v2_2026-06-28`
- Track 2 breathy detection outputs: `/localdisk/bowen/singing_identity/runs/track2_breathy_detection_probe_2026-06-28`
- Track 1 v2 script: `/home/bowen/bowen_lab/projects/singing_identity/scripts/probing/run_prompt_mismatch_accounting_v2.py`
- Breathy detection script: `/home/bowen/bowen_lab/projects/singing_identity/scripts/probing/run_breathy_detection_probe.py`
