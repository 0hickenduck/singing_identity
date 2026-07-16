# 需要人工检查的实验说明：Track 1 / SeedVC / Breathy

日期：2026-06-28 JST

## 先给结论

现在已经到了真正需要人介入的位置。不是因为实验卡住，而是因为自动数据已经把边界画出来了：

1. Track 1 的表征分解不是负结果。旧版 “M5 比 average 更差” 主要是高维小样本下的模型设定问题；低维输入 + 低秩目标以后，所有主表示都稳定超过 M0 mean-delta baseline。
2. Breathy 没有断。它不是一个干净的全局单方向，但它在 WavLM / HuBERT / MERT 里是可检测的，而且 residualize F0/duration/energy 后仍然可检测。
3. SeedVC 下游 prompt-mode 自动 proxy 有稳定差异，但它不是纯身份真值。Singing prompt 会让输出更接近 target singing reference，同时更远离 target speech reference；这很可能混入了 singing/speech 域、breathy/phonation、音质或响度相关因素。
4. SeedVC component ablation 指向 `mel + style`，不是 `prompt_seq`。这说明下游差异更像 prompt acoustic context / speaker-style embedding 在动，而不是 semantic/F0 prompt sequence 单独在动。

因此现在需要你听的是：自动 proxy 看到的 `singing prompt` / `mel+style` 优势，在人耳上到底是身份更像 target singer，还是只是更像 singing/breathy/同域/更好音质。

## 人工检查页面

### 1. Prompt mode baseline，30 pairs

HTML:

`/localdisk/bowen/singing_identity/runs/track1_seedvc_prompt_baseline_30pairs/listening_review.html`

每个 pair 里有：

- source singing
- target speech reference
- target singing reference
- generated with target speech prompt
- generated with target singing prompt
- Resemblyzer delta

请判断：

1. `target_singing_prompt` 生成音频是否真的更像 target singer？
2. 如果更像，它更像的是身份/timbre，还是 singing domain / breathy / phonation / loudness？
3. 哪些 pair 是 speech prompt 更好？
4. 哪些 pair 有 artifact、音量、断裂、内容保持问题？

### 2. SeedVC component ablation，12 pairs

HTML:

`/localdisk/bowen/singing_identity/runs/track1_seedvc_component_ablation_12pairs/component_ablation_review.html`

每个 pair 里有：

- `baseline_all_speech`
- `oracle_all_singing`
- `singing_prompt_seq_only`
- `singing_mel_only`
- `singing_style_only`
- `singing_prompt_seq_mel`
- `singing_prompt_seq_style`
- `singing_mel_style`

请重点判断：

1. `singing_mel_style` 是否听起来接近 `oracle_all_singing`？
2. 单个 component 里，`mel` 和 `style` 谁更像真正改变身份？还是都只是改变唱法/域/音质？
3. `prompt_seq_only` 是否几乎不可听或方向错误？自动 proxy 认为它基本没用。
4. 如果 `style_only` 有效果，它更像 speaker identity、vocal tract/timbre，还是 technique/phonation？

## 为什么现在必须人工检查

SeedVC 30-pair 自动结果：

| 指标 | mean | 95% bootstrap CI | sign |
|---|---:|---:|---:|
| singing prompt 对 target singing 的增益 | 0.0870 | [0.0671, 0.1064] | 28/30 positive |
| singing prompt 对 target speech 的增益 | -0.0694 | [-0.0825, -0.0566] | 0/30 positive |
| singing prompt 对 source singing 的增益 | 0.0130 | [-0.0048, 0.0299] | 19/30 positive |
| target-singing gain - target-speech gain | 0.1564 | [0.1303, 0.1813] | 30/30 positive |

这个结果说明 prompt mode effect 很稳定，但解释很危险：

- 如果它是真身份增益，那么 singing prompt 确实改善 target-singer identity。
- 如果它主要是 singing-domain 匹配，那么 Resemblyzer 只是认为 singing output 更像 singing reference，而不是更像这个人。
- 如果它主要是 breathy/phonation/音质，那么它支持 Track 2/phonation 相关机制，但不能当 Track 1 身份转换成功。

自动 QA 也提示混杂：

- raw target-singing gain 与 RMS delta 的相关较强：Spearman = -0.645。
- raw target-singing gain 与 MERT predicted delta norm 看似强相关：Spearman 约 -0.70。
- 但去掉 RMS delta 和 language 后，Track 1 predictors 与 target-singing gain residual 的最高相关只剩约 0.23，而且 control model 也能排到前面。

所以目前不能写成 “Track 1 accounting predicts SeedVC identity improvement”。更准确是：

> SeedVC prompt mode produces a robust automatic target-singing similarity shift, but this shift is not yet attributable to human-perceived identity rather than singing-domain, phonation, loudness, or prompt-acoustic-context effects.

## SeedVC component ablation 自动结果

12-pair component ablation，相对 `baseline_all_speech`：

| condition | d target singing | d target speech | d source | d RMS dB | positive target singing |
|---|---:|---:|---:|---:|---:|
| oracle_all_singing | 0.1030 | -0.0786 | 0.0214 | -0.574 | 12/12 |
| singing_mel_only | 0.0392 | -0.0324 | 0.0061 | 1.916 | 10/12 |
| singing_style_only | 0.0533 | -0.0321 | 0.0395 | -2.403 | 11/12 |
| singing_prompt_seq_only | -0.0197 | -0.0150 | -0.0148 | 2.019 | 1/12 |
| singing_prompt_seq_mel | 0.0597 | -0.0349 | 0.0092 | 1.096 | 9/12 |
| singing_prompt_seq_style | 0.0472 | -0.0415 | 0.0315 | -0.764 | 11/12 |
| singing_mel_style | 0.1001 | -0.0757 | 0.0204 | -0.491 | 12/12 |

自动 verdict：

- `prompt_seq_only` 基本不是驱动项，而且方向还偏负。
- `mel_only` 和 `style_only` 都有贡献。
- `mel + style` 几乎复现 `oracle_all_singing`。
- 因此如果人耳也听到真实身份改善，下一步应优先看 SeedVC 的 mel context / CAMPPlus style embedding，而不是只映射 semantic/F0 prompt sequence。

## Track 1：M0 / average / 负结果方向怎么解释

这里的 M0 不是 “什么都不做”。M0 是在训练 speakers 上学到一个平均 speech-to-singing Delta，然后把这个平均 Delta 加到 held-out speech prompt representation 上。

也就是说：

```text
do nothing: speech representation
M0: speech representation + train-speaker mean Delta
M1-M5: speech representation + model-predicted Delta
```

旧版 M5 比 M0 差，表示：

> 这个高维模型预测出来的 per-pair Delta 比直接用训练集平均 Delta 更差。

这不是 “我们的现象不存在”，而是典型的小样本高维泛化失败。旧版问题是：

- 目标是完整 768/1024/1536 维 Delta。
- 输入里有 exact phone histogram 和很多 sparse content features。
- speaker-disjoint split 只有约 20 个 speaker。
- 模型自由度相对样本太高。

v2 修正：

- target 只预测 train speakers 上 Delta 的 top-K PCA subspace，K=8/16/32/64。
- 输入改为低维 nuisance/content/metadata/technique/acoustic proxy。
- ridge alpha grid 包含非常大的 alpha，让模型可以 shrink 回接近 M0。
- controls 包括 metadata-only、content-only、shuffled nuisance、shuffled technique、acoustic-only。

v2 结果：

| 表示 | 最佳 K | 最佳模型 | MSE 改善 vs M0 | M0 R@1 | 最佳 R@1 |
|---|---:|---|---:|---:|---:|
| WavLM L6 | 64 | L1 nuisance | 0.0725 | 0.5225 | 0.5863 |
| WavLM L9 | 16/32 | L5 或 L1 | 0.0865 | 0.5443 | 0.6245 |
| HuBERT L12 | 64 | L5 acoustic proxy | 0.0717 | 0.6302 | 0.6943 |
| MERT L3 | 64 | L5 acoustic proxy | 0.0621 | 0.5285 | 0.5647 |
| MERT L12 | 64 | L5 acoustic proxy | 0.0574 | 0.4258 | 0.5118 |

所以现在的 Track 1 verdict 是：

> Frozen representations contain a predictable speech-to-singing prompt mismatch. The predictable part is mostly low-dimensional acoustic/prosodic/phonation-related structure. Discrete technique labels add weaker evidence. High-dimensional exact accounting overfits.

## Breathy：不是 global vector，但不是 dead end

之前 breathy 看起来“最有结构”，这件事没有被推翻。被推翻的是更强的说法：

> breathy is one clean global transferable direction.

新实验支持的是：

> breathy is encoded, but not as one clean global direction.

Detection probe：

| 表示 | residualized AUC | shuffled AUC | residualized acc |
|---|---:|---:|---:|
| WavLM L6 | 0.8238 | 0.5173 | 0.7377 |
| HuBERT L12 | 0.8198 | 0.5115 | 0.7307 |
| MERT L3 | 0.8093 | 0.5167 | 0.7370 |
| WavLM L9 | 0.7579 | 0.5077 | 0.6837 |
| MERT L12 | 0.7300 | 0.5168 | 0.6587 |
| acoustic baseline | 0.6916 | 0.5092 | 0.6354 |

Subspace probe：

| 表示 | K | target capture | wrong-technique capture | random capture |
|---|---:|---:|---:|---:|
| WavLM L6 | 64 | 0.6116 | 0.5644 | 0.0420 |
| HuBERT L12 | 64 | 0.7017 | 0.6547 | 0.0418 |
| MERT L3 | 64 | 0.6489 | 0.5877 | 0.0415 |

解释：

- Breathy 的信息确实在 representation 里。
- 它不是简单 duration/F0/energy shortcut。
- 但 breathy-specific subspace 只比 wrong-technique subspace 略强，说明这里有一个更一般的 technique/phonation/phone subspace。
- 因此不能直接做 “breathy linear steering direction” 强 claim。

## Music-oriented model 的说法

当前数据不支持 “MERT/music-oriented model 一定更完整地 encode technical latent space”。

更准确的说法是：

- MERT L3 很有竞争力。
- WavLM L6 和 HuBERT L12 也很强，很多指标不弱于 MERT。
- MERT L12 往往不如 MERT L3。

因此比较 music-oriented model 的实验还可以做，但 claim 应该谨慎写成：

> Music-oriented pretraining does not uniformly dominate general SSL representations in the current probes; lower MERT layers may be competitive for technique/phonation encoding.

## 现在需要你的 verdict

请先听两个 HTML 页面，不需要全部写长评。最有用的反馈格式是：

```text
Prompt mode 30-pair:
- singing prompt 是否经常更像 target identity: yes / no / mixed
- 如果 mixed，哪些 pair speech prompt wins:
- 主要变化更像 identity / breathy-phonation / singing-domain / loudness-quality:
- artifact-heavy pair numbers:

Component ablation 12-pair:
- mel_style 是否接近 oracle_all_singing: yes / no / mixed
- 单分量最明显: mel / style / prompt_seq / none
- style 的效果更像 identity / timbre / technique-phonation / loudness-quality:
- prompt_seq_only 是否基本无效: yes / no / mixed
```

## 人工 verdict 之后的分支

如果人耳确认 singing prompt / mel_style 主要改善 target identity：

- Track 1 下游 link 成立，可以继续做 SeedVC 内部 intervention。
- 下一步优先映射或操控 CAMPPlus style embedding 和 mel context，而不是 prompt_seq。
- 可以设计 `speech style -> singing style` mapper，或做 mel/style interpolation。

如果人耳认为主要是 singing-domain / breathy / phonation：

- 不能把 SeedVC proxy 当 identity success。
- Track 1 仍然成立为 representation mismatch accounting，但 SeedVC link 需要改成 phonation/domain transfer 解释。
- Track 2 可以接上，因为 breathy/phonation encoding 是正结果。

如果人耳认为 artifact/loudness 主导：

- 当前 SeedVC prompt-mode run 只能作为工程 debug。
- 需要 rerun 高 diffusion steps、控制 loudness、扩大 target/source speaker balance，再做听评。

## 文件索引

- Track 1 v2 main run: `/localdisk/bowen/singing_identity/runs/track1_prompt_mismatch_accounting_v2_2026-06-28`
- Track 1 v2 controls: `/localdisk/bowen/singing_identity/runs/track1_prompt_mismatch_accounting_v2_controls_2026-06-28`
- SeedVC prompt baseline 30 pairs: `/localdisk/bowen/singing_identity/runs/track1_seedvc_prompt_baseline_30pairs`
- SeedVC prompt listening HTML: `/localdisk/bowen/singing_identity/runs/track1_seedvc_prompt_baseline_30pairs/listening_review.html`
- SeedVC prompt posthoc report: `/home/bowen/bowen_lab/projects/singing_identity/results/seedvc_30pair_prompt_gap_posthoc_zh_2026-06-28.md`
- SeedVC component ablation 12 pairs: `/localdisk/bowen/singing_identity/runs/track1_seedvc_component_ablation_12pairs`
- SeedVC component listening HTML: `/localdisk/bowen/singing_identity/runs/track1_seedvc_component_ablation_12pairs/component_ablation_review.html`
- Breathy detection run: `/localdisk/bowen/singing_identity/runs/track2_breathy_detection_probe_2026-06-28`
- Breathy subspace run: `/localdisk/bowen/singing_identity/runs/track2_breathy_subspace_probe_2026-06-28`

## 本轮新增脚本

- `/home/bowen/bowen_lab/projects/singing_identity/scripts/intervention/analyze_seedvc_prompt_gap_posthoc.py`
- `/home/bowen/bowen_lab/projects/singing_identity/scripts/intervention/evaluate_seedvc_multicondition_outputs.py`
- `/home/bowen/bowen_lab/projects/singing_identity/scripts/intervention/build_multicondition_listening_review.py`
