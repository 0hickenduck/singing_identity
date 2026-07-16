# 需要人工检查：Track 1 提示模式错配核算与 Breathy 技术方向

日期：2026-06-28 JST

## 一句话结论

当前自动实验已经跑到需要人工判断的阶段：Track 1 的“全量因素核算模型”暂时没有超过全局平均残差基线，反而是 F0/时长/能量这组简单 nuisance covariates 在所有表示空间里稳定改善；同时，speaker-balanced 的 breathy 技术方向在 analogy top-1 上基本没有超过 wrong-technique 或 shuffled-direction 控制。因此下一步不是继续盲目扩展算力，而是决定论文叙事是否收窄、模型是否重做成更低维/更强正则的核算模型，以及 breathy 方向证据是否转为负结果或重新设计。

## 已完成的自动运行

### Track 1 Prompt-Mode Mismatch Accounting

- 输出根目录：`/localdisk/bowen/singing_identity/runs/track1_prompt_mismatch_accounting_2026-06-28`
- 主脚本：`/home/bowen/bowen_lab/projects/singing_identity/scripts/probing/run_prompt_mismatch_accounting.py`
- 数据：4,000 个 same-singer same-text speech/singing pairs，20 speakers，5-fold speaker-disjoint split。
- 每个表示都输出 `metrics.json`、`predictions.jsonl` 和一个 markdown summary。
- Acoustic baseline 的第一次运行发现目标泄漏：用 acoustic delta 预测 acoustic delta 会得到完美分数。脚本已补丁为当 target extractor 是 `acoustic_baseline` 时自动禁用 acoustic covariates，并重跑为 `acoustic_no_leak`。

### Track 2 Breathy Technique Directions

- 输出根目录：`/localdisk/bowen/singing_identity/runs/track2_breathy_technique_directions_2026-06-28`
- 使用 `gtsinger_phoneme_pairs.jsonl`，过滤 `<AP>/<SP>` 等静音符号。
- 第一版 cap manifest 发现许多 phone group 只来自 1 个 speaker，因此已废弃为诊断结果。
- 最终用于判断的是 speaker-balanced manifest：`voiced_phone_pairs_breathy_headline_with_controls_speaker_balanced.jsonl`。
- 该 manifest 有 3,915 个 interval pairs，其中 breathy 3,000 个；30 个 breathy phone groups，每组覆盖 2-3 speakers；非 breathy rows 只用于 wrong-technique control。

## 图解：当前实验在问什么

```text
speech reference z_speech  ----\
                              +--> Delta_z = z_singing - z_speech
singing reference z_singing --/

核算目标：哪些可测因素能预测 Delta_z？

M0: 全局平均 Delta_z
M1: 文本/phone histogram
M2: M1 + F0/prosody
M3: M2 + timing/energy
M4: M3 + acoustic/phonation proxy
M5: M4 + technique label

关键判断：
1. M5 是否比 M0 更好？
2. 简单 F0/timing/energy-only 是否已经解释主要变化？
3. 残差或 raw Delta 是否能解释 Seed-VC speech-prompt vs singing-prompt gap？
```

## Track 1 结果表

| 表示 | M0 R@1 | M5 R@1 | F0/时长/能量 R@1 | M5 MSE 改善 vs M0 | F0/时长/能量 MSE 改善 vs M0 | raw Delta vs Seed-VC gap Pearson |
|---|---:|---:|---:|---:|---:|---:|
| WavLM Base+ L6 | 0.522 | 0.455 | 0.604 | -0.095 | 0.078 | 0.567 |
| WavLM Base+ L9 | 0.544 | 0.517 | 0.634 | -0.062 | 0.094 | 0.520 |
| HuBERT Base L12 | 0.630 | 0.566 | 0.696 | -0.162 | 0.035 | 0.586 |
| MERT 95M L3 | 0.528 | 0.459 | 0.567 | -0.179 | 0.054 | 0.762 |
| MERT 95M L12 | 0.426 | 0.443 | 0.492 | -0.214 | 0.062 | 0.222 |
| Acoustic baseline（无泄漏） | 0.208 | 0.194 | 0.222 | -0.220 | 0.015 | 0.177 |

### Track 1 读法

1. `M5 MSE 改善 vs M0` 全部为负：当前全量 ridge 核算模型在 held-out speakers 上没有超过“全局平均 speech-to-singing delta”。这不是理想的正结果。
2. `F0/时长/能量-only` 在所有表示中 MSE 改善为正，且 R@1 往往高于 M0/M5。这说明当前 prompt-mode mismatch 很大一部分更像低维 prosody/acoustic 差异，而不是一个复杂 technique-label residual。
3. raw Delta norm 与 Seed-VC prompt gap 在 10 个 baseline pairs 上有中等正相关，但 n=10 太小，只能作为 sanity link，不能作为强结论。MERT L3 的 Pearson 最高（0.762），但需要扩展 Seed-VC 样本验证。
4. 技术标签从 M4 到 M5 的增益很小，甚至 R@1 有时下降。当前不能主张 “technique label 显著解释 prompt mismatch”。

## 图解：为什么现在需要人工判断

```text
如果目标是“证明有稳定身份残差”：
  当前 M5 < M0  => 证据不足

如果目标是“诊断 speech prompt 为什么不等于 singing prompt”：
  F0/时长/能量-only 稳定 > M0  => 有可写的负/诊断型结果

如果目标是“把 technique 作为核心变量”：
  M5-M4 很小 + technique direction 控制不过关 => 需要重新设计或收窄
```

## Breathy Technique Direction 结果表（speaker-balanced）

| 表示 | breathy groups | 平均 speakers/group | bootstrap angle 均值 | analogy top1 | wrong-technique top1 | shuffled-direction top1 |
|---|---:|---:|---:|---:|---:|---:|
| MERT 95M L3 | 30 | 2.5 | 31.8 | 0.225 | 0.223 | 0.226 |
| WavLM Base+ L6 | 30 | 2.5 | 37.4 | 0.222 | 0.222 | 0.223 |
| WavLM Base+ L9 | 30 | 2.5 | 42.2 | 0.243 | 0.245 | 0.243 |

### Breathy 读法

1. 平均 analogy top1 与 wrong-technique、shuffled-direction 几乎相同：这说明当前 breathy vector arithmetic 没有形成稳定、可区分的 technique direction。
2. 个别 phone group（例如 `e_es`）看起来 top1 很高，但 wrong/shuffled 也同样高，说明它更可能是 phone/speaker/local structure，而不是 breathy 方向本身。
3. bootstrap angle 在 speaker-balanced 后升高，尤其 WavLM L9 平均约 42 度；方向稳定性一般。
4. 当前不能把 breathy 作为“明天 meeting 的强正证据”。更合适的说法是：在严格去掉 silence 并做 speaker-balanced 后，breathy 方向没有明显超过控制。

## Breathy 最稳定 group 快照

### MERT 95M L3

| phone | n | speakers | angle | analogy | wrong | shuffled | within cosine | between cosine |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| e_es | 80 | 2 | 22.1 | 0.688 | 0.600 | 0.675 | 0.075 | -0.003 |
| a_es | 80 | 2 | 25.6 | 0.125 | 0.125 | 0.125 | 0.056 | -0.003 |
| AY1_en | 120 | 3 | 26.9 | 0.275 | 0.275 | 0.275 | 0.036 | 0.002 |
| o_es | 80 | 2 | 27.9 | 0.400 | 0.375 | 0.388 | 0.041 | -0.001 |
| ɐ_ko | 120 | 3 | 28.1 | 0.167 | 0.142 | 0.167 | 0.031 | 0.022 |
| N_en | 120 | 3 | 28.3 | 0.342 | 0.333 | 0.325 | 0.029 | 0.002 |
| ʌ_ko | 120 | 3 | 30.0 | 0.083 | 0.092 | 0.083 | 0.023 | 0.008 |
| i_es | 80 | 2 | 30.2 | 0.225 | 0.225 | 0.200 | 0.038 | -0.001 |

### WavLM Base+ L6

| phone | n | speakers | angle | analogy | wrong | shuffled | within cosine | between cosine |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ɐ_ko | 120 | 3 | 27.1 | 0.167 | 0.167 | 0.150 | 0.030 | 0.032 |
| AY1_en | 120 | 3 | 32.0 | 0.200 | 0.208 | 0.208 | 0.025 | 0.002 |
| e_es | 80 | 2 | 32.3 | 0.725 | 0.700 | 0.738 | 0.027 | -0.001 |
| ɭ_ko | 120 | 3 | 33.1 | 0.108 | 0.108 | 0.108 | 0.012 | 0.010 |
| a_es | 80 | 2 | 33.7 | 0.163 | 0.150 | 0.163 | 0.023 | 0.011 |
| i_ko | 120 | 3 | 34.1 | 0.217 | 0.225 | 0.208 | 0.017 | 0.010 |
| R_en | 120 | 3 | 34.3 | 0.233 | 0.233 | 0.233 | 0.018 | 0.006 |
| ə_de | 80 | 2 | 34.6 | 0.212 | 0.250 | 0.237 | 0.025 | 0.006 |

### WavLM Base+ L9

| phone | n | speakers | angle | analogy | wrong | shuffled | within cosine | between cosine |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ɐ_ko | 120 | 3 | 31.5 | 0.183 | 0.175 | 0.167 | 0.017 | 0.023 |
| ɭ_ko | 120 | 3 | 33.6 | 0.125 | 0.125 | 0.125 | 0.010 | 0.011 |
| e_es | 80 | 2 | 37.5 | 0.838 | 0.838 | 0.863 | 0.014 | 0.001 |
| i_ko | 120 | 3 | 38.1 | 0.225 | 0.233 | 0.225 | 0.008 | 0.007 |
| L_en | 120 | 3 | 38.9 | 0.242 | 0.233 | 0.233 | 0.011 | 0.003 |
| R_en | 120 | 3 | 39.1 | 0.267 | 0.275 | 0.258 | 0.012 | 0.005 |
| N_en | 120 | 3 | 39.6 | 0.342 | 0.317 | 0.333 | 0.010 | 0.002 |
| i_it | 120 | 3 | 40.4 | 0.092 | 0.075 | 0.058 | 0.007 | 0.007 |

## 需要人工确认的决策

### 决策 A：Track 1 是否继续作为主线？

建议：继续，但收窄为“prompt-mode mismatch diagnostic”，不要声称 disentanglement 或 strong identity residual。当前最稳的论点是：same-person speech/singing prompt mismatch 在 frozen representations 中存在，但可预测部分主要由 F0、duration、energy 等低维因素解释。

需要你确认：是否接受把 Track 1 的叙事改成诊断/负结果方向，而不是寻找复杂 residual 的正结果？

### 决策 B：是否重写 accounting model？

建议：需要。当前 M1 的 phone histogram 维度高，只有 20 speakers，ridge 仍可能过拟合。下一版应先做低维模型：只用 nuisance covariates、语言/音域 controls、少量 phone summary，不直接塞 400+ phone histogram；或用 PCA/PLS/ElasticNet，但仍保持 speaker-disjoint。

需要你确认：是否把下一轮计算资源放在“更干净的低维 accounting model”，而不是继续横向跑更多 encoders？

### 决策 C：Breathy 技术方向是否作为 meeting headline？

建议：不要作为强正 headline。可以作为严谨 negative/control result：去掉 `<AP>`、做 voiced-phone 和 speaker-balanced 后，breathy direction 不明显超过 wrong/shuffled control。

需要你确认：明天 1:1 是想展示这个负结果，还是希望我改成更可视化的 error analysis（例如列出为什么 `e_es` 假阳性高）？

### 决策 D：Seed-VC 下游 link 是否扩展？

建议：扩展。当前只有 10 pairs，raw Delta norm 与 prompt gap 有中等相关，但不足以支撑结论。下一步最值得跑的是把 Seed-VC prompt baseline 扩到 30-50 pairs，并覆盖非 breathy/control 技术。

需要你确认：是否允许继续消耗 GPU 时间扩展 Seed-VC 下游样本？

## 建议下一步（如果人工确认后继续）

1. 改写 `run_prompt_mismatch_accounting.py` 的模型层级：先低维 nuisance-only，再加入小规模 metadata，再加入 technique；phone histogram 改为 PCA 或少数 phone class counts。
2. 为 Track 1 加一个 permutation test：speaker labels、technique labels、singing target within speaker 都要报告置信区间。
3. Seed-VC prompt baseline 扩到至少 30 pairs；否则 downstream correlation 只能当 sanity check。
4. Breathy 方向改为 error analysis，不再用当前 vector arithmetic 当正结果。

## 主要文件索引

- Track 1 outputs: `/localdisk/bowen/singing_identity/runs/track1_prompt_mismatch_accounting_2026-06-28`
- Track 2 outputs: `/localdisk/bowen/singing_identity/runs/track2_breathy_technique_directions_2026-06-28`
- Main Track 1 script: `/home/bowen/bowen_lab/projects/singing_identity/scripts/probing/run_prompt_mismatch_accounting.py`
- WavLM L6 main summary: `/home/bowen/bowen_lab/projects/singing_identity/results/track1_prompt_mismatch_accounting_2026-06-28.md`
- Corrected acoustic summary: `/home/bowen/bowen_lab/projects/singing_identity/results/track1_prompt_mismatch_accounting_2026-06-28_acoustic_no_leak.md`
- This human-check document: `/home/bowen/bowen_lab/projects/singing_identity/results/human_check_track1_track2_zh_2026-06-28.md`
