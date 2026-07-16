# Human Verdict 与 Objective Follow-up

日期：2026-06-28 JST

## 用户听评反馈摘要

### Prompt mode 30-pair

Human verdict:

- Singing prompt 经常更像 target identity：`yes`，但身份/timbre 判断很弱，因为 target reference 与 generated output 的内容、pitch、time period/phrase 都不完全容易对齐。
- 主要听到的变化不是清晰的 timbre identity，而是 “how the content is being done”。
- 变化包含大量 breathy phonation。
- 因为两个 generated outputs 唱的是同一内容，所以能听到一些 timbre difference；但难以判断哪个 timbre 更接近原 target。合理假设 singing prompt 更接近 target singing，但这不是强主观证据。
- 没有明确标注 speech prompt wins。
- `artifact-heavy` 的定义对用户不清晰；后续不应依赖这个开放问题。这里 artifact-heavy 指明显破音、爆音、断裂、严重噪声、内容丢失、音高/节奏崩坏等。

Interpretation:

Prompt-mode 听评支持 “audible prompt effect”，但不支持强 claim：

> singing prompt improves target identity.

更可靠的 human-grounded 说法是：

> singing prompt changes content delivery, phonation, and acoustic style in a way that often sounds more compatible with target singing.

### Component ablation 12-pair

Human verdict:

- `mel_style` 接近 `oracle_all_singing`: `yes`。
- 但几乎所有 ablated outputs 都质量较好、彼此接近。
- `mel_only` 有时缺少 technique。
- `style` 有时包含 technique/phonation 信息。
- 最后三个主观问题很难判断：哪个 single component 最明显、style 是 identity/timbre/technique 还是 loudness、prompt_seq_only 是否无效。
- 用户认为这种 component-level subjective evaluation 对人非常难，不适合作为研究核心证据。

Interpretation:

Component ablation 保留为工程诊断，不作为论文主证据。它的作用是告诉我们 SeedVC 内部差异更可能经过 `mel context` / `style embedding`，而不是 prompt semantic/F0 sequence 单独驱动。

## 为什么 ablation 仍然有用，但不是 required proof

Component ablation 的目的不是证明 scientific claim，而是 debug SeedVC 机制：

```text
如果 prompt_seq_only 有效 -> 继续做 semantic/F0 prompt mapper
如果 style_only 有效 -> 看 CAMPPlus style / speaker-style embedding
如果 mel_only 有效 -> 看 prompt acoustic context
如果 mel+style 接近 oracle -> 下游差异主要在 acoustic/style side，不在 prompt_seq alone
```

现在自动结果和用户反馈都指向：

```text
prompt_seq_only: 不可靠/弱
mel + style: 接近 oracle
subjective component attribution: 太难，不应作为主证据
```

所以后续决策：

1. 不再要求用户继续做细粒度 component subjective attribution。
2. SeedVC component ablation 作为 engineering diagnostic 附录或内部指导。
3. 主线实验转向 objective acoustic / phonation / representation metrics。

## Objective acoustic follow-up

新增脚本：

`/home/bowen/bowen_lab/projects/singing_identity/scripts/intervention/evaluate_seedvc_acoustic_objective.py`

它使用以下客观声学 proxy：

- duration
- RMS / peak
- F0 mean/std/range
- voiced percentage
- spectral centroid/bandwidth/rolloff
- spectral flatness
- zero-crossing rate
- low/high spectral ratio
- high-band ratio

这些指标不是 identity labels，但能帮助判断 prompt effect 是否更像 phonation/content-delivery/acoustic-domain shift。

### Prompt mode 30-pair acoustic objective

相对 `target_speech_prompt`，`target_singing_prompt`：

| metric | mean delta |
|---|---:|
| acoustic distance to target singing | -1.4452 |
| acoustic distance to target speech | +0.0265 |
| acoustic distance to source | -0.4280 |
| RMS dB | -0.3113 |
| F0 mean Hz | -2.11 |
| F0 std Hz | +8.20 |
| spectral centroid Hz | +74.08 |
| low/high spectral ratio dB | +3.0202 |
| high-band ratio | -0.0004 |

Reading:

- Negative distance to target singing means singing prompt is objectively closer to target singing acoustic profile than speech prompt.
- It is not similarly closer to target speech.
- Spectral tilt and F0-variation movement support the user’s “content delivery / breathy phonation / acoustic style” description.
- This still does not prove target identity.

### Component ablation 12-pair acoustic objective

相对 `baseline_all_speech`：

| condition | d dist target singing | d dist target speech | d dist source | d RMS dB | d F0 std | d tilt low/high |
|---|---:|---:|---:|---:|---:|---:|
| oracle_all_singing | -1.3753 | +0.2744 | -0.1049 | -0.6148 | +2.30 | +3.8002 |
| singing_mel_style | -1.2213 | +0.2823 | -0.1453 | -0.5610 | +0.54 | +3.1187 |
| singing_style_only | -0.6789 | +0.0232 | -0.6660 | -2.4035 | -13.95 | +2.4167 |
| singing_mel_only | -0.1763 | +0.3080 | +0.0435 | +1.8593 | -3.27 | -0.4249 |
| singing_prompt_seq_only | +0.1584 | +0.0996 | +0.4908 | +2.0020 | +1.59 | -0.8113 |

Reading:

- `mel_style` is objectively close to `oracle_all_singing` in acoustic distance to target singing.
- `prompt_seq_only` moves away from target singing acoustic profile.
- `style_only` and `mel_only` each move some aspects, but differently.
- This supports using ablation as a mechanistic hint, not as a human-evaluable claim.

## Updated research stance

### What we can say

Track 1:

> Frozen representations contain a predictable speech-to-singing prompt mismatch. The predictable component is largely low-dimensional acoustic/prosodic/phonation-related, and low-rank/regularized accounting avoids the high-dimensional overfitting that made earlier M5 underperform M0.

SeedVC link:

> In SeedVC, using a target singing prompt produces a robust automatic and acoustic shift toward the target singing reference. Human listening confirms audible differences, mostly in content delivery / breathy phonation / acoustic style. This is not yet strong evidence of improved target identity.

Track 2:

> Breathy is encoded and detectable after nuisance residualization, but does not behave as one clean global transferable vector direction.

### What we should not say

- Do not say SeedVC singing prompt clearly improves target identity.
- Do not say component ablation has a reliable subjective component ranking.
- Do not use subjective component attribution as a core proof.
- Do not claim music-oriented MERT uniquely dominates WavLM/HuBERT.

## Next experiment direction

Primary:

1. Keep Track 1 v2 accounting as the main quantitative result.
2. Keep Track 2 breathy detection/subspace as the technique/phonation result.
3. Treat SeedVC as downstream engineering evidence that prompt mismatch manifests audibly and acoustically, but currently closer to phonation/acoustic-style than identity.

Optional if we continue SeedVC:

1. Build objective mel/style mapper or interpolation.
2. Evaluate with acoustic objective + speaker proxy.
3. Only use human listening at the end as demo sanity check, not as a detailed component-labeling task.

## Artifacts

- Prompt acoustic objective report: `/home/bowen/bowen_lab/projects/singing_identity/results/seedvc_prompt_30pair_acoustic_objective_zh_2026-06-28.md`
- Component acoustic objective report: `/home/bowen/bowen_lab/projects/singing_identity/results/seedvc_component_12pair_acoustic_objective_zh_2026-06-28.md`
- Prompt acoustic objective run: `/localdisk/bowen/singing_identity/runs/track1_seedvc_prompt_baseline_30pairs/acoustic_objective_summary.json`
- Component acoustic objective run: `/localdisk/bowen/singing_identity/runs/track1_seedvc_component_ablation_12pairs/acoustic_objective_summary.json`
