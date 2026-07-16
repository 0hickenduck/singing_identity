# SeedVC 30-pair Prompt Gap Posthoc

## 结论

- SeedVC 自动 proxy 上，singing prompt 相比 speech prompt 更接近 target singing：mean=0.0870, 95% bootstrap CI=[0.0671, 0.1064], sign=28/30 positive。
- 同时它更不接近 target speech：mean=-0.0694, sign=0/30 positive。 这提示 Resemblyzer 差值混有 singing/speech 域匹配，不是纯身份判断。
- 对 source singing 的平均变化很小：mean=0.0130。 说明这次 prompt 模式主要改变 target-domain 相似度，不像是简单把输出整体推向 source。
- target-singing gain 减 target-speech gain 的域优势很大：mean=0.1564。 这更像“目标 singing reference 的域/唱法匹配优势”，需要听评确认它是不是人耳身份优势。

## 自动 QA

| check | n | pearson | spearman |
|---|---:|---:|---:|
| rms_delta_db_vs_singing_domain_advantage | 30 | -0.385 | -0.408 |
| rms_delta_db_vs_source_gain | 30 | 0.107 | 0.031 |
| rms_delta_db_vs_target_singing_gain | 30 | -0.609 | -0.645 |
| rms_delta_db_vs_target_singing_gain_resid_language | 30 | -0.343 | -0.415 |
| rms_delta_db_vs_target_singing_gain_resid_rms | 30 | 0.000 | -0.100 |
| rms_delta_db_vs_target_singing_gain_resid_rms_language | 30 | 0.000 | -0.127 |
| rms_delta_db_vs_target_singing_gain_resid_rms_target | 30 | 0.000 | -0.114 |
| rms_delta_db_vs_target_speech_gain | 30 | -0.163 | -0.065 |
| singing_domain_advantage_vs_source_gain | 30 | 0.291 | 0.350 |
| singing_domain_advantage_vs_target_singing_gain | 30 | 0.866 | 0.860 |
| singing_domain_advantage_vs_target_singing_gain_resid_language | 30 | 0.800 | 0.823 |
| singing_domain_advantage_vs_target_singing_gain_resid_rms | 30 | 0.796 | 0.791 |
| singing_domain_advantage_vs_target_singing_gain_resid_rms_language | 30 | 0.784 | 0.838 |
| singing_domain_advantage_vs_target_singing_gain_resid_rms_target | 30 | 0.783 | 0.823 |
| singing_domain_advantage_vs_target_speech_gain | 30 | -0.656 | -0.671 |
| source_gain_vs_singing_domain_advantage | 30 | 0.291 | 0.350 |
| source_gain_vs_target_singing_gain | 30 | 0.349 | 0.348 |
| source_gain_vs_target_singing_gain_resid_language | 30 | 0.385 | 0.361 |
| source_gain_vs_target_singing_gain_resid_rms | 30 | 0.522 | 0.517 |
| source_gain_vs_target_singing_gain_resid_rms_language | 30 | 0.419 | 0.391 |
| source_gain_vs_target_singing_gain_resid_rms_target | 30 | 0.432 | 0.393 |
| source_gain_vs_target_speech_gain | 30 | -0.045 | -0.099 |

## Track 1 Accounting Join

这些是 SeedVC target-singing gain 与 Track 1 v2 prediction rows 的最高绝对 Spearman 相关。它们是探索性检查，不应当单独当显著性结论。

| rep | K | model | predictor | n | pearson | spearman |
|---|---:|---|---|---:|---:|---:|
| track1_prompt_mismatch_accounting_v2_2026-06-28/mert_l3 | 64 | L3_low_content | pred_delta_norm | 30 | -0.630 | -0.704 |
| track1_prompt_mismatch_accounting_v2_controls_2026-06-28/mert_l3 | 64 | L3_low_content | pred_delta_norm | 30 | -0.630 | -0.704 |
| track1_prompt_mismatch_accounting_v2_2026-06-28/mert_l3 | 8 | L4_technique | pred_delta_norm | 30 | -0.636 | -0.701 |
| track1_prompt_mismatch_accounting_v2_2026-06-28/mert_l3 | 16 | L3_low_content | pred_delta_norm | 30 | -0.633 | -0.700 |
| track1_prompt_mismatch_accounting_v2_2026-06-28/mert_l3 | 32 | L3_low_content | pred_delta_norm | 30 | -0.630 | -0.700 |
| track1_prompt_mismatch_accounting_v2_controls_2026-06-28/mert_l3 | 16 | L3_low_content | pred_delta_norm | 30 | -0.633 | -0.700 |
| track1_prompt_mismatch_accounting_v2_2026-06-28/mert_l12 | 8 | L4_technique | pred_delta_norm | 30 | -0.577 | -0.690 |
| track1_prompt_mismatch_accounting_v2_controls_2026-06-28/mert_l3 | 16 | C_l4_shuffle_technique | pred_delta_norm | 30 | -0.631 | -0.687 |
| track1_prompt_mismatch_accounting_v2_2026-06-28/mert_l3 | 8 | L3_low_content | pred_delta_norm | 30 | -0.637 | -0.686 |
| track1_prompt_mismatch_accounting_v2_2026-06-28/mert_l12 | 16 | L4_technique | pred_delta_norm | 30 | -0.571 | -0.683 |

去掉 RMS delta 和 language 后，target-singing gain residual 与 Track 1 predictors 的最高相关：

| rep | K | model | predictor | n | pearson | spearman |
|---|---:|---|---|---:|---:|---:|
| track1_prompt_mismatch_accounting_v2_controls_2026-06-28/mert_l3 | 64 | C_l4_shuffle_technique | residual_norm | 30 | 0.192 | 0.232 |
| track1_prompt_mismatch_accounting_v2_2026-06-28/mert_l3 | 64 | L3_low_content | residual_norm | 30 | 0.193 | 0.229 |
| track1_prompt_mismatch_accounting_v2_controls_2026-06-28/mert_l3 | 64 | L3_low_content | residual_norm | 30 | 0.193 | 0.229 |
| track1_prompt_mismatch_accounting_v2_controls_2026-06-28/mert_l12 | 16 | C_acoustic_only | pred_delta_norm | 30 | -0.311 | -0.225 |
| track1_prompt_mismatch_accounting_v2_2026-06-28/mert_l3 | 32 | L3_low_content | residual_norm | 30 | 0.189 | 0.220 |
| track1_prompt_mismatch_accounting_v2_controls_2026-06-28/mert_l3 | 16 | C_l4_shuffle_technique | residual_norm | 30 | 0.190 | 0.218 |
| track1_prompt_mismatch_accounting_v2_2026-06-28/wavlm_l9 | 32 | L1_nuisance | residual_norm | 30 | -0.144 | -0.214 |
| track1_prompt_mismatch_accounting_v2_controls_2026-06-28/mert_l12 | 64 | C_acoustic_only | pred_delta_norm | 30 | -0.305 | -0.212 |
| track1_prompt_mismatch_accounting_v2_2026-06-28/mert_l3 | 16 | L3_low_content | residual_norm | 30 | 0.190 | 0.212 |
| track1_prompt_mismatch_accounting_v2_controls_2026-06-28/mert_l3 | 16 | L3_low_content | residual_norm | 30 | 0.190 | 0.212 |

## 人需要介入的位置

- 现在需要人工听评的不是“是否有自动差值”，而是自动差值究竟对应人耳身份、唱法/气声、还是 speech-vs-singing 域匹配。
- 优先听 target-singing gain 高、但 target-speech gain 低的样本；这类最容易让 Resemblyzer 把域匹配当身份。
- 本轮听评页面：`listening_review.html`，每个 pair 包含 source、target speech、target singing、speech prompt generation、singing prompt generation。
