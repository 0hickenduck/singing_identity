import marimo

__generated_with = "0.17.6"
app = marimo.App(width="full")


@app.cell
def _():
    import json
    from pathlib import Path

    import marimo as mo

    REPO = Path("/home/bowen/bowen_lab/projects/singing_identity")
    PROMPT_RUN = Path("/localdisk/bowen/singing_identity/runs/track1_seedvc_prompt_baseline_30pairs")
    COMPONENT_RUN = Path("/localdisk/bowen/singing_identity/runs/track1_seedvc_component_ablation_12pairs")
    TRACK1_RUN = Path("/localdisk/bowen/singing_identity/runs/track1_prompt_mismatch_accounting_v2_2026-06-28")
    TRACK1_CONTROLS = Path("/localdisk/bowen/singing_identity/runs/track1_prompt_mismatch_accounting_v2_controls_2026-06-28")
    BREATHY_DETECTION = Path("/localdisk/bowen/singing_identity/runs/track2_breathy_detection_probe_2026-06-28")
    BREATHY_SUBSPACE = Path("/localdisk/bowen/singing_identity/runs/track2_breathy_subspace_probe_2026-06-28")
    return (
        BREATHY_DETECTION,
        BREATHY_SUBSPACE,
        COMPONENT_RUN,
        PROMPT_RUN,
        REPO,
        TRACK1_CONTROLS,
        TRACK1_RUN,
        json,
        mo,
    )


@app.cell
def _(COMPONENT_RUN, PROMPT_RUN, json):
    prompt_summary = json.loads((PROMPT_RUN / "posthoc_prompt_gap_accounting.json").read_text(encoding="utf-8"))
    component_summary = json.loads((COMPONENT_RUN / "resemblyzer_multicondition_summary.json").read_text(encoding="utf-8"))
    prompt_acoustic = json.loads((PROMPT_RUN / "acoustic_objective_summary.json").read_text(encoding="utf-8"))
    component_acoustic = json.loads((COMPONENT_RUN / "acoustic_objective_summary.json").read_text(encoding="utf-8"))
    return component_acoustic, component_summary, prompt_acoustic, prompt_summary


@app.cell
def _(mo):
    mo.md("""
    # 需要人工检查：Track 1 / SeedVC / Breathy

    这个 notebook 是本轮 human checkpoint 的入口。音频逐 pair 检查请打开下面两个专门 notebook：

    - `notebooks/track1_seedvc_prompt_review.py`
    - `notebooks/track1_seedvc_component_ablation_review.py`

    当前自动实验已经能判断：Track 1 v2 不是负结果；breathy 不是 dead end；SeedVC prompt-mode proxy 有稳定差异。

    现在不能自动判断的是：这个 SeedVC proxy 差异在人耳上到底是 target identity，还是 singing-domain / breathy-phonation / loudness-quality。
    """)
    return


@app.cell
def _(COMPONENT_RUN, PROMPT_RUN, REPO, mo):
    mo.md(f"""
    ## SSH / Marimo 打开方式

    当前机器是 lab GPU node。推荐从本地开 SSH tunnel：

    ```bash
    ssh -L 2719:127.0.0.1:2719 bowen@valkyrie03.gavo.t.u-tokyo.ac.jp
    ```

    在 lab server 上启动总入口：

    ```bash
    cd {REPO}
    uv run --with marimo marimo edit notebooks/human_check_track1_seedvc_track2_2026_06_28.py --host 127.0.0.1 --port 2719 --headless
    ```

    然后本地浏览器打开：

    ```text
    http://127.0.0.1:2719
    ```

    也可以直接打开专门听评 notebook：

    ```bash
    uv run --with marimo marimo edit notebooks/track1_seedvc_prompt_review.py --host 127.0.0.1 --port 2720 --headless
    uv run --with marimo marimo edit notebooks/track1_seedvc_component_ablation_review.py --host 127.0.0.1 --port 2721 --headless
    ```

    Run roots:

    - Prompt mode: `{PROMPT_RUN}`
    - Component ablation: `{COMPONENT_RUN}`
    """)
    return


@app.cell
def _(mo, prompt_summary):
    gaps = prompt_summary["seedvc"]["gap_summaries"]
    mo.md(
        f"""
        ## Prompt Mode 30-pair 自动结果

        | 指标 | mean | 95% bootstrap CI | sign |
        |---|---:|---:|---:|
        | singing prompt 对 target singing 的增益 | `{gaps["target_singing_gain"]["mean"]:.4f}` | `[{gaps["target_singing_gain"]["mean_ci95_bootstrap"][0]:.4f}, {gaps["target_singing_gain"]["mean_ci95_bootstrap"][1]:.4f}]` | `{gaps["target_singing_gain"]["sign_test"]["positive"]}/{gaps["target_singing_gain"]["sign_test"]["n"]}` |
        | singing prompt 对 target speech 的增益 | `{gaps["target_speech_gain"]["mean"]:.4f}` | `[{gaps["target_speech_gain"]["mean_ci95_bootstrap"][0]:.4f}, {gaps["target_speech_gain"]["mean_ci95_bootstrap"][1]:.4f}]` | `{gaps["target_speech_gain"]["sign_test"]["positive"]}/{gaps["target_speech_gain"]["sign_test"]["n"]}` |
        | singing prompt 对 source singing 的增益 | `{gaps["source_gain"]["mean"]:.4f}` | `[{gaps["source_gain"]["mean_ci95_bootstrap"][0]:.4f}, {gaps["source_gain"]["mean_ci95_bootstrap"][1]:.4f}]` | `{gaps["source_gain"]["sign_test"]["positive"]}/{gaps["source_gain"]["sign_test"]["n"]}` |
        | target-singing gain - target-speech gain | `{gaps["singing_domain_advantage"]["mean"]:.4f}` | `[{gaps["singing_domain_advantage"]["mean_ci95_bootstrap"][0]:.4f}, {gaps["singing_domain_advantage"]["mean_ci95_bootstrap"][1]:.4f}]` | `{gaps["singing_domain_advantage"]["sign_test"]["positive"]}/{gaps["singing_domain_advantage"]["sign_test"]["n"]}` |

        自动 verdict：prompt mode effect 很稳定，但它不是纯 identity 证据。它可能是 identity，也可能是 singing-domain / phonation / loudness-quality。
        """
    )
    return


@app.cell
def _(component_summary, mo):
    rows = component_summary["condition_summary"]
    order = [
        "oracle_all_singing",
        "singing_mel_only",
        "singing_style_only",
        "singing_prompt_seq_only",
        "singing_prompt_seq_mel",
        "singing_prompt_seq_style",
        "singing_mel_style",
    ]
    table = "\n".join(
        [
            "| condition | d target singing | d target speech | d source | d RMS dB | positive |",
            "|---|---:|---:|---:|---:|---:|",
            *[
                f"| `{name}` | `{rows[name]['mean_delta_to_target_singing']:.4f}` | "
                f"`{rows[name]['mean_delta_to_target_speech']:.4f}` | "
                f"`{rows[name]['mean_delta_to_source']:.4f}` | "
                f"`{rows[name]['mean_delta_rms_db']:.3f}` | "
                f"`{rows[name]['positive_delta_to_target_singing']}/{rows[name]['pairs']}` |"
                for name in order
            ],
        ]
    )
    mo.md(
        f"""
        ## Component Ablation 12-pair 自动结果

        {table}

        自动 verdict：

        - `prompt_seq_only` 基本不是驱动项，而且方向偏负。
        - `mel_only` 和 `style_only` 都有贡献。
        - `mel + style` 几乎复现 `oracle_all_singing`。
        - 但是否是 identity，需要听。
        """
    )
    return


@app.cell
def _(component_acoustic, mo, prompt_acoustic):
    prompt = prompt_acoustic["condition_summary"]["target_singing_prompt"]
    comp = component_acoustic["condition_summary"]
    mo.md(
        f"""
        ## Objective Acoustic Check

        Negative distance delta means closer to that reference than baseline in a z-scored acoustic feature space using F0, voicing, RMS, spectral centroid/rolloff/flatness, high-band ratio, and spectral tilt proxies.

        Prompt mode 30-pair, `target_singing_prompt` vs `target_speech_prompt`:

        - acoustic distance to target singing: `{prompt["delta_dist_to_target_singing_mean"]:.4f}`
        - acoustic distance to target speech: `{prompt["delta_dist_to_target_speech_mean"]:.4f}`
        - acoustic distance to source: `{prompt["delta_dist_to_source_mean"]:.4f}`
        - spectral tilt low/high delta: `{prompt["delta_low_high_ratio_db_mean"]:.4f}`
        - F0 std delta: `{prompt["delta_f0_std_hz_mean"]:.2f}`

        Component ablation 12-pair, acoustic distance to target singing:

        - `oracle_all_singing`: `{comp["oracle_all_singing"]["delta_dist_to_target_singing_mean"]:.4f}`
        - `singing_mel_style`: `{comp["singing_mel_style"]["delta_dist_to_target_singing_mean"]:.4f}`
        - `singing_style_only`: `{comp["singing_style_only"]["delta_dist_to_target_singing_mean"]:.4f}`
        - `singing_mel_only`: `{comp["singing_mel_only"]["delta_dist_to_target_singing_mean"]:.4f}`
        - `singing_prompt_seq_only`: `{comp["singing_prompt_seq_only"]["delta_dist_to_target_singing_mean"]:.4f}`

        Reading: objective acoustics support a singing/phonation/content-delivery shift. They do not prove identity.
        """
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Track 1 / Breathy 当前结论

    Track 1：

    - 旧版 M5 弱于 M0 主要是高维小样本下的 overfitting / model misspecification。
    - v2 低维输入 + 低秩 target 后，所有主表示都超过 M0 mean-delta baseline。
    - 可预测部分主要是 F0/prosody/duration/energy/acoustic proxy；discrete technique label 不是主导。

    Breathy：

    - 单个 global breathy vector 不成立。
    - 但 breathy 在 SSL representation 中可检测；residualize duration/F0/energy 后仍可检测。
    - 更准确的说法是：`breathy is encoded, but not as one clean global direction`。
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 需要你回的 verdict

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

    这个 verdict 决定下一步：

    - 如果是 identity：继续 SeedVC 内部 mel/style 或 CAMPPlus style mapping。
    - 如果是 domain/phonation：Track 1 作为 prompt mismatch accounting 保留，SeedVC link 改写成 domain/phonation mechanism。
    - 如果 artifact/loudness 主导：rerun 高 diffusion steps / 更严 loudness QA / 更均衡 pairs。
    """)
    return


@app.cell
def _(
    BREATHY_DETECTION,
    BREATHY_SUBSPACE,
    COMPONENT_RUN,
    PROMPT_RUN,
    TRACK1_CONTROLS,
    TRACK1_RUN,
    mo,
):
    mo.md(f"""
    ## Artifact Index

    - Track 1 v2 main: `{TRACK1_RUN}`
    - Track 1 v2 controls: `{TRACK1_CONTROLS}`
    - Prompt mode 30-pair: `{PROMPT_RUN}`
    - Component ablation 12-pair: `{COMPONENT_RUN}`
    - Breathy detection: `{BREATHY_DETECTION}`
    - Breathy subspace: `{BREATHY_SUBSPACE}`
    """)
    return


if __name__ == "__main__":
    app.run()
