import marimo

__generated_with = "0.17.6"
app = marimo.App(width="full")


@app.cell
def _():
    import html
    import json
    from pathlib import Path

    import marimo as mo

    RUN_ROOT = Path("/localdisk/bowen/singing_identity/runs/seedvc_native_latents_all200")
    AUDIT_ROOT = RUN_ROOT / "audit"
    return AUDIT_ROOT, RUN_ROOT, html, json, mo


@app.cell
def _(AUDIT_ROOT, json):
    def read_json(path):
        return json.loads(path.read_text(encoding="utf-8"))

    def read_jsonl(path):
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    components = ["style", "semantic_stats", "prompt_stats", "mel_stats", "pooled"]
    summaries = {
        component: read_json(AUDIT_ROOT / f"{component}_summary.json")
        for component in components
    }
    pair_metrics = {
        component: read_jsonl(AUDIT_ROOT / f"{component}_pair_metrics.jsonl")
        for component in components
    }
    return components, pair_metrics, read_json, read_jsonl, summaries


@app.cell
def _(RUN_ROOT, mo, summaries):
    first = next(iter(summaries.values()))
    heldout = ", ".join(first["heldout_speakers"])
    mo.md(
        f"""
        # Seed-VC Native Prompt Latent Audit

        This notebook reviews the all-speaker Seed-VC prompt-latent extraction at:

        `{RUN_ROOT}`

        Split: `{first["train_pairs"]}` train pairs, `{first["test_pairs"]}` heldout pairs, `{first["speakers"]}` speakers.

        Heldout speakers: `{heldout}`

        Decision use: pick the component to perturb first in the next Seed-VC intervention. The strongest current candidate is `semantic_stats`; `prompt_stats` is the secondary candidate. `style` is important because speech/singing are far apart there, but it is a harder and riskier direct injection target.
        """
    )
    return


@app.cell
def _(components, html, mo, summaries):
    def fmt(value, digits=3):
        if value is None:
            return ""
        return f"{float(value):.{digits}f}"

    headers = [
        "component",
        "dim",
        "speech-sing cos",
        "global map cos",
        "ridge map cos",
        "global delta cos",
        "ridge delta cos",
        "delta norm",
        "corr duration",
        "corr rms",
        "corr f0 voiced",
    ]
    rows = []
    for component in components:
        s = summaries[component]
        corr = s["nuisance_correlations_test"]
        rows.append(
            [
                component,
                s["vector_dim"],
                fmt(s["test_speech_to_singing_cosine_mean"]),
                fmt(s["test_global_mapped_to_singing_cosine_mean"]),
                fmt(s["test_ridge_mapped_to_singing_cosine_mean"]),
                fmt(s["test_global_delta_to_true_delta_cosine_mean"]),
                fmt(s["test_ridge_delta_to_true_delta_cosine_mean"]),
                fmt(s["test_delta_norm_mean"]),
                fmt(corr["delta_norm_vs_duration_delta"]),
                fmt(corr["delta_norm_vs_rms_delta"]),
                fmt(corr["delta_norm_vs_f0_voiced_pct_delta"]),
            ]
        )

    table = "<table><thead><tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr></thead><tbody>"
    for row in rows:
        table += "<tr>" + "".join(f"<td>{html.escape(str(cell))}</td>" for cell in row) + "</tr>"
    table += "</tbody></table>"

    mo.vstack(
        [
            mo.md("## Component Summary"),
            mo.Html(table),
            mo.md(
                """
                Reading guide:

                - `speech-sing cos`: how close the unmodified speech prompt latent already is to singing prompt latent.
                - `ridge delta cos`: how well a learned speech-conditioned residual predicts the true speech-to-singing residual direction.
                - Nuisance correlations: high absolute values mean the component may be partly duration, loudness, or voiced-F0 behavior.
                """
            ),
        ]
    )
    return fmt


@app.cell
def _(components, mo):
    metric = mo.ui.dropdown(
        options={
            "Ridge delta-to-true-delta cosine": "test_ridge_delta_to_true_delta_cosine_mean",
            "Global delta-to-true-delta cosine": "test_global_delta_to_true_delta_cosine_mean",
            "Ridge mapped-to-singing cosine": "test_ridge_mapped_to_singing_cosine_mean",
            "Speech-to-singing cosine": "test_speech_to_singing_cosine_mean",
        },
        value="test_ridge_delta_to_true_delta_cosine_mean",
        label="Metric",
    )
    selected_component = mo.ui.dropdown(options=components, value="semantic_stats", label="Component detail")
    mo.hstack([metric, selected_component], justify="start")
    return metric, selected_component


@app.cell
def _(components, html, metric, mo, summaries):
    values = [(component, float(summaries[component][metric.value])) for component in components]
    max_value = max(value for _, value in values) if values else 1.0
    bars = []
    for component, value in values:
        width = 4 if max_value == 0 else max(4, int(420 * value / max_value))
        bars.append(
            f"""
            <div style="display:flex;align-items:center;gap:10px;margin:6px 0;">
              <div style="width:130px;font-family:monospace;">{html.escape(component)}</div>
              <div style="height:18px;width:{width}px;background:#2f6f73;"></div>
              <div style="font-family:monospace;">{value:.3f}</div>
            </div>
            """
        )
    mo.vstack([mo.md("## Metric Comparison"), mo.Html("".join(bars))])
    return


@app.cell
def _(fmt, mo, selected_component, summaries):
    component = selected_component.value
    s = summaries[component]
    corr = s["nuisance_correlations_test"]
    mo.md(
        f"""
        ## `{component}` Interpretation

        - vector dim: `{s["vector_dim"]}`
        - speech-to-singing cosine: `{fmt(s["test_speech_to_singing_cosine_mean"])}`
        - ridge mapped-to-singing cosine: `{fmt(s["test_ridge_mapped_to_singing_cosine_mean"])}`
        - ridge delta-to-true-delta cosine: `{fmt(s["test_ridge_delta_to_true_delta_cosine_mean"])}`
        - delta norm mean: `{fmt(s["test_delta_norm_mean"])}`
        - nuisance correlations: duration `{fmt(corr["delta_norm_vs_duration_delta"])}`, RMS `{fmt(corr["delta_norm_vs_rms_delta"])}`, F0 voiced pct `{fmt(corr["delta_norm_vs_f0_voiced_pct_delta"])}`

        Practical reading: high ridge delta cosine means the residual is predictable from speech. High mapped-to-singing cosine alone is less useful when the unmodified speech latent is already very close to singing.
        """
    )
    return (component,)


@app.cell
def _(component, html, mo, pair_metrics):
    test_rows = [row for row in pair_metrics[component] if row["split"] == "test"]
    test_rows = sorted(test_rows, key=lambda row: row["ridge_delta_to_true_delta_cosine"], reverse=True)
    headers = [
        "pair",
        "speaker",
        "language",
        "speech-sing",
        "ridge map",
        "ridge delta",
        "delta norm",
        "dur delta",
        "rms delta",
        "f0 voiced delta",
    ]
    body = ""
    for row in test_rows:
        cells = [
            row["pair_id"],
            row["speaker_id"],
            row["language"],
            f"{row['speech_to_singing_cosine']:.3f}",
            f"{row['ridge_mapped_to_singing_cosine']:.3f}",
            f"{row['ridge_delta_to_true_delta_cosine']:.3f}",
            f"{row['true_delta_norm']:.3f}",
            f"{row['duration_delta']:.3f}",
            f"{row['rms_delta']:.4f}",
            f"{row['f0_voiced_pct_delta']:.3f}",
        ]
        body += "<tr>" + "".join(f"<td>{html.escape(str(cell))}</td>" for cell in cells) + "</tr>"

    table = "<table><thead><tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr></thead><tbody>" + body + "</tbody></table>"
    mo.vstack([mo.md(f"## Heldout Pair Detail: `{component}`"), mo.Html(table)])
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## Next Action

        The next experiment should be a Seed-VC decoder intervention with two first-pass candidates:

        1. Inject a learned `semantic_stats` residual into prompt conditioning and compare speech prompt, mapped speech prompt, and oracle singing prompt.
        2. Repeat with `prompt_stats` if semantic injection is technically awkward or acoustically unstable.

        Listening review should be packaged as another Marimo notebook with generated audio for baseline, mapped, and oracle conditions.
        """
    )
    return


if __name__ == "__main__":
    app.run()
