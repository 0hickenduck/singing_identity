import marimo

__generated_with = "0.17.6"
app = marimo.App(width="full")


@app.cell
def _():
    import json
    from collections import defaultdict
    from pathlib import Path

    import marimo as mo

    RUN_ROOT = Path("/localdisk/bowen/singing_identity/runs/track1_seedvc_semantic_intervention_4pairs")
    LISTENING_MANIFEST = RUN_ROOT / "listening_manifest_with_audio_stats.jsonl"
    SUMMARY_PATH = RUN_ROOT / "audio_summary.json"
    return LISTENING_MANIFEST, RUN_ROOT, SUMMARY_PATH, defaultdict, json, mo


@app.cell
def _(LISTENING_MANIFEST, SUMMARY_PATH, defaultdict, json):
    def read_jsonl(path):
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    rows = read_jsonl(LISTENING_MANIFEST)
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["target_pair_id"]].append(row)
    pairs = sorted(grouped.items())
    return pairs, rows, summary


@app.cell
def _(RUN_ROOT, mo, summary):
    mo.md(
        f"""
        # Track 1 Seed-VC Semantic Residual Intervention Review

        Run root:

        `{RUN_ROOT}`

        This is the first causal audio check after the native latent audit. Each pair has:

        - `baseline_speech_prompt`: source singing converted with target speech prompt.
        - `mapped_semantic_stats`: same target speech prompt, but its Seed-VC semantic prompt sequence is affine-shifted toward predicted singing semantic mean/std.
        - `oracle_singing_prompt`: source singing converted with the actual target singing prompt.

        Audio sanity: `{summary["conditions"]}` wavs, `{summary["pairs"]}` pairs, mean duration `{summary["duration_sec_mean"]:.2f}` sec, mean RMS `{summary["rms_db_mean"]:.2f}` dB.
        """
    )
    return


@app.cell
def _(mo, pairs):
    pair_index = mo.ui.slider(start=1, stop=len(pairs), step=1, value=1, label="Pair")
    pair_index
    return (pair_index,)


@app.cell
def _(mo, pair_index, pairs):
    pair_id, group = pairs[pair_index.value - 1]
    by_condition = {row["condition"]: row for row in group}
    baseline = by_condition["baseline_speech_prompt"]
    mapped = by_condition["mapped_semantic_stats"]
    oracle = by_condition["oracle_singing_prompt"]
    mo.md(
        f"""
        ## Pair {pair_index.value}: `{baseline["target_speaker_id"]}` target from `{baseline["source_speaker_id"]}` source

        Pair id: `{pair_id}`

        Listen for whether `mapped_semantic_stats` moves from baseline toward oracle in singing identity/domain, without simply becoming noisier or losing source melody/content.
        """
    )
    return baseline, mapped, oracle


@app.cell
def _(baseline, mo):
    mo.vstack(
        [
            mo.md("### References"),
            mo.hstack(
                [
                    mo.vstack([mo.md("Source singing"), mo.audio(baseline["source_wav"])]),
                    mo.vstack([mo.md("Target speech reference"), mo.audio(baseline["target_speech_wav"])]),
                    mo.vstack([mo.md("Target singing reference"), mo.audio(baseline["target_singing_wav"])]),
                ],
                justify="start",
            ),
        ]
    )
    return


@app.cell
def _(baseline, mapped, mo, oracle):
    def card(title, row):
        return mo.vstack(
            [
                mo.md(
                    f"""
                    **{title}**

                    - duration: `{row["duration_sec"]:.2f}` sec
                    - RMS dB: `{row["rms_db"]:.2f}`
                    - peak: `{row["peak"]:.3f}`
                    """
                ),
                mo.audio(row["audio_wav"]),
            ]
        )

    mo.vstack(
        [
            mo.md("### Generated Outputs"),
            mo.hstack(
                [
                    card("Baseline speech prompt", baseline),
                    card("Mapped semantic stats", mapped),
                    card("Oracle singing prompt", oracle),
                ],
                justify="start",
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
        ### What To Report Back

        Please report one of:

        - `semantic works`: mapped is audibly between baseline and oracle in the desired way.
        - `semantic weak`: mapped is valid audio but barely changes the result.
        - `semantic harmful`: mapped causes artifacts, identity loss, or worse conversion.

        Useful notes: pair numbers where mapped improves, pair numbers where mapped fails, and whether the difference sounds like identity, singing mode, loudness, breathiness, or F0/prosody.
        """
    )
    return


if __name__ == "__main__":
    app.run()
