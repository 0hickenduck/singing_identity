import marimo

__generated_with = "0.17.6"
app = marimo.App(width="full")


@app.cell
def _():
    import json
    from collections import defaultdict
    from pathlib import Path

    import marimo as mo

    RUN_ROOT = Path("/localdisk/bowen/singing_identity/runs/track1_seedvc_component_ablation_12pairs")
    LISTENING_MANIFEST = RUN_ROOT / "listening_manifest.jsonl"
    SUMMARY_PATH = RUN_ROOT / "resemblyzer_multicondition_summary.json"
    return LISTENING_MANIFEST, RUN_ROOT, SUMMARY_PATH, defaultdict, json, mo


@app.cell
def _(LISTENING_MANIFEST, SUMMARY_PATH, defaultdict, json):
    def read_jsonl(path):
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    rows = read_jsonl(LISTENING_MANIFEST)
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    deltas = {
        f"{row['target_pair_id']}::{row['condition']}": row
        for row in summary["pair_deltas"]
    }
    grouped = defaultdict(list)
    for _row in rows:
        grouped[_row["target_pair_id"]].append(_row)
    pairs = sorted(grouped.items())
    return deltas, pairs, summary


@app.cell
def _(RUN_ROOT, mo, summary):
    mo.md(f"""
    # Track 1 Seed-VC Component Ablation Review

    Run root:

    `{RUN_ROOT}`

    This is an oracle diagnostic, not a prediction experiment. It asks which Seed-VC target-prompt component makes the singing prompt sound different from the speech prompt.

    Components:

    - `prompt_seq`: Seed-VC length-regulated prompt condition from Whisper/semantic features plus F0.
    - `mel`: target prompt mel context passed to the diffusion model.
    - `style`: CAMPPlus speaker/style embedding.

    Automatic Resemblyzer triage, relative to `baseline_all_speech`:

    - `oracle_all_singing`: target-singing delta `{summary["condition_summary"]["oracle_all_singing"]["mean_delta_to_target_singing"]:.4f}`
    - `singing_mel_only`: target-singing delta `{summary["condition_summary"]["singing_mel_only"]["mean_delta_to_target_singing"]:.4f}`
    - `singing_style_only`: target-singing delta `{summary["condition_summary"]["singing_style_only"]["mean_delta_to_target_singing"]:.4f}`
    - `singing_prompt_seq_only`: target-singing delta `{summary["condition_summary"]["singing_prompt_seq_only"]["mean_delta_to_target_singing"]:.4f}`
    - `singing_mel_style`: target-singing delta `{summary["condition_summary"]["singing_mel_style"]["mean_delta_to_target_singing"]:.4f}`

    Important: these automatic deltas can reflect identity, singing-domain/phonation, loudness, or quality. Human listening is the deciding evidence.
    """)
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
    baseline = by_condition["baseline_all_speech"]
    oracle = by_condition["oracle_all_singing"]
    prompt_seq = by_condition["singing_prompt_seq_only"]
    mel = by_condition["singing_mel_only"]
    style = by_condition["singing_style_only"]
    combo_rows = {
        key: by_condition[key]
        for key in (
            "singing_prompt_seq_mel",
            "singing_prompt_seq_style",
            "singing_mel_style",
        )
    }
    mo.md(
        f"""
        ## Pair {pair_index.value}: `{baseline["target_speaker_id"]}` target from `{baseline["source_speaker_id"]}` source

        Pair id: `{pair_id}`

        Main question: which single swap moves baseline toward oracle most clearly?
        """
    )
    return baseline, combo_rows, mel, oracle, prompt_seq, style


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
def _(deltas, mo):
    def audio_card(title, row):
        delta = deltas.get(f"{row['target_pair_id']}::{row['condition']}", {})
        return mo.vstack(
            [
                mo.md(
                    f"""
                    **{title}**

                    - prompt_seq: `{row["prompt_seq_source"]}`
                    - mel: `{row["mel_source"]}`
                    - style: `{row["style_source"]}`
                    - d target singing: `{delta.get("delta_to_target_singing", float("nan")):.4f}`
                    - d target speech: `{delta.get("delta_to_target_speech", float("nan")):.4f}`
                    - d source: `{delta.get("delta_to_source", float("nan")):.4f}`
                    - d RMS dB: `{delta.get("delta_rms_db", float("nan")):.2f}`
                    - RMS dB: `{row["rms_db"]:.2f}`
                    """
                ),
                mo.audio(row["audio_wav"]),
            ]
        )
    return (audio_card,)


@app.cell
def _(audio_card, baseline, mel, mo, oracle, prompt_seq, style):
    mo.vstack(
        [
            mo.md("### Single Component Swaps"),
            mo.hstack(
                [
                    audio_card("Baseline all speech", baseline),
                    audio_card("Oracle all singing", oracle),
                ],
                justify="start",
            ),
            mo.hstack(
                [
                    audio_card("Singing prompt_seq only", prompt_seq),
                    audio_card("Singing mel only", mel),
                    audio_card("Singing style only", style),
                ],
                justify="start",
            ),
        ]
    )
    return


@app.cell
def _(mo):
    combo = mo.ui.dropdown(
        options={
            "prompt_seq + mel": "singing_prompt_seq_mel",
            "prompt_seq + style": "singing_prompt_seq_style",
            "mel + style": "singing_mel_style",
        },
        value="singing_prompt_seq_style",
        label="Combo condition",
    )
    combo
    return (combo,)


@app.cell
def _(audio_card, combo, combo_rows, mo):
    _row = combo_rows[combo.value]
    mo.vstack([mo.md("### Optional Two-Component Swap"), audio_card(combo.value, _row)])
    return


@app.cell
def _(mo):
    mo.vstack(
        [
            mo.md("### What To Report Back"),
            mo.md(
                "Please report which single component is most audible: `prompt_seq`, `mel`, `style`, or `none`; "
                "whether `singing_mel_style` gets close to `oracle_all_singing`; "
                "whether the change is identity, timbre, breathy/phonation, domain, loudness/quality, or artifact; "
                "and pair numbers where the answer is clear or unclear."
            ),
            mo.md(
                "This decision determines the next predictive experiment. If `style` dominates, we map/steer CAMPPlus style. "
                "If `mel` dominates, the residual is mostly prompt acoustic context. If `prompt_seq` dominates, the semantic/F0 path is still worth improving. "
                "If none dominates alone but a combo does, we build a combined adapter."
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
