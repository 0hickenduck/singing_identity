import marimo

__generated_with = "0.17.6"
app = marimo.App(width="full")


@app.cell
def _():
    import json
    from collections import defaultdict
    from pathlib import Path

    import marimo as mo

    RUN_ROOT = Path("/localdisk/bowen/singing_identity/runs/track1_seedvc_prompt_baseline_30pairs")
    LISTENING_MANIFEST = RUN_ROOT / "listening_manifest.jsonl"
    SCORES_PATH = RUN_ROOT / "resemblyzer_scores.jsonl"
    SUMMARY_PATH = RUN_ROOT / "resemblyzer_summary.json"
    return LISTENING_MANIFEST, SCORES_PATH, SUMMARY_PATH, defaultdict, json, mo


@app.cell
def _(LISTENING_MANIFEST, SCORES_PATH, SUMMARY_PATH, defaultdict, json):
    def read_jsonl(path):
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    rows = read_jsonl(LISTENING_MANIFEST)
    score_rows = read_jsonl(SCORES_PATH)
    scores = {
        f"{row['target_pair_id']}::{row['condition']}": row
        for row in score_rows
    }
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["target_pair_id"]].append(row)
    pairs = sorted(grouped.items())
    return pairs, scores, summary


@app.cell
def _(mo, summary):
    mo.md(f"""
    # Track 1 Seed-VC Prompt Baseline Review

    Human checkpoint: compare `target_speech_prompt` against `target_singing_prompt`.

    Run root:

    `{RUN_ROOT}`

    Automatic triage from Resemblyzer:

    - pairs: `{summary["pairs"]}`
    - generated conditions: `{summary["conditions"]}`
    - mean singing-prompt delta to target singing: `{summary["mean_singing_prompt_delta_to_target_singing"]:.4f}`
    - mean singing-prompt delta to target speech: `{summary["mean_singing_prompt_delta_to_target_speech"]:.4f}`
    - mean singing-prompt delta to source singing: `{summary["mean_singing_prompt_delta_to_source"]:.4f}`

    Listen row by row. The decision we need from you is whether the singing-prompt output is audibly closer to the target singer's singing voice while keeping the source melody/content usable.

    Important: these scores are a triage proxy. A positive target-singing delta may mean identity, but may also mean singing-domain, breathy/phonation, loudness, or audio quality.
    """)
    return


@app.cell
def _(mo, pairs):
    pair_index = mo.ui.slider(start=1, stop=len(pairs), step=1, value=1, label="Pair")
    pair_index
    return (pair_index,)


@app.cell
def _(mo, pair_index, pairs, scores):
    pair_id, group = pairs[pair_index.value - 1]
    by_condition = {row["condition"]: row for row in group}
    speech = by_condition["target_speech_prompt"]
    singing = by_condition["target_singing_prompt"]
    speech_score = scores[f"{pair_id}::target_speech_prompt"]
    singing_score = scores[f"{pair_id}::target_singing_prompt"]

    delta_target_singing = (
        singing_score["sim_to_target_singing_prompt"]
        - speech_score["sim_to_target_singing_prompt"]
    )
    delta_target_speech = (
        singing_score["sim_to_target_speech_prompt"]
        - speech_score["sim_to_target_speech_prompt"]
    )
    delta_source = (
        singing_score["sim_to_source_singing"]
        - speech_score["sim_to_source_singing"]
    )

    mo.md(
        f"""
        ## Pair {pair_index.value}: `{speech["target_speaker_id"]}` target from `{speech["source_speaker_id"]}` source

        Pair id: `{pair_id}`

        Metric deltas for singing prompt minus speech prompt:

        - target singing similarity: `{delta_target_singing:.4f}`
        - target speech similarity: `{delta_target_speech:.4f}`
        - source singing similarity: `{delta_source:.4f}`
        """
    )
    return singing, singing_score, speech, speech_score


@app.cell
def _(mo, singing, speech):
    mo.vstack(
        [
            mo.md("### References"),
            mo.hstack(
                [
                    mo.vstack([mo.md("Source singing"), mo.audio(speech["source_wav"])]),
                    mo.vstack([mo.md("Target speech reference"), mo.audio(speech["target_wav"])]),
                    mo.vstack([mo.md("Target singing reference"), mo.audio(singing["target_wav"])]),
                ],
                justify="start",
            ),
        ]
    )
    return


@app.cell
def _(mo, singing, singing_score, speech, speech_score):
    mo.vstack(
        [
            mo.md("### Generated Outputs"),
            mo.hstack(
                [
                    mo.vstack(
                        [
                            mo.md(
                                f"""
                                **Speech prompt output**

                                - sim target singing: `{speech_score["sim_to_target_singing_prompt"]:.4f}`
                                - sim target speech: `{speech_score["sim_to_target_speech_prompt"]:.4f}`
                                - sim source singing: `{speech_score["sim_to_source_singing"]:.4f}`
                                - RMS dB: `{speech["rms_db"]:.2f}`
                                """
                            ),
                            mo.audio(speech["audio_wav"]),
                        ]
                    ),
                    mo.vstack(
                        [
                            mo.md(
                                f"""
                                **Singing prompt output**

                                - sim target singing: `{singing_score["sim_to_target_singing_prompt"]:.4f}`
                                - sim target speech: `{singing_score["sim_to_target_speech_prompt"]:.4f}`
                                - sim source singing: `{singing_score["sim_to_source_singing"]:.4f}`
                                - RMS dB: `{singing["rms_db"]:.2f}`
                                """
                            ),
                            mo.audio(singing["audio_wav"]),
                        ]
                    ),
                ],
                justify="start",
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ### What To Report Back

    For this checkpoint, please tell Codex:

    - whether singing prompt is often closer to target identity: `yes`, `no`, or `mixed`
    - pair numbers where speech prompt wins
    - whether the main change sounds like `identity`, `breathy/phonation`, `singing-domain`, `loudness/quality`, or `artifact`

    Decision menu:

    - `continue`: singing-prompt or mel/style outputs are audibly better in identity often enough to justify deeper SeedVC integration.
    - `rerun`: audio quality, loudness, pair balance, or diffusion settings make this run inconclusive.
    - `stop`: prompt-mode effect is not audible as useful identity signal.
    """)
    return


if __name__ == "__main__":
    app.run()
