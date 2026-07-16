import marimo

__generated_with = "0.17.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import math
    import struct
    import wave
    from pathlib import Path

    import marimo as mo

    output_dir = Path("/localdisk/bowen/singing_identity/notebook_audio_previews")
    output_dir.mkdir(parents=True, exist_ok=True)
    return math, mo, output_dir, struct, wave


@app.cell
def _(math, output_dir, struct, wave):
    sample_rate = 16000
    duration_sec = 0.5
    wav_path = output_dir / "preview_sine_440hz.wav"

    if not wav_path.exists():
        num_samples = int(sample_rate * duration_sec)
        with wave.open(str(wav_path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(sample_rate)
            for idx in range(num_samples):
                value = int(0.15 * 32767 * math.sin(2.0 * math.pi * 440.0 * idx / sample_rate))
                handle.writeframes(struct.pack("<h", value))

    return sample_rate, wav_path


@app.cell
def _(mo, sample_rate, wav_path):
    mo.audio(str(wav_path), rate=sample_rate)
    return


if __name__ == "__main__":
    app.run()
