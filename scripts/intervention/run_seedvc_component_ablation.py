#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research_utils import ExperimentError, read_table, validate_pairs, validate_utterances, write_json, write_table  # noqa: E402


def split_pairs_like_mapper(pairs: list[dict[str, Any]], pair_split: str) -> list[dict[str, Any]]:
    if pair_split == "all":
        return pairs
    by_speaker: dict[str, list[dict[str, Any]]] = {}
    for row in pairs:
        by_speaker.setdefault(str(row["speaker_id"]), []).append(row)
    speakers = sorted(by_speaker)
    heldout = set(speakers[-max(1, len(speakers) // 5):])
    return [
        row
        for row in pairs
        if (str(row["speaker_id"]) in heldout) == (pair_split == "test")
    ]


def select_pairs_round_robin(pairs: list[dict[str, Any]], max_pairs: int) -> list[dict[str, Any]]:
    if max_pairs <= 0:
        return pairs
    by_speaker: dict[str, list[dict[str, Any]]] = {}
    for row in pairs:
        by_speaker.setdefault(str(row["speaker_id"]), []).append(row)
    selected: list[dict[str, Any]] = []
    speakers = sorted(by_speaker)
    cursor = 0
    while len(selected) < max_pairs:
        added = False
        for speaker in speakers:
            rows = by_speaker[speaker]
            if cursor < len(rows):
                selected.append(rows[cursor])
                added = True
                if len(selected) >= max_pairs:
                    break
        if not added:
            break
        cursor += 1
    return selected


def find_source_singing(target_pair: dict[str, Any], pairs: list[dict[str, Any]], utterance_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    target_speaker = str(target_pair["speaker_id"])
    target_singing = utterance_by_id[str(target_pair["singing_utt_id"])]
    candidate_pairs = [
        row
        for row in pairs
        if str(row["speaker_id"]) != target_speaker
        and str(row.get("language", "")) == str(target_pair.get("language", ""))
    ]
    if not candidate_pairs:
        candidate_pairs = [row for row in pairs if str(row["speaker_id"]) != target_speaker]
    if not candidate_pairs:
        raise ExperimentError(f"No different-speaker source candidates for {target_pair['pair_id']}")

    def score(row: dict[str, Any]) -> tuple[int, str]:
        utt = utterance_by_id[str(row["singing_utt_id"])]
        matches = 0
        for key in ("language", "song_id", "take_id", "technique", "text", "phone_seq"):
            if str(utt.get(key, "")) and str(utt.get(key, "")) == str(target_singing.get(key, "")):
                matches += 1
        return (-matches, str(row["pair_id"]))

    best = sorted(candidate_pairs, key=score)[0]
    return utterance_by_id[str(best["singing_utt_id"])]


def crossfade(chunk1: np.ndarray, chunk2: np.ndarray, overlap: int) -> np.ndarray:
    fade_out = np.cos(np.linspace(0, np.pi / 2, overlap)) ** 2
    fade_in = np.cos(np.linspace(np.pi / 2, 0, overlap)) ** 2
    if len(chunk2) < overlap:
        chunk2[:overlap] = chunk2[:overlap] * fade_in[: len(chunk2)] + (chunk1[-overlap:] * fade_out)[: len(chunk2)]
    else:
        chunk2[:overlap] = chunk2[:overlap] * fade_in + chunk1[-overlap:] * fade_out
    return chunk2


def condition_grid(include_combos: bool) -> list[dict[str, str]]:
    base = [
        {"condition": "baseline_all_speech", "prompt_seq": "speech", "mel": "speech", "style": "speech"},
        {"condition": "oracle_all_singing", "prompt_seq": "singing", "mel": "singing", "style": "singing"},
        {"condition": "singing_prompt_seq_only", "prompt_seq": "singing", "mel": "speech", "style": "speech"},
        {"condition": "singing_mel_only", "prompt_seq": "speech", "mel": "singing", "style": "speech"},
        {"condition": "singing_style_only", "prompt_seq": "speech", "mel": "speech", "style": "singing"},
    ]
    if not include_combos:
        return base
    return base + [
        {"condition": "singing_prompt_seq_mel", "prompt_seq": "singing", "mel": "singing", "style": "speech"},
        {"condition": "singing_prompt_seq_style", "prompt_seq": "singing", "mel": "speech", "style": "singing"},
        {"condition": "singing_mel_style", "prompt_seq": "speech", "mel": "singing", "style": "singing"},
    ]


def audio_stats_tensor(wave, sr: int) -> dict[str, float | int]:
    import torch

    rms = float(torch.sqrt(torch.mean(wave.square())).item())
    return {
        "sample_rate": sr,
        "num_samples": int(wave.size(-1)),
        "duration_sec": float(wave.size(-1) / sr),
        "rms": rms,
        "rms_db": float(20.0 * np.log10(max(rms, 1.0e-12))),
        "peak": float(torch.max(torch.abs(wave)).item()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Oracle ablation over Seed-VC prompt_condition, prompt mel, and style conditioning.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--seedvc-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--manifest-out", type=Path, required=True)
    parser.add_argument("--metrics-out", type=Path, required=True)
    parser.add_argument("--max-pairs", type=int, default=4)
    parser.add_argument("--pair-split", choices=["test", "train", "all"], default="test")
    parser.add_argument("--diffusion-steps", type=int, default=10)
    parser.add_argument("--inference-cfg-rate", type=float, default=0.7)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--include-combos", action="store_true")
    args = parser.parse_args()

    try:
        if not (args.seedvc_root / "inference.py").exists():
            raise ExperimentError(f"Seed-VC inference.py not found in {args.seedvc_root}")
        utterances = read_table(args.manifest)
        validate_utterances(utterances, require_audio_exists=True)
        pairs = read_table(args.pairs)
        validate_pairs(pairs, utterances)
        utterance_by_id = {str(row["utt_id"]): row for row in utterances}
        selected_pairs = select_pairs_round_robin(split_pairs_like_mapper(pairs, args.pair_split), args.max_pairs)
        if not selected_pairs:
            raise ExperimentError(f"No pairs selected for pair_split={args.pair_split}")

        cwd = Path.cwd()
        sys.path.insert(0, str(args.seedvc_root))
        os.chdir(args.seedvc_root)
        try:
            import librosa
            import torch
            import torchaudio
            import inference as seedvc_inference

            os.environ.setdefault("TMPDIR", "/tmp")
            model_args = argparse.Namespace(f0_condition=True, checkpoint=None, config=None, fp16=args.fp16)
            model, semantic_fn, f0_fn, vocoder_fn, campplus_model, mel_fn, mel_fn_args = seedvc_inference.load_models(model_args)
            device = seedvc_inference.device
            sr = int(mel_fn_args["sampling_rate"])
            hop_length = 512
            max_context_window = sr // hop_length * 30
            overlap_frame_len = 16
            overlap_wave_len = overlap_frame_len * hop_length

            def semantic_long(audio_16k):
                if audio_16k.size(-1) <= 16000 * 30:
                    return semantic_fn(audio_16k)
                chunks = []
                buffer = None
                traversed = 0
                overlapping_time = 5
                while traversed < audio_16k.size(-1):
                    if buffer is None:
                        chunk = audio_16k[:, traversed : traversed + 16000 * 30]
                    else:
                        chunk = torch.cat(
                            [buffer, audio_16k[:, traversed : traversed + 16000 * (30 - overlapping_time)]],
                            dim=-1,
                        )
                    encoded = semantic_fn(chunk)
                    chunks.append(encoded if traversed == 0 else encoded[:, 50 * overlapping_time :])
                    buffer = chunk[:, -16000 * overlapping_time :]
                    traversed += 30 * 16000 if traversed == 0 else chunk.size(-1) - 16000 * overlapping_time
                return torch.cat(chunks, dim=1)

            @torch.no_grad()
            def prompt_features(wav_path: Path) -> dict[str, Any]:
                audio = librosa.load(str(wav_path), sr=sr)[0]
                tensor = torch.tensor(audio[: sr * 25]).unsqueeze(0).float().to(device)
                wav_16k = torchaudio.functional.resample(tensor, sr, 16000)
                semantic = semantic_fn(wav_16k)
                mel = mel_fn(tensor.float())
                f0 = torch.from_numpy(f0_fn(wav_16k[0], thred=0.03)).to(device)[None]
                feat = torchaudio.compliance.kaldi.fbank(wav_16k, num_mel_bins=80, dither=0, sample_frequency=16000)
                feat = feat - feat.mean(dim=0, keepdim=True)
                style = campplus_model(feat.unsqueeze(0))
                return {"semantic": semantic, "mel": mel, "f0": f0, "style": style, "wav_path": str(wav_path)}

            @torch.no_grad()
            def source_features(wav_path: Path):
                audio = librosa.load(str(wav_path), sr=sr)[0]
                tensor = torch.tensor(audio).unsqueeze(0).float().to(device)
                wav_16k = torchaudio.functional.resample(tensor, sr, 16000)
                semantic = semantic_long(wav_16k)
                mel = mel_fn(tensor.float())
                f0 = torch.from_numpy(f0_fn(wav_16k[0], thred=0.03)).to(device)[None]
                target_lengths = torch.LongTensor([mel.size(2)]).to(mel.device)
                cond, *_ = model.length_regulator(semantic, ylens=target_lengths, n_quantizers=3, f0=f0)
                return cond

            @torch.no_grad()
            def synthesize(cond, prompt_seq_source, mel_source, style_source, output_wav: Path) -> dict[str, float | int]:
                mel2 = mel_source["mel"]
                target2_lengths = torch.LongTensor([mel2.size(2)]).to(mel2.device)
                prompt_condition, *_ = model.length_regulator(
                    prompt_seq_source["semantic"],
                    ylens=target2_lengths,
                    n_quantizers=3,
                    f0=prompt_seq_source["f0"],
                )
                max_source_window = max_context_window - mel2.size(2)
                if max_source_window <= overlap_frame_len:
                    raise ExperimentError(f"Prompt is too long for context window: {mel_source['wav_path']}")
                processed_frames = 0
                generated_wave_chunks = []
                previous_chunk = None
                while processed_frames < cond.size(1):
                    chunk_cond = cond[:, processed_frames : processed_frames + max_source_window]
                    is_last_chunk = processed_frames + max_source_window >= cond.size(1)
                    cat_condition = torch.cat([prompt_condition, chunk_cond], dim=1)
                    with torch.autocast(device_type=device.type, dtype=torch.float16 if args.fp16 else torch.float32):
                        vc_target = model.cfm.inference(
                            cat_condition,
                            torch.LongTensor([cat_condition.size(1)]).to(mel2.device),
                            mel2,
                            style_source["style"],
                            None,
                            args.diffusion_steps,
                            inference_cfg_rate=args.inference_cfg_rate,
                        )
                        vc_target = vc_target[:, :, mel2.size(-1) :]
                    vc_wave = vocoder_fn(vc_target.float()).squeeze()[None, :]
                    if processed_frames == 0:
                        if is_last_chunk:
                            generated_wave_chunks.append(vc_wave[0].cpu().numpy())
                            break
                        generated_wave_chunks.append(vc_wave[0, :-overlap_wave_len].cpu().numpy())
                        previous_chunk = vc_wave[0, -overlap_wave_len:]
                    elif is_last_chunk:
                        generated_wave_chunks.append(crossfade(previous_chunk.cpu().numpy(), vc_wave[0].cpu().numpy(), overlap_wave_len))
                        break
                    else:
                        generated_wave_chunks.append(
                            crossfade(previous_chunk.cpu().numpy(), vc_wave[0, :-overlap_wave_len].cpu().numpy(), overlap_wave_len)
                        )
                        previous_chunk = vc_wave[0, -overlap_wave_len:]
                    processed_frames += vc_target.size(2) - overlap_frame_len
                wave = torch.tensor(np.concatenate(generated_wave_chunks))[None, :].float()
                output_wav.parent.mkdir(parents=True, exist_ok=True)
                torchaudio.save(str(output_wav), wave.cpu(), sr)
                return audio_stats_tensor(wave, sr)

            rows = []
            grid = condition_grid(args.include_combos)
            for idx, target_pair in enumerate(selected_pairs, 1):
                source = find_source_singing(target_pair, pairs, utterance_by_id)
                target_speech = utterance_by_id[str(target_pair["speech_utt_id"])]
                target_singing = utterance_by_id[str(target_pair["singing_utt_id"])]
                cond = source_features(Path(str(source["wav_path"])))
                prompt_by_mode = {
                    "speech": prompt_features(Path(str(target_speech["wav_path"]))),
                    "singing": prompt_features(Path(str(target_singing["wav_path"]))),
                }
                for spec in grid:
                    output_wav = args.out_root / "audio" / str(target_pair["pair_id"]) / f"{spec['condition']}.wav"
                    stats = synthesize(
                        cond,
                        prompt_by_mode[spec["prompt_seq"]],
                        prompt_by_mode[spec["mel"]],
                        prompt_by_mode[spec["style"]],
                        output_wav,
                    )
                    rows.append(
                        {
                            "stage": "seedvc_component_ablation",
                            "condition": spec["condition"],
                            "target_pair_id": target_pair["pair_id"],
                            "target_speaker_id": target_pair["speaker_id"],
                            "source_speaker_id": source["speaker_id"],
                            "source_wav": source["wav_path"],
                            "target_speech_wav": target_speech["wav_path"],
                            "target_singing_wav": target_singing["wav_path"],
                            "prompt_seq_source": spec["prompt_seq"],
                            "mel_source": spec["mel"],
                            "style_source": spec["style"],
                            "audio_wav": str(output_wav),
                            **stats,
                        }
                    )
                print(f"synthesized {idx}/{len(selected_pairs)} pairs")
        finally:
            os.chdir(cwd)

        metrics = {
            "stage": "seedvc_component_ablation",
            "pair_split": args.pair_split,
            "pairs": len(selected_pairs),
            "conditions_per_pair": len(condition_grid(args.include_combos)),
            "conditions": len(rows),
            "include_combos": args.include_combos,
            "diffusion_steps": args.diffusion_steps,
            "inference_cfg_rate": args.inference_cfg_rate,
            "fp16": args.fp16,
            "out_root": str(args.out_root),
        }
    except (ExperimentError, OSError, RuntimeError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    write_table(rows, args.manifest_out)
    write_json(metrics, args.metrics_out)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
