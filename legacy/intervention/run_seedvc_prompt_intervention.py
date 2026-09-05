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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def component_vector(npz_path: str, component: str) -> np.ndarray:
    with np.load(npz_path) as data:
        if component == "semantic_stats":
            return np.concatenate([np.asarray(data["semantic_mean"]), np.asarray(data["semantic_std"])]).astype(np.float64)
        if component == "prompt_stats":
            return np.concatenate([np.asarray(data["prompt_mean"]), np.asarray(data["prompt_std"])]).astype(np.float64)
    raise ExperimentError(f"Unsupported intervention component: {component}")


def ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    x_aug = np.column_stack([np.ones(x.shape[0]), x])
    ident = np.eye(x_aug.shape[1], dtype=np.float64)
    ident[0, 0] = 0.0
    return np.linalg.solve(x_aug.T @ x_aug + alpha * ident, x_aug.T @ y)


def ridge_predict(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(x.shape[0]), x]) @ weights


def train_residual(latent_manifest: Path, component: str, ridge_alpha: float) -> tuple[np.ndarray, list[str]]:
    rows = load_jsonl(latent_manifest)
    by_pair: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_pair.setdefault(str(row["pair_id"]), {})[str(row["condition"])] = row
    pairs = []
    for pair_id, conds in sorted(by_pair.items()):
        if "speech_prompt" not in conds or "singing_prompt" not in conds:
            continue
        speech = conds["speech_prompt"]
        singing = conds["singing_prompt"]
        x_speech = component_vector(str(speech["latent_npz"]), component)
        x_singing = component_vector(str(singing["latent_npz"]), component)
        pairs.append(
            {
                "pair_id": pair_id,
                "speaker_id": str(speech["target_speaker_id"]),
                "x_speech": x_speech,
                "delta": x_singing - x_speech,
            }
        )
    if len(pairs) < 4:
        raise ExperimentError(f"Need at least 4 latent pairs, found {len(pairs)}")
    speakers = sorted({row["speaker_id"] for row in pairs})
    heldout = set(speakers[-max(1, len(speakers) // 5):])
    train = [row for row in pairs if row["speaker_id"] not in heldout]
    if not train:
        raise ExperimentError("No train pairs available for residual")
    x_train = np.stack([row["x_speech"] for row in train])
    y_train = np.stack([row["delta"] for row in train])
    return ridge_fit(x_train, y_train, ridge_alpha), sorted(heldout)


def affine_match_sequence(sequence, predicted_stats, eps: float):
    import torch

    dim = sequence.size(-1)
    pred = torch.as_tensor(predicted_stats, dtype=sequence.dtype, device=sequence.device)
    target_mean = pred[:dim].view(1, 1, dim)
    target_std = torch.clamp(pred[dim:].view(1, 1, dim), min=eps)
    source_mean = sequence.mean(dim=1, keepdim=True)
    source_std = torch.clamp(sequence.std(dim=1, keepdim=True), min=eps)
    return (sequence - source_mean) / source_std * target_std + target_mean


def crossfade(chunk1: np.ndarray, chunk2: np.ndarray, overlap: int) -> np.ndarray:
    fade_out = np.cos(np.linspace(0, np.pi / 2, overlap)) ** 2
    fade_in = np.cos(np.linspace(np.pi / 2, 0, overlap)) ** 2
    if len(chunk2) < overlap:
        chunk2[:overlap] = chunk2[:overlap] * fade_in[: len(chunk2)] + (chunk1[-overlap:] * fade_out)[: len(chunk2)]
    else:
        chunk2[:overlap] = chunk2[:overlap] * fade_in + chunk1[-overlap:] * fade_out
    return chunk2


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Seed-VC three-condition prompt intervention audio synthesis.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--latent-manifest", type=Path, required=True)
    parser.add_argument("--seedvc-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--manifest-out", type=Path, required=True)
    parser.add_argument("--metrics-out", type=Path, required=True)
    parser.add_argument("--max-pairs", type=int, default=4)
    parser.add_argument("--pair-split", choices=["test", "train", "all"], default="test")
    parser.add_argument("--component", choices=["semantic_stats", "prompt_stats"], default="semantic_stats")
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--diffusion-steps", type=int, default=10)
    parser.add_argument("--inference-cfg-rate", type=float, default=0.7)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--run", action="store_true")
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
        weights, residual_heldout = train_residual(args.latent_manifest, args.component, args.ridge_alpha)

        rows = []
        if args.run:
            cwd = Path.cwd()
            sys.path.insert(0, str(args.seedvc_root))
            os.chdir(args.seedvc_root)
            try:
                import librosa
                import torch
                import torchaudio
                import inference as seedvc_inference

                os.environ.setdefault("TMPDIR", "/tmp")
                os.environ.setdefault("HF_HOME", str(args.out_root / "hf_home"))
                os.environ.setdefault("HF_HUB_CACHE", str(args.out_root / "hf_home" / "hub"))

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
                def synthesize(source_wav: Path, prompt_wav: Path, output_wav: Path, intervention: bool) -> dict[str, Any]:
                    source_audio = librosa.load(str(source_wav), sr=sr)[0]
                    ref_audio = librosa.load(str(prompt_wav), sr=sr)[0]
                    source_tensor = torch.tensor(source_audio).unsqueeze(0).float().to(device)
                    ref_tensor = torch.tensor(ref_audio[: sr * 25]).unsqueeze(0).float().to(device)
                    converted_16k = torchaudio.functional.resample(source_tensor, sr, 16000)
                    ref_16k = torchaudio.functional.resample(ref_tensor, sr, 16000)
                    s_alt = semantic_long(converted_16k)
                    s_ori = semantic_fn(ref_16k)
                    mel = mel_fn(source_tensor.float())
                    mel2 = mel_fn(ref_tensor.float())
                    target_lengths = torch.LongTensor([int(mel.size(2))]).to(mel.device)
                    target2_lengths = torch.LongTensor([mel2.size(2)]).to(mel2.device)
                    feat2 = torchaudio.compliance.kaldi.fbank(ref_16k, num_mel_bins=80, dither=0, sample_frequency=16000)
                    feat2 = feat2 - feat2.mean(dim=0, keepdim=True)
                    style2 = campplus_model(feat2.unsqueeze(0))
                    f0_ori = torch.from_numpy(f0_fn(ref_16k[0], thred=0.03)).to(device)[None]
                    f0_alt = torch.from_numpy(f0_fn(converted_16k[0], thred=0.03)).to(device)[None]

                    if intervention and args.component == "semantic_stats":
                        s_np = s_ori.squeeze(0).detach().cpu().numpy()
                        x = np.concatenate([s_np.mean(axis=0), s_np.std(axis=0)]).astype(np.float64)
                        predicted = x + ridge_predict(x[None, :], weights)[0]
                        s_ori = affine_match_sequence(s_ori, predicted, eps=1.0e-5)

                    cond, *_ = model.length_regulator(s_alt, ylens=target_lengths, n_quantizers=3, f0=f0_alt)
                    prompt_condition, *_ = model.length_regulator(s_ori, ylens=target2_lengths, n_quantizers=3, f0=f0_ori)

                    if intervention and args.component == "prompt_stats":
                        p_np = prompt_condition.squeeze(0).detach().cpu().numpy()
                        x = np.concatenate([p_np.mean(axis=0), p_np.std(axis=0)]).astype(np.float64)
                        predicted = x + ridge_predict(x[None, :], weights)[0]
                        prompt_condition = affine_match_sequence(prompt_condition, predicted, eps=1.0e-5)

                    max_source_window = max_context_window - mel2.size(2)
                    if max_source_window <= overlap_frame_len:
                        raise ExperimentError(f"Prompt is too long for context window: {prompt_wav}")
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
                                style2,
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
                    vc_wave = torch.tensor(np.concatenate(generated_wave_chunks))[None, :].float()
                    output_wav.parent.mkdir(parents=True, exist_ok=True)
                    torchaudio.save(str(output_wav), vc_wave.cpu(), sr)
                    return {
                        "sample_rate": sr,
                        "duration_sec": float(vc_wave.size(-1) / sr),
                        "rms": float(torch.sqrt(torch.mean(vc_wave.square())).item()),
                    }

                for idx, target_pair in enumerate(selected_pairs, 1):
                    source = find_source_singing(target_pair, pairs, utterance_by_id)
                    target_speech = utterance_by_id[str(target_pair["speech_utt_id"])]
                    target_singing = utterance_by_id[str(target_pair["singing_utt_id"])]
                    conditions = [
                        ("baseline_speech_prompt", target_speech, False),
                        (f"mapped_{args.component}", target_speech, True),
                        ("oracle_singing_prompt", target_singing, False),
                    ]
                    for condition, prompt, intervention in conditions:
                        output_wav = args.out_root / "audio" / str(target_pair["pair_id"]) / f"{condition}.wav"
                        stats = synthesize(Path(str(source["wav_path"])), Path(str(prompt["wav_path"])), output_wav, intervention)
                        rows.append(
                            {
                                "stage": "seedvc_prompt_intervention",
                                "condition": condition,
                                "component": args.component,
                                "target_pair_id": target_pair["pair_id"],
                                "target_speaker_id": target_pair["speaker_id"],
                                "source_speaker_id": source["speaker_id"],
                                "source_wav": source["wav_path"],
                                "target_speech_wav": target_speech["wav_path"],
                                "target_singing_wav": target_singing["wav_path"],
                                "prompt_wav": prompt["wav_path"],
                                "audio_wav": str(output_wav),
                                **stats,
                            }
                        )
                    print(f"synthesized {idx}/{len(selected_pairs)} pairs")
            finally:
                os.chdir(cwd)
        else:
            for target_pair in selected_pairs:
                source = find_source_singing(target_pair, pairs, utterance_by_id)
                target_speech = utterance_by_id[str(target_pair["speech_utt_id"])]
                target_singing = utterance_by_id[str(target_pair["singing_utt_id"])]
                for condition, prompt in (
                    ("baseline_speech_prompt", target_speech),
                    (f"mapped_{args.component}", target_speech),
                    ("oracle_singing_prompt", target_singing),
                ):
                    rows.append(
                        {
                            "stage": "seedvc_prompt_intervention",
                            "condition": condition,
                            "component": args.component,
                            "target_pair_id": target_pair["pair_id"],
                            "target_speaker_id": target_pair["speaker_id"],
                            "source_speaker_id": source["speaker_id"],
                            "source_wav": source["wav_path"],
                            "target_speech_wav": target_speech["wav_path"],
                            "target_singing_wav": target_singing["wav_path"],
                            "prompt_wav": prompt["wav_path"],
                            "audio_wav": str(args.out_root / "audio" / str(target_pair["pair_id"]) / f"{condition}.wav"),
                        }
                    )

        metrics = {
            "stage": "seedvc_prompt_intervention",
            "component": args.component,
            "latent_manifest": str(args.latent_manifest),
            "ridge_alpha": args.ridge_alpha,
            "residual_heldout_speakers": residual_heldout,
            "pair_split": args.pair_split,
            "pairs": len(selected_pairs),
            "conditions": len(rows),
            "run": args.run,
            "diffusion_steps": args.diffusion_steps,
            "inference_cfg_rate": args.inference_cfg_rate,
            "fp16": args.fp16,
        }
    except (ExperimentError, OSError, RuntimeError, ImportError, np.linalg.LinAlgError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    write_table(rows, args.manifest_out)
    write_json(metrics, args.metrics_out)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
