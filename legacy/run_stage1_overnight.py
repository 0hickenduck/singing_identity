#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import signal
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from research_utils import read_table, write_json, write_table  # noqa: E402


DEFAULT_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WORK_ROOT = Path("/localdisk/bowen/singing_identity")
DEFAULT_CACHE_ROOT = Path("/localdisk/bowen/.cache/singing_identity")
DEFAULT_PYTHON = str(DEFAULT_REPO_ROOT / ".venv" / "bin" / "python")
DEFAULT_GTSINGER = str(DEFAULT_WORK_ROOT / "data" / "gtsinger_domain_eval")
HOME_PREFIX = Path("/home/bowen")


@dataclass
class ModelSpec:
    key: str
    extractor: str
    checkpoint_hash: str
    layers: list[str]
    kind: str
    model_name: str | None = None
    trust_remote_code: bool = False


MODEL_SPECS = {
    "acoustic": ModelSpec(
        key="acoustic",
        extractor="acoustic_baseline",
        checkpoint_hash="local_wave_v1",
        layers=["frame25ms_hop20ms"],
        kind="acoustic",
    ),
    "wavlm": ModelSpec(
        key="wavlm",
        extractor="wavlm_base_plus",
        checkpoint_hash="microsoft_wavlm_base_plus",
        layers=["3", "6", "9", "12"],
        kind="wavlm",
        model_name="microsoft/wavlm-base-plus",
    ),
    "hubert": ModelSpec(
        key="hubert",
        extractor="hubert_base",
        checkpoint_hash="facebook_hubert_base_ls960",
        layers=["3", "6", "9", "12"],
        kind="hf",
        model_name="facebook/hubert-base-ls960",
    ),
    "mert": ModelSpec(
        key="mert",
        extractor="mert_v1_95m",
        checkpoint_hash="m_a_p_mert_v1_95m",
        layers=["3", "6", "9", "12"],
        kind="hf",
        model_name="m-a-p/MERT-v1-95M",
        trust_remote_code=True,
    ),
}


class Stage1Runner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.repo = Path.cwd()
        self.run_root = args.run_root.resolve()
        self.jobs_root = self.run_root / "jobs"
        self.report_path = args.report.resolve()
        self.feature_root = args.feature_root.resolve()
        self.model_specs = self.resolve_models(args.models)
        self.manifest = self.run_root / "manifests" / "gtsinger_utterances.jsonl"
        self.pairs = self.run_root / "manifests" / "gtsinger_pairs.jsonl"
        self.phone_examples = self.run_root / "manifests" / "gtsinger_phone_examples.jsonl"
        self.technique_pairs = self.run_root / "manifests" / "gtsinger_phoneme_pairs.jsonl"
        self.manifest_report = self.run_root / "reports" / "manifest_report.json"
        self.disabled_models: set[str] = set()

    def resolve_models(self, model_arg: str) -> list[ModelSpec]:
        specs = []
        for key in [item.strip() for item in model_arg.split(",") if item.strip()]:
            if key == "contentvec":
                if not self.args.contentvec_model:
                    continue
                specs.append(
                    ModelSpec(
                        key="contentvec",
                        extractor="contentvec",
                        checkpoint_hash=self.args.contentvec_model.replace("/", "_").replace("-", "_"),
                        layers=[self.args.contentvec_layer],
                        kind="hf",
                        model_name=self.args.contentvec_model,
                    )
                )
                continue
            if key in MODEL_SPECS:
                specs.append(MODEL_SPECS[key])
        return specs

    def run(self) -> int:
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.jobs_root.mkdir(parents=True, exist_ok=True)
        write_json(self.resolved_config(), self.run_root / "resolved_config.json")
        failures = []
        try:
            self.preflight_paths()
            self.prepare_data()
            rows = read_table(self.manifest)
            if not rows:
                raise RuntimeError("no utterances available after data preparation")
            self.write_smoke_manifest(rows[: min(10, len(rows))])
            self.smoke_checks()
            if not self.args.smoke_only:
                self.full_feature_extraction()
                self.track1_probes()
                self.track2_probes()
        except Exception as exc:  # noqa: BLE001
            failures.append(str(exc))
            write_json(
                {"status": "failed", "error": str(exc), "traceback": traceback.format_exc()},
                self.run_root / "status.json",
            )
        finally:
            self.generate_report(extra_failures=failures)
        return 1 if failures else 0

    def preflight_paths(self) -> None:
        if self.args.synthetic or self.args.smoke_only or self.args.allow_home_heavy_io:
            return
        heavy_paths = {
            "run_root": self.run_root,
            "feature_root": self.feature_root,
            "local_cache_root": self.args.local_cache_root.resolve(),
        }
        if not self.args.existing_manifest and not self.prepared_manifests_available():
            heavy_paths["gtsinger_root"] = self.args.gtsinger_root.resolve()
        bad = [
            f"{name}={path}"
            for name, path in heavy_paths.items()
            if path.is_relative_to(HOME_PREFIX)
        ]
        if bad:
            raise RuntimeError(
                "Refusing real Stage 1 heavy I/O under /home/bowen. "
                "Use local scratch such as /localdisk/bowen for run_root, feature_root, and local_cache_root, "
                "or pass --allow-home-heavy-io only for a deliberate small/debug run. "
                f"Blocked paths: {', '.join(bad)}"
            )

    def prepared_manifests_available(self) -> bool:
        status_path = self.jobs_root / "prepare_gtsinger_manifests" / "status.json"
        if not status_path.exists():
            return False
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return False
        return (
            status.get("status") == "success"
            and self.manifest.exists()
            and self.pairs.exists()
            and self.technique_pairs.exists()
            and self.manifest_report.exists()
        )

    def resolved_config(self) -> dict[str, Any]:
        return {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "hostname": os.uname().nodename,
            "python": self.args.python,
            "device": self.args.device,
            "run_root": str(self.run_root),
            "feature_root": str(self.feature_root),
            "report": str(self.report_path),
            "models": [spec.__dict__ for spec in self.model_specs],
            "gtsinger_root": str(self.args.gtsinger_root),
            "smoke_only": self.args.smoke_only,
            "synthetic": self.args.synthetic,
            "folds": self.args.folds,
            "max_rows_per_singer": self.args.max_rows_per_singer,
            "smoke_max_rows_per_singer": self.args.smoke_max_rows_per_singer,
            "max_phone_pairs_per_group": self.args.max_phone_pairs_per_group,
            "fast_no_wav_stats": getattr(self.args, "fast_no_wav_stats", False),
            "job_timeout_sec": self.args.job_timeout_sec,
            "smoke_job_timeout_sec": self.args.smoke_job_timeout_sec,
            "extract_job_timeout_sec": self.args.extract_job_timeout_sec,
            "probe_job_timeout_sec": self.args.probe_job_timeout_sec,
            "local_cache_root": str(self.args.local_cache_root),
        }

    def prepare_data(self) -> None:
        if self.args.existing_manifest:
            self.manifest.parent.mkdir(parents=True, exist_ok=True)
            self.manifest_report.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self.args.existing_manifest, self.manifest)
            if self.args.existing_pairs:
                shutil.copyfile(self.args.existing_pairs, self.pairs)
            if self.args.existing_phone_examples:
                shutil.copyfile(self.args.existing_phone_examples, self.phone_examples)
            if self.args.existing_technique_pairs:
                shutil.copyfile(self.args.existing_technique_pairs, self.technique_pairs)
            if self.args.existing_manifest_report:
                shutil.copyfile(self.args.existing_manifest_report, self.manifest_report)
            else:
                write_json({"reused_manifest": str(self.args.existing_manifest)}, self.manifest_report)
            self.run_job(
                "prepare_reuse_existing_manifests",
                [self.args.python, "-c", "print('reused existing manifests')"],
                [self.manifest, self.pairs, self.technique_pairs, self.manifest_report],
                optional=False,
                timeout_sec=60,
            )
            return
        if self.args.synthetic:
            self.run_job(
                "prepare_synthetic_data",
                [
                    self.args.python,
                    "legacy/data_prep/make_synthetic_experiment.py",
                    "--root",
                    str(self.run_root / "synthetic"),
                    "--speakers",
                    "12",
                    "--items-per-mode",
                    "3",
                ],
                [self.run_root / "synthetic" / "synthetic_run_config.json"],
            )
            config = json.loads((self.run_root / "synthetic" / "synthetic_run_config.json").read_text())
            self.manifest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(config["track1_manifest"], self.manifest)
            shutil.copyfile(config["track1_pairs"], self.pairs)
            shutil.copyfile(config["track2_pairs"], self.technique_pairs)
            self.feature_root = Path(config["feature_root"]).resolve()
            self.model_specs = [
                ModelSpec(
                    key="synthetic",
                    extractor=config["extractor"],
                    checkpoint_hash=config["checkpoint_hash"],
                    layers=[config["layer"]],
                    kind="precomputed",
                )
            ]
            write_json({"synthetic": True, "source_config": config}, self.manifest_report)
            return

        root = Path(self.args.gtsinger_root)
        if not root.exists():
            raise RuntimeError(f"GTSinger root not found: {root}")
        max_rows = self.args.smoke_max_rows_per_singer if self.args.smoke_only else self.args.max_rows_per_singer
        max_phone_pairs = 50 if self.args.smoke_only else self.args.max_phone_pairs_per_group
        cmd = [
            self.args.python,
            "legacy/data_prep/build_gtsinger_manifests.py",
            "--root",
            str(root),
            "--utterances-out",
            str(self.manifest),
            "--pairs-out",
            str(self.pairs),
            "--phone-examples-out",
            str(self.phone_examples),
            "--technique-pairs-out",
            str(self.technique_pairs),
            "--report-out",
            str(self.manifest_report),
            "--max-rows-per-singer",
            str(max_rows),
            "--max-phone-pairs-per-group",
            str(max_phone_pairs),
        ]
        if getattr(self.args, "fast_no_wav_stats", False):
            cmd.append("--fast-no-wav-stats")
        self.run_job(
            "prepare_gtsinger_manifests",
            cmd,
            [self.manifest, self.pairs, self.technique_pairs, self.manifest_report],
        )

    def write_smoke_manifest(self, rows: list[dict[str, Any]]) -> None:
        write_table(rows, self.run_root / "manifests" / "smoke_utterances.jsonl")

    def smoke_checks(self) -> None:
        smoke_manifest = self.run_root / "manifests" / "smoke_utterances.jsonl"
        for spec in self.model_specs:
            smoke_feature_root = self.run_root / "smoke_features"
            validation_feature_root = smoke_feature_root
            if spec.kind == "precomputed":
                validation_feature_root = self.feature_root
            else:
                ok = self.extract_features(
                    spec,
                    smoke_manifest,
                    smoke_feature_root,
                    job_prefix=f"smoke_extract_{spec.key}",
                    limit=10,
                    timeout_sec=self.args.smoke_job_timeout_sec,
                )
                if not ok:
                    self.disabled_models.add(spec.key)
                    self.write_skipped_job(
                        f"smoke_validate_{spec.key}",
                        f"skipping validation because smoke extraction failed for model {spec.key}",
                    )
                    continue
            for layer in spec.layers:
                self.run_job(
                    f"smoke_validate_{spec.key}_l{layer}",
                    [
                        self.args.python,
                        "legacy/data_prep/validate_feature_cache.py",
                        "--manifest",
                        str(smoke_manifest),
                        "--feature-root",
                        str(validation_feature_root),
                        "--extractor",
                        spec.extractor,
                        "--checkpoint-hash",
                        spec.checkpoint_hash,
                        "--layer",
                        layer,
                        "--report",
                        str(self.run_root / "reports" / f"smoke_{spec.key}_l{layer}_feature_cache.json"),
                    ],
                    [self.run_root / "reports" / f"smoke_{spec.key}_l{layer}_feature_cache.json"],
                    optional=True,
                    timeout_sec=self.args.smoke_job_timeout_sec,
                )

    def full_feature_extraction(self) -> None:
        for spec in self.model_specs:
            if spec.key in self.disabled_models:
                self.write_skipped_job(f"extract_{spec.key}", f"skipping disabled model {spec.key}")
                continue
            self.extract_features(
                spec,
                self.manifest,
                self.feature_root,
                job_prefix=f"extract_{spec.key}",
                timeout_sec=self.args.extract_job_timeout_sec,
            )

    def extract_features(
        self,
        spec: ModelSpec,
        manifest: Path,
        feature_root: Path,
        job_prefix: str,
        limit: int | None = None,
        timeout_sec: int | None = None,
    ) -> bool:
        if spec.kind == "acoustic":
            cmd = [
                self.args.python,
                "legacy/data_prep/extract_acoustic_features.py",
                "--manifest",
                str(manifest),
                "--feature-root",
                str(feature_root),
                "--extractor",
                spec.extractor,
                "--checkpoint-hash",
                spec.checkpoint_hash,
                "--layer",
                spec.layers[0],
            ]
            outputs = [feature_root / spec.extractor / spec.checkpoint_hash / "metadata.json"]
        elif spec.kind == "precomputed":
            return True
        elif spec.kind == "wavlm":
            cmd = [
                self.args.python,
                "legacy/data_prep/extract_wavlm_features.py",
                "--manifest",
                str(manifest),
                "--feature-root",
                str(feature_root),
                "--model-name",
                spec.model_name or "microsoft/wavlm-base-plus",
                "--checkpoint-hash",
                spec.checkpoint_hash,
                "--layers",
                *spec.layers,
                "--device",
                self.args.device,
            ]
            outputs = [feature_root / spec.extractor / spec.checkpoint_hash / layer / "metadata.json" for layer in spec.layers]
        else:
            cmd = [
                self.args.python,
                "legacy/data_prep/extract_hf_audio_features.py",
                "--manifest",
                str(manifest),
                "--feature-root",
                str(feature_root),
                "--extractor",
                spec.extractor,
                "--model-name",
                spec.model_name or "",
                "--checkpoint-hash",
                spec.checkpoint_hash,
                "--layers",
                *spec.layers,
                "--device",
                self.args.device,
            ]
            if spec.trust_remote_code:
                cmd.append("--trust-remote-code")
            outputs = [feature_root / spec.extractor / spec.checkpoint_hash / layer / "metadata.json" for layer in spec.layers]
        if limit:
            cmd.extend(["--limit", str(limit)])
        return self.run_job(job_prefix, cmd, outputs, optional=True, timeout_sec=timeout_sec)

    def track1_probes(self) -> None:
        for spec in self.model_specs:
            for layer in spec.layers:
                self.run_track1_layer(spec, layer)

    def run_track1_layer(self, spec: ModelSpec, layer: str) -> None:
        prefix = f"{spec.key}_l{layer}"
        if spec.key in self.disabled_models:
            self.write_skipped_job(f"track1_{prefix}", f"skipping disabled model {spec.key}")
            return
        result_dir = self.run_root / "track1" / spec.key / str(layer)
        result_dir.mkdir(parents=True, exist_ok=True)
        common = [
            "--manifest",
            str(self.manifest),
            "--feature-root",
            str(self.feature_root),
            "--extractor",
            spec.extractor,
            "--checkpoint-hash",
            spec.checkpoint_hash,
            "--layer",
            layer,
        ]
        self.run_job(
            f"track1_mode_raw_{prefix}",
            [
                self.args.python,
                "legacy/probing/run_mode_probe.py",
                *common,
                "--binary-singing",
                "--folds",
                str(self.args.folds),
                "--metrics-out",
                str(result_dir / "mode_raw_metrics.json"),
                "--predictions-out",
                str(result_dir / "mode_raw_predictions.jsonl"),
            ],
            [result_dir / "mode_raw_metrics.json", result_dir / "mode_raw_predictions.jsonl"],
            optional=True,
            timeout_sec=self.args.probe_job_timeout_sec,
        )
        self.run_job(
            f"track1_mode_controlled_{prefix}",
            [
                self.args.python,
                "legacy/probing/run_mode_probe.py",
                *common,
                "--binary-singing",
                "--folds",
                str(self.args.folds),
                "--residualize-nuisance",
                "--metrics-out",
                str(result_dir / "mode_controlled_metrics.json"),
                "--predictions-out",
                str(result_dir / "mode_controlled_predictions.jsonl"),
            ],
            [result_dir / "mode_controlled_metrics.json", result_dir / "mode_controlled_predictions.jsonl"],
            optional=True,
            timeout_sec=self.args.probe_job_timeout_sec,
        )
        self.run_job(
            f"track1_mode_shuffled_{prefix}",
            [
                self.args.python,
                "legacy/probing/run_mode_probe.py",
                *common,
                "--binary-singing",
                "--folds",
                str(self.args.folds),
                "--shuffle-labels",
                "--metrics-out",
                str(result_dir / "mode_shuffled_metrics.json"),
                "--predictions-out",
                str(result_dir / "mode_shuffled_predictions.jsonl"),
            ],
            [result_dir / "mode_shuffled_metrics.json", result_dir / "mode_shuffled_predictions.jsonl"],
            optional=True,
            timeout_sec=self.args.probe_job_timeout_sec,
        )
        self.run_job(
            f"track1_retrieval_{prefix}",
            [
                self.args.python,
                "legacy/probing/run_speaker_retrieval.py",
                *common,
                "--metrics-out",
                str(result_dir / "retrieval_metrics.json"),
            ],
            [result_dir / "retrieval_metrics.json"],
            optional=True,
            timeout_sec=self.args.probe_job_timeout_sec,
        )
        self.run_job(
            f"track1_mapper_{prefix}",
            [
                self.args.python,
                "legacy/intervention/run_micro_mapper.py",
                *common,
                "--pairs",
                str(self.pairs),
                "--metrics-out",
                str(result_dir / "mapper_metrics.json"),
                "--predictions-out",
                str(result_dir / "mapper_predictions.jsonl"),
                "--model-out",
                str(result_dir / "mapper_model.npz"),
            ],
            [result_dir / "mapper_metrics.json", result_dir / "mapper_predictions.jsonl", result_dir / "mapper_model.npz"],
            optional=True,
            timeout_sec=self.args.probe_job_timeout_sec,
        )

    def track2_probes(self) -> None:
        if not self.technique_pairs.exists():
            return
        for spec in self.model_specs:
            if spec.key in self.disabled_models:
                self.write_skipped_job(f"track2_{spec.key}", f"skipping disabled model {spec.key}")
                continue
            for layer in spec.layers:
                result_dir = self.run_root / "track2" / spec.key / str(layer)
                result_dir.mkdir(parents=True, exist_ok=True)
                self.run_job(
                    f"track2_technique_{spec.key}_l{layer}",
                    [
                        self.args.python,
                        "legacy/probing/run_technique_directions.py",
                        "--pairs",
                        str(self.technique_pairs),
                        "--feature-root",
                        str(self.feature_root),
                        "--extractor",
                        spec.extractor,
                        "--checkpoint-hash",
                        spec.checkpoint_hash,
                        "--layer",
                        layer,
                        "--metrics-out",
                        str(result_dir / "technique_metrics.json"),
                        "--directions-out",
                        str(result_dir / "technique_directions.jsonl"),
                        "--bootstrap",
                        str(self.args.bootstrap),
                    ],
                    [result_dir / "technique_metrics.json", result_dir / "technique_directions.jsonl"],
                    optional=True,
                    timeout_sec=self.args.probe_job_timeout_sec,
                )

    def write_skipped_job(self, name: str, reason: str) -> None:
        job_dir = self.jobs_root / name
        job_dir.mkdir(parents=True, exist_ok=True)
        status = {
            "name": name,
            "status": "skipped",
            "reason": reason,
            "optional": True,
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "ended_at": datetime.now().isoformat(timespec="seconds"),
        }
        write_json(status, job_dir / "status.json")
        (job_dir / "run.log").write_text(reason + "\n", encoding="utf-8")
        self.write_job_summary(job_dir, status)

    def run_job(
        self,
        name: str,
        cmd: list[str],
        outputs: list[Path],
        optional: bool = False,
        timeout_sec: int | None = None,
    ) -> bool:
        job_dir = self.jobs_root / name
        job_dir.mkdir(parents=True, exist_ok=True)
        status_path = job_dir / "status.json"
        log_path = job_dir / "run.log"
        command_path = job_dir / "config.json"
        write_json({"name": name, "cmd": cmd, "outputs": [str(path) for path in outputs], "optional": optional}, command_path)
        if not self.args.force and status_path.exists():
            try:
                status = json.loads(status_path.read_text())
                if status.get("status") == "success" and all(path.exists() for path in outputs):
                    print(f"skip completed job: {name}")
                    return True
            except json.JSONDecodeError:
                pass
        print(f"run job: {name}")
        started = datetime.now().isoformat(timespec="seconds")
        timeout_sec = timeout_sec if timeout_sec is not None else getattr(self.args, "job_timeout_sec", 0)
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("HF_HUB_DISABLE_XET", "1")
        cache_root = getattr(self.args, "local_cache_root", None)
        if cache_root:
            cache_root = Path(cache_root)
            cache_root.mkdir(parents=True, exist_ok=True)
            (cache_root / "huggingface" / "datasets").mkdir(parents=True, exist_ok=True)
            (cache_root / "huggingface" / "hub").mkdir(parents=True, exist_ok=True)
            (cache_root / "huggingface" / "transformers").mkdir(parents=True, exist_ok=True)
            (cache_root / "torch").mkdir(parents=True, exist_ok=True)
            (cache_root / "pip").mkdir(parents=True, exist_ok=True)
            (cache_root / "uv").mkdir(parents=True, exist_ok=True)
            (cache_root / "nv" / "ComputeCache").mkdir(parents=True, exist_ok=True)
            (cache_root / "torch_extensions").mkdir(parents=True, exist_ok=True)
            tmp_dir = Path(os.environ.get("TMPDIR", "/localdisk/bowen/tmp"))
            tmp_dir.mkdir(parents=True, exist_ok=True)
            env.setdefault("XDG_CACHE_HOME", str(cache_root))
            env.setdefault("HF_HOME", str(cache_root / "huggingface"))
            env.setdefault("HF_DATASETS_CACHE", str(cache_root / "huggingface" / "datasets"))
            env.setdefault("HUGGINGFACE_HUB_CACHE", str(cache_root / "huggingface" / "hub"))
            env.setdefault("TRANSFORMERS_CACHE", str(cache_root / "huggingface" / "transformers"))
            env.setdefault("TORCH_HOME", str(cache_root / "torch"))
            env.setdefault("PIP_CACHE_DIR", str(cache_root / "pip"))
            env.setdefault("UV_CACHE_DIR", str(cache_root / "uv"))
            env.setdefault("UV_LINK_MODE", "copy")
            env.setdefault("CUDA_CACHE_PATH", str(cache_root / "nv" / "ComputeCache"))
            env.setdefault("TORCH_EXTENSIONS_DIR", str(cache_root / "torch_extensions"))
            env.setdefault("TMPDIR", str(tmp_dir))
        returncode: int | None = None
        timed_out = False
        pid: int | None = None
        with log_path.open("w", encoding="utf-8") as log:
            log.write("$ " + " ".join(cmd) + "\n\n")
            log.flush()
            proc = subprocess.Popen(
                cmd,
                cwd=self.repo,
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
                env=env,
            )
            pid = proc.pid
            write_json(
                {
                    "name": name,
                    "status": "running",
                    "pid": pid,
                    "optional": optional,
                    "started_at": started,
                    "timeout_sec": timeout_sec,
                    "outputs": [str(path) for path in outputs],
                    "log": str(log_path),
                },
                status_path,
            )
            deadline = time.monotonic() + timeout_sec if timeout_sec and timeout_sec > 0 else None
            while True:
                returncode = proc.poll()
                if returncode is not None:
                    break
                if deadline is not None and time.monotonic() >= deadline:
                    timed_out = True
                    log.write(f"\nTIMEOUT after {timeout_sec} seconds; sending SIGTERM to process group {pid}.\n")
                    log.flush()
                    self.terminate_process_group(pid, signal.SIGTERM)
                    time.sleep(5)
                    if proc.poll() is None:
                        log.write(f"Process group {pid} still alive; sending SIGKILL.\n")
                        log.flush()
                        self.terminate_process_group(pid, signal.SIGKILL)
                    try:
                        returncode = proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        returncode = None
                        log.write("Process did not exit after SIGKILL; recording timeout and continuing.\n")
                        log.flush()
                    break
                time.sleep(1)
        ended = datetime.now().isoformat(timespec="seconds")
        missing_outputs = [str(path) for path in outputs if not path.exists()]
        status_name = "success" if returncode == 0 and not missing_outputs else "failed"
        if timed_out:
            status_name = "timeout"
        status = {
            "name": name,
            "status": status_name,
            "returncode": returncode,
            "pid": pid,
            "optional": optional,
            "started_at": started,
            "ended_at": ended,
            "timeout_sec": timeout_sec,
            "outputs": [str(path) for path in outputs],
            "missing_outputs": missing_outputs,
            "log": str(log_path),
        }
        write_json(status, status_path)
        self.write_job_summary(job_dir, status)
        if status_name != "success" and not optional:
            raise RuntimeError(f"required job failed: {name}; see {log_path}")
        return status_name == "success"

    def write_job_summary(self, job_dir: Path, status: dict[str, Any]) -> None:
        lines = [
            f"# {status.get('name', job_dir.name)}",
            "",
            f"- Status: `{status.get('status', 'unknown')}`",
            f"- Optional: `{status.get('optional', False)}`",
        ]
        if status.get("returncode") is not None:
            lines.append(f"- Return code: `{status.get('returncode')}`")
        if status.get("started_at"):
            lines.append(f"- Started: `{status.get('started_at')}`")
        if status.get("ended_at"):
            lines.append(f"- Ended: `{status.get('ended_at')}`")
        if status.get("timeout_sec"):
            lines.append(f"- Timeout seconds: `{status.get('timeout_sec')}`")
        if status.get("reason"):
            lines.extend(["", "Reason:", "", str(status["reason"])])
        missing = status.get("missing_outputs") or []
        if missing:
            lines.extend(["", "Missing outputs:"])
            lines.extend(f"- `{path}`" for path in missing)
        if status.get("log"):
            lines.extend(["", f"Log: `{status['log']}`"])
        (job_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    @staticmethod
    def terminate_process_group(pid: int | None, sig: signal.Signals) -> None:
        if pid is None:
            return
        try:
            os.killpg(pid, sig)
        except ProcessLookupError:
            return
        except PermissionError:
            return

    def collect_job_statuses(self) -> list[dict[str, Any]]:
        statuses = []
        for path in sorted(self.jobs_root.glob("*/status.json")):
            try:
                statuses.append(json.loads(path.read_text()))
            except json.JSONDecodeError:
                statuses.append({"name": path.parent.name, "status": "invalid_status"})
        return statuses

    def generate_report(self, extra_failures: list[str] | None = None) -> None:
        statuses = self.collect_job_statuses()
        lines = [
            "# Stage 1 Overnight Report",
            "",
            f"Generated: {datetime.now().isoformat(timespec='seconds')}",
            "",
            "## Run Configuration",
            "",
            f"- Run root: `{self.run_root}`",
            f"- Feature root: `{self.feature_root}`",
            f"- Python: `{self.args.python}`",
            f"- Device: `{self.args.device}`",
            f"- Models requested: `{', '.join(spec.key for spec in self.model_specs)}`",
            f"- Smoke only: `{self.args.smoke_only}`",
            "",
            "## What We Wanted To Test",
            "",
            "- Track 1: whether cross-mode identity survives speech-to-singing, and whether the residual is global, personalized, or mostly acoustic.",
            "- Track 2: whether technique directions are stable enough across phones/singers to justify later steering.",
            "",
            "## Data Used",
            "",
        ]
        if self.manifest_report.exists():
            report = json.loads(self.manifest_report.read_text())
            lines.append("```json")
            lines.append(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True)[:6000])
            lines.append("```")
        else:
            lines.append("- Manifest report was not created.")
        failed = [status for status in statuses if status.get("status") not in {"success", "skipped"}]
        succeeded = [status for status in statuses if status.get("status") == "success"]
        skipped = [status for status in statuses if status.get("status") == "skipped"]
        lines.extend(
            [
                "",
                "## Job Status",
                "",
                f"- Successful jobs: {len(succeeded)}",
                f"- Failed jobs: {len(failed)}",
                f"- Skipped jobs: {len(skipped)}",
            ]
        )
        if failed:
            lines.append("")
            lines.append("Failed jobs:")
            for status in failed[:30]:
                lines.append(f"- `{status.get('name')}` returncode={status.get('returncode')} log=`{status.get('log')}`")
        if extra_failures:
            lines.append("")
            lines.append("Global failures:")
            for failure in extra_failures:
                lines.append(f"- {failure}")
        lines.extend(["", "## Track 1 Results", ""])
        lines.extend(self.track1_report_lines())
        lines.extend(["", "## Track 2 Results", ""])
        lines.extend(self.track2_report_lines())
        lines.extend(["", "## Human Check List", ""])
        lines.extend(
            [
                "- Check failed optional model logs before interpreting missing model comparisons.",
                "- Confirm any downloaded dataset/checkpoint licenses before thesis reporting.",
                "- For Track 1, inspect whether global residual is consistently competitive across folds and datasets.",
                "- For Track 2, inspect technique groups that pass both bootstrap and analogy tests before any steering.",
            ]
        )
        lines.extend(["", "## Decision", ""])
        lines.append(self.decision_text())
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        final_status = "failed" if extra_failures else ("complete_with_failures" if failed else "complete")
        write_json({"status": final_status, "report": str(self.report_path)}, self.run_root / "status.json")

    def track1_report_lines(self) -> list[str]:
        rows = []
        for metrics_path in sorted(self.run_root.glob("track1/*/*/mode_controlled_metrics.json")):
            try:
                controlled_data = json.loads(metrics_path.read_text())
                raw_path = metrics_path.parent / "mode_raw_metrics.json"
                shuffled_path = metrics_path.parent / "mode_shuffled_metrics.json"
                raw_data = json.loads(raw_path.read_text()) if raw_path.exists() else {}
                shuffled_data = json.loads(shuffled_path.read_text()) if shuffled_path.exists() else {}
                mapper = metrics_path.parent / "mapper_metrics.json"
                retrieval = metrics_path.parent / "retrieval_metrics.json"
                mapper_data = json.loads(mapper.read_text()) if mapper.exists() else {}
                retrieval_data = json.loads(retrieval.read_text()) if retrieval.exists() else {}
                rows.append(
                    {
                        "model": metrics_path.parts[-3],
                        "layer": metrics_path.parts[-2],
                        "raw_signed_auc": raw_data.get("metrics", {}).get("signed_auc_mean"),
                        "raw_sep_auc": raw_data.get("metrics", {}).get("separability_auc_mean"),
                        "controlled_signed_auc": controlled_data["metrics"].get("signed_auc_mean"),
                        "controlled_sep_auc": controlled_data["metrics"].get("separability_auc_mean"),
                        "shuffled_sep_auc": shuffled_data.get("metrics", {}).get("separability_auc_mean"),
                        "orientation": controlled_data.get("orientation_counts", {}),
                        "r1": retrieval_data.get("singing_to_speech_recall_at_1"),
                        "r5": retrieval_data.get("singing_to_speech_recall_at_5"),
                        "mapper": mapper_data.get("cosine_mean"),
                        "global": mapper_data.get("global_mean_residual_cosine_mean"),
                    }
                )
            except Exception:  # noqa: BLE001
                continue
        if not rows:
            return ["No Track 1 metrics were produced."]
        lines = [
            "| Model | Layer | Raw signed AUC | Raw sep AUC | Controlled signed AUC | Controlled sep AUC | Shuffled sep AUC | R@1 | R@5 | Mapper cosine | Global residual cosine |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in rows:
            lines.append(
                "| {model} | {layer} | {raw_signed_auc:.3f} | {raw_sep_auc:.3f} | {controlled_signed_auc:.3f} | {controlled_sep_auc:.3f} | {shuffled_sep_auc:.3f} | {r1:.3f} | {r5:.3f} | {mapper:.3f} | {global_cosine:.3f} |".format(
                    model=row["model"],
                    layer=row["layer"],
                    raw_signed_auc=safe_float(row["raw_signed_auc"]),
                    raw_sep_auc=safe_float(row["raw_sep_auc"]),
                    controlled_signed_auc=safe_float(row["controlled_signed_auc"]),
                    controlled_sep_auc=safe_float(row["controlled_sep_auc"]),
                    shuffled_sep_auc=safe_float(row["shuffled_sep_auc"]),
                    r1=safe_float(row["r1"]),
                    r5=safe_float(row["r5"]),
                    mapper=safe_float(row["mapper"]),
                    global_cosine=safe_float(row["global"]),
                )
            )
        return lines

    def track2_report_lines(self) -> list[str]:
        rows = []
        for metrics_path in sorted(self.run_root.glob("track2/*/*/technique_metrics.json")):
            try:
                data = json.loads(metrics_path.read_text())
                reliable = [g for g in data.get("groups", []) if g.get("num_pairs", 0) >= 3]
                stable = [
                    g
                    for g in reliable
                    if g.get("bootstrap_angle_mean_deg", 999) <= 30
                    and g.get("analogy_top1", 0) > g.get("wrong_technique_top1", 0)
                    and g.get("analogy_top1", 0) > g.get("shuffled_direction_top1", 0)
                ]
                rows.append(
                    {
                        "model": metrics_path.parts[-3],
                        "layer": metrics_path.parts[-2],
                        "reliable": len(reliable),
                        "stable": len(stable),
                        "best": sorted(reliable, key=lambda g: g.get("bootstrap_angle_mean_deg", 999))[:3],
                    }
                )
            except Exception:  # noqa: BLE001
                continue
        if not rows:
            return ["No Track 2 metrics were produced."]
        lines = ["| Model | Layer | Reliable groups | Passed groups | Best groups |", "|---|---:|---:|---:|---|"]
        for row in rows:
            best = ", ".join(
                f"{g.get('phone')}/{g.get('target_technique')} n={g.get('num_pairs', 0)} spk={g.get('num_speakers', 0)} lang={g.get('num_languages', 0)} angle={g.get('bootstrap_angle_mean_deg', 0):.1f} analogy={g.get('analogy_top1', 0):.2f} wrong={g.get('wrong_technique_top1', 0):.2f} shuffled={g.get('shuffled_direction_top1', 0):.2f}"
                for g in row["best"]
            )
            lines.append(f"| {row['model']} | {row['layer']} | {row['reliable']} | {row['stable']} | {best} |")
        return lines

    def decision_text(self) -> str:
        track1_files = list(self.run_root.glob("track1/*/*/mode_controlled_metrics.json"))
        track2_files = list(self.run_root.glob("track2/*/*/technique_metrics.json"))
        if not track1_files and not track2_files:
            return "No decision: Stage 1 metrics did not complete."
        return (
            "Use the table thresholds rather than raw AUC alone. Continue Track 1 if cross-mode retrieval and residual baselines "
            "hold across folds/models. Continue Track 2 only for techniques that pass both bootstrap stability and analogy retrieval."
        )


def safe_float(value: Any) -> float:
    try:
        if value is None:
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the robust Stage 1 overnight experiment suite.")
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_WORK_ROOT / "runs" / "stage1_overnight_run")
    parser.add_argument("--report", type=Path, default=Path("results/stage1_overnight_report.md"))
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_WORK_ROOT / "features" / "stage1_overnight")
    parser.add_argument("--gtsinger-root", type=Path, default=Path(DEFAULT_GTSINGER))
    parser.add_argument("--existing-manifest", type=Path)
    parser.add_argument("--existing-pairs", type=Path)
    parser.add_argument("--existing-phone-examples", type=Path)
    parser.add_argument("--existing-technique-pairs", type=Path)
    parser.add_argument("--existing-manifest-report", type=Path)
    parser.add_argument("--models", default="acoustic,wavlm,hubert,mert")
    parser.add_argument("--contentvec-model")
    parser.add_argument("--contentvec-layer", default="12")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--max-rows-per-singer", type=int, default=5000)
    parser.add_argument("--smoke-max-rows-per-singer", type=int, default=200)
    parser.add_argument("--max-phone-pairs-per-group", type=int, default=2000)
    parser.add_argument("--fast-no-wav-stats", action="store_true")
    parser.add_argument("--job-timeout-sec", type=int, default=7200)
    parser.add_argument("--smoke-job-timeout-sec", type=int, default=1800)
    parser.add_argument("--extract-job-timeout-sec", type=int, default=21600)
    parser.add_argument("--probe-job-timeout-sec", type=int, default=7200)
    parser.add_argument("--local-cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--allow-home-heavy-io", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runner = Stage1Runner(args)
    return runner.run()


if __name__ == "__main__":
    raise SystemExit(main())
