# Lab Server Research Workflow

This is the working manual for doing singing-identity research on the Gavo lab cluster.
Keep this file in `/home/bowen` with the repository because it is small, shared across
nodes, and should travel with the code.

## Core Rule

Use `/home/bowen` for human-readable and git-tracked research state. Use verified
`/localdisk/bowen` for active machine I/O.

| Kind of artifact | Location | Notes |
|---|---|---|
| Code, configs, scripts | `/home/bowen/bowen_lab/projects/singing_identity` | Git-tracked or small untracked files. |
| Manuscript drafts, notes, final reports | `/home/bowen/bowen_lab/projects/singing_identity` | Markdown, LaTeX, JSON summaries, small tables, small plots. |
| Marimo notebook source | `/home/bowen/bowen_lab/projects/singing_identity/notebooks` | Notebook code is small. Large outputs still go to `/localdisk`. |
| Active dataset copies | `/localdisk/bowen/singing_identity/data` | Copy from `/common/db` or `/home` before training. |
| Feature caches | `/localdisk/bowen/singing_identity/features/<run_name>` | Do not put frame-level features in git or `/home`. |
| Run roots, logs, predictions | `/localdisk/bowen/singing_identity/runs/<run_name>` | Keep high-frequency writes off NFS. |
| Checkpoints | `/localdisk/bowen/singing_identity/checkpoints/<run_name>` | Use `save_total_limit=1` or `2`; copy only final compact models back. |
| Tool/model caches | `/localdisk/bowen/.cache` | Hugging Face, uv, Torch, CUDA, temporary files. |

On 2026-06-23 from `valkyrie03`, the current facts were:

```bash
/home      poseidon:/home     nfs4  95.8T  1.9T  98%
/localdisk /dev/nvme0n1p1     ext4   1.8T 458.5G  70%
```

Always rerun `findmnt -T <path>` before a long job because node/storage facts can change.

## Before Any Long Run

Run this inside `tmux` on a `valkyrie` node, not on `athena`:

```bash
cd /home/bowen/bowen_lab/projects/singing_identity
bash scripts/check_lab_environment.sh
```

The required manual checklist is:

```bash
hostname
all_gpus
all_cpus
findmnt -T /localdisk/bowen -o TARGET,SOURCE,FSTYPE,SIZE,AVAIL,USE%
mkdir -p /localdisk/bowen/singing_identity/{data,runs,features,logs,status,checkpoints}
mkdir -p /localdisk/bowen/.cache/{huggingface,torch,uv,pip,nv,torch_extensions}
mkdir -p /localdisk/bowen/tmp
```

Do not use `valkyrie-status.sh` or direct `nvidia-smi` as the normal resource check.
If `all_gpus` or `all_cpus` is missing, fix the lab helper installation before starting
a GPU run.

## Shell Environment

`/home/bowen/.local/bin` should be on `PATH` so user-installed tools and lab helper
scripts can be found:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

Use these cache exports for compute-node work:

```bash
export XDG_CACHE_HOME=/localdisk/bowen/.cache
export HF_HOME=/localdisk/bowen/.cache/huggingface
export HF_DATASETS_CACHE=/localdisk/bowen/.cache/huggingface/datasets
export HUGGINGFACE_HUB_CACHE=/localdisk/bowen/.cache/huggingface/hub
export TRANSFORMERS_CACHE=/localdisk/bowen/.cache/huggingface/transformers
export TORCH_HOME=/localdisk/bowen/.cache/torch
export PIP_CACHE_DIR=/localdisk/bowen/.cache/pip
export UV_CACHE_DIR=/localdisk/bowen/.cache/uv
export UV_LINK_MODE=copy
export CUDA_CACHE_PATH=/localdisk/bowen/.cache/nv/ComputeCache
export TORCH_EXTENSIONS_DIR=/localdisk/bowen/.cache/torch_extensions
export TMPDIR=/localdisk/bowen/tmp
```

These exports are lightweight enough for shell startup. Slow initialization such as
`nvm` should stay inside an interactive-shell guard:

```bash
if [[ $- == *i* ]]; then
    # interactive-only initialization
fi
```

`/home/bowen/.bashrc` should keep this PATH line and route compute-node caches to
`/localdisk/bowen`. It must not point caches at `/work/bowen`; `/work` has been
observed as NFS on `valkyrie03`.

## Fast Binaries

The fast helper binaries are expected in:

```bash
/home/bowen/.local/bin
```

Current checked commands:

| Command | Purpose | Notes |
|---|---|---|
| `rg` | Fast text search | Prefer over `grep` for repository searches. |
| `dust` | Disk usage inspection | Prefer over recursive `du` on large trees. |
| `cpz` | Fast copy | Rust-native FUC command. Use for large NFS copies. |
| `rmz` | Fast remove | Rust-native FUC command. Use for large NFS deletes. |
| `all_gpus` | GPU usage check | Local compatibility wrapper currently backed by `gpu-users`. |
| `all_cpus` | CPU usage check | Local compatibility wrapper currently backed by `high-cpu-users`. |

`cpz` and `rmz` were restored from Steven's copies:

```bash
cp /home/smcintosh/.local/bin/cpz /home/bowen/.local/bin/cpz
cp /home/smcintosh/.local/bin/rmz /home/bowen/.local/bin/rmz
chmod +x /home/bowen/.local/bin/cpz /home/bowen/.local/bin/rmz
```

If a helper disappears, first check:

```bash
command -v rg dust cpz rmz all_gpus all_cpus
ls -l /home/bowen/.local/bin
```

## Python And Notebooks

Use `uv` for new environments and package operations. Do not create new Conda, Poetry,
or plain-pip environments for this project.

First-time setup:

```bash
cd /home/bowen/bowen_lab/projects/singing_identity
bash scripts/setup_public_tools.sh
uv sync
uv pip install torch torchvision torchaudio torchcodec --index-url https://download.pytorch.org/whl/cu128
```

The current `pyproject.toml` is intentionally non-packaged (`tool.uv.package = false`);
scripts are run from the repository root. Add dependencies there before installing
ad hoc packages.

Use `marimo` instead of Jupyter for exploratory analysis. The default pattern does
not require a permanent `marimo` binary:

```bash
uv run --with marimo marimo edit notebooks/stage1_analysis.py
```

For a notebook that should be edited by both human and AI:

```bash
uv run --with marimo marimo edit notebooks/experiment.py
```

Marimo notebooks are normal Python files. Cells should use `@app.cell` decorators.
When an agent edits `notebooks/experiment.py` on disk, Marimo detects the file change
and refreshes cells in the browser.

For headless execution, use `tmux` and `marimo run`:

```bash
tmux new -s marimo_experiment
cd /home/bowen/bowen_lab/projects/singing_identity
uv run --with marimo marimo run notebooks/experiment.py 2>&1 | tee /localdisk/bowen/singing_identity/logs/experiment.log
```

For audio notebooks, include a final cell with `marimo.ui.audio` for previewing small
generated wav samples when that API exists. In the currently verified Marimo 0.17.6,
use `mo.audio(...)`; see `notebooks/audio_experiment_template.py`. Store large generated
wavs under `/localdisk/bowen/singing_identity`, not inside the repository.

Notebook source belongs in `/home/bowen/.../notebooks`. Any dataframe caches, extracted
audio, predictions, or generated large media used by the notebook must be written under
`/localdisk/bowen/singing_identity`.

Final results must not depend on hidden notebook state. Promote important notebook
logic into scripts and configs before making a research claim.

## Data Movement

Never train directly from `/common/db` or large folders under `/home`. Copy the needed
subset to `/localdisk/bowen` first.

For large NFS operations, use lab-approved helpers:

```bash
cpz <source_dir> /localdisk/bowen/singing_identity/data/<dataset_name>
rmz /localdisk/bowen/singing_identity/runs/<bad_or_old_run>
```

If `cpz` or `rmz` is unavailable, do not replace it with a large recursive `cp` or
`rm` against NFS. Use `scripts/check_lab_environment.sh` to confirm the gap, then
install or locate the lab helper.

Use `rsync --bwlimit` only for deliberate, throttled migrations where `cpz` is not
available or where resume/progress behavior matters.

## Run Convention

Prefer this naming scheme:

```bash
RUN_NAME=<short_date_or_experiment_name>
WORK_ROOT=/localdisk/bowen/singing_identity
DATA_ROOT=$WORK_ROOT/data/<dataset_name>
RUN_ROOT=$WORK_ROOT/runs/$RUN_NAME
FEATURE_ROOT=$WORK_ROOT/features/$RUN_NAME
CACHE_ROOT=/localdisk/bowen/.cache/singing_identity
LOG_FILE=$WORK_ROOT/logs/$RUN_NAME.log
REPORT_OUT=/home/bowen/bowen_lab/projects/singing_identity/results/${RUN_NAME}_report.md
```

Record at least this metadata in each run log:

```bash
date
hostname
all_gpus
all_cpus
findmnt -T "$RUN_ROOT" -o TARGET,SOURCE,FSTYPE,SIZE,AVAIL,USE%
findmnt -T "$FEATURE_ROOT" -o TARGET,SOURCE,FSTYPE,SIZE,AVAIL,USE%
findmnt -T "$CACHE_ROOT" -o TARGET,SOURCE,FSTYPE,SIZE,AVAIL,USE%
python --version
which python
```

## Current Stage 1 Entrypoints

For a clean rerun where dataset, caches, features, and run outputs are born on local
NVMe:

```bash
tmux new -s singing_stage1_fresh_local
cd /home/bowen/bowen_lab/projects/singing_identity
bash scripts/run_fresh_stage1_on_localdisk.sh
```

This writes by default to:

```bash
/localdisk/bowen/singing_identity/data/gtsinger_domain_eval
/localdisk/bowen/singing_identity/runs/stage1_repaired_200_fresh_local
/localdisk/bowen/singing_identity/features/stage1_repaired_200_fresh_local
/localdisk/bowen/.cache/singing_identity
```

To resume the repaired 200-row Stage 1 run:

```bash
tmux new -s singing_stage1_resume
cd /home/bowen/bowen_lab/projects/singing_identity
bash scripts/resume_stage1_repaired_200_on_localdisk.sh
```

The old `*_on_work.sh` script names are kept only as compatibility wrappers. They now
delegate to the localdisk scripts.

Monitor a run with:

```bash
tail -f /localdisk/bowen/singing_identity/logs/<run_name>.log
cat /localdisk/bowen/singing_identity/status/<run_name>/status.json
```

## Cleanup

Clean local scratch when an experiment is finished:

1. Copy final compact reports, small JSON summaries, and selected small models back to
   `/home/bowen/bowen_lab/projects/singing_identity`.
2. Confirm the path is under `/localdisk/bowen`.
3. Remove large failed runs, stale feature caches, and temporary dataset copies.

Example:

```bash
findmnt -T /localdisk/bowen/singing_identity/runs/old_run -o TARGET,SOURCE,FSTYPE,SIZE,AVAIL,USE%
rmz /localdisk/bowen/singing_identity/runs/old_run
rmz /localdisk/bowen/singing_identity/features/old_run
```

If `rmz` is missing and the target is verified local ext4, normal `rm -rf` is acceptable
for local scratch cleanup. Do not use normal recursive delete for large NFS folders.

## What To Add Later

Add these only when they become real project needs:

- `pyproject.toml` with the exact uv-managed dependencies for this repo.
- A marimo notebook for Stage 1 analysis after metrics exist.
- A small `results/index.md` that links final compact reports and figures.
- A dataset manifest registry that records source path, localdisk copy path, checksum
  policy, and deletion date.

Do not add large audio, feature arrays, prediction dumps, or checkpoints to git.
