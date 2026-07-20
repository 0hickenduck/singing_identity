# Lab Cluster Operating Rules

These rules are mandatory for AI assistants working in this repository on the lab cluster.

## Cluster Layout

- Local laptop connects through `athena`.
- `athena` is only a jump host. Do not run code, downloads, training, or data processing there.
- Real compute runs on `valkyrie01` through `valkyrie08`.
- `/home/bowen` is shared NFS storage visible from all valkyrie nodes.
- `/common/db` is shared NFS dataset storage. Do not stream training data from it.
- On `valkyrie03` as checked on 2026-06-22, `/work` is also NFS (`poseidon:/work`), not local scratch.
- On `valkyrie03`, the real local scratch disk is `/localdisk` (`/dev/nvme0n1p1`, ext4).
- Always verify the current node with `findmnt -T <path>` before choosing a heavy I/O root. Do not rely on handbook assumptions when mount facts disagree.

## Path Policy

Keep code, configs, small documents, and final compact reports in `/home/bowen`.

Put all high-volume or high-frequency I/O on verified local scratch, currently `/localdisk/bowen` on `valkyrie03`, including:

- datasets and wav/audio copies used by experiments;
- feature caches and representation outputs;
- run roots for overnight experiments;
- model checkpoints and training outputs;
- Hugging Face, Torch, uv, pip-compatible, CUDA, and dataset caches;
- prediction JSONL files and large intermediate logs.

Do not judge safety by the assistant's working directory. Judge it by the actual paths passed to the command. A command launched from a repo in `/home` can still overload NFS if `--data`, `--feature-root`, `--run-root`, `--output`, `--checkpoint-dir`, or cache variables point into `/home`.

For large NFS copies or deletes, use the lab fast helpers `cpz` and `rmz` when available. If they are missing, do not replace them with a large `cp -r` or `rm -rf` on NFS; first install or locate the lab-approved helper.

Fast helper binaries live in `/home/bowen/.local/bin`, which must be on `PATH`.
Expected helpers:

- `rg` for fast text search;
- `dust` for disk usage inspection;
- `cpz` and `rmz` from FUC (Fast Unix Commands) for large NFS copies/deletes;
- `all_gpus` and `all_cpus` for resource checks. In this account they are compatibility wrappers backed by `gpu-users` and `high-cpu-users`.

If `cpz` or `rmz` disappears, restore them from:

```bash
cp /home/smcintosh/.local/bin/cpz /home/bowen/.local/bin/cpz
cp /home/smcintosh/.local/bin/rmz /home/bowen/.local/bin/rmz
chmod +x /home/bowen/.local/bin/cpz /home/bowen/.local/bin/rmz
```

## Required Preflight For Long Runs

Before downloads, feature extraction, training, or overnight experiments:

1. Run on a `valkyrie` compute node, not `athena`.
2. Check GPU and CPU usage with the lab helpers:

```bash
all_gpus
all_cpus
```

Do not use `valkyrie-status.sh` or direct `nvidia-smi` as the normal resource check.
3. Confirm the heavy root is local, not NFS:

```bash
findmnt -T /localdisk/bowen -o TARGET,SOURCE,FSTYPE,SIZE,AVAIL,USE%
```

4. Put heavy paths under verified local scratch, currently `/localdisk/bowen` on `valkyrie03`.
5. Create required local scratch directories before writing:

```bash
mkdir -p /localdisk/bowen/<target-subdir>
```

6. Set cache variables away from `/home` and NFS-backed `/work`:

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

7. Record `hostname`, run root, feature root, cache root, Python path, mount facts, and `all_gpus`/`all_cpus` output in the run config/log.
8. Start jobs expected to run longer than 10 minutes inside `tmux` or with `nohup ... &`. Do not run long training/extraction jobs directly in the foreground of an SSH session.
9. Stop jobs with `Ctrl+C` or a clean termination command. Do not use `Ctrl+Z` for training or extraction jobs.

## Experiment Run Convention

For this project, prefer:

```bash
RUN_ROOT=/localdisk/bowen/singing_identity/runs/<run_name>
FEATURE_ROOT=/localdisk/bowen/singing_identity/features/<run_name>
CACHE_ROOT=/localdisk/bowen/.cache/singing_identity
```

Copy only final compact artifacts back to `/home`, for example:

- final Markdown reports;
- small metrics summaries;
- research logs;
- final small models that should be preserved.

Do not archive large feature caches, wavs, model weights, or prediction dumps into git.

## VS Code / Shell Notes

- Ensure user-local tools are discoverable:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

- Keep slow shell initialization such as nvm or other interactive-only tools inside:

```bash
if [[ $- == *i* ]]; then
    # interactive-only initialization
fi
```

- Lightweight environment variable exports are acceptable outside that guard.
- Use `uv` for new project environments and Python package operations. Do not create new Conda, Poetry, or plain-pip environments for this project.
- Install PyTorch through the CUDA 12.8 index when creating a fresh uv environment:

```bash
uv pip install torch torchvision torchaudio torchcodec --index-url https://download.pytorch.org/whl/cu128
```

- Use `marimo` rather than Jupyter for exploratory notebooks:

```bash
uv run --with marimo marimo edit notebooks/<name>.py
```

- Marimo notebooks are plain Python files with `@app.cell` decorators. When an agent edits the file on disk, the running Marimo browser session should refresh the changed cells.
- For headless or long notebook runs, use `tmux` and:

```bash
uv run --with marimo marimo run notebooks/<name>.py
```

- For audio analysis notebooks, include a final audio preview cell only for small wavs. In the currently verified Marimo 0.17.6 API, use `mo.audio(...)`; `notebooks/audio_experiment_template.py` is the local template. Large generated wavs and caches must go under `/localdisk/bowen/singing_identity`.
- Notebooks are for exploration and figures. Final claims must be reproducible from scripts/configs with outputs written to `/localdisk/bowen` and compact reports copied back to `/home/bowen`.
- If VS Code Remote-SSH hangs, use `Remote-SSH: Kill VS Code Server on Host`. Do not manually delete `~/.vscode-server`.

## Human Review Server Workflow

When an experiment needs human judgment, produce a Marimo review notebook and
launch it with the project review server scripts. Do not point the reviewer only
to raw files.

After creating the review notebook(s), call:

```bash
scripts/review/start_review_servers.sh \
  2719:notebooks/<summary_review>.py \
  2720:notebooks/<listening_or_detail_review>.py
```

Use `PORT:NOTEBOOK_PATH` pairs. Choose ports explicitly and report them to the
human. Standard review ports are `2719`, `2720`, and `2721`; use other free
ports if needed. The start script:

- kills existing Marimo listeners on those ports;
- starts each notebook in a dedicated `tmux` session;
- exports `UV_CACHE_DIR=/localdisk/bowen/.cache/uv`;
- waits until all ports are listening;
- prints local URLs for the reviewer.

Always include in the review handoff:

- notebook path(s);
- port assignment(s);
- run root(s);
- the exact decision needed from the human.

The human manages their own local SSH tunnel from the Windows machine. Codex
does not need to run commands on the local machine. The local helper is:

```powershell
powershell -File C:\Users\libowen\bin\open_lab_tunnel.ps1 -Ports 2719,2720,2721
```

A repo copy of that helper lives at:

```text
scripts/review/open_lab_tunnel.ps1
```

After the reviewer replies with feedback in the conversation, call:

```bash
scripts/review/stop_review_servers.sh
```

Then incorporate the feedback into the experiment report and continue the next
objective/debug step. If `stop_review_servers.sh` reports a port is still in use,
resolve that before starting another review session.

## Default Decision Rule

If a task may repeatedly read/write many files, use verified local scratch such as `/localdisk/bowen` on `valkyrie03`.
If a task is small, low-frequency, and needs cross-node visibility, `/home/bowen` is acceptable.
When uncertain, run `findmnt -T <path>` first; do not choose an NFS path for heavy I/O.

## Git Sync Rules

- Code, configs, docs, and compact results move only through git
  (origin: `github.com/0hickenduck/singing_identity`). The rsync scripts
  (`sync_to_lab.sh`, `sync_from_lab.sh`) are only for large non-git artifacts.
- Sessions launched with the `cx` wrapper pull automatically. If
  `git pull --ff-only` fails, stop and report it; never merge or force-push.
- End every completed task by running `bash scripts/sync_push.sh "<short message>"`.
  It stages only files up to 2MB and lists anything it skipped; larger artifacts
  belong under `/localdisk/bowen`, not in the repo.

## Run Completion Contract

A run is complete only when all four of these exist:

1. `/localdisk/bowen/singing_identity/runs/<RUN_NAME>/` containing a copy of the
   exact config or command used.
2. `/localdisk/bowen/singing_identity/status/<RUN_NAME>/status.json`.
3. `results/<RUN_NAME>/` in the repo with a compact report
   (`final_experiment_report.md` or `README_results.md`): what ran, key metrics,
   verdict, and pointers to the localdisk paths. Small files only (up to 2MB
   each); row-level dumps and score matrices stay on `/localdisk`.
4. One appended row in `results/index.md`:
   `| YYYY-MM-DD | <RUN_NAME> | one-sentence verdict | [report](<RUN_NAME>/...) |`

`RUN_NAME` format: `<experiment>_<YYYY-MM-DD>[_smoke|_minimal]`. Everything on
`/localdisk` is deletable scratch; the `results/` report is the artifact of record.

## Session Start (any machine)

At the start of every session, run `git pull --ff-only`. If it fails because of
local uncommitted changes or divergence, do not merge, rebase, stash, or force
anything automatically — show the user `git status` and ask how to proceed.
Pushing is the user's (or sync_push.sh's) job at task end, never automatic
mid-session.
