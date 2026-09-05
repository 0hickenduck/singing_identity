# singing_identity — agent conventions
Global rules: ~/.codex/AGENTS.md. This file adds project specifics only.

## Topic
Speech vs singing representation geometry in frozen SSL models (WavLM/HuBERT/MERT).

## Paths (node-local; check `hostname`)
RUN_ROOT      /localdisk/bowen/singing_identity/runs/<RUN_NAME>
FEATURE_ROOT  /localdisk/bowen/singing_identity/features/<RUN_NAME>
CACHE_ROOT    /localdisk/bowen/.cache/singing_identity

## Run completion — a run is done only when all three exist
1. RUN_ROOT/ with the exact config/command used and git_commit.txt (hash + dirty flag)
2. results/<RUN_NAME>/ in the repo: report.md (what ran, key metrics, verdict, localdisk paths) plus small
   figures / short audio clips for Bowen to open in VS Code. Files ≤2 MB; row-level dumps stay on /localdisk.
3. one row appended to results/index.md:  | YYYY-MM-DD | <RUN_NAME> | one-sentence verdict | [report](<RUN_NAME>/report.md) |
RUN_NAME = <experiment>_<YYYY-MM-DD>[_smoke|_minimal]. /localdisk is deletable scratch; results/ is the record.

## Notebooks
marimo, not Jupyter: `uv run --with marimo marimo edit notebooks/<name>.py`; headless runs in tmux via `marimo run`.
Final claims must be reproducible from scripts/configs, never from notebook state.
