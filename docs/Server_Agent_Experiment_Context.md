# Valkyrie Server Agent: Experiment Context & Workflow Guidelines

> **Purpose**: This document serves as the "headboard" or core context file for AI agents operating on the `valkyrie` GPU servers. It outlines the environment constraints, execution workflows, and the current state of the Singing Identity research project.

## 1. Environment & Filesystem Constraints (Critical)

You are running on a `valkyrie` GPU server. You must obey the following storage and I/O rules to avoid crippling network performance:

- **NFS (Network File System)**: `/home/bowen/` (and `/work/`)
  - **Use for**: Code, scripts, `.venv`, git repositories, and final saved checkpoints.
  - **Commands**: NEVER use standard `cp`, `rm`, or `rsync` for large folders on NFS. Always use the Rust-optimized `cpz` and `rmz` located in `~/.local/bin/`.
- **Local SSD (Node Local)**: `/localdisk/bowen/`
  - **Use for**: Datasets, active experiment runs, intermediate checkpoints, and HuggingFace cache.
  - **Cache Variable**: Always export `HF_HOME="/localdisk/bowen/.cache/huggingface"` before running models.
  - *Note*: Data on `/localdisk` is node-specific. You must copy final results back to `/home/bowen` when a run finishes.

## 2. Python & Marimo Notebook Workflow

We use **`uv`** for Python package management and **`marimo`** for experiments. Never use `conda` or raw `pip`.

- **Virtual Environment**: Use `uv venv` and run Python via `uv run python`.
- **PyTorch**: Install specifically with `uv pip install torch torchvision torchaudio torchcodec --index-url https://download.pytorch.org/whl/cu128`.
- **Experiment Execution (Marimo)**:
  - **Debugging/Exploration**: `uv run marimo edit notebook.py` (provides reactive web UI).
  - **Long Training Runs**: `uv run marimo run notebook.py` (headless execution inside `tmux`).
- **GPU Memory Management**: 
  - If you change model architecture, tensor shapes, or hit a CUDA Out-Of-Memory (OOM) error, you **MUST restart the kernel** (`pkill -f "marimo edit notebook.py"`) before re-running. Simply re-executing a cell will leak GPU memory.
- **Results Dashboard**: Structure notebooks with a final `@app.cell` acting as a results dashboard containing loss curves (`matplotlib`) and generated audio previews (`mo.audio()`) so they can be viewed remotely via SSH port-forwarding (e.g., `http://localhost:2718`).

## 3. Research Context: Singing Identity & Timbre

**Project Root**: `/home/bowen/bowen_lab/projects/singing_identity/`

**State of the Project**:
We are investigating how singing identity and technique (specifically breathy phonation) are represented in frozen SSL models (WavLM, HuBERT, MERT). 
- **Track 1 (Prompt-Mode Mismatch)**: We found a predictable residual between speech and singing representations of the same person. It is mostly explained by acoustic/prosodic covariates, but it does NOT provide a clean disentanglement of timbre vs. technique.
- **Track 2 (Breathy Geometry)**: Breathy phonation is linearly decodable in frozen models, but it does NOT correspond to a single clean "global vector".

**Current Research Focus (The Next Experiment)**:
The focus is shifting towards the **Identity-Technology Tradeoff (Timbre Shrinkage)** hypothesis:
> *Does singing technique compress or reshape speaker identity representations? In other words, as technique/phonation separability increases, does speaker identity separability decrease?*

*Alternatively*, the agent may be asked to investigate the global vs. local geometry of breathy phonation (is it language-local, singer-local, or phone-local?).

## 4. Agent Objectives & Next Steps

When instructed to run the next phase of the experiment, your workflow is:

1. **Setup**: Ensure the `GTSinger` dataset and necessary frozen representations are properly cached in `/localdisk/bowen/singing_identity/`.
2. **Implementation**: Build/modify a Marimo notebook (`@app.cell`) to test the Timbre Shrinkage hypothesis or conduct subspace removal/projection tests.
3. **Execution**: Run the notebook headlessly in `tmux` for long runs, ensuring all intermediate outputs write to `/localdisk/bowen/singing_identity/runs/`.
4. **Wrap-up**: Once finished, use `cpz` to copy the final results, graphs, and dashboard outputs back to the NFS `results` directory.
