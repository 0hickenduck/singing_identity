# Minematsu-Saito Lab: AI Agent Environment & Workflow Guidelines

Provide this document to your AI coding agents so they understand the filesystem rules, where to find NFS-optimized utilities, and how to execute the Marimo notebook workflow.

---

## 1. Fast Binaries & NFS-Optimized Commands (Already Set Up)

We have already configured your remote environment. Below is the reference of where these utilities are located and how they were sourced (in case you need to re-install or share them):

* **Target Directory**: Binaries are stored in your remote `~/.local/bin/` folder (which is on the NFS and shared across all `valkyrie` servers).
* **Installed Commands**:
  - **`rg`** (ripgrep) and **`dust`** (disk usage analyzer): Precompiled Linux binaries installed directly.
  - **`cpz`** and **`rmz`**: High-performance Rust-native commands from the open-source **`FUC` (Fast Unix Commands)** project (GitHub: `SUPERCILEX/fuc`).
* **Source/Restoration path**:
  - The compiled Rust binaries for `cpz` and `rmz` were copied from Steven's directory: `/home/smcintosh/.local/bin/cpz` and `/home/smcintosh/.local/bin/rmz`.
  - If they are ever lost, you can copy them again and run `chmod +x ~/.local/bin/cpz ~/.local/bin/rmz`.

---

## 2. Server Storage Guidelines (NFS vs. Local SSD)

Agents must place files according to the following rules to avoid performance bottlenecks:

### A. Network File System (NFS) — `/home/USER` (and `/work`)
* **Usage**: Storing scripts, git repositories, python virtual environments (`.venv`), and final model checkpoints.
* **Operation**: For copying or deleting directories on NFS, always use the fast Rust utilities **`cpz`** and **`rmz`** rather than standard `cp`/`rm` or `rsync`.

### B. Node Local SSD — `/localdisk/USER`
* **Usage**: Storing active datasets, intermediate run checkpoints, and cache files.
* **Model Cache**: Always configure HuggingFace to download to the local disk:
  ```bash
  export HF_HOME="/localdisk/USER/.cache/huggingface"
  ```
* **Warning**: Files here are local *only* to the active node. Copy final output models back to NFS (`/home/USER`) before shutting down or switching nodes.

---

## 3. Marimo Notebook & AI Workflow

We use **`marimo`** (a reactive Python notebook stored as a clean `.py` file) for interactive ML development and audio analysis.

### Two-Phase Experiment Strategy

Choose the right mode based on the experiment stage:

| Phase | Mode | How | Why |
| :--- | :--- | :--- | :--- |
| **Exploration & Debug** | Interactive Notebook | `uv run marimo edit notebook.py` | Rapid iteration: change a parameter, see the graph update, listen to audio samples — all without reloading data. |
| **Long/Overnight Training** | Headless Script | `uv run marimo run notebook.py` (inside `tmux`) | Runs all cells top-to-bottom like a regular script. Clean process lifecycle: starts, runs, releases all GPU memory when done. No risk of memory accumulation. |

### GPU Memory & Kernel Management (Critical for AI Agents)

> **Rule**: When the AI modifies model architecture, changes tensor shapes, or encounters CUDA OOM errors, it **MUST restart the Marimo kernel** before re-running. Simply re-executing the cell will create a second copy of the model in GPU memory, causing OOM crashes.

**AI agents must follow this protocol:**
1. **Minor parameter changes** (e.g., learning rate, batch size): Safe to just re-run the affected cell. No restart needed.
2. **Structural changes** (e.g., adding layers, changing model class, modifying tensor dimensions): **Restart the kernel first**, then re-run from the data loading cell onward.
3. **After encountering any CUDA OOM or memory error**: **Always restart the kernel** and free GPU memory before retrying.

To restart from the command line (useful for AI agents):
```bash
# Kill the running marimo process, then relaunch
pkill -f "marimo edit notebook.py"
uv run marimo edit notebook.py
```

### Viewing Experiment Results via Browser (Port Forwarding)

When the AI runs an experiment on a remote server and you want to view graphs, loss curves, or listen to audio samples in your browser:

1. **AI starts Marimo on the server** (e.g., on `valkyrie03`):
   ```bash
   uv run marimo edit notebook.py --port 2718
   ```
2. **You open an SSH tunnel from your local Mac terminal** (do NOT run this on the server):
   ```bash
   ssh -NL 2718:localhost:2718 valkyrie03
   ```
3. **Open your browser** at `http://localhost:2718` to see all graphs, audio players, and experiment outputs.

> **Tip for AI agents**: When the user asks to "see results" or "check the experiment", proactively provide the port forwarding command and remind the user to open their browser.

### Building a Results Dashboard for Long Experiments

For multi-stage experiments (e.g., Stage 1: feature extraction → Stage 2: training → Stage 3: synthesis), structure your Marimo notebook so that the **final cells act as a results dashboard**:

```python
@app.cell
def results_dashboard(train_losses, val_losses, generated_audio_paths):
    import marimo as mo
    import matplotlib.pyplot as plt

    # Loss curve
    fig, ax = plt.subplots()
    ax.plot(train_losses, label="Train")
    ax.plot(val_losses, label="Val")
    ax.legend()
    plt.title("Training Progress")

    # Audio players for generated samples
    audio_players = [mo.audio(path) for path in generated_audio_paths]

    mo.vstack([fig, *audio_players])
```

This way, after a long training run finishes, you can open the notebook in your browser and immediately see loss curves and listen to generated audio samples without writing any extra code.

### AI Instruction Prompts (Copy-paste to Agent)

* **To Create/Edit Notebooks**: 
  - *Prompt*: `"Create/Modify the Marimo notebook experiment.py. Group the code into cells using @app.cell decorators. Include a results dashboard cell at the end with loss curve plots and marimo audio components to preview generated wav samples."`
* **To Run Headlessly (in tmux)**:
  - *Prompt*: `"Run the Marimo notebook experiment.py headlessly in the background using 'uv run marimo run experiment.py' inside a tmux session on valkyrie02, and redirect output to train.log."`
* **To Debug (with kernel restart awareness)**:
  - *Prompt*: `"The run of experiment.py failed with a CUDA OOM error. Restart the marimo kernel first (kill the process and relaunch), then fix the model code in the relevant cell, and re-run from the data loading cell onward."`
* **To View Results**:
  - *Prompt*: `"Start marimo edit on the server for experiment.py and give me the SSH port forwarding command so I can view the results dashboard in my browser."`

---

## 4. Python Environment

* **Always use `uv`** for virtual environments and package management. Never use `conda`, `pip` directly, or `virtualenv`.
* **Per-project venvs**: Each project gets its own environment: `mkdir my-project && cd my-project && uv venv`.
* **Run Python through uv**: Always use `uv run python` instead of bare `python`.
* **PyTorch installation** (our CUDA drivers are old):
  ```bash
  uv pip install torch torchvision torchaudio torchcodec \
    --index-url https://download.pytorch.org/whl/cu128
  ```
  > **Important**: Install all torch packages in a single command. Installing them separately can cause version conflicts.
