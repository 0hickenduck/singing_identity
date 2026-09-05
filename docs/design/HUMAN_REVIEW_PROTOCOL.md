# Human Review Protocol

This project should not ask for subjective evaluation by pointing only to raw files.

When Codex needs human judgment, it should produce a review artifact first, preferably a Marimo notebook under `notebooks/`, with the generated audio, plots, metrics, and the specific decision requested.

## Required Review Artifact

For listening or visual inspection checkpoints, provide:

1. A Marimo notebook path.
2. The command to open it.
3. The run root containing the underlying artifacts.
4. A concise decision menu, such as `continue`, `rerun`, or `stop`.
5. The exact evidence to check, such as pair numbers where one condition wins or fails.

Use static HTML only as a fallback or export artifact. The default interactive surface is Marimo.

## Launch / Stop Workflow

Server-side launch is automated through:

```bash
scripts/review/start_review_servers.sh PORT:NOTEBOOK [PORT:NOTEBOOK ...]
```

Example:

```bash
scripts/review/start_review_servers.sh \
  2719:notebooks/human_check_track1_seedvc_track2_2026_06_28.py \
  2720:notebooks/track1_seedvc_prompt_review.py \
  2721:notebooks/track1_seedvc_component_ablation_review.py
```

The start script kills existing Marimo listeners on the requested ports, starts
each notebook in tmux, exports `UV_CACHE_DIR=/localdisk/bowen/.cache/uv`, waits
for ports to listen, and prints a summary.

After the reviewer gives feedback in the conversation, stop the servers with:

```bash
scripts/review/stop_review_servers.sh
```

The reviewer opens the local SSH tunnel. A PowerShell helper is kept in the repo:

```text
scripts/review/open_lab_tunnel.ps1
```

Recommended local install path:

```text
C:\Users\libowen\bin\open_lab_tunnel.ps1
```

Local usage:

```powershell
powershell -File C:\Users\libowen\bin\open_lab_tunnel.ps1 -Ports 2719,2720,2721
```

Codex only starts/stops server-side Marimo sessions; it does not manage the
reviewer's Windows tunnel.

## Current Human Checkpoint

Track 1 / SeedVC / Breathy checkpoint:

```bash
scripts/review/start_review_servers.sh \
  2719:notebooks/human_check_track1_seedvc_track2_2026_06_28.py \
  2720:notebooks/track1_seedvc_prompt_review.py \
  2721:notebooks/track1_seedvc_component_ablation_review.py
```

Dedicated listening notebooks:

Prompt review uses port `2720`; component review uses port `2721`.

Run roots:

```text
/localdisk/bowen/singing_identity/runs/track1_seedvc_prompt_baseline_30pairs
/localdisk/bowen/singing_identity/runs/track1_seedvc_component_ablation_12pairs
```

SSH tunnel:

```bash
powershell -File C:\Users\libowen\bin\open_lab_tunnel.ps1 -Ports 2719,2720,2721
```

Decision needed:

- `continue`: singing-prompt or mel/style outputs are audibly better as target identity often enough to justify deeper SeedVC integration.
- `rerun`: audio quality, loudness, pair balance, or diffusion settings make the run inconclusive.
- `stop`: prompt-mode effect is not audible as useful identity signal.

Useful notes:

- pair numbers with artifacts,
- pair numbers where speech prompt wins,
- whether the change sounds like identity, technique/phonation, singing-domain, loudness/quality, or melody/content.
