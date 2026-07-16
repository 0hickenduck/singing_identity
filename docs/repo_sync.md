# Repository Sync

This repository is synchronized through the private GitHub remote:

```bash
git remote -v
# origin  https://github.com/0hickenduck/arti6_linearvc.git
```

## Code Sync

Use normal git for code, docs, task notes, manifests, and small reproducibility metadata:

```bash
git pull --rebase origin master
git status
git push origin master
```

Before switching machines, commit and push code changes. After switching machines, pull before editing.

The repository is organized as a workspace with separate Trellis package lanes:

* `vtuber_pipeline/` - VTuber speech/singing data curation.
* `arti6_linearvc_demo/` - ARTI-6 + LinearVC demo.
* `.trellis/`, `.agents/`, `.agent/` - Trellis harness and AI-platform
  integration.

Antigravity support lives in `.agent/`:

* `.agent/workflows/` contains the Trellis start/continue/finish workflows.
* `.agent/skills/` contains auto-triggered Trellis skills.
* `.trellis/scripts/common/cli_adapter.py` maps Antigravity non-interactive
  execution to `agy --print` and resume-by-id to `agy --conversation`.

## What Stays Out Of Git

The following should stay out of normal git:

* `data/`
* `outputs/`
* `.venv/`
* downloaded YouTube audio/video
* separated stems and temporary Demucs output
* generated training datasets
* model checkpoints and Hugging Face caches
* cookies and local secrets
* generated archives

Small demo audio that is intentionally part of a report can be force-added after review, but training data should move through out-of-band transfer.

## Data Packaging Policy

Keep generated audio datasets as expanded directories by default. On the lab
server, expanded directories are easier to preview and audit than `.tar.gz` or
`.zip` files.

Default:

* keep `data/vtuber_curated_*` as normal folders;
* keep manifests beside the audio they describe;
* use `rsync` for local/server transfer.

Only create archives when there is a specific reason:

* temporary transfer packaging when direct folder sync is inconvenient;
* portable HTML demo/report bundles that need `index.html` plus linked assets in
  one downloadable unit;
* user explicitly asks for an archive.

Archives made only for transfer should be treated as disposable local artifacts,
not as canonical dataset outputs.

## External Dependencies

`external/seed-vc` is a submodule:

```bash
git submodule update --init external/seed-vc
```

The working tree may also contain large local files inside that checkout. Those are local artifacts and should not be committed to this repository.

## Large Data Sync

Use `scripts/sync-large-data.sh` for local/server data transfer. It syncs
expanded folders with `rsync` and does not delete remote files unless explicitly
requested.

Push local generated data to the lab server:

```bash
scripts/sync-large-data.sh push bowen@valkyrie08:/home/bowen/bowen_lab/projects/arti6_linearvc
```

Pull server data back locally:

```bash
scripts/sync-large-data.sh pull bowen@valkyrie08:/home/bowen/bowen_lab/projects/arti6_linearvc
```

Dry run first:

```bash
DRY_RUN=1 scripts/sync-large-data.sh push bowen@valkyrie08:/home/bowen/bowen_lab/projects/arti6_linearvc
```

Customize synced paths with `SYNC_PATHS`, separated by spaces:

```bash
SYNC_PATHS="data/vtuber_curated_conservative_20260524_run2 data/manifests" \
  scripts/sync-large-data.sh push bowen@valkyrie08:/home/bowen/bowen_lab/projects/arti6_linearvc
```
