# Decision Log

## 2026-07-01: Keep Context Small And Hook Enforcement Objective

- decision: Use `context/` as the curated agent entry point, not raw logs or raw free recall.
- reason: Full conversations and raw notes have low signal density and consume context quickly.
- consequence: `SessionStart` injects only current state, open questions, latest session summary if present, and recent registry entries.

## 2026-07-01: Use Marker Files For Experiment Registry Enforcement

- decision: Experiment registry enforcement is based on explicit marker files, not transcript grep.
- reason: Transcript formats are not a stable hook interface, and keyword guessing is fragile.
- consequence: Scripts/wrappers should call `/home/bowen/bowen_lab/codex_hooks/mark_experiment.py` after meaningful runs.

## 2026-07-01: Keep Sync Outside Hooks

- decision: Obsidian/server bidirectional sync remains an explicit command or git workflow, not a Codex hook.
- reason: Sync has conflict semantics and needs human control.
- consequence: Hooks may remind or enforce local registry completion, but they do not push/pull notes automatically.
