# Singing Identity Context

This directory is the default human-readable entry point for AI agents.
It should stay small and curated.

Default reading order:

1. `/home/bowen/bowen_lab/LAB_AGENT_RULES.md`
2. `CURRENT_STATE.md`
3. `OPEN_QUESTIONS.md`
4. `EXPERIMENT_REGISTRY.md`
5. `DECISION_LOG.md`

Raw paper notes, free recalls, full logs, raw datasets, feature caches,
checkpoints, and prediction dumps are not part of the default context path.

Codex hooks read this directory through `.codex/project_context.json`.
The `SessionStart` hook injects a short brief from these files; the `Stop`
hook checks `EXPERIMENT_REGISTRY.md` when an experiment marker is present.

## Experiment Marker Command

Use this from scripts or wrappers after an experiment creates a run:

```bash
python scripts/mark_experiment.py \
  --run-name <run_name> \
  --run-root /localdisk/bowen/singing_identity/runs/<run_name> \
  --script scripts/<path>.py \
  --status finished
```

The marker is written under:

```text
/localdisk/bowen/singing_identity/status/codex_sessions/<session_id>/experiment_markers.jsonl
```

When Codex tries to stop, the hook requires a matching `run_name` or `run_root`
entry in `EXPERIMENT_REGISTRY.md`.
