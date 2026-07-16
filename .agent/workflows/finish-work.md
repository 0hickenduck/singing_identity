# Finish Work

Wrap up the current session: archive the active task and record the session
journal. Code commits are done before this workflow in `.trellis/workflow.md`
Phase 3.4.

## Step 1: Survey Current State

```bash
python3 ./.trellis/scripts/get_context.py --mode record
```

This prints:

- **My active tasks** - review whether any besides the current one are actually
  done and should be archived this round.
- **Git status** - quick visual on what's dirty.
- **Recent commits** - use work commit hashes in Step 4.

If `--mode record` surfaces other completed tasks not tied to the current
session, ask the user once whether to archive them too. Default is no; the
current active task is always archived in Step 3 when complete.

## Step 2: Sanity Check Dirty Paths

Run:

```bash
git status --porcelain
```

Filter out paths under `.trellis/workspace/` and `.trellis/tasks/`; those are
managed by `add_session.py` and `task.py archive`.

For each remaining dirty path, decide whether it belongs to the current task or
to other parallel work. If any remaining path is current-task work, return to
Phase 3.4 and commit it before running this workflow. If all remaining paths are
unrelated, report them once and continue.

## Step 3: Archive Task

```bash
python3 ./.trellis/scripts/task.py archive <task-name>
```

At minimum, archive the current active task if it is complete. The script
creates a `chore(task): archive ...` commit when session auto-commit is enabled.

## Step 4: Record Session Journal

```bash
python3 ./.trellis/scripts/add_session.py \
  --title "Session Title" \
  --commit "hash1,hash2" \
  --summary "Brief summary"
```

Use the work commit hashes produced in Phase 3.4. Do not include archive commit
hashes from Step 3.
