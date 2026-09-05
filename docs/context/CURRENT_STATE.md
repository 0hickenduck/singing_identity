# Current State

Updated: 2026-07-01

- The project studies same-person speech-to-singing vocal identity shift and singing technique representation.
- Canonical source lives at `/home/bowen/bowen_lab/projects/singing_identity`.
- Runtime data, runs, features, checkpoints, and large logs live at `/localdisk/bowen/singing_identity`.
- Stage 1 repaired GTSinger/JVS-oriented feature and manifest work produced reusable manifests and feature caches under `/localdisk`.
- Recent work has focused on Track 1 prompt/mode mismatch accounting, SeedVC prompt/component checks, and Track 2 technique/breathy probes.
- SeedVC prompt/component experiments need careful interpretation: current evidence may reflect singing-domain, phonation, quality, or technique effects rather than clean identity transfer.
- Marimo notebooks are used for exploration and human review; final claims should be backed by scripts and compact reports.

Default next-agent behavior:

- Read this file, `OPEN_QUESTIONS.md`, and recent `EXPERIMENT_REGISTRY.md` entries before proposing a new run.
- Do not scan raw audio, feature caches, checkpoints, prediction dumps, or full logs unless the task specifically requires it.
- If a new experiment is run, record it in `EXPERIMENT_REGISTRY.md` before ending the session.
