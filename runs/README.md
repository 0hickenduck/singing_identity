# Benchmark Runs

This directory defines the benchmark run output structure for reproducible executions.

---

## Benchmark Run Directory Standard

Each run of an experiment approach should be isolated in its own semantic path:

```text
runs/
└── <experiment_name>/
    └── <approach_name>/
        └── <run_id>/
            ├── config.json       # Exact resolved configuration for reproduction
            ├── metadata.json     # Git commit hash, timestamp, hostname, environment
            ├── metrics.json      # Final computed evaluation metrics
            ├── logs/             # Execution and stderr/stdout logs
            └── artifacts/        # Small lightweight checkpoints or predictions (<2 MB)
```

---

## Heavy Output Offloading (Lab Convention)

Per the lab conventions:
> *Never create data/, runs/, features/, outputs/ inside a repo or worktree.*  
> *Heavy data/outputs: `/localdisk/bowen/singing_identity/...`*

Heavy feature caches (`.npz` files, large audio outputs, or model checkpoint weights) must be written directly to `/localdisk/bowen/singing_identity/runs/<run_id>/` or `/localdisk/bowen/singing_identity/features/`.
