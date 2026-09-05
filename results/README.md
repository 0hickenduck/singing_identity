# Benchmark Results & Evaluations

This directory houses comparison-level evaluation results, benchmark metrics, and experiment summary reports.

---

## Directory Organization

```text
results/
├── README.md                      # This overview
├── index.md                       # Canonical historical experiment run index
├── summary.csv                    # Consolidated benchmark comparison metrics across methods
├── summary_provenance.json        # Machine-readable provenance mapping linking each row to run artifacts
├── reports/                       # Consolidated markdown reports from key experiment stages
├── track1_timbre/                 # Track 1: Speech-vs-Singing Timbre & Identity verification
└── track2_technique/              # Track 2: Vocal Technique disentanglement and probing
```

---

## Canonical Reports (`results/reports/`)

- `identity_residual_gate_report_2026-07-09.md`: Cross-modal residual verification gate verdict.
- `stage1_repaired_200_fresh_local_report.md`: Stage 1 200-speaker GTSinger benchmark validation.
- `paper_related_work_survey_2026-07-16.md`: Literature comparison and positioning vs KunquDB.
- `identity_residual_paper_closure_2026-07-15/`: Final paper closure validation runs and audit matrices.
