# Legacy Scripts Archive

This directory archives deprecated and historical development scripts from earlier research iterations.
Canonical project operations have been consolidated into dedicated entry points under `scripts/`.

## Canonical Replacements

| Legacy Script / Directory | Purpose | New Canonical Replacement |
|---|---|---|
| `legacy/data_prep/` | Manifest building and feature extraction | `scripts/data/prepare_manifests.py` |
| `legacy/extract_features.py` | Standalone feature cache extraction | Integrated into `scripts/data/prepare_manifests.py --extract-features` |
| `legacy/probing/` | Probing and verification runners | `scripts/run/run_stage1.py` & `scripts/evaluate/evaluate_verification.py` |
| `legacy/intervention/` | Latent steering & ablation experiments | `src/singing_identity/methods/` package |
| `legacy/research_utils.py` | Shared utility library | `src/singing_identity/utils/research_utils.py` |
| `legacy/run_stage1_overnight.py` | Stage 1 overnight runner | `scripts/run/run_stage1.py` |
| `legacy/run_targeted_repair_then_stage1.py` | Targeted repair + stage1 pipeline | `scripts/run/run_stage1.py` |
| `legacy/audit_stage1.py` | Stage 1 audit | `scripts/summarize/summarize_results.py --validate` |
| `legacy/validate_run_provenance.py` | Provenance validator | `scripts/summarize/summarize_results.py --validate` |
| `legacy/*.sh` | Machine-specific / ad-hoc shell wrappers | Run directly via `uv run` per `README.md` |
