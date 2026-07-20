# Final Experiment Report

This is a compact interim report for the strict residual suite run.

## Data Actually Used
- dataset: `JVS_JVSMuSiC`
- manifest: `/localdisk/bowen/singing_identity/runs/jvs_music_retrieval_2026-07-08/manifests/jvs_music_utterances.jsonl`
- feature root: `/localdisk/bowen/singing_identity/features/jvs_music_2026-07-08`

## Speaker Splits Actually Used
- split seeds: `13`
- protocol: 60/20/20 speaker-disjoint for minimal JVS audit.
- max speech per speaker: `20`
- duration balanced: `True`

## Outputs
- manifest audit: `/home/bowen/bowen_lab/projects/singing_identity/results/identity_residual_suite_duration_smoke_2026-07-09/manifest_audit`
- Experiment A: `/home/bowen/bowen_lab/projects/singing_identity/results/identity_residual_suite_duration_smoke_2026-07-09/expA_residualization_audit`
- Experiment B0/B1: `/home/bowen/bowen_lab/projects/singing_identity/results/identity_residual_suite_duration_smoke_2026-07-09/expB_global_and_reliability`

## Current Claim
No final scientific claim is supported by this driver alone until synthetic gate and full control criteria are reviewed.

## Exact Next Experiment Recommended
Review Exp A controls. If random/shuffled controls do not explain true residualization gains, expand to full seeds/layers and bootstrap CIs; otherwise audit geometry and mean-preserving behavior before mapper work.
