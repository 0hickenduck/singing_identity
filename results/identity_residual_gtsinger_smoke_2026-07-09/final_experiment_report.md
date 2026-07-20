# Final Experiment Report

This is a compact interim report for the strict residual suite run.

## Data Actually Used
- dataset: `GTSinger`
- manifest: `/localdisk/bowen/singing_identity/runs/stage1_repaired_200_fresh_local/manifests/gtsinger_utterances.jsonl`
- feature root: `/localdisk/bowen/singing_identity/features/stage1_repaired_200_fresh_local`

## Speaker Splits Actually Used
- split seeds: `13`
- protocol: `10_0_10_speaker_disjoint`
- max speech per speaker: `20`
- duration balanced: `False`

## Outputs
- manifest audit: `/home/bowen/bowen_lab/projects/singing_identity/results/identity_residual_gtsinger_smoke_2026-07-09/manifest_audit`
- Experiment A: `/home/bowen/bowen_lab/projects/singing_identity/results/identity_residual_gtsinger_smoke_2026-07-09/expA_residualization_audit`
- Experiment B0/B1: `/home/bowen/bowen_lab/projects/singing_identity/results/identity_residual_gtsinger_smoke_2026-07-09/expB_global_and_reliability`

## Current Claim
No final scientific claim is supported by this driver alone until synthetic gate and full control criteria are reviewed.

## Exact Next Experiment Recommended
Review Exp A controls. If random/shuffled controls do not explain true residualization gains, expand to full seeds/layers and bootstrap CIs; otherwise audit geometry and mean-preserving behavior before mapper work.
