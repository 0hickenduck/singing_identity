# Final Experiment Report

This is a compact interim report for the strict residual suite run.

## Data Actually Used
- dataset: `GTSinger`
- manifest: `/localdisk/bowen/singing_identity/runs/stage1_repaired_200_fresh_local/manifests/gtsinger_utterances.jsonl`
- feature root: `/localdisk/bowen/singing_identity/features/stage1_repaired_200_fresh_local`

## Speaker Splits Actually Used
- split seeds: `13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,103,107,109,113,127,131,137,139,149,151,157,163,167,173,179,181,191,193,197,199,211,223,227,229,233,239,241,251,257`
- protocol: `10_0_10_speaker_disjoint`
- max speech per speaker: `20`
- duration balanced: `False`

## Outputs
- manifest audit: `/home/bowen/bowen_lab/projects/singing_identity/results/identity_residual_gtsinger_b1_2026-07-09/manifest_audit`
- Experiment A: `/home/bowen/bowen_lab/projects/singing_identity/results/identity_residual_gtsinger_b1_2026-07-09/expA_residualization_audit`
- Experiment B0/B1: `/home/bowen/bowen_lab/projects/singing_identity/results/identity_residual_gtsinger_b1_2026-07-09/expB_global_and_reliability`

## Current Claim
No final scientific claim is supported by this driver alone until synthetic gate and full control criteria are reviewed.

## Exact Next Experiment Recommended
Review Exp A controls. If random/shuffled controls do not explain true residualization gains, expand to full seeds/layers and bootstrap CIs; otherwise audit geometry and mean-preserving behavior before mapper work.
