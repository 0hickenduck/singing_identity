# Identity Residual Protocol Robustness

## Scope

This suite reevaluates the train-estimated global displacement with verification trials, limited speech references, held-out residual geometry, speaker-level uncertainty, and gallery-size changes. All corrected test queries use only a vector estimated from train speakers.

## Threshold Policy

Genuine trials are same-speaker speech--singing centroid pairs; impostors are all cross-speaker pairs. Train speakers never occur in test trials. EER is reported as a descriptive held-out ROC crossing. TMR operating thresholds are determined from train-speaker impostor trials only; 0.1% FMR is marked insufficient when fewer than 1,000 train impostors exist.

## Low-rank Decision

Only the residual spectrum is reported. Applying train PCA directions to a held-out speaker requires a pre-specified test-speech-only rule for the coefficients; the available alternative is to fit a predictor, which would restart the individualized mapper line that this project has stopped. No target-singing or oracle projection is used.

## Output Map

- `verification_summary.csv` and `verification_paired_split_deltas.csv`: cross-mode verification and controls.
- `reference_budget_summary.csv`: one, two, five, ten, and full speech-reference budgets.
- `heldout_alignment_summary.csv` and `global_mean_explained_energy_summary.csv`: alignment, variance, and residual-spectrum accounting.
- `speaker_level_uncertainty.csv`: speaker-aggregated bootstrap and paired sign-permutation result.
- `gallery_size_summary.csv`: sampled held-out gallery robustness. A requested gallery size is omitted when the fixed split has too few held-out speakers.
