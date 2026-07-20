# Identity Residual Synthetic Gate

- seed: `13`
- speakers: `48`
- passed: `true`
- split: speaker-disjoint, train speakers only for scalers and residualizers

## Case Verdicts

- `no_nuisance_mode_dominance`: PASS; raw=1.000, mode=1.000, full_mean=1.000, full_classic=1.000, random=1.000
- `mode_nuisance_dominance`: PASS; raw=0.500, mode=0.647, full_mean=0.647, full_classic=0.618, random=0.412
- `mode_proxy`: PASS; raw=0.265, mode=0.265, full_mean=1.000, full_classic=1.000, random=0.176
- `within_mode_nuisance`: PASS; raw=0.382, mode=0.353, full_mean=1.000, full_classic=1.000, random=0.382
- `random_nuisance`: PASS; raw=1.000, mode=1.000, full_mean=1.000, full_classic=1.000, random=1.000
- `leakage_trap`: PASS; raw=0.412, mode=0.412, full_mean=0.412, full_classic=0.412, random=0.412

## Leakage Trap

The leakage case puts the nuisance-to-embedding relationship only in held-out test speakers. Because the design scaler and OLS residualizer are fit on train speaker utterances only, the full nuisance variants are expected not to improve over raw by more than +0.20 absolute R@1.
