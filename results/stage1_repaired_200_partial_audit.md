# Stage 1 Audit

- Run root: `/home/bowen/bowen_lab/projects/singing_identity/results/stage1_repaired_200_allmodels_run`
- Job counts: `{"running": 1, "success": 41, "timeout": 2}`

## Jobs To Check

- `track1_mode_controlled_wavlm_l12` status=`running` log=`/home/bowen/bowen_lab/projects/singing_identity/results/stage1_repaired_200_allmodels_run/jobs/track1_mode_controlled_wavlm_l12/run.log`
- `track1_mode_raw_wavlm_l12` status=`timeout` log=`/home/bowen/bowen_lab/projects/singing_identity/results/stage1_repaired_200_allmodels_run/jobs/track1_mode_raw_wavlm_l12/run.log`
- `track1_mode_shuffled_wavlm_l6` status=`timeout` log=`/home/bowen/bowen_lab/projects/singing_identity/results/stage1_repaired_200_allmodels_run/jobs/track1_mode_shuffled_wavlm_l6/run.log`

## Track 1

| model | layer | raw_sep | controlled_sep | shuffled_sep | R@1 | R@5 | mapper | global |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| acoustic | frame25ms_hop20ms | 0.912 | 0.912 | 0.507 | 0.350 | 0.600 | 0.617 | 0.629 |
| wavlm | 3 | 0.999 | 1.000 | 0.507 | 0.150 | 0.400 | 0.782 | 0.723 |
| wavlm | 6 | 1.000 | 1.000 | nan | 0.050 | 0.450 | 0.790 | 0.738 |
| wavlm | 9 | 0.999 | 1.000 | 0.514 | 0.100 | 0.550 | 0.797 | 0.731 |

## Track 2

No completed Track 2 metrics found.
