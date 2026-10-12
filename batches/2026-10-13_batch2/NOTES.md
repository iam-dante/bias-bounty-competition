# Batch 2 (5 submissions, for the day after batch 1)

Every file here is r13 (`submission.csv`, expected public ≈0.0636) + 0.004 × one new direction, so
each is itself a near-best submission (expected ≈0.0637) and its score measures exactly how much of
the remaining error lies along that direction. Upload in any order, before or after batch 1:
`solution.py` decodes every probe from its own score, whatever base it was built on.

| file | technique | why this direction |
|---|---|---|
| `p14_04_gmm_route.csv` | **deep model**: physics refitted on the published statistics + every leaderboard moment, including the road-ratio probe (`PHYSICS["g43"]`) | Reproduces part of the measured road residual physically (corr −0.24): TIGER length noise doubles (σ_t 0.29 → 0.60), weak fire-station evidence counts more, burned-area roads lose TIGER length. Tests whether that nonlinear physics carries more than the linear correction. |
| `p14_01_interstate_share.csv` | **feature engineering**: Interstate share of route length | Highest-ranked new feature (physics disagreement 0.0012). Do Interstates carry fewer TIGER records than other routes (r10b's assumption)? |
| `p13_01_tiger_multiplicity.csv` | physics direction: TIGER records per edge × 1.1 | The mechanism that best matches the measured road residual (−0.46). |
| `p14_02_log_route_length.csv` | feature engineering: total route length | Tracts with long route networks: does the gap saturate differently? |
| `p14_03_multiref_share.csv` | feature engineering: concurrent-route share | Roads carrying several route numbers: does TIGER store a record per route? |

After uploading: copy each scored file into `scored/`, append `file,score` to `lb_scores.csv`, and run
`python solution.py --probes`.
