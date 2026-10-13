# Batch 3 (5 submissions, the day after batch 2)

All probes on r14 (each expected ≈0.0636). If batch 2 reveals new signal, rerun
`python solution.py --probes 5` first: it rebuilds this batch on the improved base.

| file | technique |
|---|---|
| `p14_05_fire_vs_expected.csv` | Fire-station evidence vs what the tract's size predicts (the facilities half of the score). |
| `p14_06_log_route_length.csv` | Feature engineering: total route length. |
| `p14_07_route_ratio_maricopa-az.csv` | Road ratio, Maricopa only: is the measured road effect stronger in the desert metro? |
| `p14_08_multiref_share.csv` | Feature engineering: share of roads carrying several route numbers. |
| `p14_09_route_ratio_northern-ca.csv` | Road ratio, Northern California only (wildfire corridor). |
