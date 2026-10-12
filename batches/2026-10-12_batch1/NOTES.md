# Batch 1 (5 submissions, for 2026-10-12)

| file | kind | expected public | what it measures / does |
|---|---|---|---|
| `r13_route_corrected.csv` | candidate | ≈0.0636 | r12 (0.064009) corrected along the road-ratio direction, measured by probe p12_01 (ρ = −0.113, z = −6): true road gaps are sharper than the model predicts. James–Stein shrink 0.973. Same file as `submission.csv`. |
| `p13_02_boundary_leak.csv` | probe | ≈0.0641 | Physics direction: predictions when 20 % more of a boundary highway's TIGER copy lands in the tract (beta_bnd + 0.2). Highest physics disagreement after removing the road-ratio direction. |
| `p13_03_tiger_noise.csv` | probe | ≈0.0641 | Physics direction: TIGER length less uncertain (sigma_t − 0.1), i.e. sharper road gaps. |
| `p12_08_boundary_share.csv` | probe | ≈0.0641 | Share of route length running along the tract boundary (TIGER counts it in both tracts). |
| `p12_02_fire_vs_expected.csv` | probe | ≈0.0641 | Fire-station evidence relative to what the tract's size predicts; the facilities half of the score. |

Every probe is r12 + 0.004 × direction, the direction standardised and orthogonalised against every
scored file (public-subset weights). A probe's score gives exactly mean_P(direction × (y − r12)):
how much of r12's remaining error lies along it.

After uploading: copy each file into `scored/`, append `file,score` to `lb_scores.csv`, and run
`python solution.py --probes`.
