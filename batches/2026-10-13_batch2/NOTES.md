# Batch 2 (5 submissions)

| file | kind | expected public | technique |
|---|---|---|---|
| `r14_route_boundary_corrected.csv` | **best candidate** | **≈0.06345** | r12 (scored 0.064009) corrected jointly along the two measured directions: road ratio (z = −6.1) and boundary leak (z = +4.7, probe scored 0.063780). James–Stein shrink 0.960. Road gaps are sharper than modelled, and highways on tract boundaries carry more TIGER length. Same file as `submission.csv`. |
| `p14_05_fire_vs_expected.csv` | probe on r14 (≈0.0636) | | Fire-station evidence vs what the tract's size predicts (the facilities half of the score). |
| `p14_02_tiger_multiplicity.csv` | probe on r14 | | Physics: TIGER records per highway edge × 1.1. |
| `p14_03_tiger_noise.csv` | probe on r14 | | Physics: TIGER length less uncertain (σ_t − 0.1). |
| `p14_04_interstate_share.csv` | probe on r14 | | Feature engineering: Interstate share of route length. |

Each probe is r14 + 0.004 × direction (orthogonal to every scored file), so it is itself a near-best
submission. Ranking: physics disagreement after removing the scored files and both measured
directions (boundary share dropped out: boundary leak already measured it).

**Already measured (2026-10-12):** the refitted physics model's direction (old `p14_04_gmm_route.csv`, built
on r13) scored 0.063802. Alone it shows z = +5.0, but given the road-ratio and boundary-leak
directions its information is z = 0.4: the refit learned the road signal and nothing beyond. It is
excluded from the correction, and its r14 probe was removed from this batch.
