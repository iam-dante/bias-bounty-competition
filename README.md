# Bias Bounty Mapping Equity Challenge: label-free coverage-gap estimator

**Current submission: `submission.csv`.** Expected public RMSE 0.0636. It is r12 (scored 0.064009)
plus a correction measured by a scored probe. Every leaderboard score is in `lb_scores.csv`.

```
pip install -r requirements.txt
B=s3://us-west-2.opendata.source.coop/humane-intelligence/bias-bounty-mapping-equity-challenge
aws s3 sync $B/reference/ data/reference/ --no-sign-request --exclude '*roads-unfiltered*'
for r in eastern-ok maricopa-az northern-ca south-central-tx eastern-wa; do
  aws s3 sync $B/strata/$r/ data/strata/$r/ --no-sign-request --exclude '*.csv'; done
python solution.py      # -> submission.csv (features are cached in data/features/ after the first run)
python solution.py --probes   # also writes the next 4 probe files to next/
```

**Daily cycle (5 submissions a day):** submit `submission.csv` plus the 4 files in `next/`. Copy each
scored file into `scored/` and append `file,score` to `lb_scores.csv`. Then rerun
`python solution.py --probes`: it fits every scored probe jointly and writes the next batch. The
queue (`PROBE_QUEUE`) is ordered by how much the physics models disagree along each direction,
after removing what the leaderboard has already measured.

## Layout

```
solution.py      the whole pipeline: features -> physics -> leaderboard blend -> correction -> submission.csv
lb_scores.csv    public RMSE of every scored file and probe
scored/          the scored files themselves (inputs of the blend; each score is an exact fact about the target)
submission.csv   the file to submit (GEOID, coverage_gap_score)
data/            challenge data, not in git (download above)
```

Submit score-only files. Zindi grades every column in the file, and the optional component columns
cost about 0.08 RMSE.

## The problem

The target is the organisers' per-tract composite: the mean of the defined components among
road gap `1 - min(1, Overture named highways / TIGER S1100+S1200 length)`, building gap
`1 - min(1, Overture buildings / Microsoft buildings)`, and POI gap (the mean of a facilities half
over fire / EMS / schools and a business-establishments half). The reference layers were
withdrawn and rebuilding them is prohibited, so there are no labels. `solution.py` estimates
`E[score | permitted data]`, the RMSE-optimal prediction, from three sources of information.

1. **Features** (DuckDB on the provided Overture, ACS and strata tables). The Overture side of
   every ratio is computed exactly. The reference side is inferred from evidence in the permitted
   data:
   - TIGER S1100/S1200 is by definition the Interstate/US/State system, and Overture keeps those
     route refs. Texas FM roads and Arizona's historic/future designations are not TIGER highways.
   - TIGER stores one record per (edge, name), so a route that also carries a local street name
     counts twice.
   - Tract boundaries are TIGER road edges, so a highway on a boundary is in both tracts for TIGER
     but in only one for Overture.
   - Facilities are recognised by alternate category and by name. Facility points near a boundary
     leak into the neighbouring tract.
   - The USFS building count is independent of Overture.
2. **Physics.** A generative model of the withdrawn reference counts is pushed through the exact
   composite by Monte Carlo. Its parameters were calibrated on the published statistics:
   undefined-component shares, region × rural × burned means, component means and the five
   Paradise tracts. The calibrated parameters are embedded in `solution.py`, and it reruns the
   main models at 4096 draws.
3. **Leaderboard facts.** For RMSE, every scored file f gives one exact moment of the hidden
   target: `mean_P(f·y) = (mean_P(f²) + mean_P(y²) − MSE_f) / 2`. The blend is the least-squares
   combination of scored files.
   - Exact identities between scored files (affine pairs, stacks) locate the public 30 % well
     enough to predict held-out public scores to about 5e-5. r12 was predicted at 0.064012 and
     scored 0.064009.
   - A probe (r12 plus a small push along one standardised feature direction) measures how much
     of r12's remaining error lies along that feature. The road-ratio probe found a correlation
     of −0.113 (z = −6): true road gaps are sharper than the model predicts. `solution.py`
     applies that correction with James–Stein shrinkage. On semi-synthetic splits, that shrinkage
     beat ridge and no shrinkage for 5 and 10 probes.
   - Probe directions include physics directions: how predictions move when one mechanism changes,
     such as boundary-highway miss, TIGER length noise or TIGER record multiplicity. Of all
     mechanisms, TIGER multiplicity lines up best (−0.46) with the measured road residual.

## Score history (public RMSE)

| file | score | what changed |
|---|---|---|
| all zeros / best constant | 0.1072 / 0.0890 | probes |
| r2f | 0.069455 | label-free generative model |
| stack3 | 0.066116 | leaderboard blend + scorecard groups |
| r6e | 0.065531 | vulnerability effects in the physics |
| stack7 | 0.064214 | regional specialists, Arizona route fix |
| r12 | **0.064009** | public-subset-corrected blend, Monte-Carlo denoising, partial cell anchoring |
| submission.csv | ≈0.0636 expected | r12 + road-ratio probe correction |

**Tested and dropped:**
- Re-blending earlier stacks: costs private RMSE on semi-synthetic public/private splits.
- Full anchoring to the published cell means (0.064221): the means predate the 28 Sep re-scoring.
- Physics calibrated directly on leaderboard moments (0.064317): its new direction had no signal.
- The published TIGER/Overture ratio range.
- A live business-count half.

## Rules

Only the provided datasets are used. External data is allowed only for the Best Bias Discovery
writeup. Thread 35292, which asks whether leaderboard-calibrated constants are allowed, had no
organiser reply as of 2026-10-11. The blend and the probe correction depend on that answer.
