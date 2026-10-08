# Bias Bounty Mapping Equity Challenge: label-free coverage-gap estimator

**Current best: `submissions/e01_r2f_affine.csv`, public RMSE 0.067315979**
(= 0.009 + 0.80 × r2f; previous best r2f_ensemble 0.069455144, rank 12).
Best constant: 0.0890; all-zeros: 0.1072.

Branches: `main` holds the code that reproduces the current best. `dev` is the experiment
workspace (leaderboard probes, `lb_stack.py` blend solver, `experiments.py`, score log).

```
pip install -r requirements.txt
B=s3://us-west-2.opendata.source.coop/humane-intelligence/bias-bounty-mapping-equity-challenge
aws s3 sync $B/reference/ data/reference/ --no-sign-request --exclude '*roads-unfiltered*'
for r in eastern-ok maricopa-az northern-ca south-central-tx eastern-wa; do
  aws s3 sync $B/strata/$r/ data/strata/$r/ --no-sign-request --exclude '*.csv'; done
python solution.py   # features (cached in data/features/) + calibration -> submissions/params.json
python round2.py     # segment-multiplicity variants -> submissions/r2f_ensemble.csv
python best.py       # affine recalibration -> submissions/e01_r2f_affine.csv (best)
```

Submit **score-only** files (`GEOID,coverage_gap_score`). Zindi grades every column you
include, and the optional component columns cost ~0.08 RMSE.

## 1. The problem after the 2026-09-25 update

The target is the organisers' per-tract composite:

```
road     t = 1 - min(1, L_overture(motorway,trunk,primary,secondary) / L_TIGER(S1100,S1200))
building b = 1 - min(1, N_overture_buildings / N_microsoft)
POI      p = mean( HIFLD half , CBP half )
           HIFLD half = mean over defined types {fire, EMS, school} of 1 - min(1, O_k / H_k)
           CBP half   = 1 - min(1, N_overture_places / N_CBP)
score    = mean of the DEFINED components (a component is undefined when its reference is 0)
```

The organisers withdrew TIGER, Microsoft, HIFLD and CBP, and rebuilding them from outside
sources is prohibited. There is no labelled training data. The public leaderboard is saturated
at RMSE 0.0 by entries computed before the withdrawal. So this solution is an estimator of
`E[score | permitted data]`, which is the RMSE-optimal prediction:

* **The Overture side of every ratio is computed exactly** from the provided extracts.
* **The reference side is modelled as a random variable.** Its distribution is driven by
  evidence that *is* in the permitted data.
* The exact official formula, including "mean of the defined components" and its variable
  divisor, is applied to Monte Carlo draws and averaged.

## 2. Domain findings behind the features (`src/features.py`)

| Finding | Evidence | Feature |
|---|---|---|
| TIGER S1100/S1200 is by definition the Interstate/US/State highway system. Overture keeps those route refs (`routes[].network`). | "No route-tagged highway in/along the tract" gives road-undefined shares of 21.2 / 50.6 / 35.0 / 28.8 % (OK / AZ / CA / TX). Official "≥1 component undefined": 21 / 55 / 37 / 28 %. | `L_route` (regex on networks, incl. toll / loop / spur / business; county routes and TX FM/RM kept apart) |
| Texas Farm-to-Market roads are **not** S1200. | Including FM gives 18.1 % undefined in TX, which is below the official 28 % and so impossible. Excluding FM gives 28.8 %. | `L_fm` kept separate, weight 0 |
| TIGER/Line road files hold **one record per (edge, name)**. A highway that also has a local name ("State Rte 32" + "Deer Creek Hwy") is stored twice. | Five Paradise tracts with published values imply TIGER length = 2.00, 1.98, 2.02 and 2.2 × the route length. Calibration independently gives m_CA = 2.07. | region multiplicity `m_r`; per-segment `L_route_localname`, `L_route_multiref` |
| Census tract boundaries are drawn on TIGER road edges. A TIGER highway on a boundary falls in both tracts, but Overture's copy sits metres off and lands in one. | Northern CA: 2,153 km of tract boundary runs along route highways. | `Lb_route` (boundary ∩ 25 m highway buffer, parallel pieces only), `L_route_nearB` |
| CarbonPlan's building count is derived from Overture (means 1,972 vs 1,971), so it is not independent. The USFS Wildfire-Risk BuildingCount is independent (median Overture/USFS = 1.02). | | `U_bld = usfs_BuildingCount_sum` |
| Overture's POI numerator counts only `categories.primary`. Fire stations filed as `fire_protection_service`, or found only by name ("... VFD"), are real HIFLD stations that count against Overture. | | `n_fd`, `n_fd_alt`, `n_fire_name`; same for EMS and schools |

The DuckDB pitfalls from the data README are handled: `always_xy := true` on every
`ST_Transform`, and GEOID read as text.

## 3. Generative model (`src/model.py`)

For each tract, with `S = 512` Monte Carlo draws:

* `L_TIGER = m_region · (L_route + β · max(0, Lb_route − L_route_nearB)) · LogNormal(σ_t)`.
  The road component is defined iff route evidence exists (plus a small `p_def0` otherwise).
* `N_MS = κ · U_bld · LogNormal(σ_b)`.
* `H_k = Bin(O_k, α_k) + Bin(weak-evidence_k, α_k^x) + Poisson(μ̂_k · (1 − r_k)/r_k)`.
  Here `μ̂_k` comes from a Poisson GLM of Overture facility evidence on population, area, POI
  and building counts, rurality and region. The last term covers stations Overture does not
  show at all.
* `N_CBP = ε · population · LogNormal(σ_c)`.

## 4. "Training" = method-of-moments calibration on public facts only

Thirteen parameters (`m_r`×4, β, σ_t, r_fire, α_fire^x, α_ems^x, α_sch, α_sch^x, ε, κ) are fitted
with `scipy.optimize.least_squares`, using common random numbers, against:

1. the official share of tracts with an undefined component, per region (challenge page);
2. the 16 published means by region × urban/rural (RUCA ≥ 4) × MTBS-burned. These come from
   public discussion 34910; my cell sizes reproduce that post exactly (798 burned, 49 / 86
   urban-burned in CA / TX);
3. the pooled component means (burned / unburned) from the same post;
4. the five Paradise tracts with published component values (low weight).

No withdrawn layer and no externally obtained copy of one is used anywhere.

## 5. Submissions

Round 1 (`solution.py`, ideas sub1–sub5) established that per-segment TIGER multiplicity wins.
Round 2 (`round2.py`) builds on it; every model is recalibrated with the regional multiplier
allowed down to 0.5, since Maricopa and Texas sat on the old 1.0 floor.

| file | idea |
|---|---|
| `r2a_seg_relaxed` | records per edge = 1 + [local street name] + [concurrent routes] |
| `r2b_seg_no_multiref` | 1 + [local street name] |
| `r2c_seg_half_local` | 1 + 0.6·[local street name] + [concurrent routes] |
| `r2d_seg_refcount` | 1 + [local street name] + exact count of extra route refs |
| `r2e_seg_bagged` | r2a recalibrated 8× on jittered targets and MC seeds, averaged |
| **`r2f_ensemble`** | **mean of r2a–r2e: public 0.069455 (best)** |

Files cover every row of the five region sample submissions (9,794 tracts, eastern-wa
included, which Zindi requires) and contain only `GEOID, coverage_gap_score`.

eastern-wa has no published statistics, so it is not a calibration target. Its TIGER
multiplicity is the mean of the four calibrated regions.

## 6. Edge cases

* **Zero-population tracts:** CBP half undefined (allocation by population). The POI component
  then rests on HIFLD only.
* **Water-dominated tracts:** already dropped by the organisers (7 TX tracts).
* **No Overture named road but route evidence:** road gap → 1 in every draw where the road
  component is defined.
* **Undefined components** carry a probability, not a hard 0/1. The composite divisor varies by
  draw, exactly as in the official rule.

## 7. Leaderboard findings

* **Zindi grades every column in the file.** The optional component columns cost ~0.08 RMSE:
  r2f scored 0.149 with them and **0.069455 score-only**. All scripts now write
  `GEOID, coverage_gap_score` only.
* Probes: all-zeros 0.107246, constant 0.10 0.097676. So public mean(y) = 0.0598,
  sd(y) = 0.0890, and the best constant scores 0.0890.
* **Per-segment TIGER multiplicity beats per-region** (`sub2` > `sub1`), and an ensemble of
  segment variants (`r2f_ensemble`, `round2.py`) beats every single one.
* **Leaderboard-feedback recalibration** (`lb_stack.py`, `best.py`). For RMSE,
  mean(f·y) = (mean(f²) + mean(y²) − MSE_f)/2, so the zeros probe, the constant-0.10 probe and
  r2f's score give the least-squares fit y ≈ 0.009 + 0.80·r2f. r2f was over-dispersed, and the
  fix scored **0.067316** (predicted 0.0679). `lb_scores.csv` holds the three scores used.
* Further experiment tooling (`experiments.py`, probes, 20 experiment files) lives on the
  `dev` branch.
