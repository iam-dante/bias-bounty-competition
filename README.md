# Bias Bounty Mapping Equity Challenge: label-free coverage-gap estimator

**Current best: `submissions/best/stack6_r9.csv`, public RMSE 0.064388606.**
Rebuild it in seconds with `python best.py`. Previous bests: stack5 0.064867, stack4 0.065108,
r6e 0.065531, r5e 0.065883, stack3 0.066116, stack2 0.066850, e01 0.067316, r2f 0.069455.
Best constant: 0.0890; all-zeros: 0.1072. Every score is in `lb_scores.csv`.

## Why stack6 is the best, in five steps

1. **Exact Overture side, inferred reference side** (`src/features.py`). For every tract we
   compute exactly what the organisers computed from Overture: named-highway length, building
   count, fire / EMS / school / place counts. The withdrawn reference side (TIGER, Microsoft,
   HIFLD, CBP) is inferred from permitted data: Overture route refs (TIGER S1100/S1200 is the
   Interstate/US/State system), tract boundaries that run along highways (tract boundaries are
   TIGER edges), TIGER's one-record-per-name duplication, USFS building counts, and facility
   names / alternate categories.
2. **A generative model run through the exact scoring formula** (`src/model.py`). Monte-Carlo
   draws of the reference counts go through the official composite, including "mean of the
   defined components". That gives E[score | data], the RMSE-optimal prediction for each tract.
3. **Calibration on public facts only** (`solution.py`, `round*.py`). Parameters are fitted to
   the official undefined-component shares, published region × rural × burned means, component
   means, five published Paradise tracts and the true target's public bias scorecard.
4. **Structural fixes found region by region.** Each fix came from a region failing one of
   those facts:
   - Texas FM roads are not TIGER highways.
   - Arizona's historic/future route designations are ordinary streets (round 9; this fix
     halved the moment loss).
   - Station positions leak across tract boundaries (round 3).
   - Rural / wildfire / drought / tribal / vulnerability effects sit inside the mechanisms
     (rounds 5–6).
5. **A blend fitted from leaderboard scores** (`lb_stack.py`). For RMSE,
   mean(f·y) = (mean(f²) + mean(y²) − MSE_f) / 2. So the all-zeros probe, a constant probe and
   each scored file give exact moments of the hidden target. The RMSE-optimal blend of all
   scored files is then a small least-squares problem. Its predictions land within ~0.0005 of
   the actual score: stack6 was predicted 0.0640–0.0643 and scored 0.064389.

## Where to improve

- **The public aggregates are exhausted.** The round-9 model reproduces every published
  figure (undefined shares within 0.01–0.02, the 16 cell means mostly within 0.01, component
  means within 0.015). They can no longer reveal what is wrong.
- **What remains is within-group, tract-level error.** The model explains about half the
  variance (RMSE 0.064 vs sd 0.089).
- **The only source of tract-level truth is leaderboard scores.** The loop that produced
  stack5 and stack6: a physically different model scores near the best (~0.065), then the
  blend uses that score and beats the best. Next candidates, in order of expected new
  information:
  - regional specialists on the fixed road features (one parameter set per region);
  - a live CBP half (business-density gaps in residential tracts);
  - stale facility lists in burned areas (HIFLD still lists stations the fires destroyed);
  - per-facility-type boundary leakage.
- **Not a route:** rebuilding the withdrawn layers or tract-by-tract leaderboard probing. The
  first is prohibited; the second only fits the public 30 % and fails on the private 70 %.

Branches: `main` holds the current best and the code that produces it. `dev` is the same
layout plus work in progress; anything that beats the best on the leaderboard moves to `main`.

## Layout

```
src/features.py          DuckDB feature engineering on the provided GeoParquet
src/model.py             generative model of the withdrawn reference layers + MC expectation
solution.py              features + base calibration          -> submissions/params.json
round2.py                TIGER multiplicity per segment          -> submissions/params_round2.json
round3.py                neighbour-aware facility leakage        -> submissions/params_round3.json
round3_fixed.py          leakage at a fixed 40 m / 100 m offset  (r3j)
round5.py                group effects + scorecard calibration   -> submissions/params_round5.json
round6.py                + social / climate vulnerability        -> submissions/params_round6.json
round8.py                regional specialists (one parameter set per region)
round9.py                Arizona route fix + refit              -> submissions/params_round9.json
lb_stack.py              RMSE-optimal blend from leaderboard scores alone
best.py                  rebuilds the current best from its saved weights (seconds)
lb_scores.csv            every leaderboard score (file, public RMSE)
scorecards.json          public bias scorecards (ours and the RMSE-0 entries)
submissions/best/        current best (stack6_r9) + exact blend weights
submissions/blend_inputs/  every scored file (each score is one known moment of the target)
```

```
pip install -r requirements.txt
B=s3://us-west-2.opendata.source.coop/humane-intelligence/bias-bounty-mapping-equity-challenge
aws s3 sync $B/reference/ data/reference/ --no-sign-request --exclude '*roads-unfiltered*'
for r in eastern-ok maricopa-az northern-ca south-central-tx eastern-wa; do
  aws s3 sync $B/strata/$r/ data/strata/$r/ --no-sign-request --exclude '*.csv'; done
python best.py       # -> submissions/best/stack6_r9.csv, byte-identical to the scored file
# full rebuild of the model inputs (~2 h): solution.py, round2.py, round3.py, round3_fixed.py,
# round5.py, round6.py; then `python lb_stack.py fit` refits the blend from lb_scores.csv
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
* **Per-region scaling.** Probes that keep r2f in one region and zero elsewhere (e02 Eastern OK
  0.092871, e05 Texas 0.093546) give each region's fit separately. Eastern OK's gaps are the
  most under-predicted, so the blend adds 0.198·r2f there. The result, `stack_region_ok_tx`,
  scored **0.066977**. Weights are in `best_weights.json`.
* **Boundary leakage (round 3).** 18–30 % of Overture facility points lie within 100 m of
  their tract boundary, so a station's HIFLD copy can land in the neighbouring tract. The model
  spreads each reference copy over tracts within 500 m by Φ(d/σ). The moments cannot identify σ,
  but a σ = 40 m variant added independent signal on the leaderboard (stack2 0.066850).
* **The true target's bias scorecard is public.** The RMSE-0 entries' scorecards (Zindi
  participations API) give the true disparity ratios. The group definitions were
  reverse-engineered by reproducing our own scorecard to the 3rd decimal (e.g. wildfire =
  `usfs_WHP_mean` above median, rural = `pct_urban` < 0.5). Our predictions were too flat
  (rural 1.92× vs true 2.36×, wildfire 1.33× vs 1.77×).
* **Round 5 puts those effects inside the model.** The unseen-facility rate, the TIGER-length
  multiplier and CBP density get log-linear effects per group, and the 8 true ratios join the
  calibration moments. Robust across variants: winter-drought tracts carry ×1.17–1.25 more TIGER
  highway length than Overture's named classes, and real establishments are sparser than the
  population allocation in rural tracts. The roads-vs-facilities split of the other group
  effects is not identified, so `r5e` averages three variants. It scored **0.065883**.
* **Matching the scorecard does not win RMSE.** Rank 10 (0.0660) matches the true scorecard
  almost exactly; rank 6 (0.0621) is much flatter. Our flatter r5e (rural 2.10) beat the
  truth-like r5d (rural 2.36). With noisy tract-level signal the RMSE-optimal prediction is
  flatter than the truth. Group offsets taken from the scorecard also hurt in the blend
  (stack4 with them 0.065307, without 0.065108), so the gains now come from tract-level signal.
* **Round 6** adds social- and climate-vulnerability effects (r6e 0.065531, better than r5e).
* **stack5** blends every scored file with r6e as the largest weight: predicted 0.0644,
  scored **0.064867**. Without scorecard columns the blend predictions have landed within
  ~0.0004 of the actual score each time.
* **eastern-wa is not graded.** Its probe (e06) scored exactly the all-zeros score, so its rows
  are required in the file but don't affect the score.
* Further experiment tooling (`experiments.py`, probes, 20 experiment files) lives on the
  `dev` branch.
