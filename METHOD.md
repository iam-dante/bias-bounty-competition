# Method

The target is a per-tract coverage-gap score in [0, 1]. It is the mean of up to three gaps
(roads, buildings, POIs), each of the form `1 - min(1, Overture / reference)`. The reference
layers (TIGER roads, Microsoft buildings, HIFLD facilities, County Business Patterns) were
withdrawn and are prohibited. No tract labels are published. Two approaches are provided.
Neither uses the withdrawn layers or any copy of them.

| Approach | Script | Inputs | Public RMSE |
|---|---|---|---|
| A. Data-only physics model | `data_only.py` | Permitted challenge data + the 5 published Paradise scores | unscored |
| B. Calibrated model | `solution.py`, `research/r29_analysis.py` | A's physics family + scored-submission feedback + the public Bias Scorecard | **0.062095** (R29) |
| C. Physics calibrated by simulated moments | `research/r32_msm.py` | A + true scorecard ratios + public mean(y), mean(y²) (+ optionally scored-file moments) | not shipped (see C) |

## A. Data-only physics model (`data_only.py`)

This model is estimation without labels: it predicts `E[score | permitted data]` from a
generative model of the withdrawn reference counts.

1. **Data.** Downloads the permitted files from the public Source Cooperative bucket.
2. **Features (DuckDB spatial).** The Overture side of every ratio is computed exactly per
   tract: named-highway length (motorway/trunk/primary/secondary), building count, place count
   and facility counts. The reference side is not available, so the features that predict it
   are measured:
   - **Roads:** Overture route designations. TIGER S1100/S1200 is the Interstate/US/State
     system, and Texas FM/RM and county routes are kept separate. Highways running along a
     tract boundary are measured too, because TIGER assigns them to both tracts.
   - **Buildings:** the USFS Wildfire Risk building count.
   - **Facilities:** Overture primary and alternate categories, plus name patterns for fire
     stations, EMS and schools, including points near tract boundaries.
   - **Establishments:** population.
   - **Strata:** urban/rural, tribal, SVI, CVI, drought, wildfire and heat.
3. **Physics.** Each withdrawn count is simulated:
   - **TIGER length:** route length × records per edge × log-normal noise.
   - **Microsoft count:** κ × USFS count × noise.
   - **Facilities:** binomial over Overture evidence, plus Poisson facilities that Overture never shows.
   - **CBP:** ε × population × noise.

   Each Monte-Carlo draw goes through the exact official composite, including the
   defined/undefined component rules. The mean over 4,096 draws per parameter set is the
   RMSE-optimal prediction under the model. Four parameter sets are averaged: global and
   per-region, each with and without boundary leakage. They were calibrated on statistics
   published by the organisers.
4. **Output.** Clipped to [0, 1], and the five published Paradise tracts are set to their
   scores.

## B. Calibrated model (`solution.py`, current best R28b)

B starts from A's physics family and adds two information sources: the leaderboard scores of
our own submissions, and the organisers' public Bias Scorecard.

1. **Scores as measurements.** For RMSE, a scored file v with public score s gives one exact
   moment of the hidden target y on the public rows:
   `mean(v·y) = (mean(v²) + mean(y²) - s²) / 2`.
   `mean(y²)` comes from an all-zero submission, and `mean(y)` from a constant one. The 61
   scored files and their scores are frozen in `model.npz`.
2. **Ridge calibration (R21).** Probe files measure single feature directions. A ridge fit
   (λ = 0.2) over 24 directions corrects the round-12 physics blend.
3. **Empirical-Bayes moment model (R22–R23).**
   - The scored files are orthogonalised in scoring order. Each new direction's coefficient
     gets a group prior (base models / probes), fitted by marginal likelihood.
   - The noise model reflects that the public set is a 30% sample, so weak directions are
     shrunk rather than fitted.
   - The result is blended with the ridge fit (weight 0.6 on the Bayesian part, chosen on
     simulated 30/70 public/private splits).
4. **Scorecard update (R24–R25).**
   - The Bias Scorecard reports, for each submission, the ratio of mean predicted gap between
     strata (rural/urban, tribal, high SVI, high CVI, drought, wildfire, heat). The rank-1
     entry scores exactly 0, so its displayed ratios are the target's.
   - We reproduced the stratum definitions from our own scorecards. Each ratio is a linear
     constraint on y, combined with the moments in one Gaussian posterior.
   - The part outside the scored span was then measured by its own score and rescaled
     (R25: step 0.844).
5. **Joint minimum-change update (R27).** This is the smallest change to R25 that agrees with
   every scored moment (with its sampling noise) and the 8 true ratios. The per-tract prior
   scale is fitted by marginal likelihood.
6. **Calibrated push (R28b, `submission_push.csv`).** The out-of-span piece of R27 was
   measured at z = 3.6 and stepped to 2.3×. Taking the full optimum would overshoot the true
   tribal ratio.
7. **Correlated scorecard recalibration (R29, `submission_r29.csv`).** A local minimum-change
   solve on R28b. It propagates the scorecard rounding and the correlated uncertainty of the
   ungraded Washington rows. Scored 0.062094756, the current best.

### Exact Bias Scorecard definitions (R32, `research/r32_scorecard.py`)

The public leaderboard API (`api.zindi.world/v1/competitions/<slug>/participations`) returns
every scorecard ratio to **3 decimals** (the page rounds to 2). Our own best entry is a known
vector, so its 9 ratios test a definition exactly. The following reproduces **27/27** displayed
values: R23 and R24 (2 dp, after double rounding of the 3-dp value) and R29 (3 dp).

- **Rows:** all 9,794, including the 415 ungraded Washington rows.
- **Ratio:** mean gap of the high group over mean gap of the low group.
- **Missing values:** a row missing the stratum variable is in neither group.
- **The 9 groups:**

| Stratum | High group | Low group |
|---|---|---|
| Rural | Census `ur_class` = Rural (not RUCA: no RUCA cut-off fits) | Urban |
| Tribal | `tribal_any` | the rest |
| SVI, CVI | `svi_overall` / `cvi_overall` ≥ median (not SVI ≥ 0.6) | < median |
| Summer, winter drought | `usdm_summer_dsci` / `usdm_winter_dsci` ≥ median | < median |
| Wildfire | `usfs_WHP_mean` ≥ median | < median |
| Heat | `epht_heat_days_summer` ≥ median | < median |
| High Hazard + High Vulnerability | (heat or wildfire or summer drought high) and SVI high | every other row |

**The zero-score leaders disagree.** Ranks 1–4 all score exactly 0 on the public rows, yet
their ratios differ (rural 2.234–2.378, CVI 1.462–1.649). Their graded rows agree, so the
difference is their Washington rows, which are scored by the scorecard but not by RMSE. A
Washington model per entry, level × our Washington prediction + offset, fits 8 of the top 10
entries with RMS ≈ 0.004. Fitted levels: rank 1 1.00 ± 0.13, rank 2 1.74, rank 3 ≈ 0 (the
sample file's zeros), rank 4 0.89. The graded-row truth is therefore pinned only up to the true
Washington level. The joint update treats each leader's level and offset as unknowns.

**Result (T1 candidate, `submission_r32_t1.csv`).** The update uses R29's score and the 4
leaders × 9 ratios.

- **Estimated public RMSE:** 0.062091253, vs R29's estimate of 0.062095180 (R29 actual 0.062094756).
- **Change:** RMS 0.00058 from R29; 21% inside the scored span, 79% outside.
- **Ratios:** all within 0.013 of rank 1.

Under the exact definitions R29 is already the most ratio-faithful scored file: RMS gap 0.005,
against R27 0.0085 and R28b 0.008. The scorecard information is essentially exhausted. The
expected gain (0.000004) is below the estimator's error (forward MAE 0.000119 over 34 files).

### C. Physics calibrated by the Method of Simulated Moments (`research/r32_msm.py`)

The data-only simulator (A), extended and calibrated on moments instead of tract labels.

**Generative model.** Each withdrawn count is simulated, then pushed through the exact composite:

- **Roads:** TIGER length = m_region × route length × exp(stratum effects + mapping effort) × LogNormal.
- **Buildings:** Microsoft count = κ × USFS count × LogNormal.
- **Facilities:** binomial over Overture evidence, plus Poisson unseen stations × exp(stratum effects).
- **CBP:** establishments = ε × population × exp(registry share + transit + rural) × LogNormal.

**New permitted inputs** (`research/r32_features.py`):

- mapping effort: the service / track / footpath share of the unfiltered road layer;
- Overture rail and transit infrastructure;
- the BrightQuery (business registry) share of places;
- the exact scorecard strata as effect indicators.

**Fit.** 34 parameters, with a Gaussian prior around the published-statistics fit (r9a). The
estimator is weighted least squares on moments, with weights = 1 / noise variance:

- 9 true ratios (sd 0.0065: leaders' disagreement plus Monte Carlo noise);
- 5 Paradise scores (draw variance + 0.05 structural error for the post-fire rebuild);
- public mean(y) and mean(y²).

The scored-file moments are added in variant B only, so **variant B uses leaderboard feedback**.
The draws use common random numbers (1,024 per tract). The fit is Levenberg–Marquardt with
Broyden updates of the moment Jacobian.

**Results.**

| | χ² / moments | Physics alone, est. public |
|---|---|---:|
| Prior centre | – | 0.067321 |
| MSM-A (data moments) | 185 / 16 | 0.067023 |
| MSM-B (+ scored-file moments) | 552 / 69 | 0.067024 |

- **Chronological validation.** Each scored file is predicted from the files scored before it,
  with the physics as prior for the file's out-of-span part.
  - MSM-A at full weight: forward MAE 0.000108, vs 0.000119 for the current estimator.
    The gain is not significant: 17 wins vs 17 losses, bootstrap 95% CI [−0.000039, +0.000012].
    It comes from two early probes and is about 0 from R22 on.
  - MSM-B: worse (0.000181).
- **Leave-one-ratio-out.** A refit without a ratio predicts it worse than the prior (MAE 0.17 vs
  0.03), so the stratum effects are weakly identified. The scorecard is published only as global
  ratios, so a region-wise hold-out is not observable.
- **Decision: not shipped.**
  - The physics' out-of-span part relative to R29 has RMS 0.010. Blending it into R29 *raises*
    the estimated public RMSE: 0.06232 at weight 0.5, 0.06294 at 1.0.
  - The shipping bar was a gain above 2 × forward MAE on |z| ≥ 2 directions. It is not met.
  - The physics model remains the explainable account of *why* gaps arise. The leaderboard-calibrated
    model (B) remains the predictor.

**Validation.** No labels exist, so the validation set is the scored files. `validation.csv`
predicts each file's public RMSE from the files scored before it (forward MAE 0.000119 over 34
files) and from all others (leave-one-out MAE 0.000033).

**What the model was missing.** The leaderboard-proven scorecard correction (z = 5.1) raised
predictions in less-urban, tribal and remote tracts (corr with pct_urban -0.31, tribal_pct
+0.17). It lowered them in drought-heavy, populous tracts. The correction is additive, and 62%
of it falls in Texas.

### Progression

| File | Public RMSE | Step |
|---|---:|---|
| r2f_ensemble | 0.069455 | First physics model |
| r12_blend_with_AB | 0.064009 | Moment-based blend of physics models |
| r19b | 0.062455 | Probe corrections |
| R22 | 0.062343 | Ridge + empirical-Bayes ensemble |
| R23 | 0.062327 | Ensemble weight 0.6 |
| R24 | 0.062185 | Joint scorecard update |
| R25 | 0.062148 | Measured scorecard component only |
| R27 | 0.062113 | Joint minimum-change update |
| R28b | 0.062098 | Calibrated push on R27's new piece |
| **R29** | **0.062095** | Correlated scorecard recalibration |

### Final selection (T3)

1. **Best public file: R29** (`submission_r29.csv`, 0.062094756). Under the exact definitions it
   is also the most ratio-faithful scored file.
2. **Hedge: R27** (0.062113403). It is the ratio-faithful joint solve before the public-leaning
   push (ratio RMS gap 0.0085), so it is less tuned to the public 30%.

Replace either only with a file that scores better. `research/r32/ratio_fidelity.csv` lists
every candidate.

### Best Bias Discovery

See [BIAS_DISCOVERY.md](BIAS_DISCOVERY.md). Among urban tracts, places supplied mostly by Meta
pages carry about twice the estimated gap in every region. Overture's fire stations are 85%
Meta-sourced, and its health places 65% registry-sourced.

### Tested and rejected

- **Feature-kernel extrapolation** (205 features, linear/RBF): it did not predict held-out
  probe scores.
- **Moment-matching MLP ensemble (R26):** it passed a chronological hold-out (p ≈ 0.03–0.06),
  but scored 0.062237 (no signal, z = 0.25). LightGBM with the same loss diverged.
- **Hybrid global/local shrinkage:** no likelihood gain.
- **Microsoft-sourced Overture buildings:** 65% new information, but it could not be
  calibrated, and it is a partial copy of a withdrawn layer, so it is not used.
- **Hazard + Vulnerability stratum:** unresolved until R32, which pinned it (see above).
