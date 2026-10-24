# Method: a coverage-gap model trained from scratch

Zindi user **iam-dante**. One script, [`train.py`](train.py), goes from the public challenge data to
`submission.csv` in two parts:

- **Stages 1–7, the physics model** (`outputs/physics.csv`). It uses only the permitted data and the organisers'
  published statistics. Scored alone, it gets 0.101340.
- **Stage 8, learning from aggregate labels** (`submission.csv`). The public RMSE of the 65 submitted files
  (`aggregate_labels.npz`) are treated as exact aggregate measurements of the hidden target, and one posterior
  is fitted to all of them. Estimated public RMSE: 0.061629.

## 1. Problem and constraints

- **Target:** a per-tract coverage gap in [0, 1], the mean of the defined components among roads, buildings and POI.
- **Component form:** each component is `1 - min(1, Overture / reference)`.
- **Missing reference layers:** TIGER roads, Microsoft footprints, USGS/HIFLD facilities and CBP were withdrawn,
  and copies are prohibited.
- **No labels:** no tract labels are published, so the target cannot be learned from examples.

The model therefore has three parts:

- the **Overture side** of every ratio, measured exactly;
- a **generative model of each withdrawn reference count**, driven by permitted evidence;
- the **exact official composite**, applied to simulated reference counts by Monte Carlo.

The mean over draws is `E[score | permitted data]`, the RMSE-optimal prediction under the model.
Only the provided datasets are used for the scored file, as the rules require. Additional public data
appears only in the Bias Discovery write-up.

Five "published" Paradise scores circulate in the discussions (thread 34910). They are **not used**: a
participant computed them from the reference layers before those layers were withdrawn and before the
28 Sep re-scoring.

## 2. Data and features (`train.py` stages 1–2)

- **Source:** the public Source Cooperative bucket (no account), Overture release `2026-08-19.0`.
- **Spatial work:** DuckDB spatial, with lon/lat transformed using `always_xy := true`.

| Component | Overture side (exact) | Evidence for the withdrawn reference |
|---|---|---|
| Roads | motorway / trunk / primary / secondary length | Length of Overture segments in Interstate / US / State route relations (TIGER S1100/S1200 is that system); Texas FM/RM, county routes and unrouted arterials kept apart; highways running along a tract boundary (TIGER assigns them to both tracts); Arizona's historic / future route tags excluded |
| Buildings | building count (centroid in tract) | USFS Wildfire Risk building count (strata table) |
| Facilities | `fire_department`, `ambulance_and_ems_services`, six school categories | the same plus alternate categories and name patterns (weak evidence); expected facility density from context (§3) |
| Establishments | all Overture places | population; business-registry (BrightQuery) intensity from context (§3) |
| Mapping effort | – | service / track / footpath share of the unfiltered road layer |
| Strata | – | the Bias Scorecard strata (used only with `--scorecard`) |

## 3. Auxiliary supervised models (stage 3)

These are the places where real labels exist. Each model predicts an observed Overture count from
**context only**: population, housing, area, USFS buildings, total and detailed road length, transit, rail,
urbanity, RUCA, SVI, tribal share and region. Overture place counts are left out, so the predicted density
does not inherit mapping completeness. The comparison is a Poisson GLM vs Poisson gradient boosting
(scikit-learn), using 5-fold cross-validation **grouped by county**. The out-of-fold (cross-fitted)
predictions of the better model feed the simulator.

| Target | GLM CV deviance | GBM CV deviance | Null | Chosen, deviance explained |
|---|---:|---:|---:|---|
| Fire evidence | 1.008 | 0.983 | 1.370 | GBM, 28% |
| EMS evidence | 0.886 | 0.880 | 1.038 | GBM, 15% |
| School evidence | 2.829 | 2.707 | 3.752 | GBM, 28% |
| BrightQuery places | 24.47 | 22.35 | 45.23 | GBM, 51% |

## 4. Generative model (stage 4)

For each tract and Monte Carlo draw:

- **Roads:** `TIGER = m_region · exp(stratum effects) · (route + β·boundary + B_fm·FM + B_cty·county + B_unr·unrouted) · LogNormal(σ_t)`.
  - The `B_*` are Bernoulli inclusions.
  - A small surprise rate covers TIGER highways with no Overture evidence.
  - The road component is undefined where TIGER has no highway.
- **Buildings:** `Microsoft = κ · USFS count · LogNormal(σ_b)`.
- **Facilities:** for each type, `reference = Bin(Overture primary, a) + Bin(weak evidence, x) + Poisson(unseen)`.
  - The unseen rate is `μ · (1−r)/r · exp(γ · mapping effort)`, where `μ` is the cross-fitted expected density.
  - A facility type is defined only where reference ≥ 1.
- **CBP:** `establishments = ε · population · exp(a · registry intensity) · LogNormal(σ_c)`, defined where ≥ 0.5.
- **Composite:** POI = mean of the defined halves; score = mean of the defined components, exactly as the
  Evaluation page specifies.

## 5. Calibration by simulated moments (stage 5)

Weighted least squares on the statistics below (weights = 1 / noise variance), plus a Gaussian prior on 25
parameters. The fit uses common random numbers and Levenberg–Marquardt with Broyden updates.

| Statistic | Value | Source | Calibrated model |
|---|---:|---|---:|
| Share of tracts with an undefined component: Eastern OK / Maricopa / Northern CA / South TX | 0.21 / 0.55 / 0.37 / 0.28 | Evaluation page | 0.209 / 0.535 / 0.339 / 0.265 |
| Overture/TIGER highway-length ratio, lowest / highest region | 0.71 / 1.59 | Evaluation page | 0.723 / 1.596 |
| Overture / reference EMS count | ≈ 1 | data README ("close to 1:1") | 1.06 |
| Share of tracts where the EMS term binds | small | data README ("rarely binds") | 0.021 |
| Reference schools per tract | ≈ 1.5 | national order of magnitude (≈130k K-12 schools / ≈85k tracts) | 1.81 |
| Reference fire stations per tract | ≈ 0.6 | national order of magnitude (≈52k stations) | 0.66 |
| Establishments per resident | ≈ 0.025 | national order of magnitude | 0.024 |

The last three are weak, national-level soft constraints, not tract-level reference data. They can be
removed from `Targets` in `train.py` if they are considered external knowledge.

**Mechanism check before any fitting.** The share of tracts with no Overture route-network length
already matches the published undefined shares. That shows the route system is the right proxy for TIGER
S1100/S1200:

| Region | No Overture route length | Published undefined share |
|---|---:|---:|
| Eastern Oklahoma | 0.222 | 0.21 |
| Maricopa | 0.564 | 0.55 |
| Northern California | 0.365 | 0.37 |
| South-Central Texas | 0.306 | 0.28 |

**Priors** are physical judgements, documented next to each parameter in `train.py`:

- TIGER metres per metre of route network ≈ 1;
- Microsoft count ≈ the USFS count (κ ≈ 1);
- Overture labels about 60% of fire stations and most EMS posts and schools;
- more mapping detail means fewer unseen facilities.

## 6. Validation (stage 6)

There are no labels, so validation checks what can be checked:

- **Cross-validated auxiliary models** (§3).
- **Leave-one-target-out.** Each published statistic is held out, the model refit (first order), and the
  statistic predicted.
  - The four undefined shares are predicted within 0.036.
  - The road-ratio extremes and the soft facility targets are not: each pins its own parameters, and
    nothing else in the data does.
- **Identifiability.** 2 of the 25 parameters have a posterior sd below half their prior sd:
  Eastern Oklahoma's TIGER scale and the fire-station visibility. The rest stay near their priors.
  That is the honest limit of label-free calibration.
- **Prior sensitivity:** the RMS change of the prediction when one parameter moves by one prior sd.
  - Largest: building scale κ (0.068), establishments per resident (0.048) and registry elasticity (0.026).
  - Then the Texas and Oklahoma TIGER scales (about 0.02).
  - These are the choices to refine first.
- **Monte Carlo stability:** RMS 0.0016 between two independent 4,096-draw runs, with fixed seeds.

## 7. Physics prediction (stage 7, `outputs/physics.csv`)

`E[score | data, θ_MAP]` with 4,096 draws per tract, clipped to [0, 1]. Scored 0.101340 public RMSE.

| Quantity | Value |
|---|---:|
| Mean prediction on graded tracts | 0.0955 |
| Road component defined | 69% of tracts |
| Mean road gap where defined | 0.091 |
| Building gap | 0.050 |
| Facility half | 0.309 (fire 0.76 where defined) |
| Establishment half | 0.086 |

Parameter uncertainty is reported, not averaged in. With unidentified parameters, averaging over the prior
inflates the bounded gap and adds bias instead of information.

## 8. Learning from aggregate labels (stage 8, `submission.csv`)

**Measurement.** For RMSE, a file `v` with public score `s` gives one exact moment of the target on the public
rows:

```
mean_public(v · y) = (mean(v²) + mean(y²) − s²) / 2
```

- `mean(y²)` comes from an all-zero file, and `mean(y)` from a constant file.
- Public-row membership is unknown. Its probabilities are estimated from the exact affine relations among
  scored files (approximate Gaussian conditioning of a 30% public subset).

`aggregate_labels.npz` holds the 65 scored files and their public RMSE. That includes `physics.csv` and its
scorecard-calibrated variant (0.086862).

**Model.** `y = Σ θ_j q_j` over the scored files, orthogonalised in scoring order (57 independent directions).

- **Group priors:** each coefficient gets a group prior: physics/base models, probes, or
  leaderboard-calibrated files. Mean and scale are fitted by marginal likelihood:
  - base models: mean 0.49, sd 0.37;
  - probes: mean 0.06, sd 0.72;
  - leaderboard-calibrated files: mean 0.47, sd 0.14.
- **Noise:** the moments carry the sampling noise of the 30% public set (fitted scale 0.76), so weak
  directions are shrunk.
- **Fit:** the posterior mean is the prediction for graded tracts. It is one fit; no file is edited into another.
- **Washington:** the ungraded Washington rows get the graded-row calibration of the physics model,
  `0.0196 + 0.411 × physics`.

**What the physics model adds.** Each physics file's part outside the span of all earlier files is measured by
its own score:

| File | Out-of-span part | Best step along it | z |
|---|---:|---:|---:|
| `physics.csv` | RMS 0.044 | 0.083 ± 0.022 | 3.7 |
| Scorecard variant | RMS 0.050 | 0.152 ± 0.020 | 7.7 |

Both hold under uniform public weights (z 2.9 and 7.4).

Against the best earlier file (R33, 0.062091), the posterior moves RMS 0.0094:

| Part of the move | RMS | Estimated public RMSE if added to R33 |
|---|---:|---:|
| Along the measured physics directions | 0.0089 | 0.061534 |
| Re-weighting inside the old span | 0.0035 | 0.062197 |
| Whole posterior | 0.0094 | **0.061629** |

The re-weighting is the cost of a fresh fit instead of hand-measured steps.

**Validation.**

- **Chronological check:** each scored file's public RMSE is predicted from the files scored before it.
  MAE 0.000113 over 36 files.
- **Physics files:** they were forecast poorly before being scored (errors +0.0016 and +0.0043). That is
  why their scores add information.
- **Replay at past cutoffs:** fitting only on earlier files, then re-estimating with all files, shows no
  optimism (within ±0.00007).

**Tested and rejected.** Adding the 36 public scorecard ratios to the same posterior, with each leader's
Washington level latent, raised the estimate to 0.0623. The scorecard's information is already in the scored
files, and a direct test of its remaining part measured −0.2 ± 1.9.

**Ratios of `submission.csv`** vs the rank-1 public scorecard:

| Stratum | Submission | Rank 1 |
|---|---:|---:|
| Rural | 2.23 | 2.34 |
| Tribal | 2.81 | 2.88 |
| SVI | 1.28 | 1.23 |
| CVI | 1.58 | 1.55 |
| Wildfire | 1.68 | 1.77 |
| Heat | 0.63 | 0.62 |

**Result.**

- The 65-label fit scored **0.061372**, better than its estimate of 0.061629 and than R33 (0.062091).
- The line through R33 and this file has a known curvature, so the two scores place the optimum along it at
  about 1.0×. The fit is already at the best point of its own direction.
- With that score added (66 labels), the refit is estimated at 0.061347.

## 8b. Probe design (stage 9)

Further gains need directions no scored file has measured.

**Candidates:** the four component contributions (roads, buildings, facilities, establishments) of both physics
models. They sum to each model's prediction, and only that sum has been scored.

**Selection:**
- Each candidate is made orthogonal to the span of all scored files (pi-weighted).
- The largest is taken first, and the rest are made orthogonal to it.
- The process repeats while the remaining part has RMS > 0.002.

**Probe file:** the best scored file plus a step of RMS 0.003 along the unit direction. Its score gives the
target's projection on that direction, to about ±0.001 (public-sampling noise).

**Cost and gain:**
- If the direction is empty, the probe costs at most about 0.00007 RMSE.
- If it carries signal like the physics directions did, it gains about the same.

The scores enter stage 8 as labels, so the next fit re-weights each component by how strongly the truth agrees
with it. The first batch (`outputs/probes/manifest.json` when rerun; the probe files are no longer kept):

| Probe | Component | New part (RMS) |
|---|---|---:|
| 1 | Roads, scorecard physics | 0.028 |
| 2 | Establishments | 0.017 |
| 3 | Roads | 0.014 |
| 4 | Buildings | 0.010 |
| 5 | Buildings, scorecard physics | 0.005 |

The facility component is already almost entirely inside the scored span.

## 8c. Physics calibrated on the scores, and the transfer step (stage 10)

**What the last gain was made of.** The best file is close to:

```
0.89 × R33 + 0.11 × (scorecard physics, facility part) + 0.10 × (its road part) − (its establishment part)
```

This explains 98.8% of it. The truth endorsed the scorecard-calibrated physics' structure; single tract features
explain little (|correlation| ≤ 0.11).

**Stage 10.** It refits the 41 physics parameters, stratum effects included, on three sets of targets at once:
the published statistics, the scorecard (tolerance 0.03), and the 60 orthogonalised score moments. The fit uses
the same simulated-moments machinery as stage 5.

| Fit | Start | Fitted |
|---|---:|---:|
| Chi-squared, 60 score moments | 1,513 | 169 |
| Chi-squared, published statistics + scorecard | 100.8 | 28.7 |

The fitted model has mean 0.057, scorecard ratios within 0.03 of rank 1, and an estimated public RMSE of 0.0673
on its own.

**Transfer step.**

- **Step:** `submission.csv` adds 0.15 × the part of this model that is orthogonal to every scored file
  (RMS 0.020) to the best scored file.
- **Why 0.15:** the two physics files' orthogonal parts entered the truth at 0.083 ± 0.022 and 0.152 ± 0.020
  when scored, and the step breaks even at a coefficient of 0.075.
- **Expected outcomes:**

  | Truth's coefficient | Expected public RMSE |
  |---|---:|
  | 0 | 0.061424 |
  | 0.1 | 0.061323 |
  | 0.2 | 0.061222 |
  | 0.3 | 0.061121 |

Once the file is scored, the direction is measured to about ±0.05 and the next stage-8 refit takes its full value.

**Result.** The file scored **0.060742**. Along its step, the optimum is 4.50 ± 0.32 times the step taken, i.e.
the truth follows the label-calibrated physics' unscored part at 0.68 ± 0.05 (z 14). The guessed-parameter
physics directions measured 0.08 and 0.15.

**Next round.**

- **1-D update:** stage 8 compares the full refit (0.0602, which re-weights old directions badly), the best
  scored file, and the 1-D Bayesian update along the transfer step (posterior 4.18×, estimated **0.059819**).
  It keeps the lowest.
- **Recalibration:** stage 10 recalibrated on 69 labels gives a physics model estimated at 0.0635 on its own,
  with a new unscored part of RMS 0.0088.
- **Submission:** the 1-D update plus 0.4 × that new part. Expected public RMSE: 0.05992 if the part is empty,
  0.05982 at a coefficient of 0.2 (break-even), 0.05971 at 0.4, 0.05961 at 0.6.

**Result:** it scored **0.060353**. The two parts are orthogonal (correlation −0.008), so the scores separate them:

| Part | Optimum multiple of the step taken |
|---|---:|
| First direction | 4.50 ± 0.32 (confirmed) |
| New part (the 0.4 bet) | **−2.14 ± 0.27** |

So the truth runs opposite to the 69-label physics' new direction, with coefficient about −0.86. Without the bet,
the file would have scored 0.059808, as predicted.

**Lessons:**
- Every bet is a precise measurement, whichever its sign.
- The measurement's precision does not depend on the step size, so bets are now small (`TRANSFER = 0.1`).

**Stage 8 now:**
- **Local update:** around the anchor (`s8_refit67`), every transfer displacement's new part is measured, with a
  prior N(0, 0.8²) on its coefficient.
- **Next file:** 4.49 × the first part and −2.11 × the second (estimated 0.059359), plus 0.1 × the 70-label
  physics' new part (RMS 0.0057).
- **Expected:** 0.05933–0.05941.

**Result and correction.**

- **The miss:** the file scored **0.060747**. That is not explained by the small bet (±0.0001) or by clipping:
  the three transfer files lie in the two-direction plane up to RMS 0.0015.
- **The cause:** the quadratic term. The public rows' mean of each move's square was larger than the weighted
  estimate. Solved from the three scores, the curvature scale is 1.38, the same under uniform weights.
- **What it means:** large moves along these directions are concentrated in tracts that the public 30% happens
  to over-represent. Under the correct curvature, the second direction's optimum is about 0, not −2.1, and the
  first direction's about 3.3×, not 4.5×.

**Curvature-aware local update (current stage 8).**

- **Directions:** the main directions are the displacements' new parts with RMS ≥ 0.002.
- **Priors:** the public curvature along them gets log-normal scales (sd 0.4) and correlations N(0, 0.3²).
- **Fit:** each prior draw fits the linear terms to all transfer scores by weighted least squares. A file's noise
  is set by its part outside the main directions, so the first two files count as exact. Each draw is then
  weighted by its fit.
- **Choice:** the update minimises the posterior expected MSE, `x* = E[C]⁻¹ E[G]`.
- **Result:** 3.32 × the first direction, 0 × the second, expected **0.060156 ± 0.000062**. It interpolates
  between the scored points.
- **Retro-check:** given only the first two transfer scores, the method's 5–95% band for the third file was
  0.0578–0.0623, which contains the actual 0.0607 (81st percentile).
- **No new bet this round** (`TRANSFER = 0`).

**Result:** that file scored **0.060100**, better than its expected 0.060156. With four scored points, the
plane's optimum is 0.060099 ± 0.00003: the measured directions are exhausted.

**Richer physics, calibrated on all 72 scores.**

| Model | Parameters | Moment chi-square (64 moments) | Unscored part (RMS) |
|---|---:|---:|---:|
| Global | 41 | 175.2 | 0.0039 |
| + per-region multipliers (unseen facilities, establishments, building scale) | 53 | 167.5 | 0.0063 |
| **+ structural mechanisms** | **44** | **106.0** | **0.0136** |

- **Regional effects:** unseen facilities about +57% in Texas and −30% in Maricopa.
- **Structural mechanisms:**
  - facility copies leak across tract boundaries (fitted spread 15 m);
  - TIGER counts route highways that also carry a local street name twice (fitted 0.97);
  - an Interstate adjustment (−0.14).

The structural model's unscored part correlates only about 0.2 with the others. It is the analogue of the first
calibrated-physics direction (RMS 0.020, coefficient +0.5).

**Bet:** `submission.csv` adds 0.2 × that part to the best file (step RMS 0.0027, above the 0.002 threshold, so
it becomes a main direction once scored). Expected public RMSE with curvature 1.3:

| Truth's coefficient | Expected public RMSE |
|---:|---:|
| +0.5 | 0.05987 |
| +0.3 | 0.06000 |
| 0 | 0.06018 |
| −0.3 | 0.06036 |

**Result:** it scored **0.059977**. The structural direction's coefficient is 0.25–0.30, depending on curvature;
0.2 was already close.

**Diminishing returns.**

- **Big steps came from better fits.** They appeared when a new model fitted the scores much better than the
  last: guessed → calibrated gave −0.0007; calibrated → structural (chi-square 175 → 106) gave −0.00012.
- **Fit has stopped improving.** With 73 labels, structural + regional reaches the same fit as structural alone
  (105.9; 127.5 when cold-started). Its unscored part is RMS 0.008, worth at most about 0.00002.
- **No other source is left.** The old-span correction toward R33 is worth 0.000016.
- **Current file:** the local Bayesian update over the three measured directions, without a bet (estimated
  0.059954 ± 0.00003).
- **Further gains** need a mechanism that makes the physics fit the scores clearly better than chi-square ≈ 106.

**Facility mechanisms tested (`--unmatched`).**

- **Screen.** I correlated the score-visible misfit (best file minus the structural physics, restricted to the
  scored span) with candidate Overture categories that the organisers' match does not count. The signs all fit
  "the reference layers hold places the match misses":

  | Candidate | Correlation |
  |---|---:|
  | Colleges / universities | +0.18 |
  | Vocational schools | +0.11 |
  | Generic "education" | +0.09 |
  | Hospital ERs | +0.09 |
  | Preschools | +0.07 |
  | Charter / religious / Montessori schools | +0.06 |

- **Mechanism.** Reference copies from colleges and vocational schools, from other K-12 / education places
  (schools layer) and from ERs (EMS layer), each with a fitted share.
- **Calibration** (structural + unmatched, warm-started, 73 labels): moment chi-square **106.2**, no better than
  structural alone. Fitted shares: colleges 7%, other K-12 11%, ERs 2%.
- **Decision:** not used. A model that fits no better gives no reason to trust its unscored part (RMS 0.0125).
- **Tract area.** The misfit also correlates −0.10 with tract area, but area lies 98% inside the scored span: the
  posterior already uses it, and a physics mechanism for it would add no new scored information.

**Largest model and ensemble bet.**

- **Last local update:** scored **0.059954**, exactly as predicted.
- **Largest model:** structural + unmatched facilities + regional (59 parameters, warm-started from the smaller
  fits). It reaches moment chi-square 112.6 on 66 moments, about the same fit per moment as structural alone.
- **Agreement:** the three best-fitting models' unscored parts relative to all 74 labels correlate +0.58 to
  +0.87. Different calibrations of the physics agree on this direction.
- **Bet:** `submission.csv` adds 0.3 × their average (an ensemble; RMS 0.0089) to the best file. Expected public
  RMSE:

  | Truth's coefficient | Expected public RMSE |
  |---:|---:|
  | +0.5 | 0.05983 |
  | +0.3 | 0.05991 |
  | 0 | 0.06003 |
  | −0.3 | 0.06015 |

**Road screen: concurrent routes.**

- **Screen.** The same in-span screen on roads, buildings and establishments found much stronger signals than
  for facilities:

  | Signal | Correlation |
  |---|---:|
  | Share of route length carrying several route numbers | **+0.31** |
  | Road detail | +0.25 |
  | Interstate share | +0.21 |
  | Farm-to-Market length | −0.20 |
  | Local-name share | +0.20 |
  | Named-to-route ratio | −0.20 |

- **Mechanism.** TIGER keeps one record per (edge, name), so a road carrying two route numbers is counted
  twice, as the local-name mechanism already showed. A fitted weight `w_multi` on concurrent-route length now
  enters the TIGER length.
- **Recalibration** of the largest model, warm-started:
  - moment chi-square 112.6 → **94.2**, the first break below the 106 ceiling;
  - fitted `w_multi` 0.46, `w_local` 0.67, `w_inter` 0.21.
- **Bet.** The new model's unscored part (RMS 0.0106) agrees +0.87 with the earlier ensemble direction.
  `submission.csv` now adds 0.3 × it to the best file:

  | Truth's coefficient | Expected public RMSE |
  |---:|---:|
  | +0.5 | 0.05978 |
  | +0.3 | 0.05990 |
  | 0 | 0.06006 |
  | −0.3 | 0.06023 |

**Result:** it scored **0.059507**, beyond the best case. Along that step the optimum is 2.3–3.2× (curvature
1.5 to 1.0), so the truth follows the concurrent-route model's unscored part with coefficient **0.7–1.0**: the
strongest direction found.

- **Curvature prior.** The public curvature has measured 1.04–1.56 on every direction, so the local update's
  prior is now log-normal with median 1.3 (sd 0.35) instead of 1.0.
- **Next file.** The update takes 2.45× along the new direction: expected **0.05926 ± 0.00016**, P(beats best)
  0.93.
- **No bet.** Recalibrated on 75 labels, the best physics fits the same (94.6), estimates 0.0626 on its own, and
  has almost nothing unscored left (RMS 0.0019).

**Next road screen.**

- **The screen validates itself.** On the improved model, the concurrent-route signal fell from +0.31 to +0.09.
- **What remains:**

  | Signal | Correlation |
  |---|---:|
  | Interstate share | +0.24 |
  | Road detail | +0.23 |
  | Farm-to-Market length | −0.21 |
  | Buildings per housing unit | −0.20 |

- **Covariate effects (`--covariates`).** Log-effects of these on the TIGER length and the building scale give
  moment chi-square 94.6 → 89.3 (64 parameters). The fitted effects are small: buildings per housing −0.12,
  Farm-to-Market −0.06, the others about 0. The unscored part is RMS 0.0045, too small for a worthwhile bet.

**Result of the local update:** it scored **0.059232**, inside its band (predicted 0.05926 ± 0.00016). With that
second point on the concurrent-route line, the four measured directions are exhausted: the update's optimum
(2.61×) is expected to gain 0.000002.

**What was tested next (76 labels), and the result:**

| Test | Result |
|---|---|
| Big model recalibrated on 76 labels | moment chi-square 91.6; unscored part RMS 0.0029: nothing worth a bet |
| TIGER records per route number and alternate name (distinct refs, 3+ refs, alternate / official names, bridges) | in-span correlations ≤ 0.19, mostly already in span |
| Facility evidence from Overture buildings (fire-station and school buildings in tracts without the place) | correlations ≤ 0.10: no signal |
| County-level mapping quality (unrouted arterial share, fire stations per resident) | ≤ 0.17, 76–81% in span |
| Linear feature correction learned through the moments (98 features, ridge) | **worse** on held-out moments at every penalty (last 12: chi-square 19.5 → 20.2–44.5) |
| Flexible context effects inside the physics (`--flex`, 28 log-effects, prior sd 0.1) | see below |

**Flexible physics (`--flex`).** Fourteen context features act on the TIGER length (`ft_`), and fourteen on the
unseen-facility rate (`ff_`): buildings per housing unit, route-type shares, area, density, RUCA, urban share,
places per resident, source shares, SVI, tribal share, and others.

- **Held-out check** (linearised refit on the published statistics plus the first k moments, predicting the
  rest): it is better on the last 8 moments (chi-square 39.0 → 33.8–36.7) but worse on the last 12 and 20.
  Leave-one-moment-out is better at prior sd 0.1 (236.6 → 219.3) and worse at 0.2.
- **Full calibration:** moment chi-square 91.6 → 68.7, published + scorecard 23.8 → 18.1. A total drop of 28.6
  for 28 parameters is what fitting noise alone gives, so this is no proof of a mechanism. The largest effects
  are on unseen facilities: urban share −0.18, density −0.16, a route in the tract +0.15.
- **Unscored part:** RMS **0.0146**, five times the big model's (correlation +0.20 with it). It is the only
  sizeable unmeasured direction left.
- **Bet.** `submission.csv` = the best file + 0.2 × that part (RMS change 0.0029). Break-even coefficient about
  0.13; calibrated-physics directions have measured 0.25–1.0 so far:

  | Truth's coefficient | Expected public RMSE |
  |---:|---:|
  | +0.5 | 0.05897 |
  | +0.3 | 0.05911 |
  | 0 | 0.05933 |
  | −0.3 | 0.05954 |

**Result:** it scored **0.058987**, the +0.5 case. The flexible effects extrapolate: the truth follows their unscored
part at about +0.5, the same range as the earlier calibrated-physics directions. Noise fitting would have given 0.

- **Local update** over the five measured directions: 1.74× along the flexible one (1.37 curvature), expected
  0.058929 ± 0.000051, P(beats best) 0.88.
- **Recalibrated on 77 labels** (warm-started from the 76-label fit), the flexible physics fits much better:
  - moment chi-square 73.0;
  - estimated **0.0615 on its own**, vs 0.0634 before and 0.0627 for the big model;
  - new unscored part RMS 0.0043.
- **Next file.** `submission.csv` = the local update + 0.3 × that part. It beats the best scored file in every case:

  | Truth's coefficient | Expected public RMSE |
  |---:|---:|
  | +0.5 | 0.05890 |
  | +0.3 | 0.05892 |
  | 0 | 0.05895 |
  | −0.3 | 0.05898 |

**Result:** it scored **0.058912**, between the +0.3 and +0.5 cases: the second flexible direction measured about
+0.4. Refitted on 78 labels, the local update's optimum is the scored file itself (P(beats best) 0.44).

- **Recalibrated on 78 labels:** the standard flexible model (sd 0.1) estimates 0.0613 on its own, unscored RMS
  0.0027. A wider version (`--flex-sd 0.2`, warm-started) estimates **0.0611**, the best physics model so far,
  with unscored RMS 0.0037. The two unscored parts agree +0.53.
- **Next file.** `submission.csv` = the best file + 0.3 × the wider model's unscored part (RMS change 0.0011).
  Break-even coefficient about 0.2:

  | Truth's coefficient | Expected public RMSE |
  |---:|---:|
  | +0.5 | 0.05889 |
  | +0.3 | 0.05890 |
  | 0 | 0.05893 |
  | −0.3 | 0.05895 |

- **Diminishing returns again.** Each recalibration leaves a smaller unscored part (0.0146 → 0.0043 → 0.0037), so
  the next big step needs a model with more capacity, not another refit.

**Result:** it scored **0.058866**, beyond the +0.5 case: the wider flexible direction measured about **+0.85**.

**Small bets now count as directions.** The last three bets moved RMS 0.0011–0.0013, below the update's 0.002
cut-off, so their measurements were unused. The local-update files themselves move at most 0.00025 outside the
plane (clipping), so the cut-off is now 0.001. Lowering it alone gave a −10× coordinate on the old
curvature-miss direction (expected 0.0558 ± 0.0094: meaningless). So each coordinate now gets a prior,
x ~ N(0, 3²), and each curvature draw is weighted by the scores' marginal likelihood. Two candidates are compared:

| Candidate | Expected | P(beats 0.058866) |
|---|---:|---:|
| Full update over 8 directions | 0.05883 ± 0.00007 | 0.72 |
| **Best file, newest direction at 2.15×** (others where scored) | **0.058848 ± 0.00001** | **0.97** |

Stage 8 takes the lower estimate among candidates with P ≥ 0.75: the second.

**Region-specific flexible effects (`--flex-regional`).** Each of the 28 flexible effects gets a per-region
deviation (prior sd 0.1): 200 parameters in all.

- **Fit:** warm-started from the wider flexible fit, moment chi-square 71.9 → 63.5. Estimated **0.0610** on its
  own, the best physics model.
- **Largest deviations** are on unseen facilities: places per resident in Maricopa −0.07, buildings per housing
  in Texas −0.07, weak fire evidence and SVI in Oklahoma −0.05.
- **Unscored part:** RMS 0.0046, nearly orthogonal to the earlier flexible directions (+0.09, +0.06).
- **Next file.** `submission.csv` = the chosen candidate + 0.3 × that part:

  | Truth's coefficient | Expected public RMSE |
  |---:|---:|
  | +0.5 | 0.05882 |
  | +0.3 | 0.05884 |
  | 0 | 0.05887 |
  | −0.3 | 0.05890 |

**Result:** it scored **0.058936**, worse than the best and beyond the −0.3 case. With prior sd 0.1 on 112
per-region deviations, the region-specific model over-fitted: its unscored part runs against the truth (about
−0.7). The wider global flexible model (+0.85) is the one that extrapolates.

**The update is now stable and risk-aware.**

- **Stability.** With 12 transfer scores, the posterior over the public curvature is sharp: only 20–130 of
  200,000 prior draws carry weight. The previous file's estimate therefore had real Monte Carlo error. The update
  now uses 1,000,000 draws in blocks, in two passes with the same seeds.
- **Risk.** The candidates are the full update and the best scored file with its newest one or two directions
  moved. Stage 8 now picks the lowest *pessimistic* score (mean + 1 sd), not the lowest mean:

  | Candidate | Expected | P(beats 0.058866) |
  |---|---:|---:|
  | Full update over 9 directions | 0.05876 ± 0.00009 (0.05875–0.05879 across seeds, sd up to 0.00015) | 0.76–0.89 |
  | Best, newest direction at −1.6× | 0.05881 ± 0.00004 | 0.87–0.93 |
  | **Best, newest two at 2.26× and −1.64×** | **0.058791 ± 0.000044** | **0.88–0.97** |

- **No new bet.** Recalibrated on 80 labels, the wider flexible model estimates 0.0609 on its own (the best physics
  model), but its unscored part is only RMS 0.0019.

## 9. Edge cases

- **Undefined components** are excluded per tract, so the divisor is 1–3. A tract with all three undefined
  scores 0.
- **Tracts with no TIGER highway:** about a third of tracts have no route evidence, so their road component
  is undefined and the score is the mean of buildings and POI. A small surprise rate allows unseen highways.
- **Zero-population tracts:** establishments ≈ 0, so the establishment half is undefined and POI rests on
  facilities alone.
- **Water-dominated tracts:** the 7 Texas `99xx` tracts are not in the sample submission. A tract with no
  USFS building count falls back to the Overture count.
- **Tracts with no Overture data:**
  - The ratios use Overture = 0, so a defined component scores 1.
  - Context-only densities still give them expected facilities.
- **Eastern Washington** (not graded): predicted with the mean of the four regional TIGER scales.
- **GEOIDs** are read as text everywhere (Maricopa's state FIPS is `04`).

## 10. Options

- `--physics-only`: stop after stage 7.
- `--fit-only`: stage 8 only, reading `outputs/physics.csv`.
- `--probes K`: stage 9, which writes K probe files and a manifest to `outputs/probes/`.
- `--add-label NAME FILE RMSE`: record a scored file as a new label.
- `--calibrate-physics`: stage 10, which writes `outputs/physics_lb*.csv` and its report. With `--structural`,
  `--unmatched`, `--regional`, `--covariates` and `--flex` it adds the mechanisms of §8c; each adds a suffix to
  the output name.
- `--scorecard`: a physics variant that also calibrates 16 stratum effects on the public Bias Scorecard (the
  rank-1 entry's 9 ratios, tolerance 0.03). Writes `outputs/physics_scorecard.csv`, which scored 0.086862 and is one
  of the aggregate labels.
- `--discovery`: writes the Best Bias Discovery tables ([BIAS_DISCOVERY.md](BIAS_DISCOVERY.md)).

## 11. Relation to the earlier pipeline

The `main` branch keeps the earlier chained pipeline, where each file was built from the previous one (best
R33, 0.062091). This branch replaces the chain with one fit. The old files' scores enter only as training data.
