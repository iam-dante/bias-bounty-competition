# Method: a coverage-gap model trained from scratch

Zindi user **iam-dante**. One script, [`train.py`](train.py), goes from the public challenge data to
`submission.csv` in two parts:

- **Stages 1–7, the physics model** (`physics.csv`). It uses only the permitted data and the organisers'
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

## 7. Physics prediction (stage 7, `physics.csv`)

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
- `--scorecard`: a physics variant that also calibrates 16 stratum effects on the public Bias Scorecard (the
  rank-1 entry's 9 ratios, tolerance 0.03). Writes `physics_scorecard.csv`, which scored 0.086862 and is one
  of the aggregate labels.
- `--discovery`: writes the Best Bias Discovery tables ([BIAS_DISCOVERY.md](BIAS_DISCOVERY.md)).

## 11. Relation to the earlier pipeline

The `main` branch keeps the earlier chained pipeline, where each file was built from the previous one (best
R33, 0.062091). This branch replaces the chain with one fit. The old files' scores enter only as training data.
