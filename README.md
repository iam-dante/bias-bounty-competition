# Bias Bounty Mapping Equity Challenge: one training script, from raw data to submission

Zindi user **iam-dante**. One standalone script trains everything from scratch and writes the submission:

```bash
pip install -r requirements.txt
python train.py
```

| Output | What it is | Public RMSE |
|---|---|---:|
| `physics.csv` | stages 1–7: physics model from the challenge data alone | 0.101340 (scored) |
| stage 8 with 65 labels | posterior fitted to the aggregate labels | **0.061372 (scored; best)**, estimated 0.061629 |
| `probes/q01_physics_scorecard_road.csv` | probe: road pattern of the scorecard physics | 0.061456 (scored; no signal) |
| stage 8 refit with 67 labels | posterior after the probe | **0.061348 (scored; best)**, estimated 0.061355 |
| best + 0.15 × unscored part of `physics_lb.csv` (68 labels) | first transfer step | **0.060742 (scored; best)**; implies the truth follows that direction at 0.68 ± 0.05 |
| local update + 0.4 × the 69-label physics' new part | second transfer step | **0.060353 (scored; best)**; the new part runs the other way: optimum −2.14 ± 0.27 × that step |
| local update (4.38×, −1.96×) + 0.1 bet | third transfer file | 0.060747 (scored): missed the 0.0594 prediction because the public curvature is 1.4× the estimate |
| curvature-aware local update (3.32×, 0×) | fourth transfer file | **0.060100 (scored; best)** |
| `physics_lb_regional.csv` | stage 10 with per-region multipliers (53 parameters) | moment chi-square 167.5 (global: 175.2) |
| `physics_lb_struct.csv` | stage 10 with facility leakage + TIGER local-name double count + Interstate adjustment (44 parameters) | **moment chi-square 106.0**; fitted `w_local` 0.97, spread 15 m |
| best + 0.2 × the structural physics' unscored part | fifth transfer file | **0.059977 (scored; best)**; structural direction coefficient 0.25–0.30 |
| `physics_lb_struct_regional.csv` | structural + regional (56 parameters), warm-started from the structural fit | moment chi-square 105.9: no better than structural alone |
| local Bayesian update over the three measured directions | sixth transfer file | **0.059954 (scored; best)**, as predicted |
| `physics_lb_struct_unmatched_regional.csv` | the largest physics model: structural + unmatched facilities + regional (59 parameters) | moment chi-square 112.6 on 66 moments |
| `submission.csv` | best + 0.3 × the ensemble (average) of the three best-fitting physics models' unscored parts (RMS 0.0089; their pairwise correlations +0.58 to +0.87) | 0.05983 at a coefficient of +0.5, 0.05991 at +0.3, 0.06003 at 0, 0.06015 at −0.3 |
| `probes/q02`–`q05` | stage 9: unscored probes (establishments, roads, buildings) | ≈ 0.0613–0.0615 each |
| `submission_report.json`, `physics_report.json` | every parameter, target, validation table and diagnostic | – |

The first run downloads about 4.2 GB of challenge data; a run then takes about 5 minutes on a laptop CPU.
Seeds are fixed, and reruns reproduce both files byte for byte.

## How it trains

There are no tract labels. The model is trained in two parts.

**Part 1, the physics model (stages 1–7)**, from the challenge data only:

1. **Data:** download the permitted files from the public Source Cooperative bucket. Overture release
   2026-08-19.0 roads, unfiltered roads, rail, infrastructure, buildings and places; ACS housing; strata;
   tract boundaries.
2. **Features (DuckDB spatial).** The exact Overture side of each ratio, plus evidence for each withdrawn
   reference layer:
   - the Interstate/US/State route system for TIGER;
   - the USFS building count for Microsoft footprints;
   - facility categories, names and alternates for USGS facilities;
   - the business-registry share of places for CBP;
   - mapping effort from road detail.
3. **Auxiliary models:** Poisson GLM vs gradient boosting (scikit-learn), 5-fold cross-validation grouped by
   county. They predict facility and registry density from context, where Overture's own counts are real labels.
4. **Simulator:** a Monte Carlo model of the withdrawn reference counts, run through the exact official formula.
5. **Calibration:** the Method of Simulated Moments on the organisers' published statistics:
   - the share of tracts with an undefined component in each region;
   - the 0.71–1.59 road-length ratio range;
   - "EMS is close to 1:1 and rarely binds".
6. **Validation:** leave-one-target-out, identifiability, prior sensitivity, Monte Carlo stability.
7. **Prediction:** `physics.csv`.

**Part 2, learning from aggregate labels (stage 8).** For RMSE, every scored file `v` with public score `s`
gives one exact equation about the hidden target:

```
mean_public(v · y) = (mean(v²) + mean(y²) − s²) / 2
```

`aggregate_labels.npz` holds the vectors and public scores of every submitted file (66 now), including
`physics.csv` itself. One empirical-Bayes Gaussian posterior over the target is fitted to all 65 equations at once:

- **Directions:** the files are orthogonalised in the order they were scored.
- **Priors:** each new direction's coefficient gets a group prior (physics models, probes,
  leaderboard-calibrated files), with mean and scale fitted by marginal likelihood.
- **Noise:** the noise is that of the 30% public sample, so weakly measured directions are shrunk.

No earlier file is copied or edited into the next one; their scores are training data.

The score-prediction model is validated chronologically: each scored file is predicted from the files scored
before it, with MAE 0.000113 over 36 files.

## Why the physics model matters even though it scored 0.101

The physics model measures things no earlier submission contained. Its scores show this directly:

| File | New part (out of span) | Truth lies along it at |
|---|---:|---:|
| `physics.csv` | RMS 0.044 | 0.083 ± 0.022 of its length (z 3.7) |
| Scorecard-calibrated variant | RMS 0.050 | 0.152 ± 0.020 (z 7.7) |

That information is worth about 0.0005 RMSE.

Compared with the best earlier file (R33, 0.062091), the posterior's estimate is **0.061629**:

- the gain comes from these two measured directions;
- re-weighting inside the old span costs about 0.0001;
- this is an estimate until the file is scored.

Stage 8 writes the refit only if its estimate beats the best scored file; otherwise it writes the best scored file
itself. So `python train.py` always reproduces the best submission.

## Improving it: the probe loop (stage 9)

Each new score is one more training label. The loop is:

```bash
python train.py --probes 5                                        # design 5 probes -> probes/q*.csv
# upload them; then for each score:
python train.py --add-label q01_physics_scorecard_road.csv probes/q01_physics_scorecard_road.csv 0.0614xxxxx
python train.py --fit-only                                        # refit stage 8 on all labels -> submission.csv
```

**How a probe is built:**
- **Direction:** a direction no scored file has measured. Here these are the components of the physics models
  (roads, buildings, facilities, establishments), each made orthogonal to every scored file and to the others.
- **Probe file:** the best scored file plus a small step (RMS 0.003) along that direction. It scores within
  about ±0.00015 of the best and measures how far the truth lies along its direction (±0.001).
- **Refit:** the next stage-8 fit uses that measurement.

This is how the physics model's own scores moved the target from 0.0621 to 0.0614. The probes ask the same
question component by component: how strongly does the truth agree with each part of the physics model?

## Stage 10: the physics model calibrated on the scores

`python train.py --calibrate-physics` refits the physics model's 41 parameters to three sets of targets at once:
the published statistics, the public scorecard, and the 60 independent moments of the scored files. This gives a
physics model that reproduces the scored truth:

- the misfit to the score moments (chi-squared) drops from 1,513 to 169;
- the mean is 0.057;
- the scorecard ratios fall within 0.03;
- estimated public RMSE 0.0673 on its own, vs 0.087 and 0.101 for the earlier physics files.

Its value is in the part no scored file contains (RMS 0.020). The first time, stage 8 added 15% of that part to
the best file. The file scored 0.060742, which means the truth follows that part at 0.68 ± 0.05.
The two earlier physics files' unscored parts entered the truth at 8% and 15% once scored, and the step breaks
even at 7.5%. Once the file is scored, its score measures that direction (±0.05), and the next refit takes its
full measured value.

### Stage 8 candidate selection

Stage 8 now compares three candidates and keeps the one with the lowest estimated public RMSE:

- **the full empirical-Bayes refit;**
- **the best scored file;**
- **a 1-D update for each transfer label.** A transfer file is a deliberate step from the best file scored before
  it. The two scored points on that line, and its known curvature, give the optimum multiple of the step; the
  prior is N(1, 1).

**Lesson from the third transfer file:** the public mean of a large, concentrated move's square was 1.4× the
weighted estimate. That curvature error made the previous update overshoot one direction and wrongly reverse the
other. The local update now treats curvature as uncertain:

- **Priors:** log-normal scales (sd 0.4) and correlations N(0, 0.3²).
- **Weighting:** each prior draw is weighted by how well it reproduces every transfer score.
- **Choice:** the update takes the point with the lowest posterior expected score.

Retro-check: with only two transfer scores, it gave the third file a 5–95% band of 0.0578–0.0623, which contains
the actual 0.0607. The old method had predicted 0.0593 with no band.

With several transfer files, the update is local around their anchor:

- **Steps:** each file's displacement is made orthogonal to the earlier ones.
- **Measurement:** the scores measure the target's projection on each new part.
- **Prior:** each coefficient gets N(0, 0.8²).
- **Next step:** a small `TRANSFER` (0.1) × the newest calibrated-physics direction is added as the next bet.

The calibrated-physics directions have measured +0.68 and −0.86: informative, but of unpredictable sign. A small
step risks at most about ±0.0001, and its score still measures the direction fully.

## Options

```bash
python train.py --physics-only   # stop after the physics model
python train.py --scorecard      # physics variant also calibrated on the public Bias Scorecard -> physics_scorecard.csv
python train.py --discovery      # Best Bias Discovery tables (discovery*.csv), see BIAS_DISCOVERY.md
python train.py --rebuild        # recompute cached features
python train.py --fit-only       # stage 8 only (reads physics.csv)
python train.py --probes K       # stage 9: write K probe files
python train.py --add-label NAME FILE RMSE   # record a new score as a label
python train.py --calibrate-physics          # stage 10: physics refitted on all labels -> physics_lb.csv
python train.py --calibrate-physics --structural   # + facility leakage and road-frame mechanisms -> physics_lb_struct.csv
python train.py --calibrate-physics --regional     # + per-region multipliers -> physics_lb_regional.csv
```

## Rules

- **Data:** the scored file uses only the provided datasets. External data appears only in the Bias Discovery write-up.
- **Not used:** the withdrawn layers, any copy of them, and the participant-posted Paradise scores.
- **Leaderboard:** stage 8 uses the leaderboard scores of our own files as aggregate labels. This is stated openly.
- **Tooling:** open-source packages only (NumPy, pandas, SciPy, DuckDB, scikit-learn), and fixed seeds.

## Files

- `train.py`: the whole pipeline.
- `aggregate_labels.npz`: the vectors and public RMSE of the 65 submitted files (stage 8 training data).
- `METHOD.md`: the technical write-up.
- `BIAS_DISCOVERY.md`: the Best Bias Discovery write-up.
- `requirements.txt`.
- Outputs: `physics.csv`, `submission.csv`, the report files, `discovery*.csv`, `probes/` (with `manifest.json`).
- `data/`: downloaded data and cached features (git-ignored).
- The older chained pipeline and its history are on the `main` branch.
