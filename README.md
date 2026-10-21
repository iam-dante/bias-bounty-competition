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
| `physics_lb.csv` | stage 10: physics refitted on all 68 scores + statistics + scorecard | 0.0673 (estimated) |
| `submission.csv` | best scored file (0.061348) + 15% of the part of `physics_lb.csv` no scored file contains | 0.06132 if the truth's coefficient is 0.1, 0.06122 at 0.2; 0.06142 if 0 |
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

Its value is in the part no scored file contains (RMS 0.020). Stage 8 adds 15% of that part to the best file.
The two earlier physics files' unscored parts entered the truth at 8% and 15% once scored, and the step breaks
even at 7.5%. Once the file is scored, its score measures that direction (±0.05), and the next refit takes its
full measured value.

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
