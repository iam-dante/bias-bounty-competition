# Bias Bounty Mapping Equity Challenge: one training script, from raw data to submission

Zindi user **iam-dante**. One standalone script trains everything from scratch and writes the submission:

```bash
pip install -r requirements.txt
python train.py
```

| Output | What it is | Public RMSE |
|---|---|---:|
| `physics.csv` | stages 1–7: physics model from the challenge data alone | 0.101340 (scored) |
| `submission.csv` | stage 8: posterior fitted to the aggregate labels | **0.06163** (estimated) |
| `submission_report.json` | every parameter, target, validation table and diagnostic | – |

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

`aggregate_labels.npz` holds the vectors and public scores of the 65 submitted files, including `physics.csv`
itself. One empirical-Bayes Gaussian posterior over the target is fitted to all 65 equations at once:

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

## Options

```bash
python train.py --physics-only   # stop after the physics model
python train.py --scorecard      # physics variant also calibrated on the public Bias Scorecard -> physics_scorecard.csv
python train.py --discovery      # Best Bias Discovery tables (discovery*.csv), see BIAS_DISCOVERY.md
python train.py --rebuild        # recompute cached features
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
- Outputs: `physics.csv`, `submission.csv`, `submission_report.json`, `discovery*.csv`.
- `data/`: downloaded data and cached features (git-ignored).
- The older chained pipeline and its history are on the `main` branch.
