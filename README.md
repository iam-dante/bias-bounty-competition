# Bias Bounty Mapping Equity Challenge: coverage-gap model trained from scratch

Zindi user **iam-dante**. One standalone script trains the model from raw data and writes the submission:

```bash
pip install -r requirements.txt
python train.py
```

It downloads the permitted files from the public Source Cooperative bucket (no account) and builds the
features with DuckDB. It then fits auxiliary models with cross-validation, calibrates a generative model of
the withdrawn reference layers, and writes:

- `submission.csv`: 9,794 tracts, values in [0, 1];
- `submission_report.json`: every fitted parameter, calibration target, validation table and diagnostic.

The first run downloads about 4.2 GB and builds the features; later runs take about 5 minutes on a laptop CPU.
Seeds are fixed, so a rerun reproduces the file.

## What it uses

- **The challenge data only:**
  - Overture release 2026-08-19.0 roads, unfiltered roads, rail, infrastructure, buildings and places;
  - ACS housing units;
  - the strata tables;
  - the tract boundaries.
- **The organisers' own published statements:**
  - the share of tracts with an undefined component in each region;
  - the 0.71–1.59 range of the Overture/TIGER highway-length ratio;
  - "EMS is close to 1:1 and rarely binds".
- **Three weak national order-of-magnitude constraints:** schools and fire stations per tract, and business
  establishments per resident. They are documented and removable.

It does **not** use leaderboard scores, earlier submissions, parameters fitted on the leaderboard, the
withdrawn layers or any copy of them, or the participant-posted Paradise scores.

## How it works

The 7 stages are printed as they run. The details are in [METHOD.md](METHOD.md).

1. **Data:** download and size-check every permitted file.
2. **Features:** measure the exact Overture side of each ratio. Also measure the evidence for each withdrawn
   reference: the Interstate/US/State route system for TIGER, the USFS building count, facility categories,
   names and alternates, the business-registry share of places, and mapping effort.
3. **Auxiliary models:** Poisson GLM vs gradient boosting, county-grouped 5-fold CV, predicting facility and
   registry density from context. The cross-fitted predictions feed stage 4.
4. **Simulator:** Monte Carlo of the withdrawn counts through the exact official composite.
5. **Calibration:** the Method of Simulated Moments on the published statistics, with a Gaussian prior on
   25 physical parameters.
6. **Validation:**
   - leave-one-target-out;
   - identifiability of every parameter;
   - prior sensitivity;
   - Monte Carlo stability;
   - a mechanism check run before any fitting: the share of tracts without an Overture route
     (0.222 / 0.564 / 0.365 / 0.306) matches the published undefined shares (0.21 / 0.55 / 0.37 / 0.28).
7. **Prediction:** E[score | data] at the most probable parameters, with 4,096 draws per tract.

## Options

```bash
python train.py --discovery     # Best Bias Discovery tables (discovery*.csv), see BIAS_DISCOVERY.md
python train.py --scorecard     # optional variant calibrated also on the public Bias Scorecard -> submission_scorecard.csv
python train.py --rebuild       # recompute cached features
```

`--scorecard` is off by default. The scorecard ratios come from the leaderboard (another team's entry),
not from the data.

## Results and limits

- **No labels:** the score is unknown until a file is uploaded. With no labels, 2 of 25 parameters are
  pinned by the published statistics; the rest rest on documented physical priors.
- **Prediction:** mean 0.0955 on graded tracts. The facility half dominates: fire stations are labelled in
  only 29% of tracts.
- **Reference point:** an earlier label-free model of this family scored 0.069455.
- **`--scorecard` variant** (`submission_scorecard.csv`): mean 0.0738.
  - It reproduces the 9 scorecard ratios within about 0.1 (rural 2.29, tribal 2.81).
  - It strains the published statistics: lowest road ratio 0.50 vs 0.71, establishments per resident ≈ 0.
  - Uploading both files shows which physics the truth prefers.
- **Main branch:** the leaderboard-calibrated pipeline is on `main`, best R33 at 0.062090669. This branch
  replaces it with a pipeline that can be fully explained and reproduced.

## Files

- `train.py`: the whole pipeline.
- `METHOD.md`: the technical write-up (data, components, edge cases, validation).
- `BIAS_DISCOVERY.md`: the Best Bias Discovery write-up.
- `requirements.txt`: NumPy, pandas, SciPy, DuckDB, scikit-learn.
- `submission.csv`, `submission_report.json`: outputs of `python train.py`.
- `discovery*.csv`: outputs of `--discovery`.
- `data/`: downloaded challenge data and cached features (git-ignored).
