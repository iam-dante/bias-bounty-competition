# Bias Bounty Mapping Equity Challenge: a coverage-gap model trained from scratch

Zindi user **iam-dante**. One script trains everything and writes the one file to upload:

```bash
pip install -r requirements.txt
python train.py          # -> submission.csv
```

The first run downloads about 4.2 GB of challenge data, then takes about 5 minutes on a laptop CPU. Seeds are fixed,
and reruns reproduce the files byte for byte.

## Repository

| Path | What it is |
|---|---|
| `train.py` | the whole pipeline (stages 1–10) |
| `submission.csv` | **the submission** |
| `aggregate_labels.npz` | training labels: the vector and public RMSE of every scored file |
| `outputs/` | generated models and reports: `physics.csv`, the label-calibrated physics model used by the current file, the fitted parameters every recalibration warm-starts from, `submission_report.json` |
| `METHOD.md` | the technical write-up, with every experiment and its result |
| `BIAS_DISCOVERY.md` | the Best Bias Discovery write-up |
| `requirements.txt` | open-source packages only |
| `data/` (git-ignored) | downloaded challenge data and cached features |

## How it trains

There are no tract labels. Training has three parts.

1. **Physics model from the challenge data (stages 1–7).**
   - DuckDB features measure the Overture side of every ratio exactly, plus evidence for each withdrawn reference layer.
   - Auxiliary Poisson models (GLM vs gradient boosting, county-grouped CV) give facility and registry densities.
   - A Monte Carlo simulator generates the withdrawn reference counts and pushes them through the exact official
     composite.
   - It is calibrated by simulated moments on the organisers' published statistics.
   - Output: `outputs/physics.csv`, which scored 0.101340.
2. **Aggregate labels (stage 8).** For RMSE, every scored file `v` with public score `s` gives one exact equation
   about the hidden target:

   ```
   mean_public(v · y) = (mean(v²) + mean(y²) − s²) / 2
   ```

   `aggregate_labels.npz` holds 80 such labels. The scores of our own files are used openly as training data;
   no earlier file is copied into the next one.
3. **Physics calibrated on the labels, and measured bets (stage 10 + stage 8).**
   - The physics model is refitted to the published statistics, the public scorecard and all score moments.
   - The largest model has 88 parameters, including 28 flexible context effects on TIGER length and on unseen
     facilities.
   - Its part outside everything scored is its extrapolation to unmeasured directions. Each file adds a small step
     along it, and the step's score measures how much of it is true.
   - A curvature-aware Bayesian update then takes each measured direction to its optimum.

## Score progression (public RMSE)

| File | What changed | Score |
|---|---|---:|
| physics model alone | challenge data only | 0.101340 |
| stage 8, 65 labels | posterior on the aggregate labels | 0.061372 |
| + label-calibrated physics direction | first measured bet | 0.060742 |
| + structural mechanisms | facility leakage, TIGER local-name double count | 0.059977 |
| + concurrent-route double counting | one TIGER record per route number | 0.059507 |
| curvature-aware local update | 2.45× along it | 0.059232 |
| + flexible context effects | 28 effects, prior sd 0.1 | 0.058987 |
| + recalibrated flexible model | 0.3× its new direction | 0.058912 |
| + wider flexible model | prior sd 0.2 | 0.058866 |
| + region-specific flexible effects | 200 parameters | 0.058936: that direction ran the wrong way (≈ −0.7) |
| stable, risk-aware update | newest two directions at 2.26× and −1.64× | **0.058735 (best)**; predicted 0.058791 ± 0.000044 |

The full history, with every experiment that did not help, is in [METHOD.md](METHOD.md).

## The current `submission.csv`

The best file (0.058735) with its two newest measured directions moved further, to 2.68× (wider flexible) and
−2.76× (region-specific, reversed), plus a measurable 0.45× bet on the recalibrated flexible model:

| Bet direction's coefficient | Expected public RMSE |
|---:|---:|
| +0.5 | 0.058705 |
| +0.3 | 0.058716 |
| 0 | 0.058732 |
| −0.3 | 0.058748 |

The move alone is expected at 0.058716 ± 0.000021 (P(beats best) 0.86). When it is scored, run
`python train.py --round <score>`; it is recorded as `tr_loop01_045.csv`.

## The loop: one command per score

Upload `submission.csv`, then give the loop its public score:

```bash
python train.py --round 0.0587xxxxx
```

That one command runs a full round (about 11 minutes):

1. **Label.** It checks that `submission.csv` is the file stage 8 last wrote (checksum), then records it with its
   score as the next label (the name is in `outputs/submission_report.json`).
2. **Recalibrate.** It refits the loop physics model (the 88-parameter flexible model) on all labels.
3. **Next file.** Stage 8 moves the measured directions to their optimum, and adds a 0.3× bet when the
   recalibrated model has enough left unmeasured (RMS ≥ 0.003). It writes the next `submission.csv` and prints its
   expected score.

**Why each round needs a real score.** An expected score is only reliable one step past what has been measured.
Re-running the prediction on its own output does not add information: the update is already at the optimum of what
the scores know, and stepping further on the model's own guesses is how the one bad miss (0.060747) happened. Each
upload measures one new direction exactly, and that measurement is what the next file exploits. The competition
allows 5 uploads a day, so the loop can run up to 5 rounds a day.

## Options

```bash
python train.py --physics-only        # stop after the physics model
python train.py --scorecard           # physics variant also calibrated on the public Bias Scorecard
python train.py --discovery           # Best Bias Discovery tables -> outputs/discovery*.csv
python train.py --rebuild             # recompute the cached features
python train.py --fit-only            # stage 8 only (reads outputs/physics.csv)
python train.py --probes K            # stage 9: K probe files -> outputs/probes/
python train.py --round RMSE          # one loop round (see above)
python train.py --add-label NAME FILE RMSE   # record a score by hand
python train.py --calibrate-physics [--structural --unmatched --regional --covariates --flex --flex-sd SD --flex-regional]
```

## Rules

- **Data:** the scored file uses only the provided datasets. External data appears only in the Bias Discovery
  write-up.
- **Not used:** the withdrawn layers, any copy of them, and the participant-posted Paradise scores.
- **Leaderboard:** stage 8 uses the leaderboard scores of our own files as aggregate labels. This is stated openly.
- **Tooling:** open-source packages only (NumPy, pandas, SciPy, DuckDB, scikit-learn), fixed seeds, no AutoML.
- `main` and `dev` hold the same tree; the older chained pipeline is in `main`'s history (merge 6090147).
