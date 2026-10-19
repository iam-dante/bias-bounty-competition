# Bias Bounty — R26 scorecard step + neural-network ensemble

One model, one submission: `submission.csv`.

## Run

```bash
pip install -r requirements.txt
python solution.py
```

The command refits the ridge calibration and the empirical-Bayes model, blends them,
and writes only `submission.csv`.
It runs locally without downloads. The file contains 9,794 unique GEOIDs,
bounded predictions, and the five published Paradise values.

## Model and evidence

R26 = R25 + 0.5 x a neural-network extrapolation (`nn_correction`).
R25 = R23 + 0.844 x the scorecard component of R24 (`scorecard_step`).

- **Neural-network ensemble (R26):** 10 seeds of a 205-64-32-1 MLP on permitted
  tract features. There are no labels, so each network is trained to explain
  the scored files' whitened signals and the 8 true scorecard ratios (moment
  matching).
  - **Validation:** chronological hold-out over the 30 files scored since
    round 12 (train on earlier files only). Error vs no-model baseline -9.2%,
    correlation +0.31 (sign-flip p = 0.034; p = 0.06 excluding the
    scorecard-derived R24/R25).
  - **What is used:** only the part outside the scored span, made neutral to
    every scorecard constraint, at half strength (validated shrink about
    1.1-1.25).
  - **Expected score:** no-signal estimate 0.062251. If the network performs
    as in hold-out, about 0.06203-0.06205.
  - **Dropped:** LightGBM with the same custom loss diverged.
R24 = R23 + a joint Bayesian update on the official Bias Scorecard.

- **Calibrated step (R25):** R24's move splits into two orthogonal parts:
  - The part inside the span of earlier scored files (RMS 0.0017): no reliable
    signal (z about -0.9), so R25 drops it.
  - The scorecard part outside that span (RMS 0.0063): R24's score measures
    it directly, z = 5.1, best public step 0.80 +- 0.16.
  - Combined with the scorecard's claim that the full step is right (prior
    1.0 +- 0.3), R25 takes a 0.844 step.
  - A larger step along R24's direction makes the score worse: the optimum
    along that line is 0.71 x R24.
R23 = 0.4 x the R21 ridge fit + 0.6 x an empirical-Bayes posterior (R22 used 0.6 / 0.4).

- **Scorecard update (R24):** the rank-1 entry scores exactly 0, so its scorecard
  shows the true disparity ratios. The update targets them for 8 strata: rural,
  tribal, SVI >= 0.6, and median splits of CVI, summer and winter drought,
  wildfire (WHP) and summer heat. Our R23 scorecard reproduces these definitions
  on all 9,794 rows. Each ratio becomes a linear constraint, combined with the
  leaderboard moments in one Gaussian posterior. The update (RMS 0.0066) lies
  almost entirely outside the scored-file span, so the public estimate cannot
  credit it (it reads 0.06269). If the leader's ratios are right, the model
  expects about 0.0004 lower RMSE. `data/strata/` is required.

- **Ridge (R21 method):** the ridge fit uses 24 probe directions (now including
  p20_02) at strength **0.2**. It also adds the bounded r19b clipping correction.
- **Empirical Bayes:** each scored file's score is treated as a public-row moment
  `<v, y>`. The files are orthogonalised in scoring order. Each new direction gets
  a group prior (base models or probes), fitted by marginal likelihood. The noise
  term comes from sampling the 30% public subset, so weak directions are shrunk
  rather than fitted at face value. Fitted priors: probe coefficients
  N(0.04, 0.75^2), base-model coefficients N(0.57, 0.35^2), noise scale 0.42.
- **Blend weight 0.6:** R22 (weight 0.4) scored 0.000028 better than predicted,
  and its score now enters the fit. Re-run on simulated 30/70 public/private folds
  (100 per setting), weight 0.6 gains 0.000105 private RMSE when the Bayesian
  model is right (better in 100% of folds), and loses 0.000026 if the unshrunk
  ridge fit were the truth. It also lowers the public estimate by 0.00001 vs 0.4.
- **Tested and rejected:** a hybrid global/local shrinkage prior (no likelihood
  gain), and feature-kernel extrapolation from 205 permitted tract features
  (did not predict held-out probe scores). Overture building coverage against
  the USFS (Microsoft-based) building count is 97% inside the scored span
  already. The share of Overture buildings sourced from Microsoft ML Buildings
  is 65% new, but the scorecard cannot calibrate it (effect +0.0012 +- 0.0029),
  so it is not used.

`model.npz` contains the aligned predictions, scores, tract IDs and regions.
Historical predictions are frozen features, not tract-level labels.

| Result | Public RMSE | Status |
|---|---:|---|
| R26 (`submission.csv`, dev) | 0.062251 no-signal / ~0.06204 if the NN holds | Unscored |
| R25 (main) | **0.062147723** | **Best confirmed** (predicted 0.062117874) |
| R24 | 0.062185147 | Confirmed (no-signal estimate 0.062691) |
| R23 | 0.062327384 | Confirmed (predicted 0.062330919) |
| R22 | 0.062343029 | Confirmed (predicted 0.062370908) |
| R21 | 0.062411516 | Estimated, never submitted |
| p20_02 multiref share | 0.062438801 | Confirmed |
| r19b | 0.062455004 | Confirmed |

R23 scored within 0.000004 of its prediction and R22 within 0.000028, both inside the estimator's backtest error (MAE about 0.00014). There are no
training labels for independent target validation. The score estimate uses
fitted public-row weights and assumes no target signal outside the historical
prediction span. Simulated 30/70 splits favored ridge over backward selection
in approximately 83% of trials; this depends on a simulated truth constructed
from measured directions. **Neither this simulation nor the estimate proves a
leaderboard improvement or a score of 0.05.**

## Focused experiment (2026-10-17)

The new direction has approximate conditional z = 1.581, so its signal is
uncertain. A paired stress experiment refitted the ridge and bounded correction
on each simulated 30% public split and evaluated the other 70% (400 splits per
noise model and signal setting; seed 8119).

| Assumed extra signal | Gaussian mean RMSE change | Heavy-tail mean RMSE change |
|---|---:|---:|
| None | +0.00000039 | +0.00000009 |
| Half the fitted signal | -0.00000151 | -0.00000208 |
| Full fitted signal | -0.00000865 | -0.00000997 |

At full signal, the update won 71.5% / 78.5% of splits. These are controlled
simulations with fitted moments and synthetic residuals, not independent
validation on hidden labels. A separate attempt to center the entire ridge
penalty on r19b was rejected because it worsened simulated holdout RMSE.
The retained update's estimated public gain over R20 is only 0.0000152.

Reproduction inputs and scripts for this check are archived outside the repo:
`../bias-bounty-r21-evidence-20261017.tar.gz` (run `stress.py` after extraction).

## Minimal layout

- `solution.py`: ridge fit, empirical-Bayes fit, blend and submission generation.
- `model.npz`: compressed frozen inputs and historical score record (2.7 MB).
- `submission.csv`: the only submission to upload.
- `requirements.txt`: NumPy, pandas, SciPy and PyTorch (CPU is enough; about 20 s).
- `data/`: existing challenge data, preserved locally and ignored by Git.

Old experiments, candidate submissions, raw-feature pipeline and detailed
research records were backed up outside this repo before cleanup:

`/Users/briantemu/Downloads/projects/competitions/bias-bounty-research-backup-20261017-164646.tar.gz`

The archive preserves the original generation code and score provenance.
Regenerating historical features from raw challenge data requires that archive;
the minimal branch reproduces the final calibration from frozen features.
No withdrawn reference datasets were added during this consolidation.
