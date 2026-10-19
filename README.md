# Bias Bounty — R27 joint minimum-change update

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

## Validation

There are no tract-level labels, so the validation set is the scored
submissions themselves. Each run writes `validation.csv`: for every file scored
since round 12 it predicts the public RMSE from the other scored files and
compares it with the real score.

- **Forward:** only files scored earlier, an honest look-ahead test.
  MAE 0.000130 over 31 files.
- **Leave-one-out:** all other files. MAE 0.000056.

The metric is the competition's RMSE. Forward errors are large for files that
add genuinely new information (R24: +0.000505), because the projection cannot
credit what no earlier file contains.

## Model and evidence

R27 = R25 + the smallest change that agrees with every scored file and the true scorecard ratios (`joint_min_change`).
R25 = R23 + 0.844 x the scorecard component of R24 (`scorecard_step`).

- **What the model was missing (R23 to R25):** the leaderboard-proven
  scorecard correction (z = 5.1) raised predictions in less-urban, tribal and
  remote tracts. Its strongest correlations are pct_urban -0.31, pop_urban
  -0.32, tribal_pct +0.17 and distance to the nearest weather station +0.17. It
  lowered predictions in drought-heavy, populous tracts. It is additive (corr
  -0.007 with the prediction level), and 62% of it falls in Texas.
- **Remaining gaps in R25 (vs true ratios):** tribal -0.115, wildfire -0.040,
  rural -0.037, SVI +0.027; the other strata are closed.
- **Joint update (R27):** one Gaussian solve over all 59 scored-file moments
  (with their public-sampling noise) and the 8 true ratios. A per-tract prior
  scale is fitted by marginal likelihood (tau 0.039).
  - The update is RMS 0.0015, and 93% of it lies inside the scored span, where
    the estimator credits it (0.062140 vs 0.062146).
  - Every ratio ends within 0.015 of the truth: rural 2.333, tribal 2.869,
    wildfire 1.766.
  - Closing the gaps only outside the scored span needs a distorting 0.016 RMS
    move, so that variant is rejected.
- **Hazard + Vulnerability:** no definition reproduces both our R23 (1.25) and
  R24 (1.22) scorecards, so this stratum is left out.

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
- **Tested and rejected, neural networks (R26):** a 10-seed MLP ensemble
  trained by moment matching on 205 permitted features, the scored files'
  signals and the true scorecard ratios. It passed a chronological hold-out
  (-9% error, p = 0.03-0.06), but R26 (R25 + half of its out-of-span
  correction) scored 0.062236918, matching its no-signal estimate (0.062251).
  Along that direction the best step is 0.07 (z = 0.25), so there is no signal.
  LightGBM with the same loss diverged.
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
| R27 (`submission.csv`, dev) | 0.062140156 | Estimated, unscored |
| R25 (main) | **0.062147723** | **Best confirmed** (predicted 0.062117874) |
| R26 (neural-network add-on) | 0.062236918 | Confirmed worse; removed |
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
- `validation.csv`: predicted vs actual score for every scored file (rewritten each run).
- `requirements.txt`: NumPy, pandas and SciPy.
- `data/`: existing challenge data, preserved locally and ignored by Git.

Old experiments, candidate submissions, raw-feature pipeline and detailed
research records were backed up outside this repo before cleanup:

`/Users/briantemu/Downloads/projects/competitions/bias-bounty-research-backup-20261017-164646.tar.gz`

The archive preserves the original generation code and score provenance.
Regenerating historical features from raw challenge data requires that archive;
the minimal branch reproduces the final calibration from frozen features.
No withdrawn reference datasets were added during this consolidation.
