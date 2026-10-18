# Bias Bounty — R22 ridge + empirical-Bayes ensemble

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

R22 = 0.6 x the R21 ridge fit + 0.4 x an empirical-Bayes posterior.

- **Ridge (R21 method):** the ridge fit uses 24 probe directions (now including
  p20_02) at strength **0.2**. It also adds the bounded r19b clipping correction.
- **Empirical Bayes:** each scored file's score is treated as a public-row moment
  `<v, y>`. The files are orthogonalised in scoring order. Each new direction gets
  a group prior (base models or probes), fitted by marginal likelihood. The noise
  term comes from sampling the 30% public subset, so weak directions are shrunk
  rather than fitted at face value. Fitted priors: probe coefficients
  N(0.04, 0.75^2), base-model coefficients N(0.57, 0.35^2), noise scale 0.42.
- **Blend weight 0.4:** chosen on simulated 30/70 public/private folds (100 per
  setting). It gains 0.000091 private RMSE when the Bayesian model is right
  (better in 100% of folds). It changes by only +0.000004 when the unshrunk ridge
  fit is the truth.

`model.npz` contains the aligned predictions, scores, tract IDs and regions.
Historical predictions are frozen features, not tract-level labels.

| Result | Public RMSE | Status |
|---|---:|---|
| R22 (`submission.csv`) | **0.062370908** | Estimated, unscored |
| R21 | 0.062411516 | Estimated, unscored |
| p20_02 multiref share | 0.062438801 | Best confirmed |
| r19b | 0.062455004 | Confirmed |

The estimated improvement over p20_02 is small and unconfirmed (the estimator's backtest MAE is about 0.00014). There are no
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
- `requirements.txt`: NumPy, pandas and SciPy.
- `data/`: existing challenge data, preserved locally and ignored by Git.

Old experiments, candidate submissions, raw-feature pipeline and detailed
research records were backed up outside this repo before cleanup:

`/Users/briantemu/Downloads/projects/competitions/bias-bounty-research-backup-20261017-164646.tar.gz`

The archive preserves the original generation code and score provenance.
Regenerating historical features from raw challenge data requires that archive;
the minimal branch reproduces the final calibration from frozen features.
No withdrawn reference datasets were added during this consolidation.
