# Bias Bounty — R21 ridge

One model, one submission: `submission.csv`.

## Run

```bash
pip install -r requirements.txt
python solution.py
```

The command refits the fixed ridge calibration and writes only `submission.csv`.
It runs locally without downloads. The file contains 9,794 unique GEOIDs,
bounded predictions, and the five published Paradise values.

## Model and evidence

R21 keeps the 23-direction ridge fit at strength **0.2** and adds one bounded
correction: the part of the best scored r19b prediction outside that fit's
existing feature span. This captures additional information from clipping.
Positive-part shrinkage suppresses weak or negative evidence, and the added
coefficient is capped at 1; the current fit reaches that cap. Historical predictions
are frozen features; score-derived moments fit the correction. `model.npz`
contains the aligned predictions, scores, tract IDs and regions needed to
reproduce this fit. This is a model for the fixed competition tracts.

| Result | Public RMSE | Status |
|---|---:|---|
| R21 (`submission.csv`) | **0.062411516** | Estimated, unscored |
| R20 | 0.062426715 | Estimated, unscored |
| r19b | 0.062455004 | Best confirmed |
| P19_64 | 0.062474294 | Confirmed |

The estimated improvement over P19_64 is small and unconfirmed. There are no
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

- `solution.py`: complete ridge fitting and submission generation.
- `model.npz`: compressed frozen inputs and historical score record (2.7 MB).
- `submission.csv`: the only submission to upload.
- `requirements.txt`: NumPy and pandas.
- `data/`: existing challenge data, preserved locally and ignored by Git.

Old experiments, candidate submissions, raw-feature pipeline and detailed
research records were backed up outside this repo before cleanup:

`/Users/briantemu/Downloads/projects/competitions/bias-bounty-research-backup-20261017-164646.tar.gz`

The archive preserves the original generation code and score provenance.
Regenerating historical features from raw challenge data requires that archive;
the minimal branch reproduces the final calibration from frozen features.
No withdrawn reference datasets were added during this consolidation.
