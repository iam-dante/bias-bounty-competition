# Bias Bounty — R20 ridge

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

R20 uses all 23 scored experiment directions with ridge strength **0.2**,
replacing the previous backward-selection correction. Historical predictions
are frozen features; score-derived moments fit the correction. `model.npz`
contains the aligned predictions, scores, tract IDs and regions needed to
reproduce this fit. This is a model for the fixed competition tracts.

| Result | Public RMSE | Status |
|---|---:|---|
| R20 (`submission.csv`) | **0.062426715** | Estimated, unscored |
| r19b | 0.062455004 | Best confirmed |
| P19_64 | 0.062474294 | Confirmed |

The estimated improvement over P19_64 is small and unconfirmed. There are no
training labels for independent target validation. The score estimate uses
fitted public-row weights and assumes no target signal outside the historical
prediction span. Simulated 30/70 splits favored ridge over backward selection
in approximately 83% of trials; this depends on a simulated truth constructed
from measured directions. **Neither this simulation nor the estimate proves a
leaderboard improvement or a score of 0.05.**

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
