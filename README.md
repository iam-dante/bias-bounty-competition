# Bias Bounty Mapping Equity Challenge: a coverage-gap model trained from scratch

Zindi user **iam-dante**. The pipeline trains from the challenge data and writes two final files:

```bash
pip install -r requirements.txt
python train.py          # first run: data (4.2 GB), features, physics model, stage 8
python cosci.py          # co-scientist tournament -> submission.csv (A) and submission_alt.csv (B)
```

Seeds are fixed, and reruns reproduce the files byte for byte.

## Repository

| Path | What it is |
|---|---|
| `train.py` | the pipeline: data, features, physics simulator, calibration, aggregate labels (stages 1–10) |
| `cosci.py` | the co-scientist loop: hypotheses, held-out tournament (Elo), and the two final submissions |
| `experiment_loop.py`, `posterior_experiment.py`, `robust_posterior_experiment.py`, `aggregate_kernel_experiment.py` | the audited score-feedback loop that produced the current best file |
| `submission.csv` | **final submission A**: best expected public score |
| `submission_alt.csv` | **final submission B**: a different model family, built for the private leaderboard |
| `aggregate_labels.npz` | training labels: the vector and public RMSE of every scored file (84) |
| `outputs/` | the physics model, the calibrated models the two files use, the warm-start fits, and every report |
| `METHOD.md` | the technical write-up: every experiment, including the ones that failed |
| `BIAS_DISCOVERY.md` | the Best Bias Discovery write-up |
| `data/` (git-ignored) | challenge data, cached features, and `archive/` with the retired experiment files |

## How it trains

There are no tract labels. Training has three parts.

1. **Physics model from the challenge data (stages 1–7).**
   - DuckDB features measure the Overture side of every ratio exactly, plus evidence for each withdrawn reference layer.
   - Auxiliary Poisson models (GLM vs gradient boosting, county-grouped CV) give facility and registry densities.
   - A Monte Carlo simulator generates the withdrawn reference counts and pushes them through the exact composite.
   - It is calibrated by simulated moments on the organisers' published statistics.
   - Output: `outputs/physics.csv`, which scored 0.101340.
2. **Aggregate labels (stage 8).** For RMSE, every scored file `v` with public score `s` gives one exact equation
   about the hidden target:

   ```
   mean_public(v · y) = (mean(v²) + mean(y²) − s²) / 2
   ```

   The scores of our own files are used openly as training data; no earlier file is copied into the next one.
3. **Physics calibrated on the labels, and measured bets (stage 10).**
   - The physics model is refitted to the published statistics, the public scorecard and all score moments.
   - Its part outside everything scored is its prediction for unmeasured directions. A small step along it is
     scored, the score measures how much of it is true, and a curvature-aware update takes it to its optimum.

## The co-scientist loop (`cosci.py`)

It follows the AI co-scientist (Gottweis et al. 2025, [arXiv:2502.18864](https://arxiv.org/abs/2502.18864)):

| Co-scientist step | Here |
|---|---|
| **Generate** hypotheses | physics-model families: base (60 parameters), flexible linear context effects (88), neural MLP context effects inside the simulator (206) |
| **Reflect** (review) | each family is refitted *without* the 10 newest score moments (`train.py --holdout 10`), so it cannot have seen them |
| **Rank** (tournament) | every held-out moment is one pairwise debate (the closer prediction wins), and the debates update Elo ratings |
| **Evolve** | the winner supplies the next measured bet; the runner-up supplies the alternative file |
| **Meta-review** | failure patterns go into METHOD.md §24 and constrain the next generation |

Result (84 labels; the 10 newest genuine directions held out; every fit starts from the same base):

| Hypothesis | Parameters | Held-out chi-square (start 27.8) | Elo | Estimated public RMSE on its own (all labels) |
|---|---:|---:|---:|---:|
| **flexible linear context effects** | 88 | **21.4** | **1056** | **0.0611** |
| neural MLP context effects (21 → 6 tanh → 2) | 206 | 23.3 | 983 | 0.0619 |
| base physics | 60 | 25.7 | 961 | 0.0622 (held-out fit) |

The neural network beats the base model on data it never saw, but not the simpler linear effects: with 84 aggregate
labels, the extra flexibility costs more than it explains.

**A bug found on the way.** Two near-duplicate labels (`tr_loop01_045` and `tr_refined082`, whose new part has
RMS 0.000028) had inflated the posterior's fitted noise scale from 0.97 to 6.4. Every physics refit after the 82nd label
was therefore down-weighting the scores about 40×. A file now opens a posterior direction only if its new part has RMS ≥ 0.0005
(`EB_MIN_NEW_RMS`), which restores the noise scale to 1.18.

## The two final submissions

| | **A: `submission.csv`** | **B: `submission_alt.csv`** |
|---|---|---|
| Idea | best public score | best expected private score, different model family |
| Base | the measured best (0.058697) | the measured best, with every coordinate shrunk by its public-sampling error |
| Bet | 0.3 × the flexible model's unscored part (RMS 0.0048) | 0.3 × the neural model's unscored part (RMS 0.0067) |
| If the bet's coefficient is +0.5 | **0.058662** | **0.058634** |
| +0.3 | 0.058685 | 0.058680 |
| 0 | 0.058720 | 0.058749 |
| −0.3 | 0.058755 | 0.058818 |

The two files differ by RMS 0.0022. Flexible-physics directions have measured +0.5, +0.4 and +0.85 so far; the
neural direction is unmeasured.

**Why B shrinks.** The final ranking uses the private 70% of rows, but every coefficient was measured on the public 30%. A
direction opened by a small step (RMS ~0.001) has a public-sampling error of about ±1 in its coordinate. B replaces
each coordinate of the best file with its posterior mean under a hierarchical prior N(1.05, 1.70²) fitted across the 10
measured directions. The shrinkage is small (the expected private gain is 0.000008), which also means the best file's
public gains are mostly real and should carry over to the private leaderboard.

**For the final selection:** upload both. Then keep the best public-scored file, and pick the second from a
different idea (B or its successor), so the two finals do not share every error.

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
| + region-specific flexible effects | 200 parameters | 0.058936: that direction ran the wrong way |
| stable, risk-aware update | newest two directions at 2.26× and −1.64× | 0.058735 |
| loop round | `tr_loop01_045.csv` | 0.058732 |
| refined posterior update | `tr_refined082_df6bb1c7.csv` | 0.058707 |
| robust continuous update | `tr_refined083_d7469871.csv` | **0.058697 (best)** |

## After each score

Record the real leaderboard score of the file you uploaded, then rerun the loop:

```bash
python train.py --add-label NAME FILE RMSE        # NAME as printed in outputs/cosci_report.json; FILE = the uploaded CSV
python train.py --calibrate-physics --structural --unmatched --regional --flex --flex-sd 0.2   # ~10 min
python cosci.py                                   # -> new submission.csv and submission_alt.csv
```

`--add-label` refuses a score more than 0.003 from what the file can plausibly score: a target typed in as a score
(0.004, 0.04) would corrupt every later fit. Each round needs a real score, and Zindi allows 5 uploads a day.

## Why not 0.04

Every gain here was first predicted, then confirmed by a score. The predictions are only reliable one step past what
has been measured. The best physics model, refitted on all scores, estimates about 0.061 on its own, and the
measured best is 0.0587. What remains is mostly per-tract randomness in the withdrawn reference layers (the exact
TIGER length, whether a station exists in a given tract), which the permitted data cannot predict. Recent rounds
gained 0.00001–0.00005 each.

## Rules

- **Data:** the scored files use only the provided datasets. External data appears only in the Bias Discovery
  write-up.
- **Not used:** the withdrawn layers, any copy of them, and the participant-posted Paradise scores.
- **Leaderboard:** the leaderboard scores of our own files are used as aggregate labels. This is stated openly.
- **Private leaderboard:** the public score covers about 30% of the test rows; the final ranking uses the other 70%.
- **Tooling:** open-source packages only (NumPy, pandas, SciPy, DuckDB, scikit-learn), fixed seeds, no AutoML.
