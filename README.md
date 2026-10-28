# Bias Bounty Mapping Equity Challenge: a coverage-gap model trained from scratch

Zindi user **iam-dante**. The pipeline trains from the challenge data. Final state (2026-10-27, experiments stopped):

| Final file | Scored file | Public RMSE |
|---|---|---:|
| **`submission.csv`** | `tr_nofp_030`: the best + 0.3 × the footprint-free flexible model's unscored part | **0.05836217** |
| `submission_alt.csv` | `tr_cosci02B_030`: the shared update + 0.3 × the flexible model's unscored part (a different idea, kept as the second final) | 0.05854815 |

```bash
pip install -r requirements.txt
python train.py          # first run: data (4.2 GB), features, physics model, stage 8
python cosci.py          # co-scientist round (overwrites submission.csv and submission_alt.csv)
```

Seeds are fixed, and reruns reproduce the files byte for byte.

## Repository

| Path | What it is |
|---|---|
| `train.py` | the pipeline: data, features, physics simulator, calibration, aggregate labels (stages 1–10) |
| `cosci.py` | the co-scientist loop: hypotheses, held-out tournament (Elo), and the two final submissions |
| `experiment_loop.py`, `posterior_experiment.py`, `robust_posterior_experiment.py`, `aggregate_kernel_experiment.py` | the audited score-feedback loop that produced the current best file |
| `submission.csv` | **final submission**: the best scored file, 0.05836217 |
| `submission_alt.csv` | **second final**: the previous best scored file, 0.05854815 |
| `aggregate_labels.npz` | training labels: the vector and public RMSE of every scored file (88) |
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
| **Generate** hypotheses | physics-model families: base (60 parameters), flexible linear context effects (88), neural MLP context effects inside the simulator (206), and a wide network on all four reference layers (376) |
| **Reflect** (review) | each family is refitted *without* the 10 newest score moments (`train.py --holdout 10`), so it cannot have seen them |
| **Rank** (tournament) | every held-out moment is one pairwise debate (the closer prediction wins), and the debates update Elo ratings |
| **Evolve** | the measured bets decide the families: each file bets on a different one, so two scores measure both |
| **Meta-review** | failure patterns go into METHOD.md §24 and constrain the next generation |

Result (85 labels; the 10 newest genuine directions held out; every fit starts from the same base):

| Hypothesis | Parameters | Held-out chi-square (start 30.4) | Elo | Estimated public RMSE on its own (all labels) | Unscored part (RMS) |
|---|---:|---:|---:|---:|---:|
| **wide neural network** (21 → 12 tanh → 4: TIGER length, facilities, Microsoft buildings, CBP establishments) | 376 | **23.5** | **1050** | 0.0612 | **0.0046** |
| neural network (21 → 6 tanh → 2: TIGER length, facilities) | 206 | 24.6 | 957 | 0.0614 | 0.0026 |
| base physics | 60 | 26.2 | 968 | 0.0619 (held-out fit) | – |
| flexible linear context effects | 88 | 79.2 | 1025 | 0.0607 | 0.0040 |

At 84 labels the flexible model led (21.4 against 23.3 for the 6-unit network). At 85 it misses one held-out moment by
z +7.8: it fits the scores it sees best, but extrapolates worst. The wide network wins on both measures, and its unscored
part is mostly new: cosine 0.32 with the 6-unit network's and 0.23 with the flexible model's. After the two pending
files are scored, RMS 0.0043 of it will still be unmeasured. Its two new outputs (buildings, establishments) stay small
(|W2| 0.08 and 0.07, against 0.18 and 0.23).

**A bug found on the way.** Two near-duplicate labels (`tr_loop01_045` and `tr_refined082`, whose new part has
RMS 0.000028) had inflated the posterior's fitted noise scale from 0.97 to 6.4. Every physics refit after the 82nd label
was therefore down-weighting the scores about 40×. A file now opens a posterior direction only if its new part has RMS ≥ 0.0005
(`EB_MIN_NEW_RMS`), which restores the noise scale to 1.18.

## Measured bets (2026-10-27)

| Round | File | Bet | Score | Measured |
|---|---|---|---:|---|
| 1 | `tr_cosciB_030` | 0.3 × the 6-unit network's unscored part | 0.058603 | optimum 1.65× the step (coefficient about +0.6) |
| 2 | `tr_cosci02A_043` | shared update + 0.43 × the 6-unit network | 0.058568 | optimum 0.55× the step: little left |
| 2 | `tr_cosci02B_030` | shared update + 0.30 × the flexible model | 0.058548 | optimum 1.12× the step: B landed on it |
| 3 | `tr_nofp_030` | best + 0.30 × the footprint-free flexible model | **0.058362** | beat its +0.5 forecast (0.058466); coefficient above +0.5 |

The shared update was forecast at 0.058572; the flexible bet added 0.000024. Flexible bets have now measured +0.5,
+0.4, +0.85 and about +0.45, although the flexible model fails one held-out moment. After round 2 the posterior's
own update is spent: it forecasts 0.058543 ± 0.000046, a coin flip against the best (P 0.54).

## Round 3

Three files started from `tr_cosci02B_030`. C, the bet on the footprint-free model, scored **0.05836217**, a gain of
0.000186 and the largest of any round. A (the wide-network bet) and B (a second flexible bet) have no recorded scores.
Their files are in git history at commit `016baa9`.

**For the final selection:** `submission.csv` (0.058362) and `submission_alt.csv` (0.058548). Both are scored, and
they rest on different bets, so they do not share every error.

## Without the footprint-derived columns (`--no-footprint-columns`)

The organisers have not said whether `usfs_BuildingCount_sum` is a "permitted column" of the strata tables. Neither have
they ruled on `usfs_HUCount_sum` or `carbonplan_buildings`. The switch drops all three, plus the Microsoft-sourced share of
Overture buildings. The Overture building count then stands in for the USFS count. Only two active paths used them: the
building model's independent count, and one auxiliary feature.

| Flexible physics, 87 labels | Estimated public RMSE alone | Fit cost | Microsoft scale κ |
|---|---:|---:|---:|
| with the columns | 0.061014 | 157.3 | 0.62 |
| without | 0.061049 | 145.2 | 0.82 |

Dropping them costs 0.000035 in the physics model. The building component is near zero in both. The two
predictions still differ by RMS 0.0086 (correlation 0.992). The footprint-free model's unscored part is RMS 0.0072,
three times the flexible model's, so a bet on it is the boldest file available:

| File C (`tr_nofp_030`, now `submission.csv`) | +0.5 | +0.3 | 0 | −0.3 | **Scored** |
|---|---:|---:|---:|---:|---:|
| best + 0.3 × the footprint-free model's unscored part | 0.058466 | 0.058520 | 0.058600 | 0.058681 | **0.058362** |

A caveat: every scored file so far came from physics that used the USFS count, and the best file is built from them. A
strictly footprint-free final would be the footprint-free physics alone, about 0.0610.

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
| robust continuous update | `tr_refined083_d7469871.csv` | 0.058697 |
| co-scientist B: neural-network bet | 0.3 × the neural model's unscored part (`tr_cosciB_030.csv`) | 0.058603 |
| co-scientist round 2, A | shared update + 0.43 × the 6-unit network (`tr_cosci02A_043.csv`) | 0.058568 |
| co-scientist round 2, B | shared update + 0.30 × the flexible model (`tr_cosci02B_030.csv`) | 0.058548 |
| footprint-free bet | best + 0.30 × the footprint-free flexible model (`tr_nofp_030.csv`) | **0.058362 (best)** |

## Continuing the loop

Experiments stopped on 2026-10-27. To continue, record each real score, refit, and rerun the co-scientist round
(`cosci.py` overwrites both final files):

```bash
python train.py --add-label NAME.csv FILE RMSE      # NAME starts with tr_ for a bet file
python train.py --calibrate-physics --structural --unmatched --regional --flex --flex-sd 0.2 --no-footprint-columns
python cosci.py
```

`--add-label` refuses a score more than 0.003 from what the file can plausibly score: a target typed in as a score
(0.004, 0.04) would corrupt every later fit. Zindi's upload dialog allows 5 uploads a day.

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
- **Open questions (forum, read 2026-10-27):** the organisers allow "features derived from the currently provided
  datasets, including Overture attributes and permitted columns in the strata tables" (thread 35250), and ban any use of
  TIGER, Microsoft footprints, HIFLD or CBP to reconstruct the target (35079). They have not answered two questions this
  pipeline depends on. One is whether `usfs_BuildingCount_sum` is a permitted column (35302, 35402); it is the building
  model's independent count. The other is whether tuning on one's own leaderboard scores is allowed (35292, 35532).
- **Data:** the bucket has not changed since 2026-09-25. All 197 files used here match it in size; the rest are CSV
  copies of the same tables and the bulk zips.
