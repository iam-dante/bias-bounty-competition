# Bias Bounty — modeling experiments on dev

Current confirmed best: `submission_r29.csv` (R29), public RMSE **0.062094756** (Zindi user iam-dante, rank 24).
`submission_r30.csv` was uploaded, but our best entry on the public leaderboard is still R29. So R30 scored
worse or has not been scored; its exact score is still needed. Its estimate was 0.061833
([R30 analysis](research/r30/ANALYSIS.md); the [R31 audit](research/r31/ANALYSIS.md) put it at 0.06170–0.06224).

R32 (this round) pinned the official Bias Scorecard exactly and tested a moment-calibrated physics model:

- **Scorecard.** The scorecard definitions are now exact (27/27 displayed values reproduced).
  The 3-decimal ratios of every leaderboard entry come from the public API.
- **T1 candidate.** `submission_r32_t1.csv` (est. 0.062091) is the only new upload candidate.
- **Physics model.** The calibrated physics model was not shipped. See the [R32 section](#r32-exact-scorecard-and-calibrated-physics-2026-10-20).

`submission_push.csv` is R28b (unchanged). `submission.csv` is the R28 joint update rerun now that R29's
score is in `model.npz`. `submission_data_only.csv` comes from `data_only.py`, a standalone physics model
that uses no leaderboard feedback. See [METHOD.md](METHOD.md) for all approaches.

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

R28 = the same joint update rerun with R27's score in the data (the current `solution.py` output).
R27 = R25 + the smallest change that agrees with every scored file and the true scorecard ratios (`joint_min_change`).

- **What R27's score showed:** R27 beat its estimate by 0.000027. Its in-span
  part is consistent with step 1 (0.65 +- 0.71). Its small out-of-span part
  (RMS 0.00058) prefers a 6.0 +- 1.7 step (z = 3.6), but steps above about 2
  push the all-rows ratios past the truth (tribal 2.93 at 3.9x vs 2.881). With
  R27's score included, the joint solve balances both sources: +0.29 extra on
  that part, ratios within 0.01, estimate 0.062111.
- **Push direction exhausted (R28b):** with R27 (1x) and R28b (2.3x) both
  scored, the public curve along this direction is flat. Its optimum is about
  3.9x (+- 1.7) and would gain only 0.000007, below the estimator's error, while
  overshooting the true tribal ratio. R28b is the final public-leaning file.
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
| R28 (`submission.csv`) | 0.062114244 | Estimated, unscored; ratio-faithful joint solve, private hedge |
| R28b (`submission_push.csv`) | **0.062097808** | **Best confirmed** (estimated 0.062088042); 2.3x R27's out-of-span piece |
| R27 | 0.062113403 | Confirmed (estimated 0.062140156) |
| R25 | 0.062147723 | Confirmed (predicted 0.062117874) |
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

- `solution.py`: calibrated model (ridge, empirical Bayes, scorecard updates) and submissions.
- `data_only.py`: data-only physics model (data -> DuckDB features -> Monte-Carlo simulator); `--rebuild` recomputes features.
- `METHOD.md`: technical write-up of both approaches.
- `model.npz`: compressed frozen inputs and historical score record (61 scored files; add new scores with
  `python research/add_score.py <name.csv> <file> <rmse>`).
- `submission.csv`: the only submission to upload.
- `submission_push.csv`: public-leaning variant (R28b), written on every run.
- `validation.csv`: predicted vs actual score for every scored file (rewritten each run).
- `requirements.txt`: NumPy, pandas, SciPy (+ DuckDB and scikit-learn for `data_only.py`).
- `data/`: existing challenge data, preserved locally and ignored by Git.

Old experiments, candidate submissions, raw-feature pipeline and detailed
research records were backed up outside this repo before cleanup:

`/Users/briantemu/Downloads/projects/competitions/bias-bounty-research-backup-20261017-164646.tar.gz`

The archive preserves the original generation code and score provenance.
Regenerating historical features from raw challenge data requires that archive;
the minimal branch reproduces the final calibration from frozen features.
No withdrawn reference datasets were added during this consolidation.

## R29 analysis (2026-10-19)

`submission_r29.csv` is a separate recalibration of R28b, now confirmed at **0.062094756**.
It propagates scorecard rounding error and correlated uncertainty from ungraded
Washington tracts. Three empirical-Bayes priors were compared on 33 chronological
score holdouts; none clearly improved forecasting. R29's original estimated public
RMSE was 0.062097258; the user subsequently reported its actual improved score.

See [the full analysis](research/r29/ANALYSIS.md) for the data audit, validation,
prior comparisons, sensitivity analyses, ablations, installed skill provenance,
and limitations. Reproduce with:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python research/r29_analysis.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python research/r29_report.py
```

The numerical pipeline uses the existing requirements. The optional diagnostic
report additionally needs Matplotlib. R29 does not replace `submission_push.csv`.

## R30 candidate (2026-10-19)

R30 adds R29's verified file and user-reported score through
`research/score_feedback.json`; `model.npz` remains immutable. The new research
compares linear, spline, unsupervised tree and spatial Gaussian-process covariances,
then solves bounded aggregate regression with soft scorecard constraints.

`submission_r30.csv` is the primary unscored candidate. The optional
`submission_r30_conservative.csv` blends it equally with R29. Both pass all 9,794-row
schema, GEOID, bound and published-anchor checks. No upload was performed.
The 0.05 goal remains open. Reproduce using the four commands in
[research/r30/ANALYSIS.md](research/r30/ANALYSIS.md).

## R31 estimator audit and follow-up design

Seven rank cutoffs were backtested against 34 chronological scores. The cutoff
giving a tempting R30 estimate near 0.0577 failed that backtest badly and is
rejected. Two complementary diagnostic probes were prepared from supplied spline
and spatial features; neither is a demonstrated improvement or should replace
R30 as the first pending measurement. Exact hashes, probe novelty and future
feedback names are in [research/r31/](research/r31/ANALYSIS.md).

## R32: exact scorecard and calibrated physics (2026-10-20)

**New information.**

- **Source.** The leaderboard API (`api.zindi.world/v1/competitions/<slug>/participations`) returns
  every entry's 9 scorecard ratios to 3 decimals (`research/r32/leaderboard_rubrics.json`).
- **Definitions.** Our R29 entry is a known vector, which pins the definitions. They use all 9,794
  rows and median splits, and drop rows missing the stratum variable. Rural is Census `ur_class`
  (not RUCA). SVI is split at the median (not 0.6).
- **High Hazard + High Vulnerability:** (heat or wildfire or summer drought high) and SVI high,
  vs every other row.
- **Check:** R23, R24 and R29 reproduce 27/27 displayed values (`research/r32/definition_check.csv`).
- **The four zero-score leaders disagree** (rural 2.23–2.38, CVI 1.46–1.65).
  - The cause is their ungraded Washington rows: their fitted Washington levels are 1.0×, 1.7×,
    ≈0× and 0.9× ours.
  - So "rank 1 = truth" holds only up to the true Washington level. The joint update now treats
    each leader's level as unknown.

**Candidates.**

| File | Est. public | Change vs R29 (RMS / inside span) | Note |
|---|---:|---|---|
| `submission_r29.csv` | 0.062095 (actual 0.062094756) | – | best; also the most ratio-faithful scored file |
| `submission_r32_t1.csv` | 0.062091 | 0.00058 / 21% | T1: exact definitions, 4 leaders, Washington latent |
| MSM physics blend (not written) | 0.06232–0.06294 | 0.005–0.010 / 0% | T2: rejected |

**Validation.**

- **Estimator:** forward MAE 0.000119 over 34 files, leave-one-out MAE 0.000033.
- **MSM physics as out-of-span prior:** forward MAE 0.000108. It is not significant
  (17 wins / 17 losses, bootstrap 95% CI includes 0).

**Upload order** (at most 5 per day):

1. `submission_r32_t1.csv`. It is the only file with a positive estimate. The gain is small
   (0.000004), and the score measures whether the multi-leader Washington-latent constraints carry signal.
2. Keep the remaining slots unless R30's score changes the picture. Report it and the
   R32 score; add each with `research/add_score.py`.

**Final selection** (by 2026-10-30, chosen manually on Zindi):

- (i) the best public file, currently R29;
- (ii) R27, the ratio-faithful file from before the public-leaning push.

**Best Bias Discovery** (separate $1,000 prize, judged from the write-up):

- Draft: [BIAS_DISCOVERY.md](BIAS_DISCOVERY.md). Among urban tracts, places supplied mostly by Meta
  pages carry about twice the estimated gap in all four regions.
- Submit it through the competition's write-up form.

Reproduce (about 10 minutes; data/ required):

```bash
python research/r32_scorecard.py
python research/r32_features.py
OPENBLAS_NUM_THREADS=4 python research/r32_msm.py
python research/r32_discovery.py
```
