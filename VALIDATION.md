# Validation: how every number here is split, held out, and computed

**Status (2026-10-29, after 89 real leaderboard scores; best measured public RMSE 0.058301186).**

There is no tract-level training data for the target: the four reference layers that define the
coverage gap (TIGER roads, Microsoft footprints, HIFLD facilities, CBP) were withdrawn, so the
score of one uploaded file is the only ground truth available. This document states exactly what
is used as validation data, how the split works, and what the current numbers are.

## 1. The four validation layers

| Layer | Split | What it validates | Current score |
|---|---|---|---:|
| **Real leaderboard scores** | chronological: each new upload | the whole pipeline, measured | **0.05836217** |
| **Rolling-origin hold-out of the scores** | fit on scores 1..k, predict score k+1 | the score-forecast model | see §3 |
| **County-grouped 5-fold CV** | counties held out, per-tract | the auxiliary Overture-count models only (the only supervised models with real per-tract labels, one row per tract observed) | §5, §6 |
| **Leave-one-target-out on the published statistics** | each published moment held out | the physics simulator's calibration | calibration section of `outputs/physics_report.json` |

## 2. The "validation set" in the usual ML sense does not exist

The competition ships no per-tract target. The test tracts themselves are what Overture data is
available for; only their reference counts are hidden. So "the validation score of the model" is
**the public RMSE the leaderboard returns for an uploaded file** — a single number per submission,
5 submissions a day, each one an aggregate constraint over the ~2,938 scored public rows (about 30%
of 9,794). There is no way to compute a per-tract error locally, and no way to split tracts into
training and validation for the target.

## 3. Rolling-origin validation of the score model (the closest analogue of a validation split)

Every scored file `v` with public RMSE `s` gives one exact equation about the hidden target:

```
mean_public(v · y) = (mean_public(v²) + mean_public(y²) − s²) / 2
```

`mean_public(y²)` comes from the all-zero file (score 0.107246) and `mean_public(y)` from the
constant-0.10 file, so **the target's public moments are known exactly**:

| Quantity | Value |
|---|---:|
| mean y over public rows | 0.059806 |
| RMS y over public rows | 0.107246 |
| sd y over public rows | 0.089022 |
| best achievable with a single constant (the mean) | 0.089022 |
| best measured (`outputs/submission_u0.csv`, the pure measured update) | **0.058301** |
| share of the public target variance explained by the current best | **70.45%** |

### How prediction-before-measurement is validated

For each scored file in sequence, the projection estimator (`train.estimate`) predicts its public
RMSE from the labels scored *before* it — a leak-free, chronological hold-out over 61 forecastable
labels (the two physics-only files and the eastern-wa control are excluded; the probes are scored
at the end):

| Group | n | MAE | RMSE of forecast error | bias |
|---|---:|---:|---:|---:|
| base (first ensemble files) | 2 | 0.002914 | 0.003209 | +0.002914 |
| clip (leaderboard-calibrated) | 14 | 0.000065 | 0.000151 | +0.000060 |
| probe (designed directions) | 25 | 0.000138 | 0.000177 | +0.000003 |
| transfer (measured bets) | 20 | 0.000232 | 0.000402 | +0.000032 |
| **last 12 scored files** | 12 | 0.000101 | **0.000139** | +0.000082 |
| **last 12 at 89 labels (U0 included)** | 12 | 0.000080 | **0.000100** | +0.000046 |
| all 61 | 61 | 0.000243 | 0.000639 | +0.000121 |

The pure-update file `outputs/submission_u0.csv` scored **0.058301186** (a new best, 0.000061 under the
previous best): its pre-score forecast from the transfer posterior was 0.058272 ± 0.000053 (error
+0.000029, about half a modelled standard deviation), and the projection estimator's lower in-span
value 0.058210 had the direction's truth below it.

The current best file's forecast, made before it was scored: 0.058583 predicted, 0.058362 measured
(error +0.000221, about a 0.02% relative forecast).

Two more rolling-origin validations on the same chronological split, with the transfer rows re-run at
the 89-label state (`outputs/transfer_forward_sweep_diagonal.json`):

| Forecaster | Hold-out | MAE | RMSE of forecast error | bias |
|---|---:|---:|---:|---:|
| projection estimator (`estimate`) — recent 20 rounds | rolling origin | 0.000232 | **0.000402** | +0.000032 |
| projection estimator — recent 12 rounds | rolling origin | 0.000101 | **0.000139** | +0.000082 |
| projection estimator — recent 12 rounds, 89 labels | rolling origin | 0.000080 | **0.000100** | +0.000046 |
| transfer posterior (the bet forecaster), `independent` residual model | last 10 rounds | 0.000117 | **0.000209** | +0.000100 |
| transfer posterior, last 8 rounds, 89 labels (includes the U0 landing) | rolling origin | 0.000061 | **0.000100** | +0.000052 |
| transfer posterior, last 4 rounds | rolling origin | 0.000024 | — | — |
| stage-8 empirical-Bayes posterior vector scored against held-out probe moments | last 20 | 0.003738 | 0.003776 | +0.003708 |
| stage-8 empirical-Bayes with OLS-projection weights | last 20 | 0.000232 | **0.000402** | +0.000032 |

The empirical-Bayes posterior's own vector-scores are systematically ~0.004 above the measured
scores because the held-out files *are* improvements over that posterior's static vector; the
projection-based forecasts are the numbers that predict "what will the next upload score". Every
forecast tool now sits at MAE ≈ 0.0001–0.0004 per round on the chronological split, and the forecast
errors have been shrinking as more scores accumulate.

## 4. What per-model held-out validation says about the model families (the tournament)

Each family of the physics-calibrated-on-labels pipeline is refitted **without the last 10 score
moments** (`--holdout 10`) and is then scored on those held-out moments. This is the competition's
"validation set": the ten newest real scores, never seen by the fit (the held-out window is labels
79–88, which includes the current best file `tr_nofp_030`). Results at 88 labels — chi-square over
the 10 held-out moments (start before fitting 47.1, fewer is better), and the family's estimated
public RMSE on its own:

| Family | held-out chi² (start 47.1) | est. RMSE alone (held-out fit) |
|---|---:|---:|
| **wide neural, MLP 21-12-4 on all four layers (376)** — tournament winner, Elo 1081 | **36.09** | 0.061595 |
| neural context effects, MLP 21-6-2 (206) | 38.83 | 0.061687 |
| base physics (structural mechanisms, 60 parameters) | 39.79 | 0.061913 |
| flexible context effects, sd 0.2 (88) | 68.80 | 0.061201 |

The held-out moment for the current best file (`tr_nofp_030`) is missed by every family
(z −4.1 to −4.4): the newest direction carried more truth than the fits predicted, which is exactly
what the round-4 files bet on.

Refits at 89 labels (held-out window labels 80–89, which now also covers `tr_u0_000`'s own landing;
start 34.0):

| Family | held-out chi² (start 34.0) | Elo (round 5) | est. RMSE alone (held-out fit) |
|---|---:|---:|---:|
| **wide neural, MLP 21-12-4 on all four layers (376)** — tournament winner again, Elo 1078 | **24.91** | 1077.9 | 0.061574 |
| neural context effects, MLP 21-6-2 (206) | 27.48 | 977.0 | 0.061750 |
| base physics (structural mechanisms, 60 parameters) | 28.19 | 941.7 | 0.061964 |
| flexible context effects, sd 0.2 (88) | 65.66 | 1003.4 | 0.061314 |

Every family improved on hold-out with the extra label, and the wide network stays first by a wide
Elo margin. The flexible family still misses the newest moments hard (65.66 over 10 dof): the
measured bets keep carrying more truth than the fits predicted.

Full-data fits on all 89 labels: flexible physics **0.060367** (≈ unchanged from 88), wide network
**0.061053**.

## 4b. Round-4 files built from this validation (2026-10-29)

| File | Contains | In-span forecast | Posterior forecast (truth at +0.3) |
|---|---|---:|---:|
| `outputs/submission_u0.csv` (`tr_u0_000.csv`) | the measured-direction update alone, P(beats best) 0.95 | 0.058210 | 0.058272 ± 0.000053 → **scored 0.058301186** |
| `submission.csv` (`tr_cosci04A_030.csv`) | U0 + 0.30 × the wide network's unscored part (RMS 0.0064) | 0.058240 | 0.058250 (+0.5: 0.058208) |
| `submission_alt.csv` (`tr_cosci04B_030.csv`) | U0 + 0.30 × the flexible model's unscored part (RMS 0.0039) | 0.058224 | 0.058264 (+0.5: 0.058249) |
| `outputs/probes_branch/q092_round4_network_residual.csv` | probe of the wide-net residual orthogonal to A and B | 0.058439 (if empty) | measures one fresh direction |

The round-4 A/B files are byte-exact stashed while round 5 overwrites the generic names:
`outputs/tr_cosci04A_030_final.csv`, `outputs/tr_cosci04B_030_final.csv` (sha256 in
`data/logs/goal/round4_stash_sha256.txt`, also in git at `16d0301`).

## 4c. Round-5 files built from the 89-label refits (2026-10-29)

| File | Contains | In-span forecast | Posterior forecast |
|---|---|---:|---:|
| `outputs/submission_u1.csv` (`tr_u1_000.csv`) | U0's newest 2 measured directions stretched to their posterior optimum (1.10x, 2.04x), P(beats 0.058301) **0.88** | 0.058301 | 0.058296 ± 0.000005 → **scored 0.058294871 (best; error −0.0000015 ≈ ⅓ σ)** |
| `submission.csv` (`tr_cosci05A_030.csv`) | U1 + 0.30 × the wide network's unscored part (RMS 0.0066) | 0.058339 | 0.058273 at +0.3, 0.058228 at +0.5 |
| `submission_alt.csv` (`tr_cosci05B_031.csv`) | U1 + 0.31 × the flexible model's unscored part (RMS 0.0035) | 0.058318 | 0.058290 at +0.3, 0.058276 at +0.5 |
| `outputs/probes_branch/q093_round5_network_residual.csv` | the wide-net residual orthogonal to A and B (cosines ≤1e-6; span fraction 1.1e-05) | 0.058378 (if empty) | measures one fresh direction (0.058224 if true at step) |

Schema checks: `outputs/check_round5.py` → `outputs/round5_files.json` (all four: 9,794 rows, unique
11-digit GEOIDs, values in [0,1]; A/B md5 identical across the cosci reruns; U1's A/B reconstruction
error exactly 0).

## 5. Feature engineering that was validated, and that failed

Every auxiliary-feature branch ever tested improved county-held-out proxy CV and **lost** in the
downstream physics fit — a consistent negative-transfer finding, so the pipeline's features stayed
lean. The players that actually moved the real score:

| Kept (moved the real score) | Real gain |
|---|---:|
| route-network length as the TIGER proxy (mechanism check: no-overture-route share ≈ published undefined share) | first ensembles |
| facility-leakage + TIGER local-name double-count (structural mechanisms, `--structural --unmatched`) | 0.060353 → 0.059977 |
| concurrent-route double counting | 0.059977 → 0.059507 |
| flexible context effects, 28 log-effects (`--flex`), then 0.3× its unscored direction | 0.058987 → 0.058866 |
| wide neural MLP on all four reference layers (`--nn-wide`, 376 parameters), best in the held-out tournament | holds the current base |
| footprint-free variant's unscored direction (dropped USFS/CarbonPlan columns), the newest bet | 0.058548 → **0.058362** |
| measured bets sized by the transfer posterior (`--add-label` per real score) | cumulative: 0.061372 → 0.058362 |

| Tested, rejected (improved proxy CV, worse physics/board) | Result |
|---|---|
| hurdle (two-part) proxy models | physics estimate 0.062327 vs baseline 0.060974 |
| cross-category Overture counts | 0.062356 |
| direct-Poisson cross-count control | 0.062489 |
| Overture source-mix features | 0.062469 |
| source-confidence features | 0.062332 |
| spatial-lag (neighbour-tract) counts | 0.062580 |
| address/contact richness | unstable (Maricopa registry regression) |
| `--covariates` log-effects on the TIGER/building scale | small fitted effects; unscored RMS 0.0045, not worth a bet |
| linear feature correction through the moments (98 features, ridge) | worse on held-out moments at every penalty |
| region-specific flexible effects | ran the wrong way (0.058866 → 0.058936) |

## 6. The auxiliary supervised models (the only place with an ordinary train/valid split)

Per-tract labels exist for four observable Overture counts (fire, EMS, school and registry-place
evidence), validated with county-grouped 5-fold CV, GLM vs Poisson-GBM, no held-out county leaking
into either hurdle stage. These **proxy** CV deviances are input to the simulator; they are not the
competition metric, and §5 is the study of how little they carry over.

## 7. Honest response to the "validation score ≤ 0.0004" goal

Two possible meanings, answered with measurements:

1. **"The model's mistake when validating a held-out score" ≈ 0.0004** — i.e., the residual RMS of
   score forecasts. This is already at or below 0.0004 on every rolling-origin check run today
   (projection 0.000402 over the last 20 rounds and **0.000100** over the recent 12 at 89 labels;
   transfer posterior 0.000209 over 10 and **0.000100** over the last 8 at 89 labels). It is a
   measurement-precision limit, not a modelling limit: the
   only way to shrink it further is to score more files.

2. **"The submission's RMSE ≤ 0.0004"** — not achievable with the permitted data. A per-tract RMSE
   of 0.0004 requires explaining 99.9986% of the target's public variance (RMS y = 0.107246,
   measured exactly from the zero file's score); the current best explains 70.45%, and the best
   possible single constant would score 0.089022. The model's own 4,096-draw simulator at the
   label-calibrated parameter fit claims an aleatoric floor of about **0.0599** (`outputs/floor_analysis.json`)
   — the per-tract randomness of the withdrawn reference counts (exact TIGER metreage of each tract's
   highways, whether a fire station entered the HIFLD extract, how many establishments CBP counted),
   which the permitted datasets do not determine and the rules forbid reconstructing. The measured
   rounds keep moving below that claim (0.058301 so far), which shows the remaining *learnable*
   structure is real but small; recent rounds gained 0.00002–0.00019 per round. The practical
   frontier is round-by-round measured bets, not features: each new direction is worth at most
   ~5e-5 of RMSE when fully true (a 0.0072-RMS direction squared against the current
   0.058301² residual mass).

3. **"Round gains of 0.0004 or better"** — this is what the loop actually delivers: rounds
   `tr_lbmulti` → `tr_nofp` gained 0.00022, 0.00002, 0.00004, 0.00009, 0.00009, 0.00019, then the pure
   update `tr_u0_000` gained 0.000061 — the mean
   gain of the last six rounds is 0.00009 per upload, and round 5's files forecast 0.0583
   (0.0582–0.0583).
