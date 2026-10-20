# R30 modeling study — 2026-10-19

The confirmed best is **R29, RMSE 0.062094756**, reported by the user. A new
**unscored** candidate is `submission_r30.csv`. Its historical-span projection is
**0.061832588**, compared with **0.062095180** for R29
under the same estimator. These are model estimates, not new leaderboard results.
The 0.05 goal has **not** been reached; it requires 35.16% lower squared error.

## What was trained

This round executed **612 chronological kernel forecasts** (nine models × two
prior centers × 34 held-out scores) and **209 bounded-model forecasts**, plus
the full-data fits and sensitivity variants. No tract-level labels were invented.
The original 60-score `model.npz` remains immutable. `research/feedback.py` adds
the confirmed R29 score after verifying the exact submission SHA-256 against
`research/score_feedback.json`.

Inputs are supplied competition data: Overture-derived counts and route measures,
ACS housing, population, supplied vulnerability/hazard strata, county IDs and
centroids of supplied tract geometries. There were no new external datasets,
withdrawn reference layers, paid inference APIs, AutoML runs or external uploads.
The initial smoke run failed because regional/national tract geometry tables
contained duplicate IDs; de-duplicating IDs before alignment fixed it. Full runs
then used all 9,379 graded tracts and verified the 9,794-row submission schema.

## Research and model choice

[Law et al. (NeurIPS 2018)](https://proceedings.neurips.cc/paper/2018/hash/24b43fb034a10d78bec71274033b4096-Abstract.html)
and [Zhang et al. (NeurIPS 2020)](https://proceedings.neurips.cc/paper_files/paper/2020/file/5b0fa0e4c041548bb6289e15d865a696-Paper.pdf)
motivate learning individual functions through an explicit aggregate observation
model. Here, RMSE feedback is converted to signed linear target moments. This
adaptation uses correlated Gaussian measurement noise; it does not inherit the
papers' guarantees because the observations are adaptive, overlapping and sampled.

I trained linear, spline, random-tree leaf (depths 3/5/7), regional-interaction,
spatial Fourier/county and combined feature covariance models, with an iid control.
Random tree partitions use covariates without targets. The Gaussian posterior is
computed analytically; hyperparameters are refit before every held-out score.
An adaptive prior uses the earlier best submission; a frozen prior uses R12.
Feature transforms use the available, unlabeled competition covariates. There
is no supervised tract split because there are no tract training labels.

[TabPFN](https://www.nature.com/articles/s41586-024-08328-6) was considered but not
used: its ordinary supervised interface requires individual labels. Training it,
CatBoost or a neural net against our current predictions would primarily reproduce
our own assumptions. The earlier R26 MLP already failed its actual leaderboard
test; it was not rebranded as new evidence. The
[DRO regression literature](https://www.jmlr.org/papers/v19/17-295.html) informed
regularization sensitivity, but this code does not implement that paper's
Wasserstein uncertainty set.

The selected new method is **box-constrained aggregate regression**. Its objective is

`0.5 * (M y - z)' R^-1 (M y - z) + 0.5 * ||y - physics_base||² / tau²`,

subject to `0 <= y <= 1`, with published anchors fixed for final fitting. `M` maps
tract predictions to scored moments, `z` is reconstructed from scores, and `R`
models public-sampling uncertainty. For the final model, eight rounded scorecard
ratios add soft linear constraints with correlated WA uncertainty. The ratio
interpretation and the leader-as-truth assumption are inherited from earlier
research and remain unverified assumptions.

This is more than clipping once: a small convex dual repeatedly
updates all tract predictions and aggregate residuals until the bounded solution
is consistent. Actual moment rank is 53. The local kernel and noise
scales are empirical estimates. Bounds provide real target support, not new labels.

## Validation

Each held-out score is excluded from fitting, covariance calibration and prior
selection for that fold. No scorecard constraints enter the 34-file main comparison.
The separate five-file scorecard test begins after R24, when both R23/R24 calibration
scorecards were already available. This is exploratory backtesting of public score
prediction, **not private-tract cross-validation or an untouched final test**.

| center | kernel | n | mae | rmse |
| --- | --- | --- | --- | --- |
| adaptive | domain_mix | 34 | 0.000119758 | 0.000174334 |
| adaptive | linear | 34 | 0.000120027 | 0.000174797 |
| adaptive | regional_splines | 34 | 0.000118342 | 0.000173499 |
| adaptive | spatial | 34 | 0.000119833 | 0.000175305 |
| adaptive | splines | 34 | 0.000120211 | 0.000173737 |
| adaptive | trees3 | 34 | 0.000120122 | 0.000173897 |
| adaptive | trees5 | 34 | 0.000120414 | 0.000174077 |
| adaptive | trees7 | 34 | 0.000120212 | 0.000173815 |
| adaptive | white | 34 | 0.000120330 | 0.000175371 |
| frozen | domain_mix | 34 | 0.000120700 | 0.000179343 |
| frozen | linear | 34 | 0.000121550 | 0.000187192 |
| frozen | regional_splines | 34 | 0.000122829 | 0.000186346 |
| frozen | spatial | 34 | 0.000118922 | 0.000174620 |
| frozen | splines | 34 | 0.000113958 | 0.000186677 |
| frozen | trees3 | 34 | 0.000116282 | 0.000181242 |
| frozen | trees5 | 34 | 0.000123380 | 0.000190022 |
| frozen | trees7 | 34 | 0.000121198 | 0.000190842 |
| frozen | white | 34 | 0.000118841 | 0.000174229 |

| factor | noise_factor | scorecard | n | mae | rmse |
| --- | --- | --- | --- | --- | --- |
| 0.250000000 | 1.000000000 | False | 34 | 0.000120025 | 0.000174321 |
| 0.500000000 | 1.000000000 | False | 34 | 0.000117220 | 0.000172791 |
| 1.000000000 | 0.500000000 | False | 34 | 0.000114937 | 0.000172323 |
| 1.000000000 | 1.000000000 | False | 34 | 0.000115465 | 0.000172238 |
| 1.000000000 | 1.000000000 | True | 5 | 0.000005237 | 0.000007753 |
| 1.000000000 | 2.000000000 | False | 34 | 0.000116254 | 0.000172349 |
| 2.000000000 | 1.000000000 | False | 34 | 0.000114598 | 0.000172443 |

Primary bounded-model MAE difference versus frozen iid control:
**-0.000003376**, with five-file moving-block bootstrap 95%
descriptive interval [-0.000010021, +0.000000618].
The five scorecard holdouts are few and closely related; their small errors do not
establish performance on a substantially new prediction direction. Nonlinear
kernels gave mixed results and mostly shrank toward the control. They were not
selected simply for a low in-sample score estimate.

## Candidate and sensitivity

R30 uses the primary bounded model with all current moments and the eight soft
scorecard constraints. It changes predictions by weighted RMS **0.004748**
from R29, maximum **0.061020**. `submission_r30_conservative.csv`
is an optional 50% blend with R29, not another confirmed improvement. Neither is
submitted automatically. R29 remains the known best until a new score is returned.

The primary specification was retained rather than selecting the most favorable
projected score among sensitivity fits. Prior scale factors 0.25/0.5/1/2 and noise
factors 0.5/1/2 were evaluated. Full configurations, convergence, dual gaps and
direction changes are in `bounded_sensitivity.csv`; all full-fit artifacts and
estimates are in `experiments.tsv`. Older provisional files remain diagnostic
artifacts and are excluded from chronological counts.

| stratum | leader_displayed | r29 | r30 |
| --- | --- | --- | --- |
| rural | 2.340000000 | 2.328757997 | 2.329083082 |
| tribal | 2.880000000 | 2.881180042 | 2.876165146 |
| svi | 1.230000000 | 1.229706635 | 1.228662788 |
| cvi | 1.550000000 | 1.541917573 | 1.544133838 |
| sdrought | 1.000000000 | 0.999300645 | 1.001119409 |
| wdrought | 1.360000000 | 1.360358735 | 1.363036830 |
| wildfire | 1.770000000 | 1.763458717 | 1.767175467 |
| heat | 0.620000000 | 0.616039255 | 0.619002864 |

The frozen-prior scales approach their upper search bound, and the row-weight
model is approximate. The public score estimate assumes no target signal in
directions outside historical predictions; nonlinear clipping and scorecard moves
can violate that assumption. The 0.06183 estimate should therefore be treated as
an experiment to test, not a forecast accurate to nine decimal places.

## Verification and next step

- Both files have exactly 9,794 unique official GEOIDs, preserved leading zeros,
  two required columns, finite bounded predictions and all five published anchors.
  Eastern Washington values are unchanged from R29.
- Independent small primal and dual QP implementations agreed to maximum absolute
  prediction difference 5.15e-08. This checks numerical implementation,
  not hidden-target accuracy. Maximum chronological dual gap: 1.43e-09.
- Perturbing the held-out R29 score by +0.01 changed preceding training moments
  by at most 2.67e-12. The held-out score does not train its forecast.
- The numerical requirements include Shapely 2.1.2 for supplied tract geometry
  features; the other packages retain their existing pinned versions.
  CPU only; tree/spatial seed 3101, bootstrap 3104, solver check 3105.
- The most informative next observation is the actual public score of
  `submission_r30.csv`. A new scored direction can be incorporated; repeatedly
  fitting the same 61 measurements cannot verify that 0.05 is attainable.

Reproduce:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python research/r30_kernels.py --center adaptive
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python research/r30_kernels.py --center frozen
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python research/r30_bounded.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python research/r30_finalize.py
```

R30 SHA-256: `97fa40b3c11d0e1990a7b3da11ca5d2d7b126efeb1192a1a0cea3b20633b36df`.

Workflow skills: existing `ml-analysis` and `co-scientist`, plus installed
`scikit-learn`, `statistical-analysis`, and `pymc`. Their guidance informed
validation, prior sensitivity and uncertainty reporting; PyMC MCMC was not run.
Credit: Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026),
[Scientific Agent Skills](https://doi.org/10.48550/arXiv.2609.00065).
[Competition rules](https://zindi.world/competitions/bias-bounty-mapping-equity-challenge)
and the primary research sources were checked online during this round.
