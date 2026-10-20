# R31 audit and experiment design — 2026-10-19

The previous goal turn made concrete progress: it trained and verified R30 and
created two valid submission alternatives. This continuation audited the score
estimator and prepared complementary measurement files. No new leaderboard score
has been supplied; the confirmed best remains R29 at **0.062094756**, and the
0.05 goal is not achieved.

## Rank sensitivity

The score estimator projects a candidate into the historical prediction span.
Near-dependent directions can amplify estimated public-weight and rounding errors.
This audit backtested seven SVD cutoffs on the same 34 chronological score holdouts,
always excluding the held-out score. Results are in `rank_cv.csv` and
`rank_validation_summary.csv`. Candidate estimates are in `rank_candidate_estimates.csv`.

The cutoff with smallest retrospective MAE was 1e-05, MAE
0.000119126; it gives R30 estimate
0.061732458. Across all tested cutoffs, R30 estimates
range from 0.057712131 to 0.062238493.
These are conditional model calculations, not observed performance. The most
optimistic cutoff must not be chosen merely because it moves toward 0.05.
The retrospective cutoff choice is itself exploratory and needs new feedback.

In particular, the 1e-7 cutoff produces the tempting 0.057712131 estimate, but
its chronological MAE is **0.017022004**, versus about 0.000119 for stable
cutoffs. It is rejected as an estimator. Among cutoffs excluding that failure,
R30 estimates still range from about 0.06170 to 0.06224. This is a sensitivity
range, not a confidence interval. The 0.06183 estimate in the R30 report remains
one regularized estimate; its precision must not be read as certainty.

## Complementary probes

`probe_r31_splines.csv` and `probe_r31_spatial.csv` are **diagnostic experiments**,
not validated replacements for R29 or R30. They query the strongest remaining
feature-covariance directions after excluding historical predictions, the eight
scorecard indicators and R30's pending direction. The second also excludes the
first query's planned direction. This is a prior-based information design;
large design variance does not establish a real target effect.

This is a simple spectral query heuristic, not a claim of Bayes-optimal query
selection. [Alexanderian's D-optimality note](https://arxiv.org/abs/2212.11466)
describes the linear-Gaussian information criterion; [Go and Isaac (2022)](https://proceedings.mlr.press/v180/go22a.html)
show why experiment rankings can depend on the prior. Those limitations motivate
keeping distinct spline and spatial hypotheses. The code does not implement
their robust expected-information-gain ambiguity set.

Both use only already supplied competition covariates and are perturbations of
confirmed R29. Each has 9,794 unique GEOIDs, preserved leading zeros, finite bounded
predictions, all five published anchors and unchanged ungraded WA rows. Clipping
can introduce known-span components; the metadata reports the actual remaining
novelty and correlation with R30 after clipping and rounding.

The planned perturbation RMS is 0.006. The approximate public-sampling uncertainty
in a normalized direction's residual coefficient is about 0.001. This is an
assumption-based population/public comparison, not uncertainty in an already
returned leaderboard score. The goal needs roughly 35% lower MSE; these queries
are designed to acquire information, not to imply that two small perturbations
will reach that goal.

## Decision

**R30 remains the first file to score.** Keep its bytes unchanged so the returned
measurement can be matched to the existing record. Do not spend additional slots
on the probes until R30's result shows whether the new direction is useful.
No external upload was performed. The next necessary external evidence is R30's
actual score. More fitting of these same aggregate measurements cannot verify
a public score of 0.05.

`pending_measurements.json` records exact file hashes and internal score-history
names for future feedback. It records that uploads have not been authorized;
neither these files nor the continuation instruction authorize an external
submission. Add feedback only when a real score and exact file are identified.

Reproduce: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python research/r31_audit.py`.
This uses the same requirements and source data as R30. Probe seed: 3110.
