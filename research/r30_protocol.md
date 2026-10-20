# R30 protocol, 2026-10-19

Objective: reduce public RMSE from confirmed R29 0.062094756 toward 0.05.
The target requires about 35% lower MSE; no local experiment can establish this
without new leaderboard feedback or tract labels.

Research hypothesis: nonlinear feature covariance can predict additional target
directions from signed aggregate score measurements. The prior generic linear/RBF
feature kernels failed; this round tests domain splines, unsupervised tree
partitions, county effects and spatial covariance, centered on the best submission
available before each held-out score. The center is an empirical, adaptive prior.

Primary validation: chronological prediction of public scores from p12 onward.
Every model refits hyperparameters to earlier scores. At least the last eight
historical scores form a separate reporting slice; this is an exploratory
comparison, since earlier research already inspected those outcomes. Do not
describe it as an untouched test set. Do not fit artificial tract labels.

Controls: no-feature local white prior; current original empirical-Bayes score
forecasts; incumbent predictions. Compare MAE, predictive log score, large-error
cases and posterior prior-sensitivity. No scorecard ratios in chronological CV;
they can be included in final fitting as a separately identified information
source. Geographic covariates are transductive inputs, not targets.

Execution: existing local open-source Python stack, CPU only; no paid compute,
external uploads, new datasets, or AutoML. First run a small end-to-end test;
extend only successful, numerically stable configurations. Keep every trial.
Use seeds 3101/3102 for tree partitions and spatial features, then a second seed
if a family shows useful held-out signal. Stop scaling a family if it adds no
held-out predictive value. Never claim an estimated score as a measured score.

Sources to assess:
- Law et al., Variational Learning on Aggregate Outputs with Gaussian Processes,
  NeurIPS 2018: https://proceedings.neurips.cc/paper/2018/hash/24b43fb034a10d78bec71274033b4096-Abstract.html
- Chen & Paschalidis, A Robust Learning Approach for Regression Models Based on
  Distributionally Robust Optimization, JMLR 2018:
  https://www.jmlr.org/papers/v19/17-295.html
- Hollmann et al., Accurate predictions on small data with a tabular foundation
  model, Nature 2025: https://www.nature.com/articles/s41586-024-08328-6
  TabPFN is not a direct substitute here because it expects individual labels.
