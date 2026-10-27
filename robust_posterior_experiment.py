"""Choose transfer updates that survive competing residual and prior assumptions."""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import train
from posterior_experiment import TransferPosterior, forward_validation


SCENARIOS = (
    ("independent", .35, 3.), ("shared", .35, 3.),
    ("independent", .2, 3.), ("independent", .5, 3.),
    ("independent", .35, 1.5), ("independent", .35, 6.),
    ("shared", .2, 3.), ("shared", .5, 3.),
    ("shared", .35, 1.5), ("shared", .35, 6.),
)


def posterior_moments(samples):
    w = samples["weights"]
    z = np.column_stack([samples["curvature"], samples["means"]])
    mean = w @ z
    centered = z - mean
    covariance = (centered * w[:, None]).T @ centered
    k = samples["means"].shape[1]
    covariance[k:, k:] += np.einsum("d,dij->ij", w, samples["covariance"])
    return mean, covariance


def risk_delta(x, baseline, moments, uncertainty_weight=.5):
    coefficients = np.r_[x ** 2 - baseline ** 2, -2 * (x - baseline)]
    mean, covariance = moments
    return float(coefficients @ mean + uncertainty_weight * np.sqrt(max(coefficients @ covariance @ coefficients, 0)))


def optimized_candidates(model, moments):
    baseline = model.best_x
    candidates, diagnostics = {}, []
    for tail in sorted(set((1, min(2, model.k), model.k))):
        fixed = np.arange(model.k - tail, model.k)
        for radius in (.0015, .003):
            for uncertainty_weight in (.5, 1.):
                def unpack(free):
                    x = baseline.copy()
                    x[fixed] = free
                    return x

                def objective(free):
                    x = unpack(free)
                    return max(risk_delta(x, baseline, m, uncertainty_weight) for m in moments) * 1e6

                constraints = [dict(type="ineq", fun=lambda free: 1 - np.sum(model.c * (unpack(free) - baseline) ** 2) / radius ** 2)]
                fits = []
                for start in (baseline[fixed], (moments[0][0][model.k:] / moments[0][0][:model.k])[fixed]):
                    fits.append(minimize(objective, start, method="SLSQP", constraints=constraints,
                                         bounds=[(baseline[j] - radius / np.sqrt(model.c[j]),
                                                  baseline[j] + radius / np.sqrt(model.c[j])) for j in fixed],
                                         options=dict(maxiter=250, ftol=1e-9)))
                feasible = [f for f in fits if f.success and constraints[0]["fun"](f.x) >= -1e-6]
                if not feasible:
                    diagnostics.append(dict(tail=tail, radius=radius, uncertainty_weight=uncertainty_weight,
                                            accepted=False, messages=[str(f.message) for f in fits]))
                    continue
                fit = min(feasible, key=lambda f: f.fun)
                x = unpack(fit.x)
                vector = model.best.copy()
                vector[model.labels.graded] = np.clip(model.best[model.labels.graded] + model.e @ (x - baseline), 0, 1)
                name = f"robust_last{tail}_rms{radius}_risk{uncertainty_weight}"
                candidates[name] = np.round(vector, 6)
                diagnostics.append(dict(name=name, accepted=True, objective_mse=float(fit.fun / 1e6),
                                        coordinates=x.tolist(), optimizer_success=bool(fit.success)))
    return candidates, diagnostics


def run(draws=40000, forward_points=10):
    labels = train.AggregateLabels()
    models, summaries, moments, proposals = [], [], [], []
    candidates = {}
    for scenario in SCENARIOS:
        mode, k_sd, coord_sd = scenario
        model = TransferPosterior(labels, residual_model=mode, k_sd=k_sd, coord_sd=coord_sd)
        proposal = model.proposal()
        sample = model.sample(draws=draws, seed=101, proposal=proposal)
        models.append(model)
        moments.append(posterior_moments(sample))
        proposals.append(proposal)
        summaries.append(dict(residual_model=mode, k_sd=k_sd, coord_sd=coord_sd, **sample["diagnostics"]))
        if len(models) <= 2:
            candidates.update({f"{mode}__{name}": v for name, v in model.candidates(sample).items() if name != "measured_best"})
        train.log(f"Robust scenario {scenario}: ESS {sample['diagnostics']['effective_draws']:.0f}/{draws}")
    model = models[0]
    optimized, optimization = optimized_candidates(model, moments)
    candidates.update(optimized)
    candidates["measured_best"] = model.best.copy()
    unique, seen = {}, set()
    for name, vector in sorted(candidates.items(), key=lambda item: item[0] == "measured_best"):
        digest = hashlib.sha256(vector.tobytes()).hexdigest()
        if digest not in seen or name == "measured_best":
            unique[name] = vector
            seen.add(digest)
    candidates = unique
    predictions = {name: [] for name in candidates}
    for scenario, fitted, proposal in zip(SCENARIOS, models, proposals):
        sample = fitted.sample(draws=draws, seed=101, proposal=proposal)
        for name, vector in candidates.items():
            forecast = fitted.forecast(vector, sample)
            predictions[name].append(dict(scenario=list(scenario), **forecast))
    rows = []
    for name, forecasts in predictions.items():
        reference = forecasts[0]
        worst = max(r["expected_rmse"] + .5 * r["sd_rmse"] for r in forecasts)
        rms = float(np.sqrt(model.w @ (candidates[name][labels.graded] - model.best[labels.graded]) ** 2))
        rows.append(dict(name=name, expected_rmse=reference["expected_rmse"], sd_rmse=reference["sd_rmse"],
                         worst_risk_adjusted_rmse=worst, min_p_beats_best=min(r["p_beats_best"] for r in forecasts),
                         scenario_rmse_range=[min(r["expected_rmse"] for r in forecasts), max(r["expected_rmse"] for r in forecasts)],
                         step_rms=rms, scenarios=forecasts))
    eligible = [r for r in rows if r["min_p_beats_best"] >= .65 and r["worst_risk_adjusted_rmse"] < model.best_score
                and 1e-6 < r["step_rms"] <= .005]
    selected = min(eligible, key=lambda r: r["worst_risk_adjusted_rmse"]) if eligible else next(r for r in rows if r["name"] == "measured_best")
    vector = candidates[selected["name"]]
    seed_checks = []
    for fitted, proposal in zip(models[:2], proposals[:2]):
        forecasts = []
        for seed in (202, 303):
            sample = fitted.sample(draws=draws, seed=seed, proposal=proposal)
            forecasts.append(dict(seed=seed, effective_draws=sample["diagnostics"]["effective_draws"],
                                  **fitted.forecast(vector, sample)))
        seed_checks.append(dict(residual_model=fitted.residual_model, forecasts=forecasts))
        reference = selected["scenarios"][0 if fitted.residual_model == "independent" else 1]["expected_rmse"]
        if max(abs(f["expected_rmse"] - reference) for f in forecasts) >= 1e-5:
            raise ValueError("candidate forecast does not reproduce across independent seeds")
    validation = {mode: forward_validation(labels, forward_points, residual_model=mode) for mode in ("independent", "shared")}
    for mode, summary in validation.items():
        train.log(f"Validation {mode}: MAE {summary['mae']:.8f}")
    output = "outputs/submission_robust_refined.csv"
    reference_seeds = [selected["expected_rmse"], *[f["expected_rmse"] for f in seed_checks[0]["forecasts"]]]
    frame = pd.DataFrame({"GEOID": labels.geoid, "coverage_gap_score": vector})
    if not np.isfinite(vector).all() or not frame.coverage_gap_score.between(0, 1).all():
        raise ValueError("invalid candidate scores")
    frame.to_csv(output, index=False)
    report = dict(experiment="continuous transfer search robust to shared residuals and prior settings",
                  labels_sha256=hashlib.sha256(Path(train.LABELS).read_bytes()).hexdigest(),
                  measured_best=dict(file=model.best_name, score=model.best_score),
                  selected=dict(selected, mean_over_seeds=float(np.mean(reference_seeds)),
                                seed_rmse_range=float(np.ptp(reference_seeds)),
                                p_beats_best=selected["scenarios"][0]["p_beats_best"]),
                  posterior_diagnostics=summaries, optimization=optimization, candidates=rows,
                  independent_seed_checks=seed_checks, validation=validation,
                  selection_rule="minimum worst(mean + 0.5 SD) across ten scenarios; all P(improve)>=0.65; step RMS<=0.005",
                  output=output, output_sha256=hashlib.sha256(Path(output).read_bytes()).hexdigest(),
                  target_004_supported=bool(selected["worst_risk_adjusted_rmse"] <= .04),
                  caveat="Forecasts are conditional on aggregate-score models and are not independently calibrated probabilities. "
                         "Competing residual models have similar historical error; both must support the chosen update. "
                         "No private leaderboard validation is available.")
    train.atomic_json("outputs/robust_posterior_experiment.json", report)
    train.log(f"Selected {selected['name']}: {selected['expected_rmse']:.8f}; scenario range {selected['scenario_rmse_range']}")
    return report


if __name__ == "__main__":
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):
        run()
