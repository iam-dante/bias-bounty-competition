"""Chronological experiments with nonlinear priors for aggregate score observations.

Run: python aggregate_kernel_experiment.py
Only scores preceding a held-out submission enter its fit. Features use supplied
covariates, never auxiliary predictions calibrated on the complete score history.
"""
import argparse
import copy
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from sklearn.kernel_approximation import RBFSampler
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import train


def features(df, rich=False):
    x, names = train.auxiliary_feature_matrix(df, "density_strata" if rich else "density")
    cols = [c for c in df if c.startswith(("L_", "n_", "E_", "X_"))]
    raw = np.log1p(df[cols].fillna(0).to_numpy(float).clip(0))
    ratios = np.column_stack([
        np.log((df.n_bld + 1) / (df.U_bld + 1)),
        np.log((df.L_named + 100) / (df.L_route + 100)),
        np.log((df.n_poi + 1) / (df["pop"] + 100)),
        df.n_poi_meta / df.n_poi.clip(lower=1),
        df.n_poi_bq / df.n_poi.clip(lower=1),
    ])
    x = np.column_stack([x, raw, ratios])
    if not np.isfinite(x).all():
        raise ValueError("nonfinite input feature")
    x = np.clip(StandardScaler().fit_transform(x), -6, 6)
    return x[:, x.std(0) > 1e-8]


def feature_bases(df):
    for rich in (False, True):
        x = features(df, rich)
        prefix = "strata" if rich else "physical"
        yield prefix + "_linear", x
        hinges = np.column_stack([x, *[np.maximum(x - knot, 0) for knot in (-1, 0, 1)]])
        yield prefix + "_splines", hinges
        for width in (0.3, 1.0, 3.0):
            random_features = RBFSampler(gamma=width / x.shape[1], n_components=512, random_state=123).fit_transform(x)
            yield f"{prefix}_rbf_{width}", random_features


def prefix_labels(labels, heldout):
    prior = copy.copy(labels)
    names = list(labels.s)[:list(labels.s).index(heldout)]
    prior.s = {n: labels.s[n] for n in names}
    prior.V = {n: labels.V[n] for n in names}
    prior.public_weights()
    return prior


def observation_system(labels):
    g = labels.graded
    w = labels.pi / labels.pi.sum()
    anchor_name = min(labels.s, key=labels.s.get)
    anchor = labels.V[anchor_name][g]
    names = [n for n in labels.s if not n.startswith(("probe_", "e06")) and n != anchor_name]
    d = np.column_stack([np.ones(g.sum()), *[labels.V[n][g] - anchor for n in names]])
    rms = np.sqrt(w @ d ** 2)
    keep = rms > 1e-9
    target = np.r_[labels.ybar - w @ anchor,
                   [(w @ (labels.V[n][g] - anchor) ** 2 + labels.s[anchor_name] ** 2 - labels.s[n] ** 2) / 2
                    for n in names]]
    d, target = d[:, keep] / rms[keep], target[keep] / rms[keep]
    gram = d.T @ (w[:, None] * d)
    span = np.column_stack([np.ones(g.sum()), *[labels.V[n][g] for n in labels.s]])
    u, s, _ = np.linalg.svd(span * np.sqrt(w[:, None]), full_matrices=False)
    q = u[:, s > s[0] * 1e-10]
    return w, d, target, gram, q


def summarize(rows):
    errors = np.array([r["predicted"] - r["actual"] for r in rows])
    return dict(n=len(rows), mae=float(np.mean(abs(errors))), rmse=float(np.sqrt(np.mean(errors ** 2))),
                bias=float(np.mean(errors)))


def run(points=40):
    labels = train.AggregateLabels()
    df, _ = train.load()
    df = df.set_index("GEOID").loc[labels.geoid].reset_index()
    eligible = [n for n in labels.s if not n.startswith(("probe_", "e06"))][-points:]
    split = len(eligible) // 2
    folds, baseline = [], []
    for i, name in enumerate(eligible):
        prior = prefix_labels(labels, name)
        w, d, target, gram, q = observation_system(prior)
        v = labels.V[name][labels.graded]
        residual_w = v * np.sqrt(w)
        residual_w -= q @ (q.T @ residual_w)
        base = train.estimate(prior, labels.V[name])
        folds.append((w, d, target, gram, residual_w * np.sqrt(w), base))
        baseline.append(dict(index=i, file=name, actual=labels.s[name], predicted=base,
                             unseen_rms=float(np.linalg.norm(residual_w)), prior_labels=len(prior.s)))
    rows, summaries = [], []
    for label, b in feature_bases(df):
        b = b[labels.graded]
        b -= b.mean(0)
        b /= np.sqrt(np.mean(np.sum(b * b, axis=1)))
        stats = []
        for w, d, target, gram, residual_weights, base in folds:
            a = d.T @ (w[:, None] * b)
            stats.append((a, a @ a.T, target, gram, residual_weights @ b, base))
        for penalty in (0.001, 0.01, 0.1, 1.0, 10.0):
            group = []
            for i, (a, kernel, target, gram, unseen_b, base) in enumerate(stats):
                c = kernel + penalty * gram + 1e-8 * np.eye(len(target))
                coef = a.T @ cho_solve(cho_factor(c), target)
                extra_moment = float(unseen_b @ coef)
                mse = base ** 2 - 2 * extra_moment
                group.append(dict(baseline[i], basis=label, penalty=penalty,
                                  predicted=float(np.sqrt(max(mse, 0))), raw_mse=float(mse),
                                  extra_target_moment=extra_moment, baseline=base))
            rows.extend(group)
            summaries.append(dict(basis=label, penalty=penalty, selection=summarize(group[:split]),
                                  confirmation=summarize(group[split:])))
        best = min(summaries[-5:], key=lambda s: s["selection"]["mae"])
        train.log(f"{label}: selected penalty {best['penalty']}; selection MAE {best['selection']['mae']:.6f}, "
                  f"confirmation {best['confirmation']['mae']:.6f}")
    selected = min(summaries, key=lambda s: s["selection"]["mae"])
    base_selection, base_confirmation = summarize(baseline[:split]), summarize(baseline[split:])
    selected_rows = [r for r in rows if r["basis"] == selected["basis"] and r["penalty"] == selected["penalty"]]
    errors_delta = np.array([abs(r["predicted"] - r["actual"]) - abs(r["baseline"] - r["actual"])
                             for r in selected_rows[split:]])
    rng = np.random.default_rng(129)
    boot = errors_delta[rng.integers(len(errors_delta), size=(10000, len(errors_delta)))].mean(1)
    supported = (selected["selection"]["mae"] < base_selection["mae"] and
                 selected["confirmation"]["mae"] < 0.95 * base_confirmation["mae"] and
                 float(np.quantile(boot, .95)) < 0)
    report = dict(
        experiment="nonlinear aggregate regression outside the scored span",
        premise="Supplied covariates predict target residuals outside the existing score directions.",
        validation="40 chronological scores; first half selects basis and penalty, second half confirms; "
                   "weights, anchor, moments, and scored span use earlier scores only. Unlabeled feature scaling is transductive.",
        caveat="Historical scores are adaptively selected and dependent; bootstrap is descriptive, not independent target CV.",
        source="https://arxiv.org/abs/2004.06316",
        labels_sha256=hashlib.sha256(Path(train.LABELS).read_bytes()).hexdigest(),
        selection_points=split, confirmation_points=len(eligible)-split,
        baseline=dict(selection=base_selection, confirmation=base_confirmation),
        selected=selected, confirmation_mae_difference_95pct=np.quantile(boot, [.025, .975]).tolist(),
        supported_for_further_candidate_testing=bool(supported),
        decision="retain for candidate testing" if supported else "reject promotion; feature extrapolation failed confirmation",
        summaries=summaries, baseline_results=baseline, results=rows)
    train.atomic_json(train.out("aggregate_kernel_experiment.json"), report)
    train.log(f"Baseline selection {base_selection['mae']:.6f}, confirmation {base_confirmation['mae']:.6f}; {report['decision']}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=int, default=40)
    args = parser.parse_args()
    if args.points < 4:
        parser.error("--points must be at least 4")
    with threadpool_limits(limits=2):
        run(args.points)
