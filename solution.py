"""
Bias Bounty Mapping Equity Challenge -- end-to-end pipeline.

    python solution.py                 # features (cached) -> calibrate -> 5 submissions
    python solution.py --force-features

Stages
  1. src/features.py : DuckDB spatial feature engineering on the provided GeoParquet
  2. src/model.py    : generative model of the withdrawn reference layers + Monte-Carlo
                       expectation of the official composite + method-of-moments calibration
  3. this file       : the five submission ideas, diagnostics, CSVs in submissions/

Five ideas (one submission each)
  sub1_calibrated            structural model, parameters fitted to the public moments (official
                             undefined shares, published region x urban/rural x burned means, ...)
  sub2_segment_multiplicity  TIGER name multiplicity estimated per highway segment (local street
                             name / concurrent routes) instead of one constant per region
  sub3_shrunk                sub1 shrunk 25% toward the published cell means (robustness)
  sub4_posterior_bagged      8 calibrations on jittered targets + MC seeds, averaged
                             (posterior-predictive mean under parameter uncertainty)
  sub5_ensemble              mean of sub1..sub4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, replace

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import features  # noqa: E402
import model as M  # noqa: E402

OUT = "submissions"
SEED = 2026
N_SIMS = 512


def save(df_feat: pd.DataFrame, pred: pd.DataFrame, name: str, components: bool = False) -> pd.DataFrame:
    # Zindi scores EVERY column in the file: optional component columns are graded too and cost
    # ~0.08 RMSE (r2f: 0.149 with components vs 0.069 score-only), so write the score only.
    cols = ["GEOID", "transport_gap", "building_gap", "poi_gap", "coverage_gap_score"] if components \
        else ["GEOID", "coverage_gap_score"]
    sub = pred[cols].copy()
    for c in cols[1:]:
        sub[c] = sub[c].clip(0, 1).round(6)
    assert sub["GEOID"].is_unique and sub[cols].notna().all().all() and len(sub) == len(df_feat)
    os.makedirs(OUT, exist_ok=True)
    sub.to_csv(os.path.join(OUT, f"{name}.csv"), index=False)
    return sub


def segment_multiplicity_frame(df: pd.DataFrame, a: float, b: float, c: float) -> pd.DataFrame:
    """Idea 3: TIGER records per edge = a + b*[has local street name] + c*[concurrent routes].
    Folded into the model by rescaling the route length the simulator sees (m_r := 1)."""
    d = df.copy()
    eff = a * d["L_route"] + b * d["L_route_localname"] + c * d["L_route_multiref"]
    scale = np.where(d["L_route"] > 0, eff / d["L_route"].where(d["L_route"] > 0, 1), a)
    d["L_route"] = eff
    d["Lb_route"] = d["Lb_route"] * scale
    d["L_route_nearB"] = d["L_route_nearB"] * scale
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force-features", action="store_true")
    ap.add_argument("--fast", action="store_true", help="fewer MC draws / optimiser steps")
    args = ap.parse_args()
    n_sims = 128 if args.fast else N_SIMS
    nfev = 25 if args.fast else 60

    print("[1/4] features")
    for r in M.REGIONS:
        features.build_region(r, force=args.force_features)
    df = M.load()
    mu = M.prior_rates(df)
    print(f"    {len(df)} scored tracts, {df.shape[1]} columns")

    print("[2/4] baseline: structural model with domain priors (diagnostic only)")
    p_prior = M.Params()
    pred0 = M.simulate(df, mu, p_prior, n_sims=n_sims, seed=SEED, return_components=True)
    print(M.moment_report(df, pred0).to_string(index=False))

    print("[3/4] calibration (method of moments on public statistics)")
    # idea 1: calibrated structural model
    p_cal = M.calibrate(df, mu, p_prior, max_nfev=nfev)
    pred1 = M.simulate(df, mu, p_cal, n_sims=n_sims, seed=SEED, return_components=True)
    print(M.moment_report(df, pred1).to_string(index=False))

    # idea 2: per-segment TIGER multiplicity; (a, b, c) chosen by moment loss, m_r fixed at 1
    best = None
    for a, b, c in [(1.0, 1.0, 1.0), (1.3, 0.7, 0.8), (1.5, 0.5, 0.5)]:
        d2 = segment_multiplicity_frame(df, a, b, c)
        p2 = M.calibrate(d2, mu, replace(p_cal, m=(1.0, 1.0, 1.0, 1.0)), max_nfev=nfev, verbose=False)
        pr = M.simulate(d2, mu, p2, n_sims=n_sims, seed=SEED, return_components=True)
        loss = float(np.sum(M.moment_residuals(d2, pr) ** 2))
        print(f"    segment multiplicity a={a} b={b} c={c}: moment loss {loss:.5f}")
        if best is None or loss < best[0]:
            best = (loss, (a, b, c), p2, pr)
    pred2 = best[3]

    # idea 3: shrink toward the published cell means
    cell = np.zeros(len(df))
    cells = dict(M.CELL)
    for r in M.REGIONS:  # regions without published means (eastern-wa) use the 4-region average
        cells.setdefault(r, tuple(np.mean([M.CELL[q] for q in M.CALIB_REGIONS], axis=0)))
    for r in M.REGIONS:
        for (ru, bu), tgt in zip([(1, 1), (1, 0), (0, 1), (0, 0)], cells[r]):
            mm = ((df["region"] == r) & (df["rural"] == ru) & (df["burned"] == bu)).to_numpy()
            cell[mm] = tgt
    pred3 = pred1.copy()
    w = 0.25
    pred3["coverage_gap_score"] = (1 - w) * pred1["coverage_gap_score"] + w * cell

    # idea 4: posterior-predictive bagging -- recalibrate on jittered targets / MC seeds, average
    rng = np.random.default_rng(SEED)
    bag = []
    n_bags = 4 if args.fast else 8
    for k in range(n_bags):
        pk = M.calibrate(df, mu, p_cal, max_nfev=max(10, nfev // 3), verbose=False,
                         targets=M.perturbed_targets(rng), seed=1000 + k)
        bag.append(M.simulate(df, mu, pk, n_sims=n_sims, seed=SEED + k))
        print(f"    bag {k + 1}/{n_bags}: mean score {bag[-1].coverage_gap_score.mean():.4f}")
    pred4 = pred1.copy()
    for c in ["coverage_gap_score", "transport_gap", "building_gap", "poi_gap"]:
        pred4[c] = np.mean([b_[c].to_numpy() for b_ in bag], axis=0)

    # idea 5: ensemble of ideas 1-4
    pred5 = pred1.copy()
    for c in ["coverage_gap_score", "transport_gap", "building_gap", "poi_gap"]:
        pred5[c] = (pred1[c] + pred2[c] + pred3[c] + pred4[c]) / 4

    print("[4/4] writing submissions")
    subs = {"sub1_calibrated": pred1, "sub2_segment_multiplicity": pred2, "sub3_shrunk": pred3,
            "sub4_posterior_bagged": pred4, "sub5_ensemble": pred5}
    for name, pr in subs.items():
        s = save(df, pr, name)
        print(f"    {name:28s} mean={s.coverage_gap_score.mean():.4f} sd={s.coverage_gap_score.std():.4f} "
              f"-> {OUT}/{name}.csv")
    corr = pd.DataFrame({k: v["coverage_gap_score"].to_numpy() for k, v in subs.items()}).corr().round(3)
    print(corr.to_string())
    with open(os.path.join(OUT, "params.json"), "w") as fh:
        json.dump({"prior": asdict(p_prior), "calibrated": asdict(p_cal),
                   "segment_multiplicity": {"abc": best[1], **asdict(best[2])}}, fh, indent=1)


if __name__ == "__main__":
    main()
