"""
Round 9: fix the route rule in the worst-fitting region, then refit.

Region-by-region fits (round 8) showed Maricopa carries 8x Eastern Oklahoma's misfit and its
TIGER multiplier is pushed to the floor: the model predicts 50 % of Maricopa tracts have no
TIGER highway, the official figure is 55 %. Cause: Overture's `US:US:Historic` / `US:I:Future`
designations (old US 80 / US 89 alignments, proposed Interstates) ride on ordinary streets that
TIGER does not class S1100/S1200. Dropping them in Arizona moves Maricopa's undefined share
from 0.506 to 0.542 (official 0.55). In California and Oklahoma dropping them would break the
official bound road-undefined <= any-undefined (CA 0.386 > 0.37), so they are kept there.

    python round9.py    # rebuilds AZ + CA road features, refits -> submissions/round9/*.csv

  r9a_routefix_noleak      round-6 no-leakage model, recalibrated on the fixed route rule
  r9b_routefix_interstate  round-6 Interstate + leakage model, recalibrated
  r9c_routefix_ensemble    mean of r9a, r9b
  r9d_routefix_affine      0.009 + 0.80 * r9c
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import features  # noqa: E402
import model as M  # noqa: E402
from round2 import seg_frame  # noqa: E402
from round3 import interstate_frame  # noqa: E402
from round6 import GROUP_FIELDS  # noqa: E402
from solution import N_SIMS, SEED  # noqa: E402

OUT = os.path.join("submissions", "round9")


def main():
    for r in ("maricopa-az", "northern-ca"):   # CA rebuilt to undo an earlier AZ+CA variant
        features.build_region(r, force=True)
    for k in ["m0", "m1", "m2", "m3"]:
        M.BOUNDS[k] = (0.5, 3.0)
    for k in GROUP_FIELDS:
        M.BOUNDS[k] = (-1.0, 1.0) if k.startswith("gr") else (-2.5, 2.5)
    M.CALIB_FIELDS = M.CALIB_FIELDS + GROUP_FIELDS
    M.SCORECARD_W = 0.5

    df = M.attach_points(M.load())
    mu = M.prior_rates(df)
    for r in M.CALIB_REGIONS:
        d = df[df["region"] == r]
        print(f"  {r:17s} road-undefined proxy {((d.L_route <= 1) & (d.Lb_route <= 1)).mean():.3f}"
              f"  (official any-undefined {M.UNDEF[r]:.2f})", flush=True)
    P6 = json.load(open(os.path.join("submissions", "params_round6.json")))
    start = lambda k: M.Params(**{**P6[k], "m": tuple(P6[k]["m"])})
    variants = {"r9a_routefix_noleak": (seg_frame(df, 1.0, 1.0, 0.0), start("r6c_vuln_noleak")),
                "r9b_routefix_interstate": (interstate_frame(df), start("r6b_vuln_interstate"))}
    preds, fitted = {}, {}
    for name, (d, p0) in variants.items():
        before = float(np.sum(M.moment_residuals(d, M.simulate(d, mu, p0, n_sims=96, seed=123,
                                                                return_components=True)) ** 2))
        p = M.calibrate(d, mu, p0, max_nfev=40, verbose=False)
        pr = M.simulate(d, mu, p, n_sims=N_SIMS, seed=SEED, return_components=True)
        loss = float(np.sum(M.moment_residuals(d, pr) ** 2))
        und = {r: pr.loc[(d["region"] == r).to_numpy(), "p_undef_any"].mean() for r in M.CALIB_REGIONS}
        print(f"{name:24s} moment loss {before:.5f} (round-6 params on fixed features) -> {loss:.5f}", flush=True)
        print("    undefined share pred/official: " + ", ".join(f"{r.split('-')[0]} {und[r]:.3f}/{M.UNDEF[r]:.2f}"
                                                          for r in M.CALIB_REGIONS), flush=True)
        preds[name], fitted[name] = pr["coverage_gap_score"].to_numpy(), asdict(p)
    preds["r9c_routefix_ensemble"] = np.mean(list(preds.values()), axis=0)
    preds["r9d_routefix_affine"] = 0.009 + 0.80 * preds["r9c_routefix_ensemble"]

    os.makedirs(OUT, exist_ok=True)
    best = pd.read_csv(os.path.join("submissions", "best", "stack5_r6.csv"), dtype={"GEOID": str})
    assert (best["GEOID"].to_numpy() == df["GEOID"].to_numpy()).all()
    for name, v in preds.items():
        out = pd.DataFrame({"GEOID": df["GEOID"].to_numpy(), "coverage_gap_score": np.clip(v, 0, 1).round(6)})
        out.to_csv(os.path.join(OUT, f"{name}.csv"), index=False)
        r = np.sqrt(np.mean((out.coverage_gap_score.to_numpy() - best.coverage_gap_score.to_numpy()) ** 2))
        print(f"  {name:24s} mean={out.coverage_gap_score.mean():.4f}  RMS diff vs stack5={r:.4f}", flush=True)
    json.dump(fitted, open(os.path.join("submissions", "params_round9.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
