"""
Round 3b: leakage with the map-to-map offset FIXED at physical values.

When sigma_loc is free, the moment calibration pushes it to its 1 m floor: the public moments
pin only average POI levels, and leakage raises POI gaps, so the optimiser switches it off and
keeps the other facility parameters unchanged. sigma is not identified by the moments, only by
the leaderboard. So fix it at plausible geocoding offsets and refit everything else around it.

    python round3_fixed.py   # -> submissions/round3/r3h..r3k (score-only)

  r3h_sigma40     r2b multiplicity + leakage, sigma = 40 m, other parameters recalibrated
  r3i_sigma100    same, sigma = 100 m
  r3j_sigma40_affine   0.009 + 0.80 * r3h
  r3k_sigma100_affine  0.009 + 0.80 * r3i
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import model as M  # noqa: E402
from round2 import seg_frame  # noqa: E402
from round3 import write  # noqa: E402
from solution import N_SIMS, SEED  # noqa: E402


def main():
    for k in ["m0", "m1", "m2", "m3"]:
        M.BOUNDS[k] = (0.5, 3.0)
    df = M.attach_points(M.load())
    mu = M.prior_rates(df)
    r2 = json.load(open(os.path.join("submissions", "params_round2.json")))["r2b_seg_no_multiref"]
    d = seg_frame(df, 1.0, 1.0, 0.0)
    out = {}
    for sig, name in [(40.0, "r3h_sigma40"), (100.0, "r3i_sigma100")]:
        p0 = M.Params(**{**r2, "m": tuple(r2["m"]), "use_points": True, "sigma_loc": sig})
        p = M.calibrate(d, mu, p0, max_nfev=60, verbose=False)      # sigma_loc not in CALIB_FIELDS
        pr = M.simulate(d, mu, p, n_sims=N_SIMS, seed=SEED, return_components=True)
        loss = float(np.sum(M.moment_residuals(d, pr) ** 2))
        print(f"{name:14s} moment loss {loss:.5f}  mean poi {pr.poi_gap.mean():.4f}  "
              f"alpha_fire_x={p.alpha_fire_x:.2f} r_fire={p.r_fire:.2f}", flush=True)
        out[name] = pr["coverage_gap_score"].to_numpy()
        write(df, out[name], name)
    write(df, 0.009 + 0.80 * out["r3h_sigma40"], "r3j_sigma40_affine")
    write(df, 0.009 + 0.80 * out["r3i_sigma100"], "r3k_sigma100_affine")


if __name__ == "__main__":
    main()
