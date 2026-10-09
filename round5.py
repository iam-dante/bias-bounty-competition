"""
Round 5: group effects inside the model, from first principles, not as post-hoc offsets.

stack3 (public 0.066116) fixed our too-flat scorecard by adding group offsets after the fact.
Here the mechanisms that create those disparities are part of the generative model, and the true
target's public scorecard ratios join the calibration moments:

  * unseen-facility rate x exp(gf . groups): rural / wildland / dryland fire protection is
    volunteer and state stations (CAL FIRE, Oklahoma Forestry) that HIFLD lists but Overture
    often misses, so the share of reference facilities Overture never shows rises with hazard
    exposure and rurality;
  * TIGER-length multiplier x exp(gr . groups): rural state highways are tagged below
    primary/secondary in OSM and carry extra TIGER name records (route + county road name);
  * CBP density x exp(gc_rural): Overture's place coverage thins faster than real establishments.

The calibration decides how much of each disparity is roads vs facilities, inside the exact
scoring formula, so every tract's prediction moves for a physical reason.

    python round5.py     # -> submissions/round5/*.csv (score-only)

  r5a_groups_leak40      r2b multiplicity + leakage (sigma 40 m) + group effects
  r5b_groups_interstate  r5a with Interstates carrying half a record less
  r5c_groups_noleak      r5a without leakage
  r5d_groups_ensemble    mean of r5a..r5c
  r5e_groups_affine      0.009 + 0.80 * r5d   (e01's leaderboard recalibration, transferred)
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, replace

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import model as M  # noqa: E402
from round2 import seg_frame  # noqa: E402
from round3 import interstate_frame  # noqa: E402
from solution import N_SIMS, SEED  # noqa: E402

OUT = os.path.join("submissions", "round5")
GROUP_FIELDS = ["gf_rural", "gf_wild", "gf_drw", "gf_tribal", "gf_svi", "gf_heat",
                "gr_rural", "gr_wild", "gr_drw", "gr_heat", "gc_rural"]


def write(df, v, name):
    os.makedirs(OUT, exist_ok=True)
    out = pd.DataFrame({"GEOID": df["GEOID"].to_numpy(), "coverage_gap_score": np.clip(v, 0, 1).round(6)})
    assert out["GEOID"].is_unique and out.notna().all().all()
    out.to_csv(os.path.join(OUT, f"{name}.csv"), index=False)
    return out


def main():
    for k in ["m0", "m1", "m2", "m3"]:
        M.BOUNDS[k] = (0.5, 3.0)
    for k in GROUP_FIELDS:
        M.BOUNDS[k] = (-1.0, 1.0) if k.startswith("gr") else (-2.5, 2.5)
    M.CALIB_FIELDS = M.CALIB_FIELDS + GROUP_FIELDS
    M.SCORECARD_W = 0.5

    df = M.attach_points(M.load())
    mu = M.prior_rates(df)
    r3 = json.load(open(os.path.join("submissions", "params_round3.json")))["r3a_leak_seg_b"]
    base = M.Params(**{**r3, "m": tuple(r3["m"]), "use_points": True, "sigma_loc": 40.0})

    variants = {"r5a_groups_leak40": (seg_frame(df, 1.0, 1.0, 0.0), base),
                "r5b_groups_interstate": (interstate_frame(df), base),
                "r5c_groups_noleak": (seg_frame(df, 1.0, 1.0, 0.0), replace(base, use_points=False))}
    truth = M.scorecard_truth()
    preds, fitted = {}, {}
    for name, (d, p0) in variants.items():
        p = M.calibrate(d, mu, p0, max_nfev=40, verbose=False)
        pr = M.simulate(d, mu, p, n_sims=N_SIMS, seed=SEED, return_components=True)
        loss = float(np.sum(M.moment_residuals(d, pr) ** 2))
        sc = M.scorecard_ratios(d, pr["coverage_gap_score"].to_numpy())
        print(f"{name:22s} moment loss {loss:.5f}", flush=True)
        print("    scorecard pred/truth: " + ", ".join(f"{k.split('_')[0]} {sc[k]:.2f}/{truth[k]:.2f}" for k in truth))
        print("    effects: " + ", ".join(f"{k}={getattr(p, k):+.2f}" for k in GROUP_FIELDS), flush=True)
        preds[name], fitted[name] = pr["coverage_gap_score"].to_numpy(), asdict(p)

    preds["r5d_groups_ensemble"] = np.mean(list(preds.values()), axis=0)
    preds["r5e_groups_affine"] = 0.009 + 0.80 * preds["r5d_groups_ensemble"]
    ref = pd.read_csv(os.path.join("submissions", "round3", "stack3_scorecard_groups.csv"), dtype={"GEOID": str})
    for name, v in preds.items():
        out = write(df, v, name)
        r = np.sqrt(np.mean((out.coverage_gap_score.to_numpy() - ref.coverage_gap_score.to_numpy()) ** 2))
        print(f"  {name:22s} mean={out.coverage_gap_score.mean():.4f}  RMS diff vs stack3={r:.4f}", flush=True)
    json.dump(fitted, open(os.path.join("submissions", "params_round5.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
