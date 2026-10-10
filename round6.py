"""
Round 6: round 5 plus social- and climate-vulnerability effects on roads and facilities, and the
"High Hazard + High Vulnerability" ratio as a ninth scorecard moment (r5e: SVI 1.14 vs true 1.22,
CVI 1.40 vs 1.51, intersectional 1.12 vs 1.21).

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

    python round6.py     # -> submissions/round6/*.csv (score-only)

  r6a_vuln_leak40      r5a + vulnerability effects, started from r5a's fitted parameters
  r6b_vuln_interstate  same for r5b (Interstates carry half a record less)
  r6c_vuln_noleak      same for r5c (no leakage)
  r6d_vuln_ensemble    mean of r6a..r6c
  r6e_vuln_affine      0.009 + 0.80 * r6d   (same recalibration as the current best, r5e)
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

OUT = os.path.join("submissions", "round6")
GROUP_FIELDS = ["gf_rural", "gf_wild", "gf_drw", "gf_tribal", "gf_svi", "gf_heat",
                "gr_rural", "gr_wild", "gr_drw", "gr_heat", "gc_rural",
                "gf_cvi", "gr_svi", "gr_cvi"]


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
    r5 = json.load(open(os.path.join("submissions", "params_round5.json")))
    start = lambda k: M.Params(**{**r5[k], "m": tuple(r5[k]["m"])})

    variants = {"r6a_vuln_leak40": (seg_frame(df, 1.0, 1.0, 0.0), start("r5a_groups_leak40")),
                "r6b_vuln_interstate": (interstate_frame(df), start("r5b_groups_interstate")),
                "r6c_vuln_noleak": (seg_frame(df, 1.0, 1.0, 0.0), start("r5c_groups_noleak"))}
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

    preds["r6d_vuln_ensemble"] = np.mean(list(preds.values()), axis=0)
    preds["r6e_vuln_affine"] = 0.009 + 0.80 * preds["r6d_vuln_ensemble"]
    ref = pd.read_csv(os.path.join("submissions", "round5", "r5e_groups_affine.csv"), dtype={"GEOID": str})
    for name, v in preds.items():
        out = write(df, v, name)
        r = np.sqrt(np.mean((out.coverage_gap_score.to_numpy() - ref.coverage_gap_score.to_numpy()) ** 2))
        print(f"  {name:22s} mean={out.coverage_gap_score.mean():.4f}  RMS diff vs r5e={r:.4f}", flush=True)
    json.dump(fitted, open(os.path.join("submissions", "params_round6.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
