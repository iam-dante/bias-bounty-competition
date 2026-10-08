"""
Round 2: build on the best public-LB submission (sub2_segment_multiplicity).

Leaderboard finding: estimating TIGER's (edge, name) multiplicity per highway segment beats one
constant per region. In that run the regional multipliers for Maricopa and Texas sat on their
lower bound (1.0), i.e. the per-segment count over-counts there, so every variant below lets
m_r go down to 0.5 and recalibrates.

    python round2.py           # -> submissions/r2_*.csv

  r2a_seg_relaxed     a=1, b=1, c=1 (same terms as sub2), m_r bounds relaxed to [0.5, 3]
  r2b_seg_no_multiref a=1, b=1, c=0   concurrent routes add no extra TIGER record
  r2c_seg_half_local  a=1, b=0.6, c=1 a local street name adds a record only part of the time
  r2d_seg_refcount    a=1, b=1, one extra record per additional route ref (exact count)
  r2e_seg_bagged      r2a recalibrated on 8 jittered-target / MC-seed bags, averaged
  r2f_ensemble        mean of r2a..r2e
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
from solution import N_SIMS, OUT, SEED, save  # noqa: E402

COLS = ["coverage_gap_score", "transport_gap", "building_gap", "poi_gap"]


def seg_frame(df: pd.DataFrame, a: float, b: float, c: float, refcount: bool = False) -> pd.DataFrame:
    """Effective TIGER route length = a*L + b*L[local name] + c*L[extra route refs]."""
    d = df.copy()
    if refcount:  # L_route_mult = sum L * (max(n_ref,1) + local_name) -> extra refs, counted exactly
        extra = (d["L_route_mult"] - d["L_route"] - d["L_route_localname"]).clip(lower=0)
    else:
        extra = d["L_route_multiref"]
    eff = a * d["L_route"] + b * d["L_route_localname"] + c * extra
    scale = np.where(d["L_route"] > 0, eff / d["L_route"].where(d["L_route"] > 0, 1), a)
    d["L_route"] = eff
    d["Lb_route"] = d["Lb_route"] * scale
    d["L_route_nearB"] = d["L_route_nearB"] * scale
    return d


def main():
    for k in ["m0", "m1", "m2", "m3"]:
        M.BOUNDS[k] = (0.5, 3.0)
    df = M.load()
    mu = M.prior_rates(df)
    with open(os.path.join(OUT, "params.json")) as fh:
        start = json.load(fh)["segment_multiplicity"]
    start.pop("abc")
    start["m"] = tuple(start["m"])
    p0 = M.Params(**start)

    variants = {"r2a_seg_relaxed": (1.0, 1.0, 1.0, False), "r2b_seg_no_multiref": (1.0, 1.0, 0.0, False),
                "r2c_seg_half_local": (1.0, 0.6, 1.0, False), "r2d_seg_refcount": (1.0, 1.0, 1.0, True)}
    preds, fitted = {}, {}
    for name, (a, b, c, rc) in variants.items():
        d = seg_frame(df, a, b, c, rc)
        p = M.calibrate(d, mu, p0, max_nfev=60, verbose=False)
        pr = M.simulate(d, mu, p, n_sims=N_SIMS, seed=SEED, return_components=True)
        loss = float(np.sum(M.moment_residuals(d, pr) ** 2))
        print(f"{name:22s} moment loss {loss:.5f}  m={tuple(round(x, 3) for x in p.m)}  "
              f"mean={pr.coverage_gap_score.mean():.4f}", flush=True)
        preds[name], fitted[name] = pr, asdict(p)

    d = seg_frame(df, 1.0, 1.0, 1.0)
    p_a = M.Params(**{**fitted["r2a_seg_relaxed"], "m": tuple(fitted["r2a_seg_relaxed"]["m"])})
    rng = np.random.default_rng(SEED)
    bag = []
    for k in range(8):
        pk = M.calibrate(d, mu, p_a, max_nfev=20, verbose=False, targets=M.perturbed_targets(rng), seed=1000 + k)
        bag.append(M.simulate(d, mu, pk, n_sims=N_SIMS, seed=SEED + k))
        print(f"  bag {k + 1}/8 mean {bag[-1].coverage_gap_score.mean():.4f}", flush=True)
    pe = preds["r2a_seg_relaxed"].copy()
    for c in COLS:
        pe[c] = np.mean([b_[c].to_numpy() for b_ in bag], axis=0)
    preds["r2e_seg_bagged"] = pe

    pf = pe.copy()
    for c in COLS:
        pf[c] = np.mean([preds[n][c].to_numpy() for n in list(preds)], axis=0)
    preds["r2f_ensemble"] = pf

    best = pd.read_csv(os.path.join(OUT, "sub2_segment_multiplicity.csv"), dtype={"GEOID": str})
    for name, pr in preds.items():
        s = save(df, pr, name)
        r = np.sqrt(np.mean((s.coverage_gap_score.to_numpy() - best.coverage_gap_score.to_numpy()) ** 2))
        print(f"  {name:22s} mean={s.coverage_gap_score.mean():.4f}  RMS diff vs sub2={r:.4f}")
    with open(os.path.join(OUT, "params_round2.json"), "w") as fh:
        json.dump(fitted, fh, indent=1)


if __name__ == "__main__":
    main()
