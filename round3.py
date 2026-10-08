"""
Round 3: neighbour-aware ("leakage") facility model on top of the round-2 winners.

Physics: every comparison counts the same real objects in two independently surveyed maps and
asks which tract polygon each copy falls in. A fire station 40 m from a tract boundary can be
inside tract A in Overture but inside neighbour B in HIFLD: A then has O=1, H=0 (type
undefined) and B has O=0, H=1 (gap = 1). With a map-to-map offset sigma, the reference copy of
a point at signed distance d from a tract lands there with probability ~ Phi(d / sigma).
18-30 % of Overture facility points lie within 100 m of their tract boundary, so this matters.
Roads already carry the same effect (TIGER highways *are* tract boundaries).

    python round3.py        # -> submissions/round3/*.csv (score-only)

  r3a_leak_seg_b       r2b multiplicity (1 + local name) + leakage, sigma_loc calibrated
  r3b_leak_seg_a       r2a multiplicity (1 + local name + concurrent routes) + leakage
  r3c_leak_refcount    r2d multiplicity (exact extra route refs) + leakage
  r3d_leak_interstate  r3a, Interstates carry half a record less (usually "I- 35" only)
  r3e_leak_bagged      r3a recalibrated on 6 jittered-target bags, averaged
  r3f_leak_ensemble    mean of r3a..r3e                       (the r2f analogue)
  r3g_leak_affine      0.009 + 0.80 * r3f  (e01's leaderboard recalibration, transferred)
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, replace

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import features  # noqa: E402
import model as M  # noqa: E402
from round2 import seg_frame  # noqa: E402
from solution import N_SIMS, SEED  # noqa: E402

OUT = os.path.join("submissions", "round3")


def write(df, pred, name):
    os.makedirs(OUT, exist_ok=True)
    out = pd.DataFrame({"GEOID": df["GEOID"].to_numpy(),
                        "coverage_gap_score": np.clip(pred, 0, 1).round(6)})
    assert out["GEOID"].is_unique and out.notna().all().all()
    out.to_csv(os.path.join(OUT, f"{name}.csv"), index=False)
    print(f"  {name:22s} mean={out.coverage_gap_score.mean():.4f}", flush=True)
    return out


def interstate_frame(df, k=0.5):
    d = seg_frame(df, 1.0, 1.0, 0.0)
    d["L_route"] = (d["L_route"] - k * d["L_interstate"]).clip(lower=0)
    return d


def main():
    for r in features.REGIONS:
        features.facility_points(r)
    for k in ["m0", "m1", "m2", "m3"]:
        M.BOUNDS[k] = (0.5, 3.0)
    M.CALIB_FIELDS = M.CALIB_FIELDS + ["sigma_loc"]
    M.BOUNDS["sigma_loc"] = (1.0, 300.0)

    df = M.attach_points(M.load())
    mu = M.prior_rates(df)
    r2 = json.load(open(os.path.join("submissions", "params_round2.json")))
    start = lambda name: M.Params(**{**r2[name], "m": tuple(r2[name]["m"]), "use_points": True, "sigma_loc": 60.0})

    frames = {"r3a_leak_seg_b": (seg_frame(df, 1.0, 1.0, 0.0), "r2b_seg_no_multiref"),
              "r3b_leak_seg_a": (seg_frame(df, 1.0, 1.0, 1.0), "r2a_seg_relaxed"),
              "r3c_leak_refcount": (seg_frame(df, 1.0, 1.0, 1.0, True), "r2d_seg_refcount"),
              "r3d_leak_interstate": (interstate_frame(df), "r2b_seg_no_multiref")}
    preds, fitted = {}, {}
    for name, (d, base) in frames.items():
        p = M.calibrate(d, mu, start(base), max_nfev=60, verbose=False)
        pr = M.simulate(d, mu, p, n_sims=N_SIMS, seed=SEED, return_components=True)
        loss = float(np.sum(M.moment_residuals(d, pr) ** 2))
        print(f"{name:22s} moment loss {loss:.5f}  sigma_loc={p.sigma_loc:.0f}m  "
              f"m={tuple(round(x, 2) for x in p.m)}", flush=True)
        preds[name], fitted[name] = pr["coverage_gap_score"].to_numpy(), asdict(p)

    da = frames["r3a_leak_seg_b"][0]
    pa = M.Params(**{**fitted["r3a_leak_seg_b"], "m": tuple(fitted["r3a_leak_seg_b"]["m"])})
    rng = np.random.default_rng(SEED)
    bag = []
    for k in range(6):
        pk = M.calibrate(da, mu, pa, max_nfev=20, verbose=False, targets=M.perturbed_targets(rng), seed=2000 + k)
        bag.append(M.simulate(da, mu, pk, n_sims=N_SIMS, seed=SEED + k)["coverage_gap_score"].to_numpy())
        print(f"  bag {k + 1}/6  sigma_loc={pk.sigma_loc:.0f}m", flush=True)
    preds["r3e_leak_bagged"] = np.mean(bag, axis=0)
    preds["r3f_leak_ensemble"] = np.mean([preds[n] for n in list(preds)], axis=0)
    preds["r3g_leak_affine"] = 0.009 + 0.80 * preds["r3f_leak_ensemble"]

    print("[writing]")
    best = pd.read_csv(os.path.join("submissions", "r2f_ensemble.csv"), dtype={"GEOID": str})
    for name, v in preds.items():
        out = write(df, v, name)
        r = np.sqrt(np.mean((out.coverage_gap_score.to_numpy() - best.coverage_gap_score.to_numpy()) ** 2))
        print(f"      RMS diff vs r2f = {r:.4f}")
    json.dump(fitted, open(os.path.join("submissions", "params_round3.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
