"""
Round 8: regional specialists, assembled; plus tract-level probes for the blend.

Until now one parameter set served all four regions (only the TIGER multiplier was regional).
The regions differ physically: CAL FIRE stations in the Northern California wildland, a gridded
desert metro in Maricopa, volunteer departments on Eastern Oklahoma's tribal land, FM-road
country in Texas. Each region now gets its own road / facility / business parameters, fitted on
that region's published statistics (undefined share, rural x burned cell means, Paradise tracts
for Northern California) with a ridge prior that pulls them toward the round-6 global fit, so
five moments per region cannot overfit eight parameters. The specialists are stitched into one
file; eastern-wa (not graded) keeps the global fit.

    python round8.py    # -> submissions/round8/*.csv (score-only)

  r8a_regional_noleak     round-6 no-leakage model, one specialist per region
  r8b_regional_interstate round-6 Interstate + leakage model, one specialist per region
  r8c_regional_ensemble   mean of r8a, r8b
  r8d_regional_affine     0.009 + 0.80 * r8c  (same recalibration as r5e / r6e)
  p8_transport, p8_poi    round-6 ensemble's expected road / POI components (blend columns:
                          how much tract-level signal each component carries)
  p8_best_top20           current best (stack5) kept only on its top 20 % of graded tracts
                          (blend column: are the highest predictions over-called?)
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, replace

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import model as M  # noqa: E402
from round2 import seg_frame  # noqa: E402
from round3 import interstate_frame  # noqa: E402
from solution import N_SIMS, SEED  # noqa: E402

OUT = os.path.join("submissions", "round8")
# per-region free parameters and the prior scale of each (ridge pulls toward the global value)
REG_FIELDS = {"beta_bnd": 0.5, "sigma_t": 0.2, "r_fire": 0.1, "alpha_fire_x": 0.2,
              "gf_rural": 0.5, "gr_rural": 0.2, "gr_wild": 0.2, "gr_drw": 0.2}
LAMBDA = 0.02


def write(geoid, v, name):
    os.makedirs(OUT, exist_ok=True)
    out = pd.DataFrame({"GEOID": geoid, "coverage_gap_score": np.clip(v, 0, 1).round(6)})
    assert out["GEOID"].is_unique and out.notna().all().all()
    out.to_csv(os.path.join(OUT, f"{name}.csv"), index=False)
    print(f"  {name:26s} mean={out.coverage_gap_score.mean():.4f}", flush=True)
    return out


def region_residuals(df, pred, region):
    m = (df["region"] == region).to_numpy()
    res = [3.0 * (pred.loc[m, "p_undef_any"].mean() - M.UNDEF[region])]
    for (ru, bu), tgt in zip([(1, 1), (1, 0), (0, 1), (0, 0)], M.CELL[region]):
        mm = m & (df["rural"] == ru).to_numpy() & (df["burned"] == bu).to_numpy()
        res.append((2.0 if bu == 0 else 1.0) * (pred.loc[mm, "coverage_gap_score"].mean() - tgt))
    if region == "northern-ca":
        idx = pd.Series(np.arange(len(df)), index=df["GEOID"].to_numpy())
        for g, (cs, rd, _) in M.PARADISE.items():
            i = idx[g]
            res += [0.3 * (pred["coverage_gap_score"].iat[i] - cs), 0.3 * (pred["transport_gap"].iat[i] - rd)]
    return np.asarray(res)


def specialist(d, mu, p0: M.Params, region: str, m_idx: int) -> M.Params:
    """Fit this region's multiplier + REG_FIELDS on its own moments, ridge-pulled to p0."""
    names = ["m"] + list(REG_FIELDS)
    x0 = np.array([p0.m[m_idx]] + [getattr(p0, k) for k in REG_FIELDS])
    scale = np.array([0.3] + list(REG_FIELDS.values()))
    lo = np.array([0.5, 0.0, 0.05, 0.3, 0.0, -2.5, -1.0, -1.0, -1.0])
    hi = np.array([3.0, 2.0, 0.8, 0.99, 1.0, 2.5, 1.0, 1.0, 1.0])

    def to_params(x):
        m = list(p0.m)
        m[m_idx] = x[0]
        return replace(p0, m=tuple(m), **dict(zip(names[1:], x[1:])))

    def f(x):
        pred = M.simulate(d, mu, to_params(x), n_sims=96, seed=123, return_components=True)
        return np.r_[region_residuals(d, pred, region), LAMBDA * (x - x0) / scale]

    sol = least_squares(f, np.clip(x0, lo + 1e-3, hi - 1e-3), bounds=(lo, hi), diff_step=0.05,
                        max_nfev=30, x_scale="jac")
    r0, r1 = f(x0)[:-len(x0)], sol.fun[:-len(x0)]
    print(f"    {region:17s} region loss {np.sum(r0 ** 2):.5f} -> {np.sum(r1 ** 2):.5f}  "
          + ", ".join(f"{k}={v:+.2f}" for k, v in zip(names, sol.x)), flush=True)
    return to_params(sol.x)


def main():
    for k in ["m0", "m1", "m2", "m3"]:
        M.BOUNDS[k] = (0.5, 3.0)
    df = M.attach_points(M.load())
    mu = M.prior_rates(df)
    geoid = df["GEOID"].to_numpy()
    P6 = json.load(open(os.path.join("submissions", "params_round6.json")))
    glob6 = lambda k: M.Params(**{**P6[k], "m": tuple(P6[k]["m"])})

    variants = {"r8a_regional_noleak": (seg_frame(df, 1.0, 1.0, 0.0), glob6("r6c_vuln_noleak")),
                "r8b_regional_interstate": (interstate_frame(df), glob6("r6b_vuln_interstate"))}
    preds, fitted = {}, {}
    for name, (d, pg) in variants.items():
        print(f"[{name}]", flush=True)
        assembled = M.simulate(d, mu, pg, n_sims=N_SIMS, seed=SEED)["coverage_gap_score"].to_numpy().copy()
        fitted[name] = {"global": asdict(pg)}
        for i, region in enumerate(M.CALIB_REGIONS):
            pr = specialist(d, mu, pg, region, i)
            fitted[name][region] = asdict(pr)
            m = (d["region"] == region).to_numpy()
            assembled[m] = M.simulate(d, mu, pr, n_sims=N_SIMS, seed=SEED)["coverage_gap_score"].to_numpy()[m]
        preds[name] = assembled

    preds["r8c_regional_ensemble"] = np.mean(list(preds.values()), axis=0)
    preds["r8d_regional_affine"] = 0.009 + 0.80 * preds["r8c_regional_ensemble"]
    best = pd.read_csv(os.path.join("submissions", "best", "stack5_r6.csv"), dtype={"GEOID": str})
    assert (best["GEOID"].to_numpy() == geoid).all()
    for name, v in preds.items():
        out = write(geoid, v, name)
        r = np.sqrt(np.mean((out.coverage_gap_score.to_numpy() - best.coverage_gap_score.to_numpy()) ** 2))
        print(f"      RMS diff vs stack5 = {r:.4f}", flush=True)

    # tract-level probes for the blend
    frames = {"r6a_vuln_leak40": seg_frame(df, 1.0, 1.0, 0.0), "r6b_vuln_interstate": interstate_frame(df),
              "r6c_vuln_noleak": seg_frame(df, 1.0, 1.0, 0.0)}
    comps = [M.simulate(frames[k], mu, glob6(k), n_sims=N_SIMS, seed=SEED) for k in frames]
    write(geoid, np.mean([c["transport_gap"].to_numpy() for c in comps], axis=0), "p8_transport")
    write(geoid, np.mean([c["poi_gap"].to_numpy() for c in comps], axis=0), "p8_poi")
    f = best["coverage_gap_score"].to_numpy()
    graded = (df["region"] != "eastern-wa").to_numpy()
    cut = np.quantile(f[graded], 0.8)
    write(geoid, np.where(f >= cut, f, 0.0), "p8_best_top20")
    print(f"      top-20% band: stack5 >= {cut:.3f}")
    json.dump(fitted, open(os.path.join("submissions", "params_round8.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
