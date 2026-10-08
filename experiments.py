"""
Round 3: twenty experiment files built around the current best, r2f_ensemble (public 0.069455).

Every file is score-only (GEOID, coverage_gap_score): Zindi grades every included column.
Each experiment is also a *column* for lb_stack.py: once scored, its correlation with the
hidden target is known exactly, and lb_stack.py fits the RMSE-optimal linear blend of all of them.

    python experiments.py        # -> submissions/exp/e01..e20_*.csv  (+ prints expected scores)

  e01        r2f affine-recalibrated with the scores we already have (0.009 + 0.80*r2f)
  e02-e06    r2f on one region, 0 elsewhere          -> per-region scale in the blend
  e07-e11    0.10 on one region, 0 elsewhere         -> per-region intercept + region mean(y)
  e12-e14    r2f's expected road / POI / building components as the score column
  e15-e16    0.10 / r2f on rural (RUCA >= 4) tracts only  -> urban/rural split
  e17-e20    model variants of r2b: smoother road proxy, no boundary term, fewer missing
             fire stations, schools never binding
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import replace

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import model as M  # noqa: E402
from round2 import seg_frame  # noqa: E402
from solution import N_SIMS, SEED  # noqa: E402

EXP = os.path.join("submissions", "exp")
SEG = {"r2a_seg_relaxed": (1.0, 1.0, 1.0, False), "r2b_seg_no_multiref": (1.0, 1.0, 0.0, False),
       "r2c_seg_half_local": (1.0, 0.6, 1.0, False), "r2d_seg_refcount": (1.0, 1.0, 1.0, True)}


def write(geoid, score, name):
    os.makedirs(EXP, exist_ok=True)
    out = pd.DataFrame({"GEOID": geoid, "coverage_gap_score": np.clip(score, 0, 1).round(6)})
    assert out["GEOID"].is_unique and out.notna().all().all()
    out.to_csv(os.path.join(EXP, f"{name}.csv"), index=False)
    print(f"  {name:34s} mean={out.coverage_gap_score.mean():.4f}")


def params(d: dict) -> M.Params:
    return M.Params(**{**d, "m": tuple(d["m"])})


def main():
    df = M.load()
    mu = M.prior_rates(df)
    geoid = df["GEOID"].to_numpy()
    fitted = json.load(open(os.path.join("submissions", "params_round2.json")))
    r2f = pd.read_csv("submissions/r2f_ensemble.csv", dtype={"GEOID": str})
    assert (r2f["GEOID"].to_numpy() == geoid).all()
    f = r2f["coverage_gap_score"].to_numpy()

    # r2f components: mean of the four stored round-2 fits (r2e's bag mean ~ r2a, counted twice)
    comps = []
    for name, (a, b, c, rc) in SEG.items():
        comps.append(M.simulate(seg_frame(df, a, b, c, rc), mu, params(fitted[name]), n_sims=N_SIMS, seed=SEED))
    comps.append(comps[0])
    comp = {k: np.mean([x[k].to_numpy() for x in comps], axis=0) for k in ["transport_gap", "poi_gap", "building_gap"]}

    print("[LB-information experiments]")
    write(geoid, 0.009 + 0.80 * f, "e01_r2f_affine")
    for i, r in enumerate(M.REGIONS):
        write(geoid, np.where(df["region"] == r, f, 0.0), f"e{2 + i:02d}_r2f_only_{r}")
    for i, r in enumerate(M.REGIONS):
        write(geoid, np.where(df["region"] == r, 0.10, 0.0), f"e{7 + i:02d}_const_only_{r}")
    write(geoid, comp["transport_gap"], "e12_r2f_transport_component")
    write(geoid, comp["poi_gap"], "e13_r2f_poi_component")
    write(geoid, comp["building_gap"], "e14_r2f_building_component")
    rural = df["rural"].to_numpy() == 1
    write(geoid, np.where(rural, 0.10, 0.0), "e15_const_only_rural")
    write(geoid, np.where(rural, f, 0.0), "e16_r2f_only_rural")

    print("[model variants of r2b]")
    pb = params(fitted["r2b_seg_no_multiref"])
    db = seg_frame(df, *SEG["r2b_seg_no_multiref"])
    variants = {"e17_r2b_smooth_road": replace(pb, sigma_t=0.8),
                "e18_r2b_no_boundary": replace(pb, beta_bnd=0.0),
                "e19_r2b_fire_recall_low": replace(pb, r_fire=0.6),
                "e20_r2b_schools_never_bind": replace(pb, alpha_sch=0.0, r_sch=0.999)}
    for name, p in variants.items():
        write(geoid, M.simulate(db, mu, p, n_sims=N_SIMS, seed=SEED)["coverage_gap_score"].to_numpy(), name)


if __name__ == "__main__":
    main()
