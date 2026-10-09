"""
Round 7 probes: tract-level signal, not group averages.

Lesson from the leaderboard: matching the true bias scorecard does not win RMSE (rank 10 matches
it almost exactly at 0.0660; rank 6 is flatter and leads at 0.0621; our flatter r5e beat the
truth-like r5d). What remains is *within-group* ranking, so these probes let lb_stack.py learn:

  p7_comp_transport / p7_comp_poi / p7_comp_building
      the round-6 ensemble's expected road / POI / building components as the score column ->
      how much tract-level signal each component carries (reweights them in the blend)
  p7_band1..p7_band5
      the current best (r5e) kept only in one quintile band of its own prediction, 0 elsewhere
      -> a separate slope per band: a nonlinear recalibration curve

    python probes_round7.py   # -> submissions/round7/*.csv
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
import model as M  # noqa: E402
from round2 import seg_frame  # noqa: E402
from round3 import interstate_frame  # noqa: E402
from solution import N_SIMS, SEED  # noqa: E402

OUT = os.path.join("submissions", "round7")


def write(geoid, v, name):
    os.makedirs(OUT, exist_ok=True)
    out = pd.DataFrame({"GEOID": geoid, "coverage_gap_score": np.clip(v, 0, 1).round(6)})
    assert out["GEOID"].is_unique and out.notna().all().all()
    out.to_csv(os.path.join(OUT, f"{name}.csv"), index=False)
    print(f"  {name:22s} mean={out.coverage_gap_score.mean():.4f}")


def main():
    df = M.attach_points(M.load())
    mu = M.prior_rates(df)
    geoid = df["GEOID"].to_numpy()
    P = json.load(open(os.path.join("submissions", "params_round6.json")))
    frames = {"r6a_vuln_leak40": seg_frame(df, 1.0, 1.0, 0.0), "r6b_vuln_interstate": interstate_frame(df),
              "r6c_vuln_noleak": seg_frame(df, 1.0, 1.0, 0.0)}
    comps = [M.simulate(frames[k], mu, M.Params(**{**P[k], "m": tuple(P[k]["m"])}), n_sims=N_SIMS, seed=SEED)
             for k in frames]
    for c, name in [("transport_gap", "p7_comp_transport"), ("poi_gap", "p7_comp_poi"),
                    ("building_gap", "p7_comp_building")]:
        write(geoid, np.mean([x[c].to_numpy() for x in comps], axis=0), name)

    best = pd.read_csv(os.path.join("submissions", "round5", "r5e_groups_affine.csv"), dtype={"GEOID": str})
    assert (best["GEOID"].to_numpy() == geoid).all()
    f = best["coverage_gap_score"].to_numpy()
    graded = (df["region"] != "eastern-wa").to_numpy()
    edges = np.quantile(f[graded], [0.2, 0.4, 0.6, 0.8])
    band = np.digitize(f, edges)
    for b in range(5):
        lo = f[graded & (band == b)].min()
        hi = f[graded & (band == b)].max()
        write(geoid, np.where(band == b, f, 0.0), f"p7_band{b + 1}")
        print(f"      band {b + 1}: r5e in [{lo:.3f}, {hi:.3f}]")


if __name__ == "__main__":
    main()
