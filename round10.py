"""
Round 10: regional specialists on the fixed road features (round 8's method on round 9's model).

Round 8 fitted one parameter set per region but ran before the Arizona route fix; round 9 fixed
the roads but kept one shared parameter set. Here each region refits its TIGER multiplier and
road / facility parameters on its own published moments (ridge-pulled toward the round-9 global
fit), on the corrected features, and the specialists are stitched into one file.

The public aggregates are already matched, so the purpose of this file is tract-level: it is a
physically different, near-best model whose leaderboard score gives the blend (lb_stack.py) a
new exact moment of the target, which is how stack5 and stack6 beat the previous best.

    python round10.py   # -> submissions/round10/*.csv (score-only)

  r10a_regional_noleak      round-9 no-leakage model, one specialist per region
  r10b_regional_interstate  round-9 Interstate + leakage model, one specialist per region
  r10c_regional_ensemble    mean of r10a, r10b
  r10d_regional_affine      0.009 + 0.80 * r10c   (the candidate to submit)
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import model as M  # noqa: E402
from round2 import seg_frame  # noqa: E402
from round3 import interstate_frame  # noqa: E402
from round8 import specialist  # noqa: E402
from solution import N_SIMS, SEED  # noqa: E402

OUT = os.path.join("submissions", "round10")


def main():
    for k in ["m0", "m1", "m2", "m3"]:
        M.BOUNDS[k] = (0.5, 3.0)
    df = M.attach_points(M.load())
    mu = M.prior_rates(df)
    geoid = df["GEOID"].to_numpy()
    P9 = json.load(open(os.path.join("submissions", "params_round9.json")))
    glob9 = lambda k: M.Params(**{**P9[k], "m": tuple(P9[k]["m"])})
    variants = {"r10a_regional_noleak": (seg_frame(df, 1.0, 1.0, 0.0), glob9("r9a_routefix_noleak")),
                "r10b_regional_interstate": (interstate_frame(df), glob9("r9b_routefix_interstate"))}
    preds, fitted = {}, {}
    for name, (d, pg) in variants.items():
        print(f"[{name}]", flush=True)
        out = M.simulate(d, mu, pg, n_sims=N_SIMS, seed=SEED)["coverage_gap_score"].to_numpy().copy()
        fitted[name] = {"global": asdict(pg)}
        for i, region in enumerate(M.CALIB_REGIONS):
            pr = specialist(d, mu, pg, region, i)
            fitted[name][region] = asdict(pr)
            m = (d["region"] == region).to_numpy()
            out[m] = M.simulate(d, mu, pr, n_sims=N_SIMS, seed=SEED)["coverage_gap_score"].to_numpy()[m]
        preds[name] = out
    preds["r10c_regional_ensemble"] = np.mean(list(preds.values()), axis=0)
    preds["r10d_regional_affine"] = 0.009 + 0.80 * preds["r10c_regional_ensemble"]

    os.makedirs(OUT, exist_ok=True)
    best = pd.read_csv(os.path.join("submissions", "best", "stack6_r9.csv"), dtype={"GEOID": str})
    assert (best["GEOID"].to_numpy() == geoid).all()
    for name, v in preds.items():
        sub = pd.DataFrame({"GEOID": geoid, "coverage_gap_score": np.clip(v, 0, 1).round(6)})
        assert sub["GEOID"].is_unique and sub.notna().all().all()
        sub.to_csv(os.path.join(OUT, f"{name}.csv"), index=False)
        r = np.sqrt(np.mean((sub.coverage_gap_score.to_numpy() - best.coverage_gap_score.to_numpy()) ** 2))
        print(f"  {name:26s} mean={sub.coverage_gap_score.mean():.4f}  RMS diff vs stack6={r:.4f}", flush=True)
    json.dump(fitted, open(os.path.join("submissions", "params_round10.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
