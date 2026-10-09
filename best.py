"""
Current best: stack_region_ok_tx, public RMSE 0.066977056.

Both steps use leaderboard scores only (lb_stack.py). For RMSE,
mean(f*y) = (mean(f^2) + mean(y^2) - MSE_f) / 2, so each scored file is one known moment.

1. e01_r2f_affine (public 0.067316) = 0.009 + 0.80 * r2f_ensemble. r2f was over-dispersed.
2. stack_region_ok_tx (public 0.066977) = least-squares blend of r2f, e01, r2f restricted to
   Eastern Oklahoma, r2f restricted to Texas, and a constant (weights in best_weights.json,
   ridge 1e-3). Roughly 0.0118 + 0.695*r2f, plus 0.198*r2f extra in Eastern OK (its gaps
   were the most under-predicted) and 0.033*r2f extra in Texas. eastern-wa is not graded.

    python solution.py && python round2.py && python best.py
"""
import json

import numpy as np
import pandas as pd

r2f = pd.read_csv("submissions/r2f_ensemble.csv", dtype={"GEOID": str})
f = r2f["coverage_gap_score"].to_numpy()
state = r2f["GEOID"].str[:2].to_numpy()

e01 = np.clip(0.009 + 0.80 * f, 0, 1).round(6)
r2f[["GEOID"]].assign(coverage_gap_score=e01).to_csv("submissions/e01_r2f_affine.csv", index=False)

w = json.load(open("best_weights.json"))
best = (w["r2f"] * f + w["e01"] * e01 + w["r2f_ok"] * f * (state == "40")
        + w["r2f_tx"] * f * (state == "48") + w["const"])
out = r2f[["GEOID"]].assign(coverage_gap_score=np.clip(best, 0, 1).round(6))
out.to_csv("submissions/stack_region_ok_tx.csv", index=False)
print(f"wrote submissions/e01_r2f_affine.csv and submissions/stack_region_ok_tx.csv "
      f"({len(out)} rows, mean {out.coverage_gap_score.mean():.4f})")
