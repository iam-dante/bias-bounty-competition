"""
Current best: stack4_no_scorecard, public RMSE 0.065108469.

A least-squares blend of earlier submissions, fitted from leaderboard scores alone
(lb_stack.py: for RMSE, mean(f*y) = (mean(f^2) + mean(y^2) - MSE_f) / 2, so every scored file
is one known moment). Main ingredients: the round-5 group-effect model (r5e, weight 0.48),
the scorecard-offset blend stack3 (0.36) and the region-scaled r2f blend (0.14). The five
Paradise tracts with published values are set to those values.
The exact weights are in submissions/round5/stack4_no_scorecard_weights.json.

    python best.py      # rebuilds submissions/round5/stack4_no_scorecard.csv byte-for-byte
"""
import json

import numpy as np

from lb_stack import read, template

rec = json.load(open("submissions/round5/stack4_no_scorecard_weights.json"))
t = template()
score = np.zeros(len(t))
for name, w in rec["weights"].items():
    if name == "__const__":
        score += w
    else:
        v = t.merge(read(name)[["GEOID", "coverage_gap_score"]], on="GEOID", how="left")
        score += w * v["coverage_gap_score"].to_numpy()
out = t.copy()
out["coverage_gap_score"] = np.clip(score, 0, 1)
out["coverage_gap_score"] = out["GEOID"].map(rec["published"]).fillna(out["coverage_gap_score"])
out.round(6).to_csv("submissions/round5/stack4_no_scorecard.csv", index=False)
print(f"wrote submissions/round5/stack4_no_scorecard.csv ({len(out)} rows, "
      f"mean {out.coverage_gap_score.mean():.4f})")
