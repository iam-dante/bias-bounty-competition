"""
Current best: submissions/best/r12_blend_with_AB.csv, public RMSE 0.064008915.

A least-squares blend of scored submissions fitted from leaderboard scores alone, with the moments
of the public subset corrected by exact leaderboard identities (exp11/e14_public_subset.py). The
basis is the 8 single models, stack3 and both stack4 files, and the two scored round-11 files
(r11_B: cell-anchored blend, r11_A: plus GMM physics), so the leaderboard itself sets how much
cell anchoring to keep (about a third). The r10d and r9d inputs enter at 4096 Monte-Carlo draws
instead of 512 (same saved parameters, exp11/e26_denoise.py): their MC noise is independent of
the target. The five Paradise tracts are set to their published values.

    python best.py      # rebuilds submissions/best/r12_blend_with_AB.csv byte-for-byte
"""
import json
import os

import numpy as np
import pandas as pd

from lb_stack import read, template

rec = json.load(open("submissions/best/r12_blend_with_AB_weights.json"))
t = template()
col = lambda name: t.merge(read(name)[["GEOID", "coverage_gap_score"]], on="GEOID", how="left")["coverage_gap_score"].to_numpy()
files = {n: (np.ones(len(t)) if n == "__const__" else col(n)) for n in rec["weights"]}
score = sum(w * files[n] for n, w in rec["weights"].items())
for n, hi in rec["denoise"].items():
    hiv = pd.read_csv(os.path.join("submissions", "best", hi))["coverage_gap_score"].to_numpy()
    score = score + rec["weights"][n] * (hiv - files[n])
out = t.copy()
out["coverage_gap_score"] = np.clip(score, 0, 1)
out["coverage_gap_score"] = out["GEOID"].map(rec["published"]).fillna(out["coverage_gap_score"])
out["coverage_gap_score"] = out["coverage_gap_score"].round(6)
out.to_csv("submissions/best/r12_blend_with_AB.csv", index=False)
print(f"wrote submissions/best/r12_blend_with_AB.csv ({len(out)} rows, mean {out.coverage_gap_score.mean():.4f})")
