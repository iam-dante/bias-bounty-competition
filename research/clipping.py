"""Clipping at 0/1 after the correction: how many tracts, and its upper-bound effect on public RMSE.
y is in [0, 1], so clipping can only lower the error; the expected-score formula ignores it.
    python research/clipping.py"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, ".")
import solution as S
df = S.load(); mu = S.prior_rates(df); lb = S.Leaderboard(df); lb.public_subset(); g = lb.graded
r12 = lb.vec(S.BASE)
rows = []
for order in ["scoring", "backward"]:
    corr, note, gain = S.probe_correction(lb, df, mu, r12, order=order)
    raw = (r12 + corr)[g]
    lo, hi = raw < 0, raw > 1
    exp_noclip = np.sqrt(lb.s[S.BASE] ** 2 - gain)
    # if y >= 0, clipping a negative prediction p to 0 lowers the squared error by at least p^2 (y = 0 is the worst case,
    # y > |p| the best case 2|p|y + p^2); the public share is weighted by pi
    min_gain = lb.AP(np.where(lo, raw ** 2, 0.0))
    rows.append((order, int(lo.sum()), int(hi.sum()), float(raw[lo].min()) if lo.any() else 0.0, exp_noclip,
                 np.sqrt(exp_noclip ** 2 - min_gain)))
t = pd.DataFrame(rows, columns=["selection", "n_below_0", "n_above_1", "most_negative", "expected_no_clip", "expected_with_min_clip_gain"])
print(t.round(6).to_string(index=False))
t.round(7).to_csv("research/clipping.csv", index=False)
