"""Uncertainty of the expected public RMSE from not knowing which tracts are public: redraw the public set
from the estimated membership probabilities, refit the correction, recompute the expected score.
    python research/expected_uncertainty.py"""
import functools, io, contextlib, sys, numpy as np, pandas as pd
sys.path.insert(0, ".")
import solution as S
df = S.load(); mu = S.prior_rates(df)
lb = S.Leaderboard(df); lb.public_subset(); lb.vec = functools.lru_cache(None)(lb.vec)
pi0 = lb.pi.copy(); r12 = lb.vec(S.BASE)
rng = np.random.default_rng(11); out = []
for rep in range(30):
    lb.pi = (rng.random(len(pi0)) < pi0).astype(float)
    with contextlib.redirect_stdout(io.StringIO()):
        corr, note, gain = S.probe_correction(lb, df, mu, r12, order="backward")
    out.append(np.sqrt(lb.s[S.BASE] ** 2 - gain))
lb.pi = pi0
o = np.array(out)
print(f"expected public RMSE of r19b over 30 redrawn public sets: mean {o.mean():.6f}, sd {o.std():.6f}, range {o.min():.6f}-{o.max():.6f}")
pd.DataFrame({"expected_public": o}).round(7).to_csv("research/expected_uncertainty.csv", index=False)
