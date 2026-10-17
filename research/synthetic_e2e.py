"""End-to-end test on a planted truth: generate every leaderboard score from a known y, run the real
correction and validation on those scores, and compare predicted vs true public and private RMSE.
    python research/synthetic_e2e.py [replicates]"""
import functools, io, contextlib, sys, numpy as np, pandas as pd
sys.path.insert(0, ".")
import solution as S

df = S.load(); mu = S.prior_rates(df)
real = S.Leaderboard(df)
real.vec = functools.lru_cache(None)(real.vec)
g = real.graded
files = list(real.s)
V = {f: real.vec(f) for f in files if not f.startswith("probe_")}
r19 = pd.read_csv("submission.csv")["coverage_gap_score"].to_numpy()
feat = S.probe_features(df, mu, ["ems_evidence"])["ems_evidence"].to_numpy()   # an unprobed direction
reps = int(sys.argv[1]) if len(sys.argv) > 1 else 20
rng = np.random.default_rng(7)
rows = []
for rep in range(reps):
    n = g.sum()
    m = np.zeros(len(df), bool); m[np.flatnonzero(g)[rng.random(n) < 0.3]] = True     # true public rows
    tau = [0.0, 0.004][rep % 2]                                                       # hidden signal off / on
    y = np.clip(r19 + tau * feat + 0.06 * rng.standard_normal(len(df)) * (r19 > 0.01), 0, 1)
    rmse = lambda v, mask: float(np.sqrt(((v - y)[mask] ** 2).mean()))
    lb = S.Leaderboard(df); lb.vec = real.vec
    lb.s = {f: (rmse(V[f], m) if f in V else None) for f in files}
    lb.s["probe_zeros.csv"] = rmse(np.zeros(len(df)), m); lb.s["probe_const_0.10.csv"] = rmse(np.full(len(df), 0.1), m)
    lb.Z = lb.s["probe_zeros.csv"] ** 2
    lb.ybar = (0.01 + lb.Z - lb.s["probe_const_0.10.csv"] ** 2) / 0.2
    lb.pi = np.full(n, 0.3); lb.public_subset()
    r12 = lb.vec(S.BASE)
    with contextlib.redirect_stdout(io.StringIO()):
        corr, note, gain = S.probe_correction(lb, df, mu, r12)
        val = S.validate(lb, out="/dev/null")
    out = S.finish(r12 + corr, df["GEOID"])
    pred = np.sqrt(lb.s[S.BASE] ** 2 - gain)
    priv = g & ~m
    e = val[val.file.str.match(r"^(r1[2-9]|p1[2-9])_")]
    rows.append(dict(rep=rep, hidden=tau, predicted_public=pred, true_public=rmse(out, m), true_private=rmse(out, priv),
                     r12_private=rmse(r12, priv), oracle_private=rmse(np.clip(r19 + tau * feat, 0, 1), priv),
                     backtest_mae=e.err_before.abs().mean(), backtest_max=e.err_before.abs().max()))
    print(f"rep {rep:2d} hidden {tau}: predicted public {pred:.6f} true public {rows[-1]['true_public']:.6f} "
          f"true private {rows[-1]['true_private']:.6f} (r12 {rows[-1]['r12_private']:.6f}) backtest max {rows[-1]['backtest_max']:.6f}", flush=True)
t = pd.DataFrame(rows); t.round(7).to_csv("research/synthetic_e2e.csv", index=False)
t["err_public"] = t.predicted_public - t.true_public
t["gap"] = t.true_private - t.true_public
print(f"\npredicted vs true public: MAE {t.err_public.abs().mean():.6f}, max {t.err_public.abs().max():.6f}")
print(f"true private - true public: mean {t.gap.mean():+.6f}, sd {t.gap.std():.6f}")
print(f"correction effect on private (r12 -> corrected): mean {(t.true_private - t.r12_private).mean():+.6f}")
print(f"step-6 backtest on synthetic scores: mean MAE {t.backtest_mae.mean():.6f}, worst max {t.backtest_max.max():.6f}")
