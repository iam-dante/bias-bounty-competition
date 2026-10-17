"""How much of the public-to-private loss comes from the 14-number blend, and would a smaller or shrunk blend
transfer better? Truth = the real blend + noise; refit on random 30 % public sets, score on the 70 %.
    python research/blend_optimism.py [gauss|t4]"""
import sys, numpy as np
sys.path.insert(0, ".")
import solution as S
df = S.load(); lb = S.Leaderboard(df); lb.public_subset(); g = lb.graded
B = np.column_stack([np.ones(g.sum())] + [lb.vec(f)[g] for f in S.BASIS])
_, w_real, _ = lb.blend(S.BASIS)
w_real = np.array(list(w_real.values()))
truth_mean = B @ w_real
n = len(truth_mean); sig = 0.0625
rng = np.random.default_rng(3)
noise = (sys.argv[1:] or ["gauss"])[0]
# a 5-file basis: the single physics models without the stacks and anchored blends
small = [0] + [1 + S.BASIS.index(f) for f in ["r2f_ensemble.csv", "r6e_vuln_affine.csv", "r9d_routefix_affine.csv",
                                              "r10d_regional_affine.csv", "r11_B_blend_anchored.csv"]]
def fit(X, y, ridge=0.0, prior=None):
    G = X.T @ X / len(y); c = X.T @ y / len(y)
    p = np.zeros(X.shape[1]) if prior is None else prior
    lam = ridge * np.trace(G) / X.shape[1]
    return np.linalg.solve(G + lam * np.eye(X.shape[1]), c + lam * p)
pol = {"14-number blend (used)": lambda Xm, ym: (slice(None), fit(Xm, ym, 1e-7)),
       "5-file blend": lambda Xm, ym: (small, fit(Xm[:, small], ym, 1e-7)),
       "14, ridge to equal-weight blend 0.01": lambda Xm, ym: (slice(None), fit(Xm, ym, 0.01, np.r_[0, np.full(Xm.shape[1] - 1, 1 / (Xm.shape[1] - 1))])),
       "14, ridge to equal-weight blend 0.1": lambda Xm, ym: (slice(None), fit(Xm, ym, 0.1, np.r_[0, np.full(Xm.shape[1] - 1, 1 / (Xm.shape[1] - 1))]))}
res = {k: [] for k in pol}
for sim in range(200):
    eta = sig * (rng.standard_t(4, n) / np.sqrt(2) if noise == "t4" else rng.standard_normal(n))
    y = truth_mean + eta
    m = rng.random(n) < 0.3
    for k, f in pol.items():
        cols, w = f(B[m], y[m])
        pred = B[:, cols] @ w
        res[k].append((np.sqrt(((pred - y)[m] ** 2).mean()), np.sqrt(((pred - y)[~m] ** 2).mean()), np.sqrt((eta[~m] ** 2).mean())))
print(f"noise {noise}, 200 splits")
base = np.array(res["14-number blend (used)"])
print(f"{'blend':40s} {'public':>9s} {'private':>9s} {'gap':>9s} {'private vs oracle':>18s} {'paired vs used':>16s}")
for k in pol:
    a = np.array(res[k]); d = a[:, 1] - base[:, 1]
    print(f"{k:40s} {a[:,0].mean():9.6f} {a[:,1].mean():9.6f} {(a[:,1]-a[:,0]).mean():+9.6f} {(a[:,1]-a[:,2]).mean():+18.6f} {d.mean():+10.6f}±{d.std()/np.sqrt(len(d)):.6f}")
