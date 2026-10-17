"""Semi-synthetic public/private test of the probe-correction policy.  python research/semisynth.py [gauss|t4]"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, ".")
import solution as S
df = S.load(); lb = S.Leaderboard(df); lb.public_subset(); g = lb.graded; AP = lb.AP
r12 = lb.vec(S.BASE)[g]
basis = S.BASIS + [S.BASE, "r11_blend_denoised.csv"]
A, G, c = lb.system(basis)
scored = [f for f in lb.s if S.probe_direction_of(f)]
P = []
for f in scored:
    dd = lb.vec(f)[g] - r12
    a = np.linalg.solve(G, (A * lb.pi[:, None]).T @ dd / lb.pi.sum()); P.append(dd - A @ a)
P = np.column_stack(P); n, K = P.shape
# "truth" coefficients: the real-data unshrunk fit on all probes
cy = lambda v, sc: (AP(v ** 2) + lb.Z - sc ** 2) / 2
ry = cy(r12, lb.s[S.BASE])
b = np.array([(cy(lb.vec(f)[g], lb.s[f]) - ry) - np.linalg.solve(G, (A * lb.pi[:, None]).T @ (lb.vec(f)[g] - r12) / lb.pi.sum()) @ c
              - AP(P[:, k] * r12) for k, f in enumerate(scored)])
Gp = (P * lb.pi[:, None]).T @ P / lb.pi.sum()
theta = np.linalg.lstsq(Gp, b, rcond=None)[0]
sig = 0.0625
rng = np.random.default_rng(0)

def fit(Pm, e_m, policy):
    Gm = Pm.T @ Pm / len(e_m); bm = Pm.T @ e_m / len(e_m); s2 = (e_m ** 2).mean()
    chi = lambda idx: float(bm[idx] @ np.linalg.solve(Gm[np.ix_(idx, idx)], bm[idx])) / s2 * len(e_m) if idx else 0.0
    if policy == "all, no shrink":
        sel = list(range(K)); shrink = 1.0
    else:
        thr = {"z>=2 + JS shrink (current)": 4.0, "z>=3 + JS shrink": 9.0, "z>=2, no shrink": 4.0, "z>=1.5 + JS shrink": 2.25, "z>=2 + 2x shrink": 4.0}[policy]
        sel = []
        for j in range(K):
            if chi(sel + [j]) - chi(sel) >= thr: sel.append(j)
        if not sel: return np.zeros(K)
        shrink = 1.0 if "no shrink" in policy else max(0.0, 1 - (2 if "2x" in policy else 1) * len(sel) / chi(sel))
    gam = np.zeros(K)
    gam[sel] = shrink * np.linalg.solve(Gm[np.ix_(sel, sel)], bm[sel])
    return gam

policies = ["z>=2 + JS shrink (current)", "z>=2 + 2x shrink", "z>=1.5 + JS shrink", "z>=3 + JS shrink", "all, no shrink"]
res = {p: [] for p in policies}
for sim in range(300):
    eta = sig * (rng.standard_t(4, n) / np.sqrt(2) if (sys.argv[1:] or ["gauss"])[0] == "t4" else rng.standard_normal(n))          # heavy-tailed residual, var = sig^2
    e = P @ theta + eta                                    # y - r12
    m = rng.random(n) < 0.3
    for p in policies:
        gam = fit(P[m], e[m], p)
        r = e - P @ gam
        res[p].append((np.sqrt((r[m] ** 2).mean()), np.sqrt((r[~m] ** 2).mean()), np.sqrt((e[~m] ** 2).mean()),
                       np.sqrt(((eta[~m]) ** 2).mean())))
print(f"{K} probe directions, 300 simulated 30/70 splits")
print(f"{'policy':30s} {'public':>9s} {'private':>9s} {'gap':>9s} {'private gain':>13s} {'vs oracle':>10s}")
for p in policies:
    a = np.array(res[p])
    print(f"{p:30s} {a[:,0].mean():9.6f} {a[:,1].mean():9.6f} {(a[:,1]-a[:,0]).mean():+9.6f} {(a[:,2]-a[:,1]).mean():13.6f} {(a[:,1]-a[:,3]).mean():+10.6f}")
print("split-to-split sd of private RMSE: %.6f" % np.array(res[policies[0]])[:, 1].std())
