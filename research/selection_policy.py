"""Selection order for the correction: scoring order (used) vs strongest-first vs backward elimination,
on semi-synthetic 30/70 splits.  python research/selection_policy.py [gauss|t4]"""
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
    if policy == "scoring order (used)":
        sel = []
        for j in range(K):
            if chi(sel + [j]) - chi(sel) >= 4.0: sel.append(j)
    elif policy == "strongest first":
        sel, rest = [], list(range(K))
        while rest:
            gains = [chi(sel + [j]) - chi(sel) for j in rest]
            j = int(np.argmax(gains))
            if gains[j] < 4.0: break
            sel.append(rest.pop(j))
    elif policy == "backward elimination":
        sel = list(range(K))
        while sel:
            drops = [chi(sel) - chi([k for k in sel if k != j]) for j in sel]
            j = int(np.argmin(drops))
            if drops[j] >= 4.0: break
            sel.pop(j)
    if not sel: return np.zeros(K)
    shrink = max(0.0, 1 - len(sel) / chi(sel))
    gam = np.zeros(K); gam[sel] = shrink * np.linalg.solve(Gm[np.ix_(sel, sel)], bm[sel])
    return gam

policies = ["scoring order (used)", "strongest first", "backward elimination"]
res = {p: [] for p in policies}
for sim in range(150):
    eta = sig * (rng.standard_t(4, n) / np.sqrt(2) if (sys.argv[1:] or ["gauss"])[0] == "t4" else rng.standard_normal(n))          # heavy-tailed residual, var = sig^2
    e = P @ theta + eta                                    # y - r12
    m = rng.random(n) < 0.3
    for p in policies:
        gam = fit(P[m], e[m], p)
        r = e - P @ gam
        res[p].append((np.sqrt((r[m] ** 2).mean()), np.sqrt((r[~m] ** 2).mean()), np.sqrt((e[~m] ** 2).mean()),
                       np.sqrt(((eta[~m]) ** 2).mean())))
print(f"{K} probe directions, 150 simulated 30/70 splits")
print(f"{'policy':30s} {'public':>9s} {'private':>9s} {'gap':>9s} {'private gain':>13s} {'vs oracle':>10s}")
for p in policies:
    a = np.array(res[p])
    print(f"{p:30s} {a[:,0].mean():9.6f} {a[:,1].mean():9.6f} {(a[:,1]-a[:,0]).mean():+9.6f} {(a[:,2]-a[:,1]).mean():13.6f} {(a[:,1]-a[:,3]).mean():+10.6f}")
print("split-to-split sd of private RMSE: %.6f" % np.array(res[policies[0]])[:, 1].std())
base = np.array(res[policies[0]])[:, 1]
for p in policies[1:]:
    dd = np.array(res[p])[:, 1] - base
    print(f"paired private difference {p} - {policies[0]}: {dd.mean():+.6f} +- {dd.std() / np.sqrt(len(dd)):.6f} (se); better in {np.mean(dd < 0):.0%} of splits")
