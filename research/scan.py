"""Candidate scan: unexplained share of every column after the scored files.  python research/scan.py"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, ".")
import solution as S
df = S.load(); lb = S.Leaderboard(df); lb.public_subset(); g = lb.graded; pi = lb.pi; W = pi.sum()
files = [f for f in lb.s if not f.startswith("probe_")]
A = np.column_stack([np.ones(g.sum())] + [lb.vec(f)[g] for f in files])
Q, _ = np.linalg.qr(A * np.sqrt(pi / W)[:, None])
num = df.select_dtypes("number").drop(columns=["ridx"], errors="ignore")
num = num.loc[:, num.notna().mean() > 0.8]
cand = {}
for c in num.columns:
    x = num[c].astype(float)
    if x.nunique() < 3: continue
    x = np.sign(x) * np.log1p(np.abs(x)) if (x.abs().max() > 50 and x.skew() > 2) else x
    x = x.fillna(x.median()).groupby(df["region"]).transform(lambda v: (v - v.mean()) / (v.std() + 1e-12)).fillna(0)
    cand[c] = x.to_numpy()[g]
X = np.column_stack(list(cand.values())) * np.sqrt(pi / W)[:, None]
R = X - Q @ (Q.T @ X)
share = (R ** 2).sum(0) / (X ** 2).sum(0)
names = list(cand)
tab = pd.DataFrame({"column": names, "unexplained_share": share}).sort_values("unexplained_share", ascending=False)
tab.round(4).to_csv("research/candidate_scan.csv", index=False)
print(f"{len(names)} candidate columns; unexplained share quantiles:", np.quantile(share, [0.1, 0.5, 0.9]).round(3))
# greedy diverse selection: largest residual after removing scored files AND already-picked candidates
picked, Rw = [], R / np.sqrt((X ** 2).sum(0))
for _ in range(12):
    norms = (Rw ** 2).sum(0)
    j = int(np.argmax(norms)); picked.append((names[j], norms[j]))
    q = Rw[:, j] / np.sqrt(norms[j]); Rw = Rw - np.outer(q, q @ Rw)
print("greedy diverse pick (share of new variance left at pick time):")
for n_, v in picked: print(f"  {n_:45s} {v:.3f}")
