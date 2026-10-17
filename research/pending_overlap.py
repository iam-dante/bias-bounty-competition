"""Overlap between the pending probes in next/: each probe's push (probe minus the base it was built on),
with the scored files removed, correlated pairwise. Two highly correlated probes measure the same thing.
    python research/pending_overlap.py"""
import glob, os, sys, numpy as np, pandas as pd
sys.path.insert(0, ".")
import solution as S
df = S.load(); lb = S.Leaderboard(df); lb.public_subset(); g = lb.graded; w = np.sqrt(lb.pi / lb.pi.sum())
files = [f for f in lb.s if not f.startswith("probe_")]
Q, _ = np.linalg.qr(np.column_stack([np.ones(g.sum())] + [lb.vec(f)[g] for f in files]) * w[:, None])
P = {}
for f in sorted(glob.glob("next/p*.csv")):
    v = pd.read_csv(f)["coverage_gap_score"].to_numpy()[g] * w
    r = v - Q @ (Q.T @ v)            # the probe minus everything the scored files span = its new push
    P[os.path.basename(f)[:-4]] = r / np.linalg.norm(r)
names = list(P); M = np.array([P[n] for n in names])
C = M @ M.T
out = pd.DataFrame(C, index=[n[:22] for n in names], columns=[n[4:7] for n in names]).round(2)
out.to_csv("research/pending_overlap.csv")
print(out.to_string())
print("\npairs with |corr| > 0.5:")
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        if abs(C[i, j]) > 0.5: print(f"  {names[i]}  ~  {names[j]}: {C[i, j]:+.2f}")
