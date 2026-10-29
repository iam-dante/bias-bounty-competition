import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import train as TR

L = TR.AggregateLabels()
g = L.graded
w = L.pi / L.pi.sum()
names = [n for n in L.s if not n.startswith("probe_")]
V = np.column_stack([L.V[n][g] for n in names])
ones = np.ones((g.sum(), 1))
X = np.column_stack([ones, V]) * np.sqrt(w)[:, None]
s = np.linalg.svd(X, compute_uv=False)
print(f"scored files: {len(names)}, graded rows: {int(g.sum())}")
print(f"span rank of scored files: {int((s > s[0] * max(X.shape) * 1e-12).sum())}")
near = np.cumsum(s ** 2) / (s ** 2).sum()
for k in (10, 40, 60, 80, 86):
    print(f"top {k:2d} directions carry {near[k - 1]:.6f} of the span's weighted energy")
print(f"singular spectrum tail: {s[-4:].round(6)}")

# exact information carried by the aggregate scores: 75 linear functionals of y + mean + mean-square
best = min(names, key=L.s.get)
segments = {"north": "tr_nofp_030.csv"}
print(f"\nbest measured {L.s[best]:.9f}; span dimension {X.shape[1]} against {int(g.sum())} unknown target rows ->")
print(f"the public set still has ~{int(g.sum()) - X.shape[1]} unconstrained directions the scores cannot see;")
print(f"only a feature model (the physics prior) can choose among them.")
