"""E20: are the published cell means consistent with the leaderboard? (guards the anchoring step)
  (a) overall mean implied by the 16 published cells vs the public mean from the probes
  (b) the in-span conflict (proj_err) the anchoring creates: real value vs the synthetic-world
      distribution, where the published means are exact by construction."""
import numpy as np, pandas as pd, harness as H, synth as S, final as F
D = H.data(); df = D["df"]
lb = H.LBc(F.BASIS); g = lb.graded
cells = S.cell_index(df); tg = S.published_targets(df)
n = {k: ((cells == k) & g).sum() for k in tg}
full_mean = sum(n[k] * tg[k] for k in tg) / sum(n.values())
se = 0.089 * np.sqrt(0.7 / (0.3 * g.sum()))
print(f"(a) published-cells full mean {full_mean:.5f} vs public mean {lb.ybar:.5f}  (z = {(lb.ybar - full_mean) / se:+.2f}, se {se:.4f})")
B0, _, _ = F.build(anchor=False); B1, _, _ = F.build(anchor=True)
real = lb.validate(B1)["proj_err"]
# synthetic: exact P-moments, blend on the same basis columns, then anchor; proj_err of the anchored blend
sd = S.eps_draws().std(0)
W = S.World(kappa=1.0, seed=7)
names = list(lb.cols)
X = np.column_stack([lb.cols[nm] if nm == "__const__" else lb.vec(nm)[g] for nm in names])
X[:, 0] = 1.0
vals = []
for rep in range(60):
    y, P, Q, sc, pub = W.replicate()
    Pg = P[g]; yg = y[g]
    G = X[Pg].T @ X[Pg] / Pg.sum(); c = X[Pg].T @ yg[Pg] / Pg.sum()
    w = np.linalg.solve(G + 1e-10 * np.eye(len(c)), c)
    Bs = np.zeros(len(df)); Bs[g] = X @ w
    Ba = S.anchor_sd(Bs, cells, pub, g, sd)
    b = c - X[Pg].T @ Ba[g][Pg] / Pg.sum()
    vals.append(np.sqrt(b @ np.linalg.solve(G, b)))
vals = np.array(vals)
print(f"(b) anchoring in-span conflict: real {real:.5f}; synthetic (published exact) mean {vals.mean():.5f}, 95th pct {np.quantile(vals, .95):.5f}, max {vals.max():.5f}")
H.log("E20", "consistency of published cells with LB", "validation-lb", {"z_mean": (lb.ybar - full_mean) / se, "proj_conflict_real": real, "proj_conflict_synth_p95": np.quantile(vals, .95)})
