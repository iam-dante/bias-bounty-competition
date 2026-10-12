"""E09-E11: shape of the cell-mean anchoring (additive / variance-weighted / multiplicative)."""
import numpy as np, pandas as pd, synth as S, harness as H

def anchor_w(f, cells, targets, graded, wt):
    g = f.copy()
    for k, mu in targets.items():
        m = (cells == k) & graded; mm = cells == k
        g[mm] += (mu - f[m].mean()) * wt[mm] / wt[m].mean()
    return np.clip(g, 0, 1)

def anchor_mult(f, cells, targets, graded):
    g = f.copy()
    for k, mu in targets.items():
        m = (cells == k) & graded
        g[cells == k] *= mu / max(f[m].mean(), 1e-6)
    return np.clip(g, 0, 1)

W0 = S.World(kappa=1.0)
sd = W0.E.std(0)
res = []
for kappa in (0.5, 1.0, 2.0):
    W = S.World(kappa=kappa, seed=100 + int(kappa * 10))
    for rep in range(60):
        y, P, Q, sc, pub = W.replicate()
        B, _ = W.fit_blend(sc)
        cand = {"additive": S.anchor(B, W.cells, pub, W.graded),
                "var_weighted": anchor_w(B, W.cells, pub, W.graded, sd ** 2 + 1e-4),
                "sd_weighted": anchor_w(B, W.cells, pub, W.graded, sd + 1e-3),
                "half_var_half_add": anchor_w(B, W.cells, pub, W.graded, 0.5 * (sd ** 2 + 1e-4) / np.mean(sd ** 2) + 0.5),
                "multiplicative": anchor_mult(B, W.cells, pub, W.graded)}
        base = S.rmse(B, y, Q)
        for k, v in cand.items():
            res.append({"kappa": kappa, "shape": k, "private_vs_blend": S.rmse(v, y, Q) - base, "public_vs_blend": S.rmse(v, y, P) - S.rmse(B, y, P)})
T = pd.DataFrame(res).groupby(["kappa", "shape"]).agg(private=("private_vs_blend", "mean"), public=("public_vs_blend", "mean"),
                                                        win=("private_vs_blend", lambda x: (x < 0).mean())).reset_index()
print(T.round(6).to_string(index=False))
k1 = T[T.kappa == 1.0].set_index("shape")
for eid, sh in [("E09", "var_weighted"), ("E10", "multiplicative"), ("E11", "half_var_half_add")]:
    H.log(eid, f"synthetic: anchoring shape {sh}", "post-processing", {"synth_private_gain": -k1.loc[sh, "private"], "synth_private_gain_additive": -k1.loc["additive", "private"]})
