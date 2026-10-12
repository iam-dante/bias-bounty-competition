"""E12: blend with cell-indicator columns (moments from published cell means), then sd-anchor."""
import numpy as np, pandas as pd, synth as S, harness as H
sd = S.World().E.std(0)
res = []
for kappa in (0.5, 1.0, 2.0):
    W = S.World(kappa=kappa, seed=200 + int(kappa * 10))
    for rep in range(60):
        y, P, Q, sc, pub = W.replicate()
        B, _ = W.fit_blend(sc)
        Bc = W and S.fit_blend_cells(W, sc, pub)
        cand = {"blend+sd_anchor": S.anchor_sd(B, W.cells, pub, W.graded, sd),
                "cellblend": Bc, "cellblend+sd_anchor": S.anchor_sd(Bc, W.cells, pub, W.graded, sd)}
        for k, v in cand.items():
            res.append({"kappa": kappa, "proc": k, "private": S.rmse(v, y, Q) - S.rmse(B, y, Q), "public": S.rmse(v, y, P) - S.rmse(B, y, P)})
T = pd.DataFrame(res).groupby(["kappa", "proc"]).agg(private=("private", "mean"), public=("public", "mean"), win=("private", lambda x: (x < 0).mean())).reset_index()
print(T.round(6).to_string(index=False))
k1 = T[T.kappa == 1.0].set_index("proc")
H.log("E12", "synthetic: blend with cell-indicator columns + sd anchor", "blend", {"synth_private_gain": -k1.loc["cellblend+sd_anchor", "private"], "synth_private_gain_blend_sd_anchor": -k1.loc["blend+sd_anchor", "private"]})
