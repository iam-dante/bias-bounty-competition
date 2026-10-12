"""E05-E08: blend / post-processing procedures under semi-synthetic public/private validation."""
import numpy as np, pandas as pd, synth as S, harness as H

res = []
for kappa in (0.5, 1.0, 2.0):
    W = S.World(kappa=kappa, seed=int(kappa * 10))
    r10d = W.files["r10d_regional_affine.csv"]
    for rep in range(60):
        y, P, Q, sc, pub = W.replicate()
        row = {"kappa": kappa}
        B, w = W.fit_blend(sc)
        row["blend"] = (S.rmse(B, y, P), S.rmse(B, y, Q))
        for rg in (1e-4, 1e-3, 1e-2):
            Br, _ = W.fit_blend(sc, ridge=rg); row[f"blend_ridge{rg:g}"] = (S.rmse(Br, y, P), S.rmse(Br, y, Q))
        Bo, _ = W.fit_blend(sc, P=P); row["blend_oracleP"] = (S.rmse(Bo, y, P), S.rmse(Bo, y, Q))
        for st in (0.5, 1.0):
            A = S.anchor(B, W.cells, pub, W.graded, st); row[f"blend_anchor{st}"] = (S.rmse(A, y, P), S.rmse(A, y, Q))
        row["r10d"] = (S.rmse(r10d, y, P), S.rmse(r10d, y, Q))
        A = S.anchor(r10d, W.cells, pub, W.graded); row["r10d_anchor"] = (S.rmse(A, y, P), S.rmse(A, y, Q))
        row["truth_T"] = (S.rmse(W.T, y, P), S.rmse(W.T, y, Q))
        res.append(row)
out = []
for kappa in (0.5, 1.0, 2.0):
    rows = [r for r in res if r["kappa"] == kappa]
    for k in rows[0]:
        if k == "kappa": continue
        pub_ = np.mean([r[k][0] for r in rows]); prv = np.mean([r[k][1] for r in rows])
        d = np.array([r[k][1] - r["blend"][1] for r in rows])
        out.append({"kappa": kappa, "procedure": k, "public": pub_, "private": prv, "private_vs_blend": d.mean(), "se": d.std() / np.sqrt(len(d)), "win_rate": (d < 0).mean()})
T = pd.DataFrame(out); print(T.round(6).to_string(index=False)); T.to_csv("exp11/e05_synth_blend.csv", index=False)
k1 = T[T.kappa == 1.0].set_index("procedure")
H.log("E05", "synthetic: LB blend (baseline procedure)", "validation-synthetic", {"synth_private_rmse": k1.loc["blend", "private"], "synth_public_rmse": k1.loc["blend", "public"]})
H.log("E06", "synthetic: blend ridge 1e-2", "blend", {"synth_private_rmse": k1.loc["blend_ridge0.01", "private"], "synth_public_rmse": k1.loc["blend_ridge0.01", "public"]})
H.log("E07", "synthetic: blend + cell anchoring", "post-processing", {"synth_private_rmse": k1.loc["blend_anchor1.0", "private"], "synth_public_rmse": k1.loc["blend_anchor1.0", "public"]})
H.log("E08", "synthetic: blend fitted with exact public-subset moments (oracle)", "validation-synthetic", {"synth_private_rmse": k1.loc["blend_oracleP", "private"], "synth_public_rmse": k1.loc["blend_oracleP", "public"]})
