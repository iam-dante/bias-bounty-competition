"""E13: does re-blending earlier stack files (stack7's recipe) overfit the public subset?
Replays the history in the synthetic world: after each new single model a stack is fitted on the
files scored so far, clipped and scored on P; the final blend uses singles + every stack."""
import numpy as np, pandas as pd, synth as S, harness as H
sd = S.World().E.std(0)
hist = [["__const__", "r2f_ensemble.csv", "e02_r2f_only_eastern-ok.csv", "e05_r2f_only_south-central-tx.csv"],
        ["r3j_sigma40_affine.csv"], ["r5e_groups_affine.csv"], ["r6e_vuln_affine.csv"], ["r9d_routefix_affine.csv"], ["r10d_regional_affine.csv"]]
res = []
for kappa in (0.5, 1.0, 2.0):
    W = S.World(kappa=kappa, seed=300 + int(kappa * 10))
    base_files = dict(W.files)
    for rep in range(40):
        y, P, Q, sc, pub = W.replicate()
        W.files = dict(base_files)
        avail = []; stacks = {}
        for step in hist:
            avail += step
            Bk, _ = W.fit_blend(sc, names=avail + list(stacks), ridge=1e-3)
            nm = f"stack{len(stacks)}"
            stacks[nm] = Bk; W.files[nm] = Bk; sc[nm] = np.sqrt(np.mean((Bk[P] - y[P]) ** 2))
        last = stacks[nm]
        Ball, _ = W.fit_blend(sc, ridge=1e-3)                  # singles + all stacks (stack7 recipe)
        Bs, _ = W.fit_blend(sc, names=list(base_files))         # singles only
        for k, v in {"last_stack(stack7-like)": last, "refit_all": Ball, "singles_only": Bs,
                     "singles_only+sd_anchor": S.anchor_sd(Bs, W.cells, pub, W.graded, sd),
                     "last_stack+sd_anchor": S.anchor_sd(last, W.cells, pub, W.graded, sd)}.items():
            res.append({"kappa": kappa, "proc": k, "public": S.rmse(v, y, P), "private": S.rmse(v, y, Q),
                        "d_private": S.rmse(v, y, Q) - S.rmse(Bs, y, Q)})
    W.files = base_files
T = pd.DataFrame(res).groupby(["kappa", "proc"]).agg(public=("public", "mean"), private=("private", "mean"), d_private_vs_singles=("d_private", "mean")).reset_index()
print(T.round(6).to_string(index=False))
k1 = T[T.kappa == 1.0].set_index("proc")
H.log("E13", "synthetic: stack-of-stacks (stack7 recipe) vs singles-only blend", "blend",
      {"synth_private_rmse": k1.loc["last_stack(stack7-like)", "private"], "synth_private_rmse_singles": k1.loc["singles_only", "private"]})
