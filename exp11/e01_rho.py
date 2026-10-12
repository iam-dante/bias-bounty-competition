"""E01: how much NEW signal each past physical model carried (novel-part correlation rho with y)."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from lb_stack import read, template, region_of, UNSCORED
s = pd.read_csv("lb_scores.csv"); s = dict(zip(s.file, s.score))
Z = s["probe_zeros.csv"]**2; ybar = (0.01 + Z - s["probe_const_0.10.csv"]**2)/0.2
t = template(); reg = region_of(t.GEOID); g = ~reg.isin(UNSCORED).to_numpy()
files = [f for f in s if not f.startswith("probe")]
X = {}; c = {}
for f in files:
    v = t.merge(read(f)[["GEOID","coverage_gap_score"]], on="GEOID", how="left").coverage_gap_score.to_numpy()[g]
    X[f] = v; c[f] = (np.mean(v**2) + Z - s[f]**2)/2
X["__const__"] = np.ones(g.sum()); c["__const__"] = ybar
# only single models + region probes: these are the "information" columns; stacks are near-linear combos
singles = ["__const__","r2f_ensemble.csv","e02_r2f_only_eastern-ok.csv","e05_r2f_only_south-central-tx.csv",
           "r3j_sigma40_affine.csv","r5e_groups_affine.csv","r6e_vuln_affine.csv","r9d_routefix_affine.csv","r10d_regional_affine.csv"]
def fit(S, ridge=1e-6):
    A = np.column_stack([X[n] for n in S]); G = A.T@A/len(A); cs = np.array([c[n] for n in S])
    w = np.linalg.solve(G + ridge*np.trace(G)/len(S)*np.eye(len(S)), cs); return A, G, cs, w, w@G@w - 2*w@cs + Z
for k in range(1, len(singles)):
    S = singles[:k]; f = singles[k]
    A,G,cs,w,mse = fit(S)
    a = np.linalg.lstsq(A, X[f], rcond=None)[0]; r = X[f]-A@a
    cross = c[f]-a@cs; nr = np.sqrt(np.mean(r**2))
    rho = cross/(nr*np.sqrt(mse))
    print(f"{f:36s} score={s[f]:.6f} blend-before={np.sqrt(mse):.6f} |novel|={nr:.4f} rho={rho:+.3f}")
A,G,cs,w,mse = fit(singles); print("singles-only blend expected", np.sqrt(mse), dict(zip(singles, np.round(w,3))))
# check: consistency of exact affine relations e01 = 0.009+0.8 r2f ?
d = X["e01_r2f_affine.csv"] - (0.009+0.8*X["r2f_ensemble.csv"]); print("e01 affine resid max", np.abs(d).max())
pred_c = 0.009*ybar + 0.8*c["r2f_ensemble.csv"]; print("e01 c pred vs from score", pred_c, c["e01_r2f_affine.csv"], "-> implied MSE", np.sqrt(np.mean(X["e01_r2f_affine.csv"]**2)+Z-2*pred_c), "actual", s["e01_r2f_affine.csv"])
for a_,b_ in [("r5e_groups_affine.csv","r5d_groups_ensemble.csv"),("r6e_vuln_affine.csv","r6d_vuln_ensemble.csv"),("r9d_routefix_affine.csv","r9c_routefix_ensemble.csv")]:
    d = X[a_]-(0.009+0.8*X[b_]); pc = 0.009*ybar+0.8*c[b_]
    print(a_, "resid", np.abs(d).max().round(6), "implied", np.sqrt(np.mean(X[a_]**2)+Z-2*pc), "actual", s[a_])
