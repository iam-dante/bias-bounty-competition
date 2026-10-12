"""
Final assembly (round 11).

  1. blend: leaderboard-moment least squares on const + 8 single models + stack3 + stack4 (both),
     with public-subset-corrected moments (E14); stack5 and the exact-combination stacks are left
     out (collinear, E13)
  2. new physics: each unscored round-11 model enters through its novel part (orthogonal to the
     basis) with weight rho_prior * sigma_blend / ||novel||, the RMSE-optimal weight when its
     correlation with the target is the historical rho (E01: every past model had rho > 0)
  3. sd-weighted anchoring to the 16 published region x rural x burned means (E09-E11: wins 100 %
     of synthetic public/private replicates)
  4. the five published Paradise tracts
"""
import numpy as np
import pandas as pd
import harness as H
import synth as S

M = H.M
BASIS = H.BASIS + ["stack3_scorecard_groups.csv", "stack4_with_scorecard.csv", "stack4_no_scorecard.csv"]


DENOISED = {"r10d_regional_affine.csv": "submissions/round11/e26_r10d_hiprec.csv",
            "r9d_routefix_affine.csv": "submissions/round11/e26_r9d_hiprec.csv"}


def build(new_models: dict | None = None, rho=0.06, anchor=True, ridge=1e-7, basis=BASIS, denoise=False):
    lb = H.LBc(basis)
    D = H.data(); df = D["df"]
    w, mse = lb.blend(ridge)
    files = {n: (lb.vec(n) if n != "__const__" else np.ones(len(df))) for n in w}
    B = sum(w[n] * files[n] for n in w)
    if denoise:   # E26: same weights, MC noise of the 512-draw inputs removed (independent of y)
        for n, path in DENOISED.items():
            B = B + w[n] * (pd.read_csv(path)["coverage_gap_score"].to_numpy() - files[n])
    info = {"blend_expected_public": np.sqrt(mse)}
    if new_models:
        names, A, G, c = lb._sys(ridge)
        Afull = np.column_stack([files[n] for n in names])
        novels = []
        for nm, v in new_models.items():
            gv = v[lb.graded]
            a = np.linalg.solve(G, (A * lb.pi[:, None]).T @ gv / lb.pi.sum())
            novels.append(v - Afull @ a)
        d = np.mean(novels, axis=0)
        nd = np.sqrt(lb.AP(d[lb.graded] ** 2))
        a = rho * np.sqrt(mse) / nd
        B = B + a * d
        info.update(novel_norm=nd, novel_weight=a, expected_gain_mse=(rho * np.sqrt(mse)) ** 2)
    if anchor:
        sd = S.eps_draws().std(0)
        B = S.anchor_sd(B, S.cell_index(df), S.published_targets(df), lb.graded, sd)
    return np.clip(B, 0, 1), info, lb


def report(name, v, lb):
    D = H.data(); df = D["df"]
    cells = S.cell_index(df); tg = S.published_targets(df)
    err = [v[(cells == k) & lb.graded].mean() - mu for k, mu in tg.items()]
    met = lb.validate(v); met["cell_rms_err"] = float(np.sqrt(np.mean(np.square(err))))
    return met
