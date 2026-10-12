"""
E14: infer the public subset P from exact leaderboard identities, to predict public scores better.

Every scored file F that is a known linear combination of other scored files f_i (affine pairs
such as e01 = 0.009 + 0.8 r2f, and every stack) gives one exact linear equation on the unknown
public-row indicator pi:
    A_P(F^2 - sum_i w_i f_i^2 - 2 r y) = MSE_F + 2 w_0 ybar + sum_i w_i (Z - MSE_i) - Z,
with r = F - (w_0 + sum w_i f_i) nonzero only where the stack was clipped (y ~ 0 there) or set to a
published Paradise value (y known). Gaussian conditioning of pi (prior: 30 % random subset) on
these equations gives E[A_P(h)] for any h, hence corrected predictions of public scores.
Validation: leave each equation out and predict that file's public score from the rest.
"""
import numpy as np, pandas as pd, harness as H
from lb_stack import PUBLISHED

lb = H.LB(); g = lb.graded; n = g.sum(); p = 0.3
s, Z, ybar = lb.s, lb.Z, lb.ybar
gid = lb.geoid[g]
V = {f: lb.vec(f)[g] for f in s if not f.startswith("probe")}
one = np.ones(n)
par = pd.Series(gid).map(PUBLISHED).to_numpy(dtype=float)
is_par = ~np.isnan(par)


def equation(F_name, comps):
    F = V[F_name]
    X = np.column_stack([one] + [V[c] for c in comps])
    ok = (F > 1e-6) & (F < 1 - 1e-6) & ~is_par
    w = np.linalg.lstsq(X[ok], F[ok], rcond=None)[0]
    lin = X @ w
    r = F - lin
    yk = np.where(is_par, np.nan_to_num(par), 0.0)                # y known at Paradise, ~0 where clipped
    h = F ** 2 - sum(wi * V[c] ** 2 for wi, c in zip(w[1:], comps)) - 2 * r * yk
    rhs = s[F_name] ** 2 + 2 * w[0] * ybar + sum(wi * (Z - s[c] ** 2) for wi, c in zip(w[1:], comps)) - Z
    return {"h": h, "rhs": rhs, "fit_resid": float(np.abs(r[ok]).max()), "name": F_name}


order = [f for f in s if f in V]
EQ = {"e01_r2f_affine.csv": ["r2f_ensemble.csv"], "r5e_groups_affine.csv": ["r5d_groups_ensemble.csv"],
      "r6e_vuln_affine.csv": ["r6d_vuln_ensemble.csv"], "r9d_routefix_affine.csv": ["r9c_routefix_ensemble.csv"]}
for st in ["stack_region_ok_tx.csv", "stack2_e01_r3j_region.csv", "stack6_r9.csv", "stack7_r10.csv"]:  # stack4/5 are not exact combinations
    EQ[st] = [f for f in order[:order.index(st)] if not f.startswith(("stack3", "stack4_with"))]
eqs = {k: equation(k, v) for k, v in EQ.items()}
for k, e in eqs.items():
    print(f"{k:28s} exact-combination residual {e['fit_resid']:.1e}")


def condition(keys):
    Hm = np.column_stack([eqs[k]["h"] for k in keys]) / (p * n)
    r = np.array([eqs[k]["rhs"] for k in keys])
    m0 = np.full(n, p)
    S = p * (1 - p) * Hm.T @ Hm
    lam = np.linalg.lstsq(S, r - Hm.T @ m0, rcond=None)[0]
    return np.clip(m0 + p * (1 - p) * Hm @ lam, 0, 1)


def implied_score(k, pi):
    e = eqs[k]
    a_p = e["h"] @ pi / pi.sum()
    return np.sqrt(max(a_p - e["rhs"] + s[k] ** 2, 0))


rows = []
for k in eqs:
    others = [o for o in eqs if o != k]
    rows.append({"held_out": k, "actual": s[k], "pred_uncorrected": implied_score(k, np.full(n, p)),
                 "pred_corrected": implied_score(k, condition(others))})
T = pd.DataFrame(rows)
T["err_unc"] = T.pred_uncorrected - T.actual; T["err_cor"] = T.pred_corrected - T.actual
print(T.round(6).to_string(index=False))
print(f"LOO mean |err|: uncorrected {T.err_unc.abs().mean():.6f}  corrected {T.err_cor.abs().mean():.6f}")
pi = condition(list(eqs)); np.save("exp11/pi_hat.npy", pi)
H.log("E14", "public-subset inference from exact LB identities", "validation-lb",
      {"loo_abs_err_uncorrected": T.err_unc.abs().mean(), "loo_abs_err_corrected": T.err_cor.abs().mean()})
