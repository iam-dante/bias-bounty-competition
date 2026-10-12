"""
Round 11 experiment harness: shared data, leaderboard moments and validation metrics.

There are no labels, so "validation" is built from what the leaderboard already told us.
For RMSE on the public subset P, every scored file f_i gives one exact moment
    c_i = mean_P(f_i * y) = (mean_P(f_i^2) + mean_P(y^2) - MSE_i) / 2.
For any candidate g, the error e = y - g splits into its projection on the span of the scored
files (computable exactly) and the rest:
    MSE(g) = ||P e||^2 + ||P_perp y||^2 + ||P_perp g||^2 - 2 <P_perp y, P_perp g>
  * ||P e||^2 = b' G^-1 b with b_i = c_i - mean(f_i g)            (known)
  * ||P_perp y||^2 = MSE of the best blend of scored files          (known)
  * the cross term is the new information g carries, rho * ||P_perp y|| * ||P_perp g||.
Every past physical model had rho > 0 (0.01 to 0.65, median ~0.10; exp11/e00_rho.py), so the
validation score V(g) reports the RMSE at rho = 0 (neutral) and at the historical prior rho = 0.08.

Only the single models and region probes enter the projection basis: the stack files are
near-linear combinations of them and add collinearity, not information.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "exp11"))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.chdir(ROOT)
import model as M  # noqa: E402
from lb_stack import read, region_of, template, UNSCORED  # noqa: E402

OUT = os.path.join("submissions", "round11")
LOG = os.path.join("exp11", "log.csv")
RHO_PRIOR = 0.08
BASIS = ["r2f_ensemble.csv", "e02_r2f_only_eastern-ok.csv", "e05_r2f_only_south-central-tx.csv",
         "r3j_sigma40_affine.csv", "r5e_groups_affine.csv", "r6e_vuln_affine.csv",
         "r9d_routefix_affine.csv", "r10d_regional_affine.csv"]


class LB:
    """Leaderboard moments on the graded rows (eastern-wa is not graded)."""

    def __init__(self, basis=BASIS):
        s = pd.read_csv("lb_scores.csv")
        self.s = dict(zip(s["file"], s["score"].astype(float)))
        self.Z = self.s["probe_zeros.csv"] ** 2
        self.ybar = (0.01 + self.Z - self.s["probe_const_0.10.csv"] ** 2) / 0.2
        self.t = template()
        self.geoid = self.t["GEOID"].to_numpy()
        self.graded = ~region_of(self.t["GEOID"]).isin(UNSCORED).to_numpy()
        self.cols, self.c = {"__const__": np.ones(self.graded.sum())}, {"__const__": self.ybar}
        for f in basis:
            self.add(f)
        self.full = {}

    def vec(self, name):
        return self.t.merge(read(name)[["GEOID", "coverage_gap_score"]], on="GEOID",
                            how="left")["coverage_gap_score"].to_numpy()

    def add(self, f, score=None):
        v = self.vec(f)[self.graded] if isinstance(f, str) else f[1][self.graded]
        name = f if isinstance(f, str) else f[0]
        sc = self.s[name] if score is None else score
        self.cols[name] = v
        self.c[name] = (np.mean(v ** 2) + self.Z - sc ** 2) / 2

    def _sys(self, ridge=1e-7):
        names = list(self.cols)
        A = np.column_stack([self.cols[n] for n in names])
        G = A.T @ A / len(A)
        G = G + ridge * np.trace(G) / len(names) * np.eye(len(names))
        return names, A, G, np.array([self.c[n] for n in names])

    def blend(self, ridge=1e-7):
        names, A, G, c = self._sys(ridge)
        w = np.linalg.solve(G, c)
        return dict(zip(names, w)), float(w @ G @ w - 2 * w @ c + self.Z)

    def validate(self, g_full: np.ndarray) -> dict:
        """Validation metrics for a full-length (all 9,794 rows) candidate."""
        g = g_full[self.graded]
        names, A, G, c = self._sys()
        Gi = np.linalg.inv(G)
        b = c - A.T @ g / len(A)
        pe2 = float(b @ Gi @ b)                        # ||P e||^2, exact
        _, mse_b = self.blend()
        a = Gi @ (A.T @ g / len(A))
        perp = g - A @ a
        n2 = float(np.mean(perp ** 2))                 # ||P_perp g||^2
        sb = np.sqrt(max(mse_b, 0))
        v0 = pe2 + mse_b + n2
        vp = v0 - 2 * RHO_PRIOR * sb * np.sqrt(n2)
        return {"lb_proj_rmse": np.sqrt(v0), "lb_prior_rmse": np.sqrt(max(vp, 0)), "proj_err": np.sqrt(pe2),
                "novel": np.sqrt(n2), "mean": float(g.mean())}


_DATA = {}


def data():
    """df (with facility points), mu, and the two road frames used since round 5."""
    if not _DATA:
        from round2 import seg_frame
        from round3 import interstate_frame
        for k in ["m0", "m1", "m2", "m3"]:
            M.BOUNDS[k] = (0.5, 3.0)
        df = M.attach_points(M.load())
        mu = M.prior_rates(df)
        _DATA.update(df=df, mu=mu, noleak=seg_frame(df, 1.0, 1.0, 0.0), inter=interstate_frame(df))
    return _DATA


def params(path, key):
    P = json.load(open(path))[key]
    return M.Params(**{**P, "m": tuple(P["m"])})


def regional(path, key):
    """Round-10 specialists: {region: Params}."""
    P = json.load(open(path))[key]
    return {r: M.Params(**{**P[r], "m": tuple(P[r]["m"])}) for r in P}


def sim_regional(d, mu, pr: dict, n_sims=512, seed=2026, comps=False):
    out = M.simulate(d, mu, pr["global"], n_sims=n_sims, seed=seed, return_components=comps).copy()
    for r in M.CALIB_REGIONS:
        m = (d["region"] == r).to_numpy()
        o = M.simulate(d, mu, pr[r], n_sims=n_sims, seed=seed, return_components=comps)
        out.loc[m] = o.loc[m].to_numpy()
    return out


def moment_loss(d, pred) -> float:
    return float(np.sum(M.moment_residuals(d, pred) ** 2))


def log(eid, name, category, metrics: dict, note=""):
    row = {"id": eid, "name": name, "category": category, **{k: round(float(v), 7) for k, v in metrics.items()},
           "note": note, "time": time.strftime("%H:%M:%S")}
    old = pd.read_csv(LOG) if os.path.exists(LOG) else pd.DataFrame()
    if len(old):
        old = old[old["id"] != eid]
    pd.concat([old, pd.DataFrame([row])], ignore_index=True).to_csv(LOG, index=False)
    print("LOG", json.dumps(row), flush=True)


def write(geoid, v, name, published=True):
    from lb_stack import PUBLISHED
    os.makedirs(OUT, exist_ok=True)
    out = pd.DataFrame({"GEOID": geoid, "coverage_gap_score": np.clip(v, 0, 1)})
    if published:
        out["coverage_gap_score"] = out["GEOID"].map(PUBLISHED).fillna(out["coverage_gap_score"])
    out["coverage_gap_score"] = out["coverage_gap_score"].round(6)
    assert out["GEOID"].is_unique and out.notna().all().all() and len(out) == 9794
    p = os.path.join(OUT, f"{name}.csv")
    out.to_csv(p, index=False)
    return out["coverage_gap_score"].to_numpy()


class LBc(LB):
    """LB with public-row weights pi_hat (E14): mean_P(h) ~ pi_hat . h / sum(pi_hat) instead of the
    mean over all graded rows. Predicts held-out public scores to ~5e-5 (vs ~5e-4 uncorrected)."""

    def __init__(self, basis=BASIS, pi_path="exp11/pi_hat.npy"):
        self.pi = np.load(pi_path)
        super().__init__(basis)

    def AP(self, h):
        return float(h @ self.pi / self.pi.sum())

    def add(self, f, score=None):
        v = self.vec(f)[self.graded] if isinstance(f, str) else f[1][self.graded]
        name = f if isinstance(f, str) else f[0]
        sc = self.s[name] if score is None else score
        self.cols[name] = v
        self.c[name] = (self.AP(v ** 2) + self.Z - sc ** 2) / 2

    def _sys(self, ridge=1e-7):
        names = list(self.cols)
        A = np.column_stack([self.cols[n] for n in names])
        G = (A * self.pi[:, None]).T @ A / self.pi.sum()
        G = G + ridge * np.trace(G) / len(names) * np.eye(len(names))
        return names, A, G, np.array([self.c[n] for n in names])

    def public_score(self, g_full):
        """Predicted public RMSE of a candidate in the span of the basis (exact up to pi_hat)."""
        g = g_full[self.graded]
        names, A, G, c = self._sys()
        a = np.linalg.lstsq(A, g, rcond=None)[0]
        resid = g - A @ a
        mse = self.AP(g ** 2) - 2 * a @ c - 2 * self.AP(resid * g) * 0 + self.Z
        return np.sqrt(max(mse, 0)), float(np.sqrt(np.mean(resid ** 2)))

    def validate(self, g_full):
        g = g_full[self.graded]
        names, A, G, c = self._sys()
        Gi = np.linalg.inv(G)
        b = c - (A * self.pi[:, None]).T @ g / self.pi.sum()
        pe2 = float(b @ Gi @ b)
        _, mse_b = self.blend()
        a = Gi @ ((A * self.pi[:, None]).T @ g / self.pi.sum())
        perp = g - A @ a
        n2 = self.AP(perp ** 2)
        sb = np.sqrt(max(mse_b, 0))
        v0 = pe2 + mse_b + n2
        vp = v0 - 2 * RHO_PRIOR * sb * np.sqrt(n2)
        return {"lb_proj_rmse": np.sqrt(v0), "lb_prior_rmse": np.sqrt(max(vp, 0)), "proj_err": np.sqrt(pe2),
                "novel": np.sqrt(n2), "mean": float(g.mean())}
