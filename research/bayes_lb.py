"""Bayesian moment model of the leaderboard.

Every scored file f gives c_f = <v_f, y>_pub (estimated with public-row weights pi).
Model: y = Q theta + Phi w + eps on graded rows, where Q is the pi-orthogonal
chronological Gram-Schmidt basis of the scored files, Phi are permitted tract
features, theta_k ~ N(mu_g, tau_g^2) per group, w ~ N(0, tau_phi^2 / m).
Noise: public-vs-pi sampling of v (y - v/2) plus pubmean(v eps).
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

REPO = Path("/Users/briantemu/Downloads/projects/competitions/bias-bounty-competition")
sys.path.insert(0, str(REPO))
import solution as S  # noqa: E402

P_PUB = 0.3


def load():
    lb = S.Leaderboard()
    lb.public_subset()
    g = lb.graded
    w = lb.pi / lb.pi.sum()
    names = [f for f in lb.s if f not in ("probe_zeros.csv", "e06_r2f_only_eastern-wa.csv")]
    V = np.column_stack([lb.vec(f)[g] for f in names])
    c = np.array([(lb.AP(V[:, i] ** 2) + lb.Z - lb.s[f] ** 2) / 2 for i, f in enumerate(names)])
    return lb, g, w, names, V, c


def group_of(name):
    if name.startswith("probe_const"):
        return "const"
    if re.match(r"^p\d+_\d+_", name):
        return "probe"
    if name.startswith(("r14", "r19b")):
        return "clip"
    return "base"


def gram_schmidt(V, w, names, tol=1e-5):
    """pi-weighted chronological Gram-Schmidt. Returns kept file indices, Q, R (V_kept = Q R)."""
    n, k = V.shape
    Q, keep = [], []
    for i in range(k):
        v = V[:, i].copy()
        for q in Q:  # two passes for stability
            v -= (w @ (q * v)) / (w @ (q * q)) * q
        for q in Q:
            v -= (w @ (q * v)) / (w @ (q * q)) * q
        rel = np.sqrt(w @ v ** 2) / max(np.sqrt(w @ V[:, i] ** 2), 1e-12)
        if rel > tol:
            Q.append(v)
            keep.append(i)
    Q = np.column_stack(Q)
    Vk = V[:, keep]
    R = np.linalg.lstsq(Q * np.sqrt(w)[:, None], Vk * np.sqrt(w)[:, None], rcond=None)[0]
    return keep, Q, R


def noise_cov(V, w, yhat, sigma2):
    """Cov of c-estimation error over random public sets (pi-weighted proxy)."""
    n_pub = P_PUB * len(w)
    U = V * (yhat[:, None] - V / 2)
    Uc = U - w @ U
    C1 = (Uc * w[:, None]).T @ Uc
    C2 = (V * w[:, None]).T @ V * sigma2
    return ((1 - P_PUB) * C1 + C2) / n_pub


def features(geoid_graded):
    frames = []
    for f in sorted((REPO / "data/features").glob("*.parquet")):
        if "facpts" in f.name:
            continue
        frames.append(pd.read_parquet(f))
    F = pd.concat(frames, ignore_index=True).drop_duplicates("GEOID").set_index("GEOID")
    F = F.reindex(pd.Index(geoid_graded.astype(str)))
    region = F["region"].astype(str)
    num = F.select_dtypes(include=[np.number]).copy()
    drop = [c for c in num.columns if re.search(r"(_year|year_|_covered$|record_|window_|edition|vintage|STATEFP|COUNTYFP|frac_inside)", c)]
    num = num.drop(columns=drop)
    area = (F["area_m2"].astype(float) / 1e6).clip(lower=1e-3)
    hu = F["housing_units"].astype(float).clip(lower=1)
    for col in [c for c in num.columns if c.startswith(("L_", "Lb_", "n_"))]:
        num[col + "_per_km2"] = num[col] / area
        num[col + "_per_hu"] = num[col] / hu
    num["hu_per_km2"] = hu / area
    out = {}
    for col in num.columns:
        x = num[col].astype(float)
        if x.notna().mean() < 0.5 or x.nunique() < 3:
            continue
        if x.min() >= 0 and x.skew() > 2:
            x = np.log1p(x)
        x = (x - x.mean()) / (x.std() + 1e-12)
        out[col] = x.fillna(0).clip(-5, 5)
    X = pd.DataFrame(out)
    for r in sorted(region.unique()):
        X["reg_" + r] = (region == r).astype(float)
    return X


class Model:
    def __init__(self, Q, R, keep, names, c, Nv, Phi=None):
        self.Q, self.keep = Q, keep
        self.names = [names[i] for i in keep]
        self.groups = np.array([group_of(f) for f in self.names])
        self.Rinv_T = np.linalg.inv(R).T
        self.o = self.Rinv_T @ c[keep]                     # observations in Q basis
        self.N = self.Rinv_T @ Nv[np.ix_(keep, keep)] @ self.Rinv_T.T
        self.Phi = Phi

    def setup(self, w):
        self.D = w @ (self.Q * self.Q)                     # |q_k|^2
        self.QtPhi = None if self.Phi is None else (self.Q * w[:, None]).T @ self.Phi

    def prior(self, p):
        mu = np.zeros(len(self.names))
        var = np.zeros(len(self.names))
        for gname in ("const", "base", "probe", "clip"):
            m = self.groups == gname
            mu[m] = p.get("mu_" + gname, 0.0)
            var[m] = p["tau_" + gname] ** 2
        return mu, var

    def cov_obs(self, p):
        mu, var = self.prior(p)
        C = (self.D[:, None] * var * self.D[None, :]) * np.eye(len(var)) + p["alpha_N"] * self.N
        if self.QtPhi is not None and p.get("tau_phi", 0) > 0:
            C = C + p["tau_phi"] ** 2 / self.Phi.shape[1] * self.QtPhi @ self.QtPhi.T
        return self.D * mu, C

    def nll(self, p):
        m, C = self.cov_obs(p)
        L = np.linalg.cholesky(C + 1e-30 * np.eye(len(m)))
        r = np.linalg.solve(L, self.o - m)
        return 0.5 * r @ r + np.log(np.diag(L)).sum()

    def posterior_y(self, p, idx=None):
        """Posterior mean of y on graded rows given observations idx (default all)."""
        idx = np.arange(len(self.o)) if idx is None else np.asarray(idx)
        mu, var = self.prior(p)
        m, C = self.cov_obs(p)
        a = np.linalg.solve(C[np.ix_(idx, idx)], (self.o - m)[idx])
        # Cov(theta, o_idx) = var * D (diag) ; Cov(w, o_idx) = tau^2/m Phi^T Pi Q
        theta = mu.copy()
        theta[idx] += var[idx] * self.D[idx] * a
        y = self.Q @ theta
        if self.QtPhi is not None and p.get("tau_phi", 0) > 0:
            wpost = p["tau_phi"] ** 2 / self.Phi.shape[1] * self.QtPhi[idx].T @ a
            y = y + self.Phi @ wpost
        return y

    def predict_obs(self, p, train, test):
        m, C = self.cov_obs(p)
        a = np.linalg.solve(C[np.ix_(train, train)], (self.o - m)[train])
        mean = m[test] + C[np.ix_(test, train)] @ a
        var = np.diag(C[np.ix_(test, test)] - C[np.ix_(test, train)] @ np.linalg.solve(C[np.ix_(train, train)], C[np.ix_(train, test)]))
        return mean, var
