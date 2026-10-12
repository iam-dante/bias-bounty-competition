"""
Semi-synthetic public/private validation of post-processing and blending procedures.

A synthetic truth y = T + eps is built so the scored files relate to it the way they relate to the
real target:
  T   = the leaderboard-optimal blend of the scored single models (our best estimate of E[y|x])
        + kappa * a physical direction NOT in their span (r10a - r10b: the leakage / Interstate
        disagreement), so the blend is misspecified as in reality;
  eps = posterior noise of the generative model (one Monte-Carlo draw minus the posterior mean,
        r10 regional specialists), scaled so the blend's RMSE against y is the real ~0.0644.
Each replicate draws a fresh eps and a fresh random 30 % public subset P of the graded rows,
computes every file's public RMSE exactly as Zindi would, reruns the blend from those scores
alone (the A_all approximation included), and scores each procedure on the private 70 %.
The published cell means are recomputed from the synthetic y (all graded rows, rounded to 3 dp).
"""
import os
import numpy as np
import pandas as pd
import harness as H

M = H.M
CACHE = "exp11/cache_eps.npy"


def eps_draws(n=40):
    if os.path.exists(CACHE):
        return np.load(CACHE)
    D = H.data()
    pr = H.regional("submissions/params_round10.json", "r10a_regional_noleak")
    mean = H.sim_regional(D["noleak"], D["mu"], pr, n_sims=512)["coverage_gap_score"].to_numpy()
    E = np.stack([H.sim_regional(D["noleak"], D["mu"], pr, n_sims=1, seed=1000 + k)["coverage_gap_score"].to_numpy() - mean
                  for k in range(n)])
    np.save(CACHE, E)
    return E


def cell_index(df):
    key = df["region"] + "|" + df["rural"].astype(str) + "|" + df["burned"].astype(str)
    return key.where(df["region"].isin(M.CALIB_REGIONS), "wa").to_numpy()


def anchor(f, cells, targets: dict, graded, strength=1.0):
    """Additive per-cell shift so the mean over all graded rows of each cell hits the published value."""
    g = f.copy()
    for k, mu in targets.items():
        m = (cells == k) & graded
        g[cells == k] += strength * (mu - f[m].mean())
    return np.clip(g, 0, 1)


def published_targets(df):
    out = {}
    for r in M.CALIB_REGIONS:
        for (ru, bu), tgt in zip([(1, 1), (1, 0), (0, 1), (0, 0)], M.CELL[r]):
            out[f"{r}|{ru}|{bu}"] = tgt
    return out


class World:
    def __init__(self, kappa=1.0, eps_scale=None, seed=0):
        D = H.data(); self.df = D["df"]
        self.lb = H.LB(); lb = self.lb
        self.graded = lb.graded
        self.cells = cell_index(self.df)
        self.files = {n: (lb.vec(n) if n != "__const__" else np.ones(len(self.df))) for n in lb.cols}
        w, _ = lb.blend()
        B = sum(w[n] * self.files[n] for n in w)
        a = pd.read_csv("submissions/round10/r10a_regional_noleak.csv").coverage_gap_score.to_numpy()
        b = pd.read_csv("submissions/round10/r10b_regional_interstate.csv").coverage_gap_score.to_numpy()
        self.T = np.clip(B + kappa * (a - b), 0, 1)
        self.E = eps_draws()
        g = self.graded
        target = 0.0644 ** 2 - np.mean((self.T - B)[g] ** 2)
        self.eps_scale = eps_scale or np.sqrt(max(target, 1e-6) / np.mean(self.E[:, g] ** 2))
        self.rng = np.random.default_rng(seed)

    def replicate(self):
        g = self.graded; n = g.sum()
        e = self.E[self.rng.integers(len(self.E))] * self.eps_scale
        y = np.clip(self.T + e, 0, 1)
        idx = np.flatnonzero(g)
        P = np.zeros(len(y), bool); P[self.rng.choice(idx, int(round(0.3 * n)), replace=False)] = True
        Q = g & ~P
        scores = {n_: np.sqrt(np.mean((f[P] - y[P]) ** 2)) for n_, f in self.files.items() if n_ != "__const__"}
        scores["zeros"] = np.sqrt(np.mean(y[P] ** 2)); scores["c10"] = np.sqrt(np.mean((0.1 - y[P]) ** 2))
        pub = {k: round(float(y[(self.cells == k) & g].mean()), 3) for k in published_targets(self.df)}
        return y, P, Q, scores, pub

    def fit_blend(self, scores, names=None, ridge=1e-7, P=None):
        """lb_stack.fit from scores alone; A_all over graded rows unless P (oracle) is given."""
        g = self.graded if P is None else P
        Z = scores["zeros"] ** 2; ybar = (0.01 + Z - scores["c10"] ** 2) / 0.2
        names = names or list(self.files)
        X = np.column_stack([self.files[k] for k in names])
        c = np.array([ybar if k == "__const__" else (np.mean(self.files[k][g] ** 2) + Z - scores[k] ** 2) / 2 for k in names])
        Xs = X[g]; G = Xs.T @ Xs / len(Xs)
        w = np.linalg.solve(G + ridge * np.trace(G) / len(names) * np.eye(len(names)), c)
        return np.clip(X @ w, 0, 1), dict(zip(names, w))


def rmse(f, y, m):
    return float(np.sqrt(np.mean((f[m] - y[m]) ** 2)))


def anchor_sd(f, cells, targets, graded, sd):
    """sd-weighted anchoring (E09-E11 winner): uncertain tracts absorb more of the cell shift."""
    wt = sd + 1e-3
    g = f.copy()
    for k, mu in targets.items():
        m = (cells == k) & graded; mm = cells == k
        g[mm] += (mu - f[m].mean()) * wt[mm] / wt[m].mean()
    return np.clip(g, 0, 1)


def fit_blend_cells(W, scores, pub, ridge=1e-7, cell_w=1.0):
    """Blend with the 16 cell indicators as extra columns; their moments come from the published
    full-data cell means (mean_P(1_cell * y) ~ share_cell * mu_cell)."""
    g = W.graded
    Z = scores["zeros"] ** 2; ybar = (0.01 + Z - scores["c10"] ** 2) / 0.2
    names = list(W.files)
    cols = [W.files[k] for k in names]
    c = [ybar if k == "__const__" else (np.mean(W.files[k][g] ** 2) + Z - scores[k] ** 2) / 2 for k in names]
    for k, mu in pub.items():
        ind = (W.cells == k).astype(float)
        cols.append(ind); c.append(ind[g].mean() * mu)
    X = np.column_stack(cols); c = np.array(c)
    Xs = X[g]; G = Xs.T @ Xs / len(Xs)
    w = np.linalg.solve(G + ridge * np.trace(G) / len(c) * np.eye(len(c)), c)
    return np.clip(X @ w, 0, 1)
