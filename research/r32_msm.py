"""R32 / T2: the data_only.py physics simulator calibrated by the Method of Simulated Moments.

Run from the repo root: OPENBLAS_NUM_THREADS=4 python research/r32_msm.py

Generative model (each withdrawn reference count is simulated, then pushed through the exact
official composite; the mean over draws is E[y | permitted data]):
  roads      TIGER S1100+S1200 length = m_region * route length * exp(stratum effects + effort) * LogNormal
  buildings  Microsoft count          = kappa * USFS building count * LogNormal
  facilities reference count          = Bin(Overture primary, alpha) + Bin(weak evidence, alpha_x)
                                        + Poisson(unseen rate * (1 - r) / r * exp(stratum effects))
  CBP        establishments           = eps * population * exp(registry + transit + rural) * LogNormal
New permitted inputs vs data_only.py: mapping effort from the unfiltered road layer (service /
track / footpath share), Overture transit stops, the BrightQuery (business registry) share of
places, and the exact Bias Scorecard strata as effect indicators.

Calibration (MSM): weighted least squares on moments, weights = 1 / noise variance, plus a
Gaussian prior on the 34 parameters around the published-statistics fit (r9a). Common random
numbers; the moment Jacobian is built once at the prior centre by central differences and the
fit iterates Gauss-Newton steps re-simulating the residual (chord method).
Moment sets:
  A  "data": 9 true scorecard ratios, 5 Paradise scores, public mean(y) and mean(y^2)
  B  A + the scored-file moments mean(v * y) (leaderboard feedback; stated in METHOD.md)
Validation: chronological hold-out over scored files (refit on moments of files scored before
k, predict k's public RMSE) and leave-one-ratio-out prediction of the scorecard.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))
import data_only as D
import solution as S
import r32_scorecard as SC

OUT = ROOT / "research/r32"
SEED, S_FIT, S_JAC = 7, 1024, 1024
STRATA = ["rural", "tribal", "svi", "cvi", "sdrought", "wdrought", "wildfire", "heat"]
R9A = D.PHYSICS["r9a"]["global"]
LOGIT = lambda p: float(np.log(p / (1 - p)))
SIG = lambda x: 1 / (1 + np.exp(-x))
# name, prior centre (transformed scale), prior sd
PARAMS = [("log_mscale", 0.0, 0.3), ("log_sigma_t", np.log(R9A["sigma_t"]), 0.4), ("beta_bnd", R9A["beta_bnd"], 0.3),
          ("w_unr", 0.0, 0.3), ("w_tert", 0.0, 0.2), ("ge_effort", 0.0, 0.3),
          ("log_kappa", np.log(R9A["kappa_b"]), 0.3), ("log_sigma_b", np.log(0.10), 0.5),
          ("logit_r_fire", LOGIT(R9A["r_fire"]), 1.0), ("logit_r_ems", LOGIT(0.90), 1.0), ("logit_r_sch", LOGIT(0.95), 1.0),
          ("logit_ax_fire", LOGIT(R9A["alpha_fire_x"]), 1.0), ("logit_a_sch", LOGIT(R9A["alpha_sch"]), 1.0),
          ("log_eps", np.log(R9A["eps_cbp"]), 0.5), ("log_sigma_c", np.log(0.35), 0.4), ("a_bq", 0.0, 0.5), ("a_tr", 0.0, 0.3)]
GF0 = dict(rural=R9A["gf_rural"], tribal=R9A["gf_tribal"], svi=R9A["gf_svi"], cvi=0.0, sdrought=0.0,
           wdrought=R9A["gf_drw"], wildfire=R9A["gf_wild"], heat=R9A["gf_heat"])
GR0 = dict(rural=R9A["gr_rural"], tribal=0.0, svi=R9A["gr_svi"], cvi=R9A["gr_cvi"], sdrought=0.0,
           wdrought=R9A["gr_drw"], wildfire=R9A["gr_wild"], heat=R9A["gr_heat"])
PARAMS += [(f"gf_{k}", GF0[k], 0.5) for k in STRATA] + [(f"gr_{k}", GR0[k], 0.5) for k in STRATA]
PARAMS += [("gc_rural", R9A["gc_rural"], 0.5)]
NAMES = [p[0] for p in PARAMS]
THETA0 = np.array([p[1] for p in PARAMS])
PRIOR_SD = np.array([p[2] for p in PARAMS])


def zscore(x):
    x = np.asarray(x, float)
    return (x - x.mean()) / (x.std() + 1e-12)


def frame(lb):
    """Simulator inputs aligned to the model.npz GEOID order (all 9,794 rows)."""
    df = D.load()
    mu = D.prior_rates(df)
    df = D.road_frame(df)
    extra = pd.concat([pd.read_parquet(ROOT / D.FEAT / f"{r}-extra.parquet") for r in D.REGIONS]).drop_duplicates("GEOID")
    df = df.merge(extra, on="GEOID", how="left")
    df[[c for c in extra.columns if c != "GEOID"]] = df[[c for c in extra.columns if c != "GEOID"]].fillna(0)
    df = pd.concat([df, mu], axis=1).set_index("GEOID").loc[lb.geoid.astype(str)].reset_index()
    groups = SC.scorecard_groups(lb.geoid)
    F = {k: df[c].to_numpy(float) for k, c in [("N", "L_named"), ("L_route", "L_route"), ("L_unr", "L_unrouted"),
                                                ("L_tert", "L_tertiary"), ("O_b", "n_bld"), ("U_b", "U_bld"),
                                                ("pop", "pop"), ("n_poi", "n_poi")]}
    F["bnd"] = (df["Lb_route"] - df["L_route_nearB"]).clip(lower=0).to_numpy(float)
    F["m_r"] = np.asarray(list(R9A["m"]) + [float(np.mean(R9A["m"]))])[df["ridx"].to_numpy()]
    for k in ["fire", "ems", "sch"]:
        F[f"mu_{k}"] = df[f"mu_{k}"].to_numpy(float)
        F[f"X_{k}"] = df[f"X_{k}"].to_numpy().astype(int)
    F["O_fire"], F["O_ems"], F["O_sch"] = (df[c].to_numpy().astype(int) for c in ["n_fd", "n_ems", "n_sch"])
    detail = (df["L_service"] + df["L_track"] + df["L_ped"]) / (df["L_unf"] + 1.0)
    F["z_effort"] = zscore(np.log((detail + 0.01) / (1.01 - detail)))
    F["z_bq"] = zscore(np.log((df["n_poi_bq"] + 1) / (df["pop"] + 100)))
    F["z_tr"] = zscore(np.log1p(df["n_transit"]))
    for k in STRATA:
        F[f"s_{k}"] = groups[k][0].astype(float)
    F["n"] = len(df)
    return F, groups, df


def unpack(th):
    p = dict(zip(NAMES, th))
    return p


def simulate(F, th, S=S_FIT, seed=SEED):
    """E[y] and E[y^2] per tract under parameters th (common random numbers via seed)."""
    p = unpack(th)
    rng = np.random.default_rng(seed)
    n = F["n"]
    eff = lambda pre: np.exp(sum(p[f"{pre}_{k}"] * F[f"s_{k}"] for k in STRATA))
    ln = lambda sig, shape: np.exp(sig * rng.standard_normal(shape) - 0.5 * sig ** 2)
    # roads
    G = np.clip(F["L_route"] + p["beta_bnd"] * F["bnd"] + p["w_unr"] * F["L_unr"] + p["w_tert"] * F["L_tert"], 0, None)
    has = G > 1.0
    Tm = np.exp(p["log_mscale"]) * F["m_r"] * G * eff("gr") * np.exp(p["ge_effort"] * F["z_effort"])
    T = Tm[None, :] * ln(np.exp(p["log_sigma_t"]), (S, n))
    surprise = (~has)[None, :] & (rng.random((S, n)) < 0.03)
    T = np.where(surprise, 400.0 * ln(1.0, (S, n)), T)
    dT = has[None, :] | surprise
    t = np.where(dT, 1 - np.minimum(1, F["N"][None, :] / np.maximum(T, 1e-9)), 0.0)
    # buildings
    MS = (np.exp(p["log_kappa"]) * F["U_b"])[None, :] * ln(np.exp(p["log_sigma_b"]), (S, n))
    dB = MS > 0.5
    b = np.where(dB, 1 - np.minimum(1, F["O_b"][None, :] / np.maximum(MS, 1e-9)), 0.0)
    # facilities
    fac = eff("gf")
    spec = {"fire": (0.90, SIG(p["logit_ax_fire"]), SIG(p["logit_r_fire"])),
            "ems": (0.90, R9A["alpha_ems_x"], SIG(p["logit_r_ems"])),
            "sch": (SIG(p["logit_a_sch"]), R9A["alpha_sch_x"], SIG(p["logit_r_sch"]))}
    gs, ds = [], []
    for k, (a, ax, r) in spec.items():
        lam = F[f"mu_{k}"] * (1 - r) / r * fac
        H = (rng.binomial(np.broadcast_to(F[f"O_{k}"], (S, n)), a) + rng.binomial(np.broadcast_to(F[f"X_{k}"], (S, n)), ax)
             + rng.poisson(np.broadcast_to(lam, (S, n))))
        d = H >= 1
        gs.append(np.where(d, 1 - np.minimum(1, F[f"O_{k}"][None, :] / np.maximum(H, 1)), 0.0))
        ds.append(d)
    nd = ds[0].astype(int) + ds[1] + ds[2]
    dH = nd > 0
    h = np.where(dH, (gs[0] + gs[1] + gs[2]) / np.maximum(nd, 1), 0.0)
    cm = np.exp(p["log_eps"]) * F["pop"] * np.exp(p["a_bq"] * F["z_bq"] + p["a_tr"] * F["z_tr"] + p["gc_rural"] * F["s_rural"])
    CBP = cm[None, :] * ln(np.exp(p["log_sigma_c"]), (S, n))
    dC = CBP >= 0.5
    c = np.where(dC, 1 - np.minimum(1, F["n_poi"][None, :] / np.maximum(CBP, 1e-9)), 0.0)
    nP = dH.astype(int) + dC
    pg = np.where(nP > 0, (h + c) / np.maximum(nP, 1), 0.0)
    nd_all = dT.astype(int) + dB + (nP > 0)
    y = np.where(nd_all > 0, (t + b + pg) / np.maximum(nd_all, 1), 0.0)
    return y.mean(0), (y ** 2).mean(0)


class Moments:
    """Observed moments, their noise and the map from simulated (E[y], E[y^2]) to model moments."""

    def __init__(self, lb, groups, use_lb=False, files=None):
        self.lb, self.groups, self.use_lb = lb, groups, use_lb
        g = lb.graded
        self.w = lb.pi[...] / lb.pi.sum()
        lead = SC.leaders(1)[0]["rubric"]
        self.ratio_obs = np.array([lead[SC.API[k]] for k in groups])
        self.ratio_sd = np.full(len(groups), 0.0065)     # leaders' WA disagreement (T1) + Monte Carlo at 1,024 draws
        gid = lb.geoid.astype(str).to_numpy()
        self.par_idx = np.array([np.where(gid == k)[0][0] for k in S.PUBLISHED])
        self.par_obs = np.array(list(S.PUBLISHED.values()))
        if use_lb:
            self.Q, self.o, self.N = moment_basis(lb, files)

    def model(self, Ey, Ey2):
        g = self.lb.graded
        r = np.array([Ey[h].mean() / Ey[l].mean() for h, l in self.groups.values()])
        m = [r, Ey[self.par_idx], [self.w @ Ey[g]], [self.w @ Ey2[g]]]
        if self.use_lb:
            m.append((self.Q * self.w[:, None]).T @ Ey[g])
        return np.concatenate(m)

    def observed(self):
        m = [self.ratio_obs, self.par_obs, [self.lb.ybar], [self.lb.Z]]
        if self.use_lb:
            m.append(self.o)
        return np.concatenate(m)

    def cov(self, Ey, Ey2):
        g = self.lb.graded
        var = np.clip(Ey2 - Ey ** 2, 1e-6, None)
        p, n = S.P_PUBLIC, g.sum()
        w = self.w
        sd_ybar = np.sqrt(((1 - p) * (w @ (Ey[g] - w @ Ey[g]) ** 2) + w @ var[g]) / (p * n))
        var_y2 = 4 * Ey2[g] * var[g] + 2 * var[g] ** 2
        sd_z = np.sqrt(((1 - p) * (w @ (Ey2[g] - w @ Ey2[g]) ** 2) + w @ var_y2) / (p * n))
        if self.use_lb:              # the constant probe's moment already carries mean(y)
            sd_ybar = 1e3
        # Paradise: one realisation each, plus 0.05 structural error (post-fire rebuild the simulator cannot see)
        d = np.r_[self.ratio_sd ** 2, var[self.par_idx] + 0.05 ** 2, sd_ybar ** 2, sd_z ** 2]
        C = np.diag(d)
        if self.use_lb:
            k = len(self.o)
            C = np.block([[C, np.zeros((len(d), k))], [np.zeros((k, len(d))), self.N]])
        return C


def moment_basis(lb, files, yhat=None):
    """eb_posterior's orthogonalised moment basis for a subset of scored files (noise scale 1)."""
    g = lb.graded
    w = lb.pi / lb.pi.sum()
    if yhat is None:
        yhat = lb.vec("r12_blend_with_AB.csv")[g]
    V = np.column_stack([lb.vec(f)[g] for f in files])
    c = np.array([(lb.AP(V[:, i] ** 2) + lb.Z - lb.s[f] ** 2) / 2 for i, f in enumerate(files)])
    U = V * (yhat[:, None] - V / 2)
    Uc = U - w @ U
    Nv = ((1 - S.P_PUBLIC) * (Uc * w[:, None]).T @ Uc + 0.0624 ** 2 * (V * w[:, None]).T @ V) / (S.P_PUBLIC * len(w))
    Q, keep = [], []
    for i in range(V.shape[1]):
        v = V[:, i].copy()
        for _ in range(2):
            for q in Q:
                v -= (w @ (q * v)) / (w @ (q * q)) * q
        if np.sqrt(w @ v ** 2) > 1e-5 * np.sqrt(w @ V[:, i] ** 2):
            Q.append(v)
            keep.append(i)
    Q = np.column_stack(Q)
    sw = np.sqrt(w)[:, None]
    Rm = np.linalg.lstsq(Q * sw, V[:, keep] * sw, rcond=None)[0]
    Rit = np.linalg.inv(Rm).T
    return Q, Rit @ c[keep], Rit @ Nv[np.ix_(keep, keep)] @ Rit.T


def jacobian(F, th, S=S_JAC):
    """Central differences of (E[y], E[y^2]) in the transformed parameters, common random numbers."""
    Jy, Jy2 = np.zeros((F["n"], len(th))), np.zeros((F["n"], len(th)))
    for j in range(len(th)):
        h = 0.1 * PRIOR_SD[j]
        up, dn = th.copy(), th.copy()
        up[j] += h
        dn[j] -= h
        a, a2 = simulate(F, up, S)
        b, b2 = simulate(F, dn, S)
        Jy[:, j], Jy2[:, j] = (a - b) / (2 * h), (a2 - b2) / (2 * h)
    return Jy, Jy2


def moment_jacobian(M, Jy, Jy2, Ey, Ey2, eps=1e-7):
    """d moments / d theta by the chain rule (moments are explicit functions of E[y], E[y^2])."""
    base = M.model(Ey, Ey2)
    J = np.zeros((len(base), Jy.shape[1]))
    for j in range(Jy.shape[1]):
        J[:, j] = (M.model(Ey + eps * Jy[:, j], Ey2 + eps * Jy2[:, j]) - base) / eps
    return J


def calibrate(F, M, Jm, th0=THETA0, iters=25, linear=None):
    """Regularised MSM fit: minimise r' W r + (th - THETA0)' P^-1 (th - THETA0).
    Nonlinear: Levenberg-Marquardt on re-simulated residuals (common random numbers), the moment
    Jacobian updated by Broyden after each accepted step. With `linear=(Ey0, Ey20, Jy, Jy2, th_lin)`
    the simulator is replaced by its first-order expansion around th_lin (chronological folds)."""
    obs = M.observed()
    Pinv = np.diag(1 / PRIOR_SD ** 2)

    def run(th):
        if linear is None:
            return simulate(F, th)
        Ey0, Ey20, Jy, Jy2, th_lin = linear
        return Ey0 + Jy @ (th - th_lin), Ey20 + Jy2 @ (th - th_lin)

    def cost(th, Ey, Ey2):
        r = M.model(Ey, Ey2) - obs
        W = np.linalg.inv(M.cov(Ey, Ey2))
        return float(r @ W @ r + (th - THETA0) @ Pinv @ (th - THETA0)), r, W

    th = th0.copy()
    Ey, Ey2 = run(th)
    c, r, W = cost(th, Ey, Ey2)
    Jm, lam = Jm.copy(), (1e-3 if linear is not None else 0.3)
    for _ in range(iters):
        A = Jm.T @ W @ Jm + Pinv
        grad = Jm.T @ W @ r + Pinv @ (th - THETA0)
        step = np.linalg.solve(A + lam * np.diag(np.diag(A)), -grad)
        th_new = th + step
        Ey_n, Ey2_n = run(th_new)
        c_new, r_new, W_new = cost(th_new, Ey_n, Ey2_n)
        if c_new < c:
            Jm += np.outer((r_new - r) - Jm @ step, step) / (step @ step)      # Broyden update
            gain = c - c_new
            th, Ey, Ey2, c, r, W, lam = th_new, Ey_n, Ey2_n, c_new, r_new, W_new, lam / 3
            if gain < 1e-3 * c:
                break
        else:
            lam *= 4
            if lam > 1e6:
                break
    chi2 = float(r @ W @ r)
    return th, Ey, Ey2, chi2


def forward_validation(lb, groups, EyA, lin, first="p12_01_route_ratio.csv", gammas=(0.25, 0.5, 1.0)):
    """Chronological hold-out: for each scored file k, predict its public RMSE as the span
    projection on files scored before k plus gamma times the physics value of k's out-of-span
    part (gamma = 0 is the current estimator). A = the LB-free fit; B = refit adding the moments
    of files scored before k (first-order expansion around the A fit)."""
    g = lb.graded
    w = lb.pi / lb.pi.sum()
    files = [f for f in lb.s if not f.startswith(("probe_", "e06"))]
    order = [f for f in lb.s if f not in ("probe_zeros.csv", "e06_r2f_only_eastern-wa.csv")]
    vec = {f: lb.vec(f)[g] for f in files}
    mom = {f: (lb.AP(vec[f] ** 2) + lb.Z - lb.s[f] ** 2) / 2 for f in files}
    Ey_l, Ey2_l, Jy, Jy2, th_l = lin
    rows = []
    for i, f in enumerate(files):
        if i < files.index(first):
            continue
        X = np.column_stack([np.ones(g.sum())] + [vec[q] for q in files[:i]])
        m = np.array([lb.ybar] + [mom[q] for q in files[:i]])
        gram = (X * w[:, None]).T @ X
        gram += 1e-7 * np.trace(gram) / len(m) * np.eye(len(m))
        coef = np.linalg.solve(gram, (X * w[:, None]).T @ vec[f])
        r = vec[f] - X @ coef
        MB = Moments(lb, groups, use_lb=True, files=order[:order.index(f)])
        _, EyB, _, _ = calibrate(None, MB, moment_jacobian(MB, Jy, Jy2, Ey_l, Ey2_l), th0=th_l, linear=lin)
        row = dict(file=f, actual=lb.s[f])
        for tag, Ey in [("A", EyA), ("B", EyB)]:
            for gm in (0.0,) + tuple(gammas):
                cy = coef @ m + gm * (w @ (r * Ey[g]))
                row[f"{tag}_g{gm}"] = float(np.sqrt(max(lb.AP(vec[f] ** 2) - 2 * cy + lb.Z, 0))) - lb.s[f]
        rows.append(row)
    return pd.DataFrame(rows)


def leave_one_ratio_out(lb, groups, lin0):
    """Refit without one scorecard ratio (data moments only, first-order around the prior
    centre, which uses no ratio) and predict the held-out ratio."""
    Ey0, Ey20, Jy, Jy2, th0 = lin0
    rows = []
    for j, k in enumerate(groups):
        M = Moments(lb, groups)
        M.ratio_sd = M.ratio_sd.copy()
        M.ratio_sd[j] = 1e3
        _, Ey, _, _ = calibrate(None, M, moment_jacobian(M, Jy, Jy2, Ey0, Ey20), th0=th0, linear=lin0)
        hi, lo = groups[k]
        rows.append(dict(stratum=k, truth=M.ratio_obs[j], prior=Ey0[hi].mean() / Ey0[lo].mean(),
                         held_out_fit=Ey[hi].mean() / Ey[lo].mean()))
    return pd.DataFrame(rows)


def summarise(lb, M, th, Ey, Ey2, chi2, base, groups):
    res = M.model(Ey, Ey2) - M.observed()
    pull = res / np.sqrt(np.diag(M.cov(Ey, Ey2)))
    v = base.copy()
    v[lb.graded] = Ey[lb.graded]
    out = S.finish(v, lb.geoid)
    return out, dict(chi2=chi2, n_moments=len(res), est_public_physics=SC.estimate(lb, out),
                     ratios=dict(zip(groups, np.round(M.model(Ey, Ey2)[:9], 3).tolist())),
                     ratio_pulls=dict(zip(groups, np.round(pull[:9], 2).tolist())),
                     paradise=np.round(Ey[M.par_idx], 3).tolist(), ybar_pull=float(pull[14]), z_pull=float(pull[15]),
                     theta={k: round(float(x), 4) for k, x in zip(NAMES, th)},
                     theta_shift_sd={k: round(float(x), 2) for k, x in zip(NAMES, (th - THETA0) / PRIOR_SD)})


def main():
    t0 = time.time()
    lb = S.Leaderboard()
    lb.public_subset()
    F, groups, df = frame(lb)
    base = lb.vec(SC.BASE_NAME)
    clock = lambda: f"{time.time() - t0:.0f}s"
    Ey0, Ey20 = simulate(F, THETA0)
    J0, J20 = jacobian(F, THETA0)
    print(f"prior centre: mean {Ey0.mean():.4f}; Jacobian {J0.shape} ({clock()})", flush=True)
    results = {}
    MP = Moments(lb, groups)
    out0, results["prior"] = summarise(lb, MP, THETA0, Ey0, Ey20, float("nan"), base, groups)

    MA = Moments(lb, groups)
    thA, EyA, Ey2A, chi2 = calibrate(F, MA, moment_jacobian(MA, J0, J20, Ey0, Ey20))
    outA, results["A"] = summarise(lb, MA, thA, EyA, Ey2A, chi2, base, groups)
    JA, J2A = jacobian(F, thA)
    print(f"MSM-A fitted ({clock()})", flush=True)
    files_all = [f for f in lb.s if f not in ("probe_zeros.csv", "e06_r2f_only_eastern-wa.csv")]
    MB = Moments(lb, groups, use_lb=True, files=files_all)
    thB, EyB, Ey2B, chi2 = calibrate(F, MB, moment_jacobian(MB, JA, J2A, EyA, Ey2A), th0=thA)
    outB, results["B"] = summarise(lb, MB, thB, EyB, Ey2B, chi2, base, groups)
    np.savez_compressed(OUT / "msm_fit.npz", Ey0=Ey0, Ey20=Ey20, J0=J0, J20=J20, thA=thA, EyA=EyA, Ey2A=Ey2A,
                        JA=JA, J2A=J2A, thB=thB, EyB=EyB, Ey2B=Ey2B)
    for tag, out in [("prior", out0), ("A", outA), ("B", outB)]:
        np.save(OUT / f"msm_{tag}.npy", out)
        r = results[tag]
        print(f"{tag:5s}: chi2 {r['chi2']:.1f} on {r['n_moments']} moments; physics alone est public {r['est_public_physics']:.6f}")
        print(f"   ratios {r['ratios']}\n   pulls {r['ratio_pulls']}; paradise {r['paradise']}; ybar pull {r['ybar_pull']:.2f}, Z pull {r['z_pull']:.2f}")
        print(f"   parameters moved > 1 prior sd: { {k: x for k, x in r['theta_shift_sd'].items() if abs(x) > 1} }", flush=True)

    lr = leave_one_ratio_out(lb, groups, (Ey0, Ey20, J0, J20, THETA0))
    lr.to_csv(OUT / "msm_leave_one_ratio_out.csv", index=False)
    print(lr.round(3).to_string())
    print(f"leave-one-ratio-out MAE: prior {np.abs(lr.prior - lr.truth).mean():.4f}, "
          f"refit without it {np.abs(lr.held_out_fit - lr.truth).mean():.4f}", flush=True)

    fv = forward_validation(lb, groups, EyA, (EyA, Ey2A, JA, J2A, thA))
    fv.to_csv(OUT / "msm_forward_validation.csv", index=False)
    summ = {c: float(fv[c].abs().mean()) for c in fv.columns if c.startswith(("A_", "B_"))}
    print("forward MAE (n=%d): " % len(fv) + ", ".join(f"{k} {v:.6f}" for k, v in summ.items()))
    results["forward_mae"] = summ
    results["leave_one_ratio_out"] = lr.to_dict(orient="records")
    (OUT / "msm_summary.json").write_text(json.dumps(results, indent=2))
    print(f"done ({clock()})")


if __name__ == "__main__":
    main()
