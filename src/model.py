"""
Label-free estimator of the organiser coverage-gap score.

There is no training target in this challenge: the reference layers (TIGER roads, Microsoft
footprints, HIFLD facilities, CBP establishments) were withdrawn on 2026-09-25 and rebuilding
them from outside sources is prohibited. So instead of fitting a regressor to labels we:

  1. write down a generative model of each withdrawn reference count, driven by evidence that
     *is* in the permitted data (Overture route refs, tract-boundary geometry, USFS building
     counts, Overture facility names/alternate categories, population ...);
  2. push Monte-Carlo draws of those references through the EXACT official formula
     (per-component 1 - min(1, overture/reference), "mean of the defined components",
     POI = mean of the HIFLD and CBP halves, HIFLD half = mean over defined facility types),
     which yields E[score | features] -- the RMSE-optimal prediction;
  3. calibrate the ~10 structural parameters by the method of moments against the only
     public facts about the target: the official share of tracts with an undefined component,
     region x urban/rural x burned means and component means published on the discussion
     board, and five tracts with published component values.

The Overture side of every ratio (named-highway length, building count, facility counts) is
computed exactly, so all uncertainty is on the reference side.
"""
from __future__ import annotations

import os
import warnings

warnings.filterwarnings("ignore")
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor

DATA = os.environ.get("BB_DATA", "data")
REGIONS = ["eastern-ok", "maricopa-az", "northern-ca", "south-central-tx", "eastern-wa"]
# regions with published calibration statistics; eastern-wa (added 2026-09-08) has none, so it
# borrows the mean of the calibrated regional parameters
CALIB_REGIONS = REGIONS[:4]
RIDX = {r: i for i, r in enumerate(REGIONS)}


# ----------------------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------------------
def load() -> pd.DataFrame:
    frames = []
    for r in REGIONS:
        f = pd.read_parquet(f"{DATA}/features/{r}.parquet")
        ss = pd.read_csv(f"{DATA}/reference/{r}/{r}-sample-submission.csv", dtype={"GEOID": str})
        f = ss[["GEOID"]].merge(f, on="GEOID", how="left")  # authoritative scored list & order
        f["region"] = r
        frames.append(f)
    df = pd.concat(frames, ignore_index=True)
    zero_cols = [c for c in df.columns if c.startswith(("L_", "Lb_", "n_"))] + ["housing_units"]
    df[zero_cols] = df[zero_cols].fillna(0)
    df["area_km2"] = df["area_m2"] / 1e6
    df["rural"] = (df["ruca_primary"] >= 4).astype(int)
    df["burned"] = df["mtbs_wildfire_ever"].fillna(False).astype(bool).astype(int)
    # bias-scorecard groups (definitions reproduce our public scorecard to the 3rd decimal;
    # medians over all rows of the submission, eastern-wa included)
    above = lambda c: (df[c] > df[c].median()).astype(int)
    df["g_rural"] = (df["pct_urban"] < 0.5).astype(int)
    df["g_tribal"] = df["tribal_any"].fillna(False).astype(bool).astype(int)
    df["g_svi"], df["g_cvi"] = above("svi_overall"), above("cvi_overall")
    df["g_drs"], df["g_drw"] = above("usdm_summer_dsci"), above("usdm_winter_dsci")
    df["g_wild"], df["g_heat"] = above("usfs_WHP_mean"), above("epht_heat_days_summer")
    # scorecard "High Hazard + High Vulnerability": any hazard high AND social vulnerability high
    df["g_inter"] = (((df["g_wild"] + df["g_heat"] + df["g_drs"]) > 0) & (df["g_svi"] == 1)).astype(int)
    df["ridx"] = df["region"].map(RIDX)
    df["pop"] = df["pop_total"].fillna(0).clip(lower=0)
    df["county"] = df["GEOID"].str[:5]
    # evidence counts for each HIFLD facility type (anything in Overture that looks like one)
    # primary-category evidence is what Overture's numerator counts; "_x" is weaker evidence
    # (alternate category / facility-like name) of a facility filed under another category
    df["E_fire"] = df["n_fd"] + df["n_fd_alt"] + df["n_fire_name"]
    df["E_ems"] = df["n_ems"] + df["n_ems_alt"] + df["n_ems_name"]
    df["E_sch"] = df["n_sch"] + df["n_sch_alt"] + df["n_sch_name"]
    df["X_fire"] = df["n_fd_alt"] + df["n_fire_name"]
    df["X_ems"] = df["n_ems_alt"] + df["n_ems_name"]
    df["X_sch"] = df["n_sch_alt"] + df["n_sch_name"]
    # building reference proxy (USFS Wildfire-Risk BuildingCount; CarbonPlan is Overture-derived)
    u = df["usfs_BuildingCount_sum"]
    df["U_bld"] = u.where(u > 0, df["n_bld"]).fillna(df["n_bld"])
    return df


def prior_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Expected count of *Overture-visible* facility evidence given tract covariates (Poisson GLM).
    Used for the stations Overture does not show at all: unseen ~ Poisson(mu * (1 - r) / r)."""
    X = np.column_stack([
        np.log1p(df["pop"]), np.log1p(df["area_km2"]), np.log1p(df["n_poi"]), np.log1p(df["n_bld"]),
        df["rural"], np.log1p(df["housing_units"]), *[(df["ridx"] == i).astype(float) for i in range(len(REGIONS))]])
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    out = {}
    for k in ["E_fire", "E_ems", "E_sch"]:
        m = PoissonRegressor(alpha=1e-3, max_iter=1000).fit(X, df[k].clip(upper=20))
        out[f"mu_{k[2:]}"] = m.predict(X)
    return pd.DataFrame(out, index=df.index)


# ----------------------------------------------------------------------------------------------
# structural parameters
# ----------------------------------------------------------------------------------------------
@dataclass
class Params:
    # roads: TIGER length = m_r * (route geometry + w_fm * FM geometry + beta * boundary miss) * noise
    m: tuple = (1.7, 1.7, 2.0, 1.7)          # TIGER records per highway edge (per region)
    w_fm: float = 0.0                         # share of Texas FM/RM length that TIGER calls S1200
    beta_bnd: float = 0.5                     # boundary highway whose Overture copy is next door
    sigma_t: float = 0.25                     # log-noise of the TIGER proxy
    p_def0: float = 0.03                      # P(TIGER highway | no route evidence at all)
    # buildings: MS count = kappa * USFS count * noise
    kappa_b: float = 0.97
    sigma_b: float = 0.10
    # HIFLD facilities: H = Bin(primary-category count, alpha) + Bin(weak evidence, alpha_x)
    #                      + Poisson(mu * (1 - r) / r)   [stations Overture does not show at all]
    alpha_fire: float = 0.90
    alpha_fire_x: float = 0.60
    r_fire: float = 0.60
    alpha_ems: float = 0.90
    alpha_ems_x: float = 0.05
    r_ems: float = 0.90
    alpha_sch: float = 0.85
    alpha_sch_x: float = 0.10
    r_sch: float = 0.95
    # CBP establishments allocated to tracts by population: CBP = eps * pop * noise
    eps_cbp: float = 0.025
    sigma_c: float = 0.35
    use_cbp: bool = True
    # neighbour-aware facilities: each Overture facility point's HIFLD copy lands in a candidate
    # tract with probability Phi(d / sigma_loc) (d = signed distance to that tract, metres)
    use_points: bool = False
    sigma_loc: float = 60.0
    # burned-regime specialist (MTBS-burned tracts): multiplier on the unseen-facility rate and on
    # the TIGER length. 1.0 = shared parameters (rounds 1-3)
    burn_unseen: float = 1.0
    burn_road: float = 1.0
    # first-principles group effects (log multipliers, 0 = off). Rural / wildland / dryland fire
    # protection is volunteer and state stations that HIFLD lists but Overture often misses
    # (gf_*: unseen-facility rate); rural state highways are tagged below primary/secondary in OSM
    # and carry extra TIGER name records (gr_*: TIGER-length multiplier); Overture's place
    # coverage thins faster than real establishments in sparse areas (gc_rural: CBP density).
    gf_rural: float = 0.0
    gf_wild: float = 0.0
    gf_drw: float = 0.0
    gf_tribal: float = 0.0
    gf_svi: float = 0.0
    gf_heat: float = 0.0
    gr_rural: float = 0.0
    gr_wild: float = 0.0
    gr_drw: float = 0.0
    gr_heat: float = 0.0
    gc_rural: float = 0.0
    gf_cvi: float = 0.0
    gr_svi: float = 0.0
    gr_cvi: float = 0.0


def attach_points(df: pd.DataFrame) -> pd.DataFrame:
    """Load the facility point-tract tables (features.facility_points) into df.attrs."""
    from scipy import sparse
    idx = pd.Series(np.arange(len(df)), index=df["GEOID"].to_numpy())
    pts = pd.concat([pd.read_parquet(f"{DATA}/features/{r}-facpts.parquet") for r in REGIONS])
    pts = pts[pts["GEOID"].isin(idx.index)]
    out = {}
    for k in ["fire", "ems", "sch"]:
        q = pts[pts["typ"] == k].sort_values(["id", "d"], ascending=[True, False]).reset_index(drop=True)
        pid, j = np.unique(q["id"].to_numpy(), return_inverse=True)
        strong = q["strong"].fillna(False).astype(bool).groupby(j).first().to_numpy()
        A = sparse.csr_matrix((np.ones(len(q)), (idx[q["GEOID"]].to_numpy(), np.arange(len(q)))),
                              shape=(len(df), len(q)))
        out[k] = {"j": j, "d": q["d"].to_numpy(), "strong": strong, "A": A, "n_pts": len(pid)}
    df.attrs["pts"] = out
    return df


def _point_counts(rng, P: dict, alpha: float, alpha_x: float, sigma: float, S: int) -> np.ndarray:
    """(S, n_tracts) draws of reference facility counts implied by the Overture points."""
    from scipy.stats import norm
    j, d = P["j"], P["d"]
    w = norm.cdf(d / max(sigma, 1e-3))
    tot = np.bincount(j, weights=w, minlength=P["n_pts"])
    w = w / np.maximum(tot[j], 1.0)                       # corners: never more than one copy
    c = np.cumsum(w)
    start = np.r_[0, np.cumsum(np.bincount(j, minlength=P["n_pts"]))[:-1]]
    base = np.r_[0.0, c][start][j]                        # cumulative weight before this point
    hi, lo = c - base, c - base - w
    real = rng.random((S, P["n_pts"])) < np.where(P["strong"], alpha, alpha_x)
    u = rng.random((S, P["n_pts"]))
    sel = real[:, j] & (u[:, j] >= lo) & (u[:, j] < hi)
    return np.asarray((P["A"] @ sel.T.astype(np.float32)).T)


def _lognoise(rng, sigma, shape):
    return np.exp(sigma * rng.standard_normal(shape) - 0.5 * sigma ** 2)


def simulate(df: pd.DataFrame, mu: pd.DataFrame, p: Params, n_sims: int = 256, seed: int = 0,
             return_components: bool = False):
    """Monte-Carlo E[score | features] under the exact official composite rule."""
    rng = np.random.default_rng(seed)
    n, S = len(df), n_sims
    m_all = list(p.m) + [float(np.mean(p.m))] * (len(REGIONS) - len(p.m))
    m_r = np.asarray(m_all)[df["ridx"].to_numpy()]
    burned = df["burned"].to_numpy() == 1
    gcol = lambda c: df[c].to_numpy() if c in df else np.zeros(len(df))
    road_mult = np.exp(p.gr_rural * gcol("g_rural") + p.gr_wild * gcol("g_wild")
                       + p.gr_drw * gcol("g_drw") + p.gr_heat * gcol("g_heat")
                       + p.gr_svi * gcol("g_svi") + p.gr_cvi * gcol("g_cvi"))
    fac_mult = np.exp(p.gf_rural * gcol("g_rural") + p.gf_wild * gcol("g_wild") + p.gf_drw * gcol("g_drw")
                      + p.gf_tribal * gcol("g_tribal") + p.gf_svi * gcol("g_svi") + p.gf_heat * gcol("g_heat")
                      + p.gf_cvi * gcol("g_cvi"))

    # ---- road component --------------------------------------------------------------------
    N = df["L_named"].to_numpy()
    G = (df["L_route"] + p.w_fm * df["L_fm"]
         + p.beta_bnd * (df["Lb_route"] - df["L_route_nearB"]).clip(lower=0)).to_numpy()
    has_ev = G > 1.0
    T = (m_r * G * np.where(burned, p.burn_road, 1.0) * road_mult)[None, :] * _lognoise(rng, p.sigma_t, (S, n))
    # rare "TIGER highway with no route evidence": short piece, Overture usually has it as named
    surprise = (~has_ev)[None, :] & (rng.random((S, n)) < p.p_def0)
    T = np.where(surprise, 400.0 * _lognoise(rng, 1.0, (S, n)), T)
    dT = has_ev[None, :] | surprise
    t = np.where(dT, 1.0 - np.minimum(1.0, N[None, :] / np.maximum(T, 1e-9)), 0.0)

    # ---- building component ----------------------------------------------------------------
    O_b = df["n_bld"].to_numpy()
    MS = (p.kappa_b * df["U_bld"].to_numpy())[None, :] * _lognoise(rng, p.sigma_b, (S, n))
    dB = np.ones((S, n), bool) & (MS > 0.5)
    b = np.where(dB, 1.0 - np.minimum(1.0, O_b[None, :] / np.maximum(MS, 1e-9)), 0.0)

    # ---- POI component: HIFLD half ---------------------------------------------------------
    gs, ds = [], []
    for k, O_col, alpha, alpha_x, r in [("fire", "n_fd", p.alpha_fire, p.alpha_fire_x, p.r_fire),
                                        ("ems", "n_ems", p.alpha_ems, p.alpha_ems_x, p.r_ems),
                                        ("sch", "n_sch", p.alpha_sch, p.alpha_sch_x, p.r_sch)]:
        O = df[O_col].to_numpy()
        X = df[f"X_{k}"].to_numpy().astype(int)
        lam_unseen = (mu[f"mu_{k}"].to_numpy() * (1 - r) / r * np.where(burned, p.burn_unseen, 1.0)
                      * fac_mult)
        if p.use_points and "pts" in df.attrs:
            H = (_point_counts(rng, df.attrs["pts"][k], alpha, alpha_x, p.sigma_loc, S)
                 + rng.poisson(np.broadcast_to(lam_unseen, (S, n))))
        else:
            H = (rng.binomial(np.broadcast_to(O.astype(int), (S, n)), alpha)
                 + rng.binomial(np.broadcast_to(X, (S, n)), alpha_x)
                 + rng.poisson(np.broadcast_to(lam_unseen, (S, n))))
        O = O[None, :]
        d = H >= 1
        gs.append(np.where(d, 1.0 - np.minimum(1.0, O / np.maximum(H, 1)), 0.0))
        ds.append(d)
    nd = ds[0].astype(int) + ds[1] + ds[2]
    dH = nd > 0
    h = np.where(dH, (gs[0] + gs[1] + gs[2]) / np.maximum(nd, 1), 0.0)

    # ---- POI component: CBP half -----------------------------------------------------------
    if p.use_cbp:
        CBP = (p.eps_cbp * df["pop"].to_numpy() * np.exp(p.gc_rural * gcol("g_rural")))[None, :] * _lognoise(rng, p.sigma_c, (S, n))
        dC = CBP >= 0.5
        c = np.where(dC, 1.0 - np.minimum(1.0, df["n_poi"].to_numpy()[None, :] / np.maximum(CBP, 1e-9)), 0.0)
    else:
        dC = np.zeros((S, n), bool)
        c = np.zeros((S, n))
    nP = dH.astype(int) + dC
    dP = nP > 0
    pg = np.where(dP, (h + c) / np.maximum(nP, 1), 0.0)

    # ---- composite: mean of the defined components ----------------------------------------
    nd_all = dT.astype(int) + dB + dP
    score = np.where(nd_all > 0, (t + b + pg) / np.maximum(nd_all, 1), 0.0)

    out = pd.DataFrame({
        "GEOID": df["GEOID"].to_numpy(),
        "coverage_gap_score": score.mean(0),
        # optional component columns: E[component written to the CSV] (0 when undefined)
        "transport_gap": t.mean(0), "building_gap": b.mean(0), "poi_gap": pg.mean(0),
    })
    if return_components:
        out["p_undef_any"] = (~(dT & dB & dP)).mean(0)
        out["p_def_T"] = dT.mean(0)
        out["t_if_def"] = t.sum(0) / np.maximum(dT.sum(0), 1)
    return out


# ----------------------------------------------------------------------------------------------
# public moments used for calibration (all published, none derived from withdrawn layers)
# ----------------------------------------------------------------------------------------------
# Official challenge page: share of tracts with >= 1 undefined component.
UNDEF = {"eastern-ok": 0.21, "northern-ca": 0.37, "south-central-tx": 0.28, "maricopa-az": 0.55}
# Discussion 34910 (public): mean coverage_gap_score, rural = RUCA >= 4, burned = MTBS ever.
#                (rural_burned, rural_unburned, urban_burned, urban_unburned)
CELL = {"eastern-ok": (0.176, 0.191, 0.176, 0.081), "maricopa-az": (0.122, 0.136, 0.096, 0.029),
        "northern-ca": (0.122, 0.120, 0.096, 0.039), "south-central-tx": (0.076, 0.111, 0.069, 0.040)}
# Same post: pooled component means (burned, unburned).
COMP = {"poi_gap": (0.146, 0.043), "transport_gap": (0.194, 0.104), "building_gap": (0.008, 0.005)}
# Same post: five Paradise (Butte Co.) tracts -> (coverage, road, poi).
PARADISE = {"06007002300": (0.252, 0.496, 0.250), "06007001602": (0.254, 0.254, 0.500),
            "06007001703": (0.191, 0.000, 0.381), "06007002200": (0.249, 0.246, 0.500),
            "06007001601": (0.323, 0.506, 0.375)}


def perturbed_targets(rng: np.random.Generator, rel_sd: float = 0.08) -> dict:
    """Jitter the public moments by their (rough) sampling noise, for posterior-style bagging."""
    j = lambda x: float(x * np.exp(rel_sd * rng.standard_normal()))
    return {"UNDEF": {k: j(v) for k, v in UNDEF.items()},
            "CELL": {k: tuple(j(x) for x in v) for k, v in CELL.items()},
            "COMP": {k: tuple(j(x) for x in v) for k, v in COMP.items()}}


# True target's public bias scorecard (the RMSE-0 leaderboard entries, averaged), if available.
SCORECARD_W = 0.0   # weight on log(pred ratio / true ratio); round5.py turns it on
_SC_GROUPS = {"urban_rural": "g_rural", "tribal_vs_nontribal": "g_tribal", "high_svi_vs_low_svi": "g_svi",
              "high_cvi_vs_low_cvi": "g_cvi", "drought_summer": "g_drs", "drought_winter": "g_drw",
              "wildfire": "g_wild", "heat_summer": "g_heat", "intersectional": "g_inter"}


def scorecard_truth() -> dict:
    import json
    if not os.path.exists("scorecards.json"):
        return {}
    t = json.load(open("scorecards.json"))["truth"]
    return {k: float(np.mean([x[k] for x in t])) for k in _SC_GROUPS}


def scorecard_ratios(df: pd.DataFrame, score: np.ndarray) -> dict:
    out = {}
    for k, g in _SC_GROUPS.items():
        m = df[g].to_numpy() == 1
        out[k] = score[m].mean() / max(score[~m].mean(), 1e-9)
    return out


def moment_residuals(df: pd.DataFrame, pred: pd.DataFrame, targets: dict | None = None) -> np.ndarray:
    tg = targets or {"UNDEF": UNDEF, "CELL": CELL, "COMP": COMP}
    UNDEF_, CELL_, COMP_ = tg["UNDEF"], tg["CELL"], tg["COMP"]
    res = []
    cal = df["region"].isin(CALIB_REGIONS).to_numpy()
    for r in CALIB_REGIONS:
        m = (df["region"] == r).to_numpy()
        res.append(3.0 * (pred.loc[m, "p_undef_any"].mean() - UNDEF_[r]))
        for (ru, bu), tgt in zip([(1, 1), (1, 0), (0, 1), (0, 0)], CELL_[r]):
            mm = m & (df["rural"] == ru).to_numpy() & (df["burned"] == bu).to_numpy()
            w = 2.0 if bu == 0 else 1.0                       # unburned cells are bigger and cleaner
            res.append(w * (pred.loc[mm, "coverage_gap_score"].mean() - tgt))
    for col, (tb, tu) in COMP_.items():
        bmask = df["burned"].to_numpy() == 1
        res.append(1.5 * (pred.loc[cal & bmask, col].mean() - tb))
        res.append(3.0 * (pred.loc[cal & ~bmask, col].mean() - tu))
    if SCORECARD_W > 0:
        truth = scorecard_truth()
        ours = scorecard_ratios(df, pred["coverage_gap_score"].to_numpy())
        res += [SCORECARD_W * np.log(ours[k] / truth[k]) for k in truth]
    idx = pd.Series(np.arange(len(df)), index=df["GEOID"].to_numpy())
    for g, (cs, rd, po) in PARADISE.items():
        if g in idx:
            i = idx[g]
            res += [0.3 * (pred["coverage_gap_score"].iat[i] - cs), 0.3 * (pred["transport_gap"].iat[i] - rd)]
    return np.asarray(res)


CALIB_FIELDS = ["m0", "m1", "m2", "m3", "beta_bnd", "sigma_t", "r_fire", "alpha_fire_x",
                "alpha_ems_x", "alpha_sch", "alpha_sch_x", "eps_cbp", "kappa_b"]
BOUNDS = {"m0": (1.0, 3.0), "m1": (1.0, 3.0), "m2": (1.0, 3.0), "m3": (1.0, 3.0),
          "beta_bnd": (0.0, 2.0), "sigma_t": (0.05, 0.8), "r_fire": (0.3, 0.99), "alpha_fire_x": (0.0, 1.0),
          "alpha_ems_x": (0.0, 0.6), "alpha_sch": (0.4, 1.0), "alpha_sch_x": (0.0, 0.6),
          "eps_cbp": (0.0005, 0.06), "kappa_b": (0.7, 1.10),
          "burn_unseen": (0.2, 30.0), "burn_road": (0.5, 2.0)}


def _vec_to_params(v, base: Params) -> Params:
    d = dict(zip(CALIB_FIELDS, v))
    return replace(base, m=(d["m0"], d["m1"], d["m2"], d["m3"]),
                   **{k: d[k] for k in CALIB_FIELDS if not k.startswith("m")})


def _params_to_vec(p: Params):
    return [*p.m] + [getattr(p, k) for k in CALIB_FIELDS[4:]]


def calibrate(df, mu, base: Params, n_sims: int = 96, max_nfev: int = 60, verbose: bool = True,
              targets: dict | None = None, seed: int = 123) -> Params:
    from scipy.optimize import least_squares
    lo = np.array([BOUNDS[k][0] for k in CALIB_FIELDS])
    hi = np.array([BOUNDS[k][1] for k in CALIB_FIELDS])
    x0 = np.clip(_params_to_vec(base), lo + 1e-3, hi - 1e-3)

    def f(v):
        p = _vec_to_params(v, base)
        pred = simulate(df, mu, p, n_sims=n_sims, seed=seed, return_components=True)
        return moment_residuals(df, pred, targets)

    r0 = f(x0)
    sol = least_squares(f, x0, bounds=(lo, hi), diff_step=0.05, max_nfev=max_nfev, x_scale="jac")
    if verbose:
        print(f"    moment loss {np.sum(r0 ** 2):.5f} -> {np.sum(sol.fun ** 2):.5f}")
        print("    " + ", ".join(f"{k}={v:.3f}" for k, v in zip(CALIB_FIELDS, sol.x)))
    return _vec_to_params(sol.x, base)


def moment_report(df, pred) -> pd.DataFrame:
    rows = []
    for r in CALIB_REGIONS:
        m = df["region"] == r
        row = {"region": r, "undef_pred": pred.loc[m, "p_undef_any"].mean(), "undef_true": UNDEF[r]}
        for (ru, bu), tgt, nm in zip([(1, 1), (1, 0), (0, 1), (0, 0)], CELL[r], ["RB", "RU", "UB", "UU"]):
            mm = m & (df["rural"] == ru) & (df["burned"] == bu)
            row[f"{nm}_pred"], row[f"{nm}_true"] = pred.loc[mm, "coverage_gap_score"].mean(), tgt
        rows.append(row)
    return pd.DataFrame(rows).round(3)
