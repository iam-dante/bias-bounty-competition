"""
GMM regional specialists: round 10's one-parameter-set-per-region idea, but each region is fitted on
its own published moments AND the global leaderboard moments (the other regions held at the global
GMM fit), ridge-pulled toward the global GMM parameters. One coordinate-descent pass.

    python gmm_regional.py <name> <frame> <global params json> <lambda>
"""
import json, sys
from dataclasses import asdict, replace
import numpy as np
from scipy.optimize import least_squares
import harness as H, gmm
from round8 import region_residuals

M = H.M
REG = {"beta_bnd": 0.5, "sigma_t": 0.2, "r_fire": 0.1, "alpha_fire_x": 0.2, "gf_rural": 0.5, "gr_rural": 0.2, "gr_wild": 0.2, "gr_drw": 0.2}
LO = np.array([0.5, 0.0, 0.05, 0.3, 0.0, -2.5, -1.0, -1.0, -1.0]); HI = np.array([3.0, 2.0, 1.2, 0.99, 1.0, 2.5, 1.0, 1.0, 1.0])

if __name__ == "__main__":
    name, frame, pjson, lam = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
    D = H.data(); d, mu = D[frame], D["mu"]
    P = json.load(open(pjson))["params"]; pg = M.Params(**{**P, "m": tuple(P["m"])})
    lbr = gmm.LBResid()
    cur = M.simulate(d, mu, pg, n_sims=96, seed=123, return_components=True)
    specs = {"global": pg}
    for i, region in enumerate(M.CALIB_REGIONS):
        mask = (d["region"] == region).to_numpy()
        names = ["m"] + list(REG)
        x0 = np.array([pg.m[i]] + [getattr(pg, k) for k in REG]); scale = np.array([0.3] + list(REG.values()))

        def to_p(x):
            m = list(pg.m); m[i] = x[0]
            return replace(pg, m=tuple(m), **dict(zip(names[1:], x[1:])))

        def f(x):
            pr = M.simulate(d, mu, to_p(x), n_sims=96, seed=123, return_components=True)
            full = cur.copy(); full.loc[mask] = pr.loc[mask].to_numpy()
            return np.r_[region_residuals(d, pr, region), lam * lbr(full["coverage_gap_score"].to_numpy()), 0.02 * (x - x0) / scale]

        r0 = f(x0)
        sol = least_squares(f, np.clip(x0, LO + 1e-3, HI - 1e-3), bounds=(LO, HI), diff_step=0.05, max_nfev=30, x_scale="jac")
        nreg = len(r0) - len(x0) - len(lbr.c)
        print(f"  {region:17s} region loss {np.sum(r0[:nreg]**2):.5f} -> {np.sum(sol.fun[:nreg]**2):.5f}  LB ||Pe|| {np.linalg.norm(r0[nreg:nreg+len(lbr.c)])/lam:.5f} -> {np.linalg.norm(sol.fun[nreg:nreg+len(lbr.c)])/lam:.5f}  "
              + ", ".join(f"{k}={v:+.2f}" for k, v in zip(names, sol.x)), flush=True)
        specs[region] = to_p(sol.x)
        pr = M.simulate(d, mu, specs[region], n_sims=96, seed=123, return_components=True)
        cur.loc[mask] = pr.loc[mask].to_numpy()
    out = H.sim_regional(d, mu, specs, n_sims=512, comps=True)
    v = out["coverage_gap_score"].to_numpy()
    H.write(D["df"]["GEOID"].to_numpy(), v, name, published=False)
    json.dump({k: asdict(p) for k, p in specs.items()}, open(f"exp11/params_{name}.json", "w"), indent=1)
    met = lbr.lb.validate(v); met["moment_loss"] = H.moment_loss(d, out)
    print({k: round(x, 6) for k, x in met.items()}, flush=True)
