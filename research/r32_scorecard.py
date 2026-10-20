"""R32 / T1: exact Bias Scorecard definitions and the joint update they imply.

Run from the repo root: OPENBLAS_NUM_THREADS=1 python research/r32_scorecard.py

1. Definitions. The public leaderboard API returns every scorecard ratio to 3
   decimals (research/r32/leaderboard_rubrics.json). Our own best entry (R29)
   is a known vector, so its 9 ratios test each candidate definition exactly.
2. Leaders. Ranks 1-4 score exactly 0 on the public rows yet show different
   ratios. The ratios are taken over all 9,794 rows, including the 415 ungraded
   Washington (WA) rows, so each leader's WA treatment is a latent variable.
3. Joint update. The smallest change to R29 that agrees with every scored
   moment and with the leaders' ratios, each leader's WA level being unknown.
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))
import solution as S
import feedback

OUT = ROOT / "research/r32"
BASE_FILE = "submission_r29.csv"
BASE_NAME = "r29_correlated_scorecard.csv"
# API keys of the 9 ratios, in our stratum order
API = dict(rural="urban_rural", tribal="tribal_vs_nontribal", svi="high_svi_vs_low_svi", cvi="high_cvi_vs_low_cvi",
           sdrought="drought_summer", wdrought="drought_winter", wildfire="wildfire", heat="heat_summer",
           hv="intersectional")
ROUND_SD = 0.0005 / np.sqrt(3)          # uniform rounding of a 3-decimal ratio


def strata_table(geoid):
    t = pd.concat([pd.read_parquet(f) for f in glob.glob(str(ROOT / "data/strata/*/*-strata-tract-table.parquet"))
                   if "national" not in f])
    return t.drop_duplicates("GEOID").set_index("GEOID").reindex(geoid.astype(str))


def scorecard_groups(geoid):
    """(high, low) row masks over all 9,794 rows. Reproduces our R23, R24 (2 dp) and
    R29 (3 dp) scorecards: median splits over all rows, rows missing the stratum
    variable excluded from both groups; rural = Census ur_class (not RUCA)."""
    t = strata_table(geoid)

    def split(col):
        x = t[col]
        return (x >= x.median()).to_numpy(), (x < x.median()).to_numpy()

    g = {"rural": (t.ur_class.eq("Rural").to_numpy(), t.ur_class.eq("Urban").to_numpy())}
    tribal = t.tribal_any.fillna(False).astype(bool).to_numpy()
    g["tribal"] = (tribal, ~tribal)
    for k, col in [("svi", "svi_overall"), ("cvi", "cvi_overall"), ("sdrought", "usdm_summer_dsci"),
                   ("wdrought", "usdm_winter_dsci"), ("wildfire", "usfs_WHP_mean"), ("heat", "epht_heat_days_summer")]:
        g[k] = split(col)
    # High Hazard + High Vulnerability: (heat or wildfire or summer drought high) and SVI high, vs every other row
    hv = (g["heat"][0] | g["wildfire"][0] | g["sdrought"][0]) & g["svi"][0]
    g["hv"] = (hv, ~hv)
    return g


def ratios(v, groups):
    return {k: float(v[h].mean() / v[l].mean()) for k, (h, l) in groups.items()}


def definition_check(lb, groups):
    """Calculated vs displayed ratios for our three scored files with known scorecards.
    The page shows the 3-dp API value rounded again to 2 dp (double rounding)."""
    shown = {"r23_ridge_eb_blend.csv": ([2.18, 2.70, 1.28, 1.49, 1.02, 1.43, 1.66, 0.61, 1.25], 2),
             "r24_scorecard_joint.csv": ([2.32, 2.85, 1.23, 1.54, 1.00, 1.35, 1.75, 0.63, 1.22], 2),
             BASE_NAME: ([2.337, 2.881, 1.238, 1.543, 1.000, 1.361, 1.763, 0.616, 1.225], 3)}
    rows = []
    for name, (values, dp) in shown.items():
        calc = ratios(lb.vec(name), groups)
        for (k, c), s in zip(calc.items(), values):
            disp = round(round(c, 3), 2) if dp == 2 else round(c, 3)
            rows.append(dict(file=name, stratum=k, calculated=c, displayed=s, reproduced=bool(abs(disp - s) < 1e-9)))
    return pd.DataFrame(rows)


def leaders(max_rank=4):
    data = json.loads((OUT / "leaderboard_rubrics.json").read_text())["entries"]
    return [e for e in data if e["rank"] <= max_rank]


def estimate(lb, v):
    """Projection estimate of the public RMSE (the solution.py estimator)."""
    g = lb.graded
    files = [f for f in lb.s if not f.startswith("probe_")]
    X = np.column_stack([np.ones(g.sum())] + [lb.vec(f)[g] for f in files])
    m = np.array([lb.ybar] + [(lb.AP(lb.vec(f)[g] ** 2) + lb.Z - lb.s[f] ** 2) / 2 for f in files])
    gram = (X * lb.pi[:, None]).T @ X / lb.pi.sum()
    gram += 1e-7 * np.trace(gram) / len(m) * np.eye(len(m))
    coef = np.linalg.solve(gram, (X * lb.pi[:, None]).T @ v[g] / lb.pi.sum())
    return float(np.sqrt(max(lb.AP(v[g] ** 2) - 2 * coef @ m + lb.Z, 0)))


def span_share(lb, d):
    """Share of a graded-row change's pi-weighted squared size inside the scored-file span."""
    g = lb.graded
    w = lb.pi / lb.pi.sum()
    files = [f for f in lb.s if not f.startswith("probe_")]
    X = np.column_stack([np.ones(g.sum())] + [lb.vec(f)[g] for f in files])
    gram = (X * w[:, None]).T @ X
    gram += 1e-7 * np.trace(gram) / X.shape[1] * np.eye(X.shape[1])
    inside = X @ np.linalg.solve(gram, (X * w[:, None]).T @ d)
    return float((w @ inside ** 2) / max(w @ d ** 2, 1e-30))


def joint_update(lb, base, parts, groups, entries, sd_level=0.5, sd_offset=0.03, sd_tract=0.03,
                 fixed_level=None, tau=None):
    """Minimum change to `base` (graded rows) agreeing with the scored moments and each
    entry's 9 ratios. Entry k's WA rows are (1 + c_k) * base_WA + b_k + e, with
    c_k ~ N(0, sd_level^2), b_k ~ N(0, sd_offset^2), e ~ N(0, sd_tract^2) per tract.
    `fixed_level` pins rank 1's c to a value (sensitivity scan). Returns the new
    vector and the posterior of every entry's (c, b)."""
    g = lb.graded
    Q, w, o, noise = parts["Q"], parts["w"], parts["o"], parts["noise"]
    k, ng = Q.shape[1], g.sum()
    w0 = base[~g]
    H, rhs, Rr, cols = [], [], [], []
    for j, e in enumerate(entries):
        for name, (hi, lo) in groups.items():
            r = e["rubric"][API[name]]
            h = np.where(hi, 1 / hi.sum(), 0.0) - np.where(lo, r / lo.sum(), 0.0)
            H.append(h)
            rhs.append(-h @ base)
            mean_lo = base[lo].mean()
            extra = (e["public_score"] * 3) ** 2 * np.sum(h[g] ** 2)   # non-exact graded rows
            Rr.append((ROUND_SD * mean_lo) ** 2 + extra)
            cols.append(j)
    H, cols = np.array(H), np.array(cols)
    nE = len(entries)
    # latent WA columns: level (x base_WA) and offset per entry
    L = np.zeros((len(H), 2 * nE))
    for i, j in enumerate(cols):
        L[i, 2 * j] = H[i, ~g] @ w0
        L[i, 2 * j + 1] = H[i, ~g].sum()
    prior_lat = np.tile([sd_level ** 2, sd_offset ** 2], nE)
    if fixed_level is not None:
        rhs = np.array(rhs) - L[:, 0] * fixed_level
        prior_lat[0] = 0.0
    rhs = np.asarray(rhs)
    R = np.zeros((k + len(H), k + len(H)))
    R[:k, :k] = noise
    Rw = np.zeros((len(H), len(H)))
    for j in range(nE):
        s = cols == j
        Hw = H[s][:, ~g]
        Rw[np.ix_(s, s)] = sd_tract ** 2 * Hw @ Hw.T
    R[k:, k:] = Rw + np.diag(Rr)
    Mg = np.vstack([(Q * w[:, None]).T, H[:, g]])
    Ml = np.vstack([np.zeros((k, 2 * nE)), L])
    obs = np.r_[o - (Q * w[:, None]).T @ base[g], rhs]
    MMg = Mg @ Mg.T
    C_lat = (Ml * prior_lat) @ Ml.T

    def nll(log_tau):
        Lc = cho_factor(np.exp(2 * log_tau) * MMg + C_lat + R, lower=True)
        return 0.5 * obs @ cho_solve(Lc, obs) + np.log(np.diag(Lc[0])).sum()

    if tau is None:
        tau = float(np.exp(minimize_scalar(nll, bounds=(-12, -2), method="bounded").x))
    C = tau ** 2 * MMg + C_lat + R
    a = np.linalg.solve(C, obs)
    delta = tau ** 2 * Mg.T @ a
    lat_mean = prior_lat * (Ml.T @ a)
    lat_cov = np.diag(prior_lat) - (Ml * prior_lat).T @ np.linalg.solve(C, Ml * prior_lat)
    out = base.copy()
    out[g] = base[g] + delta
    lat_sd = np.sqrt(np.clip(np.diag(lat_cov), 0, None))
    latent = {e["rank"]: dict(level=float(lat_mean[2 * j] + (fixed_level if (fixed_level is not None and j == 0) else 0)),
                              level_sd=float(lat_sd[2 * j]), offset=float(lat_mean[2 * j + 1]),
                              offset_sd=float(lat_sd[2 * j + 1]))
              for j, e in enumerate(entries)}
    return S.finish(out, lb.geoid), dict(tau=tau, nll=float(nll(np.log(tau))), latent=latent)


def report(lb, base, v, groups, target):
    g = lb.graded
    w = lb.pi / lb.pi.sum()
    d = (v - base)[g]
    rat = ratios(v, groups)
    return dict(est_public=estimate(lb, v), rms_change=float(np.sqrt(w @ d ** 2)), max_change=float(np.abs(d).max()),
                in_span=span_share(lb, d) if np.any(d) else float("nan"),
                ratios={k: round(x, 4) for k, x in rat.items()},
                ratio_gap={k: round(rat[k] - target[k], 4) for k in rat})


def forward_mae(lb, first="p12_01_route_ratio.csv"):
    """solution.validation_log on the current score record, without rewriting validation.csv."""
    g = lb.graded
    w = lb.pi / lb.pi.sum()
    files = [f for f in lb.s if not f.startswith(("probe_", "e06"))]
    vec = {f: lb.vec(f)[g] for f in files}
    mom = {f: (lb.AP(vec[f] ** 2) + lb.Z - lb.s[f] ** 2) / 2 for f in files}

    def predict(target, use):
        X = np.column_stack([np.ones(g.sum())] + [vec[f] for f in use])
        m = np.array([lb.ybar] + [mom[f] for f in use])
        gram = (X * w[:, None]).T @ X
        gram += 1e-7 * np.trace(gram) / len(m) * np.eye(len(m))
        return float(np.sqrt(max(lb.AP(vec[target] ** 2) - 2 * np.linalg.solve(gram, (X * w[:, None]).T @ vec[target]) @ m + lb.Z, 0)))

    err = [predict(f, files[:i]) - lb.s[f] for i, f in enumerate(files) if i >= files.index(first)]
    return float(np.mean(np.abs(err))), len(err)


CANDIDATE = "T1c_leaders1-4_WAlatent"
CANDIDATE_FILE = ROOT / "submission_r32_t1.csv"


def main():
    OUT.mkdir(exist_ok=True)
    lb = feedback.load()
    groups = scorecard_groups(lb.geoid)
    check = definition_check(lb, groups)
    check.to_csv(OUT / "definition_check.csv", index=False)
    print(f"Definitions: {check.reproduced.sum()}/{len(check)} displayed ratios reproduced (R23, R24 2 dp; R29 3 dp)")
    print(check[~check.reproduced].to_string() if (~check.reproduced).any() else "  all reproduced")

    base = lb.vec(BASE_NAME).copy()
    g = lb.graded
    _, _, _, parts = S.eb_posterior(lb, base[g])
    lead = leaders(4)
    target = {k: lead[0]["rubric"][API[k]] for k in groups}
    results = {"R29 (base)": report(lb, base, base, groups, target)}
    runs = {
        # rank 1 only, its WA taken as ours (the R24-R29 assumption), exact definitions and 3-dp targets
        "T1a_rank1_WAfixed": dict(entries=lead[:1], sd_level=0.0, sd_offset=0.0, sd_tract=0.0624),
        # rank 1 only, its WA level unknown
        "T1b_rank1_WAlatent": dict(entries=lead[:1]),
        # all four zero-score leaders, each with its own unknown WA level and offset
        "T1c_leaders1-4_WAlatent": dict(entries=lead),
    }
    vectors = {}
    for tag, kw in runs.items():
        v, info = joint_update(lb, base, parts, groups, **kw)
        vectors[tag] = v
        results[tag] = report(lb, base, v, groups, target) | info
    # sensitivity: rank 1's WA level pinned (c = level - 1), leaders 1-4 jointly
    scan = []
    for c in [-0.6, -0.3, 0.0, 0.3, 0.6]:
        v, info = joint_update(lb, base, parts, groups, lead, fixed_level=c)
        rep = report(lb, base, v, groups, target)
        scan.append(dict(rank1_wa_level=1 + c, est_public=rep["est_public"], rms_change=rep["rms_change"],
                         in_span=rep["in_span"], nll=info["nll"], **{f"r_{k}": x for k, x in rep["ratios"].items()}))
    pd.DataFrame(scan).to_csv(OUT / "wa_level_scan.csv", index=False)
    print(pd.DataFrame(scan).round(6).to_string())
    for tag, v in vectors.items():
        np.save(OUT / f"{tag}.npy", v)
    (OUT / "summary.json").write_text(json.dumps(results, indent=2))
    for tag, r in results.items():
        lat = r.get("latent", {})
        print(f"{tag:26s} est {r['est_public']:.9f}  rms {r['rms_change']:.5f}  in-span {r['in_span']:.2f}  "
              f"tau {r.get('tau', float('nan')):.4f}")
        print("   gap vs rank 1: " + " ".join(f"{k} {x:+.4f}" for k, x in r["ratio_gap"].items()))
        for rk, p in lat.items():
            print(f"   rank {rk}: WA level {1 + p['level']:.2f} +- {p['level_sd']:.2f}, offset {p['offset']:+.3f} +- {p['offset_sd']:.3f}")

    v = vectors[CANDIDATE]
    out = pd.DataFrame({"GEOID": lb.geoid.astype(str), "coverage_gap_score": v})
    anchors = sum(float(out.loc[out.GEOID.eq(k), "coverage_gap_score"].iloc[0]) == x for k, x in S.PUBLISHED.items())
    assert len(out) == 9794 and out.GEOID.is_unique and out.coverage_gap_score.between(0, 1).all() and anchors == 5
    assert np.array_equal(v[~g], base[~g])                  # WA rows untouched
    out.to_csv(CANDIDATE_FILE, index=False)
    mae, n = forward_mae(lb)
    r = results[CANDIDATE]
    print(f"\n{CANDIDATE_FILE.name}: est public {r['est_public']:.9f} (R29 est {results['R29 (base)']['est_public']:.9f}, "
          f"actual 0.062094756); RMS change {r['rms_change']:.5f}; inside scored span {r['in_span']:.0%}, "
          f"outside {1 - r['in_span']:.0%}; forward MAE {mae:.6f} over {n} files")
    print("  ratios vs rank 1: " + " ".join(f"{k} {r['ratios'][k]:.3f}/{target[k]:.3f}" for k in groups))


if __name__ == "__main__":
    main()
