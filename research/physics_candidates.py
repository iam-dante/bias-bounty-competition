"""Rank unprobed physics mechanisms as probe candidates: how predictions move when one parameter moves
(r10a regional specialists, common random numbers), with the scored files and measured probes removed.
    python research/physics_candidates.py"""
import sys, time, numpy as np, pandas as pd
from dataclasses import replace
sys.path.insert(0, ".")
import solution as S
df = S.load(); mu = S.prior_rates(df)
lb = S.Leaderboard(df); lb.public_subset(); g = lb.graded; pi = lb.pi; W = pi.sum()
MECH = {   # name: (perturbation, one-sided physical sign known?)
    "ms_vs_usfs_buildings":  (lambda p: replace(p, kappa_b=p.kappa_b * 1.05), False),
    "building_count_noise":  (lambda p: replace(p, sigma_b=p.sigma_b + 0.05), False),
    "fire_unseen_share":     (lambda p: replace(p, r_fire=max(0.3, p.r_fire - 0.05)), False),
    "fire_weak_evidence":    (lambda p: replace(p, alpha_fire_x=min(1, p.alpha_fire_x + 0.1)), False),
    "school_unseen_share":   (lambda p: replace(p, r_sch=max(0.3, p.r_sch - 0.03)), False),
    "ems_unseen_share":      (lambda p: replace(p, r_ems=max(0.3, p.r_ems - 0.05)), False),
    "cbp_per_person":        (lambda p: replace(p, eps_cbp=p.eps_cbp * 1.2), False),
    "cbp_noise":             (lambda p: replace(p, sigma_c=p.sigma_c + 0.15), False),
    "cbp_rural":             (lambda p: replace(p, gc_rural=p.gc_rural + 0.3), False),
    "burned_unseen":         (lambda p: replace(p, burn_unseen=p.burn_unseen * 1.3), False),
    "burned_road":           (lambda p: replace(p, burn_road=p.burn_road * 1.1), False),
    "fac_rural":             (lambda p: replace(p, gf_rural=p.gf_rural + 0.2), False),
    "fac_tribal":            (lambda p: replace(p, gf_tribal=p.gf_tribal + 0.3), False),
    "road_rural":            (lambda p: replace(p, gr_rural=p.gr_rural + 0.1), False),
    "road_heat":             (lambda p: replace(p, gr_heat=p.gr_heat + 0.1), False),
    "surprise_highway":      (lambda p: replace(p, p_def0=p.p_def0 + 0.02), True),
    "fm_as_s1200":           (lambda p: replace(p, w_fm=p.w_fm + 0.1), True),
}
frame = S.road_frame(df)
sets = {k: S.Params(**v) for k, v in S.PHYSICS["r10a"].items()}
def run(sets_):
    v = S.simulate(frame, mu, sets_["global"], 512, 77)
    for r in S.CALIB_REGIONS:
        mm = (frame["region"] == r).to_numpy(); v[mm] = S.simulate(frame, mu, sets_[r], 512, 77)[mm]
    return 0.8 * v
t0 = time.time(); base = run(sets)
files = [f for f in lb.s if not f.startswith("probe_")]
A = np.column_stack([np.ones(g.sum())] + [lb.vec(f)[g] for f in files]) * np.sqrt(pi / W)[:, None]
Q, _ = np.linalg.qr(A)
rows, D = [], {}
for name, (pert, onesided) in MECH.items():
    d = (run({k: pert(p) for k, p in sets.items()}) - base)[g] * np.sqrt(pi / W)
    raw = float(d @ d)
    if raw < 1e-14: rows.append((name, 0, 0, onesided)); continue
    r = d - Q @ (Q.T @ d); D[name] = r
    rows.append((name, np.sqrt(raw), float(r @ r) / raw, onesided))
    print(f"  {name:22s} size {np.sqrt(raw):.5f} unexplained {float(r @ r) / raw:.3f} ({time.time() - t0:.0f}s)", flush=True)
t = pd.DataFrame(rows, columns=["mechanism", "size", "unexplained_share", "one_sided"]).sort_values("unexplained_share", ascending=False)
names = [n for n in t.mechanism if n in D]
C = np.array([[D[a] @ D[b] / np.sqrt((D[a] @ D[a]) * (D[b] @ D[b])) for b in names] for a in names])
t["max_corr_with_other"] = [np.max(np.abs(np.delete(C[i], i))) if n in D else np.nan for i, n in enumerate(names)] + [np.nan] * (len(t) - len(names))
t.round(4).to_csv("research/physics_candidates.csv", index=False)
print(t.round(3).to_string(index=False))
