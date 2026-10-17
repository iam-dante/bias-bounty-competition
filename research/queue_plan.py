"""Plan the probe queue for the rest of the competition: build every candidate direction (physics at
8 x 512 draws, regional splits, features), remove the scored files AND the pending probes in next/, and order
the candidates greedily by the share of new variance each still adds.
    python research/queue_plan.py"""
import glob, os, sys, time, numpy as np, pandas as pd
sys.path.insert(0, ".")
import solution as S
t0 = time.time()
df = S.load(); mu = S.prior_rates(df); lb = S.Leaderboard(df); lb.public_subset(); g = lb.graded; w = np.sqrt(lb.pi / lb.pi.sum())
files = [f for f in lb.s if not f.startswith("probe_")]
cols = [np.ones(g.sum())] + [lb.vec(f)[g] for f in files]
pending = sorted(glob.glob("next/p*.csv"))
cols += [pd.read_csv(f)["coverage_gap_score"].to_numpy()[g] for f in pending]
Q, _ = np.linalg.qr(np.column_stack(cols) * w[:, None])
done = {S.probe_direction_of(f) for f in lb.s} | {S.probe_direction_of(os.path.basename(f)) for f in pending}
physics = ["building_count_noise", "fire_unseen_share", "fire_weak_evidence", "school_unseen_share", "ems_unseen_share",
           "cbp_per_person", "burned_road", "fac_rural", "road_rural", "road_heat", "surprise_highway",
           "boundary_leak_clean", "county_s1200_clean", "tertiary_s1200_clean"]
regional = [f"unrouted_s1200_clean_{r}" for r in ["maricopa-az", "south-central-tx", "northern-ca", "eastern-ok"]] + \
           [f"tiger_noise_clean_{r}" for r in ["maricopa-az", "south-central-tx", "northern-ca", "eastern-ok"]] + \
           [f"poi_per_person_{r}" for r in ["maricopa-az", "south-central-tx", "northern-ca", "eastern-ok"]]
features = ["heat_wbgt", "multiref_share", "log_route_length", "route_ratio_northern-ca", "boundary_share",
            "fire_vs_expected_south-central-tx", "building_coverage_maricopa-az", "log_pop_density_south-central-tx"]
names = [n for n in physics + regional + features if n not in done]
F = S.probe_features(df, mu, names)
print(f"built {len(names)} candidate directions in {time.time() - t0:.0f}s", flush=True)
R = {}
for n in names:
    x = F[n].to_numpy(float)[g] * w
    if np.linalg.norm(x) < 1e-12: continue
    r = x - Q @ (Q.T @ x); R[n] = (r, float(r @ r) / float(x @ x))
# rank by ABSOLUTE new share: residual relative to the candidate's own size (normalising the residual itself
# would rank already-measured directions first, on numerical residue); drop those < 20 % unexplained
norm0 = {n: np.linalg.norm(F[n].to_numpy(float)[g] * w) for n in R}
order, rows, Rw = [], [], {n: r / norm0[n] for n, (r, share) in R.items() if share >= 0.2}
left = dict(Rw)
while left:
    norms = {n: float(v @ v) for n, v in left.items()}
    n = max(norms, key=norms.get)
    rows.append((len(order) + 1, n, R[n][1], norms[n])); order.append(n)
    q = left.pop(n) / np.sqrt(norms[n])
    left = {k: v - q * (q @ v) for k, v in left.items()}
t = pd.DataFrame(rows, columns=["rank", "direction", "unexplained_share", "absolute_new_share_at_pick"])
t.round(4).to_csv("research/queue_plan.csv", index=False)
print(t.round(3).to_string(index=False))
