"""Monte-Carlo noise in physics probe directions: build each direction with two seeds (512 draws each) and
with 8 seeds averaged, after removing the scored files. Correlation between seeds = share that is mechanism.
    python research/mc_noise.py"""
import sys, numpy as np, pandas as pd
from dataclasses import replace
sys.path.insert(0, ".")
import solution as S
df = S.load(); mu = S.prior_rates(df)
lb = S.Leaderboard(df); lb.public_subset(); g = lb.graded; w = np.sqrt(lb.pi / lb.pi.sum())
files = [f for f in lb.s if not f.startswith("probe_")]
Q, _ = np.linalg.qr(np.column_stack([np.ones(g.sum())] + [lb.vec(f)[g] for f in files]) * w[:, None])
frame = S.road_frame(df); sets = {k: S.Params(**v) for k, v in S.PHYSICS["r10a"].items()}
def run(sets_, seed):
    v = S.simulate(frame, mu, sets_["global"], 512, seed)
    for r in S.CALIB_REGIONS:
        m = (frame["region"] == r).to_numpy(); v[m] = S.simulate(frame, mu, sets_[r], 512, seed)[m]
    return 0.8 * v
def direction(name, seeds):
    pert = {k: S.PHYSICS_DIRECTIONS[name](p) for k, p in sets.items()}
    d = np.mean([run(pert, s) - run(sets, s) for s in seeds], axis=0)[g] * w
    return d - Q @ (Q.T @ d)
rows = []
for name in ["fm_s1200", "tiger_noise", "unrouted_s1200", "burned_unseen", "cbp_noise", "ms_vs_usfs_buildings"]:
    a, b = direction(name, [77]), direction(name, [78])
    big = direction(name, range(100, 108))
    corr = float(a @ b / np.sqrt((a @ a) * (b @ b)))
    # correlation of a 512-draw direction with the 4096-draw one estimates how much of it is mechanism
    c512 = float(a @ big / np.sqrt((a @ a) * (big @ big)))
    rows.append((name, corr, c512, corr / (1 + corr) * 2 if corr > 0 else 0))
    print(f"  {name:22s} seed-vs-seed corr {corr:.3f}; 512 vs 4096 draws corr {c512:.3f}", flush=True)
t = pd.DataFrame(rows, columns=["direction", "corr_seed77_vs_78", "corr_512_vs_4096", "signal_share_est"])
t.round(4).to_csv("research/mc_noise.csv", index=False)
