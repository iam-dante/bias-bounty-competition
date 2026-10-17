"""What the leaderboard has taught us, in physical terms: regress r19b's correction (r19b - r12) on the
physics mechanism directions (each = response to one parameter step), and read each coefficient as an
implied parameter change.  python research/correction_anatomy.py"""
import sys, time, numpy as np, pandas as pd
sys.path.insert(0, ".")
import solution as S
t0 = time.time()
df = S.load(); mu = S.prior_rates(df); lb = S.Leaderboard(df); lb.public_subset(); g = lb.graded
w = lb.pi / lb.pi.sum()
corr = (pd.read_csv("submission.csv")["coverage_gap_score"].to_numpy() - lb.vec(S.BASE))[g]
step = {"tiger_multiplicity": "m x1.1", "boundary_leak": "beta_bnd +0.2", "tiger_noise": "sigma_t -0.1",
        "county_s1200": "w_cty +0.5", "unrouted_s1200": "w_unr +0.2", "tertiary_s1200": "w_tert +0.1",
        "fm_s1200": "w_fm +0.1", "burned_unseen": "burn_unseen x1.3", "fac_tribal": "gf_tribal +0.3",
        "cbp_noise": "sigma_c +0.15", "ms_vs_usfs_buildings": "kappa_b x1.05", "building_count_noise": "sigma_b +0.05",
        "fire_unseen_share": "r_fire -0.05", "fire_weak_evidence": "alpha_fire_x +0.1", "school_unseen_share": "r_sch -0.03",
        "ems_unseen_share": "r_ems -0.05", "cbp_per_person": "eps_cbp x1.2", "burned_road": "burn_road x1.1",
        "fac_rural": "gf_rural +0.2", "road_rural": "gr_rural +0.1", "road_heat": "gr_heat +0.1", "surprise_highway": "p_def0 +0.02"}
D = {}
for k in step:
    D[k] = S.physics_direction(df, mu, k, seeds=range(77, 79))[g]
    print(f"  {k:22s} built ({time.time() - t0:.0f}s)", flush=True)
X = np.column_stack([np.ones(g.sum())] + [D[k] for k in step])
Xw = X * np.sqrt(w)[:, None]; yw = corr * np.sqrt(w)
beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
fit = X @ beta
r2 = 1 - np.sum(w * (corr - fit) ** 2) / np.sum(w * (corr - np.sum(w * corr)) ** 2)
rows = []
for i, k in enumerate(step, 1):
    d = D[k]; c = np.sum(w * (d - np.sum(w * d)) * (corr - np.sum(w * corr))) / np.sqrt(np.sum(w * (d - np.sum(w * d)) ** 2) * np.sum(w * (corr - np.sum(w * corr)) ** 2))
    rows.append((k, step[k], c, beta[i]))
t = pd.DataFrame(rows, columns=["mechanism", "step", "corr_with_correction", "coef_in_steps"]).sort_values("corr_with_correction", key=abs, ascending=False)
t.round(4).to_csv("research/correction_anatomy.csv", index=False)
print(f"\ncorrection rms {np.sqrt(np.sum(w * corr ** 2)):.5f}; R^2 on 22 mechanisms: {r2:.3f}")
print(t.round(3).to_string(index=False))
