"""E27-E30: final candidates, each scored by the validators.
E27  blend11 + MC denoising                                   (in-span: exact public estimate)
E28  E27 + sd-weighted cell anchoring + Paradise              -> candidate B (no unscored bets)
E29  E28 + novel direction of the round-11 GMM physics, rho 0.06 -> candidate A
E30  published-facts-only hedge: r10c raw + anchoring (no leaderboard-derived constants)"""
import numpy as np, pandas as pd, harness as H, final as F, synth as S

met = eval(open("exp11/log_e24.txt").read().strip().splitlines()[-1].replace("np.float64", ""))
H.log("E24", "GMM regional specialists (Interstate), published + LB moments", "model-gmm", met)
D = H.data(); df = D["df"]; geo = df["GEOID"].to_numpy()
rd = lambda n: pd.read_csv(f"submissions/round11/{n}.csv")["coverage_gap_score"].to_numpy()
gmm = {n: rd(n) for n in ["e25_gmm_noleak_2048", "e16_gmm_inter_l10", "e23_gmm_regional_noleak", "e24_gmm_regional_inter"]}

c27, i27, lb = F.build(anchor=False, denoise=True)
c28, _, _ = F.build(anchor=True, denoise=True)
c29, i29, _ = F.build(new_models=gmm, rho=0.06, anchor=True, denoise=True)
r10c = pd.read_csv("submissions/round10/r10c_regional_ensemble.csv")["coverage_gap_score"].to_numpy()
c30 = S.anchor_sd(r10c, S.cell_index(df), S.published_targets(df), lb.graded, S.eps_draws().std(0))
print("novel-direction info:", {k: round(float(v), 6) for k, v in i29.items()})
out = {}
for eid, nm, v, cat in [("E27", "r11_blend_denoised", c27, "blend"), ("E28", "r11_B_blend_anchored", c28, "final"),
                        ("E29", "r11_A_blend_anchored_gmm", c29, "final"), ("E30", "r11_hedge_published_only", c30, "hedge")]:
    v = H.write(geo, v, nm, published=True)
    m = F.report(nm, v, lb)
    m["exact_public_if_in_span"] = lb.public_score(v)[0]
    out[nm] = v
    H.log(eid, nm, cat, m)
s7 = lb.vec("stack7_r10.csv")
for nm, v in out.items():
    print(f"{nm:28s} rms diff vs stack7 {np.sqrt(np.mean((v - s7)[lb.graded] ** 2)):.5f}   mean {v[lb.graded].mean():.5f}")
print(f"A vs B rms diff {np.sqrt(np.mean((out['r11_A_blend_anchored_gmm'] - out['r11_B_blend_anchored'])[lb.graded] ** 2)):.5f}")
