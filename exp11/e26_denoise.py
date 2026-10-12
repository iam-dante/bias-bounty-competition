"""E26: Monte-Carlo denoising of blend inputs. Each single model is a 512-draw posterior mean; its MC
noise (~0.003 rms per model) is independent of the target, so replacing a scored input by a
high-precision re-simulation from the SAME saved parameters removes noise without changing what
the leaderboard measured. First check that the saved parameters reproduce the scored file."""
import json, numpy as np, pandas as pd, harness as H
M = H.M
D = H.data(); mu = D["mu"]
SEEDS = range(3000, 3008)   # 8 x 512 = 4096 draws


def regional_specs(path, key):
    P = json.load(open(path))[key]
    return {r: M.Params(**{**P[r], "m": tuple(P[r]["m"])}) for r in P}


jobs = {"r10a": (D["noleak"], regional_specs("submissions/params_round10.json", "r10a_regional_noleak")),
        "r10b": (D["inter"], regional_specs("submissions/params_round10.json", "r10b_regional_interstate")),
        "r9a": (D["noleak"], {"global": H.params("submissions/params_round9.json", "r9a_routefix_noleak")}),
        "r9b": (D["inter"], {"global": H.params("submissions/params_round9.json", "r9b_routefix_interstate")})}
hi = {}
for k, (d, specs) in jobs.items():
    sim = (lambda s, n: H.sim_regional(d, mu, specs, n_sims=n, seed=s)) if len(specs) > 1 else \
          (lambda s, n: M.simulate(d, mu, specs["global"], n_sims=n, seed=s))
    rep = sim(2026, 512)["coverage_gap_score"].to_numpy()
    sub = {"r10a": "round10/r10a_regional_noleak", "r10b": "round10/r10b_regional_interstate",
           "r9a": "round9/r9a_routefix_noleak", "r9b": "round9/r9b_routefix_interstate"}[k]
    orig = pd.read_csv(f"submissions/{sub}.csv").coverage_gap_score.to_numpy()
    print(f"{k}: reproduction max |diff| vs saved file {np.abs(np.clip(rep,0,1).round(6) - orig).max():.2e}", flush=True)
    hi[k] = np.mean([sim(s, 512)["coverage_gap_score"].to_numpy() for s in SEEDS], axis=0)
    print(f"   4096-draw vs 512-draw rms {np.sqrt(np.mean((hi[k] - rep) ** 2)):.5f}", flush=True)
geo = D["df"]["GEOID"].to_numpy()
r10d_hi = 0.009 + 0.8 * (hi["r10a"] + hi["r10b"]) / 2
r9d_hi = 0.009 + 0.8 * (hi["r9a"] + hi["r9b"]) / 2
H.write(geo, r10d_hi, "e26_r10d_hiprec", published=False)
H.write(geo, r9d_hi, "e26_r9d_hiprec", published=False)
lb = H.LBc(H.BASIS + ["stack3_scorecard_groups.csv", "stack4_with_scorecard.csv", "stack4_no_scorecard.csv"])
w, _ = lb.blend()
g = lb.graded
r10d = lb.vec("r10d_regional_affine.csv"); r9d = lb.vec("r9d_routefix_affine.csv")
noise = w["r10d_regional_affine.csv"] * (r10d - r10d_hi) + w["r9d_routefix_affine.csv"] * (r9d - r9d_hi)
gain = np.mean(noise[g] ** 2)
print(f"blend weights r10d {w['r10d_regional_affine.csv']:+.3f}, r9d {w['r9d_routefix_affine.csv']:+.3f}; removed noise rms {np.sqrt(gain):.5f}; expected MSE gain {gain:.2e}")
H.log("E26", "MC denoising of r10d/r9d blend inputs (4096 draws)", "model-precision", {"removed_noise_rms": np.sqrt(gain), "expected_mse_gain": gain})
