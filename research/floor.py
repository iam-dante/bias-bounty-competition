"""Irreducible error under the physics model (python research/floor.py [sigma_t_shift]): for each tract, the spread of the official composite across
Monte-Carlo draws of the withdrawn reference counts. sqrt(mean variance) is the RMSE that even a perfect
E[score | permitted data] would have if the model were right; the gap to the leaderboard is model error.
    python research/floor.py"""
import inspect, sys, textwrap, numpy as np, pandas as pd
sys.path.insert(0, ".")
import solution as S
src = textwrap.dedent(inspect.getsource(S.simulate)).replace("def simulate(", "def simulate_draws(")
src = src.replace(".mean(0)\n", "\n")      # keep every draw instead of the mean
ns = dict(vars(S)); exec(src, ns); simulate_draws = ns["simulate_draws"]
df = S.load(); mu = S.prior_rates(df)
from dataclasses import replace
shift = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
frame = S.road_frame(df); sets = {k: replace(S.Params(**v), sigma_t=max(0.02, S.Params(**v).sigma_t + shift)) for k, v in S.PHYSICS["r10a"].items()}
print(f"TIGER length noise sigma_t shifted by {shift:+.2f} (global {sets['global'].sigma_t:.3f})")
draws = []
for seed in range(4):
    v = simulate_draws(frame, mu, sets["global"], 512, 500 + seed)
    for r in S.CALIB_REGIONS:
        m = (frame["region"] == r).to_numpy(); v[:, m] = simulate_draws(frame, mu, sets[r], 512, 500 + seed)[:, m]
    draws.append(v)
D = np.concatenate(draws)                  # (2048, tracts)
g = (df["region"] != "eastern-wa").to_numpy()
var = D[:, g].var(0); mean = D[:, g].mean(0)
floor = float(np.sqrt(var.mean()))
by_region = pd.Series(var, index=df.loc[g, "region"].values).groupby(level=0).mean() ** 0.5
print(f"model-implied irreducible RMSE (graded tracts, 2048 draws): {floor:.5f}")
print("by region:", by_region.round(5).to_dict())
print(f"r19b expected public RMSE 0.062470 -> model error part: sqrt(0.062470^2 - floor^2) = {np.sqrt(max(0.062470**2 - floor**2, 0)):.5f}")
if shift == 0: pd.DataFrame({"region": list(by_region.index) + ["all"], "irreducible_rmse": list(by_region.values) + [floor]}).round(6).to_csv("research/floor.csv", index=False)
