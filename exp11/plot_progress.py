"""Progress plot for round 11: validation RMSE of every candidate file, synthetic private gains of
every procedure, and validator accuracy. Reads exp11/log.csv -> exp11/round11_progress.png"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
L = pd.read_csv("exp11/log.csv")
L["n"] = L["id"].str[1:].astype(int)
L = L.sort_values("n")
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(3, 1, figsize=(10, 12.5), gridspec_kw={"height_ratios": [1.5, 1.1, 0.6]})

# A: validation RMSE (public-subset-corrected, rho-prior for unseen directions) of each candidate file
ax = axes[0]
c = L[L["lb_prior_rmse"].notna()].copy()
ax.axhline(0.064214, color=MUTED, lw=1, ls="--")
ax.text(4, 0.064214 + 0.00003, "stack7: current best, actual public 0.064214", color=MUTED, fontsize=8, va="bottom")
TOP = 0.0662
off = c["lb_prior_rmse"] > TOP
ax.scatter(c.loc[~off, "n"], c.loc[~off, "lb_prior_rmse"], s=46, color=BLUE, zorder=3, edgecolor="white", linewidth=1.5, label="candidate file (validation estimate)")
for _, r in c[off].iterrows():
    ax.annotate(f"{r['id']} off scale ({r['lb_prior_rmse']:.4f})", (r["n"], TOP), xytext=(r["n"] - 3.2, TOP - 0.0001), ha="center", fontsize=7, color=MUTED,
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.8))
best = c["lb_prior_rmse"].cummin()
ax.step(c["n"], best, where="post", color=ORANGE, lw=2, label="running best")
for _, r in c[~off].iterrows():
    ax.annotate(r["id"], (r["n"], r["lb_prior_rmse"]), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=7, color=MUTED)
ax.set_ylabel("validation RMSE (public-subset estimate)")
ax.set_title("A. Candidate submissions: estimated public RMSE by experiment (lower is better)", loc="left", color=INK, fontsize=10.5)
ax.set_ylim(0.0636, TOP)
ax.grid(axis="y", color=GRID, lw=0.8); ax.legend(frameon=False, loc="upper left", fontsize=8)
ax.set_xlabel("experiment")

# B: synthetic private-RMSE change of each procedure vs the current blend recipe
ax = axes[1]
proc = {"E06": ("ridge 1e-2 on blend", "synth_private_rmse", "E05"), "E08": ("exact public-subset moments", "synth_private_rmse", "E05"),
        "E07": ("cell anchoring (additive)", "synth_private_rmse", "E05")}
rows = []
base = L.set_index("id")
b5 = base.loc["E05", "synth_private_rmse"]
for k in ["E06", "E08", "E07"]:
    rows.append((proc[k][0], base.loc[k, "synth_private_rmse"] - b5))
for k, lab in [("E10", "anchoring (multiplicative)"), ("E09", "anchoring (variance-weighted)"), ("E11", "anchoring (half variance, half additive)")]:
    rows.append((lab, -base.loc[k, "synth_private_gain"]))
rows.append(("blend + cell columns + anchor", -base.loc["E12", "synth_private_gain"]))
rows.append(("blend + sd-weighted anchoring", -base.loc["E12", "synth_private_gain_blend_sd_anchor"]))
rows.append(("stack-of-stacks (stack7 recipe)", base.loc["E13", "synth_private_rmse"] - base.loc["E13", "synth_private_rmse_singles"]))
R = pd.DataFrame(rows, columns=["proc", "d"]).sort_values("d")
cols = [AQUA if d < 0 else ORANGE for d in R["d"]]
ax.barh(R["proc"], R["d"], color=cols, height=0.62)
ax.axvline(0, color=MUTED, lw=1)
for i, (p_, d) in enumerate(zip(R["proc"], R["d"])):
    ax.text(d + (0.00002 if d >= 0 else -0.00002), i, f"{d:+.5f}", va="center", ha="left" if d >= 0 else "right", fontsize=8, color=INK)
lim = max(abs(R["d"]).max() * 1.35, 0.0005)
ax.set_xlim(-lim * 0.5, lim)
ax.set_xlabel("change in private RMSE vs current blend recipe (semi-synthetic, 60 public/private splits; negative = better)")
ax.set_title("B. Post-processing and blending procedures, validated on synthetic public/private splits", loc="left", color=INK, fontsize=10.5)
ax.grid(axis="x", color=GRID, lw=0.8)

# C: validator accuracy (E14)
ax = axes[2]
e14 = base.loc["E14"]
vals = [e14["loo_abs_err_uncorrected"], e14["loo_abs_err_corrected"]]
ax.barh(["all-rows approximation (before)", "public-subset corrected (E14)"], vals, color=[ORANGE, AQUA], height=0.5)
for i, v in enumerate(vals):
    ax.text(v + 0.000005, i, f"{v:.6f}", va="center", fontsize=8, color=INK)
ax.set_xlim(0, max(vals) * 1.3)
ax.set_xlabel("mean |predicted - actual| public RMSE, leave-one-file-out over 8 scored files")
ax.set_title("C. How well the validator predicts real leaderboard scores it has not seen", loc="left", color=INK, fontsize=10.5)
ax.grid(axis="x", color=GRID, lw=0.8)
fig.tight_layout()
fig.savefig("exp11/round11_progress.png", dpi=150, facecolor="white")
print("wrote exp11/round11_progress.png")
