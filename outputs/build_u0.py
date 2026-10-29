"""Rebuild round 4's pure base update (U0) exactly as cosci.py built it, verify against the A/B
files, and write outputs/submission_u0.csv."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd
import train as TR

L = TR.AggregateLabels()
g = L.graded

# --- reproduce cosci.update_base: the measured-direction move that passed the stage-8 rule
best = min((n for n in L.s if not n.startswith("probe_")), key=L.s.get)
moves = [u for u in TR.transfer_updates(L) if u["p_beats_best"] >= TR.MIN_MOVE_PROB and u["expected_rmse"] < L.s[best]]
assert moves, "no qualifying base move — cosci would have fallen back to the measured best"
u = min(moves, key=lambda u_: u_["expected_rmse"] + u_["sd_rmse"] / 2)
base = u["vector"].copy()
base[~g] = L.V[best][~g]
base = np.round(base, 6)
print(f"base move: {u['label']}; expected {u['expected_rmse']:.6f} +- {u['sd_rmse']:.6f}, P(beats {L.s[best]:.6f}) {u['p_beats_best']:.3f}")

pd.DataFrame({"GEOID": L.geoid, "coverage_gap_score": base}).to_csv("outputs/submission_u0.csv", index=False)

# --- verify A and B reproduce from this base with their recorded share/physics
rep = {}
nnw = pd.read_csv(TR.out("physics_lb_struct_unmatched_regional_nnw.csv"), dtype={"GEOID": str}).set_index("GEOID")["coverage_gap_score"].loc[L.geoid].to_numpy(float)
flex = pd.read_csv(TR.out("physics_lb_struct_unmatched_regional_flex20.csv"), dtype={"GEOID": str}).set_index("GEOID")["coverage_gap_score"].loc[L.geoid].to_numpy(float)
v, share_a = TR.transfer_step(L, base.copy(), nnw, rep, u["expected_rmse"], "nnw")
v = np.round(v, 6)
A_file = pd.read_csv("submission.csv", dtype={"GEOID": str})["coverage_gap_score"].to_numpy(float)
print(f"A check: share {share_a:.3f}, max|recon - submission.csv| = {np.abs(v - A_file).max():.9f}")
v2, share_b = TR.transfer_step(L, base.copy(), flex, rep, u["expected_rmse"], "flex")
v2 = np.round(v2, 6)
B_file = pd.read_csv("submission_alt.csv", dtype={"GEOID": str})["coverage_gap_score"].to_numpy(float)
print(f"B check: share {share_b:.3f}, max|recon - submission_alt.csv| = {np.abs(v2 - B_file).max():.9f}")

est = TR.estimate(L, base)
print(f"U0 in-span projection estimate: {est:.6f}")
print("wrote outputs/submission_u0.csv")
print("label when scored: tr_u0_000.csv  (record with: python train.py --add-label tr_u0_000.csv outputs/submission_u0.csv <public RMSE>)")
