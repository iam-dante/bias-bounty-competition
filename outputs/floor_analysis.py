"""Per-tract uncertainty floor of the current physics model.

Rebuilds stage 3-4 inputs, takes the label-calibrated flexible physics' fitted parameters
(outputs/physics_lb_struct_unmatched_regional_flex20_report.json), and runs the simulator at
4,096 draws. The per-tract E[y^2] - E[y]^2 is the conditional (aleatoric) variance the model
itself assigns: the RMSE no prediction built from the permitted data can go below under the
model, because the withdrawn reference counts are not determined by them.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd
import train as TR

np.random.seed(0)
L = TR.AggregateLabels()
g = L.graded

df, groups = TR.load()
print("auxiliary models (county-grouped CV) — rebuilding the cross-fitted densities")
mu, aux = TR.fit_auxiliary(df)
F = TR.inputs(df, mu)

th = np.array([p["fitted"] for p in json.load(open(TR.out("physics_lb_struct_unmatched_regional_flex20_report.json")))["parameters"]])
TR.set_priors(True, regional=True, structural=True, unmatched=True, flex=True)   # same parameter set as the flex20 fit
th = np.array([p["fitted"] for p in json.load(open(TR.out("physics_lb_struct_unmatched_regional_flex20_report.json")))["parameters"]])
assert len(th) == len(TR.NAMES), f"theta length {len(th)} vs {len(TR.NAMES)}"
S = 4096
sim = TR.simulate(F, th, S=S, seed=100)
var_floor = np.maximum(sim["Ey2"] - sim["Ey"] ** 2, 0.0)
pred = np.clip(sim["Ey"], 0, 1)
w = L.pi / L.pi.sum()    # pi is already restricted to the graded rows
wg = w
rms_pub = float(np.sqrt(wg @ (var_floor[g])))

mean_contrib = {k: float(w @ sim[k][g]) for k in ("c_road", "c_building", "c_facility", "c_establishment")}
gaps = {k: float(w @ sim[k][g]) for k in ("road_gap", "building_gap", "facility_gap", "establishment_gap")}
print(f"\nmodel: label-calibrated flexible physics (88 labels), 4096 draws/tract")
print(f"mean predicted score over graded rows: {float(wg @ sim['Ey'][g]):.6f}")
print(f"component contributions to the predicted gap:")
for k, v in mean_contrib.items():
    print(f"  {k:14s} {v:.6f}")
print(f"component means where defined: {gaps}")
print(f"\naleatoric RMSE floor claimed by the model: {rms_pub:.6f}")
print(f"measured best today:                        {L.s[min((v, n) for n, v in L.s.items() if not n.startswith('probe_'))[1]]:.6f}")
est = TR.estimate(L, L.V[min((v, n) for n, v in L.s.items() if not n.startswith('probe_'))[1]])
print(f"in-span projection estimate of that file:   {est:.6f}")
print(f"specification slack left between model floor and its in-span estimate: {est - rms_pub:.6f}")

out = dict(S=S, theta_source="physics_lb_struct_unmatched_regional_flex20_report.json",
           mean_prediction=float(w @ sim["Ey"][g]), mean_contrib=mean_contrib, gaps=gaps,
           aleatoric_rmse=rms_pub, in_span_estimate=est,
           best_measured=float(min(v for n, v in L.s.items() if not n.startswith("probe_"))))
with open("outputs/floor_analysis.json", "w") as f:
    json.dump(out, f, indent=1)
print("\nwrote outputs/floor_analysis.json")
