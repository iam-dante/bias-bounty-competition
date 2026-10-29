import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd
import train as TR

np.random.seed(0)

L = TR.AggregateLabels()
print(f"labels: {len(L.s)}, graded tracts: {int(L.graded.sum())},zenie public share {TR.P_PUBLIC}")

best = min((n for n in L.s if not n.startswith("probe_")), key=L.s.get)
print(f"best scored: {best} = {L.s[best]:.9f}")

# --- 1. chronological (rolling-origin) hold-out: predict each scored file's RMSE from earlier scores
fv = TR.forward_validation(L)
fv["abs_error"] = fv.error.abs()
by_group = fv.groupby(lambda i: TR.label_group(fv.file[i]))
rows = []
for grp, d in fv.groupby(fv.file.map(TR.label_group)):
    rows.append(dict(group=grp, n=len(d), mae=float(d.abs_error.mean()), rmse=float(np.sqrt((d.error**2).mean())),
                     bias=float(d.error.mean())))
recent = fv.tail(12)
rows.append(dict(group="last12", n=len(recent), mae=float(recent.abs_error.mean()),
                 rmse=float(np.sqrt((recent.error**2).mean())), bias=float(recent.error.mean())))
overall = dict(n=len(fv), mae=float(fv.abs_error.mean()), rmse=float(np.sqrt((fv.error**2).mean())),
               bias=float(fv.error.mean()))
print("\nforward validation (forecast each score from the labels before it):")
print(pd.DataFrame(rows).round(6).to_string(index=False))
print("overall:", {k: round(v, 6) if isinstance(v, float) else v for k, v in overall.items()})

# recent rows detail
print("\nlast 12 scored files, forecast vs actual:")
print(recent[["file", "predicted", "actual", "error"]].round(6).to_string(index=False))

# --- 2. what the current labels pin down about the target
Z = float(L.Z); ybar = float(L.ybar)
rms_y = float(np.sqrt(Z))
std_y = float(np.sqrt(max(Z - ybar ** 2, 0)))
print(f"\ntarget moments on the public set: mean y = {ybar:.6f}, RMS y = {rms_y:.6f}, sd y = {std_y:.6f}")
print(f"best constant (mean-only) public RMSE would be {std_y:.6f}; best measured {L.s[best]:.6f}")
for r2 in (0.9, 0.99, 0.999, 0.999986):
    print(f"variance explained {r2:.6f} -> RMSE {rms_y * np.sqrt(1 - r2):.6f}")
ach = 1 - (L.s[best] / rms_y) ** 2
print(f"current best explains {ach:.4f} of the public target variance")

# --- 3. forecast accuracy of the projection estimator on the newest label (tr_nofp_030)
prefix = [f for f in L.s if not f.startswith(("probe_", "e06"))]
i = prefix.index(best)
est = TR.estimate(L, L.V[best], prefix[:i])
print(f"\nforecast for the newest label ({best}) before it was scored: {est:.6f} vs actual {L.s[best]:.6f} "
      f"(error {est - L.s[best]:+.6f})")

out = dict(best_file=best, best_rmse=float(L.s[best]), n_labels=len(L.s), ybar=ybar, rms_y=rms_y, std_y=std_y,
           best_constant_rmse=std_y, forward_validation=dict(overall=overall, groups=rows,
                                                        last12=recent[["file", "predicted", "actual", "error"]]
                                                        .round(6).to_dict(orient="records")),
           newest_label_forecast=dict(file=best, predicted=float(est), actual=float(L.s[best])))
with open("outputs/validation_now.json", "w") as f:
    json.dump(out, f, indent=1)
print("\nwrote outputs/validation_now.json")
