"""Round-4 schema checks and forecasts."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd
import train as TR

L = TR.AggregateLabels()
files = {
    "U1  pure base update": ("outputs/submission_u1.csv", "tr_u1_000.csv"),
    "A   base + 0.30 x wide-net unscored": ("submission.csv", "tr_cosci05A_030.csv"),
    "B   base + 0.30 x flexible unscored": ("submission_alt.csv", "tr_cosci05B_031.csv"),
    "C   probe (nnw residual, orthogonal to A/B)": ("outputs/probes_branch/q093_round5_network_residual.csv", "q093_round5_network_residual.csv"),
}
expected = pd.concat([pd.read_csv(f"{TR.DATA}/reference/{r}/{r}-sample-submission.csv", dtype={"GEOID": str})[["GEOID"]]
                      for r in TR.REGIONS], ignore_index=True)
out = []
for label, (path, next_label) in files.items():
    df = pd.read_csv(path, dtype={"GEOID": str})
    assert df.columns.tolist() == ["GEOID", "coverage_gap_score"], path
    assert len(df) == len(expected) == 9794 and df.GEOID.is_unique and set(df.GEOID) == set(expected.GEOID), path
    v = df.coverage_gap_score.to_numpy(float)
    assert np.isfinite(v).all() and v.min() >= 0 and v.max() <= 1 and df.GEOID.str.fullmatch(r"\d{11}").all(), path
    est = TR.estimate(L, v)
    best_name = min((s_, n) for n, s_ in L.s.items() if not n.startswith("probe_"))[1]
    newpart = df.coverage_gap_score.to_numpy(float) - L.V[best_name]
    rms_new = float(np.sqrt(TR.P_PUBLIC and (L.pi @ (newpart[L.graded] ** 2)) / L.pi.sum()))
    rows = dict(file=label, path=path, next_label=next_label, rows=len(df),
                forecast_estimate=round(est, 9), step_rms_from_best=round(rms_new, 6),
                min=round(float(v.min()), 6), max=round(float(v.max()), 6),
                md5=TR.file_md5(path))
    out.append(rows)
    print(rows)
with open("outputs/round5_files.json", "w") as f:
    json.dump(out, f, indent=1)
print("\nwrote outputs/round5_files.json")
