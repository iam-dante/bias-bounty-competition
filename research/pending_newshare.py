"""For each pending probe in next/: how much of its push (probe minus its base) is new, i.e. not spanned by
the scored files. The base is the probe's own projection on the scored files plus r19b's span.
    python research/pending_newshare.py"""
import glob, os, sys, numpy as np, pandas as pd
sys.path.insert(0, ".")
import solution as S
df = S.load(); lb = S.Leaderboard(df); lb.public_subset(); g = lb.graded; w = np.sqrt(lb.pi / lb.pi.sum())
files = [f for f in lb.s if not f.startswith("probe_")]
sub = pd.read_csv("submission.csv")["coverage_gap_score"].to_numpy()[g]
Q, _ = np.linalg.qr(np.column_stack([np.ones(g.sum())] + [lb.vec(f)[g] for f in files] + [sub]) * w[:, None])
rows = []
for f in sorted(glob.glob("next/p*.csv")):
    v = pd.read_csv(f)["coverage_gap_score"].to_numpy()[g]
    r = v * w - Q @ (Q.T @ (v * w))                 # the part of the probe no scored file (or r19b) explains
    push = (v - sub) * w                            # the probe's push away from the current best
    rows.append((os.path.basename(f)[:-4], float(np.linalg.norm(push)), float(np.linalg.norm(r)),
                 float(np.linalg.norm(r) / max(np.linalg.norm(push), 1e-12))))
t = pd.DataFrame(rows, columns=["probe", "push_rms_vs_r19b", "new_rms", "new_share_of_push"]).sort_values("new_rms", ascending=False)
t.round(5).to_csv("research/pending_newshare.csv", index=False)
print(t.round(4).to_string(index=False))
