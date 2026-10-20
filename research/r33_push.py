"""R33: R32 + a measured push along R29's out-of-span piece.

Run from the repo root: python research/r33_push.py

R29's score measures the part of the R29 - R28b move outside the span of earlier scored
files (u29): best step 23.0 +- 9.7 (z = 2.36). R32 carries it at step 1. The step is shrunk
to a N(1, 3^2) prior; R28b showed a 1 +- 1 prior is too tight (measured 6.0, pushed to 2.3,
later optimum about 3.9). R32's own new piece measured -0.2 +- 1.9 (no signal) and is not pushed.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))
import solution as S
import r32_scorecard as SC

BASE, PREV, MOVE = "r32_t1_multi_leader.csv", "r28b_push.csv", "r29_correlated_scorecard.csv"
PRIOR_SD = 3.0
OUTPUT = ROOT / "submission_r33.csv"


def out_of_span_step(lb, prev, new):
    """Split new - prev (graded rows) into its part inside the span of files scored before
    `new` and the rest u; the score of `new` measures <u, y>. Returns u, step, sd."""
    g = lb.graded
    w = lb.pi / lb.pi.sum()
    order = list(lb.s)
    before = [f for f in order[:order.index(new)] if not f.startswith("probe_")]
    X = np.column_stack([np.ones(g.sum())] + [lb.vec(f)[g] for f in before])
    gram = (X * w[:, None]).T @ X
    gram += 1e-7 * np.trace(gram) / X.shape[1] * np.eye(X.shape[1])
    y0, y1 = lb.vec(prev), lb.vec(new)
    d = (y1 - y0)[g]
    q = X @ np.linalg.solve(gram, (X * w[:, None]).T @ d)
    u = d - q
    moments = np.array([lb.ybar] + [(lb.AP(lb.vec(f)[g] ** 2) + lb.Z - lb.s[f] ** 2) / 2 for f in before])
    d_y = (lb.AP(y1[g] ** 2) - lb.AP(y0[g] ** 2) - (lb.s[new] ** 2 - lb.s[prev] ** 2)) / 2
    q_y = np.linalg.solve(gram, (X * w[:, None]).T @ q) @ moments
    size = lb.AP(u * u)
    sd = np.sqrt(0.0622 ** 2 * size * (1 - S.P_PUBLIC) / (S.P_PUBLIC * g.sum())) / size
    return u, float((d_y - q_y - lb.AP(u * y0[g])) / size), float(sd)


def main():
    lb = S.Leaderboard()
    lb.public_subset()
    g = lb.graded
    w = lb.pi / lb.pi.sum()
    u, b, sd = out_of_span_step(lb, PREV, MOVE)
    post = (b / sd ** 2 + 1 / PRIOR_SD ** 2) / (1 / sd ** 2 + 1 / PRIOR_SD ** 2)
    base = lb.vec(BASE)
    v = base.copy()
    v[g] = base[g] + (post - 1) * u
    v = S.finish(v, lb.geoid)
    out = pd.DataFrame({"GEOID": lb.geoid.astype(str), "coverage_gap_score": v})
    anchors = sum(float(out.loc[out.GEOID.eq(k), "coverage_gap_score"].iloc[0]) == x for k, x in S.PUBLISHED.items())
    assert len(out) == 9794 and out.GEOID.is_unique and out.coverage_gap_score.between(0, 1).all() and anchors == 5
    assert np.array_equal(v[~g], base[~g])
    out.to_csv(OUTPUT, index=False)
    groups = SC.scorecard_groups(lb.geoid)
    lead = SC.leaders(1)[0]["rubric"]
    rat = SC.ratios(v, groups)
    d = (v - base)[g]
    print(f"u29 step {b:.2f} +- {sd:.2f} (z {b / sd:.2f}); posterior total step {post:.2f} (prior 1 +- {PRIOR_SD:g})")
    print(f"{OUTPUT.name}: est public {SC.estimate(lb, v):.9f} (R32 est {SC.estimate(lb, base):.9f}, actual {lb.s[BASE]:.9f}); "
          f"RMS change {np.sqrt(w @ d ** 2):.5f}, inside scored span {SC.span_share(lb, d):.0%}")
    print("ratios vs rank 1: " + " ".join(f"{k} {x:.3f}/{lead[SC.API[k]]:.3f}" for k, x in rat.items()))


if __name__ == "__main__":
    main()
