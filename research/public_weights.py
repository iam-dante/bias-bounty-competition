"""Public-subset weights: backtest with uniform vs conditioned weights, and probe decoding.  python research/public_weights.py"""
import os, sys, numpy as np, pandas as pd, io, contextlib
sys.path.insert(0, ".")
import solution as S
df = S.load()
out = {}
for name, cond in [("uniform 30 %", False), ("conditioned (current)", True)]:
    lb = S.Leaderboard(df)
    if cond: lb.public_subset()
    with contextlib.redirect_stdout(io.StringIO()):
        t = S.validate(lb, out=os.devnull)
    m = t[t.file.str.match(r"^(r1[2-9]|p1[2-9])_")]
    out[name] = m
    for k in ["err_before", "err_loo"]:
        e = m[k].dropna().abs()
        print(f"A  pi {name:22s} {k:10s} MAE {e.mean():.6f} max {e.max():.6f} within 0.0005 {(e<5e-4).sum()}/{len(e)}")
    # Exp B: decoded moment of each probe's own new direction relative to the file it was built on, z-scale
    g = lb.graded; Z = lb.Z
    if name.startswith("uniform"): lb_u = lb
    else: lb_c = lb
def decode(lb, f, base):
    g = lb.graded; v, b0 = lb.vec(f)[g], lb.vec(base)[g]
    # exact identity: mean_P((f - base) * y) = cy(f) - cy(base)
    cy = lambda v, s: (lb.AP(v ** 2) + lb.Z - s ** 2) / 2
    d = v - b0; return (cy(v, lb.s[f]) - cy(b0, lb.s[base]) - lb.AP(d * b0)) / np.sqrt(lb.AP(d ** 2))
pairs = [("p13_02_boundary_leak.csv", "r12_blend_with_AB.csv"), ("p12_01_route_ratio.csv", "r12_blend_with_AB.csv")]
for f, b in pairs:
    print(f"B  {f:28s} rho-moment (uniform pi) {decode(lb_u, f, b):+.6f}  (conditioned pi) {decode(lb_c, f, b):+.6f}")
