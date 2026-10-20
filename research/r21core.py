"""R21 core from solution.py main(), parameterised by a Leaderboard whose scores may be synthetic."""
import numpy as np
import solution as S
def r21(lb, ridge=S.RIDGE, incumbent=True, linear=False):
    graded = lb.graded
    base = lb.vec(S.BASE)[graded]
    basis = S.BASIS + [S.BASE, "r11_blend_denoised.csv"]
    A, gram, c = lb.system(basis)
    cy = lambda v, s: (lb.AP(v**2) + lb.Z - s**2) / 2
    base_y = cy(base, lb.s[S.BASE])
    cols, mom = [], []
    dirs = [n for n in lb.s if S.probe_direction_of(n)]
    for name in dirs:
        v = lb.vec(name)[graded]; d = v - base
        coef = np.linalg.solve(gram, (A * lb.pi[:, None]).T @ d / lb.pi.sum())
        r = d - A @ coef
        cols.append(r); mom.append((cy(v, lb.s[name]) - base_y) - coef @ c - lb.AP(r * base))
    P = np.column_stack(cols); b = np.asarray(mom)
    gp = (P * lb.pi[:, None]).T @ P / lb.pi.sum()
    gamma = np.linalg.solve(gp + ridge * np.trace(gp) / len(dirs) * np.eye(len(dirs)), b)
    raw = lb.vec(S.BASE).copy(); raw[graded] += P @ gamma
    if linear: return raw
    cand = S.finish(raw, lb.geoid)
    if not incumbent: return cand
    span = np.column_stack([A, P])
    sm = np.r_[c, b + (P * lb.pi[:, None]).T @ base / lb.pi.sum()]
    inc = lb.vec("r19b_backward_selection.csv")[graded]
    wt = np.sqrt(lb.pi)
    pr = np.linalg.lstsq(span * wt[:, None], inc * wt, rcond=1e-8)[0]
    extra = inc - span @ pr
    extra_y = cy(inc, lb.s["r19b_backward_selection.csv"]) - pr @ sm
    var = lb.AP(extra**2); cov = extra_y - lb.AP(extra * cand[graded])
    noise = lb.s["r19b_backward_selection.csv"]**2 / lb.pi.sum()
    z = cov / np.sqrt(var * noise) if var > 1e-16 else 0.0
    st = float(np.clip(cov / var * (1 - 1 / z**2), 0, 1)) if z > 1 else 0.0
    cand[graded] += st * extra
    return S.finish(cand, lb.geoid)
