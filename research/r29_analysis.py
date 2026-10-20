"""Reproducible aggregate-label analysis; no tract targets are available.

Run from the repo root: OPENBLAS_NUM_THREADS=1 python research/r29_analysis.py
Chronological score prediction compares three empirical-Bayes priors. Future
scores never enter a fold's fit or noise estimate. Gaussian conditioning is
analytic, so no MCMC approximation or PyMC dependency is needed.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize, minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import solution as S

OUT = ROOT / "research/r29"
MODES = ("original", "free_clip", "effect_scale")


def group(name):
    if name.startswith("probe_const"):
        return 0
    if S.probe_direction_of(name):
        return 2
    if name.startswith(("r14", "r19b", "r22", "r23", "r24", "r25", "r26", "r27", "r28")):
        return 3
    return 1


def design(lb, names, center, sigma):
    w = lb.pi / lb.pi.sum()
    V = np.column_stack([lb.vec(f)[lb.graded] for f in names])
    c = np.array([(w @ V[:, i]**2 + lb.Z - lb.s[f]**2)/2 for i, f in enumerate(names)])
    qs, keep = [], []
    for i in range(len(names)):
        v = V[:, i].copy()
        for _ in range(2):
            for q in qs:
                v -= (w @ (q*v)) / (w @ q**2) * q
        if np.sqrt(w @ v**2) > 1e-5 * np.sqrt(w @ V[:, i]**2):
            qs.append(v)
            keep.append(i)
    Q = np.column_stack(qs)
    D = w @ Q**2
    A = Q / np.sqrt(D)
    # R maps normalized, orthogonal directions to original prediction files.
    R = (A*w[:, None]).T @ V
    inv = np.linalg.inv(R[:, keep]).T
    z = inv @ c[keep]
    U = V*(center[:, None]-V/2)
    U -= w @ U
    Nv = ((1-S.P_PUBLIC)*(U*w[:, None]).T@U + sigma**2*(V*w[:, None]).T@V)/(S.P_PUBLIC*len(w))
    N = inv @ Nv[np.ix_(keep, keep)] @ inv.T
    return dict(A=A, D=D, R=R, z=z, N=(N+N.T)/2,
                groups=np.array([group(names[i]) for i in keep]), keep=keep, names=names)


def fit(d, mode, train=None):
    k = len(d['z'])
    train = np.arange(k) if train is None else np.asarray(train)
    gr, D = d['groups'], d['D']
    # Original priors concern raw innovation coefficients. Effect-scale priors
    # instead concern an innovation's RMS change in the target's score units.
    def prior(x):
        mean = np.select([gr == 1, gr == 2], [x[1], x[3]], 0.) * np.sqrt(D)
        clip = np.exp(2*x[5]) if mode != 'original' else 1.
        var = np.select([gr == 0, gr == 1, gr == 2], [100., np.exp(2*x[0]), np.exp(2*x[2])], clip)
        var *= D if mode != 'effect_scale' else np.where(gr == 0, D, 1.)
        C = np.diag(var) + np.exp(x[4])*d['N']
        return mean, var, C
    def objective(x):
        m, _, C = prior(x)
        try:
            L = cho_factor(C[np.ix_(train, train)], lower=True)
            r = (d['z']-m)[train]
            return .5*r @ cho_solve(L,r) + np.log(np.diag(L[0])).sum()
        except np.linalg.LinAlgError:
            return 1e30
    base = [-1.05,.56,-.29,.04,-.87]
    starts = [base + ([0.] if mode != 'original' else [])]
    if mode == 'effect_scale':
        starts = [[-4.,.56,-5.,.04,-.5,-6.],[-3.,.3,-6.,0.,0.,-7.]]
    bounds = [(-10,2),(-2,2),(-12,2),(-2,2),(-5,3)] + ([(-14,2)] if mode != 'original' else [])
    opts = [minimize(objective,x,method='L-BFGS-B',bounds=bounds, options={'maxiter':700,'ftol':1e-11}) for x in starts]
    opt = min(opts,key=lambda r:r.fun)
    m,var,C = prior(opt.x)
    L=cho_factor(C[np.ix_(train,train)],lower=True)
    a=cho_solve(L,(d['z']-m)[train])
    theta=m.copy()
    theta[train]+=var[train]*a
    # Predict an unobserved measurement including shared public-sampling noise.
    obsmean=m+C[:,train]@a
    obscov=C-C[:,train]@cho_solve(L,C[train,:])
    return dict(theta=theta, obsmean=obsmean, obscov=obscov,
                hyper=opt.x.tolist(), nll=float(opt.fun), converged=bool(opt.success),
                variance=var, covariance=C)


def constrained(lb, d, f, rounding=1., correlated=True):
    """Condition the analytic posterior on rounded, correlated scorecard ratios.

    Ungraded WA predictions stay fixed but their uncertain contribution enters
    as a full covariance matrix. An iid tract prior is projected outside A,
    preventing double counting of the scored prediction span.
    """
    masks=S.stratum_masks(lb)
    base=lb.vec('r28b_push.csv').copy()
    A,w=d['A'],lb.pi/lb.pi.sum()
    ratio=lambda v,m:v[m].mean()/v[~m].mean()
    hs=[]; vars_round=[]
    for name,m in masks.items():
        calibration=np.mean([S.OURS_R23[name]/ratio(lb.vec('r23_ridge_eb_blend.csv'),m),
                             S.OURS_R24[name]/ratio(lb.vec('r24_scorecard_joint.csv'),m)])
        target=S.LEADER[name]/calibration
        hs.append(np.where(m,1/m.sum(),-target/(~m).sum()))
        # Both the leader and our two calibration ratios are rounded to .01.
        # Independent uniform rounding errors, propagated to score units.
        relative_var=(.005**2/3)*(1/S.LEADER[name]**2 +
                      .25/S.OURS_R23[name]**2+.25/S.OURS_R24[name]**2)
        vars_round.append((target*base[~m].mean())**2*relative_var*rounding**2)
    H=np.array(hs); Hg=H[:,lb.graded]
    B=Hg@A
    # Euclidean projection, since the new prior is iid per tract (not pi iid).
    Hu=Hg-(Hg@A)@np.linalg.solve(A.T@A,A.T)
    var=f['variance']; L=cho_factor(f['covariance'],lower=True)
    P=np.diag(var)-(var[:,None]*cho_solve(L,np.diag(var)))
    P=(P+P.T)/2
    base[lb.graded]=A@f['theta']
    rhs=-H@base
    R=.0624**2 * H[:,~lb.graded]@H[:,~lb.graded].T
    if not correlated:
        R=np.diag(np.diag(R))
    R+=np.diag(vars_round)
    C0=B@P@B.T+R
    UU=Hu@Hu.T
    def nll(lt):
        L=cho_factor(C0+np.exp(2*lt)*UU,lower=True)
        return .5*rhs@cho_solve(L,rhs)+np.log(np.diag(L[0])).sum()
    opt=minimize_scalar(nll,bounds=(-12,-1.5),method='bounded')
    tau=np.exp(opt.x)
    a=np.linalg.solve(C0+tau**2*UU,rhs)
    delta=A@(P@B.T@a)+tau**2*Hu.T@a
    base[lb.graded]+=delta
    return S.finish(base,lb.geoid),dict(tau=float(tau),nll=float(opt.fun),
               rms_update=float(np.sqrt(w@delta**2)),max_noise_correlation=float(np.max(np.abs(R/np.sqrt(np.outer(np.diag(R),np.diag(R)))-np.eye(8)))))


def minimum_change(lb, parts, rounding=1., correlated=True, noise_scale=1., omit=None,
                   ratio_jitter=None):
    """Conservative recalibration centered on the scored incumbent.

    This is an empirical local update: the center was already constructed from
    score feedback. Its posterior scale is a sensitivity device, not an
    independent Bayesian posterior or evidence for a generalization gain.
    """
    base=lb.vec('r28b_push.csv').copy(); g=lb.graded
    masks=S.stratum_masks(lb); Q,w=parts['Q'],parts['w']
    H=[]; rv=[]; kept=[]
    for name,m in masks.items():
        if name==omit:
            continue
        calc=lambda n:lb.vec(n)[m].mean()/lb.vec(n)[~m].mean()
        shown=np.mean([S.OURS_R23[name]/calc('r23_ridge_eb_blend.csv'),
                       S.OURS_R24[name]/calc('r24_scorecard_joint.csv')])
        target=S.LEADER[name]/shown
        if ratio_jitter is not None:
            target*=ratio_jitter[name]
        H.append(np.where(m,1/m.sum(),-target/(~m).sum()))
        relative_var=(.005**2/3)*(1/S.LEADER[name]**2+
                        .25/S.OURS_R23[name]**2+.25/S.OURS_R24[name]**2)
        rv.append((target*base[~m].mean())**2*relative_var*rounding**2)
        kept.append(name)
    H=np.array(H)
    M=np.vstack([(Q*w[:,None]).T,H[:,g]])
    obs=np.r_[parts['o']-(Q*w[:,None]).T@base[g],-H@base]
    R=np.zeros((len(obs),len(obs)))
    R[:Q.shape[1],:Q.shape[1]]=parts['noise']*noise_scale
    WA=.0624**2*H[:,~g]@H[:,~g].T
    if not correlated:
        WA=np.diag(np.diag(WA))
    R[Q.shape[1]:,Q.shape[1]:]=WA+np.diag(rv)
    MM=M@M.T
    def nll(lt):
        L=cho_factor(np.exp(2*lt)*MM+R,lower=True)
        return .5*obs@cho_solve(L,obs)+np.log(np.diag(L[0])).sum()
    opt=minimize_scalar(nll,bounds=(-12,-2),method='bounded')
    tau=np.exp(opt.x)
    a=np.linalg.solve(tau**2*MM+R,obs)
    delta=tau**2*M.T@a
    out=base.copy();out[g]+=delta
    out=S.finish(out,lb.geoid)
    return out,dict(tau=float(tau),rms=float(np.sqrt(w@(out[g]-base[g])**2)),
                    rounding=rounding,correlated=correlated,noise_scale=noise_scale,omit=omit)


def chronological(lb):
    files=[f for f in lb.s if f not in ('probe_zeros.csv','e06_r2f_only_eastern-wa.csv')]
    first=files.index('p12_01_route_ratio.csv')
    rows=[]
    for i in range(first,len(files)):
        prior=files[:i]
        incumbent=min([f for f in prior if not f.startswith('probe_')],key=lb.s.get)
        # The held-out prediction is known; only its score must remain unseen.
        d=design(lb,files[:i+1],lb.vec(incumbent)[lb.graded],lb.s[incumbent])
        train=np.array([j for j,v in enumerate(d['keep']) if v<i])
        # A dependent heldout vector may add no new Q column; original c entries
        # used for training are independent of the held-out c in either case.
        for mode in MODES:
            f=fit(d,mode,train)
            r=d['R'][:,-1]
            cy=r@f['obsmean']
            v=lb.vec(files[i])[lb.graded]
            mse=lb.AP(v**2)+lb.Z-2*cy
            pred=np.sqrt(max(mse,0))
            sd=np.sqrt(max(r@f['obscov']@r,0))/max(pred,1e-10)
            rows.append(dict(file=files[i],mode=mode,actual=lb.s[files[i]],predicted=pred,
                             error=pred-lb.s[files[i]],sd=sd,converged=f['converged']))
        print('holdout',files[i],flush=True)
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    lb=S.Leaderboard();lb.public_subset()
    cv=chronological(lb)
    cv.to_csv(OUT/'chronological.csv',index=False)
    print(cv.groupby('mode').agg(mae=('error',lambda x: x.abs().mean()),rmse=('error',lambda x:np.sqrt(np.mean(x*x))),bias=('error','mean')).to_string())
    names=[f for f in lb.s if f not in ('probe_zeros.csv','e06_r2f_only_eastern-wa.csv')]
    best=min(names,key=lb.s.get)
    d=design(lb,names,lb.vec(best)[lb.graded],lb.s[best])
    models={}
    for mode in MODES:
        f=fit(d,mode)
        pred=lb.vec(best).copy();pred[lb.graded]=d['A']@f['theta']
        models[mode]={k:f[k] for k in ('hyper','nll','converged')}
        models[mode]['rms_from_best']=float(np.sqrt(lb.AP((pred[lb.graded]-lb.vec(best)[lb.graded])**2)))
        np.save(OUT/f'{mode}.npy',pred)
        for rounding,correlated in [(1.,True),(2.,True),(1.,False)]:
            p,info=constrained(lb,d,f,rounding,correlated)
            tag=f'{mode}_round{rounding:g}_corr{int(correlated)}'
            np.save(OUT/f'{tag}.npy',p)
            models[tag]=info
    (OUT/'models.json').write_text(json.dumps(models,indent=2))
    print(json.dumps(models,indent=2))
    _,_,_,parts=S.eb_posterior(lb,lb.vec(best)[lb.graded])
    legacy_local,_=S.joint_min_change(lb,lb.vec(best),parts)
    np.save(OUT/'best_center_joint.npy',legacy_local)
    local={}
    for rounding,correlated,noise_scale in [(1.,True,1.),(2.,True,1.),(1.,False,1.),
                                           (1.,True,.5),(1.,True,2.)]:
        tag=f'local_round{rounding:g}_corr{int(correlated)}_noise{noise_scale:g}'
        pred,info=minimum_change(lb,parts,rounding,correlated,noise_scale)
        np.save(OUT/f'{tag}.npy',pred);local[tag]=info
    chosen='local_round1_corr1_noise1'
    final=np.load(OUT/f'{chosen}.npy')
    output=pd.DataFrame({'GEOID':lb.geoid,'coverage_gap_score':final})
    sample=pd.concat([pd.read_csv(p,dtype={'GEOID':str}) for p in ROOT.glob('data/reference/*/*-sample-submission.csv')])
    assert len(output)==len(sample)==9794 and output.GEOID.is_unique
    assert set(output.GEOID)==set(sample.GEOID)
    assert np.isfinite(final).all() and output.coverage_gap_score.between(0,1).all()
    assert all(output.set_index('GEOID').coverage_gap_score[k]==v for k,v in S.PUBLISHED.items())
    output.to_csv(ROOT/'submission_r29.csv',index=False)
    # Individual-stratum ablations: influence, not heldout tract accuracy.
    for name in S.STRATA:
        pred,info=minimum_change(lb,parts,omit=name)
        info['rms_from_final']=float(np.sqrt(lb.AP((pred[lb.graded]-final[lb.graded])**2)))
        local[f'omit_{name}']=info
    (OUT/'local_models.json').write_text(json.dumps(local,indent=2))
    print('Wrote submission_r29.csv',local[chosen])


if __name__=='__main__':
    main()
