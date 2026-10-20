"""Box-constrained aggregate regression via a small convex dual problem.

Unlike clipping a fitted model once, every iteration solves the actual bounded
problem, so feasible tracts feed back into the other aggregate constraints.
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.linalg import cho_factor,cho_solve
from scipy.optimize import minimize

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import solution as S
from feedback import load
from r29_analysis import design
from r30_kernels import fit_kernel

OUT=ROOT/'research/r30'


def bounded(M,z,N,base,tau,noise_scale,anchors=None):
    R=N*noise_scale
    scale=np.sqrt(np.diag(tau*tau*(M@M.T)+R)).clip(min=1e-9)
    lower=np.zeros(len(base));upper=np.ones(len(base))
    if anchors is not None:
        for i,v in anchors.items():lower[i]=upper[i]=v
    def objective(t):
        alpha=t/scale;s=M.T@alpha
        y=np.clip(base+tau*tau*s,lower,upper)
        conjugate=s@y-.5*np.sum((y-base)**2)/tau**2
        value=.5*alpha@R@alpha+conjugate-z@alpha
        grad=(R@alpha+M@y-z)/scale
        return value,grad
    opt=minimize(objective,np.zeros(len(z)),jac=True,method='L-BFGS-B',
                 options={'maxiter':4000,'ftol':1e-13,'gtol':1e-7,'maxls':50})
    alpha=opt.x/scale
    y=np.clip(base+tau*tau*(M.T@alpha),lower,upper)
    primal=.5*np.sum((y-base)**2)/tau**2+.5*(z-M@y)@np.linalg.solve(R,z-M@y)
    gap=primal+opt.fun
    return y,alpha,dict(success=bool(opt.success),iterations=int(opt.nit),dual_gap=float(gap),
                        max_scaled_gradient=float(np.max(np.abs(opt.jac))),tau=float(tau),noise_scale=float(noise_scale))


def constraints(lb,base):
    H=[];rv=[];masks=S.stratum_masks(lb)
    for name,m in masks.items():
        ratio=lambda f:lb.vec(f)[m].mean()/lb.vec(f)[~m].mean()
        calibration=np.mean([S.OURS_R23[name]/ratio('r23_ridge_eb_blend.csv'),S.OURS_R24[name]/ratio('r24_scorecard_joint.csv')])
        target=S.LEADER[name]/calibration
        H.append(np.where(m,1/m.sum(),-target/(~m).sum()))
        relative_var=.005**2/3*(1/S.LEADER[name]**2+.25/S.OURS_R23[name]**2+.25/S.OURS_R24[name]**2)
        rv.append((target*base[~m].mean())**2*relative_var)
    H=np.array(H)
    rhs=-H[:,~lb.graded]@base[~lb.graded]
    R=.0624**2*H[:,~lb.graded]@H[:,~lb.graded].T+np.diag(rv)
    return H[:,lb.graded],rhs,R


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    lb=load();g=lb.graded;w=lb.pi/lb.pi.sum()
    files=[f for f in lb.s if f not in ('probe_zeros.csv','e06_r2f_only_eastern-wa.csv')]
    base=lb.vec(S.BASE)[g];rows=[]
    for i in range(files.index('p12_01_route_ratio.csv'),len(files)):
        d=design(lb,files[:i+1],base,lb.s[S.BASE]);M=(d['A']*w[:,None]).T
        train=np.array([j for j,v in enumerate(d['keep']) if v<i])
        _,_,_,hyper=fit_kernel(M,d['z'],d['N'],sparse.csr_matrix((len(base),0)),base,train)
        tau=np.exp(hyper['hyper'][1]);alphaN=np.exp(hyper['hyper'][2])
        for factor,noise_factor in [(1.,1.),(.25,1.),(.5,1.),(2.,1.),(1.,.5),(1.,2.)]:
            y,a,info=bounded(M[train],d['z'][train],d['N'][np.ix_(train,train)],base,tau*factor,alphaN*noise_factor)
            mean=M@y+alphaN*noise_factor*d['N'][:,train]@a
            cy=d['R'][:,-1]@mean
            v=lb.vec(files[i])[g];pred=np.sqrt(max(lb.AP(v*v)+lb.Z-2*cy,0))
            rows.append(dict(file=files[i],actual=lb.s[files[i]],predicted=pred,error=pred-lb.s[files[i]],
                             factor=factor,noise_factor=noise_factor,scorecard=False,**info))
        if i>files.index('r24_scorecard_joint.csv'):
            H,rhs,Rh=constraints(lb,lb.vec(S.BASE))
            Mt=np.vstack([M[train],H]);zt=np.r_[d['z'][train],rhs]
            Rt=np.zeros((len(zt),len(zt)));Rt[:len(train),:len(train)]=alphaN*d['N'][np.ix_(train,train)]
            Rt[len(train):,len(train):]=Rh
            y,a,info=bounded(Mt,zt,Rt,base,tau,1.)
            mean=M@y+alphaN*d['N'][:,train]@a[:len(train)]
            cy=d['R'][:,-1]@mean
            v=lb.vec(files[i])[g];pred=np.sqrt(max(lb.AP(v*v)+lb.Z-2*cy,0))
            rows.append(dict(file=files[i],actual=lb.s[files[i]],predicted=pred,error=pred-lb.s[files[i]],
                             factor=1.,noise_factor=1.,scorecard=True,**info))
        print(files[i],pred-lb.s[files[i]],info['dual_gap'],flush=True)
        pd.DataFrame(rows).to_csv(OUT/'cv_bounded.csv',index=False)
    d=design(lb,files,base,lb.s[S.BASE]);M=(d['A']*w[:,None]).T
    _,_,_,hyper=fit_kernel(M,d['z'],d['N'],sparse.csr_matrix((len(base),0)),base)
    tau=np.exp(hyper['hyper'][1]);alphaN=np.exp(hyper['hyper'][2])
    report={}
    for with_scorecard,factor,noise_factor in [(False,1.,1.),(True,1.,1.),(True,.25,1.),(True,.5,1.),
                                              (True,2.,1.),(True,1.,.5),(True,1.,2.)]:
        Ma,z,R=M,d['z'],d['N']*alphaN*noise_factor
        if with_scorecard:
            H,rhs,Rh=constraints(lb,lb.vec(S.BASE))
            Ma=np.vstack([M,H]);z=np.r_[z,rhs]
            R=np.zeros((len(z),len(z)));R[:len(M),:len(M)]=d['N']*alphaN*noise_factor;R[len(M):,len(M):]=Rh
        anchor={i:S.PUBLISHED[geoid] for i,geoid in enumerate(lb.geoid[g]) if geoid in S.PUBLISHED}
        pred,_,info=bounded(Ma,z,R,base,tau*factor,1.,anchors=anchor)
        out=lb.vec('r29_correlated_scorecard.csv').copy();out[g]=pred
        tag='bounded_scorecard' if with_scorecard else 'bounded'
        if factor!=1. or noise_factor!=1.:tag+=f'_tau{factor:g}_noise{noise_factor:g}'
        np.save(OUT/f'{tag}.npy',S.finish(out,lb.geoid));report[tag]=info
    (OUT/'models_bounded.json').write_text(json.dumps(report,indent=2))
    cv=pd.DataFrame(rows);print(cv.groupby(['factor','noise_factor','scorecard']).error.agg(mae=lambda x:x.abs().mean(),rmse=lambda x:np.sqrt(np.mean(x*x))).to_string())


if __name__=='__main__':main()
