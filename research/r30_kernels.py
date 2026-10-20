"""Train residual Gaussian processes from signed aggregate score observations.

No pseudo-labels: each training outcome is a moment implied by an actual score.
Random trees partition permitted covariates without target labels. All learning
and score predictions use a chronological fit of the covariance parameters.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.linalg import cho_factor,cho_solve
from scipy.optimize import minimize
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.preprocessing import SplineTransformer

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import solution as S
from feedback import load
from r29_analysis import design

OUT=ROOT/'research/r30'


def covariates(lb, seed=3101):
    frame=pd.concat([pd.read_parquet(p) for p in ROOT.glob('data/features/*.parquet') if 'facpts' not in p.name])
    frame=frame.set_index('GEOID').loc[lb.geoid]
    raw={}
    for col in ['L_named','L_route','L_route_multiref','L_interstate','L_fm','L_tertiary','L_county','L_unrouted',
                'Lb_route','n_bld','n_poi','n_poi_meta','n_poi_bq','n_fd','n_fd_alt','n_fire_name','n_ems','n_ems_name',
                'n_sch','n_sch_alt','n_sch_name','housing_units','area_m2','pop_total','pct_urban','svi_overall',
                'cvi_overall','tribal_pct','usfs_BuildingCount_sum','usfs_WHP_mean','usdm_summer_dsci',
                'usdm_winter_dsci','epht_heat_days_summer','ghcn_any_nearest_km']:
        x=frame[col].astype(float).to_numpy()
        if np.nanmin(x)>=0 and col!='pct_urban': x=np.log1p(x)
        raw[col]=x
    for numerator,denominator in [('n_bld','usfs_BuildingCount_sum'),('n_bld','housing_units'),
          ('n_poi','pop_total'),('n_poi_bq','n_poi'),('n_poi_meta','n_poi'),('L_named','L_route'),
          ('L_unrouted','L_route'),('L_tertiary','L_named'),('L_fm','L_route'),('L_route_multiref','L_route'),
          ('Lb_route','L_route'),('pop_total','area_m2'),('n_bld','area_m2'),('n_fd','pop_total'),
          ('n_ems','pop_total'),('n_sch','pop_total')]:
        raw[numerator+'/'+denominator]=np.log1p(frame[numerator].to_numpy(float))-np.log1p(frame[denominator].to_numpy(float))
    X=pd.DataFrame(raw,index=frame.index)
    missing=X.isna().astype(float)
    X=X.fillna(X.median()).fillna(0)
    X=(X-X.mean())/X.std().clip(lower=.01)
    X=X.clip(-5,5)
    X=pd.concat([X,missing.loc[:,missing.nunique()>1].add_suffix('_missing')],axis=1)
    # One feature map per kernel, normalized to mean diagonal one.
    linear=sparse.csr_matrix(X.to_numpy())
    splines=sparse.csr_matrix(SplineTransformer(n_knots=4,degree=2,include_bias=False).fit_transform(X))
    region=pd.get_dummies(frame.region,dtype=float)
    county=pd.get_dummies(frame.index.str[:5],dtype=float)
    county=sparse.csr_matrix(county.to_numpy())
    def normalize(P):
        P=sparse.csr_matrix(P)
        return P/np.sqrt(np.asarray(P.multiply(P).sum(1)).mean())
    maps={'white':sparse.csr_matrix((len(X),0)),'linear':normalize(linear),'splines':normalize(splines)}
    for depth in (3,5,7):
        forest=RandomTreesEmbedding(n_estimators=96,max_depth=depth,min_samples_leaf=20,
                  random_state=seed,n_jobs=1)
        maps[f'trees{depth}']=normalize(forest.fit_transform(X))
    # Regional interactions let the same road ratio have a different meaning
    # across study regions; all region labels are supplied competition data.
    regional=sparse.hstack([linear.multiply(region.iloc[:,j].to_numpy()[:,None]) for j in range(region.shape[1])])
    maps['regional_splines']=normalize(sparse.hstack([splines,regional,county*np.sqrt(10)]))
    from shapely import from_wkb,centroid,get_x,get_y
    geo=pd.concat([pd.read_parquet(p,columns=['GEOID','geometry']) for p in ROOT.glob('data/strata/*/*-census-tracts.parquet')])
    geo=geo.drop_duplicates('GEOID').set_index('GEOID').loc[lb.geoid]
    assert len(geo)==len(frame) and geo.geometry.notna().all()
    centers=centroid(from_wkb(geo.geometry.to_numpy()))
    xy=np.column_stack([get_x(centers)*85.,get_y(centers)*111.])
    rng=np.random.default_rng(seed)
    P=[]
    for scale in (10.,40.,120.):
        W=rng.normal(size=(2,128))/scale;b=rng.uniform(0,2*np.pi,128)
        P.append(np.sqrt(2)*np.cos(xy@W+b))
    maps['spatial']=normalize(sparse.hstack([sparse.csr_matrix(np.column_stack(P)),county*np.sqrt(100)]))
    maps['domain_mix']=normalize(sparse.hstack([maps['splines'],maps['trees5'],maps['spatial']]))
    return maps,list(X.columns)


def fit_kernel(M,z,N,Phi,center,train=None, start=None):
    k=len(z);train=np.arange(k) if train is None else np.asarray(train)
    L=np.asarray(M@Phi)
    F=L@L.T;W=M@M.T
    residual=z-M@center
    def covariance(x):
        return np.exp(2*x[0])*F+np.exp(2*x[1])*W+np.exp(x[2])*N
    def objective(x):
        C=covariance(x)[np.ix_(train,train)]
        try:
            chol=cho_factor(C,lower=True)
            r=residual[train]
            # Weak, explicit log-scale regularization. No unbounded scale fit.
            penalty=.5*((x[0]-np.log(.02))/2)**2+.5*(x[2]/1.)**2
            return .5*r@cho_solve(chol,r)+np.log(np.diag(chol[0])).sum()+penalty
        except np.linalg.LinAlgError:
            return 1e25
    initial=[np.log(.02),np.log(.03),0.] if start is None else start
    opt=minimize(objective,initial,method='L-BFGS-B',bounds=[(-9,-1.5),(-7,-1.5),(-3,3)],
                 options={'maxiter':300,'ftol':1e-10})
    C=covariance(opt.x);chol=cho_factor(C[np.ix_(train,train)],lower=True)
    a=cho_solve(chol,residual[train])
    mean=M@center+C[:,train]@a
    predvar=np.diag(C-C[:,train]@cho_solve(chol,C[train,:]))
    # Only signal covariance, not observation noise, changes tract estimates.
    coef=np.exp(2*opt.x[0])*L[train].T@a
    delta=np.asarray(Phi@coef).ravel()+np.exp(2*opt.x[1])*M[train].T@a
    return center+delta,mean,predvar,dict(hyper=opt.x.tolist(),objective=float(opt.fun),
                              success=bool(opt.success),coef=coef)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');parser.add_argument('--seed',type=int,default=3101)
    parser.add_argument('--center',choices=['adaptive','frozen'],default='adaptive')
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    suffix=f'{args.seed}_{args.center}'
    lb=load();g=lb.graded;w=lb.pi/lb.pi.sum()
    maps,cols=covariates(lb,args.seed)
    if args.smoke: maps={k:maps[k] for k in ('white','trees5','domain_mix')}
    (OUT/f'feature_manifest_{args.seed}.json').write_text(json.dumps({'features':cols,'maps':{k:v.shape[1] for k,v in maps.items()}},indent=2))
    files=[f for f in lb.s if f not in ('probe_zeros.csv','e06_r2f_only_eastern-wa.csv')]
    indices=list(range(files.index('p12_01_route_ratio.csv'),len(files)))
    if args.smoke:indices=indices[-3:]
    rows=[]
    for i in indices:
        best=min([f for f in files[:i] if not f.startswith('probe_')],key=lb.s.get) if args.center=='adaptive' else S.BASE
        center=lb.vec(best)[g]
        d=design(lb,files[:i+1],center,lb.s[best]);M=(d['A']*w[:,None]).T
        train=np.array([j for j,v in enumerate(d['keep']) if v<i])
        for name,Phi in maps.items():
            _,mean,variance,info=fit_kernel(M,d['z'],d['N'],Phi[g],center,train)
            # Full held-out score moment, including correlated measurement noise.
            Cmean=d['R'][:,-1]@mean
            v=lb.vec(files[i])[g]
            pred=np.sqrt(max(lb.AP(v*v)+lb.Z-2*Cmean,0))
            rows.append(dict(file=files[i],kernel=name,actual=lb.s[files[i]],predicted=pred,error=pred-lb.s[files[i]],
                             seed=args.seed,center=args.center,success=info['success'],tau_kernel=np.exp(info['hyper'][0]),
                             tau_white=np.exp(info['hyper'][1]),noise_scale=np.exp(info['hyper'][2])))
        print('holdout',i,files[i],flush=True)
        pd.DataFrame(rows).to_csv(OUT/f'cv_{suffix}{"_smoke" if args.smoke else ""}.csv',index=False)
    cv=pd.DataFrame(rows)
    print(cv.groupby('kernel').error.agg(mae=lambda x:x.abs().mean(),rmse=lambda x:np.sqrt(np.mean(x*x))).to_string(),flush=True)
    if args.smoke:return
    best=min(files,key=lb.s.get) if args.center=='adaptive' else S.BASE
    center=lb.vec(best)[g]
    d=design(lb,files,center,lb.s[best]);M=(d['A']*w[:,None]).T
    info_all={}
    for name,Phi in maps.items():
        p,_,_,info=fit_kernel(M,d['z'],d['N'],Phi[g],center)
        out=lb.vec(best).copy();out[g]=p
        np.save(OUT/f'{name}_{suffix}.npy',out)
        np.save(OUT/f'{name}_{suffix}_coefficients.npy',info.pop('coef'))
        info['rms_change']=float(np.sqrt(w@(p-center)**2));info_all[name]=info
    (OUT/f'models_{suffix}.json').write_text(json.dumps(info_all,indent=2))


if __name__=='__main__': main()
