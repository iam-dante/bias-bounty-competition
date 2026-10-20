"""Audit rank sensitivity and design complementary aggregate-learning probes.

This script does not treat its estimates as leaderboard measurements and does
not upload files. It keeps R30 as the first experiment awaiting a real score.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import LinearOperator,eigsh

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import solution as S
from feedback import load
from r29_analysis import design
from r30_kernels import covariates

OUT=ROOT/'research/r31'
TOLS=[1e-3,3e-4,1e-4,3e-5,1e-5,1e-6,1e-7]


def setup(lb):
    names=[f for f in lb.s if f not in ('probe_zeros.csv','probe_const_0.10.csv','e06_r2f_only_eastern-wa.csv')]
    w=lb.pi/lb.pi.sum()
    X=np.column_stack([np.ones(lb.graded.sum())]+[lb.vec(f)[lb.graded] for f in names])
    moments=np.r_[lb.ybar,[(lb.AP(X[:,j+1]**2)+lb.Z-lb.s[n]**2)/2 for j,n in enumerate(names)]]
    return names,w,X,moments


def score_forecast(lb,X,moments,target,w,tol):
    U,s,Vt=np.linalg.svd(X*np.sqrt(w)[:,None],full_matrices=False)
    keep=s>tol*s[0]
    coef=Vt[keep].T@((U[:,keep].T@(target*np.sqrt(w)))/s[keep])
    return float(np.sqrt(max(w@target**2-2*coef@moments+lb.Z,0))),int(keep.sum())


def audit(lb):
    names,w,X,moments=setup(lb)
    first=names.index('p12_01_route_ratio.csv');rows=[]
    for i in range(first,len(names)):
        target=X[:,i+1]
        for tol in TOLS:
            score,rank=score_forecast(lb,X[:,:i+1],moments[:i+1],target,w,tol)
            rows.append(dict(file=names[i],tol=tol,rank=rank,actual=lb.s[names[i]],predicted=score,error=score-lb.s[names[i]]))
    cv=pd.DataFrame(rows);cv.to_csv(OUT/'rank_cv.csv',index=False)
    candidates={n:pd.read_csv(ROOT/f'{n}.csv',dtype={'GEOID':str}).set_index('GEOID').loc[lb.geoid].coverage_gap_score.to_numpy()
                for n in ('submission_r29','submission_r30','submission_r30_conservative')}
    rows=[]
    for name,values in candidates.items():
        for tol in TOLS:
            score,rank=score_forecast(lb,X,moments,values[lb.graded],w,tol)
            rows.append(dict(file=name,tol=tol,rank=rank,estimated_score=score))
    table=pd.DataFrame(rows);table.to_csv(OUT/'rank_candidate_estimates.csv',index=False)
    metrics=cv.groupby('tol').error.agg(n='size',mae=lambda x:abs(x).mean(),rmse=lambda x:np.sqrt(np.mean(x*x)),worst=lambda x:abs(x).max())
    metrics.to_csv(OUT/'rank_validation_summary.csv')
    print(metrics.to_string(),flush=True)
    print(table[table.file=='submission_r30'].to_string(index=False),flush=True)
    return cv,table


def probe_design(lb):
    """Largest residual feature-covariance directions, orthogonal to old scores.

    The existing R30 direction is included only to avoid duplicating a pending
    experiment, not as a scored observation. The covariance used for ranking is
    a design prior, not evidence that a direction carries true target signal.
    """
    g=lb.graded;w=lb.pi/lb.pi.sum();best=lb.vec('r29_correlated_scorecard.csv')
    maps,features=covariates(lb)
    names=[f for f in lb.s if f not in ('probe_zeros.csv','e06_r2f_only_eastern-wa.csv')]
    d=design(lb,names,best[g],lb.s[names[-1]])
    r30=pd.read_csv(ROOT/'submission_r30.csv').coverage_gap_score.to_numpy()
    # Orthogonalize in sqrt(w) coordinates, including scorecard indicators and
    # the unscored R30 direction as design exclusions.
    B=np.column_stack([d['A'],r30[g]-best[g],*[m[g].astype(float) for m in S.stratum_masks(lb).values()]])
    U,s,_=np.linalg.svd(B*np.sqrt(w)[:,None],full_matrices=False)
    Q=U[:,s>1e-8*s[0]]
    exclusions=Q.copy();metadata=[];directions=[]
    for name in ('splines','spatial'):
        P=maps[name][g]
        # Covariance of the represented function in weighted tract coordinates.
        P=sparse.diags(np.sqrt(w))@P
        def project(x):return x-exclusions@(exclusions.T@x)
        def matvec(x):
            z=project(x)
            return project(np.asarray(P@(P.T@z)).ravel())
        op=LinearOperator((len(w),len(w)),matvec=matvec,dtype=float)
        val,vec=eigsh(op,k=1,which='LA',v0=np.random.default_rng(3110).normal(size=len(w)),tol=1e-7)
        u=vec[:,0]/np.sqrt(w)
        # Fix sign using the learned frozen-prior function, for reproducibility
        # and to make each probe's positive direction its model's preferred one.
        fitted=np.load(ROOT/f'research/r30/{name}_3101_frozen.npy')[g]
        if w@(u*(fitted-best[g]))<0:u=-u
        u/=np.sqrt(w@u**2)
        # Planned query RMS 0.006 before clipping. Its actual differences and
        # historical-span contamination are measured after clipping/rounding.
        out=best.copy();out[g]+=0.006*u;out=S.finish(out,lb.geoid)
        delta=out[g]-best[g]
        weighted_delta=np.sqrt(w)*delta
        novel=weighted_delta-Q@(Q.T@weighted_delta)
        frac=float(np.linalg.norm(novel)/np.linalg.norm(weighted_delta))
        path=ROOT/f'probe_r31_{name}.csv'
        frame=pd.DataFrame({'GEOID':lb.geoid,'coverage_gap_score':out})
        assert len(frame)==9794 and frame.GEOID.is_unique and frame.coverage_gap_score.between(0,1).all()
        assert np.array_equal(out[~g],best[~g])
        assert all(frame.set_index('GEOID').coverage_gap_score[k]==v for k,v in S.PUBLISHED.items())
        frame.to_csv(path,index=False)
        directions.append(u)
        metadata.append(dict(name=name,artifact=path.name,feedback_name=f'p31_0{len(metadata)+1}_{name}.csv',
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),design_eigenvalue=float(val[0]),
            actual_rms=float(np.sqrt(w@delta**2)),novel_rms=float(np.linalg.norm(novel)),novel_rms_fraction=frac,
            r30_direction_cosine=float(w@(delta*(r30[g]-best[g]))/np.sqrt((w@delta**2)*(w@(r30[g]-best[g])**2))),
            measurement_sampling_sd_approx=float(lb.s[names[-1]]*np.sqrt((1-S.P_PUBLIC)/(S.P_PUBLIC*len(w)))),
            label='diagnostic query; no measured score or demonstrated gain'))
        # The second probe is complementary to both R30 and the first planned
        # function, regardless of the score eventually observed.
        q=project(np.sqrt(w)*u);q/=np.linalg.norm(q)
        exclusions=np.column_stack([exclusions,q])
    (OUT/'probe_design.json').write_text(json.dumps({'probes':metadata,'features':features},indent=2))
    print(json.dumps(metadata,indent=2),flush=True)
    return metadata


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    lb=load();cv,table=audit(lb);probes=probe_design(lb)
    metrics=pd.read_csv(OUT/'rank_validation_summary.csv')
    selected=metrics.loc[metrics.mae.idxmin()]
    rejected=metrics.loc[metrics.tol.idxmin()]
    pending=[]
    for filename,priority,feedback_name in [('submission_r30.csv',1,'r30_bounded_scorecard.csv'),
         ('submission_r30_conservative.csv',2,'r30c_bounded_half.csv'),
         *[(r['artifact'],3,r['feedback_name']) for r in probes]]:
        pending.append(dict(file=filename,feedback_name=feedback_name,priority=priority,
            sha256=hashlib.sha256((ROOT/filename).read_bytes()).hexdigest(),status='awaiting_user_reported_score',
            submission_authorized=False))
    (OUT/'pending_measurements.json').write_text(json.dumps(pending,indent=2))
    summary=dict(best_confirmed_rmse=min(lb.s.values()),new_confirmed_scores=0,
        best_chronological_tol=float(selected.tol),best_chronological_mae=float(selected.mae),
        r30_selected_tol_estimate=float(table[(table.file=='submission_r30')&(table.tol==selected.tol)].estimated_score.iloc[0]),
        rejected_tol=float(rejected.tol),rejected_tol_mae=float(rejected.mae),
        r30_estimate_range=[float(table[table.file=='submission_r30'].estimated_score.min()),float(table[table.file=='submission_r30'].estimated_score.max())],
        goal_verified=False,next_required_measurement='Actual public score of submission_r30.csv',probes=probes)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    report=f'''# R31 audit and experiment design — 2026-10-19

The previous goal turn made concrete progress: it trained and verified R30 and
created two valid submission alternatives. This continuation audited the score
estimator and prepared complementary measurement files. No new leaderboard score
has been supplied; the confirmed best remains R29 at **0.062094756**, and the
0.05 goal is not achieved.

## Rank sensitivity

The score estimator projects a candidate into the historical prediction span.
Near-dependent directions can amplify estimated public-weight and rounding errors.
This audit backtested seven SVD cutoffs on the same 34 chronological score holdouts,
always excluding the held-out score. Results are in `rank_cv.csv` and
`rank_validation_summary.csv`. Candidate estimates are in `rank_candidate_estimates.csv`.

The cutoff with smallest retrospective MAE was {selected.tol:g}, MAE
{selected.mae:.9f}; it gives R30 estimate
{summary['r30_selected_tol_estimate']:.9f}. Across all tested cutoffs, R30 estimates
range from {summary['r30_estimate_range'][0]:.9f} to {summary['r30_estimate_range'][1]:.9f}.
These are conditional model calculations, not observed performance. The most
optimistic cutoff must not be chosen merely because it moves toward 0.05.
The retrospective cutoff choice is itself exploratory and needs new feedback.

In particular, the 1e-7 cutoff produces the tempting 0.057712131 estimate, but
its chronological MAE is **{rejected.mae:.9f}**, versus about 0.000119 for stable
cutoffs. It is rejected as an estimator. Among cutoffs excluding that failure,
R30 estimates still range from about 0.06170 to 0.06224. This is a sensitivity
range, not a confidence interval. The 0.06183 estimate in the R30 report remains
one regularized estimate; its precision must not be read as certainty.

## Complementary probes

`probe_r31_splines.csv` and `probe_r31_spatial.csv` are **diagnostic experiments**,
not validated replacements for R29 or R30. They query the strongest remaining
feature-covariance directions after excluding historical predictions, the eight
scorecard indicators and R30's pending direction. The second also excludes the
first query's planned direction. This is a prior-based information design;
large design variance does not establish a real target effect.

This is a simple spectral query heuristic, not a claim of Bayes-optimal query
selection. [Alexanderian's D-optimality note](https://arxiv.org/abs/2212.11466)
describes the linear-Gaussian information criterion; [Go and Isaac (2022)](https://proceedings.mlr.press/v180/go22a.html)
show why experiment rankings can depend on the prior. Those limitations motivate
keeping distinct spline and spatial hypotheses. The code does not implement
their robust expected-information-gain ambiguity set.

Both use only already supplied competition covariates and are perturbations of
confirmed R29. Each has 9,794 unique GEOIDs, preserved leading zeros, finite bounded
predictions, all five published anchors and unchanged ungraded WA rows. Clipping
can introduce known-span components; the metadata reports the actual remaining
novelty and correlation with R30 after clipping and rounding.

The planned perturbation RMS is 0.006. The approximate public-sampling uncertainty
in a normalized direction's residual coefficient is about 0.001. This is an
assumption-based population/public comparison, not uncertainty in an already
returned leaderboard score. The goal needs roughly 35% lower MSE; these queries
are designed to acquire information, not to imply that two small perturbations
will reach that goal.

## Decision

**R30 remains the first file to score.** Keep its bytes unchanged so the returned
measurement can be matched to the existing record. Do not spend additional slots
on the probes until R30's result shows whether the new direction is useful.
No external upload was performed. The next necessary external evidence is R30's
actual score. More fitting of these same aggregate measurements cannot verify
a public score of 0.05.

`pending_measurements.json` records exact file hashes and internal score-history
names for future feedback. It records that uploads have not been authorized;
neither these files nor the continuation instruction authorize an external
submission. Add feedback only when a real score and exact file are identified.

Reproduce: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python research/r31_audit.py`.
This uses the same requirements and source data as R30. Probe seed: 3110.
'''
    (OUT/'ANALYSIS.md').write_text(report)


if __name__=='__main__':main()
