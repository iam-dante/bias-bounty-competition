"""Verify R30, write submissions, and build an evidence-backed experiment report."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.linalg import cho_factor,cho_solve
from scipy.optimize import minimize

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import solution as S
from feedback import load
from r29_analysis import design
from r30_bounded import bounded

OUT=ROOT/'research/r30'
lb=load();g=lb.graded;w=lb.pi/lb.pi.sum()
best=lb.vec('r29_correlated_scorecard.csv')
candidate=np.load(OUT/'bounded_scorecard.npy')
files=[n for n in lb.s if not n.startswith('probe_')]
X=np.column_stack([np.ones(g.sum())]+[lb.vec(n)[g] for n in files])
G=(X*w[:,None]).T@X;G+=1e-7*np.trace(G)/len(G)*np.eye(len(G))
c=np.r_[lb.ybar,[(lb.AP(lb.vec(n)[g]**2)+lb.Z-lb.s[n]**2)/2 for n in files]]
def estimate(v):
    coef=np.linalg.solve(G,(X*w[:,None]).T@v[g])
    return float(np.sqrt(max(lb.AP(v[g]**2)-2*coef@c+lb.Z,0)))
sample=pd.concat([pd.read_csv(p,dtype={'GEOID':str}) for p in ROOT.glob('data/reference/*/*-sample-submission.csv')])
checks={}
for tag,factor in [('submission_r30',1.),('submission_r30_conservative',.5)]:
    v=S.finish(best+factor*(candidate-best),lb.geoid)
    output=pd.DataFrame({'GEOID':lb.geoid,'coverage_gap_score':v})
    assert len(output)==9794 and output.GEOID.is_unique and set(output.GEOID)==set(sample.GEOID)
    assert np.isfinite(v).all() and output.coverage_gap_score.between(0,1).all()
    assert all(output.set_index('GEOID').coverage_gap_score[k]==value for k,value in S.PUBLISHED.items())
    assert np.array_equal(v[~g],best[~g])
    path=ROOT/f'{tag}.csv';output.to_csv(path,index=False)
    checks[tag]=dict(rows=len(v),estimated_public_rmse=estimate(v),
        rms_change=float(np.sqrt(lb.AP((v[g]-best[g])**2))),max_abs_change=float(np.max(np.abs(v-best))),
        changed_rows=int(np.sum(v!=best)),sha256=hashlib.sha256(path.read_bytes()).hexdigest())

# Use the current frozen/adaptive file naming, accepting the preserved first run.
adaptive=OUT/'cv_3101_adaptive.csv'
if not adaptive.exists():adaptive=OUT/'cv_3101.csv'
ca=pd.read_csv(adaptive).assign(center='adaptive')
cf=pd.read_csv(OUT/'cv_3101_frozen.csv').assign(center='frozen')
kb=pd.concat([ca,cf],ignore_index=True)
kbsum=kb.groupby(['center','kernel']).error.agg(n='size',mae=lambda x:abs(x).mean(),rmse=lambda x:np.sqrt(np.mean(x*x)))
kbsum.to_csv(OUT/'kernel_validation.csv')
cv=pd.read_csv(OUT/'cv_bounded.csv')
bsum=cv.groupby(['factor','noise_factor','scorecard']).error.agg(n='size',mae=lambda x:abs(x).mean(),rmse=lambda x:np.sqrt(np.mean(x*x)))
bsum.to_csv(OUT/'bounded_validation.csv')
control=cf[cf.kernel=='white'].set_index('file').error
primary=cv[(cv.factor==1)&(cv.noise_factor==1)&(~cv.scorecard)].set_index('file').error
diff=primary.abs()-control.loc[primary.index].abs()
rng=np.random.default_rng(3104)
idx=(rng.integers(0,len(diff),size=(4000,7,1))+np.arange(5))%len(diff)
samples=diff.to_numpy()[idx.reshape(4000,-1)[:,:len(diff)]].mean(1)
comparison=dict(mae_difference=float(diff.mean()),block5_interval95=np.quantile(samples,[.025,.975]).tolist())

variants=[]
for path in sorted(OUT.glob('*.npy')):
    if 'coefficients' in path.stem:continue
    v=np.load(path)
    if v.shape!=best.shape:continue
    valid=S.finish(v,lb.geoid)
    variants.append(dict(run_id=path.stem,status='candidate' if path.stem=='bounded_scorecard' else 'diagnostic',
         estimated_public_rmse=estimate(valid),rms_from_r29=float(np.sqrt(lb.AP((valid[g]-best[g])**2))),
         raw_out_of_bounds=int(np.sum((v<0)|(v>1))),artifact=str(path.relative_to(ROOT))))
pd.DataFrame(variants).to_csv(OUT/'experiments.tsv',sep='\t',index=False)
pd.concat([kb.assign(family='kernel'),cv.assign(family='bounded')],ignore_index=True).to_csv(OUT/'all_chronological_forecasts.csv',index=False)

# Independent small primal solve verifies dual implementation, not scientific accuracy.
rng=np.random.default_rng(3105);n,k=45,7
Mt=rng.normal(size=(k,n))/n;truth=rng.uniform(0,.4,n);zt=Mt@truth
Nt=np.diag(rng.uniform(.001,.003,k)**2);base=rng.normal(.1,.15,n);tau=.17
yd,_,diagnostics=bounded(Mt,zt,Nt,base,tau,1.)
chol=cho_factor(Nt,lower=True)
def primal(y):
    residual=Mt@y-zt
    loss=.5*np.sum((y-base)**2)/tau**2+.5*residual@cho_solve(chol,residual)
    grad=(y-base)/tau**2+Mt.T@cho_solve(chol,residual)
    return loss,grad
opt=minimize(primal,np.clip(base,0,1),jac=True,bounds=[(0,1)]*n,method='L-BFGS-B',
            options={'maxiter':3000,'ftol':1e-13,'gtol':1e-8})
solver_error=float(np.max(np.abs(opt.x-yd)))
assert opt.success and solver_error<1e-5 and diagnostics['dual_gap']<1e-6

# Held-out R29 score must not affect earlier training moments, even through the
# chronological basis construction.
ordered=[f for f in lb.s if f not in ('probe_zeros.csv','e06_r2f_only_eastern-wa.csv')]
center=lb.vec(S.BASE)[g]
d1=design(lb,ordered,center,lb.s[S.BASE]);train=np.array([j for j,v in enumerate(d1['keep']) if v<len(ordered)-1])
score=lb.s[ordered[-1]];lb.s[ordered[-1]]+=.01
d2=design(lb,ordered,center,lb.s[S.BASE]);lb.s[ordered[-1]]=score
isolation=float(np.max(np.abs(d1['z'][train]-d2['z'][train])))
assert isolation<1e-9

ratios=[]
for name,m in S.stratum_masks(lb).items():
    ratios.append(dict(stratum=name,leader_displayed=S.LEADER[name],r29=float(best[m].mean()/best[~m].mean()),
                       r30=float(candidate[m].mean()/candidate[~m].mean())))
pd.DataFrame(ratios).to_csv(OUT/'strata.csv',index=False)
bounded_models=json.loads((OUT/'models_bounded.json').read_text())
sens=[]
for tag,info in bounded_models.items():
    v=np.load(OUT/f'{tag}.npy');dv=v[g]-best[g];dp=candidate[g]-best[g]
    sens.append(dict(model=tag,**info,rms_from_primary=float(np.sqrt(lb.AP((v[g]-candidate[g])**2))),
                     direction_cosine=float(lb.AP(dv*dp)/np.sqrt(lb.AP(dv**2)*lb.AP(dp**2)))))
pd.DataFrame(sens).to_csv(OUT/'bounded_sensitivity.csv',index=False)
summary=dict(goal_rmse=.05,best_confirmed_rmse=lb.s[ordered[-1]],
       required_mse_reduction=float(1-.05**2/lb.s[ordered[-1]]**2),
       n_historical_scores=len(lb.s),n_orthogonal_measurements=len(d1['z']),
       chronological_kernel_forecasts=len(kb),chronological_bounded_forecasts=len(cv),
       comparison=comparison,submission_checks=checks,solver_primal_dual_max_error=solver_error,
       heldout_score_training_moment_change=isolation,
       baseline_projection_estimate=estimate(best),
       primary_max_cv_duality_gap=float(cv.dual_gap.abs().max()))
(OUT/'summary.json').write_text(json.dumps(summary,indent=2))

def table(df):
    lines=['| '+' | '.join(map(str,df.columns))+' |','| '+' | '.join(['---']*len(df.columns))+' |']
    for row in df.itertuples(index=False,name=None):
        lines.append('| '+' | '.join(f'{v:.9f}' if isinstance(v,float) else str(v) for v in row)+' |')
    return '\n'.join(lines)
bm=bsum.reset_index();r30=checks['submission_r30']
report=f'''# R30 modeling study — 2026-10-19

The confirmed best is **R29, RMSE 0.062094756**, reported by the user. A new
**unscored** candidate is `submission_r30.csv`. Its historical-span projection is
**{r30['estimated_public_rmse']:.9f}**, compared with **{summary['baseline_projection_estimate']:.9f}** for R29
under the same estimator. These are model estimates, not new leaderboard results.
The 0.05 goal has **not** been reached; it requires 35.16% lower squared error.

## What was trained

This round executed **{len(kb)} chronological kernel forecasts** (nine models × two
prior centers × 34 held-out scores) and **{len(cv)} bounded-model forecasts**, plus
the full-data fits and sensitivity variants. No tract-level labels were invented.
The original 60-score `model.npz` remains immutable. `research/feedback.py` adds
the confirmed R29 score after verifying the exact submission SHA-256 against
`research/score_feedback.json`.

Inputs are supplied competition data: Overture-derived counts and route measures,
ACS housing, population, supplied vulnerability/hazard strata, county IDs and
centroids of supplied tract geometries. There were no new external datasets,
withdrawn reference layers, paid inference APIs, AutoML runs or external uploads.
The initial smoke run failed because regional/national tract geometry tables
contained duplicate IDs; de-duplicating IDs before alignment fixed it. Full runs
then used all 9,379 graded tracts and verified the 9,794-row submission schema.

## Research and model choice

[Law et al. (NeurIPS 2018)](https://proceedings.neurips.cc/paper/2018/hash/24b43fb034a10d78bec71274033b4096-Abstract.html)
and [Zhang et al. (NeurIPS 2020)](https://proceedings.neurips.cc/paper_files/paper/2020/file/5b0fa0e4c041548bb6289e15d865a696-Paper.pdf)
motivate learning individual functions through an explicit aggregate observation
model. Here, RMSE feedback is converted to signed linear target moments. This
adaptation uses correlated Gaussian measurement noise; it does not inherit the
papers' guarantees because the observations are adaptive, overlapping and sampled.

I trained linear, spline, random-tree leaf (depths 3/5/7), regional-interaction,
spatial Fourier/county and combined feature covariance models, with an iid control.
Random tree partitions use covariates without targets. The Gaussian posterior is
computed analytically; hyperparameters are refit before every held-out score.
An adaptive prior uses the earlier best submission; a frozen prior uses R12.
Feature transforms use the available, unlabeled competition covariates. There
is no supervised tract split because there are no tract training labels.

[TabPFN](https://www.nature.com/articles/s41586-024-08328-6) was considered but not
used: its ordinary supervised interface requires individual labels. Training it,
CatBoost or a neural net against our current predictions would primarily reproduce
our own assumptions. The earlier R26 MLP already failed its actual leaderboard
test; it was not rebranded as new evidence. The
[DRO regression literature](https://www.jmlr.org/papers/v19/17-295.html) informed
regularization sensitivity, but this code does not implement that paper's
Wasserstein uncertainty set.

The selected new method is **box-constrained aggregate regression**. Its objective is

`0.5 * (M y - z)' R^-1 (M y - z) + 0.5 * ||y - physics_base||² / tau²`,

subject to `0 <= y <= 1`, with published anchors fixed for final fitting. `M` maps
tract predictions to scored moments, `z` is reconstructed from scores, and `R`
models public-sampling uncertainty. For the final model, eight rounded scorecard
ratios add soft linear constraints with correlated WA uncertainty. The ratio
interpretation and the leader-as-truth assumption are inherited from earlier
research and remain unverified assumptions.

This is more than clipping once: a 46-ish-dimensional convex dual repeatedly
updates all tract predictions and aggregate residuals until the bounded solution
is consistent. Actual moment rank is {summary['n_orthogonal_measurements']}. The local kernel and noise
scales are empirical estimates. Bounds provide real target support, not new labels.

## Validation

Each held-out score is excluded from fitting, covariance calibration and prior
selection for that fold. No scorecard constraints enter the 34-file main comparison.
The separate five-file scorecard test begins after R24, when both R23/R24 calibration
scorecards were already available. This is exploratory backtesting of public score
prediction, **not private-tract cross-validation or an untouched final test**.

{table(kbsum.reset_index())}

{table(bm)}

Primary bounded-model MAE difference versus frozen iid control:
**{comparison['mae_difference']:+.9f}**, with five-file moving-block bootstrap 95%
descriptive interval [{comparison['block5_interval95'][0]:+.9f}, {comparison['block5_interval95'][1]:+.9f}].
The five scorecard holdouts are few and closely related; their small errors do not
establish performance on a substantially new prediction direction. Nonlinear
kernels gave mixed results and mostly shrank toward the control. They were not
selected simply for a low in-sample score estimate.

## Candidate and sensitivity

R30 uses the primary bounded model with all current moments and the eight soft
scorecard constraints. It changes predictions by weighted RMS **{r30['rms_change']:.6f}**
from R29, maximum **{r30['max_abs_change']:.6f}**. `submission_r30_conservative.csv`
is an optional 50% blend with R29, not another confirmed improvement. Neither is
submitted automatically. R29 remains the known best until a new score is returned.

The primary specification was retained rather than selecting the most favorable
projected score among sensitivity fits. Prior scale factors 0.25/0.5/1/2 and noise
factors 0.5/1/2 were evaluated. Full configurations, convergence, dual gaps and
direction changes are in `bounded_sensitivity.csv`; all full-fit artifacts and
estimates are in `experiments.tsv`. Older provisional files remain diagnostic
artifacts and are excluded from chronological counts.

{table(pd.DataFrame(ratios))}

The frozen-prior scales approach their upper search bound, and the row-weight
model is approximate. The public score estimate assumes no target signal in
directions outside historical predictions; nonlinear clipping and scorecard moves
can violate that assumption. The 0.06183 estimate should therefore be treated as
an experiment to test, not a forecast accurate to nine decimal places.

## Verification and next step

- Both files have exactly 9,794 unique official GEOIDs, preserved leading zeros,
  two required columns, finite bounded predictions and all five published anchors.
  Eastern Washington values are unchanged from R29.
- Independent small primal and dual QP implementations agreed to maximum absolute
  prediction difference {solver_error:.3g}. This checks numerical implementation,
  not hidden-target accuracy. Maximum chronological dual gap: {summary['primary_max_cv_duality_gap']:.3g}.
- Perturbing the held-out R29 score by +0.01 changed preceding training moments
  by at most {isolation:.3g}. The held-out score does not train its forecast.
- The numerical requirements include Shapely 2.1.2 for supplied tract geometry
  features; the other packages retain their existing pinned versions.
  CPU only; tree/spatial seed 3101, bootstrap 3104, solver check 3105.
- The most informative next observation is the actual public score of
  `submission_r30.csv`. A new scored direction can be incorporated; repeatedly
  fitting the same 61 measurements cannot verify that 0.05 is attainable.

Reproduce:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python research/r30_kernels.py --center adaptive
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python research/r30_kernels.py --center frozen
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python research/r30_bounded.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python research/r30_finalize.py
```

R30 SHA-256: `{r30['sha256']}`.

Workflow skills: existing `ml-analysis` and `co-scientist`, plus installed
`scikit-learn`, `statistical-analysis`, and `pymc`. Their guidance informed
validation, prior sensitivity and uncertainty reporting; PyMC MCMC was not run.
Credit: Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026),
[Scientific Agent Skills](https://doi.org/10.48550/arXiv.2609.00065).
[Competition rules](https://zindi.world/competitions/bias-bounty-mapping-equity-challenge)
and the primary research sources were checked online during this round.
'''
(OUT/'ANALYSIS.md').write_text(report.replace('a 46-ish-dimensional convex dual','a small convex dual'))
print(json.dumps(summary,indent=2))
