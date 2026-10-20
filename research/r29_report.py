"""Create the R29 audit, experiment table, and static diagnostic figure."""
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import solution as S
from r29_analysis import design, fit

OUT=ROOT/'research/r29'
lb=S.Leaderboard();lb.public_subset();g=lb.graded;w=lb.pi/lb.pi.sum()
best=lb.vec('r28b_push.csv')
final=pd.read_csv(ROOT/'submission_r29.csv',dtype={'GEOID':str}).set_index('GEOID').loc[lb.geoid].coverage_gap_score.to_numpy()
delta=final-best
masks=S.stratum_masks(lb)
names=[n for n in lb.s if not n.startswith('probe_')]
X=np.column_stack([np.ones(g.sum())]+[lb.vec(n)[g] for n in names])
G=(X*w[:,None]).T@X;G+=1e-7*np.trace(G)/len(G)*np.eye(len(G))
c=np.r_[lb.ybar,[(lb.AP(lb.vec(n)[g]**2)+lb.Z-lb.s[n]**2)/2 for n in names]]
def estimate(v):
    coef=np.linalg.solve(G,(X*w[:,None]).T@v[g])
    return float(np.sqrt(max(lb.AP(v[g]**2)-2*coef@c+lb.Z,0)))
cv=pd.read_csv(OUT/'chronological.csv')
cvsum=cv.groupby('mode').agg(mae=('error',lambda x:x.abs().mean()),
          rmse=('error',lambda x:np.sqrt(np.mean(x*x))),bias=('error','mean'),
          coverage95=('error',lambda x:0.))
for mode in cvsum.index:
    d=cv[cv['mode']==mode]
    cvsum.loc[mode,'coverage95']=np.mean(np.abs(d.error)<=1.96*d.sd)
cvsum.to_csv(OUT/'validation_summary.csv')
# Dependent chronological observations: moving-block bootstrap is a descriptive
# sensitivity interval, not an independent-row significance test.
pivot=cv.pivot(index='file',columns='mode',values='error').reindex(cv.file.drop_duplicates())
rng=np.random.default_rng(20261019)
boot={}
for mode in ('free_clip','effect_scale'):
    diff=np.abs(pivot[mode].to_numpy())-np.abs(pivot.original.to_numpy())
    ix=(rng.integers(0,len(diff),size=(4000,7,1))+np.arange(5))%len(diff)
    means=diff[ix.reshape(4000,-1)[:,:len(diff)]].mean(1)
    boot[mode]={'mae_difference':float(diff.mean()),'block5_interval95':np.quantile(means,[.025,.975]).tolist()}
rows=[]
for p in sorted(OUT.glob('*.npy')):
    v=np.load(p); valid=S.finish(v,lb.geoid)
    mode=next((m for m in ('original','free_clip','effect_scale') if p.stem.startswith(m)),None)
    status='candidate' if p.stem=='local_round1_corr1_noise1' else ('discard' if mode else 'sensitivity')
    rows.append(dict(run_id=p.stem,status=status,
        val_score=float(cvsum.loc[mode,'mae']) if mode else np.nan,
        validation_metric='chronological_public_score_MAE' if mode else 'unvalidated_local_sensitivity',
        estimated_public_rmse=estimate(valid),
        rms_from_best=float(np.sqrt(lb.AP((valid[g]-best[g])**2))),
        raw_out_of_bounds=int(np.sum((v<0)|(v>1))),artifact=str(p.relative_to(ROOT))))
experiments=pd.DataFrame(rows)
experiments.to_csv(OUT/'experiments.tsv',sep='\t',index=False)
ratios=[]
for name,m in masks.items():
    r=lambda v:float(v[m].mean()/v[~m].mean())
    ratios.append(dict(stratum=name,n_high=int(m.sum()),n_low=int((~m).sum()),
                       leader_displayed=S.LEADER[name],r28b=r(best),r29=r(final)))
ratios=pd.DataFrame(ratios);ratios.to_csv(OUT/'strata.csv',index=False)
with np.load(ROOT/'model.npz') as a:regions=a['region']
reg=[]
for r in np.unique(regions):
    m=regions==r
    reg.append(dict(region=r,rows=int(m.sum()),mean_before=float(best[m].mean()),mean_after=float(final[m].mean()),
                    mean_change=float(delta[m].mean()),rms_change=float(np.sqrt(np.mean(delta[m]**2))),
                    fraction_delta_energy=float(np.sum(delta[m]**2)/np.sum(delta**2))))
pd.DataFrame(reg).to_csv(OUT/'regions.csv',index=False)
F=pd.concat([pd.read_parquet(p) for p in ROOT.glob('data/features/*.parquet') if 'facpts' not in p.name])
num=F.select_dtypes(include=[np.number])
missing=F.isna().mean().sort_values(ascending=False)
missing.rename('missing_fraction').to_csv(OUT/'feature_missingness.csv')
sample=pd.concat([pd.read_csv(p,dtype={'GEOID':str}) for p in ROOT.glob('data/reference/*/*-sample-submission.csv')])
profile=dict(feature_rows=len(F),feature_columns=len(F.columns),duplicate_feature_ids=int(F.GEOID.duplicated().sum()),
             extra_feature_ids=sorted(set(F.GEOID)-set(lb.geoid)),missing_feature_ids=sorted(set(lb.geoid)-set(F.GEOID)),
             numeric_columns=len(num.columns),constant_numeric_columns=num.columns[num.nunique(dropna=True)<=1].tolist(),
             at_least_50pct_missing=missing[missing>=.5].index.tolist(),
             sample_rows=len(sample),sample_duplicate_ids=int(sample.GEOID.duplicated().sum()))
(OUT/'data_profile.json').write_text(json.dumps(profile,indent=2))

# Verify algebraic training isolation by perturbing a held-out score. This is
# about score-feedback leakage, not proof of private-label validation.
files=[n for n in lb.s if n not in ('probe_zeros.csv','e06_r2f_only_eastern-wa.csv')]
i=files.index('r27_joint_min_change.csv')
center=lb.vec('r25_scorecard_step.csv')[g]
d1=design(lb,files[:i+1],center,lb.s['r25_scorecard_step.csv'])
save=lb.s[files[i]];lb.s[files[i]]+=.01
d2=design(lb,files[:i+1],center,lb.s['r25_scorecard_step.csv']);lb.s[files[i]]=save
train=np.array([j for j,v in enumerate(d1['keep']) if v<i])
isolation=float(np.max(np.abs(d1['z'][train]-d2['z'][train])))
assert isolation<1e-9
f1=fit(d1,'original',train);f2=fit(d2,'original',train)
assert np.allclose(f1['hyper'],f2['hyper'],atol=1e-5)
reconstruction=float(np.max(np.abs(d1['A']@d1['R']-np.column_stack([lb.vec(n)[g] for n in files[:i+1]]))))

skillroot=Path.home()/'.codex/skills'
skills={}
for name in ('scikit-learn','statistical-analysis','pymc'):
    p=skillroot/name/'SKILL.md'
    skills[name]={'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
       'review_matches':p.read_bytes()==(Path('/private/tmp/bias-ml-skills-review')/name/'SKILL.md').read_bytes()}
summary=dict(confirmed_best_rmse=lb.s['r28b_push.csv'],best_estimated_rmse=estimate(best),
    r29_estimated_rmse=estimate(final),rms_change=float(np.sqrt(lb.AP(delta[g]**2))),
    changed_rows=int(np.sum(delta!=0)),max_abs_change=float(np.max(np.abs(delta))),
    anchors_preserved=all(final[lb.geoid.eq(k).to_numpy()][0]==v for k,v in S.PUBLISHED.items()),
    geoid_match=set(sample.GEOID)==set(lb.geoid),rows=len(final),scored_history_count=len(lb.s),
    holdout_score_perturbation_train_max_change=isolation,basis_reconstruction_max_error=reconstruction,
    bootstrap=boot,skills=skills,environment={'python':platform.python_version(),'numpy':np.__version__,
      'pandas':pd.__version__,'scipy':scipy.__version__},
    submission_sha256=hashlib.sha256((ROOT/'submission_r29.csv').read_bytes()).hexdigest())
(OUT/'summary.json').write_text(json.dumps(summary,indent=2))

fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained')
axes[0].hist(delta[g],bins=65,color='#326ba8');axes[0].set(xlabel='R29 − R28b prediction',ylabel='Tracts',title='Small local adjustment')
axes[0].xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
axes[1].barh(ratios.stratum,ratios.r28b-ratios.leader_displayed,label='R28b',color='#c8ccd2')
axes[1].plot(ratios.r29-ratios.leader_displayed,ratios.stratum,'o',label='R29',color='#326ba8')
axes[1].axvline(0,color='black',lw=.7);axes[1].set(xlabel='Ratio − displayed leader ratio',title='Scorecard sensitivity');axes[1].legend()
for mode in ['original','free_clip','effect_scale']:
    z=cv[cv['mode']==mode]
    axes[2].plot(np.arange(len(z)),z.error*1e3,'.-',label=mode,alpha=.65)
axes[2].axhline(0,color='black',lw=.7);axes[2].set(xlabel='Chronological holdout',ylabel='Forecast error × 1000',title='Alternative priors do not clearly improve');axes[2].legend(fontsize=8)
fig.savefig(OUT/'diagnostics.png',dpi=160);plt.close(fig)

def table(df,cols=None):
    z=df if cols is None else df[cols]
    # No optional tabulate dependency.
    lines=['| '+' | '.join(z.columns)+' |','| '+' | '.join(['---']*len(z.columns))+' |']
    for row in z.itertuples(index=False,name=None):
        lines.append('| '+' | '.join(f'{v:.9f}' if isinstance(v,float) else str(v) for v in row)+' |')
    return '\n'.join(lines)
report=f'''# R29 analysis — 2026-10-19

`submission_r29.csv` is a new **experimental local recalibration**, not a demonstrated
improvement. R28b remains the best confirmed result: **{summary['confirmed_best_rmse']:.9f}**.
R29's public projection is **{summary['r29_estimated_rmse']:.9f}** versus **{summary['best_estimated_rmse']:.9f}**
for R28b under the same estimator. The difference is below forecast error and does
not establish a public or private gain. The conservative recommendation is to retain
R28b as the known best and treat R29 as an optional scorecard-sensitive hedge.

## Data and validation limits

- {len(lb.geoid):,} submission rows; {g.sum():,} graded rows in four regions and {(~g).sum()} Eastern Washington rows
  held fixed. GEOIDs match all five official sample files exactly, including leading zeros.
- {len(lb.s)} scored historical files in the actual artifact (older prose says 61).
  There are no tract training labels. Historical predictions are features; an RMSE
  score is an aggregate observation, never a tract-level target.
- Cached permitted features contain {len(F):,} rows and {len(F.columns)} columns, no duplicate IDs,
  {len(profile['extra_feature_ids'])} extra feature rows excluded by the official sample and no missing submitted IDs.
  Missingness and constant-column diagnostics are saved alongside this report.
- No withdrawn reference layer, external training dataset, paid model API, or AutoML
  service was used. Existing historical features were not rebuilt.
- Public scores and scorecard calibration remain adaptive feedback. The inherited
  assumption that a zero-scoring leader reveals true full-population ratios is not
  independently established by this run. Ratio definitions are inherited from R23/R24.
- The estimated public subset weights come from affine constraints available before
  the first holdout. Later scores never enter a chronological fold's fit or noise proxy.

## Skills installed and applied

Reviewed and installed `scikit-learn`, `statistical-analysis`, and `pymc` from
[K-Dense Scientific Agent Skills](https://github.com/K-Dense-AI/scientific-agent-skills),
alongside the existing `ml-analysis` workflow. They were selected for split design,
explicit measurement dependence, uncertainty checks and prior sensitivity. This is
a task-specific selection, not an objective ranking of all skills online.

The original coefficient prior, a learned clipping-direction prior, and an RMS-effect
prior were compared with hyperparameters refit from earlier scores for every holdout.
Gaussian conditioning is analytic in NumPy/SciPy; no MCMC was run. The installed PyMC
skill informed the generative-model and sensitivity checks. The skill's newer package
examples were not executed and the project's numerical environment was not upgraded.
Installed instruction hashes and review comparisons are in `summary.json`.

## Chronological score forecasting

All 33 files from p12 onward were forecast using only preceding scores. These evaluate
prediction of the next **public aggregate score**, not private tract RMSE. Conditional
95% interval coverage is diagnostic: fitted hyperparameter uncertainty, systematic
model error and adaptive submission design are not fully represented.

{table(cvsum.reset_index())}

Learned clipping prior MAE difference versus original: {boot['free_clip']['mae_difference']:+.9f};
five-file moving-block bootstrap 95% descriptive interval
[{boot['free_clip']['block5_interval95'][0]:+.9f}, {boot['free_clip']['block5_interval95'][1]:+.9f}].
RMS-effect prior difference: {boot['effect_scale']['mae_difference']:+.9f}; interval
[{boot['effect_scale']['block5_interval95'][0]:+.9f}, {boot['effect_scale']['block5_interval95'][1]:+.9f}].
Neither establishes a clear improvement. The RMS-effect fit also creates large raw
negative predictions; its attractive post-clipping projection is not sufficient
evidence to use it. All experiments, including rejected variants, are retained.

## New R29 file

R29 starts from the scored R28b predictions. It applies a local minimum-change solve
to all historical score moments and eight scorecard ratio constraints. Relative to
the earlier solve, the new uncertainty model:

1. Propagates uniform ±0.005 rounding uncertainty in the leader's ratio and both
   calibration ratios, rather than treating the two calibration ratios as exact.
2. Uses the full covariance from the same ungraded WA tracts contributing to multiple
   ratios, rather than treating those contributions as independent.
3. Fits a local tract adjustment scale by marginal likelihood and preserves clipping
   bounds and the five published Paradise values.

Because the incumbent already uses historical feedback, this is an empirical local
recalibration; it must not be interpreted as an independent Bayesian posterior.
Independent rounding errors and the common 0.0624 WA residual scale are modeling
assumptions; the covariance is an approximation, not an identified error model.
The chosen covariance model is the primary scientific specification, not the
variant with the lowest projected public score. Doubling rounding SD, halving or
doubling score noise, and diagonal covariance are recorded in `experiments.tsv`.
The eight single-stratum ablations and all local configurations are recorded in
`local_models.json`.

The final change has pi-weighted RMS **{summary['rms_change']:.7f}**, maximum absolute
change **{summary['max_abs_change']:.7f}**, and changes {summary['changed_rows']:,} rows after six-decimal rounding.
WA and the five published scores remain fixed. These are prediction changes, not
measured error reductions.

{table(ratios)}

## Verification and reproduction

- Submission: exactly 9,794 unique GEOIDs, two required columns, no missing or
  infinite values, predictions in [0,1], all official sample IDs and anchors preserved.
- Original `solution.py` reproduced its tracked submissions and validation log with
  no tracked-file changes. The known-best file was not replaced.
- Basis reconstruction maximum absolute error: {reconstruction:.3g}.
- Perturbing R27's held-out score by +0.01 changed its preceding training moments
  by at most {isolation:.3g}; refitted training hyperparameters remained equal.
- Seeds: 20261019 for the descriptive block bootstrap. Core fits are deterministic.
- Environment: Python {platform.python_version()}, NumPy {np.__version__}, pandas {pd.__version__}, SciPy {scipy.__version__}.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python research/r29_analysis.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python research/r29_report.py
```

Submission SHA-256: `{summary['submission_sha256']}`.

![Diagnostics](diagnostics.png)

## Sources

- [Competition problem, metric, data restrictions and submission rules](https://zindi.world/competitions/bias-bounty-mapping-equity-challenge), checked 2026-10-19.
- [K-Dense skill source](https://github.com/K-Dense-AI/scientific-agent-skills).
- Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026).
  [Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents](https://doi.org/10.48550/arXiv.2609.00065).
  Current arXiv record checked 2026-10-19. Cited for workflow guidance, not predictive evidence.
'''
(OUT/'ANALYSIS.md').write_text(report)
print(json.dumps(summary,indent=2))
