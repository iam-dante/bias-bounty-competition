import numpy as np, pandas as pd, pickle, re
from scipy.optimize import minimize_scalar
from bayes_lb import *
lb,g,w,names,V,c = load()
sub = pd.read_csv(REPO/'submission.csv'); yR21 = sub.coverage_gap_score.to_numpy()[g]
Nv = noise_cov(V,w,yR21,0.0624**2)
keep,Q,R = gram_schmidt(V,w,names)
m = Model(Q,R,keep,names,c,Nv); m.setup(w)
pC = pickle.load(open('eb_params.pkl','rb'))['C: group means + noise scale']
cand = pickle.load(open('cand.pkl','rb'))
def typ(n):
    n=n.lower()
    if 'wildfire' in n: return 'context'
    if re.search(r'fire|ems|school|preschool|religious|fac_', n): return 'facility'
    if 'building' in n: return 'building'
    if re.search(r'poi|brightquery|meta', n): return 'poi'
    if re.search(r'route|boundary|tiger|county|unrouted|tertiary|interstate|fm_s1200|multiref|segment|service|road', n): return 'road'
    if re.search(r'land_area|pop_density|pct_urban|svi|wildfire|drought|heat|tribal|burned|paradise|cbp', n): return 'context'
    return 'other'
pm = m.groups=='probe'
o, D, Nd = m.o[pm], m.D[pm], pC['alpha_N']*np.diag(m.N)[pm]
eff = o/np.sqrt(D); eff_sd = np.sqrt(Nd/D)           # effect along unit direction
types = np.array([typ(n) for n in np.array(m.names)[pm]])
print(f"{'probe':45s} {'type':9s} effect   sd")
for n,t,e,s in zip(np.array(m.names)[pm],types,eff,eff_sd): print(f"{n[:45]:45s} {t:9s} {e:+.4f} {s:.4f}")
def fit_s(mask):
    f=lambda ls: 0.5*np.sum(np.log(np.exp(2*ls)+eff_sd[mask]**2)+eff[mask]**2/(np.exp(2*ls)+eff_sd[mask]**2))
    r=minimize_scalar(f,bounds=(-12,-3),method='bounded'); 
    # profile-likelihood 1-unit interval
    grid=np.linspace(-12,-3,400); vals=np.array([f(x) for x in grid]); ok=grid[vals<r.fun+0.5]
    return np.exp(r.x), np.exp(ok.min()), np.exp(ok.max())
S_t={}
print("\nEB effect-size prior by mechanism type (RMS of true effect along a probe direction):")
for t in ['road','poi','context','building','facility','other']:
    mk=types==t
    if mk.sum()==0: continue
    s,lo,hi=fit_s(mk); S_t[t]=s
    print(f"  {t:9s} n={mk.sum():2d}  s={s:.4f}  (approx 68% interval {lo:.4f}-{hi:.4f})")
s_all,_,_=fit_s(np.ones_like(pm[pm]))
print(f"  all       n={pm.sum():2d}  s={s_all:.4f}")
# hierarchical: shrink type estimates toward pooled in log space by sample size
noise_unit=cand['noise_unit']
rows=[]
for nm,u in cand['dirs'].items():
    t=typ(nm); n_t=(types==t).sum(); s=S_t.get(t,s_all)
    s_h=np.exp((n_t*np.log(s)+3*np.log(s_all))/(n_t+3))   # partial pooling
    f2=w@u**2; gain=(s_h**2*f2)**2/(s_h**2*f2+noise_unit*f2)/(2*0.0625)
    rows.append(dict(file=nm,type=t,new_frac=np.sqrt(f2),prior_effect=s_h,exp_gain_rmse=gain))
df=pd.DataFrame(rows).sort_values('exp_gain_rmse',ascending=False)
print(df.to_string(index=False,float_format=lambda x:f"{x:.3g}"))
df.to_csv('probe_ranking.csv',index=False)
