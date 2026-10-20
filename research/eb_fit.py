import numpy as np, pandas as pd
from scipy.optimize import minimize
from bayes_lb import *
lb,g,w,names,V,c = load()
sub = pd.read_csv(REPO/'submission.csv'); yR21 = sub.coverage_gap_score.to_numpy()[g]
Nv = noise_cov(V,w,yR21,0.0624**2)
keep,Q,R = gram_schmidt(V,w,names)
m = Model(Q,R,keep,names,c,Nv); m.setup(w)
G = m.groups; z = m.o/np.sqrt(np.diag(m.N))
for gr in ['base','probe','clip']:
    print(gr, 'n',(G==gr).sum(),'mean z^2 %.2f'%np.mean(z[G==gr]**2))

def unpack(x, cfg):
    p = dict(tau_const=10.0, mu_const=0.0, alpha_N=1.0, tau_clip=1.0, mu_clip=0.0)
    for k,v in zip(cfg, x):
        p[k] = np.exp(v) if k.startswith(('tau','alpha')) else v
    return p
def fit(cfg, x0):
    f = lambda x: m.nll(unpack(x,cfg))
    r = minimize(f, x0, method='Nelder-Mead', options=dict(maxiter=4000, xatol=1e-4, fatol=1e-6))
    return unpack(r.x,cfg), r.fun
cfgs = {
 'A: zero-mean, shared tau':           (['tau_base','tau_probe'], [0,0]),
 'B: group means':                     (['tau_base','mu_base','tau_probe','mu_probe'], [0,0.5,0,0]),
 'C: group means + noise scale':       (['tau_base','mu_base','tau_probe','mu_probe','alpha_N'], [0,0.5,0,0,0]),
 'D: C + clip tau':                    (['tau_base','mu_base','tau_probe','mu_probe','alpha_N','tau_clip'], [0,0.5,0,0,0,0]),
}
res = {}
for k,(cfg,x0) in cfgs.items():
    p,nll = fit(cfg,x0); res[k]=p
    print(f"{k:34s} nll={nll:9.3f} ", {kk:round(v,3) for kk,v in p.items() if kk in cfg})
import pickle; pickle.dump(res, open('eb_params.pkl','wb'))
