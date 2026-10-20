import numpy as np, pandas as pd, pickle
from scipy.optimize import minimize_scalar
from bayes_lb import *
lb,g,w,names,V,c = load()
sub = pd.read_csv(REPO/'submission.csv'); yR21 = sub.coverage_gap_score.to_numpy()[g]
Nv = noise_cov(V,w,yR21,0.0624**2)
keep,Q,R = gram_schmidt(V,w,names)
X = features(lb.geoid[g].to_numpy())
print('features', X.shape)
Xs = X.to_numpy()
rng = np.random.default_rng(0)
# PCA for RBF
U,sv,Vt = np.linalg.svd(Xs - Xs.mean(0), full_matrices=False)
Z = U[:,:30]*sv[:30]; Z /= Z.std(0)
def rff(Z, ell, m=1500):
    W = rng.standard_normal((Z.shape[1], m))/ell; b = rng.uniform(0,2*np.pi,m)
    return np.sqrt(2)*np.cos(Z@W+b)
regions = lb.geoid[g].map(dict(zip(lb.geoid, np.load(REPO/'model.npz')['region']))).to_numpy()
kernels = {
  'linear(all feats)': Xs/np.sqrt(1.0),
  'rbf ell=2': rff(Z,2.0), 'rbf ell=4': rff(Z,4.0), 'rbf ell=8': rff(Z,8.0),
}
res = pickle.load(open('eb_params.pkl','rb')); pC = dict(res['C: group means + noise scale'])
m0 = Model(Q,R,keep,names,c,Nv); m0.setup(w)
G = m0.groups; probe_idx = np.where(np.isin(G,['probe','clip']))[0]
def chrono_scores(m, p):
    """predict each probe's new-signal from all earlier files; return errors in score units and log-lik"""
    errs, ll = [], 0
    for k in probe_idx:
        mean,var = m.predict_obs(p, np.arange(k), np.array([k]))
        # score units: c_f changes by R_kk * delta o_k
        f = m.names[k]; s = lb.s[f]; rkk = np.linalg.inv(m.Rinv_T.T)[k,k] if False else None
        err = (m.o[k]-mean[0]); ll += -0.5*err**2/var[0] - 0.5*np.log(2*np.pi*var[0])
        errs.append(err)
    return np.array(errs), ll
Rmat = np.linalg.inv(m0.Rinv_T).T  # V_kept = Q R  -> c_kept = R^T o
def to_score(errs):
    out=[]
    for k,e in zip(probe_idx, errs):
        f=m0.names[k]; out.append(abs(Rmat[k,k]*e)/lb.s[f])
    return np.array(out)
e0, ll0 = chrono_scores(m0, pC)
print(f"{'no features (EB-C)':22s} nll={m0.nll(pC):9.3f} chrono loglik={ll0:8.2f} score-MAE={to_score(e0).mean():.2e}")
for kname, Phi in kernels.items():
    m = Model(Q,R,keep,names,c,Nv,Phi=Phi); m.setup(w)
    def f(lt):
        p=dict(pC); p['tau_phi']=np.exp(lt); return m.nll(p)
    r = minimize_scalar(f, bounds=(-12,4), method='bounded')
    p=dict(pC); p['tau_phi']=np.exp(r.x)
    e,ll = chrono_scores(m,p)
    print(f"{kname:22s} nll={r.fun:9.3f} tau_phi={p['tau_phi']:.3e} chrono loglik={ll:8.2f} score-MAE={to_score(e).mean():.2e}")
    # also scan tau to see LOO behaviour
    for lt in [r.x-2, r.x+2]:
        pp=dict(pC); pp['tau_phi']=np.exp(lt); e2,ll2=chrono_scores(m,pp)
        print(f"      tau_phi={np.exp(lt):.2e}: chrono loglik={ll2:8.2f} score-MAE={to_score(e2).mean():.2e}")
