"""Retrain the ECDC DeepRV decoders over a WIDENED lengthscale range.

Motivation: the audit found that the fitted posterior for the spatial lengthscale
(mean 3.72, 95% [2.11,6.20]) runs past the largest lengthscale in the original
training sample (4.60), so the decoder was extrapolating over ~19% of posterior
mass. A conditional surrogate must be accurate wherever the sampler goes, and
that region is set by the POSTERIOR, not by the prior it was trained from.

Fix: sample the conditioning lengthscales log-uniformly over a bracket that
contains both the prior's 99.9% interval and the observed posterior with margin,
rather than from the inference prior itself. The inference prior is unchanged;
only the decoder's training support is widened. Everything else (architecture,
optimiser, steps, PCA latent sizing, unit-variance normalisation) is identical to
script 113.

Writes to decoders_v2/ so the original decoders are preserved.
"""
from pathlib import Path
import os, time, json
PROJECT_ROOT = Path("/Users/Zhuanz/Desktop/ECDC_measles_HSGP_vs_DeepRV")
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT/".matplotlib"))
import numpy as np, pandas as pd, jax, jax.numpy as jnp
import flax.linen as nn, optax
from flax.training import train_state as flax_ts
from scipy.spatial.distance import cdist

DATA = PROJECT_ROOT/"data"/"processed"
PANEL = DATA/"ecdc_measles_panel.csv"
OUTD = PROJECT_ROOT/"decoders_v2b"; OUTD.mkdir(exist_ok=True)

N_TRAIN=6000; N_TEST=1000; STEPS=6000; BATCH=128; LR=1e-3; HIDDEN=512
Z_S=15; Z_T=12; Z_W=360

# widened training brackets (log-uniform). Each contains the prior 99.9% interval
# and the fitted posterior with margin.
RANGE = {"space":(0.08,12.0), "time":(4.0,200.0),
         "inter_s":(0.015,1.0), "inter_t":(0.8,30.0)}

def se(x,ell): return np.exp(-cdist(x,x,"sqeuclidean")/(2.0*ell**2))
def dc(m): return m-m.mean(0,keepdims=True)-m.mean(1,keepdims=True)+m.mean()
def loguni(lo,hi,n,rng): return np.exp(rng.uniform(np.log(lo),np.log(hi),n))

def gen_1d(pts,lo,hi,n,seed):
    rng=np.random.default_rng(seed); ells=loguni(lo,hi,n,rng); D=pts.shape[0]
    f=np.empty((n,D),np.float32); le=np.empty((n,1),np.float32)
    for i,e in enumerate(ells):
        L=np.linalg.cholesky(se(pts,e)+1e-6*np.eye(D))
        v=L@rng.standard_normal(D); v=v-v.mean(); s=v.std()
        f[i]=v/s if s>1e-8 else v; le[i,0]=np.log(e)
    return f,le

def gen_inter(coords,tpts,n,seed):
    rng=np.random.default_rng(seed)
    es=loguni(*RANGE["inter_s"],n,rng); et=loguni(*RANGE["inter_t"],n,rng)
    S=coords.shape[0]; T=tpts.shape[0]
    f=np.empty((n,S*T),np.float32); le=np.empty((n,2),np.float32)
    for i,(a,b) in enumerate(zip(es,et)):
        Ls=np.linalg.cholesky(se(coords,a)+1e-6*np.eye(S))
        Lt=np.linalg.cholesky(se(tpts,b)+1e-6*np.eye(T))
        w=dc(Ls@rng.standard_normal((S,T))@Lt.T); s=w.std()
        f[i]=(w/s if s>1e-8 else w).ravel(); le[i]=[np.log(a),np.log(b)]
    return f,le

def pca(f,z):
    bias=f.mean(0); fc=f-bias; sc=fc/np.sqrt(len(f)-1)
    _,sv,vt=np.linalg.svd(sc,full_matrices=False)
    comp=vt[:z]; ss=sv[:z]+1e-8
    return ((fc@comp.T)/ss).astype(np.float32),comp,ss,bias,float((sv**2)[:z].sum()/(sv**2).sum())

class Dec(nn.Module):
    hidden_dim:int; output_dim:int
    @nn.compact
    def __call__(self,z,log_ell):
        x=jnp.concatenate([z,log_ell],axis=-1)
        x=nn.tanh(nn.Dense(self.hidden_dim)(x)); x=nn.tanh(nn.Dense(self.hidden_dim)(x))
        return nn.Dense(self.output_dim)(x)

@jax.jit
def step(state,zb,eb,fb):
    def loss_fn(p): return jnp.mean((state.apply_fn({"params":p},zb,eb)-fb)**2)
    l,g=jax.value_and_grad(loss_fn)(state.params); return state.apply_gradients(grads=g),l

def flat(params,pre=""):
    o={}
    for k,v in params.items():
        pth=f"{pre}/{k}" if pre else str(k)
        if hasattr(v,"items"): o.update(flat(v,pth))
        else: o[pth]=np.asarray(v)
    return o

def train(name,z_dim,f_tr,le_tr,f_te,le_te):
    out_dim=f_tr.shape[1]; n_ell=le_tr.shape[1]
    z_tr,comp,ss,bias,frac=pca(f_tr,z_dim)
    m=Dec(hidden_dim=HIDDEN,output_dim=out_dim)
    params=m.init(jax.random.PRNGKey(0),jnp.zeros((1,z_dim)),jnp.zeros((1,n_ell)))["params"]
    st=flax_ts.TrainState.create(apply_fn=m.apply,params=params,
        tx=optax.adamw(learning_rate=LR,weight_decay=1e-5))
    rng=np.random.default_rng(42+z_dim); t0=time.perf_counter()
    for i in range(STEPS):
        idx=rng.choice(len(f_tr),BATCH,replace=False)
        st,l=step(st,jnp.asarray(z_tr[idx]),jnp.asarray(le_tr[idx]),jnp.asarray(f_tr[idx]))
    el=time.perf_counter()-t0
    z_te=((f_te-bias)@comp.T)/ss
    fh=np.asarray(st.apply_fn({"params":st.params},jnp.asarray(z_te,jnp.float32),jnp.asarray(le_te)))
    a=fh-fh.mean(1,keepdims=True); b=f_te-f_te.mean(1,keepdims=True)
    rmse=float(np.sqrt(np.mean((a-b)**2))); vr=float(a.var()/max(b.var(),1e-12))
    np.savez_compressed(OUTD/f"ecdc_deeprv_{name}.npz",**flat(st.params),
        hidden_dim=HIDDEN,output_dim=out_dim,z_dim=z_dim,n_ell=n_ell,
        pca_components=comp,pca_score_scale=ss,pca_bias=bias,
        prior_rmse=rmse,var_ratio=vr,pca_frac=frac,
        train_ell_min=float(np.exp(le_tr[:,0].min())),train_ell_max=float(np.exp(le_tr[:,0].max())))
    print(f"  [{name}] {el:.1f}s  PCA={frac:.4f}  prior_rmse={rmse:.4f}  var_ratio={vr:.4f}  "
          f"ell in [{np.exp(le_tr[:,0].min()):.4f},{np.exp(le_tr[:,0].max()):.2f}]")
    return dict(name=name,z_dim=z_dim,seconds=el,pca_frac=frac,prior_rmse=rmse,var_ratio=vr)

if __name__=="__main__":
    p=pd.read_csv(PANEL)
    coords=p.drop_duplicates("state_index").sort_values("state_index")[["x_coord","y_coord"]].to_numpy()
    T=p["time_index"].nunique(); tpts=(np.arange(T,dtype=float)-np.arange(T).mean())[:,None]
    print(f"[widened retrain] S={len(coords)} T={T}  ranges={RANGE}")
    res=[]
    fs,ls=gen_1d(coords,*RANGE["space"],N_TRAIN,1001); fsT,lsT=gen_1d(coords,*RANGE["space"],N_TEST,1002)
    res.append(train("space",Z_S,fs,ls,fsT,lsT))
    ft,lt=gen_1d(tpts,*RANGE["time"],N_TRAIN,2001);   ftT,ltT=gen_1d(tpts,*RANGE["time"],N_TEST,2002)
    res.append(train("time",Z_T,ft,lt,ftT,ltT))
    fw,lw=gen_inter(coords,tpts,N_TRAIN,3001);        fwT,lwT=gen_inter(coords,tpts,N_TEST,3002)
    res.append(train("inter",Z_W,fw,lw,fwT,lwT))
    (OUTD/"training_summary.json").write_text(json.dumps({"ranges":RANGE,"results":res},indent=2))
    print(f"[saved] {OUTD}")
