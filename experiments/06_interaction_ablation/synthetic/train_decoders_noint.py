"""Train the two conditional DeepRV decoders needed by the NO-INTERACTION model
(space g over 20 regions, time q over 72 months). No interaction decoder needed.

Pipeline copied from script 108: draw exact GP fields with lengthscales sampled
from the SAME prior the inference uses (the new containment priors), centre +
normalise to unit variance (amplitude is applied OUTSIDE the decoder at inference),
PCA -> z targets, then fit [z, log_ell] -> field with a 2-layer tanh MLP.

Saves decoders/{space,time}_decoder.npz with the flat params + hidden/output/z dims,
in the same format load_decoder() expects.
Env: N_TRAIN N_TEST STEPS HIDDEN ZS ZT"""
from pathlib import Path
import os, time, json
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT/".matplotlib"))
import sys; sys.path.insert(0, str(ROOT/"scripts"))
import numpy as np, pandas as pd, jax, jax.numpy as jnp
import flax.linen as nn, optax
from flax.training import train_state as flax_ts
from scipy.spatial.distance import cdist
from scipy.stats import invgamma as sp_invgamma
import _common as C

N_TRAIN=int(os.environ.get("N_TRAIN","6000")); N_TEST=int(os.environ.get("N_TEST","1000"))
STEPS=int(os.environ.get("STEPS","5000")); BATCH=128; LR=1e-3
HIDDEN=int(os.environ.get("HIDDEN","512"))
Z_S=int(os.environ.get("ZS","15")); Z_T=int(os.environ.get("ZT","30"))
OUTD=ROOT/"decoders"; OUTD.mkdir(exist_ok=True)

# lengthscale priors == the no-interaction containment priors used at inference
PRIOR={"space":(2.3740,0.2489), "time":(2.6641,15.6605)}

def se(x,ell): return np.exp(-cdist(x,x,"sqeuclidean")/(2.0*ell**2))

def gen(kind, pts, n, seed):
    rng=np.random.default_rng(seed); a,b=PRIOR[kind]
    ells=sp_invgamma.rvs(a=a,scale=b,size=n,random_state=rng)
    D=pts.shape[0]; f=np.empty((n,D),np.float32); le=np.empty((n,1),np.float32)
    for i,e in enumerate(ells):
        K=se(pts,e)+1e-6*np.eye(D); L=np.linalg.cholesky(K)
        v=L@rng.standard_normal(D); v=v-v.mean(); s=v.std()
        f[i]=(v/s if s>1e-8 else v); le[i,0]=np.log(e)
    return f, le

def pca(f, z_dim):
    bias=f.mean(0); fc=f-bias; sc=fc/np.sqrt(len(f)-1)
    _,sv,vt=np.linalg.svd(sc,full_matrices=False)
    comp=vt[:z_dim]; scale=sv[:z_dim]+1e-8
    z=(fc@comp.T)/scale
    ev=(sv**2); frac=ev[:z_dim].sum()/ev.sum()
    return z.astype(np.float32), comp, scale, bias, float(frac)

class Dec(nn.Module):
    hidden_dim:int; output_dim:int
    @nn.compact
    def __call__(self,z,log_ell):
        x=jnp.concatenate([z,log_ell],axis=-1)
        x=nn.tanh(nn.Dense(self.hidden_dim)(x)); x=nn.tanh(nn.Dense(self.hidden_dim)(x))
        return nn.Dense(self.output_dim)(x)

@jax.jit
def step(state,zb,eb,fb):
    def loss_fn(p):
        return jnp.mean((state.apply_fn({"params":p},zb,eb)-fb)**2)
    l,g=jax.value_and_grad(loss_fn)(state.params)
    return state.apply_gradients(grads=g), l

def flatten(params,prefix=""):
    out={}
    for k,v in params.items():
        path=f"{prefix}/{k}" if prefix else str(k)
        if hasattr(v,"items"): out.update(flatten(v,path))
        else: out[path]=np.asarray(v)
    return out

def train(kind, pts, z_dim, seed):
    f,le=gen(kind,pts,N_TRAIN,seed); ft,let=gen(kind,pts,N_TEST,seed+777)
    z,comp,scale,bias,frac=pca(f,z_dim)
    # test z via the same projection
    zt=((ft-bias)@comp.T)/scale
    model=Dec(hidden_dim=HIDDEN,output_dim=pts.shape[0])
    params=model.init(jax.random.PRNGKey(0),jnp.zeros((1,z_dim)),jnp.zeros((1,1)))["params"]
    state=flax_ts.TrainState.create(apply_fn=model.apply,params=params,
        tx=optax.adamw(learning_rate=LR,weight_decay=1e-5))
    rng=np.random.default_rng(42); t0=time.perf_counter()
    for i in range(STEPS):
        idx=rng.integers(0,N_TRAIN,BATCH)
        state,l=step(state,jnp.asarray(z[idx]),jnp.asarray(le[idx]),jnp.asarray(f[idx]))
    el=time.perf_counter()-t0
    fh=np.asarray(state.apply_fn({"params":state.params},jnp.asarray(zt),jnp.asarray(let)))
    fhc=fh-fh.mean(1,keepdims=True); fc=ft-ft.mean(1,keepdims=True)
    rmse=float(np.sqrt(np.mean((fhc-fc)**2))); vr=float(fhc.var()/max(fc.var(),1e-12))
    out=OUTD/f"{kind}_decoder.npz"
    np.savez_compressed(out, **flatten(state.params),
        hidden_dim=HIDDEN, output_dim=pts.shape[0], z_dim=z_dim, n_ell=1,
        pca_frac=frac, prior_a=PRIOR[kind][0], prior_b=PRIOR[kind][1])
    print(f"  {kind:6s} z_dim={z_dim:3d}  PCA var kept={frac:.4f}  prior-RMSE={rmse:.4f}  var_ratio={vr:.4f}  ({el:.0f}s)  -> {out.name}")
    return rmse,vr,frac

if __name__=="__main__":
    _,regions=C.load_regime_regions()
    coords=regions[["x_coord","y_coord"]].to_numpy()-0.5
    tpts=(np.arange(72,dtype=float)-np.arange(72).mean())[:,None]
    print(f"[train] N_TRAIN={N_TRAIN} steps={STEPS} hidden={HIDDEN}")
    train("space",coords,Z_S,1); train("time",tpts,Z_T,2)
