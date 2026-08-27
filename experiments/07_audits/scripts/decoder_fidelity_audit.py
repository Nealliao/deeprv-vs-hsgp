"""Audit of the ECDC DeepRV decoders: reconstruction fidelity as a function of the
conditioning lengthscale, and whether the posterior explores lengthscales the
decoder was actually trained on.

For each decoder we (i) recover the training lengthscale distribution (the same
inverse-gamma prior used at inference, sampled with the training seed), (ii) draw
fresh exact-GP test fields on a grid of lengthscales, (iii) push them through the
PCA->decoder pipeline and measure variance ratio and RMSE per lengthscale bin, and
(iv) overlay the fitted posterior for that lengthscale.

Writes results/decoder_fidelity.json and figures/decoder_fidelity.png.
"""
from pathlib import Path
import os, json
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import numpy as np, jax, jax.numpy as jnp
import flax.linen as nn
from flax.core import freeze
from scipy.spatial.distance import cdist
from scipy.stats import invgamma as sp_ig
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import pandas as pd

ECDC = Path("/Users/Zhuanz/Desktop/ECDC_measles_HSGP_vs_DeepRV")
DEC  = ECDC/"decoders"
PANEL= ECDC/"data"/"processed"/"ecdc_measles_panel.csv"
if not PANEL.exists(): PANEL = ECDC/"data"/"ecdc_measles_panel.csv"
POST = ECDC/"data"/"processed"/"ecdc"/"ecdc_deeprv_enhanced.npz"

# training-time priors (script 113) == inference priors
PRIOR = {"space": (6.1091,3.3175), "time": (6.5707,167.3694),
         "inter_s": (5.3661,0.3648), "inter_t": (6.2718,27.0192)}
N_TRAIN = 6000   # draws per component, script 113 default

class Dec(nn.Module):
    hidden_dim:int; output_dim:int
    @nn.compact
    def __call__(self,z,log_ell):
        x=jnp.concatenate([z,log_ell],axis=-1)
        x=nn.tanh(nn.Dense(self.hidden_dim)(x)); x=nn.tanh(nn.Dense(self.hidden_dim)(x))
        return nn.Dense(self.output_dim)(x)

def unflat(f):
    root={}
    for k,v in f.items():
        parts=k.split("/"); cur=root
        for p in parts[:-1]: cur=cur.setdefault(p,{})
        cur[parts[-1]]=jnp.asarray(v)
    return freeze(root)

def load(p):
    b=np.load(p)
    params=unflat({k:b[k] for k in b.files if "/" in k})
    m=Dec(hidden_dim=int(b["hidden_dim"]),output_dim=int(b["output_dim"]))
    return m,params,int(b["z_dim"]),b["pca_components"],b["pca_score_scale"],b["pca_bias"]

def se(x,ell): return np.exp(-cdist(x,x,"sqeuclidean")/(2.0*ell**2))
def dc(m): return m-m.mean(0,keepdims=True)-m.mean(1,keepdims=True)+m.mean()

def field_1d(pts,ell,rng,n):
    D=pts.shape[0]; L=np.linalg.cholesky(se(pts,ell)+1e-6*np.eye(D))
    out=np.empty((n,D),np.float32)
    for i in range(n):
        v=L@rng.standard_normal(D); v=v-v.mean(); s=v.std()
        out[i]=v/s if s>1e-8 else v
    return out

def field_2d(coords,tpts,es,et,rng,n):
    S=coords.shape[0]; T=tpts.shape[0]
    Ls=np.linalg.cholesky(se(coords,es)+1e-6*np.eye(S))
    Lt=np.linalg.cholesky(se(tpts,et)+1e-6*np.eye(T))
    out=np.empty((n,S*T),np.float32)
    for i in range(n):
        w=dc(Ls@rng.standard_normal((S,T))@Lt.T); s=w.std()
        out[i]=(w/s if s>1e-8 else w).ravel()
    return out

def fidelity(model,params,comp,ss,bias,f,le):
    z=((f-bias)@comp.T)/ss
    fh=np.asarray(model.apply({"params":params},jnp.asarray(z,jnp.float32),jnp.asarray(le,jnp.float32)))
    a=fh-fh.mean(1,keepdims=True); b=f-f.mean(1,keepdims=True)
    return float(a.var()/max(b.var(),1e-12)), float(np.sqrt(np.mean((a-b)**2)))

def main():
    p=pd.read_csv(PANEL)
    coords=p.drop_duplicates("state_index").sort_values("state_index")[["x_coord","y_coord"]].to_numpy()
    T=p["time_index"].nunique(); tpts=(np.arange(T,dtype=float)-np.arange(T).mean())[:,None]
    post=np.load(POST,allow_pickle=True)
    def psamp(k):
        for c in (f"post_{k}",k,f"trace_{k}"):
            if c in post.files: return np.asarray(post[c]).ravel()
        return None
    out={}
    fig,axes=plt.subplots(1,3,figsize=(17,4.6))

    specs=[("space","ecdc_deeprv_space.npz","space_length",PRIOR["space"],r"$\ell_g$ (spatial)"),
           ("time","ecdc_deeprv_time.npz","time_length",PRIOR["time"],r"$\ell_q$ (trend, months)"),
           ("inter","ecdc_deeprv_inter.npz","interaction_space_length",PRIOR["inter_s"],r"$\ell_{ws}$ (interaction, space)")]

    for ax,(name,fn,pkey,pr,xlabel) in zip(axes,specs):
        model,params,zd,comp,ss,bias=load(DEC/fn)
        rng_tr=np.random.default_rng(0)
        ell_train=sp_ig.rvs(a=pr[0],scale=pr[1],size=N_TRAIN,random_state=rng_tr)
        lo_tr,hi_tr=ell_train.min(),ell_train.max()
        q=np.quantile(ell_train,[0.001,0.999])
        grid=np.geomspace(max(q[0]*0.6,1e-3),q[1]*1.6,14)
        vr=[];rm=[]
        rng=np.random.default_rng(7)
        for e in grid:
            if name=="inter":
                et=float(sp_ig.ppf(0.5,PRIOR["inter_t"][0],scale=PRIOR["inter_t"][1]))
                f=field_2d(coords,tpts,e,et,rng,40)
                le=np.column_stack([np.full(40,np.log(e)),np.full(40,np.log(et))]).astype(np.float32)
            else:
                pts=coords if name=="space" else tpts
                f=field_1d(pts,e,rng,40); le=np.full((40,1),np.log(e),np.float32)
            v,r=fidelity(model,params,comp,ss,bias,f,le); vr.append(v); rm.append(r)
        vr=np.array(vr)
        ps=psamp(pkey)
        inside=float(((ps>=lo_tr)&(ps<=hi_tr)).mean()) if ps is not None else float("nan")
        out[name]=dict(z_dim=zd,train_ell_min=float(lo_tr),train_ell_max=float(hi_tr),
            post_ell_mean=float(ps.mean()) if ps is not None else None,
            post_ell_q025=float(np.quantile(ps,.025)) if ps is not None else None,
            post_ell_q975=float(np.quantile(ps,.975)) if ps is not None else None,
            frac_posterior_in_training_support=inside,
            grid=grid.tolist(), var_ratio=vr.tolist(), rmse=rm)
        ax.axvspan(lo_tr,hi_tr,color="#dfeaf7",label="decoder training support")
        ax.plot(grid,vr,"o-",color="#C44E52",lw=2,ms=5,label=r"reconstruction fidelity $\rho^2$")
        if ps is not None:
            ax2=ax.twinx(); ax2.hist(ps,bins=45,color="#777",alpha=.4,density=True)
            ax2.set_yticks([]); ax2.set_ylabel("posterior density",fontsize=8,color="#777")
        ax.axhline(1.0,color="k",ls=":",lw=1)
        ax.set_xscale("log"); ax.set_xlabel(xlabel); ax.set_ylim(0,1.15)
        ax.set_ylabel(r"$\rho^2$"); ax.set_title(f"{name}  (z={zd}, {100*inside:.1f}% of posterior in support)",
                                                 fontsize=10.5,fontweight="bold")
        ax.legend(fontsize=8,loc="lower right")
    fig.suptitle("DeepRV decoder audit: reconstruction fidelity against conditioning lengthscale, "
                 "with the fitted posterior overlaid (grey)",fontsize=12.5,fontweight="bold")
    fig.tight_layout(rect=[0,0,1,0.9])
    (ROOT/"figures").mkdir(exist_ok=True); (ROOT/"results").mkdir(exist_ok=True)
    fig.savefig(ROOT/"figures"/"decoder_fidelity.png",dpi=140)
    (ROOT/"results"/"decoder_fidelity.json").write_text(json.dumps(out,indent=2))
    for k,v in out.items():
        print(f"{k:6s} z={v['z_dim']:4d}  train ell [{v['train_ell_min']:.4f},{v['train_ell_max']:.3f}]  "
              f"post {v['post_ell_mean']:.4f} [{v['post_ell_q025']:.4f},{v['post_ell_q975']:.4f}]  "
              f"in-support {100*v['frac_posterior_in_training_support']:.2f}%")
    print("saved figures/decoder_fidelity.png + results/decoder_fidelity.json")

if __name__=="__main__": main()
