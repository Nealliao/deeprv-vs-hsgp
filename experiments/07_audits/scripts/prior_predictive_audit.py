"""Prior-predictive audit: are the three representations actually placing the same
prior on the latent FUNCTION, or only on the named parameters?

Both HSGP and DeepRV use alpha ~ HalfNormal(1), but alpha multiplies different
objects (a basis expansion vs a variance-normalised decoder output), so identical
parameter priors do not by themselves imply identical function priors. This script
compares the INDUCED prior at fixed hyperparameters:

  (i)  marginal variance  Var{f(x) | ell}
  (ii) correlation vs distance  Corr{f(x),f(x') | ell}  against the exact SE kernel
  (iii) covariance error across a range of lengthscales

Writes figures/prior_predictive_audit.png and results/prior_predictive.json.
"""
from pathlib import Path
import os, json
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import numpy as np, pandas as pd, jax, jax.numpy as jnp
import flax.linen as nn
from flax.core import freeze
from scipy.spatial.distance import cdist, pdist, squareform
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from numpyro.contrib.hsgp.laplacian import eigenfunctions
from numpyro.contrib.hsgp.spectral_densities import diag_spectral_density_squared_exponential as _spd

ECDC = Path("/Users/Zhuanz/Desktop/ECDC_measles_HSGP_vs_DeepRV")
PANEL = ECDC/"data"/"processed"/"ecdc_measles_panel.csv"
DEC   = ECDC/"decoders_final" if (ECDC/"decoders_final").exists() else ECDC/"decoders"
NDRAW = 4000
SPACE_ELL, SPACE_M = [1.4,1.4], [8,8]

class Dec(nn.Module):
    hidden_dim:int; output_dim:int
    @nn.compact
    def __call__(self,z,le):
        x=jnp.concatenate([z,le],-1); x=nn.tanh(nn.Dense(self.hidden_dim)(x)); x=nn.tanh(nn.Dense(self.hidden_dim)(x))
        return nn.Dense(self.output_dim)(x)

def unflat(f):
    root={}
    for k,v in f.items():
        pa=k.split("/"); cur=root
        for q in pa[:-1]: cur=cur.setdefault(q,{})
        cur[pa[-1]]=jnp.asarray(v)
    return freeze(root)

def load(name):
    b=np.load(DEC/f"ecdc_deeprv_{name}.npz")
    return (Dec(hidden_dim=int(b["hidden_dim"]),output_dim=int(b["output_dim"])),
            unflat({k:b[k] for k in b.files if "/" in k}), int(b["z_dim"]))

def se_cov(x,ell): return np.exp(-cdist(x,x,"sqeuclidean")/(2*ell**2))

def draws_exact(coords, ell, n, rng):
    S=len(coords); L=np.linalg.cholesky(se_cov(coords,ell)+1e-8*np.eye(S))
    return (L@rng.standard_normal((S,n))).T

def draws_hsgp(coords, ell, n, rng):
    phi=np.asarray(eigenfunctions(x=jnp.asarray(coords), ell=SPACE_ELL, m=SPACE_M))
    spd=np.sqrt(np.asarray(_spd(alpha=1.0, length=ell, ell=SPACE_ELL, m=SPACE_M, dim=2)))
    beta=rng.standard_normal((n, phi.shape[1]))
    return (phi*spd)@beta.T if False else beta@(phi*spd).T

def draws_deeprv(model, params, zdim, ell, n, rng):
    z=jnp.asarray(rng.standard_normal((n,zdim)),jnp.float32)
    le=jnp.full((n,1), np.log(ell), jnp.float32)
    return np.asarray(model.apply({"params":params}, z, le))

def corr_by_distance(F, D, bins):
    """average Corr(f(x),f(x')) in distance bins, from draws F (n x S)."""
    C=np.corrcoef(F.T)
    iu=np.triu_indices(C.shape[0],1); d=D[iu]; c=C[iu]
    out=[]
    for lo,hi in zip(bins[:-1],bins[1:]):
        m=(d>=lo)&(d<hi)
        out.append(np.nan if m.sum()<3 else c[m].mean())
    return np.array(out)

def main():
    p=pd.read_csv(PANEL)
    coords=p.drop_duplicates("state_index").sort_values("state_index")[["x_coord","y_coord"]].to_numpy()
    D=squareform(pdist(coords))
    model,params,zdim=load("space")
    rng=np.random.default_rng(3)
    ells=[0.25,0.53,1.0,2.0,4.0]
    bins=np.linspace(0,D.max(),9); ctr=0.5*(bins[:-1]+bins[1:])
    res={}
    fig,axes=plt.subplots(1,3,figsize=(17,4.6))

    # (i) marginal variance vs lengthscale
    v_ex,v_hs,v_dr=[],[],[]
    for e in ells:
        v_ex.append(draws_exact(coords,e,NDRAW,rng).var())
        v_hs.append(draws_hsgp(coords,e,NDRAW,rng).var())
        v_dr.append(draws_deeprv(model,params,zdim,e,NDRAW,rng).var())
    ax=axes[0]
    ax.plot(ells,v_ex,"o-",color="#2CA02C",lw=2,label="exact GP")
    ax.plot(ells,v_hs,"s-",color="#4C72B0",lw=2,label="HSGP basis")
    ax.plot(ells,v_dr,"^-",color="#C44E52",lw=2,label="DeepRV decoder")
    ax.set_xscale("log"); ax.set_xlabel(r"$\ell$"); ax.set_ylabel(r"$\mathrm{Var}\{f(x)\mid\ell\}$")
    ax.set_title(r"(a) induced marginal variance at $\alpha=1$",fontsize=10.5,fontweight="bold")
    ax.legend(fontsize=9)

    # (ii) correlation vs distance at the fitted lengthscale
    e=0.53
    ce=corr_by_distance(draws_exact(coords,e,NDRAW,rng),D,bins)
    ch=corr_by_distance(draws_hsgp(coords,e,NDRAW,rng),D,bins)
    cd=corr_by_distance(draws_deeprv(model,params,zdim,e,NDRAW,rng),D,bins)
    ax=axes[1]
    ax.plot(ctr,ce,"o-",color="#2CA02C",lw=2,label="exact GP")
    ax.plot(ctr,ch,"s-",color="#4C72B0",lw=2,label="HSGP basis")
    ax.plot(ctr,cd,"^-",color="#C44E52",lw=2,label="DeepRV decoder")
    ax.axhline(0,color="k",lw=.8,ls=":")
    ax.set_xlabel("distance"); ax.set_ylabel(r"$\mathrm{Corr}\{f(x),f(x')\}$")
    ax.set_title(rf"(b) induced correlation, $\ell={e}$",fontsize=10.5,fontweight="bold"); ax.legend(fontsize=9)

    # (iii) covariance error vs lengthscale, with the normalisation control
    def nrm(F):
        F=F-F.mean(1,keepdims=True); return F/np.maximum(F.std(1,keepdims=True),1e-8)
    eh,ed,edn=[],[],[]
    for x in ells:
        Fe=draws_exact(coords,x,NDRAW,rng)
        Ke=np.corrcoef(Fe.T); Ken=np.corrcoef(nrm(Fe).T)
        Kh=np.corrcoef(draws_hsgp(coords,x,NDRAW,rng).T)
        Kd=np.corrcoef(draws_deeprv(model,params,zdim,x,NDRAW,rng).T)
        iu=np.triu_indices(len(Ke),1)
        eh.append(np.sqrt(np.mean((Kh[iu]-Ke[iu])**2)))
        ed.append(np.sqrt(np.mean((Kd[iu]-Ke[iu])**2)))
        edn.append(np.sqrt(np.mean((Kd[iu]-Ken[iu])**2)))
    ax=axes[2]
    ax.plot(ells,eh,"s-",color="#4C72B0",lw=2,label="HSGP vs exact")
    ax.plot(ells,ed,"^-",color="#C44E52",lw=2,label="DeepRV vs exact")
    ax.plot(ells,edn,"^--",color="#C44E52",lw=1.6,alpha=.65,label="DeepRV vs normalised exact")
    ax.set_xscale("log"); ax.set_xlabel(r"$\ell$"); ax.set_ylabel("RMSE of induced correlation")
    ax.set_title("(c) deviation from the exact prior",fontsize=10.5,fontweight="bold"); ax.legend(fontsize=9)

    fig.suptitle("Prior-predictive audit of the spatial component: do the three representations induce the same function prior?",
                 fontsize=12.5,fontweight="bold")
    fig.tight_layout(rect=[0,0,1,0.9])
    (ROOT/"figures").mkdir(exist_ok=True); (ROOT/"results").mkdir(exist_ok=True)
    fig.savefig(ROOT/"figures"/"prior_predictive_audit.png",dpi=140)
    res=dict(ells=ells, var_exact=v_ex, var_hsgp=v_hs, var_deeprv=v_dr,
             corr_err_hsgp=eh, corr_err_deeprv=ed, corr_err_deeprv_normctrl=edn,
             corr_dist_centres=ctr.tolist(), corr_exact=ce.tolist(), corr_hsgp=ch.tolist(), corr_deeprv=cd.tolist())
    (ROOT/"results"/"prior_predictive.json").write_text(json.dumps(res,indent=2,default=float))
    print("%-6s %9s %9s %9s | %8s %8s %8s"%("ell","Var ex","Var HS","Var DR","errHS","errDR","errDR*"))
    for i,x in enumerate(ells):
        print("%-6.2f %9.3f %9.3f %9.3f | %8.4f %8.4f %8.4f"%(x,v_ex[i],v_hs[i],v_dr[i],eh[i],ed[i],edn[i]))
    print("  (errDR* = DeepRV against the exact prior after matching its unit-variance normalisation)")
    print("saved figures/prior_predictive_audit.png")

if __name__=="__main__": main()
