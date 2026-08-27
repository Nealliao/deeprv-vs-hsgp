"""ECDC held-out CV — HSGP. Fit on 85% train, predict 15% held-out. Same shared
mask + same held-out ELPD/RMSE/coverage as the exact-GP and DeepRV holdout runs.
Model/priors/basis identical to script 111."""
from pathlib import Path
import os, json, time
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT/".matplotlib"))
import numpy as np, pandas as pd, jax, jax.numpy as jnp, numpyro
import numpyro.distributions as dist
from numpyro.contrib.hsgp.laplacian import eigenfunctions
from numpyro.contrib.hsgp.spectral_densities import diag_spectral_density_squared_exponential as _spd
from numpyro.diagnostics import summary as nsummary
from numpyro.infer import MCMC, NUTS, init_to_value
from scipy.special import gammaln, logsumexp

PANEL = ROOT/"data"/"processed"/"ecdc_measles_panel.csv"
if not PANEL.exists(): PANEL = ROOT/"data"/"ecdc_measles_panel.csv"
OUT = Path(__file__).resolve().parent/"results"
MASK = np.load(Path(__file__).resolve().parent/"mask.npz"); TRAIN=MASK["train"]; HOLD=MASK["holdout"]
WARMUP=int(os.environ.get("W","1000")); SAMPLES=int(os.environ.get("S","800")); CHAINS=int(os.environ.get("C","2"))
SPACE_ELL,SPACE_M=[1.4,1.4],[8,8]; TIME_ELL,TIME_M=75.0,35
IS_ELL,IS_M=[1.0,1.0],[8,8]; IT_ELL,IT_M=75.0,15

PRIORS={"space_alpha":dist.HalfNormal(1.0),"space_length":dist.InverseGamma(2.0065, 0.1995),
 "time_alpha":dist.HalfNormal(1.0),"time_length":dist.InverseGamma(6.5707,167.3694),
 "interaction_alpha":dist.HalfNormal(1.0),"interaction_space_length":dist.InverseGamma(5.3661,0.3648),
 "interaction_time_length":dist.InverseGamma(6.2718,27.0192),"sigma_h":dist.HalfNormal(0.45),
 "ell_h":dist.InverseGamma(3.9335,1.9878)}
_HP_INIT={"space_alpha":0.5,"space_length":0.467,"time_alpha":0.5,"time_length":22.0,
 "interaction_alpha":0.3,"interaction_space_length":0.057,"interaction_time_length":3.7,"sigma_h":0.30,"ell_h":0.40}

def _ss(x,eps=1e-20): sx=jnp.where(x>eps,x,jnp.ones_like(x)); return jnp.where(x>eps,jnp.sqrt(sx),jnp.zeros_like(x))
def hsgp_se(name,x,alpha,length,ell,m):
    dim=jnp.shape(x)[-1] if jnp.ndim(x)>1 else 1; phi=eigenfunctions(x=x,ell=ell,m=m)
    spd=_ss(_spd(alpha=alpha**2,length=length,ell=ell,m=m,dim=dim))
    beta=numpyro.sample(f"{name}_bw",dist.Normal(0,1).expand([phi.shape[-1]])); return phi@(spd*beta)
def inter_hsgp(si,ti,cc,tc,alpha,sl,tl):
    ps=eigenfunctions(x=cc,ell=IS_ELL,m=IS_M); ss=_ss(_spd(alpha=1.0,length=sl,ell=IS_ELL,m=IS_M,dim=2))
    pt=eigenfunctions(x=tc,ell=IT_ELL,m=IT_M); st=_ss(_spd(alpha=1.0,length=tl,ell=IT_ELL,m=IT_M,dim=1))
    ws=ps[si]*ss; wt=pt[ti]*st; nb=ws.shape[-1]*wt.shape[-1]
    beta=numpyro.sample("interaction_bw",dist.Normal(0,1).expand([nb])).reshape((ws.shape[-1],wt.shape[-1]))
    return alpha*jnp.einsum("ns,nt,st->n",ws,wt,beta)
def _per(sig,ell): m=jnp.arange(12.); d=jnp.abs(m[:,None]-m[None,:]); return sig**2*jnp.exp(-2*jnp.sin(jnp.pi*d/12)**2/ell**2)

def model(count,log_population,state_index,time_index,month_index,coords_c,time_c,beta0_prior_mean,train_mask):
    beta0=numpyro.sample("beta0",dist.Normal(beta0_prior_mean,3.0))
    isk=numpyro.sample("inv_sqrt_kappa",dist.HalfNormal(1.0)); kappa=numpyro.deterministic("kappa",isk**(-2))
    sh=numpyro.sample("sigma_h",PRIORS["sigma_h"]); eh=numpyro.sample("ell_h",PRIORS["ell_h"])
    Lh=jnp.linalg.cholesky(_per(sh,eh)+1e-6*jnp.eye(12)); zh=numpyro.sample("z_h",dist.Normal(0,1).expand([12]))
    hraw=Lh@zh; hm=hraw-hraw.mean()
    sa=numpyro.sample("space_alpha",PRIORS["space_alpha"]); sl=numpyro.sample("space_length",PRIORS["space_length"])
    ta=numpyro.sample("time_alpha",PRIORS["time_alpha"]); tl=numpyro.sample("time_length",PRIORS["time_length"])
    ia=numpyro.sample("interaction_alpha",PRIORS["interaction_alpha"])
    isl=numpyro.sample("interaction_space_length",PRIORS["interaction_space_length"])
    itl=numpyro.sample("interaction_time_length",PRIORS["interaction_time_length"])
    g=hsgp_se("space",coords_c,sa,sl,SPACE_ELL,SPACE_M); q=hsgp_se("time",time_c,ta,tl,TIME_ELL,TIME_M)
    w=inter_hsgp(state_index,time_index,coords_c,time_c,ia,isl,itl)
    g=g-g.mean(); q=q-q.mean(); w=w-w.mean()
    eta=jnp.clip(log_population+beta0+hm[month_index]+g[state_index]+q[time_index]+w,-30.,20.)
    rate=jnp.exp(eta); numpyro.deterministic("rate",rate)
    with numpyro.handlers.mask(mask=train_mask):
        numpyro.sample("obs",dist.NegativeBinomial2(mean=rate,concentration=kappa),obs=count)

def run():
    OUT.mkdir(exist_ok=True); numpyro.set_host_device_count(CHAINS)
    p=pd.read_csv(PANEL); obs=p[p["observed"]==1].copy()
    coords=p.drop_duplicates("state_index").sort_values("state_index")[["x_coord","y_coord"]].to_numpy()
    T=p["time_index"].nunique(); tc=(np.arange(T,dtype=float)-np.arange(T).mean())[:,None]
    y=obs["cases"].to_numpy()
    data=dict(count=jnp.asarray(y.astype(np.int32)),
        log_population=jnp.asarray(obs["log_population"].to_numpy(np.float32)),
        state_index=jnp.asarray(obs["state_index"].to_numpy(np.int32)),
        time_index=jnp.asarray(obs["time_index"].to_numpy(np.int32)),
        month_index=jnp.asarray((obs["month_of_year"].to_numpy()-1).astype(np.int32)),
        coords_c=jnp.asarray(coords.astype(np.float32)),time_c=jnp.asarray(tc.astype(np.float32)),
        beta0_prior_mean=float(np.log(obs["cases"].mean()+1)-obs["log_population"].mean()),
        train_mask=jnp.asarray(TRAIN))
    print(f"[HSGP holdout] train={TRAIN.sum()} holdout={HOLD.sum()}")
    mcmc=MCMC(NUTS(model,target_accept_prob=0.95,max_tree_depth=12,init_strategy=init_to_value(values=_HP_INIT)),
        num_warmup=WARMUP,num_samples=SAMPLES,num_chains=CHAINS,chain_method="sequential",progress_bar=True)
    t0=time.perf_counter(); mcmc.run(jax.random.PRNGKey(11),**data,extra_fields=("diverging",))
    jax.block_until_ready(mcmc.get_samples()); el=time.perf_counter()-t0
    sbc=mcmc.get_samples(group_by_chain=True)
    flat={k:np.asarray(v).reshape((-1,)+np.asarray(v).shape[2:]) for k,v in sbc.items()}
    rate=flat["rate"]; kap=flat["kappa"]
    rh=rate[:,HOLD]; yh=y[HOLD]; kp=kap[:,None]
    ll=(gammaln(yh[None,:]+kp)-gammaln(kp)-gammaln(yh[None,:]+1)+kp*np.log(kp/(kp+rh))+yh[None,:]*np.log(rh/(kp+rh)))
    elpd_i=logsumexp(ll,0)-np.log(ll.shape[0]); elpd=float(elpd_i.sum()); elpd_se=float(np.sqrt(len(elpd_i)*elpd_i.var()))
    rng=np.random.default_rng(0); pp=rng.negative_binomial(kp,kp/(kp+rh))
    predmean=pp.mean(0); lo=np.quantile(pp,.025,0); hi=np.quantile(pp,.975,0)
    rmse=float(np.sqrt(np.mean((predmean-yh)**2))); cov=float(((yh>=lo)&(yh<=hi)).mean())
    ndiv=int(np.asarray(mcmc.get_extra_fields()["diverging"]).sum())
    diag=nsummary({k:v for k,v in sbc.items() if "_bw" not in k and k not in("rate","kappa")})
    mrh=float(np.nanmax([np.nanmax(diag[k]["r_hat"]) for k in diag]))
    row=dict(method="hsgp",n_train=int(TRAIN.sum()),n_hold=int(HOLD.sum()),
        holdout_elpd=elpd,holdout_elpd_se=elpd_se,holdout_rmse=rmse,holdout_cov95=cov,
        max_rhat=mrh,n_div=ndiv,runtime=el,
        holdout_pred_mean=predmean.tolist(),holdout_y=yh.tolist(),holdout_elpd_i=elpd_i.tolist())
    (OUT/"holdout_hsgp_v4.json").write_text(json.dumps(row))
    print(f"  HELDOUT elpd={elpd:.1f}+/-{elpd_se:.1f}  rmse={rmse:.2f}  cov95={cov:.3f}  | rhat={mrh:.3f} div={ndiv} t={el:.0f}s")

if __name__=="__main__": run()
