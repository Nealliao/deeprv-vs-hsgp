"""ECDC NO-INTERACTION ablation: f = g_s + q_t + season (interaction REMOVED),
fitted with exact GP / HSGP / DeepRV. Mirrors the synthetic no-interaction study
but on real ECDC data (NB2 likelihood, data-driven periodic season).

Priors: with no interaction there is nothing to separate from, so Betancourt's
S4.1 non-overlap constraint is void and the lengthscale priors are re-derived
from pure S3 containment over the full informative range.
  ECDC geometry: S=29, min-NN=0.030, span=1.331, T=120 months
    space_length ~ InvGamma(2.0065,0.1995)  98% [0.030,1.33]   (old: [0.25,1.80])
    time_length  ~ InvGamma(2.3850,14.6784) 98% [2,60]          (old: [12,80])

No ground truth here, so evaluation is posterior-predictive (coverage, RMSE) and
pointwise loglik for WAIC/LOO. Saves everything needed for plotting.

Env: METHOD=exactgp|hsgp|deeprv  PRIORSET=noint|legacy  W S C  SEED"""
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
from scipy.special import gammaln

PANEL = ROOT/"data"/"ecdc_measles_panel.csv"
OUT = ROOT/"results"; OUT.mkdir(exist_ok=True)
METHOD=os.environ.get("METHOD","exactgp"); PRIORSET=os.environ.get("PRIORSET","noint")
WARMUP=int(os.environ.get("W","1000")); SAMPLES=int(os.environ.get("S","800"))
CHAINS=int(os.environ.get("C","4")); TARGET=float(os.environ.get("TARGET","0.9")); MTD=12
SEED=int(os.environ.get("SEED","20260722")); JITTER=1e-4
SPACE_ELL,SPACE_M=[1.4,1.4],[8,8]; TIME_ELL,TIME_M=75.0,35

if PRIORSET=="legacy":
    _LEN={"space_length":dist.InverseGamma(6.1091,3.3175),"time_length":dist.InverseGamma(6.5707,167.3694)}
else:
    _LEN={"space_length":dist.InverseGamma(2.0065,0.1995),"time_length":dist.InverseGamma(2.3850,14.6784)}
PRIORS={"space_alpha":dist.HalfNormal(1.0),"space_length":_LEN["space_length"],
    "time_alpha":dist.HalfNormal(1.0),"time_length":_LEN["time_length"],
    "sigma_h":dist.HalfNormal(0.45),"ell_h":dist.InverseGamma(3.9335,1.9878)}
HP6=list(PRIORS.keys())
_HP_INIT={"space_alpha":0.5,"space_length":0.30,"time_alpha":0.5,"time_length":20.0,
    "sigma_h":0.30,"ell_h":0.40}

if METHOD=="deeprv":
    import flax.linen as _nn
    from flax.core import freeze as _freeze
    class _Dec(_nn.Module):
        hidden_dim:int; output_dim:int
        @_nn.compact
        def __call__(self,z,log_ell):
            x=jnp.concatenate([z,log_ell],axis=-1)
            x=_nn.tanh(_nn.Dense(self.hidden_dim)(x)); x=_nn.tanh(_nn.Dense(self.hidden_dim)(x))
            return _nn.Dense(self.output_dim)(x)
    def _unflat(f):
        root={}
        for k,v in f.items():
            parts=k.split("/"); cur=root
            for p in parts[:-1]: cur=cur.setdefault(p,{})
            cur[parts[-1]]=jnp.asarray(v)
        return _freeze(root)
    def _load(p):
        b=np.load(p); pr=_unflat({k:b[k] for k in b.files if "/" in k})
        return _Dec(hidden_dim=int(b["hidden_dim"]),output_dim=int(b["output_dim"])),pr,int(b["z_dim"])
    _SDEC,_SP,_ZS=_load(ROOT/"decoders"/"space_decoder.npz")
    _TDEC,_TP,_ZT=_load(ROOT/"decoders"/"time_decoder.npz")

def _se(x,ell): d2=jnp.sum((x[:,None,:]-x[None,:,:])**2,-1); return jnp.exp(-0.5*d2/ell**2)
def _chol(x,ell): K=_se(x,ell); return jnp.linalg.cholesky(K+JITTER*jnp.eye(K.shape[0]))
def _ss(x,eps=1e-20): sx=jnp.where(x>eps,x,jnp.ones_like(x)); return jnp.where(x>eps,jnp.sqrt(sx),jnp.zeros_like(x))
def hsgp_se(name,x,alpha,length,ell,m):
    dim=jnp.shape(x)[-1] if jnp.ndim(x)>1 else 1; phi=eigenfunctions(x=x,ell=ell,m=m)
    spd=_ss(_spd(alpha=alpha**2,length=length,ell=ell,m=m,dim=dim))
    beta=numpyro.sample(f"{name}_bw",dist.Normal(0,1).expand([phi.shape[-1]])); return phi@(spd*beta)
def _per(sig,ell): m=jnp.arange(12.); d=jnp.abs(m[:,None]-m[None,:]); return sig**2*jnp.exp(-2*jnp.sin(jnp.pi*d/12)**2/ell**2)

def model(count,log_population,state_index,time_index,month_index,coords_c,time_c,beta0_prior_mean):
    S=coords_c.shape[0]; T=time_c.shape[0]
    beta0=numpyro.sample("beta0",dist.Normal(beta0_prior_mean,3.0))
    isk=numpyro.sample("inv_sqrt_kappa",dist.HalfNormal(1.0)); kappa=numpyro.deterministic("kappa",isk**(-2))
    sh=numpyro.sample("sigma_h",PRIORS["sigma_h"]); eh=numpyro.sample("ell_h",PRIORS["ell_h"])
    Lh=jnp.linalg.cholesky(_per(sh,eh)+1e-6*jnp.eye(12))
    zh=numpyro.sample("z_h",dist.Normal(0,1).expand([12])); hraw=Lh@zh; hm=hraw-hraw.mean()
    sa=numpyro.sample("space_alpha",PRIORS["space_alpha"]); sl=numpyro.sample("space_length",PRIORS["space_length"])
    ta=numpyro.sample("time_alpha",PRIORS["time_alpha"]); tl=numpyro.sample("time_length",PRIORS["time_length"])
    if METHOD=="exactgp":
        zg=numpyro.sample("z_g",dist.Normal(0,1).expand([S])); g=sa*(_chol(coords_c,sl)@zg)
        zq=numpyro.sample("z_q",dist.Normal(0,1).expand([T])); q=ta*(_chol(time_c,tl)@zq)
    elif METHOD=="deeprv":
        zs=numpyro.sample("z_space",dist.Normal(0,1).expand([_ZS]))
        zt=numpyro.sample("z_time", dist.Normal(0,1).expand([_ZT]))
        g=sa*_SDEC.apply({"params":_SP},zs[None,:],jnp.array([[jnp.log(sl)]]))[0]
        q=ta*_TDEC.apply({"params":_TP},zt[None,:],jnp.array([[jnp.log(tl)]]))[0]
    else:
        g=hsgp_se("space",coords_c,sa,sl,SPACE_ELL,SPACE_M); q=hsgp_se("time",time_c,ta,tl,TIME_ELL,TIME_M)
    g=g-g.mean(); q=q-q.mean()
    latent=g[state_index]+q[time_index]                       # NO interaction
    eta=jnp.clip(log_population+beta0+hm[month_index]+latent,-30.,20.)
    numpyro.deterministic("latent_f",latent); numpyro.deterministic("eta",eta)
    numpyro.deterministic("g_contrib",g[state_index]); numpyro.deterministic("q_contrib",q[time_index])
    numpyro.deterministic("h_monthly",hm)
    numpyro.sample("obs",dist.NegativeBinomial2(mean=jnp.exp(eta),concentration=kappa),obs=count)

def run():
    numpyro.set_host_device_count(CHAINS)
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
        beta0_prior_mean=float(np.log(obs["cases"].mean()+1)-obs["log_population"].mean()))
    print(f"[ECDC no-interaction] method={METHOD} priors={PRIORSET}  n={len(y)}")
    mcmc=MCMC(NUTS(model,target_accept_prob=TARGET,max_tree_depth=MTD,
        init_strategy=init_to_value(values=_HP_INIT)),num_warmup=WARMUP,num_samples=SAMPLES,
        num_chains=CHAINS,chain_method="sequential",progress_bar=True)
    t0=time.perf_counter(); mcmc.run(jax.random.PRNGKey(SEED),**data,extra_fields=("diverging","accept_prob"))
    jax.block_until_ready(mcmc.get_samples()); el=time.perf_counter()-t0
    sbc=mcmc.get_samples(group_by_chain=True)
    flat={k:np.asarray(v).reshape((-1,)+np.asarray(v).shape[2:]) for k,v in sbc.items()}
    def thin(a,k=400):
        a=np.asarray(a); return a[::max(1,len(a)//k)]
    eta_t=thin(flat["eta"]); rate_t=np.exp(eta_t); kap_t=thin(flat["kappa"])[:,None]
    rng=np.random.default_rng(0); pp=rng.negative_binomial(kap_t,kap_t/(kap_t+rate_t))
    lo=np.quantile(pp,.025,0); hi=np.quantile(pp,.975,0)
    pp_cov=float(((y>=lo)&(y<=hi)).mean()); pp_rmse=float(np.sqrt(np.mean((pp.mean(0)-y)**2)))
    ll=(gammaln(y[None,:]+kap_t)-gammaln(kap_t)-gammaln(y[None,:]+1)
        +kap_t*np.log(kap_t/(kap_t+rate_t))+y[None,:]*np.log(rate_t/(kap_t+rate_t)))
    diag=nsummary({k:v for k,v in sbc.items() if "_bw" not in k and k not in
                   ("latent_f","eta","g_contrib","q_contrib","h_monthly","kappa")})
    dn=list(diag.keys())
    drh=np.array([float(np.nanmax(diag[k]["r_hat"])) for k in dn],np.float32)
    des=np.array([float(np.nanmin(diag[k]["n_eff"])) for k in dn],np.float32)
    rh=float(np.nanmax(drh)); es=float(np.nanmin(des))
    ndiv=int(np.asarray(mcmc.get_extra_fields()["diverging"]).sum())
    traces={f"trace_{k}":np.asarray(v,np.float32) for k,v in sbc.items() if np.asarray(v).ndim==2}
    np.savez_compressed(OUT/f"ecdc_noint_{METHOD}_{PRIORSET}.npz",
        cases=y, state_index=obs["state_index"].to_numpy(), time_index=obs["time_index"].to_numpy(),
        country=obs["RegionCode"].to_numpy().astype(str), log_population=obs["log_population"].to_numpy(),
        month_index=(obs["month_of_year"].to_numpy()-1), coords=coords,
        latent_f_mean=flat["latent_f"].mean(0).astype(np.float32),
        latent_f_sd=flat["latent_f"].std(0).astype(np.float32),
        latent_f_thin=thin(flat["latent_f"]).astype(np.float32), eta_thin=eta_t.astype(np.float32),
        g_contrib_thin=thin(flat["g_contrib"]).astype(np.float32),
        q_contrib_thin=thin(flat["q_contrib"]).astype(np.float32),
        h_monthly_thin=thin(flat["h_monthly"]).astype(np.float32),
        pp_draws=pp[:200].astype(np.int32), loglik=ll.astype(np.float32),
        hp_names=np.array(dn), hp_rhat=drh, hp_ess=des,
        max_rhat=rh, min_ess=es, div=ndiv, runtime=el, nchains=CHAINS,
        method=METHOD, priorset=PRIORSET, pp_cov95=pp_cov, pp_rmse=pp_rmse,
        **{f"post_{h}":flat[h].astype(np.float32) for h in HP6 if h in flat},
        post_kappa=flat["kappa"].astype(np.float32), post_beta0=flat["beta0"].astype(np.float32),
        **traces)
    row=dict(method=METHOD,priorset=PRIORSET,model="no_interaction",n=len(y),
        pp_cov95=pp_cov,pp_rmse=pp_rmse,max_rhat=rh,min_ess=es,n_divergences=ndiv,runtime_seconds=el,
        kappa_post_mean=float(flat["kappa"].mean()))
    for h in HP6: row[h+"_post_mean"]=float(flat[h].mean())
    (OUT/f"ecdc_noint_{METHOD}_{PRIORSET}.json").write_text(json.dumps(row,indent=2))
    print(f"  pp_cov95={pp_cov:.3f} pp_rmse={pp_rmse:.1f} kappa={row['kappa_post_mean']:.2f} "
          f"| rhat={rh:.3f} ess={es:.0f} div={ndiv} t={el:.0f}s")

if __name__=="__main__": run()
