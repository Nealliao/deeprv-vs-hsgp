"""NO-INTERACTION ablation on the synthetic benchmark (truth known).
Fits f = g_s + q_t + season (interaction term REMOVED) with either the exact
full-rank GP or HSGP, so we can test whether dropping the (badly-resolved)
interaction actually helps HSGP/DeepRV or hurts.

Truth is known, so we report TWO errors:
  rmse_full : vs latent_f_true = g+q+w   (honest ablation cost: w is unrepresentable)
  rmse_gq   : vs g_true+q_true           (did dropping w corrupt the g/q estimates?)

Saves EVERYTHING needed for later plotting into results/noint_<method>_seed<k>.npz
(per-param rhat/ess, per-chain traces, pointwise loglik, posterior-predictive
draws, thinned latent/eta/component draws, seasonal curve draws, all truths,
coords/pop/indices) plus a compact .json summary.

Env: METHOD=exactgp|hsgp  W S C (warmup/samples/chains)  SEEDS SEED_BASE"""
from pathlib import Path
import os, json, time
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT/".matplotlib"))
import sys; sys.path.insert(0, str(ROOT/"scripts"))
import numpy as np, jax, jax.numpy as jnp, numpyro
import numpyro.distributions as dist
from numpyro.contrib.hsgp.laplacian import eigenfunctions
from numpyro.contrib.hsgp.spectral_densities import diag_spectral_density_squared_exponential as _spd
from numpyro.diagnostics import summary as nsummary
from numpyro.infer import MCMC, NUTS, init_to_value
from scipy.special import gammaln
import _common as C

METHOD = os.environ.get("METHOD", "exactgp")
WARMUP=int(os.environ.get("W","1000")); SAMPLES=int(os.environ.get("S","1000"))
CHAINS=int(os.environ.get("C","4")); TARGET=float(os.environ.get("TARGET","0.9")); MTD=10
N_SEEDS=int(os.environ.get("SEEDS","1")); SEED_BASE=int(os.environ.get("SEED_BASE","1"))
JITTER=float(os.environ.get("JITTER","1e-4"))
SPACE_M=[8,8]; SPACE_ELL=[1.1,1.1]; TIME_M=28; TIME_ELL=45.0
OUT=ROOT/"results"; OUT.mkdir(exist_ok=True)

# Only 6 HPs survive when the interaction is removed. With no interaction there is
# nothing to separate from, so Betancourt's S4.1 non-overlap constraint is moot and
# the lengthscale priors are re-derived from pure S3 containment over the FULL
# informative range [min nearest-neighbour, domain span]:
#   spatial geometry: min-NN=0.034, span=1.03   temporal: 1 month .. 72 months
# The legacy priors deliberately ceded the short half to the interaction
# (space [0.25,1.80], time [10,48]) and would doubly handicap a no-interaction fit.
PRIORSET = os.environ.get("PRIORSET", "noint")   # "noint" (containment) | "legacy"
if PRIORSET == "legacy":
    _LEN = {"space_length": dist.InverseGamma(6.1091, 3.3175),      # P[0.25,1.80]
            "time_length":  dist.InverseGamma(9.3607, 179.029)}     # P[10,48]
else:
    _LEN = {"space_length": dist.InverseGamma(2.3740, 0.2489),      # P[0.034,1.03]
            "time_length":  dist.InverseGamma(2.6641, 15.6605)}     # P[2,48]
PRIORS={"space_alpha":dist.HalfNormal(0.5),"space_length":_LEN["space_length"],
    "time_alpha":dist.HalfNormal(0.5),"time_length":_LEN["time_length"],
    "sigma_h":dist.HalfNormal(0.45),"ell_h":dist.InverseGamma(3.9335,1.9878)}
HP6=list(PRIORS.keys())
_HP_INIT={"space_alpha":0.30,"space_length":0.467,"time_alpha":0.20,"time_length":17.28,
    "sigma_h":0.20,"ell_h":0.40}

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
    def _unflat(flat):
        root={}
        for k,v in flat.items():
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

def model(count,log_population,b_h,state_index,time_index,coords_c,time_c,beta0_prior_mean):
    S=coords_c.shape[0]; T=time_c.shape[0]
    beta0=numpyro.sample("beta0",dist.Normal(beta0_prior_mean,1.0))
    sh=numpyro.sample("sigma_h",PRIORS["sigma_h"]); eh=numpyro.sample("ell_h",PRIORS["ell_h"])
    Lh=jnp.linalg.cholesky(_per(sh,eh)+1e-6*jnp.eye(12))
    zh=numpyro.sample("z_h",dist.Normal(0,1).expand([12])); hraw=b_h+Lh@zh; hm=hraw-hraw.mean()
    sa=numpyro.sample("space_alpha",PRIORS["space_alpha"]); sl=numpyro.sample("space_length",PRIORS["space_length"])
    ta=numpyro.sample("time_alpha",PRIORS["time_alpha"]); tl=numpyro.sample("time_length",PRIORS["time_length"])
    if METHOD=="exactgp":
        zg=numpyro.sample("z_g",dist.Normal(0,1).expand([S])); g=sa*(_chol(coords_c,sl)@zg)
        zq=numpyro.sample("z_q",dist.Normal(0,1).expand([T])); q=ta*(_chol(time_c,tl)@zq)
    elif METHOD=="deeprv":
        zs=numpyro.sample("z_space",dist.Normal(0,1).expand([_ZS]))
        zt=numpyro.sample("z_time", dist.Normal(0,1).expand([_ZT]))
        g=sa*_SDEC.apply({"params":_SP}, zs[None,:], jnp.array([[jnp.log(sl)]]))[0]
        q=ta*_TDEC.apply({"params":_TP}, zt[None,:], jnp.array([[jnp.log(tl)]]))[0]
    else:
        g=hsgp_se("space",coords_c,sa,sl,SPACE_ELL,SPACE_M); q=hsgp_se("time",time_c,ta,tl,TIME_ELL,TIME_M)
    g=g-g.mean(); q=q-q.mean()
    latent=g[state_index]+q[time_index]                      # NO interaction term
    eta=jnp.clip(log_population+beta0+hm[time_index%12]+latent,-35.,20.)
    numpyro.deterministic("latent_f",latent); numpyro.deterministic("eta",eta)
    numpyro.deterministic("g_contrib",g[state_index]); numpyro.deterministic("q_contrib",q[time_index])
    numpyro.deterministic("h_monthly",hm)
    numpyro.sample("obs",dist.Poisson(jnp.exp(eta)),obs=count)

def make_data(panel,regions):
    coords=regions[["x_coord","y_coord"]].to_numpy()-0.5
    T=panel["time_index"].nunique(); tc=(np.arange(T,dtype=float)-np.arange(T).mean())[:,None]
    return dict(count=jnp.asarray(panel["count"].to_numpy(np.int32)),
        log_population=jnp.asarray(panel["log_population"].to_numpy(np.float32)),
        b_h=jnp.asarray(C._b_h.astype(np.float32)),
        state_index=jnp.asarray(panel["state_index"].to_numpy(np.int32)),
        time_index=jnp.asarray(panel["time_index"].to_numpy(np.int32)),
        coords_c=jnp.asarray(coords.astype(np.float32)),time_c=jnp.asarray(tc.astype(np.float32)),
        beta0_prior_mean=float(np.log(1.2e-5))), coords

def run():
    numpyro.set_host_device_count(CHAINS)
    regime,regions=C.load_regime_regions()
    for off in range(N_SEEDS):
        seed=SEED_BASE+off
        panel=C.generate_panel(regime,regions,72,seed); data,coords=make_data(panel,regions)
        print(f"[seed {seed}] NO-INTERACTION  method={METHOD}  priors={PRIORSET}")
        mcmc=MCMC(NUTS(model,target_accept_prob=TARGET,max_tree_depth=MTD,
            init_strategy=init_to_value(values=_HP_INIT)),num_warmup=WARMUP,num_samples=SAMPLES,
            num_chains=CHAINS,chain_method="sequential",progress_bar=True)
        t0=time.perf_counter(); mcmc.run(jax.random.PRNGKey(seed*61+5),**data,
            extra_fields=("diverging","accept_prob","num_steps"))
        jax.block_until_ready(mcmc.get_samples()); el=time.perf_counter()-t0
        sbc=mcmc.get_samples(group_by_chain=True)
        flat={k:np.asarray(v).reshape((-1,)+np.asarray(v).shape[2:]) for k,v in sbc.items()}

        # ── truths ──
        f_true=panel["latent_f_true"].to_numpy(); g_true=panel["g_true"].to_numpy()
        q_true=panel["q_true"].to_numpy(); w_true=panel["w_true"].to_numpy()
        gq_true=g_true+q_true; y=panel["count"].to_numpy()
        lat=flat["latent_f"]; mf=lat.mean(0)
        lo=np.quantile(lat,.025,0); hi=np.quantile(lat,.975,0)
        rmse_full=float(np.sqrt(np.mean((mf-f_true)**2)))
        rmse_gq=float(np.sqrt(np.mean((mf-gq_true)**2)))
        cov_full=float(((f_true>=lo)&(f_true<=hi)).mean()); cov_gq=float(((gq_true>=lo)&(gq_true<=hi)).mean())

        diag=nsummary({k:v for k,v in sbc.items() if "_bw" not in k and k not in
                       ("latent_f","eta","g_contrib","q_contrib","h_monthly")})
        dn=list(diag.keys())
        drh=np.array([float(np.nanmax(diag[k]["r_hat"])) for k in dn],np.float32)
        des=np.array([float(np.nanmin(diag[k]["n_eff"])) for k in dn],np.float32)
        rh=float(np.nanmax(drh)); es=float(np.nanmin(des))
        ex=mcmc.get_extra_fields(); ndiv=int(np.asarray(ex["diverging"]).sum())

        # ── posterior predictive + pointwise loglik (Poisson) ──
        def thin(a,k=400):
            a=np.asarray(a); return a[::max(1,len(a)//k)]
        eta_t=thin(flat["eta"]); rate_t=np.exp(eta_t)
        rng=np.random.default_rng(0); pp=rng.poisson(rate_t)
        ll=(y[None,:]*np.log(np.maximum(rate_t,1e-300))-rate_t-gammaln(y[None,:]+1.0))
        pp_lo=np.quantile(pp,.025,0); pp_hi=np.quantile(pp,.975,0)
        pp_cov=float(((y>=pp_lo)&(y<=pp_hi)).mean()); pp_rmse=float(np.sqrt(np.mean((pp.mean(0)-y)**2)))

        # ── seasonal curve draws ──
        h_draws=thin(flat["h_monthly"]) if "h_monthly" in flat else None

        traces={f"trace_{k}":np.asarray(v,np.float32) for k,v in sbc.items() if np.asarray(v).ndim==2}
        np.savez_compressed(OUT/f"noint_{METHOD}_{PRIORSET}_seed{seed}.npz",
            # data + geometry
            count=y, state_index=panel["state_index"].to_numpy(), time_index=panel["time_index"].to_numpy(),
            log_population=panel["log_population"].to_numpy(), coords=coords,
            month_index=(panel["time_index"].to_numpy()%12),
            # truths
            latent_f_true=f_true, g_true=g_true, q_true=q_true, w_true=w_true, gq_true=gq_true,
            seasonal_h_true=panel["seasonal_h_true"].to_numpy(),
            # posterior fields
            latent_f_mean=mf.astype(np.float32), latent_f_sd=lat.std(0).astype(np.float32),
            latent_f_lo=lo.astype(np.float32), latent_f_hi=hi.astype(np.float32),
            latent_f_thin=thin(lat).astype(np.float32), eta_thin=eta_t.astype(np.float32),
            g_contrib_thin=thin(flat["g_contrib"]).astype(np.float32),
            q_contrib_thin=thin(flat["q_contrib"]).astype(np.float32),
            h_monthly_thin=(h_draws.astype(np.float32) if h_draws is not None else np.zeros(0)),
            # predictive + loglik
            pp_draws=pp[:200].astype(np.int32), loglik=ll.astype(np.float32),
            # diagnostics
            hp_names=np.array(dn), hp_rhat=drh, hp_ess=des,
            max_rhat=rh, min_ess=es, div=ndiv, runtime=el, nchains=CHAINS,
            method=METHOD, seed=seed, priorset=PRIORSET,
            rmse_full=rmse_full, rmse_gq=rmse_gq, coverage_full=cov_full, coverage_gq=cov_gq,
            pp_cov95=pp_cov, pp_rmse=pp_rmse,
            **{f"post_{h}":flat[h].astype(np.float32) for h in HP6 if h in flat},
            **traces)

        row=dict(method=METHOD,model="no_interaction",priorset=PRIORSET,dgp_seed=seed,
            rmse_full=rmse_full,rmse_gq=rmse_gq,coverage_full=cov_full,coverage_gq=cov_gq,
            pp_cov95=pp_cov,pp_rmse=pp_rmse,max_rhat=rh,min_ess=es,n_divergences=ndiv,runtime_seconds=el,
            sd_w_true=float(w_true.std()))
        for h in HP6:
            if h in flat:
                s=flat[h]; row[h+"_post_mean"]=float(s.mean()); row[h+"_truth"]=C.TRUE_HP[h]
                row[h+"_bias"]=float(s.mean())-C.TRUE_HP[h]
        (OUT/f"noint_{METHOD}_{PRIORSET}_seed{seed}.json").write_text(json.dumps(row,indent=2))
        print(f"  rmse_full={rmse_full:.4f} (vs g+q+w)   rmse_gq={rmse_gq:.4f} (vs g+q)")
        print(f"  cover_full={cov_full:.3f} cover_gq={cov_gq:.3f} | rhat={rh:.3f} ess={es:.0f} div={ndiv} t={el:.0f}s")
        print(f"  [saved] noint_{METHOD}_{PRIORSET}_seed{seed}.npz + .json   (sd of true w = {w_true.std():.3f})")

if __name__=="__main__": run()
