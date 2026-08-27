"""EXACT full-rank separable GP on the synthetic benchmark (truth known).
Gold-standard reference: matches the DGP structure, so it should recover the
true field best. Same 9-HP priors / season / Poisson likelihood as baseline HSGP
(script 106); ONLY the latent representation differs (exact non-centred Cholesky
instead of HSGP basis). Truth-metrics: RMSE(f), coverage, variance decomposition,
per-HP bias. Writes only under this project's results/.

Env: EG_WARMUP EG_SAMPLES EG_CHAINS EG_TARGET EG_SEEDS EG_SEED_BASE"""
from pathlib import Path
import os, json, time
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))
import sys; sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np, jax, jax.numpy as jnp, numpyro
import numpyro.distributions as dist
from numpyro.diagnostics import summary as nsummary
from numpyro.infer import MCMC, NUTS, init_to_value
import _common as C

WARMUP=int(os.environ.get("EG_WARMUP","1000")); SAMPLES=int(os.environ.get("EG_SAMPLES","1000"))
CHAINS=int(os.environ.get("EG_CHAINS","4")); TARGET=float(os.environ.get("EG_TARGET","0.9"))
MTD=10; N_SEEDS=int(os.environ.get("EG_SEEDS","1")); SEED_BASE=int(os.environ.get("EG_SEED_BASE","1"))
JITTER=float(os.environ.get("EG_JITTER","1e-4"))  # 1e-4 matches ECDC (cond~1e5); 1e-5 was marginally stiff
OUT=ROOT/"results"; OUT.mkdir(exist_ok=True)

PRIORS={"space_alpha":dist.HalfNormal(0.5),"space_length":dist.InverseGamma(6.1091,3.3175),
    "time_alpha":dist.HalfNormal(0.5),"time_length":dist.InverseGamma(9.3607,179.029),
    "interaction_alpha":dist.HalfNormal(0.5),"interaction_space_length":dist.InverseGamma(5.3661,0.3648),
    "interaction_time_length":dist.InverseGamma(7.3012,30.009),"sigma_h":dist.HalfNormal(0.45),
    "ell_h":dist.InverseGamma(3.9335,1.9878)}
_HP_INIT={"space_alpha":0.30,"space_length":0.467,"time_alpha":0.20,"time_length":17.28,
    "interaction_alpha":0.18,"interaction_space_length":0.057,"interaction_time_length":3.62,
    "sigma_h":0.20,"ell_h":0.40}

def _se(x,ell):
    d2=jnp.sum((x[:,None,:]-x[None,:,:])**2,-1); return jnp.exp(-0.5*d2/ell**2)
def _chol(x,ell):
    K=_se(x,ell); return jnp.linalg.cholesky(K+JITTER*jnp.eye(K.shape[0]))
def _per(sig,ell):
    m=jnp.arange(12.); d=jnp.abs(m[:,None]-m[None,:]); return sig**2*jnp.exp(-2*jnp.sin(jnp.pi*d/12)**2/ell**2)

def model(count,log_population,b_h,state_index,time_index,coords_c,time_c,beta0_prior_mean):
    S=coords_c.shape[0]; T=time_c.shape[0]
    beta0=numpyro.sample("beta0",dist.Normal(beta0_prior_mean,1.0))
    sh=numpyro.sample("sigma_h",PRIORS["sigma_h"]); eh=numpyro.sample("ell_h",PRIORS["ell_h"])
    Lh=jnp.linalg.cholesky(_per(sh,eh)+1e-6*jnp.eye(12))
    zh=numpyro.sample("z_h",dist.Normal(0,1).expand([12])); hraw=b_h+Lh@zh; hm=hraw-hraw.mean()
    sa=numpyro.sample("space_alpha",PRIORS["space_alpha"]); sl=numpyro.sample("space_length",PRIORS["space_length"])
    ta=numpyro.sample("time_alpha",PRIORS["time_alpha"]); tl=numpyro.sample("time_length",PRIORS["time_length"])
    ia=numpyro.sample("interaction_alpha",PRIORS["interaction_alpha"])
    isl=numpyro.sample("interaction_space_length",PRIORS["interaction_space_length"])
    itl=numpyro.sample("interaction_time_length",PRIORS["interaction_time_length"])
    Lg=_chol(coords_c,sl); zg=numpyro.sample("z_g",dist.Normal(0,1).expand([S])); g=sa*(Lg@zg)
    Lq=_chol(time_c,tl);   zq=numpyro.sample("z_q",dist.Normal(0,1).expand([T])); q=ta*(Lq@zq)
    Lws=_chol(coords_c,isl); Lwt=_chol(time_c,itl)
    Z=numpyro.sample("z_w",dist.Normal(0,1).expand([S,T])); W=ia*(Lws@Z@Lwt.T)
    w=W[state_index,time_index]
    g=g-g.mean(); q=q-q.mean(); w=w-w.mean()
    latent=g[state_index]+q[time_index]+w
    eta=jnp.clip(log_population+beta0+hm[time_index%12]+latent,-35.,20.)
    numpyro.deterministic("latent_f",latent)
    numpyro.deterministic("g_contrib",g[state_index]); numpyro.deterministic("q_contrib",q[time_index])
    numpyro.deterministic("w_contrib",w)
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
        beta0_prior_mean=float(np.log(1.2e-5)))

def run():
    numpyro.set_host_device_count(CHAINS)
    regime,regions=C.load_regime_regions()
    for off in range(N_SEEDS):
        seed=SEED_BASE+off
        panel=C.generate_panel(regime,regions,72,seed); data=make_data(panel,regions)
        print(f"[seed {seed}] exact GP  latent=z_g(20)+z_q(72)+z_w(20x72)+z_h(12)+9HP")
        init=init_to_value(values=_HP_INIT)
        mcmc=MCMC(NUTS(model,target_accept_prob=TARGET,max_tree_depth=MTD,init_strategy=init),
            num_warmup=WARMUP,num_samples=SAMPLES,num_chains=CHAINS,chain_method="sequential",progress_bar=True)
        t0=time.perf_counter(); mcmc.run(jax.random.PRNGKey(seed*97+3),**data,extra_fields=("diverging","accept_prob"))
        jax.block_until_ready(mcmc.get_samples()); el=time.perf_counter()-t0
        sbc=mcmc.get_samples(group_by_chain=True)
        flat={k:np.asarray(v).reshape((-1,)+np.asarray(v).shape[2:]) for k,v in sbc.items()}
        f_true=panel["latent_f_true"].to_numpy(); mf=flat["latent_f"].mean(0)
        lo=np.quantile(flat["latent_f"],.025,0); hi=np.quantile(flat["latent_f"],.975,0)
        rmse=float(np.sqrt(np.mean((mf-f_true)**2))); cov=float(((f_true>=lo)&(f_true<=hi)).mean())
        diag=nsummary({k:v for k,v in sbc.items() if k not in("g_contrib","q_contrib","w_contrib","latent_f")})
        rh=np.nanmax([np.nanmax(diag[k]["r_hat"]) for k in diag]); es=np.nanmin([np.nanmin(diag[k]["n_eff"]) for k in diag])
        ndiv=int(np.asarray(mcmc.get_extra_fields()["diverging"]).sum())
        row=dict(method="exactgp",dgp_seed=seed,rmse_f=rmse,coverage_95=cov,max_rhat=float(rh),
            min_ess=float(es),n_divergences=ndiv,runtime_seconds=el,
            state_index=panel["state_index"].tolist(),time_index=panel["time_index"].tolist(),
            latent_f_true=f_true.tolist(),latent_f_post_mean=mf.tolist(),
            latent_f_post_sd=flat["latent_f"].std(0).tolist(),count=panel["count"].tolist())
        vt=0.
        for c in("g_contrib","q_contrib","w_contrib"): v=float(flat[c].var(1).mean()); row["var_"+c]=v; vt+=v
        for c in("g_contrib","q_contrib","w_contrib"): row["frac_"+c]=row["var_"+c]/vt
        # true variance decomposition (from the DGP components of f_true is not separable here; report recovered)
        for hp in C.HP_NAMES:
            s=flat[hp]; row[hp+"_post_mean"]=float(s.mean()); row[hp+"_truth"]=C.TRUE_HP[hp]
            row[hp+"_bias"]=float(s.mean())-C.TRUE_HP[hp]
            row[hp+"_post_q025"]=float(np.quantile(s,.025)); row[hp+"_post_q975"]=float(np.quantile(s,.975))
            row[hp+"_samples"]=s.tolist()
        (OUT/f"exactgp_seed{seed}.json").write_text(json.dumps(row))
        print(f"  RMSE={rmse:.4f} cover={cov:.3f} rhat={rh:.3f} ess={es:.0f} div={ndiv} t={el:.0f}s")
        print(f"  var g:q:w = {row['frac_g_contrib']:.2f}:{row['frac_q_contrib']:.2f}:{row['frac_w_contrib']:.2f}")
        for hp in C.HP_NAMES:
            print(f"    {hp:26s} truth={C.TRUE_HP[hp]:7.3f} post={row[hp+'_post_mean']:7.3f} bias={row[hp+'_bias']:+.3f}")

if __name__=="__main__": run()
