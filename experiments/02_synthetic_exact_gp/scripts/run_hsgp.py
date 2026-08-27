"""HSGP on the synthetic benchmark, with the INTERACTION basis count made
configurable so we can sweep capacity (baseline -> optimised -> larger) and, with
truth known, see whether more basis recovers the true field better (good) or
overfits noise (RMSE-vs-truth worsens). Model/priors/season/Poisson identical to
script 106; only the interaction basis dims (and boundary) vary.

Env: HS_ISM (inter-space m/dim, default 8)   HS_ITM (inter-time m, default 12)
     HS_ISELL (inter-space boundary, 0.8)     HS_ITELL (inter-time boundary, 45)
     HS_WARMUP HS_SAMPLES HS_CHAINS HS_TARGET HS_SEEDS HS_SEED_BASE HS_TAG"""
from pathlib import Path
import os, json, time
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT/".matplotlib"))
import sys; sys.path.insert(0,str(ROOT/"scripts"))
import numpy as np, jax, jax.numpy as jnp, numpyro
import numpyro.distributions as dist
from numpyro.contrib.hsgp.laplacian import eigenfunctions
from numpyro.contrib.hsgp.spectral_densities import diag_spectral_density_squared_exponential as _spd
from numpyro.diagnostics import summary as nsummary
from numpyro.infer import MCMC, NUTS, init_to_value
import _common as C

ISM=int(os.environ.get("HS_ISM","8")); ITM=int(os.environ.get("HS_ITM","12"))
ISELL=float(os.environ.get("HS_ISELL","0.8")); ITELL=float(os.environ.get("HS_ITELL","45"))
SPACE_M=int(os.environ.get("HS_SPACE_M","8")); TIME_M=int(os.environ.get("HS_TIME_M","28"))
WARMUP=int(os.environ.get("HS_WARMUP","1000")); SAMPLES=int(os.environ.get("HS_SAMPLES","1000"))
CHAINS=int(os.environ.get("HS_CHAINS","4")); TARGET=float(os.environ.get("HS_TARGET","0.9")); MTD=10
N_SEEDS=int(os.environ.get("HS_SEEDS","1")); SEED_BASE=int(os.environ.get("HS_SEED_BASE","1"))
TAG=os.environ.get("HS_TAG",f"ism{ISM}_itm{ITM}")
OUT=ROOT/"results"; OUT.mkdir(exist_ok=True)

PRIORS={"space_alpha":dist.HalfNormal(0.5),"space_length":dist.InverseGamma(6.1091,3.3175),
    "time_alpha":dist.HalfNormal(0.5),"time_length":dist.InverseGamma(9.3607,179.029),
    "interaction_alpha":dist.HalfNormal(0.5),"interaction_space_length":dist.InverseGamma(5.3661,0.3648),
    "interaction_time_length":dist.InverseGamma(7.3012,30.009),"sigma_h":dist.HalfNormal(0.45),
    "ell_h":dist.InverseGamma(3.9335,1.9878)}
_HP_INIT={"space_alpha":0.30,"space_length":0.467,"time_alpha":0.20,"time_length":17.28,
    "interaction_alpha":0.18,"interaction_space_length":0.057,"interaction_time_length":3.62,
    "sigma_h":0.20,"ell_h":0.40}

def _ss(x,eps=1e-20):
    sx=jnp.where(x>eps,x,jnp.ones_like(x)); return jnp.where(x>eps,jnp.sqrt(sx),jnp.zeros_like(x))
def hsgp_se(name,x,alpha,length,ell,m):
    dim=jnp.shape(x)[-1] if jnp.ndim(x)>1 else 1
    phi=eigenfunctions(x=x,ell=ell,m=m)
    spd=_ss(_spd(alpha=alpha**2,length=length,ell=ell,m=m,dim=dim))
    beta=numpyro.sample(f"{name}_bw",dist.Normal(0,1).expand([phi.shape[-1]]))
    return phi@(spd*beta)
def inter_hsgp(si,ti,cc,tc,alpha,sl,tl):
    ps=eigenfunctions(x=cc,ell=[ISELL,ISELL],m=[ISM,ISM])
    ss=_ss(_spd(alpha=1.0,length=sl,ell=[ISELL,ISELL],m=[ISM,ISM],dim=2))
    pt=eigenfunctions(x=tc,ell=ITELL,m=ITM)
    st=_ss(_spd(alpha=1.0,length=tl,ell=ITELL,m=ITM,dim=1))
    ws=ps[si]*ss; wt=pt[ti]*st; nb=ws.shape[-1]*wt.shape[-1]
    beta=numpyro.sample("interaction_bw",dist.Normal(0,1).expand([nb])).reshape((ws.shape[-1],wt.shape[-1]))
    return alpha*jnp.einsum("ns,nt,st->n",ws,wt,beta)
def _per(sig,ell):
    m=jnp.arange(12.); d=jnp.abs(m[:,None]-m[None,:]); return sig**2*jnp.exp(-2*jnp.sin(jnp.pi*d/12)**2/ell**2)

def model(count,log_population,b_h,state_index,time_index,coords_c,time_c,beta0_prior_mean):
    beta0=numpyro.sample("beta0",dist.Normal(beta0_prior_mean,1.0))
    sh=numpyro.sample("sigma_h",PRIORS["sigma_h"]); eh=numpyro.sample("ell_h",PRIORS["ell_h"])
    Lh=jnp.linalg.cholesky(_per(sh,eh)+1e-6*jnp.eye(12))
    zh=numpyro.sample("z_h",dist.Normal(0,1).expand([12])); hraw=b_h+Lh@zh; hm=hraw-hraw.mean()
    sa=numpyro.sample("space_alpha",PRIORS["space_alpha"]); sl=numpyro.sample("space_length",PRIORS["space_length"])
    ta=numpyro.sample("time_alpha",PRIORS["time_alpha"]); tl=numpyro.sample("time_length",PRIORS["time_length"])
    ia=numpyro.sample("interaction_alpha",PRIORS["interaction_alpha"])
    isl=numpyro.sample("interaction_space_length",PRIORS["interaction_space_length"])
    itl=numpyro.sample("interaction_time_length",PRIORS["interaction_time_length"])
    g=hsgp_se("space",coords_c,sa,sl,[1.1,1.1],[SPACE_M,SPACE_M])
    q=hsgp_se("time",time_c,ta,tl,45.0,TIME_M)
    w=inter_hsgp(state_index,time_index,coords_c,time_c,ia,isl,itl)
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
    nb=ISM*ISM*ITM
    for off in range(N_SEEDS):
        seed=SEED_BASE+off
        panel=C.generate_panel(regime,regions,72,seed); data=make_data(panel,regions)
        print(f"[seed {seed}] HSGP tag={TAG}  inter-basis={ISM}x{ISM}x{ITM}={nb} (boundary s={ISELL})")
        init=init_to_value(values=_HP_INIT)
        mcmc=MCMC(NUTS(model,target_accept_prob=TARGET,max_tree_depth=MTD,init_strategy=init),
            num_warmup=WARMUP,num_samples=SAMPLES,num_chains=CHAINS,chain_method="sequential",progress_bar=True)
        t0=time.perf_counter(); mcmc.run(jax.random.PRNGKey(seed*131+7),**data,extra_fields=("diverging",))
        jax.block_until_ready(mcmc.get_samples()); el=time.perf_counter()-t0
        sbc=mcmc.get_samples(group_by_chain=True)
        flat={k:np.asarray(v).reshape((-1,)+np.asarray(v).shape[2:]) for k,v in sbc.items()}
        f_true=panel["latent_f_true"].to_numpy(); mf=flat["latent_f"].mean(0)
        lo=np.quantile(flat["latent_f"],.025,0); hi=np.quantile(flat["latent_f"],.975,0)
        rmse=float(np.sqrt(np.mean((mf-f_true)**2))); cov=float(((f_true>=lo)&(f_true<=hi)).mean())
        diag=nsummary({k:v for k,v in sbc.items() if "_bw" not in k and k not in("g_contrib","q_contrib","w_contrib","latent_f")})
        rh=np.nanmax([np.nanmax(diag[k]["r_hat"]) for k in diag]); es=np.nanmin([np.nanmin(diag[k]["n_eff"]) for k in diag])
        ndiv=int(np.asarray(mcmc.get_extra_fields()["diverging"]).sum())
        row=dict(method="hsgp",tag=TAG,inter_basis=nb,ism=ISM,itm=ITM,dgp_seed=seed,rmse_f=rmse,coverage_95=cov,
            max_rhat=float(rh),min_ess=float(es),n_divergences=ndiv,runtime_seconds=el,
            state_index=panel["state_index"].tolist(),time_index=panel["time_index"].tolist(),
            latent_f_true=f_true.tolist(),latent_f_post_mean=mf.tolist(),latent_f_post_sd=flat["latent_f"].std(0).tolist(),
            count=panel["count"].tolist())
        vt=0.
        for c in("g_contrib","q_contrib","w_contrib"): v=float(flat[c].var(1).mean()); row["var_"+c]=v; vt+=v
        for c in("g_contrib","q_contrib","w_contrib"): row["frac_"+c]=row["var_"+c]/vt
        for hp in C.HP_NAMES:
            s=flat[hp]; row[hp+"_post_mean"]=float(s.mean()); row[hp+"_truth"]=C.TRUE_HP[hp]; row[hp+"_bias"]=float(s.mean())-C.TRUE_HP[hp]
            row[hp+"_post_q025"]=float(np.quantile(s,.025)); row[hp+"_post_q975"]=float(np.quantile(s,.975)); row[hp+"_samples"]=s.tolist()
        (OUT/f"hsgp_{TAG}_seed{seed}.json").write_text(json.dumps(row))
        print(f"  RMSE={rmse:.4f} cover={cov:.3f} rhat={rh:.3f} ess={es:.0f} div={ndiv} t={el:.0f}s  var w={row['frac_w_contrib']:.2f}")

if __name__=="__main__": run()
