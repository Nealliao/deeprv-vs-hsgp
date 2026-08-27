"""ECDC held-out CV — DeepRV. Fit on 85% train, predict 15% held-out. Same shared
mask + held-out ELPD/RMSE/coverage as exact-GP and HSGP holdout runs. Conditional
decoders + model identical to script 114."""
from pathlib import Path
import os, json, time
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT/".matplotlib"))
import numpy as np, pandas as pd, jax, jax.numpy as jnp
import flax.linen as nn
from flax.core import freeze
import numpyro, numpyro.distributions as dist
from numpyro.diagnostics import summary as nsummary
from numpyro.infer import MCMC, NUTS, init_to_value
from scipy.special import gammaln, logsumexp

PANEL = ROOT/"data"/"processed"/"ecdc_measles_panel.csv"
if not PANEL.exists(): PANEL = ROOT/"data"/"ecdc_measles_panel.csv"
MODEL_DIR = ROOT/"data"/"processed"/"official_deeprv_models"
if not (MODEL_DIR/"ecdc_deeprv_inter.npz").exists(): MODEL_DIR = ROOT/"decoders"
OUT = Path(__file__).resolve().parent/"results"
MASK = np.load(Path(__file__).resolve().parent/"mask.npz"); TRAIN=MASK["train"]; HOLD=MASK["holdout"]
WARMUP=int(os.environ.get("W","1000")); SAMPLES=int(os.environ.get("S","800")); CHAINS=int(os.environ.get("C","2"))

PRIORS={"space_alpha":dist.HalfNormal(1.0),"space_length":dist.InverseGamma(6.1091,3.3175),
 "time_alpha":dist.HalfNormal(1.0),"time_length":dist.InverseGamma(6.5707,167.3694),
 "interaction_alpha":dist.HalfNormal(1.0),"interaction_space_length":dist.InverseGamma(5.3661,0.3648),
 "interaction_time_length":dist.InverseGamma(6.2718,27.0192),"sigma_h":dist.HalfNormal(0.45),
 "ell_h":dist.InverseGamma(3.9335,1.9878)}
_HP_INIT={"space_alpha":0.5,"space_length":0.467,"time_alpha":0.5,"time_length":22.0,
 "interaction_alpha":0.3,"interaction_space_length":0.057,"interaction_time_length":3.7,"sigma_h":0.30,"ell_h":0.40}

class ConditionalDecoder(nn.Module):
    hidden_dim: int; output_dim: int
    @nn.compact
    def __call__(self, z, log_ell):
        x=jnp.concatenate([z,log_ell],axis=-1)
        x=nn.tanh(nn.Dense(self.hidden_dim)(x)); x=nn.tanh(nn.Dense(self.hidden_dim)(x))
        return nn.Dense(self.output_dim)(x)
def _unflatten(flat):
    root={}
    for key,value in flat.items():
        parts=key.split("/"); cur=root
        for p in parts[:-1]: cur=cur.setdefault(p,{})
        cur[parts[-1]]=jnp.asarray(value)
    return freeze(root)
def load_decoder(path):
    b=np.load(path); params=_unflatten({k:b[k] for k in b.files if "/" in k})
    return ConditionalDecoder(hidden_dim=int(b["hidden_dim"]),output_dim=int(b["output_dim"])),params,int(b["z_dim"])
def _per(sig,ell): m=jnp.arange(12.); d=jnp.abs(m[:,None]-m[None,:]); return sig**2*jnp.exp(-2*jnp.sin(jnp.pi*d/12)**2/ell**2)

def make_model(sd,sp,zsd,td,tp,ztd,idc,ip,zwd):
    def model(count,log_population,state_index,time_index,month_index,beta0_prior_mean,S,T,train_mask):
        beta0=numpyro.sample("beta0",dist.Normal(beta0_prior_mean,3.0))
        isk=numpyro.sample("inv_sqrt_kappa",dist.HalfNormal(1.0)); kappa=numpyro.deterministic("kappa",isk**(-2))
        sh=numpyro.sample("sigma_h",PRIORS["sigma_h"]); eh=numpyro.sample("ell_h",PRIORS["ell_h"])
        Lh=jnp.linalg.cholesky(_per(sh,eh)+1e-6*jnp.eye(12)); zh=numpyro.sample("z_h",dist.Normal(0,1).expand([12]))
        hm=Lh@zh; hm=hm-hm.mean()
        sa=numpyro.sample("space_alpha",PRIORS["space_alpha"]); sl=numpyro.sample("space_length",PRIORS["space_length"])
        ta=numpyro.sample("time_alpha",PRIORS["time_alpha"]); tl=numpyro.sample("time_length",PRIORS["time_length"])
        ia=numpyro.sample("interaction_alpha",PRIORS["interaction_alpha"])
        isl=numpyro.sample("interaction_space_length",PRIORS["interaction_space_length"])
        itl=numpyro.sample("interaction_time_length",PRIORS["interaction_time_length"])
        z_s=numpyro.sample("z_space",dist.Normal(0,1).expand([zsd])); z_t=numpyro.sample("z_time",dist.Normal(0,1).expand([ztd]))
        z_w=numpyro.sample("z_inter",dist.Normal(0,1).expand([zwd]))
        g=sa*sd.apply({"params":sp},z_s[None,:],jnp.array([[jnp.log(sl)]]))[0]; g=g-g.mean()
        q=ta*td.apply({"params":tp},z_t[None,:],jnp.array([[jnp.log(tl)]]))[0]; q=q-q.mean()
        w=ia*idc.apply({"params":ip},z_w[None,:],jnp.array([[jnp.log(isl),jnp.log(itl)]]))[0]; w=w-w.mean()
        wm=w.reshape(S,T)
        eta=jnp.clip(log_population+beta0+hm[month_index]+g[state_index]+q[time_index]+wm[state_index,time_index],-30.,20.)
        rate=jnp.exp(eta); numpyro.deterministic("rate",rate)
        with numpyro.handlers.mask(mask=train_mask):
            numpyro.sample("obs",dist.NegativeBinomial2(mean=rate,concentration=kappa),obs=count)
    return model

def run():
    OUT.mkdir(exist_ok=True); numpyro.set_host_device_count(CHAINS)
    sd,sp,zsd=load_decoder(MODEL_DIR/"ecdc_deeprv_space.npz"); td,tp,ztd=load_decoder(MODEL_DIR/"ecdc_deeprv_time.npz")
    idc,ip,zwd=load_decoder(MODEL_DIR/"ecdc_deeprv_inter.npz")
    print(f"[DeepRV holdout] decoders z_s={zsd} z_t={ztd} z_w={zwd}  train={TRAIN.sum()} holdout={HOLD.sum()}")
    p=pd.read_csv(PANEL); obs=p[p["observed"]==1].copy(); y=obs["cases"].to_numpy()
    data=dict(count=jnp.asarray(y.astype(np.int32)),log_population=jnp.asarray(obs["log_population"].to_numpy(np.float32)),
        state_index=jnp.asarray(obs["state_index"].to_numpy(np.int32)),time_index=jnp.asarray(obs["time_index"].to_numpy(np.int32)),
        month_index=jnp.asarray((obs["month_of_year"].to_numpy()-1).astype(np.int32)),
        beta0_prior_mean=float(np.log(obs["cases"].mean()+1)-obs["log_population"].mean()),
        S=int(p["state_index"].nunique()),T=int(p["time_index"].nunique()),train_mask=jnp.asarray(TRAIN))
    mcmc=MCMC(NUTS(make_model(sd,sp,zsd,td,tp,ztd,idc,ip,zwd),target_accept_prob=0.9,max_tree_depth=12,
        init_strategy=init_to_value(values=_HP_INIT)),num_warmup=WARMUP,num_samples=SAMPLES,num_chains=CHAINS,
        chain_method="sequential",progress_bar=True)
    t0=time.perf_counter(); mcmc.run(jax.random.PRNGKey(11),**data,extra_fields=("diverging",))
    jax.block_until_ready(mcmc.get_samples()); el=time.perf_counter()-t0
    sbc=mcmc.get_samples(group_by_chain=True)
    flat={k:np.asarray(v).reshape((-1,)+np.asarray(v).shape[2:]) for k,v in sbc.items()}
    rate=flat["rate"]; kap=flat["kappa"]; rh=rate[:,HOLD]; yh=y[HOLD]; kp=kap[:,None]
    ll=(gammaln(yh[None,:]+kp)-gammaln(kp)-gammaln(yh[None,:]+1)+kp*np.log(kp/(kp+rh))+yh[None,:]*np.log(rh/(kp+rh)))
    elpd_i=logsumexp(ll,0)-np.log(ll.shape[0]); elpd=float(elpd_i.sum()); elpd_se=float(np.sqrt(len(elpd_i)*elpd_i.var()))
    rng=np.random.default_rng(0); pp=rng.negative_binomial(kp,kp/(kp+rh))
    predmean=pp.mean(0); lo=np.quantile(pp,.025,0); hi=np.quantile(pp,.975,0)
    rmse=float(np.sqrt(np.mean((predmean-yh)**2))); cov=float(((yh>=lo)&(yh<=hi)).mean())
    ndiv=int(np.asarray(mcmc.get_extra_fields()["diverging"]).sum())
    diag=nsummary({k:v for k,v in sbc.items() if k not in("rate","kappa")})
    mrh=float(np.nanmax([np.nanmax(diag[k]["r_hat"]) for k in diag]))
    row=dict(method="deeprv",n_train=int(TRAIN.sum()),n_hold=int(HOLD.sum()),
        holdout_elpd=elpd,holdout_elpd_se=elpd_se,holdout_rmse=rmse,holdout_cov95=cov,max_rhat=mrh,n_div=ndiv,runtime=el,
        holdout_pred_mean=predmean.tolist(),holdout_y=yh.tolist(),holdout_elpd_i=elpd_i.tolist())
    (OUT/"holdout_deeprv.json").write_text(json.dumps(row))
    print(f"  HELDOUT elpd={elpd:.1f}+/-{elpd_se:.1f}  rmse={rmse:.2f}  cov95={cov:.3f}  | rhat={mrh:.3f} div={ndiv} t={el:.0f}s")

if __name__=="__main__": run()
