"""
114_ecdc_deeprv_joint_hp_nb.py
================================================================================
DeepRV joint 9-HP inference on real ECDC measles (2010-2019), NB likelihood.
Counterpart of HSGP script 111, using conditional DeepRV decoders (script 113)
instead of the HSGP basis. Same DGP-free real data, same NB, same data-driven
seasonal, same priors — only the latent prior representation differs.

f_{c,t} = sigma_s*g_norm[c] + sigma_t*q_norm[t] + sigma_w*w_norm[c,t]
  g_norm = space_dec(z_s, log_ell_s)   z_s ~ N(0,I_15)
  q_norm = time_dec (z_t, log_ell_t)   z_t ~ N(0,I_12)
  w_norm = inter_dec(z_w, log_ell_ws, log_ell_wt)   z_w ~ N(0,I_220)
  h_t    = data-driven cyclic GP (no fixed mean)
  Y ~ NB2( E_c * exp(beta0 + h_t + f), kappa ),  1/sqrt(kappa) ~ HalfNormal(1)

Env: ECDC_NUM_WARMUP/SAMPLES/CHAINS, ECDC_TARGET_ACCEPT, ECDC_SEED
"""

from pathlib import Path
import os
import json
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

import numpy as np
import pandas as pd

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.core import freeze
import numpyro
import numpyro.distributions as dist
from numpyro.diagnostics import summary as numpyro_summary
from numpyro.infer import MCMC, NUTS, init_to_value

DATA_DIR  = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = Path("/Users/Zhuanz/Desktop/ECDC_measles_HSGP_vs_DeepRV/decoders_final2")
PANEL     = DATA_DIR / "ecdc_measles_panel.csv"
OUTPUT_DIR = DATA_DIR / "ecdc"

NUM_WARMUP  = int(os.environ.get("ECDC_NUM_WARMUP",  "1000"))
NUM_SAMPLES = int(os.environ.get("ECDC_NUM_SAMPLES", "1000"))
NUM_CHAINS  = int(os.environ.get("ECDC_NUM_CHAINS",  "4"))
TARGET_ACCEPT  = float(os.environ.get("ECDC_TARGET_ACCEPT", "0.95"))
MAX_TREE_DEPTH = 12
SEED = int(os.environ.get("ECDC_SEED", "20260628"))

HP_NAMES = ["space_alpha", "space_length", "time_alpha", "time_length",
            "interaction_alpha", "interaction_space_length", "interaction_time_length",
            "sigma_h", "ell_h"]

PRIORS = {
    "space_alpha":               dist.HalfNormal(1.0),
    "space_length":              dist.InverseGamma(6.1091, 3.3175),
    "time_alpha":                dist.HalfNormal(1.0),
    "time_length":               dist.InverseGamma(6.5707, 167.3694),
    "interaction_alpha":         dist.HalfNormal(1.0),
    "interaction_space_length":  dist.InverseGamma(5.3661, 0.3648),
    "interaction_time_length":   dist.InverseGamma(6.2718, 27.0192),
    "sigma_h":                   dist.HalfNormal(0.45),
    "ell_h":                     dist.InverseGamma(3.9335, 1.9878),
}
_HP_INIT = {
    "space_alpha": 0.5, "space_length": 0.467, "time_alpha": 0.5, "time_length": 22.0,
    "interaction_alpha": 0.3, "interaction_space_length": 0.057, "interaction_time_length": 3.7,
    "sigma_h": 0.30, "ell_h": 0.40,
}


# ─── Decoder ──────────────────────────────────────────────────────────────────

class ConditionalDecoder(nn.Module):
    hidden_dim: int
    output_dim: int

    @nn.compact
    def __call__(self, z, log_ell):
        x = jnp.concatenate([z, log_ell], axis=-1)
        x = nn.tanh(nn.Dense(self.hidden_dim)(x))
        x = nn.tanh(nn.Dense(self.hidden_dim)(x))
        return nn.Dense(self.output_dim)(x)


def _unflatten(flat):
    root = {}
    for key, value in flat.items():
        parts = key.split("/"); cur = root
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = jnp.asarray(value)
    return freeze(root)


def load_decoder(path):
    b = np.load(path)
    params = _unflatten({k: b[k] for k in b.files if "/" in k})
    model = ConditionalDecoder(hidden_dim=int(b["hidden_dim"]), output_dim=int(b["output_dim"]))
    return model, params, int(b["z_dim"]), int(b["n_ell"])


def _periodic_kernel_jax(sigma_h, ell_h):
    m12 = jnp.arange(12, dtype=jnp.float32)
    diff = jnp.abs(m12[:, None] - m12[None, :])
    return sigma_h**2 * jnp.exp(-2.0 * jnp.sin(jnp.pi * diff / 12.0)**2 / ell_h**2)


# ─── Data ─────────────────────────────────────────────────────────────────────

def load_data():
    p = pd.read_csv(PANEL)
    obs = p[p["observed"] == 1].copy()
    beta0_guess = float(np.log(obs["cases"].mean() + 1) - obs["log_population"].mean())
    data = {
        "count":          jnp.asarray(obs["cases"].to_numpy(dtype=np.int32)),
        "log_population": jnp.asarray(obs["log_population"].to_numpy(dtype=np.float32)),
        "state_index":    jnp.asarray(obs["state_index"].to_numpy(dtype=np.int32)),
        "time_index":     jnp.asarray(obs["time_index"].to_numpy(dtype=np.int32)),
        "month_index":    jnp.asarray((obs["month_of_year"].to_numpy() - 1).astype(np.int32)),
        "beta0_prior_mean": beta0_guess,
        "S": int(p["state_index"].nunique()), "T": int(p["time_index"].nunique()),
    }
    return data, obs, p


# ─── Model ────────────────────────────────────────────────────────────────────

def make_model(space_dec, sp, z_s_dim, time_dec, tp, z_t_dim, inter_dec, ip, z_w_dim):
    def model(count, log_population, state_index, time_index, month_index, beta0_prior_mean, S, T):
        beta0 = numpyro.sample("beta0", dist.Normal(beta0_prior_mean, 3.0))
        inv_sqrt_kappa = numpyro.sample("inv_sqrt_kappa", dist.HalfNormal(1.0))
        kappa = numpyro.deterministic("kappa", inv_sqrt_kappa ** (-2))

        sigma_h = numpyro.sample("sigma_h", PRIORS["sigma_h"])
        ell_h   = numpyro.sample("ell_h",   PRIORS["ell_h"])
        L_h = jnp.linalg.cholesky(_periodic_kernel_jax(sigma_h, ell_h) + 1e-6 * jnp.eye(12))
        z_h = numpyro.sample("z_h", dist.Normal(0.0, 1.0).expand([12]))
        h_monthly = (L_h @ z_h); h_monthly = h_monthly - h_monthly.mean()

        sa = numpyro.sample("space_alpha", PRIORS["space_alpha"])
        sl = numpyro.sample("space_length", PRIORS["space_length"])
        ta = numpyro.sample("time_alpha", PRIORS["time_alpha"])
        tl = numpyro.sample("time_length", PRIORS["time_length"])
        ia = numpyro.sample("interaction_alpha", PRIORS["interaction_alpha"])
        isl = numpyro.sample("interaction_space_length", PRIORS["interaction_space_length"])
        itl = numpyro.sample("interaction_time_length", PRIORS["interaction_time_length"])

        z_s = numpyro.sample("z_space", dist.Normal(0.0, 1.0).expand([z_s_dim]))
        z_t = numpyro.sample("z_time",  dist.Normal(0.0, 1.0).expand([z_t_dim]))
        z_w = numpyro.sample("z_inter", dist.Normal(0.0, 1.0).expand([z_w_dim]))

        g_norm = space_dec.apply({"params": sp}, z_s[None, :], jnp.array([[jnp.log(sl)]]))[0]
        q_norm = time_dec.apply({"params": tp}, z_t[None, :], jnp.array([[jnp.log(tl)]]))[0]
        w_norm = inter_dec.apply({"params": ip}, z_w[None, :], jnp.array([[jnp.log(isl), jnp.log(itl)]]))[0]

        g = sa * g_norm; g = g - jnp.mean(g)
        q = ta * q_norm; q = q - jnp.mean(q)
        w = ia * w_norm; w = w - jnp.mean(w)
        w_mat = w.reshape(S, T)

        latent_f = g[state_index] + q[time_index] + w_mat[state_index, time_index]
        eta = log_population + beta0 + h_monthly[month_index] + latent_f
        eta = jnp.clip(eta, -30.0, 20.0)
        rate = jnp.exp(eta)
        numpyro.deterministic("latent_f", latent_f)
        numpyro.deterministic("rate", rate)
        numpyro.deterministic("g_contrib", g[state_index])
        numpyro.deterministic("q_contrib", q[time_index])
        numpyro.deterministic("w_contrib", w_mat[state_index, time_index])
        numpyro.sample("obs", dist.NegativeBinomial2(mean=rate, concentration=kappa), obs=count)
    return model


# ─── Run ──────────────────────────────────────────────────────────────────────

def run():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    numpyro.set_host_device_count(NUM_CHAINS)
    space_dec, sp, zsd, _ = load_decoder(MODEL_DIR / "ecdc_deeprv_space.npz")
    time_dec, tp, ztd, _  = load_decoder(MODEL_DIR / "ecdc_deeprv_time.npz")
    inter_dec, ip, zwd, _ = load_decoder(MODEL_DIR / "ecdc_deeprv_inter.npz")
    print(f"[decoders] z_s={zsd} z_t={ztd} z_w={zwd}  total latent={zsd+ztd+zwd}")

    data, obs, panel = load_data()
    print(f"[data] S={data['S']} T={data['T']} observed={len(obs)} beta0_guess={data['beta0_prior_mean']:.2f}")
    model = make_model(space_dec, sp, zsd, time_dec, tp, ztd, inter_dec, ip, zwd)

    kernel = NUTS(model, target_accept_prob=TARGET_ACCEPT, max_tree_depth=MAX_TREE_DEPTH,
                  init_strategy=init_to_value(values=_HP_INIT))
    mcmc = MCMC(kernel, num_warmup=NUM_WARMUP, num_samples=NUM_SAMPLES,
                num_chains=NUM_CHAINS, chain_method="sequential", progress_bar=True)
    t0 = time.perf_counter()
    mcmc.run(jax.random.PRNGKey(SEED), **data, extra_fields=("accept_prob", "diverging", "num_steps"))
    jax.block_until_ready(mcmc.get_samples())
    elapsed = time.perf_counter() - t0

    sbc = mcmc.get_samples(group_by_chain=True)
    flat = {k: np.asarray(v).reshape((-1,) + tuple(np.asarray(v).shape[2:])) for k, v in sbc.items()}
    _skip = ("latent_f", "rate", "kappa", "g_contrib", "q_contrib", "w_contrib")
    diag = numpyro_summary({k: sbc[k] for k in sbc if k not in _skip})
    rhats, esss = [], []
    for k in diag:
        rhats += np.asarray(diag[k]["r_hat"]).ravel().tolist(); esss += np.asarray(diag[k]["n_eff"]).ravel().tolist()
    extra = mcmc.get_extra_fields(); n_div = int(np.asarray(extra["diverging"]).sum())

    rate_s = flat["rate"]; kappa_s = flat["kappa"]; rng = np.random.default_rng(0)
    idx = rng.choice(len(rate_s), size=min(400, len(rate_s)), replace=False)
    pp = np.array([rng.negative_binomial(kappa_s[i], kappa_s[i] / (kappa_s[i] + rate_s[i])) for i in idx])
    y = obs["cases"].to_numpy()
    lo = np.quantile(pp, 0.025, axis=0); hi = np.quantile(pp, 0.975, axis=0); pred = pp.mean(axis=0)
    coverage = float(((y >= lo) & (y <= hi)).mean())
    rmse = float(np.sqrt(np.mean((pred - y) ** 2))); mae = float(np.mean(np.abs(pred - y)))
    # ── enhanced NPZ save (per-param rhat/ess, traces, loglik, pp_draws, eta) for full figure suite ──
    from scipy.special import gammaln as _gl
    def _thinE(a, k=400):
        a = np.asarray(a); return a[::max(1, len(a) // k)]
    _dn = list(diag.keys())
    _drh = np.array([float(np.nanmax(np.asarray(diag[k]["r_hat"]))) for k in _dn], np.float32)
    _des = np.array([float(np.nanmin(np.asarray(diag[k]["n_eff"]))) for k in _dn], np.float32)
    _trace = {f"trace_{k}": np.asarray(v).astype(np.float32) for k, v in sbc.items() if np.asarray(v).ndim == 2}
    _rt = _thinE(rate_s); _kp = _thinE(kappa_s)
    _ll = (_gl(y[None, :] + _kp[:, None]) - _gl(_kp[:, None]) - _gl(y[None, :] + 1)
           + _kp[:, None] * np.log(_kp[:, None] / (_kp[:, None] + _rt)) + y[None, :] * np.log(_rt / (_kp[:, None] + _rt)))
    np.savez_compressed(OUTPUT_DIR / "ecdc_deeprv_v3_enhanced.npz",
        cases=y, state_index=np.asarray(obs["state_index"]), time_index=np.asarray(obs["time_index"]),
        country=np.asarray(obs["RegionCode"]).astype(str),
        eta_mean=np.log(rate_s).mean(0).astype(np.float32), eta_thin=_thinE(np.log(rate_s)).astype(np.float32),
        hp_names=np.array(_dn), hp_rhat=_drh, hp_ess=_des,
        loglik=_ll.astype(np.float32), pp_draws=pp[:200].astype(np.int32),
        max_rhat=float(np.nanmax(rhats)), min_ess=float(np.nanmin(esss)), div=int(n_div),
        runtime=float(elapsed), pp_cov95=float(coverage), pp_rmse=float(rmse), nchains=int(NUM_CHAINS), **_trace)
    print(f"[saved] {OUTPUT_DIR / 'ecdc_deeprv_v3_enhanced.npz'}")

    row = {
        "dataset": "ecdc_measles_2010_2019", "method": "deeprv_nuts_nb",
        "z_s": zsd, "z_t": ztd, "z_w": zwd, "total_latent": zsd + ztd + zwd,
        "S": data["S"], "T": data["T"], "n_observed": int(len(obs)),
        "max_rhat": float(np.nanmax(rhats)), "min_ess": float(np.nanmin(esss)),
        "n_divergences": n_div, "mean_accept_prob": float(np.asarray(extra["accept_prob"]).mean()),
        "runtime_seconds": elapsed, "ess_per_sec": float(np.nanmin(esss)) / elapsed,
        "pp_coverage_95": coverage, "pp_rmse_count": rmse, "pp_mae_count": mae,
        "kappa_post_mean": float(kappa_s.mean()), "beta0_post_mean": float(flat["beta0"].mean()),
        "state_index": obs["state_index"].tolist(), "time_index": obs["time_index"].tolist(),
        "country": obs["RegionCode"].tolist(), "cases": y.tolist(),
        "rate_post_mean": rate_s.mean(axis=0).tolist(),
        "latent_f_post_mean": flat["latent_f"].mean(axis=0).tolist(),
        "latent_f_post_sd": flat["latent_f"].std(axis=0).tolist(),
    }
    vtot = 0.0
    for comp in ["g_contrib", "q_contrib", "w_contrib"]:
        v = float(flat[comp].var(axis=1).mean()); row[f"var_{comp}"] = v; vtot += v
    for comp in ["g_contrib", "q_contrib", "w_contrib"]:
        row[f"frac_{comp}"] = row[f"var_{comp}"] / vtot
    print(f"  variance decomposition (g:q:w) = {row['frac_g_contrib']:.2f} : "
          f"{row['frac_q_contrib']:.2f} : {row['frac_w_contrib']:.2f}")

    z_h_s = flat["z_h"]; sig = flat["sigma_h"]; el = flat["ell_h"]
    m12 = np.arange(12); dm = np.abs(m12[:, None] - m12[None, :]); hs = []
    for i in range(len(z_h_s)):
        K = sig[i]**2 * np.exp(-2 * np.sin(np.pi * dm / 12)**2 / el[i]**2)
        h = np.linalg.cholesky(K + 1e-6 * np.eye(12)) @ z_h_s[i]; hs.append(h - h.mean())
    hs = np.stack(hs)
    row["h_monthly_post_mean"] = hs.mean(0).tolist()
    row["h_monthly_post_q025"] = np.quantile(hs, 0.025, 0).tolist()
    row["h_monthly_post_q975"] = np.quantile(hs, 0.975, 0).tolist()
    for hp in HP_NAMES:
        s = flat[hp]
        row[f"{hp}_post_mean"] = float(s.mean()); row[f"{hp}_post_sd"] = float(s.std())
        row[f"{hp}_post_q025"] = float(np.quantile(s, 0.025)); row[f"{hp}_post_q975"] = float(np.quantile(s, 0.975))
        row[f"{hp}_samples"] = s.tolist()

    (OUTPUT_DIR / "ecdc_deeprv_v3_joint9hp_nb.json").write_text(json.dumps(row, indent=2))
    print(f"\n  max_rhat={row['max_rhat']:.3f} min_ess={row['min_ess']:.0f} div={n_div} "
          f"t={elapsed:.0f}s accept={row['mean_accept_prob']:.3f}")
    print(f"  PP coverage_95={coverage:.3f} rmse={rmse:.1f} mae={mae:.1f} kappa={row['kappa_post_mean']:.3f}")
    for hp in HP_NAMES:
        print(f"    {hp:28s} {row[f'{hp}_post_mean']:8.3f} [{row[f'{hp}_post_q025']:7.3f}, {row[f'{hp}_post_q975']:7.3f}]")
    print(f"\n[saved] {OUTPUT_DIR / 'ecdc_deeprv_v3_joint9hp_nb.json'}")


if __name__ == "__main__":
    run()
