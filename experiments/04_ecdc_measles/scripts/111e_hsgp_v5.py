"""
111_ecdc_hsgp_joint_hp_nb.py
================================================================================
HSGP joint 9-hyperparameter inference on REAL ECDC measles data (2010-2019),
Negative-Binomial likelihood. Counterpart of script 106 (synthetic, Poisson).

Model:
  Y_{c,t} ~ NegBin2( mean = E_{c,year(t)} * exp(beta0 + h_t + f_{c,t}), kappa )
  f_{c,t} = g_c + q_t + w_{c,t}          (HSGP: space + time + interaction)
  h_t     = seasonal cyclic GP, DATA-DRIVEN (no fixed sinusoid mean)

Sampled jointly (9 HP + 2 extra):
  space_alpha, space_length, time_alpha, time_length,
  interaction_alpha, interaction_space_length, interaction_time_length,
  sigma_h, ell_h,                       (the 9 HP)
  beta0, kappa (NB concentration)       (extra)

Real-data modelling choices (vs synthetic):
  - NB instead of Poisson (overdispersion var/mean ~1134)
  - exposure = Eurostat yearly population
  - seasonal has NO fixed sinusoid mean: periodic GP learns the shape from data
  - amplitudes ~ HalfNormal(1.0): wider, real measles log-risk spans countries widely
  - time lengthscale priors recomputed for 120 months (Betancourt containment)
  - NO ground truth -> evaluate by in-sample posterior-predictive checks
    (predictive coverage, RMSE/MAE on counts) + HP posteriors + diagnostics
  - likelihood only on OBSERVED cells (0.9% missing masked out)

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
import numpyro
import numpyro.distributions as dist
from numpyro.contrib.hsgp.laplacian import eigenfunctions
from numpyro.contrib.hsgp.spectral_densities import (
    diag_spectral_density_squared_exponential,
)
from numpyro.diagnostics import summary as numpyro_summary
from numpyro.infer import MCMC, NUTS, init_to_value

DATA_DIR  = PROJECT_ROOT / "data" / "processed"
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

# Priors. Lengthscales: Betancourt InverseGamma containment.
#   space reused from synthetic (geometry comparable: span ~1.3, min-NN ~0.08).
#   time recomputed for 120 months: P[12<ell<80] and P[2<ell<14].
#   amplitudes widened to HalfNormal(1.0) for real-data dynamic range.
PRIORS = {
    "space_alpha":               dist.HalfNormal(1.0),
    "space_length":              dist.LogUniform(0.030, 1.331),    # P[0.25<l<1.80]
    "time_alpha":                dist.HalfNormal(1.0),
    "time_length":               dist.InverseGamma(6.5707, 167.3694),  # P[12<l<80] (120mo)
    "interaction_alpha":         dist.HalfNormal(1.0),
    "interaction_space_length":  dist.InverseGamma(5.3661, 0.3648),    # P[0.03<l<0.25]
    "interaction_time_length":   dist.InverseGamma(6.2718, 27.0192),   # P[2<l<14] (120mo)
    "sigma_h":                   dist.HalfNormal(0.45),                # seasonal amp (Tycho-cal)
    "ell_h":                     dist.InverseGamma(3.9335, 1.9878),    # P[0.2<l<2.5] periodic
}
_HP_INIT = {
    "space_alpha": 0.5, "space_length": 0.30,
    "time_alpha": 0.5, "time_length": 22.0,
    "interaction_alpha": 0.3, "interaction_space_length": 0.057, "interaction_time_length": 3.7,
    "sigma_h": 0.30, "ell_h": 0.40,
}

# HSGP basis dims (space reused; time scaled up for 120 months)
SPACE_ELL, SPACE_M = [1.4, 1.4], [8, 8]
TIME_ELL,  TIME_M  = 75.0, 35
INTER_SPACE_ELL, INTER_SPACE_M = [1.0, 1.0], [8, 8]
INTER_TIME_ELL,  INTER_TIME_M  = 75.0, 15


# ─── HSGP helpers (from script 106) ───────────────────────────────────────────

def _safe_sqrt(x, eps=1e-20):
    safe_x = jnp.where(x > eps, x, jnp.ones_like(x))
    return jnp.where(x > eps, jnp.sqrt(safe_x), jnp.zeros_like(x))


def hsgp_se(name, x, alpha, length, ell, m):
    dim = jnp.shape(x)[-1] if jnp.ndim(x) > 1 else 1
    phi = eigenfunctions(x=x, ell=ell, m=m)
    spd = _safe_sqrt(diag_spectral_density_squared_exponential(
        alpha=alpha**2, length=length, ell=ell, m=m, dim=dim))
    beta = numpyro.sample(f"{name}_basis_weights", dist.Normal(0.0, 1.0).expand([phi.shape[-1]]))
    return phi @ (spd * beta)


def separable_interaction_hsgp(name, state_index, time_index, coords_c, time_c,
                                alpha, space_length, time_length):
    phi_s = eigenfunctions(x=coords_c, ell=INTER_SPACE_ELL, m=INTER_SPACE_M)
    spd_s = _safe_sqrt(diag_spectral_density_squared_exponential(
        alpha=1.0, length=space_length, ell=INTER_SPACE_ELL, m=INTER_SPACE_M, dim=2))
    phi_t = eigenfunctions(x=time_c, ell=INTER_TIME_ELL, m=INTER_TIME_M)
    spd_t = _safe_sqrt(diag_spectral_density_squared_exponential(
        alpha=1.0, length=time_length, ell=INTER_TIME_ELL, m=INTER_TIME_M, dim=1))
    ws = phi_s[state_index] * spd_s
    wt = phi_t[time_index] * spd_t
    n_basis = ws.shape[-1] * wt.shape[-1]
    beta = numpyro.sample(f"{name}_basis_weights",
                          dist.Normal(0.0, 1.0).expand([n_basis])).reshape((ws.shape[-1], wt.shape[-1]))
    return alpha * jnp.einsum("ns,nt,st->n", ws, wt, beta)


def _periodic_kernel_jax(sigma_h, ell_h):
    m12 = jnp.arange(12, dtype=jnp.float32)
    diff = jnp.abs(m12[:, None] - m12[None, :])
    return sigma_h**2 * jnp.exp(-2.0 * jnp.sin(jnp.pi * diff / 12.0)**2 / ell_h**2)


# ─── Data ─────────────────────────────────────────────────────────────────────

def load_data():
    p = pd.read_csv(PANEL)
    obs = p[p["observed"] == 1].copy()                 # mask missing cells
    coords = p.drop_duplicates("state_index").sort_values("state_index")[["x_coord", "y_coord"]].to_numpy()
    n_months = p["time_index"].nunique()
    time_vals = np.arange(n_months, dtype=float)[:, None]
    time_c = time_vals - time_vals.mean()
    beta0_guess = float(np.log(obs["cases"].mean() + 1) - obs["log_population"].mean())
    data = {
        "count":          jnp.asarray(obs["cases"].to_numpy(dtype=np.int32)),
        "log_population": jnp.asarray(obs["log_population"].to_numpy(dtype=np.float32)),
        "state_index":    jnp.asarray(obs["state_index"].to_numpy(dtype=np.int32)),
        "time_index":     jnp.asarray(obs["time_index"].to_numpy(dtype=np.int32)),
        "month_index":    jnp.asarray((obs["month_of_year"].to_numpy() - 1).astype(np.int32)),
        "coords_c":       jnp.asarray(coords.astype(np.float32)),
        "time_c":         jnp.asarray(time_c.astype(np.float32)),
        "beta0_prior_mean": beta0_guess,
    }
    return data, obs, p, coords


# ─── Model ────────────────────────────────────────────────────────────────────

def model(count, log_population, state_index, time_index, month_index,
          coords_c, time_c, beta0_prior_mean):
    beta0 = numpyro.sample("beta0", dist.Normal(beta0_prior_mean, 3.0))
    # NB overdispersion: reciprocal parameterization (Stan Prior Choice
    # Recommendations; Simpson et al. 2017, Stat. Sci. 32(1), PC priors).
    # 1/sqrt(kappa) ~ HalfNormal(1): Poisson (kappa->inf) is the contained
    # default at 0, data must justify overdispersion. kappa scale is hard to
    # interpret directly, the reciprocal is not.
    inv_sqrt_kappa = numpyro.sample("inv_sqrt_kappa", dist.HalfNormal(1.0))
    kappa = numpyro.deterministic("kappa", inv_sqrt_kappa ** (-2))

    # seasonal: data-driven periodic GP (no fixed sinusoid mean)
    sigma_h = numpyro.sample("sigma_h", PRIORS["sigma_h"])
    ell_h   = numpyro.sample("ell_h",   PRIORS["ell_h"])
    L_h = jnp.linalg.cholesky(_periodic_kernel_jax(sigma_h, ell_h) + 1e-6 * jnp.eye(12))
    z_h = numpyro.sample("z_h", dist.Normal(0.0, 1.0).expand([12]))
    h_raw = L_h @ z_h
    h_monthly = h_raw - h_raw.mean()

    sa = numpyro.sample("space_alpha", PRIORS["space_alpha"])
    sl = numpyro.sample("space_length", PRIORS["space_length"])
    ta = numpyro.sample("time_alpha", PRIORS["time_alpha"])
    tl = numpyro.sample("time_length", PRIORS["time_length"])
    ia = numpyro.sample("interaction_alpha", PRIORS["interaction_alpha"])
    isl = numpyro.sample("interaction_space_length", PRIORS["interaction_space_length"])
    itl = numpyro.sample("interaction_time_length", PRIORS["interaction_time_length"])

    g = hsgp_se("space", coords_c, alpha=sa, length=sl, ell=SPACE_ELL, m=SPACE_M)
    q = hsgp_se("time", time_c, alpha=ta, length=tl, ell=TIME_ELL, m=TIME_M)
    w = separable_interaction_hsgp("interaction", state_index, time_index,
                                   coords_c, time_c, alpha=ia, space_length=isl, time_length=itl)
    g = g - jnp.mean(g); q = q - jnp.mean(q); w = w - jnp.mean(w)

    latent_f = g[state_index] + q[time_index] + w
    h_t = h_monthly[month_index]
    eta = log_population + beta0 + h_t + latent_f
    eta = jnp.clip(eta, -30.0, 20.0)
    rate = jnp.exp(eta)
    numpyro.deterministic("latent_f", latent_f)
    numpyro.deterministic("rate", rate)
    # per-cell contribution of each component (for comparable variance decomposition)
    numpyro.deterministic("g_contrib", g[state_index])
    numpyro.deterministic("q_contrib", q[time_index])
    numpyro.deterministic("w_contrib", w)
    numpyro.sample("obs", dist.NegativeBinomial2(mean=rate, concentration=kappa), obs=count)


# ─── Run + metrics ────────────────────────────────────────────────────────────

def run():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    numpyro.set_host_device_count(NUM_CHAINS)
    data, obs, panel, coords = load_data()
    print(f"[data] S={panel['state_index'].nunique()} T={panel['time_index'].nunique()} "
          f"observed={len(obs)} beta0_guess={data['beta0_prior_mean']:.2f}")

    init = init_to_value(values=_HP_INIT)
    kernel = NUTS(model, target_accept_prob=TARGET_ACCEPT, max_tree_depth=MAX_TREE_DEPTH, init_strategy=init)
    mcmc = MCMC(kernel, num_warmup=NUM_WARMUP, num_samples=NUM_SAMPLES,
                num_chains=NUM_CHAINS, chain_method="sequential", progress_bar=True)
    t0 = time.perf_counter()
    mcmc.run(jax.random.PRNGKey(SEED), **data,
             extra_fields=("accept_prob", "diverging", "num_steps"))
    jax.block_until_ready(mcmc.get_samples())
    elapsed = time.perf_counter() - t0

    sbc = mcmc.get_samples(group_by_chain=True)
    flat = {k: np.asarray(v).reshape((-1,) + tuple(np.asarray(v).shape[2:])) for k, v in sbc.items()}
    _skip = ("latent_f", "rate", "kappa", "g_contrib", "q_contrib", "w_contrib")
    diag = numpyro_summary({k: sbc[k] for k in sbc if k not in _skip})
    rhats, esss = [], []
    for k in diag:
        rhats += np.asarray(diag[k]["r_hat"]).ravel().tolist()
        esss += np.asarray(diag[k]["n_eff"]).ravel().tolist()
    extra = mcmc.get_extra_fields()
    n_div = int(np.asarray(extra["diverging"]).sum())

    # posterior predictive checks (no truth): NB predictive on observed cells
    rate_s = flat["rate"]; kappa_s = flat["kappa"]
    rng = np.random.default_rng(0)
    idx = rng.choice(len(rate_s), size=min(400, len(rate_s)), replace=False)
    pp = []
    for i in idx:
        r = rate_s[i]; kp = kappa_s[i]
        p_nb = kp / (kp + r)                       # numpy NB: n=kappa, p
        pp.append(rng.negative_binomial(kp, p_nb))
    pp = np.array(pp)                              # (n_draw, n_obs)
    y = obs["cases"].to_numpy()
    lo = np.quantile(pp, 0.025, axis=0); hi = np.quantile(pp, 0.975, axis=0)
    pred_mean = pp.mean(axis=0)
    coverage = float(((y >= lo) & (y <= hi)).mean())
    rmse = float(np.sqrt(np.mean((pred_mean - y) ** 2)))
    mae = float(np.mean(np.abs(pred_mean - y)))
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
    np.savez_compressed(OUTPUT_DIR / "ecdc_hsgp_v5_enhanced.npz",
        cases=y, state_index=np.asarray(obs["state_index"]), time_index=np.asarray(obs["time_index"]),
        country=np.asarray(obs["RegionCode"]).astype(str),
        eta_mean=np.log(rate_s).mean(0).astype(np.float32), eta_thin=_thinE(np.log(rate_s)).astype(np.float32),
        hp_names=np.array(_dn), hp_rhat=_drh, hp_ess=_des,
        loglik=_ll.astype(np.float32), pp_draws=pp[:200].astype(np.int32),
        max_rhat=float(np.nanmax(rhats)), min_ess=float(np.nanmin(esss)), div=int(n_div),
        runtime=float(elapsed), pp_cov95=float(coverage), pp_rmse=float(rmse), nchains=int(NUM_CHAINS), **_trace)
    print(f"[saved] {OUTPUT_DIR / 'ecdc_hsgp_v5_enhanced.npz'}")

    row = {
        "dataset": "ecdc_measles_2010_2019", "method": "hsgp_nuts_nb", "likelihood": "negbin",
        "S": int(panel["state_index"].nunique()), "T": int(panel["time_index"].nunique()),
        "n_observed": int(len(obs)),
        "max_rhat": float(np.nanmax(rhats)), "min_ess": float(np.nanmin(esss)),
        "n_divergences": n_div, "mean_accept_prob": float(np.asarray(extra["accept_prob"]).mean()),
        "runtime_seconds": elapsed, "ess_per_sec": float(np.nanmin(esss)) / elapsed,
        "pp_coverage_95": coverage, "pp_rmse_count": rmse, "pp_mae_count": mae,
        "kappa_post_mean": float(kappa_s.mean()), "beta0_post_mean": float(flat["beta0"].mean()),
        # arrays for plotting
        "state_index": obs["state_index"].tolist(), "time_index": obs["time_index"].tolist(),
        "country": obs["RegionCode"].tolist(),
        "cases": y.tolist(),
        "rate_post_mean": rate_s.mean(axis=0).tolist(),
        "latent_f_post_mean": flat["latent_f"].mean(axis=0).tolist(),
        "latent_f_post_sd": flat["latent_f"].std(axis=0).tolist(),
    }
    # variance decomposition: per-draw var of each component over cells, avg over draws
    vtot = 0.0
    for comp in ["g_contrib", "q_contrib", "w_contrib"]:
        v = float(flat[comp].var(axis=1).mean()); row[f"var_{comp}"] = v; vtot += v
    for comp in ["g_contrib", "q_contrib", "w_contrib"]:
        row[f"frac_{comp}"] = row[f"var_{comp}"] / vtot
    print(f"  variance decomposition (g:q:w) = {row['frac_g_contrib']:.2f} : "
          f"{row['frac_q_contrib']:.2f} : {row['frac_w_contrib']:.2f}")

    # seasonal posterior (per-sample reconstruction)
    z_h_s = flat["z_h"]; sig = flat["sigma_h"]; el = flat["ell_h"]
    m12 = np.arange(12); dm = np.abs(m12[:, None] - m12[None, :])
    hs = []
    for i in range(len(z_h_s)):
        K = sig[i]**2 * np.exp(-2 * np.sin(np.pi * dm / 12)**2 / el[i]**2)
        L = np.linalg.cholesky(K + 1e-6 * np.eye(12))
        h = L @ z_h_s[i]; hs.append(h - h.mean())
    hs = np.stack(hs)
    row["h_monthly_post_mean"] = hs.mean(axis=0).tolist()
    row["h_monthly_post_q025"] = np.quantile(hs, 0.025, axis=0).tolist()
    row["h_monthly_post_q975"] = np.quantile(hs, 0.975, axis=0).tolist()
    # HP posteriors
    for hp in HP_NAMES:
        s = flat[hp]
        row[f"{hp}_post_mean"] = float(s.mean()); row[f"{hp}_post_sd"] = float(s.std())
        row[f"{hp}_post_q025"] = float(np.quantile(s, 0.025))
        row[f"{hp}_post_q975"] = float(np.quantile(s, 0.975))
        row[f"{hp}_samples"] = s.tolist()

    (OUTPUT_DIR / "ecdc_hsgp_v5_joint9hp_nb.json").write_text(json.dumps(row, indent=2))
    print(f"\n  max_rhat={row['max_rhat']:.3f}  min_ess={row['min_ess']:.0f}  div={n_div}  "
          f"t={elapsed:.0f}s  accept={row['mean_accept_prob']:.3f}")
    print(f"  PP coverage_95={coverage:.3f}  rmse={rmse:.1f}  mae={mae:.1f}  "
          f"kappa={row['kappa_post_mean']:.3f}")
    print("  HP posteriors:")
    for hp in HP_NAMES:
        print(f"    {hp:28s} {row[f'{hp}_post_mean']:8.3f} "
              f"[{row[f'{hp}_post_q025']:7.3f}, {row[f'{hp}_post_q975']:7.3f}]")
    print(f"\n[saved] {OUTPUT_DIR / 'ecdc_hsgp_v5_joint9hp_nb.json'}")


if __name__ == "__main__":
    run()
