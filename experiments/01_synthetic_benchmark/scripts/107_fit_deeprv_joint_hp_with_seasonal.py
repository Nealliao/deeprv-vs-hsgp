"""
107_fit_deeprv_joint_hp_with_seasonal.py
================================================================================
NUTS inference with conditional DeepRV — ALL 9 hyperparameters sampled jointly.

Extends script 104 (which samples the 7 main-effect HPs via a conditional
decoder) by additionally sampling the 2 seasonal GP hyperparameters
(sigma_h, ell_h). The seasonal effect is an independent cyclic GP that does
NOT pass through the decoder, so it is sampled exactly as in script 106 (HSGP):
the 12x12 periodic-kernel Cholesky is recomputed inside the model.

Model:
  f_{s,t} = sigma_s * g_norm[s] + sigma_t * q_norm[t] + sigma_w * w_norm[s,t]
  g_norm = space_decoder(z_s, log_ell_s)
  q_norm = time_decoder(z_t,  log_ell_t)
  w_norm = inter_decoder(z_w, log_ell_ws, log_ell_wt)
  h_m    = b_h + L_h(sigma_h, ell_h) @ z_h   (centred)    <-- seasonal now sampled

  Y_{s,t} ~ Poisson(E_s * exp(beta0 + h_t + f_{s,t}))

Sampled HPs (9):
  main:     space_alpha, space_length, time_alpha, time_length,
            interaction_alpha, interaction_space_length, interaction_time_length
  seasonal: sigma_h ~ HalfNormal(0.45)
            ell_h   ~ InverseGamma(3.9335, 1.9878)   P[0.2<ell_h<2.5]=0.98 (periodic)

Counterpart of script 106 stage 3 (HSGP, all 9 HP joint). Same DGP, same priors.

Env vars:
  CDRV9_NUM_WARMUP, CDRV9_NUM_SAMPLES, CDRV9_NUM_CHAINS,
  CDRV9_N_SEEDS, CDRV9_SEED_BASE, CDRV9_SKIP_EXISTING, CDRV9_TARGET_ACCEPT
"""

from pathlib import Path
import os
import json
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.core import freeze
import numpyro
import numpyro.distributions as dist
from numpyro.diagnostics import summary as numpyro_summary
from numpyro.infer import MCMC, NUTS, init_to_value

DATA_DIR  = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = DATA_DIR / "official_deeprv_models"
OUTPUT_DIR = DATA_DIR / "formal_synthetic" / "conditional_deeprv_seasonal"

# ─── Config ───────────────────────────────────────────────────────────────────

NUM_WARMUP  = int(os.environ.get("CDRV9_NUM_WARMUP",  "1000"))
NUM_SAMPLES = int(os.environ.get("CDRV9_NUM_SAMPLES", "1000"))
NUM_CHAINS  = int(os.environ.get("CDRV9_NUM_CHAINS",  "4"))
N_SEEDS     = int(os.environ.get("CDRV9_N_SEEDS",     "1"))
SEED_BASE   = int(os.environ.get("CDRV9_SEED_BASE",   "1"))
SKIP_EXISTING = os.environ.get("CDRV9_SKIP_EXISTING", "1") == "1"

TARGET_ACCEPT  = float(os.environ.get("CDRV9_TARGET_ACCEPT", "0.95"))
MAX_TREE_DEPTH = 10

# Seasonal sinusoidal prior mean (fixed shape) + true HP (DGP)
SEASONAL_A       = 0.35
SEASONAL_M_STAR  = 2
TRUE_SIGMA_H     = 0.12
TRUE_ELL_H       = 0.75

_m12   = np.arange(12)
_diff  = np.abs(_m12[:, None] - _m12[None, :])
_K_h_fixed = TRUE_SIGMA_H**2 * np.exp(-2 * np.sin(np.pi * _diff / 12)**2 / TRUE_ELL_H**2)
_L_h_fixed = np.linalg.cholesky(_K_h_fixed + 1e-6 * np.eye(12))
_b_h   = SEASONAL_A * np.cos(2 * np.pi * (_m12 - SEASONAL_M_STAR) / 12)
_b_h  -= _b_h.mean()

TRUE_HP = {
    "space_alpha": 0.30,   "space_length": 0.55,
    "time_alpha":  0.20,   "time_length":  18.0,
    "interaction_alpha": 0.18,
    "interaction_space_length": 0.15,
    "interaction_time_length":  6.0,
    "sigma_h": TRUE_SIGMA_H,
    "ell_h":   TRUE_ELL_H,
}

# InvGamma containment priors (same as scripts 100/104/106)
PRIORS = {
    "space_alpha":               dist.HalfNormal(0.5),
    "space_length":              dist.InverseGamma(concentration=6.1091, rate=3.3175),
    "time_alpha":                dist.HalfNormal(0.5),
    "time_length":               dist.InverseGamma(concentration=9.3607, rate=179.029),
    "interaction_alpha":         dist.HalfNormal(0.5),
    "interaction_space_length":  dist.InverseGamma(concentration=5.3661, rate=0.3648),
    "interaction_time_length":   dist.InverseGamma(concentration=7.3012, rate=30.009),
    # seasonal (periodic) — same as script 106
    "sigma_h":                   dist.HalfNormal(0.45),
    "ell_h":                     dist.InverseGamma(concentration=3.9335, rate=1.9878),
}

_HP_INIT = {
    "space_alpha":               0.30,
    "space_length":              0.467,
    "time_alpha":                0.20,
    "time_length":               17.28,
    "interaction_alpha":         0.18,
    "interaction_space_length":  0.057,
    "interaction_time_length":   3.62,
    "sigma_h":                   0.20,
    "ell_h":                     0.40,
}

HP_NAMES = ["space_alpha", "space_length", "time_alpha", "time_length",
            "interaction_alpha", "interaction_space_length", "interaction_time_length",
            "sigma_h", "ell_h"]


# ─── Decoder architecture (must match training in script 103) ─────────────────

class ConditionalDecoder(nn.Module):
    hidden_dim: int
    output_dim: int

    @nn.compact
    def __call__(self, z, log_ell):
        x = jnp.concatenate([z, log_ell], axis=-1)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.tanh(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.tanh(x)
        x = nn.Dense(self.output_dim)(x)
        return x


def _unflatten(flat):
    root = {}
    for key, value in flat.items():
        parts = key.split("/")
        cursor = root
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = jnp.asarray(value)
    return freeze(root)


def load_decoder(path):
    bundle = np.load(path)
    flat = {k: bundle[k] for k in bundle.files if "/" in k}
    params = _unflatten(flat)
    hidden_dim = int(bundle["hidden_dim"])
    output_dim = int(bundle["output_dim"])
    z_dim      = int(bundle["z_dim"])
    n_ell      = int(bundle["n_ell"])
    model = ConditionalDecoder(hidden_dim=hidden_dim, output_dim=output_dim)
    return model, params, z_dim, n_ell


# ─── Seasonal periodic kernel (JAX, dynamic) ─────────────────────────────────

def _periodic_kernel_jax(sigma_h, ell_h):
    m12 = jnp.arange(12, dtype=jnp.float32)
    diff = jnp.abs(m12[:, None] - m12[None, :])
    return sigma_h**2 * jnp.exp(-2.0 * jnp.sin(jnp.pi * diff / 12.0)**2 / ell_h**2)


def generate_cyclic_seasonal(rng):
    delta = rng.multivariate_normal(np.zeros(12), _K_h_fixed + 1e-6 * np.eye(12))
    h_raw = _b_h + delta
    return h_raw - h_raw.mean()


# ─── DGP helpers (shared with scripts 100/104) ───────────────────────────────

def _se_cov(x, length_scale, variance):
    d = cdist(x, x)
    return variance * np.exp(-(d ** 2) / (2 * length_scale ** 2))


def _double_center(m):
    return (m
            - m.mean(axis=0, keepdims=True)
            - m.mean(axis=1, keepdims=True)
            + m.mean())


def generate_panel(regime, regions, n_months, dgp_seed):
    rng    = np.random.default_rng(dgp_seed)
    coords = regions[["x_coord", "y_coord"]].to_numpy()
    time   = np.arange(n_months)[:, None]

    cov_s = _se_cov(coords, regime["space_length_scale"], regime["space_sd"] ** 2)
    cov_t = _se_cov(time,   regime["time_length_scale"],  regime["time_sd"] ** 2)
    g  = rng.multivariate_normal(np.zeros(len(coords)), cov_s + 1e-6 * np.eye(len(coords)))
    g -= g.mean()
    q  = rng.multivariate_normal(np.zeros(n_months), cov_t + 1e-6 * np.eye(n_months))
    q -= q.mean()

    cov_ws = _se_cov(coords, regime["interaction_space_length_scale"], 1.0)
    cov_wt = _se_cov(time,   regime["interaction_time_length_scale"],  1.0)
    Ls = np.linalg.cholesky(cov_ws + 1e-6 * np.eye(len(coords)))
    Lt = np.linalg.cholesky(cov_wt + 1e-6 * np.eye(n_months))
    noise = rng.standard_normal((len(coords), n_months))
    w = Ls @ noise @ Lt.T
    w = _double_center(w)
    w = w / w.std() * regime["interaction_sd"]

    h_monthly = generate_cyclic_seasonal(rng)

    alpha  = np.log(regime["baseline_rate_per_100k"] / 100_000)
    months = pd.date_range("1961-01-01", periods=n_months, freq="MS")
    rows = []
    for _, sr in regions.iterrows():
        s = int(sr["state_index"])
        for t, month in enumerate(months):
            seasonal = float(h_monthly[t % 12])
            latent_f = g[s] + q[t] + w[s, t]
            log_lam  = alpha + seasonal + latent_f
            rows.append({
                "state": sr["state"], "state_index": s,
                "month": month, "time_index": t,
                "population": int(sr["population"]),
                "log_population": float(np.log(sr["population"])),
                "seasonal_h_true": seasonal,
                "latent_f_true": float(latent_f),
                "count": int(rng.poisson(sr["population"] * np.exp(log_lam))),
            })
    return pd.DataFrame(rows).sort_values(["state_index", "time_index"]).reset_index(drop=True)


def make_data(panel, regions):
    coords = regions[["x_coord", "y_coord"]].to_numpy()
    coords_c = coords - coords.mean(axis=0, keepdims=True)
    n_months = panel["time_index"].nunique()
    time_vals = np.arange(n_months, dtype=float)
    return {
        "count":           jnp.asarray(panel["count"].to_numpy(dtype=np.int32)),
        "log_population":  jnp.asarray(panel["log_population"].to_numpy(dtype=np.float32)),
        "b_h":             jnp.asarray(_b_h.astype(np.float32)),   # (12,) fixed prior mean
        "state_index":     jnp.asarray(panel["state_index"].to_numpy(dtype=np.int32)),
        "time_index":      jnp.asarray(panel["time_index"].to_numpy(dtype=np.int32)),
        "coords_c":        jnp.asarray(coords_c.astype(np.float32)),
        "time_vals":       jnp.asarray(time_vals.astype(np.float32)),
        "beta0_prior_mean": float(np.log(1.2e-5)),
    }


# ─── NumPyro model ────────────────────────────────────────────────────────────

def make_model(space_dec, space_params, z_s_dim,
               time_dec,  time_params,  z_t_dim,
               inter_dec, inter_params, z_w_dim):

    def model(count, log_population, b_h,
              state_index, time_index, coords_c, time_vals, beta0_prior_mean):

        beta0 = numpyro.sample("beta0", dist.Normal(beta0_prior_mean, 1.0))

        # ── Seasonal HPs (now sampled) + dynamic periodic Cholesky ─────────────
        sigma_h = numpyro.sample("sigma_h", PRIORS["sigma_h"])
        ell_h   = numpyro.sample("ell_h",   PRIORS["ell_h"])
        K_h = _periodic_kernel_jax(sigma_h, ell_h) + 1e-6 * jnp.eye(12)
        L_h = jnp.linalg.cholesky(K_h)
        z_h   = numpyro.sample("z_h", dist.Normal(0.0, 1.0).expand([12]))
        h_raw = b_h + L_h @ z_h
        h_monthly = h_raw - h_raw.mean()

        # ── Main-effect HPs (amplitude factoring: sigma outside decoder) ───────
        sigma_s  = numpyro.sample("space_alpha",              PRIORS["space_alpha"])
        ell_s    = numpyro.sample("space_length",             PRIORS["space_length"])
        sigma_t  = numpyro.sample("time_alpha",               PRIORS["time_alpha"])
        ell_t    = numpyro.sample("time_length",              PRIORS["time_length"])
        sigma_w  = numpyro.sample("interaction_alpha",        PRIORS["interaction_alpha"])
        ell_ws   = numpyro.sample("interaction_space_length", PRIORS["interaction_space_length"])
        ell_wt   = numpyro.sample("interaction_time_length",  PRIORS["interaction_time_length"])

        # ── Latent codes z ~ N(0, I) ───────────────────────────────────────────
        z_s = numpyro.sample("z_space", dist.Normal(0.0, 1.0).expand([z_s_dim]))
        z_t = numpyro.sample("z_time",  dist.Normal(0.0, 1.0).expand([z_t_dim]))
        z_w = numpyro.sample("z_inter", dist.Normal(0.0, 1.0).expand([z_w_dim]))

        # ── Decode: f_norm = decoder(z, log_ell), then scale by sigma ─────────
        log_ell_s  = jnp.array([[jnp.log(ell_s)]])
        log_ell_t  = jnp.array([[jnp.log(ell_t)]])
        log_ell_w  = jnp.array([[jnp.log(ell_ws), jnp.log(ell_wt)]])

        g_norm = space_dec.apply({"params": space_params}, z_s[None, :], log_ell_s)[0]
        q_norm = time_dec.apply({"params": time_params},  z_t[None, :], log_ell_t)[0]
        w_norm = inter_dec.apply({"params": inter_params}, z_w[None, :], log_ell_w)[0]

        g = sigma_s * g_norm; g = g - jnp.mean(g)
        q = sigma_t * q_norm; q = q - jnp.mean(q)
        w = sigma_w * w_norm; w = w - jnp.mean(w)

        S = coords_c.shape[0]
        T = time_vals.shape[0]
        w_mat = w.reshape(S, T)

        latent_f = g[state_index] + q[time_index] + w_mat[state_index, time_index]
        h_t = h_monthly[time_index % 12]
        eta = log_population + beta0 + h_t + latent_f
        eta = jnp.clip(eta, -35.0, 20.0)
        numpyro.deterministic("latent_f", latent_f)
        numpyro.deterministic("g_contrib", g[state_index])
        numpyro.deterministic("q_contrib", q[time_index])
        numpyro.deterministic("w_contrib", w_mat[state_index, time_index])
        numpyro.sample("obs", dist.Poisson(rate=jnp.exp(eta)), obs=count)

    return model


# ─── NUTS runner ──────────────────────────────────────────────────────────────

def run_nuts(model_fn, data, nuts_seed):
    numpyro.set_host_device_count(NUM_CHAINS)
    init_vals = {hp: _HP_INIT[hp] for hp in _HP_INIT}
    kernel = NUTS(
        model_fn,
        target_accept_prob=TARGET_ACCEPT,
        max_tree_depth=MAX_TREE_DEPTH,
        init_strategy=init_to_value(values=init_vals),
    )
    mcmc = MCMC(kernel, num_warmup=NUM_WARMUP, num_samples=NUM_SAMPLES,
                num_chains=NUM_CHAINS, chain_method="sequential", progress_bar=True)
    t0 = time.perf_counter()
    mcmc.run(jax.random.PRNGKey(nuts_seed), **data,
             extra_fields=("accept_prob", "num_steps", "diverging", "potential_energy"))
    jax.block_until_ready(mcmc.get_samples())
    elapsed = time.perf_counter() - t0
    return mcmc, elapsed


# ─── Metrics ─────────────────────────────────────────────────────────────────

def compute_metrics(mcmc, panel, elapsed, dgp_seed):
    samples_by_chain = mcmc.get_samples(group_by_chain=True)
    flat = {k: np.asarray(v).reshape((-1,) + tuple(np.asarray(v).shape[2:]))
            for k, v in samples_by_chain.items()}

    latent = flat["latent_f"]
    mean_f  = latent.mean(axis=0)
    ci_low  = np.quantile(latent, 0.025, axis=0)
    ci_high = np.quantile(latent, 0.975, axis=0)
    f_true  = panel["latent_f_true"].to_numpy()

    rmse     = float(np.sqrt(np.mean((mean_f - f_true) ** 2)))
    coverage = float(((f_true >= ci_low) & (f_true <= ci_high)).mean())

    diag = numpyro_summary({k: samples_by_chain[k] for k in samples_by_chain
                            if k not in ("latent_f", "g_contrib", "q_contrib", "w_contrib")})
    rhats, esss = [], []
    for k in diag:
        rhats.extend(np.asarray(diag[k]["r_hat"]).flatten().tolist())
        esss.extend(np.asarray(diag[k]["n_eff"]).flatten().tolist())
    max_rhat = float(np.nanmax(rhats))
    min_ess  = float(np.nanmin(esss))

    extra = mcmc.get_extra_fields()
    n_div        = int(np.asarray(extra["diverging"]).sum())
    mean_accept  = float(np.asarray(extra["accept_prob"]).mean())

    row = {
        "method": "conditional_deeprv_seasonal",
        "dgp_seed": dgp_seed,
        "n_free_params": len(HP_NAMES),
        "rmse_f": rmse,
        "coverage_95": coverage,
        "max_rhat": max_rhat,
        "min_ess": min_ess,
        "n_divergences": n_div,
        "mean_accept_prob": mean_accept,
        "runtime_seconds": elapsed,
        "ess_per_sec": min_ess / elapsed,
        "state_index":          panel["state_index"].tolist(),
        "time_index":           panel["time_index"].tolist(),
        "latent_f_true":        f_true.tolist(),
        "latent_f_post_mean":   mean_f.tolist(),
        "latent_f_post_sd":     np.std(latent, axis=0).tolist(),
        "latent_f_ci_low":      ci_low.tolist(),
        "latent_f_ci_high":     ci_high.tolist(),
    }
    # recovered variance decomposition (per-draw var of each component, avg over draws)
    vtot = 0.0
    for comp in ["g_contrib", "q_contrib", "w_contrib"]:
        v = float(flat[comp].var(axis=1).mean()); row[f"var_{comp}"] = v; vtot += v
    for comp in ["g_contrib", "q_contrib", "w_contrib"]:
        row[f"frac_{comp}"] = row[f"var_{comp}"] / vtot
    print(f"  var decomposition (g:q:w) recovered = "
          f"{row['frac_g_contrib']:.2f} : {row['frac_q_contrib']:.2f} : {row['frac_w_contrib']:.2f}")

    # Seasonal posterior (per-sample reconstruction with sampled sigma_h, ell_h)
    if "z_h" in flat:
        z_h_s = flat["z_h"]
        sig_s = flat.get("sigma_h", np.full(len(z_h_s), TRUE_SIGMA_H))
        ell_s = flat.get("ell_h",   np.full(len(z_h_s), TRUE_ELL_H))
        m12   = np.arange(12)
        diff  = np.abs(m12[:, None] - m12[None, :])
        h_list = []
        for i in range(len(z_h_s)):
            K = sig_s[i]**2 * np.exp(-2.0 * np.sin(np.pi * diff / 12.0)**2 / ell_s[i]**2)
            L = np.linalg.cholesky(K + 1e-6 * np.eye(12))
            h_raw = _b_h + L @ z_h_s[i]
            h_list.append(h_raw - h_raw.mean())
        h_monthly_s = np.stack(h_list)
        h_true = np.array([
            panel[panel["time_index"] == m]["seasonal_h_true"].iloc[0] for m in range(12)
        ])
        row["h_monthly_true"]      = h_true.tolist()
        row["h_monthly_post_mean"] = h_monthly_s.mean(axis=0).tolist()
        row["h_monthly_post_sd"]   = h_monthly_s.std(axis=0).tolist()
        row["h_monthly_ci_low"]    = np.quantile(h_monthly_s, 0.025, axis=0).tolist()
        row["h_monthly_ci_high"]   = np.quantile(h_monthly_s, 0.975, axis=0).tolist()

    # Per-HP posterior for all 9 hyperparameters
    for hp in HP_NAMES:
        if hp in flat:
            samps = flat[hp]
            row[f"{hp}_samples"]   = samps.tolist()
            row[f"{hp}_post_mean"] = float(samps.mean())
            row[f"{hp}_post_sd"]   = float(samps.std())
            row[f"{hp}_post_q025"] = float(np.quantile(samps, 0.025))
            row[f"{hp}_post_q975"] = float(np.quantile(samps, 0.975))
            row[f"{hp}_truth"]     = TRUE_HP[hp]
            row[f"{hp}_bias"]      = float(samps.mean()) - TRUE_HP[hp]

    return row


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[load] conditional DeepRV decoders …")
    space_dec, space_params, z_s_dim, _ = load_decoder(
        MODEL_DIR / "conditional_deeprv_space.npz")
    time_dec,  time_params,  z_t_dim, _ = load_decoder(
        MODEL_DIR / "conditional_deeprv_time.npz")
    inter_dec, inter_params, z_w_dim, _ = load_decoder(
        MODEL_DIR / "conditional_deeprv_inter.npz")

    total_latent = z_s_dim + z_t_dim + z_w_dim
    print(f"  z_s={z_s_dim}  z_t={z_t_dim}  z_w={z_w_dim}  total latent={total_latent}")
    print(f"  sampling all 9 HP (7 main + sigma_h, ell_h)")

    regimes = pd.read_csv(DATA_DIR / "synthetic_measles_interaction_regimes.csv")
    regime  = regimes.loc[regimes["name"] == "interaction_smooth"].iloc[0].to_dict()
    regions = pd.read_csv(DATA_DIR / "synthetic_measles_regions.csv") \
                .sort_values("state_index").reset_index(drop=True)

    print(f"\n[config] warmup={NUM_WARMUP}  samples={NUM_SAMPLES}  "
          f"chains={NUM_CHAINS}  seeds={N_SEEDS}")

    model_fn = make_model(space_dec, space_params, z_s_dim,
                          time_dec,  time_params,  z_t_dim,
                          inter_dec, inter_params, z_w_dim)

    all_rows = []
    for offset in range(N_SEEDS):
        dgp_seed = SEED_BASE + offset
        out_path = OUTPUT_DIR / f"cdrv9_seed{dgp_seed:04d}.json"

        if SKIP_EXISTING and out_path.exists():
            print(f"\n[seed {dgp_seed}] skipping (exists)")
            all_rows.append(json.loads(out_path.read_text()))
            continue

        print(f"\n{'='*60}")
        print(f"[seed {dgp_seed}] generating DGP …")
        panel = generate_panel(regime, regions, n_months=72, dgp_seed=dgp_seed)
        data  = make_data(panel, regions)

        nuts_seed = dgp_seed * 10_000 + 777
        print(f"[seed {dgp_seed}] running NUTS (nuts_seed={nuts_seed}) …")
        mcmc, elapsed = run_nuts(model_fn, data, nuts_seed)

        row = compute_metrics(mcmc, panel, elapsed, dgp_seed)
        all_rows.append(row)

        print(f"  RMSE={row['rmse_f']:.4f}  cover={row['coverage_95']:.4f}  "
              f"Rhat={row['max_rhat']:.4f}  ESS={row['min_ess']:.0f}  "
              f"div={row['n_divergences']}  t={elapsed:.1f}s  ESS/s={row['ess_per_sec']:.1f}")
        for hp in HP_NAMES:
            print(f"  {hp:28s} truth={TRUE_HP[hp]:7.3f}  "
                  f"post={row.get(f'{hp}_post_mean', float('nan')):7.3f}  "
                  f"bias={row.get(f'{hp}_bias', float('nan')):+7.3f}  "
                  f"95%CI=[{row.get(f'{hp}_post_q025', float('nan')):.3f}, "
                  f"{row.get(f'{hp}_post_q975', float('nan')):.3f}]")

        out_path.write_text(json.dumps(row, indent=2))

    df = pd.DataFrame(all_rows)
    summary_cols = ["method", "rmse_f", "coverage_95", "max_rhat",
                    "min_ess", "n_divergences", "runtime_seconds", "ess_per_sec"]
    print(f"\n{'='*60}\nSummary (mean across seeds)\n{'='*60}")
    print(df[summary_cols].mean(numeric_only=True).round(4).to_string())

    df.to_csv(OUTPUT_DIR / "cdrv9_all_results.csv", index=False)
    df[summary_cols].to_csv(OUTPUT_DIR / "cdrv9_summary.csv", index=False)
    print(f"\n[saved] {OUTPUT_DIR}/cdrv9_*.csv")


if __name__ == "__main__":
    main()
