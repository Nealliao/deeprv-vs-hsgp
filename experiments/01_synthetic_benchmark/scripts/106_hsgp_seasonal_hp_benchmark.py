"""
106_hsgp_seasonal_hp_benchmark.py
================================================================================
Joint hyperparameter inference benchmark including the seasonal GP.

Extends script 100 by adding the two seasonal GP hyperparameters (sigma_h,
ell_h) to the set of sampleable hyperparameters, so all 9 hyperparameters
(7 main-effect + 2 seasonal) can be inferred jointly.

  Stage 0  fixed_all      all 9 HPs fixed at truth (baseline)
  Stage 1  free_seasonal  only sigma_h, ell_h sampled; 7 main effects fixed
  Stage 2  free_main      only 7 main effects sampled; seasonal fixed (= script 100 stage3)
  Stage 3  free_all       all 9 HPs sampled jointly  <-- full joint

Priors (Betancourt 2020 containment principle):
  Main-effect lengthscales:  InverseGamma containment, P(l < rho < u) = 0.98
  Main-effect amplitudes:    HalfNormal(0.5)
  sigma_h ~ HalfNormal(0.45)                  Tycho calibration (script 101)
  ell_h   ~ InverseGamma(3.9335, 1.9878)      P(0.2 < ell_h < 2.5) = 0.98 (periodic)

True DGP values:
  space_alpha=0.30  space_length=0.55  time_alpha=0.20  time_length=18.0
  interaction_alpha=0.18  interaction_space_length=0.15  interaction_time_length=6.0
  sigma_h=0.12  ell_h=0.75

Env vars:
  SHP_N_SEEDS, SHP_SEED_BASE, SHP_NUM_WARMUP, SHP_NUM_SAMPLES, SHP_NUM_CHAINS,
  SHP_STAGES (default "3" = full joint), SHP_SKIP_EXISTING
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
import numpyro
import numpyro.distributions as dist
from numpyro.contrib.hsgp.laplacian import eigenfunctions
from numpyro.contrib.hsgp.spectral_densities import (
    diag_spectral_density_squared_exponential,
)
from numpyro.diagnostics import summary as numpyro_summary
from numpyro.infer import MCMC, NUTS, init_to_value


DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = DATA_DIR / "formal_synthetic" / "seasonal_hp_benchmark"

# ─── Config ───────────────────────────────────────────────────────────────────

N_SEEDS = int(os.environ.get("SHP_N_SEEDS", "1"))
SEED_BASE = int(os.environ.get("SHP_SEED_BASE", "1"))
NUM_WARMUP = int(os.environ.get("SHP_NUM_WARMUP", "1000"))
NUM_SAMPLES = int(os.environ.get("SHP_NUM_SAMPLES", "1000"))
NUM_CHAINS = int(os.environ.get("SHP_NUM_CHAINS", "4"))
SKIP_EXISTING = os.environ.get("SHP_SKIP_EXISTING", "1") == "1"

_stage_env = os.environ.get("SHP_STAGES", "3")   # default: full joint
STAGES = [int(s) for s in _stage_env.split(",") if s.strip()]

TARGET_ACCEPT = 0.90
MAX_TREE_DEPTH = 10

# Seasonal sinusoidal prior mean (fixed shape)
TRUE_SIGMA_H = 0.12
TRUE_ELL_H   = 0.75
SEASONAL_A      = 0.35   # sinusoidal prior-mean amplitude
SEASONAL_M_STAR = 2      # peak month (0=Jan → 2=Mar)

# Precompute fixed Cholesky for DGP generation and for fixed-seasonal stages
_m12  = np.arange(12)
_diff = np.abs(_m12[:, None] - _m12[None, :])
_K_h_fixed = (
    TRUE_SIGMA_H**2
    * np.exp(-2.0 * np.sin(np.pi * _diff / 12.0) ** 2 / TRUE_ELL_H**2)
)
_L_h_fixed = np.linalg.cholesky(_K_h_fixed + 1e-6 * np.eye(12))
_b_h       = SEASONAL_A * np.cos(2.0 * np.pi * (_m12 - SEASONAL_M_STAR) / 12.0)
_b_h      -= _b_h.mean()

# True values for all 9 hyperparameters
TRUE_HP = {
    "space_alpha":               0.30,
    "space_length":              0.55,
    "time_alpha":                0.20,
    "time_length":               18.0,
    "interaction_alpha":         0.18,
    "interaction_space_length":  0.15,
    "interaction_time_length":   6.0,
    "sigma_h":                   TRUE_SIGMA_H,
    "ell_h":                     TRUE_ELL_H,
}

MAIN_HP   = ["space_alpha", "space_length", "time_alpha", "time_length",
             "interaction_alpha", "interaction_space_length", "interaction_time_length"]
SEASONAL_HP = ["sigma_h", "ell_h"]

# Priors following Betancourt (2020) "Robust Gaussian Process Modeling":
#   Lengthscales: InverseGamma containment, P(l < rho < u) = 0.98.
#   Amplitudes:   HalfNormal.
#     space_length  (g_s global spatial):  P[0.25 < rho < 1.80] = 0.98
#     inter_s_len   (w_{st} local spatial): P[0.03 < rho < 0.25] = 0.98
#     time_length   (q_t global temporal): P[10   < rho < 48]    = 0.98
#     inter_t_len   (w_{st} local temporal):P[2   < rho < 12]    = 0.98
#   Additive separation (Section 4.1) verified: space↔inter_space overlap 1.9%,
#     time↔inter_time overlap 3.5%.
#   Seasonal (periodic kernel):
#     sigma_h ~ HalfNormal(0.45)             Tycho pre-vaccine amplitude (script 101)
#     ell_h   ~ InverseGamma(3.9335,1.9878)  P[0.2 < ell_h < 2.5]=0.98; lower=near
#       white noise, upper=near flat (confounds with time main effect q_t).
PRIORS = {
    "space_alpha":               dist.HalfNormal(0.5),
    "space_length":              dist.InverseGamma(concentration=6.1091, rate=3.3175),
    "time_alpha":                dist.HalfNormal(0.5),
    "time_length":               dist.InverseGamma(concentration=9.3607, rate=179.029),
    "interaction_alpha":         dist.HalfNormal(0.5),
    "interaction_space_length":  dist.InverseGamma(concentration=5.3661, rate=0.3648),
    "interaction_time_length":   dist.InverseGamma(concentration=7.3012, rate=30.009),
    "sigma_h":                   dist.HalfNormal(0.45),
    "ell_h":                     dist.InverseGamma(concentration=3.9335, rate=1.9878),
}

STAGE_FREE = {
    0: [],                              # all 9 fixed
    1: SEASONAL_HP,                     # seasonal only
    2: MAIN_HP,                         # main only (= script 100 stage3)
    3: MAIN_HP + SEASONAL_HP,           # full joint (all 9)
}
STAGE_NAMES = {
    0: "fixed_all",
    1: "free_seasonal",
    2: "free_main",
    3: "free_all",
}

# Init values near prior modes / truth (Betancourt: avoid InvGamma tails)
_HP_INIT = {
    "space_alpha":               0.30,
    "space_length":              0.467,   # InvGamma(6.1091,3.3175) mode
    "time_alpha":                0.20,
    "time_length":               17.28,   # InvGamma(9.3607,179.029) mode
    "interaction_alpha":         0.18,
    "interaction_space_length":  0.057,   # InvGamma(5.3661,0.3648) mode
    "interaction_time_length":   3.62,    # InvGamma(7.3012,30.009) mode
    "sigma_h":                   0.20,
    "ell_h":                     0.40,     # InvGamma(3.9335,1.9878) mode
}


# ─── DGP helpers (identical to script 100) ────────────────────────────────────

def se_cov(x, length_scale, variance):
    d = cdist(x, x)
    return variance * np.exp(-(d**2) / (2.0 * length_scale**2))


def centered_gp_draw(rng, cov):
    draw = rng.multivariate_normal(np.zeros(cov.shape[0]), cov + 1e-6 * np.eye(cov.shape[0]))
    return draw - draw.mean()


def double_center(matrix):
    return (
        matrix
        - matrix.mean(axis=0, keepdims=True)
        - matrix.mean(axis=1, keepdims=True)
        + matrix.mean()
    )


def sample_interaction(rng, coords, time, regime):
    cov_s = se_cov(coords, regime["interaction_space_length_scale"], 1.0)
    cov_t = se_cov(time, regime["interaction_time_length_scale"], 1.0)
    Ls = np.linalg.cholesky(cov_s + 1e-6 * np.eye(cov_s.shape[0]))
    Lt = np.linalg.cholesky(cov_t + 1e-6 * np.eye(cov_t.shape[0]))
    noise = rng.normal(size=(coords.shape[0], time.shape[0]))
    interaction = Ls @ noise @ Lt.T
    interaction = double_center(interaction)
    interaction = interaction / interaction.std() * regime["interaction_sd"]
    return interaction


def generate_cyclic_seasonal(rng):
    delta = rng.multivariate_normal(np.zeros(12), _K_h_fixed + 1e-6 * np.eye(12))
    h_raw = _b_h + delta
    return h_raw - h_raw.mean()


def generate_panel(regime, regions, n_months, dgp_seed):
    rng = np.random.default_rng(dgp_seed)
    coords = regions[["x_coord", "y_coord"]].to_numpy()
    time = np.arange(n_months)[:, None]

    cov_s = se_cov(coords, regime["space_length_scale"], regime["space_sd"] ** 2)
    cov_t = se_cov(time, regime["time_length_scale"], regime["time_sd"] ** 2)
    spatial_eff = centered_gp_draw(rng, cov_s)
    temporal_eff = centered_gp_draw(rng, cov_t)
    interaction_eff = sample_interaction(rng, coords, time, regime)
    h_monthly = generate_cyclic_seasonal(rng)

    months = pd.date_range("1961-01-01", periods=n_months, freq="MS")
    alpha = np.log(regime["baseline_rate_per_100k"] / 100_000)
    rows = []
    for _, state_row in regions.iterrows():
        s = int(state_row["state_index"])
        for t, month in enumerate(months):
            seasonal_eff = float(h_monthly[t % 12])
            latent_f = spatial_eff[s] + temporal_eff[t] + interaction_eff[s, t]
            log_lambda = alpha + seasonal_eff + latent_f
            expected = state_row["population"] * np.exp(log_lambda)
            count = rng.poisson(expected)
            rows.append({
                "state_index": s,
                "time_index": t,
                "log_population": float(np.log(state_row["population"])),
                "seasonal_h_true": seasonal_eff,
                "latent_f_true": float(latent_f),
                "count": int(count),
            })
    return pd.DataFrame(rows).sort_values(["state_index", "time_index"]).reset_index(drop=True)


# ─── HSGP helpers ─────────────────────────────────────────────────────────────

def _safe_sqrt(x, eps=1e-20):
    safe_x = jnp.where(x > eps, x, jnp.ones_like(x))
    return jnp.where(x > eps, jnp.sqrt(safe_x), jnp.zeros_like(x))


def hsgp_se(name, x, alpha, length, ell, m):
    dim = jnp.shape(x)[-1] if jnp.ndim(x) > 1 else 1
    phi = eigenfunctions(x=x, ell=ell, m=m)
    spd = _safe_sqrt(
        diag_spectral_density_squared_exponential(
            alpha=alpha**2, length=length, ell=ell, m=m, dim=dim,
        )
    )
    beta = numpyro.sample(f"{name}_basis_weights", dist.Normal(0.0, 1.0).expand([phi.shape[-1]]))
    return phi @ (spd * beta)


def separable_interaction_hsgp(name, state_index, time_index,
                                coords_centered, time_centered,
                                alpha, space_length, time_length):
    phi_space = eigenfunctions(x=coords_centered, ell=[0.8, 0.8], m=[8, 8])
    spd_space = _safe_sqrt(
        diag_spectral_density_squared_exponential(
            alpha=1.0, length=space_length, ell=[0.8, 0.8], m=[8, 8], dim=2,
        )
    )
    phi_time = eigenfunctions(x=time_centered, ell=45.0, m=12)
    spd_time = _safe_sqrt(
        diag_spectral_density_squared_exponential(
            alpha=1.0, length=time_length, ell=45.0, m=12, dim=1,
        )
    )
    weighted_space = phi_space[state_index] * spd_space
    weighted_time  = phi_time[time_index]  * spd_time
    basis_count = weighted_space.shape[-1] * weighted_time.shape[-1]
    beta = numpyro.sample(
        f"{name}_basis_weights",
        dist.Normal(0.0, 1.0).expand([basis_count]),
    ).reshape((weighted_space.shape[-1], weighted_time.shape[-1]))
    return alpha * jnp.einsum("ns,nt,st->n", weighted_space, weighted_time, beta)


def _periodic_kernel_jax(sigma_h, ell_h):
    m12 = jnp.arange(12, dtype=jnp.float32)
    diff = jnp.abs(m12[:, None] - m12[None, :])
    return sigma_h**2 * jnp.exp(-2.0 * jnp.sin(jnp.pi * diff / 12.0)**2 / ell_h**2)


# ─── Model factory ────────────────────────────────────────────────────────────

def make_hsgp_model(free_params: list[str]):
    """free_params: subset of the 9 HP names to sample. Rest fixed at truth."""
    free_set = set(free_params)
    seasonal_free = bool(free_set & set(SEASONAL_HP))

    def model(
        count, log_population,
        b_h, L_h_fixed,
        state_index, time_index,
        coords_centered, time_centered,
        beta0_prior_mean,
        # all 9 HP fixed values (used when the HP is not in free_set)
        space_alpha_fixed, space_length_fixed,
        time_alpha_fixed, time_length_fixed,
        interaction_alpha_fixed,
        interaction_space_length_fixed, interaction_time_length_fixed,
        sigma_h_fixed, ell_h_fixed,
    ):
        beta0 = numpyro.sample("beta0", dist.Normal(beta0_prior_mean, 1.0))

        def hp(name, fixed_val):
            if name in free_set:
                return numpyro.sample(name, PRIORS[name])
            return fixed_val

        space_alpha  = hp("space_alpha",              space_alpha_fixed)
        space_length = hp("space_length",             space_length_fixed)
        time_alpha   = hp("time_alpha",               time_alpha_fixed)
        time_length  = hp("time_length",              time_length_fixed)
        inter_alpha  = hp("interaction_alpha",         interaction_alpha_fixed)
        inter_s_len  = hp("interaction_space_length",  interaction_space_length_fixed)
        inter_t_len  = hp("interaction_time_length",   interaction_time_length_fixed)
        sigma_h      = hp("sigma_h",                   sigma_h_fixed)
        ell_h        = hp("ell_h",                     ell_h_fixed)

        # Seasonal Cholesky: recompute dynamically only if a seasonal HP is free
        if seasonal_free:
            K_h = _periodic_kernel_jax(sigma_h, ell_h) + 1e-6 * jnp.eye(12)
            L_h = jnp.linalg.cholesky(K_h)
        else:
            L_h = L_h_fixed

        z_h   = numpyro.sample("z_h", dist.Normal(0.0, 1.0).expand([12]))
        h_raw = b_h + L_h @ z_h
        h_monthly = h_raw - h_raw.mean()

        space_raw = hsgp_se(
            "space", coords_centered,
            alpha=space_alpha, length=space_length, ell=[1.1, 1.1], m=[8, 8],
        )
        time_raw = hsgp_se(
            "time", time_centered,
            alpha=time_alpha, length=time_length, ell=45.0, m=28,
        )
        interaction_raw = separable_interaction_hsgp(
            "interaction", state_index, time_index,
            coords_centered, time_centered,
            alpha=inter_alpha, space_length=inter_s_len, time_length=inter_t_len,
        )

        space_effect       = space_raw       - jnp.mean(space_raw)
        time_effect        = time_raw        - jnp.mean(time_raw)
        interaction_effect = interaction_raw - jnp.mean(interaction_raw)
        latent_f = space_effect[state_index] + time_effect[time_index] + interaction_effect
        h_t = h_monthly[time_index % 12]

        eta = log_population + beta0 + h_t + latent_f
        eta = jnp.clip(eta, -35.0, 20.0)
        numpyro.deterministic("latent_f", latent_f)
        numpyro.deterministic("g_contrib", space_effect[state_index])
        numpyro.deterministic("q_contrib", time_effect[time_index])
        numpyro.deterministic("w_contrib", interaction_effect)
        numpyro.sample("obs", dist.Poisson(rate=jnp.exp(eta)), obs=count)

    return model


# ─── Data prep ────────────────────────────────────────────────────────────────

def make_data(panel, regions):
    coords = regions[["x_coord", "y_coord"]].to_numpy()
    coords_centered = coords - 0.5
    n_months = panel["time_index"].nunique()
    time_values = np.arange(n_months, dtype=float)[:, None]
    time_centered = time_values - time_values.mean()
    return {
        "count":            jnp.asarray(panel["count"].to_numpy(dtype=np.int32)),
        "log_population":   jnp.asarray(panel["log_population"].to_numpy(dtype=np.float32)),
        "b_h":              jnp.asarray(_b_h.astype(np.float32)),
        "L_h_fixed":        jnp.asarray(_L_h_fixed.astype(np.float32)),
        "state_index":      jnp.asarray(panel["state_index"].to_numpy(dtype=np.int32)),
        "time_index":       jnp.asarray(panel["time_index"].to_numpy(dtype=np.int32)),
        "coords_centered":  jnp.asarray(coords_centered.astype(np.float32)),
        "time_centered":    jnp.asarray(time_centered.astype(np.float32)),
        "beta0_prior_mean": float(np.log(1.2e-5)),
        "space_alpha_fixed":              float(TRUE_HP["space_alpha"]),
        "space_length_fixed":             float(TRUE_HP["space_length"]),
        "time_alpha_fixed":               float(TRUE_HP["time_alpha"]),
        "time_length_fixed":              float(TRUE_HP["time_length"]),
        "interaction_alpha_fixed":        float(TRUE_HP["interaction_alpha"]),
        "interaction_space_length_fixed": float(TRUE_HP["interaction_space_length"]),
        "interaction_time_length_fixed":  float(TRUE_HP["interaction_time_length"]),
        "sigma_h_fixed":                  float(TRUE_HP["sigma_h"]),
        "ell_h_fixed":                    float(TRUE_HP["ell_h"]),
    }


# ─── NUTS runner ──────────────────────────────────────────────────────────────

def run_nuts(model_fn, data, nuts_seed, free_params=()):
    numpyro.set_host_device_count(NUM_CHAINS)
    hp_init_vals = {hp: _HP_INIT[hp] for hp in free_params if hp in _HP_INIT}
    init_strategy = init_to_value(values=hp_init_vals) if hp_init_vals else None

    kernel = NUTS(
        model_fn,
        target_accept_prob=TARGET_ACCEPT,
        max_tree_depth=MAX_TREE_DEPTH,
        **({"init_strategy": init_strategy} if init_strategy is not None else {}),
    )
    mcmc = MCMC(
        kernel,
        num_warmup=NUM_WARMUP,
        num_samples=NUM_SAMPLES,
        num_chains=NUM_CHAINS,
        chain_method="sequential",
        progress_bar=True,
    )
    t0 = time.perf_counter()
    mcmc.run(
        jax.random.PRNGKey(nuts_seed),
        **data,
        extra_fields=("accept_prob", "num_steps", "diverging", "potential_energy"),
    )
    jax.block_until_ready(mcmc.get_samples())
    elapsed = time.perf_counter() - t0
    return mcmc, elapsed


# ─── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(mcmc, panel, elapsed, stage, dgp_seed, free_params):
    samples_by_chain = mcmc.get_samples(group_by_chain=True)
    flat = {
        name: np.asarray(value).reshape((-1,) + tuple(np.asarray(value).shape[2:]))
        for name, value in samples_by_chain.items()
    }

    latent_samples = flat["latent_f"]
    mean_f  = latent_samples.mean(axis=0)
    ci_low  = np.quantile(latent_samples, 0.025, axis=0)
    ci_high = np.quantile(latent_samples, 0.975, axis=0)
    f_true  = panel["latent_f_true"].to_numpy()

    rmse     = float(np.sqrt(np.mean((mean_f - f_true) ** 2)))
    coverage = float(((f_true >= ci_low) & (f_true <= ci_high)).mean())

    diag = numpyro_summary({k: v for k, v in samples_by_chain.items()
                            if k not in ("g_contrib", "q_contrib", "w_contrib")},
                           prob=0.95, group_by_chain=True)
    all_rhat = np.concatenate([
        np.asarray(v.get("r_hat", np.nan), dtype=float).ravel() for v in diag.values()
    ])
    all_ess = np.concatenate([
        np.asarray(v.get("n_eff", np.nan), dtype=float).ravel() for v in diag.values()
    ])
    max_rhat = float(np.nanmax(all_rhat))
    min_ess  = float(np.nanmin(all_ess))

    extra  = mcmc.get_extra_fields(group_by_chain=True)
    n_div  = int(np.asarray(extra["diverging"]).sum())
    mean_accept = float(np.asarray(extra["accept_prob"]).mean())

    row = {
        "stage":            stage,
        "stage_name":       STAGE_NAMES[stage],
        "n_free_params":    len(free_params),
        "free_params":      ",".join(free_params) if free_params else "none",
        "dgp_seed":         dgp_seed,
        "rmse_f":           rmse,
        "coverage_95":      coverage,
        "max_rhat":         max_rhat,
        "min_ess":          min_ess,
        "n_divergences":    n_div,
        "mean_accept_prob": mean_accept,
        "runtime_seconds":  elapsed,
        "ess_per_sec":      min_ess / elapsed,
        "state_index":         panel["state_index"].tolist(),
        "time_index":          panel["time_index"].tolist(),
        "latent_f_true":       f_true.tolist(),
        "latent_f_post_mean":  mean_f.tolist(),
        "latent_f_post_sd":    np.std(latent_samples, axis=0).tolist(),
        "latent_f_ci_low":     ci_low.tolist(),
        "latent_f_ci_high":    ci_high.tolist(),
    }

    # recovered variance decomposition (per-draw var of each component, avg over draws)
    vtot = 0.0
    for comp in ["g_contrib", "q_contrib", "w_contrib"]:
        v = float(flat[comp].var(axis=1).mean()); row[f"var_{comp}"] = v; vtot += v
    for comp in ["g_contrib", "q_contrib", "w_contrib"]:
        row[f"frac_{comp}"] = row[f"var_{comp}"] / vtot
    print(f"    var decomposition (g:q:w) recovered = "
          f"{row['frac_g_contrib']:.2f} : {row['frac_q_contrib']:.2f} : {row['frac_w_contrib']:.2f}")

    # Seasonal curve posterior (reconstruct h_monthly per sample)
    if "z_h" in flat:
        z_h_s = flat["z_h"]                          # (n_samples, 12)
        if ("sigma_h" in flat) or ("ell_h" in flat):
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
        else:
            h_raw_s = _b_h + z_h_s @ _L_h_fixed.T
            h_monthly_s = h_raw_s - h_raw_s.mean(axis=1, keepdims=True)

        h_true = np.array([
            panel[panel["time_index"] == m]["seasonal_h_true"].iloc[0] for m in range(12)
        ])
        row["h_monthly_true"]      = h_true.tolist()
        row["h_monthly_post_mean"] = h_monthly_s.mean(axis=0).tolist()
        row["h_monthly_post_sd"]   = h_monthly_s.std(axis=0).tolist()
        row["h_monthly_ci_low"]    = np.quantile(h_monthly_s, 0.025, axis=0).tolist()
        row["h_monthly_ci_high"]   = np.quantile(h_monthly_s, 0.975, axis=0).tolist()

    # Per-HP posterior for every free hyperparameter
    for hp in free_params:
        if hp in flat:
            samps = flat[hp]
            row[f"{hp}_samples"]    = samps.tolist()
            row[f"{hp}_post_mean"]  = float(samps.mean())
            row[f"{hp}_post_sd"]    = float(samps.std())
            row[f"{hp}_post_q025"]  = float(np.quantile(samps, 0.025))
            row[f"{hp}_post_q975"]  = float(np.quantile(samps, 0.975))
            row[f"{hp}_truth"]      = TRUE_HP[hp]
            row[f"{hp}_bias"]       = float(samps.mean()) - TRUE_HP[hp]

    return row


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    regimes = pd.read_csv(DATA_DIR / "synthetic_measles_interaction_regimes.csv")
    regime  = regimes.loc[regimes["name"] == "interaction_smooth"].iloc[0].to_dict()
    regions = (
        pd.read_csv(DATA_DIR / "synthetic_measles_regions.csv")
        .sort_values("state_index")
        .reset_index(drop=True)
    )
    n_months = 72

    print(f"[config] stages={STAGES}  N_SEEDS={N_SEEDS}  warmup={NUM_WARMUP}  "
          f"samples={NUM_SAMPLES}  chains={NUM_CHAINS}")
    print(f"[priors] main effects: Betancourt containment; "
          f"sigma_h~HalfNormal(0.45)  ell_h~InvGamma(3.93,1.99)")

    all_rows = []

    for dgp_seed_offset in range(N_SEEDS):
        dgp_seed = SEED_BASE + dgp_seed_offset
        print(f"\n{'='*70}")
        print(f"[seed {dgp_seed}] Generating DGP ...")
        panel = generate_panel(regime, regions, n_months, dgp_seed)
        data  = make_data(panel, regions)

        for stage in STAGES:
            free_params = STAGE_FREE[stage]
            stage_name  = STAGE_NAMES[stage]
            out_path = OUTPUT_DIR / f"stage{stage}_{stage_name}_seed{dgp_seed:04d}.json"

            if SKIP_EXISTING and out_path.exists():
                print(f"  [stage {stage} | {stage_name}] skipping (exists)")
                all_rows.append(json.loads(out_path.read_text()))
                continue

            print(f"\n  [stage {stage} | {stage_name}] free={free_params or 'none'}")
            model_fn  = make_hsgp_model(free_params)
            nuts_seed = dgp_seed * 10_000 + stage * 1_000
            mcmc, elapsed = run_nuts(model_fn, data, nuts_seed, free_params=free_params)
            row = compute_metrics(mcmc, panel, elapsed, stage, dgp_seed, free_params)

            print(f"    RMSE={row['rmse_f']:.4f}  cover={row['coverage_95']:.4f}  "
                  f"Rhat={row['max_rhat']:.4f}  ESS={row['min_ess']:.0f}  "
                  f"div={row['n_divergences']}  t={elapsed:.1f}s  ESS/s={row['ess_per_sec']:.1f}")
            for hp in free_params:
                print(f"    {hp:28s} truth={TRUE_HP[hp]:7.3f}  "
                      f"post={row.get(f'{hp}_post_mean', float('nan')):7.3f}  "
                      f"bias={row.get(f'{hp}_bias', float('nan')):+7.3f}  "
                      f"95%CI=[{row.get(f'{hp}_post_q025', float('nan')):.3f}, "
                      f"{row.get(f'{hp}_post_q975', float('nan')):.3f}]")

            out_path.write_text(json.dumps(row, indent=2))
            all_rows.append(row)

    df = pd.DataFrame(all_rows)
    metrics_cols = ["rmse_f", "coverage_95", "max_rhat", "min_ess",
                    "n_divergences", "runtime_seconds", "ess_per_sec"]
    summary = (
        df.groupby(["stage", "stage_name", "n_free_params"])[metrics_cols]
        .mean().round(4).reset_index()
    )
    df.to_csv(OUTPUT_DIR / "seasonal_hp_all_results.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "seasonal_hp_summary.csv", index=False)

    print(f"\n{'='*70}")
    print("Summary (mean across seeds)")
    print(f"{'='*70}")
    print(summary.to_string(index=False))
    print(f"\n[saved] {OUTPUT_DIR / 'seasonal_hp_all_results.csv'}")
    print(f"[saved] {OUTPUT_DIR / 'seasonal_hp_summary.csv'}")


if __name__ == "__main__":
    main()
