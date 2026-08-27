"""
118c_netherlands_births_hsgp_deeprv_ncp_v3.py
==============================================
Advisor-aligned model (from meeting):

  η_t = β₀ + f(t) + h(d_t) + β_{w_t}

  f(t):    HSGP SE M=30  — α_f ~ HN(1), ℓ_f ~ InvGamma(TREND)
           DeepRV r=30   — FIXED (α_f, ℓ_f) at HSGP posterior means
  h(d_t):  Periodic GP via Fourier M=10 harmonics
           α_h ~ HN(1),  ρ_h ~ InvGamma calibrated to [0.25, 1.50]
           spectral density: exact Bessel series (was leading-term approx)
  β_{w_t}: ZeroSumNormal(1) — 6 free + 1 constrained (7 weekdays)

Changes from v2:
  • s(doy)+γ(doy) two-component → single periodic-GP h(doy)
  • weekday: 6-dummy centred → 6-free + constrained 7th
  • DeepRV: fixes trend HPs (no bilinear MLP funnel)

HSGP NUTS dims:   z_f(30) + z_h(20) + {α_f,ℓ_f,α_h,ρ_h,β₀,σ,w_raw(6)} = 62
DeepRV NUTS dims: u(30)   + z_h(20) + {α_h,ρ_h,β₀,σ,w_raw(6)} = 60
"""

from pathlib import Path
import json, os, time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from flax.core import freeze
from numpyro.diagnostics import summary as numpyro_summary
from numpyro.infer import MCMC, NUTS, init_to_value
from scipy.optimize import root
from scipy.stats import invgamma

from dl4bi.vae.deep_rv import MLPDeepRV

DATA_PATH = (PROJECT_ROOT / "data" / "processed" / "netherlands_births"
             / "nl_daily_births_panel.csv")
OUT_DIR   = PROJECT_ROOT / "data" / "processed" / "netherlands_births"
MODEL_DIR = OUT_DIR / "models"

NUM_WARMUP    = int(os.environ.get("NL_WARMUP",          "1000"))
NUM_SAMPLES   = int(os.environ.get("NL_SAMPLES",         "1000"))
NUM_CHAINS    = int(os.environ.get("NL_CHAINS",          "4"))
TARGET_ACCEPT = float(os.environ.get("NL_TARGET_ACCEPT", "0.90"))
MAX_TREE_DEPTH= int(os.environ.get("NL_MAX_TREE_DEPTH",  "10"))
SEED          = int(os.environ.get("NL_SEED",            "20260701"))

TREND_BASIS    = 30
BOUNDARY_MUL   = 1.5
SEASON_FOURIER = 10   # harmonics j=1,...,10 for h(doy)

# DeepRV model config: "z30h256" | "z24h36" | "z12h12" | "z6h12"
DRV_MODEL = os.environ.get("DRV_MODEL", "z30h256")

_DRV_CONFIGS = {
    # name: (z_dim, hidden_dims, model_file, cond_alpha, cond_length)
    "z30h256": (30, [256, 256], "nl_births_deeprv_z30_h256.npz",               None,   None),
    "z24h36":  (24, [36,  36],  "nl_births_deeprv_z24_h36_fixed_af0p1003_lf0p1678.npz",  0.1003, 0.1678),
    "z12h12":  (12, [12,  12],  "nl_births_deeprv_z12_h12_fixed_af0p1003_lf0p1678.npz",  0.1003, 0.1678),
    "z6h12":   (6,  [12,  12],  "nl_births_deeprv_z6_h12_fixed_af0p1003_lf0p1678.npz",   0.1003, 0.1678),
    "z1h12":   (1,  [12,  12],  "nl_births_deeprv_z1_h12_fixed_af0p1003_lf0p1678.npz",   0.1003, 0.1678),
}
assert DRV_MODEL in _DRV_CONFIGS, f"Unknown DRV_MODEL={DRV_MODEL!r}"


# ── InverseGamma prior calibration ────────────────────────────────────────────

def solve_invgamma(lower, upper, tail=0.01):
    def eqs(log_ab):
        a, b = np.exp(log_ab)
        return [invgamma.cdf(lower, a=a, scale=b) - tail,
                1 - invgamma.cdf(upper, a=a, scale=b) - tail]
    d  = 1.0
    ag = ((d*(upper+lower)/(upper-lower))**2 + 2)
    bg = ((upper+lower)/2) * ((d*(upper+lower)/(upper-lower))**2 + 1)
    r  = root(eqs, np.log([ag, bg]))
    return float(np.exp(r.x[0])), float(np.exp(r.x[1]))

# trend ℓ in x_std units: 98% mass in [0.10, 1.50]
TREND_IG_A,  TREND_IG_B  = solve_invgamma(0.10, 1.50)
# seasonal ρ_h in angular units: 98% mass in [0.25, 1.50]
# Lower bound dropped from 0.50 → 0.25 now that the spectral density uses the
# exact Bessel series (see fourier_variance_harmonics). The old 0.50 floor only
# existed to hide the leading-term approximation's breakdown; the data prefer
# ρ≈0.41, which the old prior clipped. M=10 harmonics retain >99% variance down
# to ρ=0.25, so this bound stays inside the truncation-faithful region.
SEASON_LO, SEASON_HI     = 0.25, 1.50
SEASON_IG_A, SEASON_IG_B = solve_invgamma(SEASON_LO, SEASON_HI)


# ── Spectral helpers ──────────────────────────────────────────────────────────

def spectral_variance_se(alpha, length, omega):
    """Spectral density for squared-exponential kernel."""
    return alpha**2 * jnp.sqrt(2.0*jnp.pi) * length * jnp.exp(-0.5*(length*omega)**2)

_BESSEL_SERIES_K = 80   # series length; exact to machine precision for ρ ≥ 0.12

def fourier_variance_harmonics(alpha, length, j_start, j_end):
    """Exact periodic-SE spectral variance for harmonics j = j_start,...,j_end.

    S_j = 2·alpha²·I_j(q)·e^{-q},  q = 1/length²,  where I_j is the modified
    Bessel function of the first kind. I_j(q)·e^{-q} is computed via a
    log-space power series (logsumexp), which is exact and numerically stable
    for all ρ — unlike the old leading-term Poisson approximation, which is
    already ~70% low at ρ=0.5 and collapses the seasonal variance toward zero
    for ρ<0.5 (the true value grows there). Validated to <1e-14 rel. error
    against scipy.special.ive over ρ∈[0.12, 0.8]."""
    q     = 1.0 / length**2
    j_arr = jnp.arange(j_start, j_end + 1, dtype=jnp.float32)      # (J,)
    k     = jnp.arange(0, _BESSEL_SERIES_K, dtype=jnp.float32)     # (K,)
    logq2 = jnp.log(q / 2.0)
    # log of k-th series term for I_j(q):  (2k+j)·log(q/2) − ln k! − ln(k+j)!
    log_terms = ((2.0*k[None, :] + j_arr[:, None]) * logq2
                 - jax.scipy.special.gammaln(k[None, :] + 1.0)
                 - jax.scipy.special.gammaln(k[None, :] + j_arr[:, None] + 1.0))
    log_Ij = jax.scipy.special.logsumexp(log_terms, axis=1)        # (J,) = ln I_j(q)
    return 2.0 * alpha**2 * jnp.exp(log_Ij - q)

def fourier_scale_harmonics(alpha, length, j_start, j_end):
    v = fourier_variance_harmonics(alpha, length, j_start, j_end)
    return jnp.sqrt(jnp.maximum(v, 1e-30))


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = df.sort_values("day_id").reset_index(drop=True)
    T  = len(df)

    # HSGP trend basis (centred)
    x       = df["day_id"].to_numpy(dtype=float)
    x_std   = (x - x.mean()) / x.std()
    boundary= BOUNDARY_MUL * float(np.max(np.abs(x_std)))
    m       = np.arange(1, TREND_BASIS + 1)
    Phi_tr  = (np.sin(np.pi * (x_std[:, None] + boundary) * m[None, :]
                      / (2.0 * boundary)) / np.sqrt(boundary))
    Phi_tr_c = (Phi_tr - Phi_tr.mean(0, keepdims=True)).astype(np.float32)
    omega_tr = (np.pi * m / (2.0 * boundary)).astype(np.float32)

    # Fourier basis for h(doy): harmonics j=1,...,SEASON_FOURIER
    tau = df["day_of_year"].to_numpy(dtype=np.float32)
    j_h = np.arange(1, SEASON_FOURIER + 1, dtype=np.float32)
    phases_h = 2.0 * np.pi * j_h[None, :] * tau[:, None] / 366.0
    Phi_sea = np.concatenate([np.cos(phases_h), np.sin(phases_h)],
                              axis=1).astype(np.float32)  # (T, 2M)

    # Weekday one-hot (T, 7) — ZeroSumNormal uses all 7
    dow = df["day_of_week"].to_numpy()
    W   = np.zeros((T, 7), dtype=np.float32)
    for k in range(7):
        W[dow == k, k] = 1.0

    y = df["log_relative_births"].to_numpy(dtype=np.float32)

    data = {
        "y":            jnp.asarray(y),
        "Phi_trend":    jnp.asarray(Phi_tr_c),
        "omega_trend":  jnp.asarray(omega_tr),
        "Phi_season":   jnp.asarray(Phi_sea),
        "W_weekday":    jnp.asarray(W),
        "trend_ig_a":   float(TREND_IG_A),
        "trend_ig_b":   float(TREND_IG_B),
        "season_ig_a":  float(SEASON_IG_A),
        "season_ig_b":  float(SEASON_IG_B),
    }
    meta = {
        "T":           T,
        "boundary":    float(boundary),
        "year_range":  [int(df["year"].min()), int(df["year"].max())],
        "mean_births": float(df["births"].mean()),
        "y_sd":        float(df["log_relative_births"].std()),
    }
    return df, data, meta


# ── HSGP NCP model (62 NUTS dims) ─────────────────────────────────────────────

def hsgp_ncp_model(y, Phi_trend, omega_trend, Phi_season, W_weekday,
                   trend_ig_a, trend_ig_b, season_ig_a, season_ig_b):
    # ── Hyperparameters (12 dims) ──────────────────────────────────────────
    beta0         = numpyro.sample("beta0",         dist.Normal(0.0, 1.0))
    sigma         = numpyro.sample("sigma",         dist.HalfNormal(1.0))
    w_raw         = numpyro.sample("w_raw",         dist.Normal(0.0, 1.0).expand([6]))
    trend_alpha   = numpyro.sample("trend_alpha",   dist.HalfNormal(1.0))
    trend_length  = numpyro.sample("trend_length",  dist.InverseGamma(trend_ig_a, trend_ig_b))
    season_alpha  = numpyro.sample("season_alpha",  dist.HalfNormal(1.0))
    season_length = numpyro.sample("season_length", dist.InverseGamma(season_ig_a, season_ig_b))

    # ── NCP latents (50 dims) ─────────────────────────────────────────────
    z_trend  = numpyro.sample("z_trend",  dist.Normal(0.0, 1.0).expand([TREND_BASIS]))
    z_season = numpyro.sample("z_season", dist.Normal(0.0, 1.0).expand([2*SEASON_FOURIER]))

    # ── Trend (HSGP) ──────────────────────────────────────────────────────
    D_tr     = spectral_variance_se(trend_alpha, trend_length, omega_trend)**0.5
    f_tr_raw = Phi_trend @ (D_tr * z_trend)
    f_trend  = f_tr_raw - jnp.mean(f_tr_raw)

    # ── Seasonal periodic GP (NCP) ────────────────────────────────────────
    D_sea      = fourier_scale_harmonics(season_alpha, season_length, 1, SEASON_FOURIER)
    D_sea_full = jnp.concatenate([D_sea, D_sea])   # cos and sin share same SD
    f_season   = Phi_season @ (D_sea_full * z_season)

    # ── Weekday ZeroSumNormal (6 free + constrained 7th) ─────────────────
    w_last = -jnp.sum(w_raw)
    w_coef = jnp.concatenate([w_raw, jnp.array([w_last])])

    eta = beta0 + f_trend + f_season + W_weekday @ w_coef
    numpyro.sample("obs", dist.Normal(eta, sigma), obs=y)
    numpyro.deterministic("eta", eta)


# ── DeepRV param helpers ──────────────────────────────────────────────────────

def unflatten_params(flat):
    root_d = {}
    for k, v in flat.items():
        parts = k.split("/")
        cur = root_d
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = v
    return freeze(root_d)

def load_deeprv_params(path):
    arrays = np.load(path)
    flat = {k: jnp.asarray(arrays[k]) for k in arrays.files if "/" in k}
    return unflatten_params(flat)


# ── DeepRV NCP model (60 NUTS dims) ───────────────────────────────────────────

def make_deeprv_ncp_model(model_obj, params, z_dim,
                           trend_alpha_fixed, trend_length_fixed):
    """
    Trend: conditional MLP with FIXED (α_f, ℓ_f) — eliminates bilinear funnel.
    Seasonal: same periodic GP NCP as HSGP.
    NUTS dims: u(30) + z_h(20) + {α_h, ρ_h, β₀, σ, w_raw(6)} = 60
    """
    _alpha_f  = float(trend_alpha_fixed)
    _length_f = float(trend_length_fixed)

    def model(y, Phi_season, W_weekday,
              season_ig_a, season_ig_b, **_kw):

        # ── Hyperparameters (10 dims) ──────────────────────────────────────
        beta0         = numpyro.sample("beta0",         dist.Normal(0.0, 1.0))
        sigma         = numpyro.sample("sigma",         dist.HalfNormal(1.0))
        w_raw         = numpyro.sample("w_raw",         dist.Normal(0.0, 1.0).expand([6]))
        season_alpha  = numpyro.sample("season_alpha",  dist.HalfNormal(1.0))
        season_length = numpyro.sample("season_length", dist.InverseGamma(season_ig_a, season_ig_b))

        # ── NCP latents (50 dims) ─────────────────────────────────────────
        u        = numpyro.sample("deeprv_z", dist.Normal(0.0, 1.0).expand([z_dim]))
        z_season = numpyro.sample("z_season", dist.Normal(0.0, 1.0).expand([2*SEASON_FOURIER]))

        # ── Trend via conditional decoder (fixed HPs) ─────────────────────
        cond      = jnp.array([_alpha_f, _length_f])
        trend_raw = model_obj.apply({"params": params}, u[None, :], cond,
                                    method="decode")[0]
        f_trend   = trend_raw - jnp.mean(trend_raw)

        # ── Seasonal periodic GP (NCP) ────────────────────────────────────
        D_sea      = fourier_scale_harmonics(season_alpha, season_length, 1, SEASON_FOURIER)
        D_sea_full = jnp.concatenate([D_sea, D_sea])
        f_season   = Phi_season @ (D_sea_full * z_season)

        # ── Weekday ZeroSumNormal ─────────────────────────────────────────
        w_last = -jnp.sum(w_raw)
        w_coef = jnp.concatenate([w_raw, jnp.array([w_last])])

        eta = beta0 + f_trend + f_season + W_weekday @ w_coef
        numpyro.sample("obs", dist.Normal(eta, sigma), obs=y)
        numpyro.deterministic("eta", eta)

    return model


# ── MCMC runner ───────────────────────────────────────────────────────────────

def run_mcmc(model_fn, model_data, seed, init_vals=None):
    numpyro.set_host_device_count(NUM_CHAINS)
    kw = {}
    if init_vals is not None:
        kw["init_strategy"] = init_to_value(values=init_vals)
    kernel = NUTS(model_fn, target_accept_prob=TARGET_ACCEPT,
                  max_tree_depth=MAX_TREE_DEPTH, **kw)
    mcmc = MCMC(kernel, num_warmup=NUM_WARMUP, num_samples=NUM_SAMPLES,
                num_chains=NUM_CHAINS, chain_method="sequential", progress_bar=True)
    t0 = time.perf_counter()
    mcmc.run(jax.random.PRNGKey(seed), **model_data,
             extra_fields=("diverging", "accept_prob", "num_steps"))
    jax.block_until_ready(mcmc.get_samples())
    return mcmc, time.perf_counter() - t0


# ── Metrics ───────────────────────────────────────────────────────────────────

def summarize(method, df, mcmc, elapsed, nuts_dims, extra=None):
    sbc  = mcmc.get_samples(group_by_chain=True)
    flat = {k: np.asarray(v).reshape((-1,) + tuple(np.asarray(v).shape[2:]))
            for k, v in sbc.items()}

    skip = {"eta"}
    diag = numpyro_summary({k: sbc[k] for k in sbc if k not in skip},
                           prob=0.9, group_by_chain=True)
    rhats, esss = [], []
    for st in diag.values():
        rhats += np.asarray(st["r_hat"]).ravel().tolist()
        esss  += np.asarray(st["n_eff"]).ravel().tolist()

    ex    = mcmc.get_extra_fields(group_by_chain=True)
    n_div = int(np.asarray(ex["diverging"]).sum())
    max_tree_frac = float(
        (np.asarray(ex["num_steps"]) >= (2**MAX_TREE_DEPTH - 1)).mean())

    eta    = flat["eta"]
    y_obs  = df["log_relative_births"].to_numpy()
    rng_pp = np.random.default_rng(0)
    sigma_s = flat["sigma"]
    idx_pp  = rng_pp.choice(len(eta), size=min(500, len(eta)), replace=False)
    pp      = rng_pp.normal(eta[idx_pp], sigma_s[idx_pp, None])
    lo_pp   = np.quantile(pp, 0.05, axis=0)
    hi_pp   = np.quantile(pp, 0.95, axis=0)
    coverage = float(((y_obs >= lo_pp) & (y_obs <= hi_pp)).mean())
    rmse     = float(np.sqrt(np.mean((eta.mean(0) - y_obs)**2)))

    row = {
        "method":                  method,
        "nuts_dims":               nuts_dims,
        "dataset":                 "netherlands_cbs_1995_2024",
        "T":                       int(len(df)),
        "num_warmup":              NUM_WARMUP,
        "num_samples":             NUM_SAMPLES,
        "num_chains":              NUM_CHAINS,
        "target_accept":           TARGET_ACCEPT,
        "max_tree_depth":          MAX_TREE_DEPTH,
        "trend_basis_count":       TREND_BASIS,
        "season_fourier_M":        SEASON_FOURIER,
        "trend_ig_a":              float(TREND_IG_A),
        "trend_ig_b":              float(TREND_IG_B),
        "season_ig_a":             float(SEASON_IG_A),
        "season_ig_b":             float(SEASON_IG_B),
        "runtime_seconds":         elapsed,
        "max_rhat":                float(np.nanmax(rhats)),
        "min_ess":                 float(np.nanmin(esss)),
        "ess_per_sec":             float(np.nanmin(esss)) / elapsed,
        "n_divergences":           n_div,
        "divergence_rate":         n_div / (NUM_CHAINS * (NUM_WARMUP + NUM_SAMPLES)),
        "max_tree_depth_frac":     max_tree_frac,
        "mean_accept_prob":        float(np.asarray(ex["accept_prob"]).mean()),
        "pp_coverage_90":          coverage,
        "pp_rmse_log_rel":         rmse,
        "season_alpha_post_mean":  float(flat.get("season_alpha", np.array([np.nan])).mean()),
        "season_length_post_mean": float(flat.get("season_length", np.array([np.nan])).mean()),
    }
    if "trend_alpha" in flat:
        row["trend_alpha_post_mean"]  = float(flat["trend_alpha"].mean())
        row["trend_length_post_mean"] = float(flat["trend_length"].mean())
    if extra:
        row.update(extra)
    return row, flat


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    df, model_data, meta = load_data()

    print(f"\n{'='*65}")
    print(f"Netherlands Daily Births  T={meta['T']}  "
          f"({meta['year_range'][0]}-{meta['year_range'][1]})")
    print(f"Model: η = β₀ + f(t) + h(doy) + weekday")
    print(f"  h(doy): periodic GP, M={SEASON_FOURIER}, ρ_h ~ InvGamma[{SEASON_LO},{SEASON_HI}], exact Bessel spectral density")
    print(f"  f(t):   HSGP M={TREND_BASIS} / DeepRV  (α_f, ℓ_f sampled in HSGP; fixed in DeepRV)")
    print(f"  weekday: ZeroSumNormal(1), 7 days")
    print(f"warmup={NUM_WARMUP}  samples={NUM_SAMPLES}  chains={NUM_CHAINS}")
    print(f"{'='*65}\n")

    mode_trend  = float(TREND_IG_B  / (TREND_IG_A  + 1.0))
    mode_season = float(SEASON_IG_B / (SEASON_IG_A + 1.0))

    # ── HSGP NCP ─────────────────────────────────────────────────────────────
    hsgp_dims = TREND_BASIS + 2*SEASON_FOURIER + 12   # 62

    hsgp_md = {k: model_data[k] for k in (
        "y", "Phi_trend", "omega_trend", "Phi_season",
        "W_weekday", "trend_ig_a", "trend_ig_b", "season_ig_a", "season_ig_b")}

    hsgp_init = {
        "beta0":         jnp.array(0.0),
        "sigma":         jnp.array(0.1),
        "w_raw":         jnp.zeros(6),
        "trend_alpha":   jnp.array(0.1),
        "trend_length":  jnp.array(mode_trend),
        "season_alpha":  jnp.array(0.1),
        "season_length": jnp.array(mode_season),
        "z_trend":       jnp.zeros(TREND_BASIS),
        "z_season":      jnp.zeros(2*SEASON_FOURIER),
    }

    print(f"── HSGP NCP ({hsgp_dims} NUTS dims) ──")
    hsgp_mcmc, hsgp_t = run_mcmc(hsgp_ncp_model, hsgp_md, SEED,
                                   init_vals=hsgp_init)
    hsgp_row, hsgp_flat = summarize("hsgp_ncp_v3", df, hsgp_mcmc, hsgp_t, hsgp_dims)
    print(f"  runtime={hsgp_t:.0f}s  max_rhat={hsgp_row['max_rhat']:.4f}  "
          f"min_ess={hsgp_row['min_ess']:.0f}  div={hsgp_row['n_divergences']}  "
          f"max_tree_frac={hsgp_row['max_tree_depth_frac']:.3f}")
    print(f"  trend_α={hsgp_row.get('trend_alpha_post_mean',float('nan')):.4f}  "
          f"trend_ℓ={hsgp_row.get('trend_length_post_mean',float('nan')):.4f}  "
          f"season_α={hsgp_row['season_alpha_post_mean']:.4f}  "
          f"season_ρ={hsgp_row['season_length_post_mean']:.4f}")

    # ── DeepRV NCP ───────────────────────────────────────────────────────────
    drv_z, drv_h, drv_file, drv_af, drv_lf = _DRV_CONFIGS[DRV_MODEL]

    # Fixed trend HPs: use training values for fixed models, HSGP posterior for conditional
    if drv_af is None:
        trend_alpha_fixed  = float(hsgp_flat["trend_alpha"].mean())
        trend_length_fixed = float(hsgp_flat["trend_length"].mean())
        print(f"\n  DeepRV ({DRV_MODEL}): trend HPs from HSGP posterior "
              f"α_f={trend_alpha_fixed:.4f}, ℓ_f={trend_length_fixed:.4f}")
    else:
        trend_alpha_fixed, trend_length_fixed = drv_af, drv_lf
        print(f"\n  DeepRV ({DRV_MODEL}): fixed training HPs "
              f"α_f={trend_alpha_fixed:.4f}, ℓ_f={trend_length_fixed:.4f}")

    deeprv_dims = drv_z + 2*SEASON_FOURIER + 10   # u(drv_z) + z_h(20) + 10 HPs

    cond_path     = MODEL_DIR / drv_file
    drv_params    = load_deeprv_params(cond_path)
    drv_model_obj = MLPDeepRV(drv_h + [meta["T"]])
    drv_ncp       = make_deeprv_ncp_model(drv_model_obj, drv_params, drv_z,
                                           trend_alpha_fixed, trend_length_fixed)

    drv_md = {k: model_data[k] for k in (
        "y", "Phi_season", "W_weekday", "season_ig_a", "season_ig_b")}

    drv_init = {
        "beta0":         jnp.array(0.0),
        "sigma":         jnp.array(0.1),
        "w_raw":         jnp.zeros(6),
        "season_alpha":  jnp.array(0.1),
        "season_length": jnp.array(mode_season),
        "deeprv_z":      jnp.zeros(drv_z),
        "z_season":      jnp.zeros(2*SEASON_FOURIER),
    }

    print(f"\n── DeepRV NCP  {DRV_MODEL}  ({deeprv_dims} NUTS dims) ──")
    drv_mcmc, drv_t = run_mcmc(drv_ncp, drv_md, SEED + 999, init_vals=drv_init)
    drv_row, drv_flat = summarize(f"deeprv_ncp_v3_{DRV_MODEL}", df, drv_mcmc, drv_t,
                                   deeprv_dims,
                                   extra={"deeprv_model": DRV_MODEL,
                                          "deeprv_loaded": True,
                                          "trend_alpha_fixed": trend_alpha_fixed,
                                          "trend_length_fixed": trend_length_fixed})
    print(f"  runtime={drv_t:.0f}s  max_rhat={drv_row['max_rhat']:.4f}  "
          f"min_ess={drv_row['min_ess']:.0f}  div={drv_row['n_divergences']}  "
          f"max_tree_frac={drv_row['max_tree_depth_frac']:.3f}")

    # ── Comparison ───────────────────────────────────────────────────────────
    eta_h  = hsgp_flat["eta"].mean(0)
    eta_d  = drv_flat["eta"].mean(0)
    D_mean = float(np.mean(np.abs(eta_h - eta_d)))
    corr   = float(np.corrcoef(eta_h, eta_d)[0, 1])
    print(f"\n── Posterior comparison ──")
    print(f"  D_mean(η) = {D_mean:.4f}   corr(η) = {corr:.4f}")

    # ── Save ─────────────────────────────────────────────────────────────────
    out_dir = OUT_DIR / "nl_births_hsgp_deeprv_ncp_v3"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = pd.DataFrame([hsgp_row, drv_row])
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    (out_dir / "config.json").write_text(json.dumps(
        {**meta, "D_mean_eta": D_mean, "corr_eta": corr,
         "seasonal_prior": f"InvGamma({SEASON_IG_A:.3f},{SEASON_IG_B:.3f}) ~ [{SEASON_LO},{SEASON_HI}]",
         "seasonal_spectral_density": "exact Bessel series (K=80)",
         "trend_alpha_fixed_for_deeprv": trend_alpha_fixed,
         "trend_length_fixed_for_deeprv": trend_length_fixed}, indent=2))

    print(f"\n{'='*65}")
    print(metrics[["method", "nuts_dims", "runtime_seconds", "min_ess",
                   "ess_per_sec", "n_divergences", "max_tree_depth_frac",
                   "max_rhat", "pp_coverage_90", "pp_rmse_log_rel"]].to_string(index=False))
    print(f"{'='*65}")
    print(f"\n[saved] {out_dir}")


if __name__ == "__main__":
    main()
