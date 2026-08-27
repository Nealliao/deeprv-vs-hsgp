"""
118d_netherlands_births_plot_components.py
==========================================
Component-decomposition plots for Netherlands daily births 1995-2024.

Layout: 4-row × 2-column figure
  Rows:  Fitted mean | Slow trend | Yearly seasonal | Day of week
  Cols:  HSGP NCP   | DeepRV z12h12

Runs a quick MCMC (NL_WARMUP=500, NL_SAMPLES=500, 1 chain) with
additional deterministic sites for each component.

Usage:
  python 118d_netherlands_births_plot_components.py
  DRV_MODEL=z12h12 NL_WARMUP=500 NL_SAMPLES=500 python 118d_...py
"""

from pathlib import Path
import os, time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from flax.core import freeze
from numpyro.infer import MCMC, NUTS, init_to_value
from scipy.optimize import root
from scipy.stats import invgamma

from dl4bi.vae.deep_rv import MLPDeepRV

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH  = (PROJECT_ROOT / "data" / "processed" / "netherlands_births"
              / "nl_daily_births_panel.csv")
OUT_DIR    = PROJECT_ROOT / "data" / "processed" / "netherlands_births"
MODEL_DIR  = OUT_DIR / "models"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"

# ── MCMC config ───────────────────────────────────────────────────────────────
NUM_WARMUP    = int(os.environ.get("NL_WARMUP",  "500"))
NUM_SAMPLES   = int(os.environ.get("NL_SAMPLES", "500"))
NUM_CHAINS    = 1
TARGET_ACCEPT = 0.90
MAX_TREE_DEPTH= 10
SEED          = int(os.environ.get("NL_SEED",    "20260701"))

# ── Model config ──────────────────────────────────────────────────────────────
TREND_BASIS    = 30
BOUNDARY_MUL   = 1.5
SEASON_FOURIER = 10

DRV_MODEL = os.environ.get("DRV_MODEL", "z12h12")
_DRV_CONFIGS = {
    "z12h12": (12, [12, 12], "nl_births_deeprv_z12_h12_fixed_af0p1003_lf0p1678.npz", 0.1003, 0.1678),
    "z24h36": (24, [36, 36], "nl_births_deeprv_z24_h36_fixed_af0p1003_lf0p1678.npz", 0.1003, 0.1678),
    "z6h12":  (6,  [12, 12], "nl_births_deeprv_z6_h12_fixed_af0p1003_lf0p1678.npz",  0.1003, 0.1678),
}
assert DRV_MODEL in _DRV_CONFIGS, f"Unknown DRV_MODEL={DRV_MODEL!r}"

# ── InvGamma calibration ──────────────────────────────────────────────────────
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

TREND_IG_A,  TREND_IG_B  = solve_invgamma(0.10, 1.50)
SEASON_IG_A, SEASON_IG_B = solve_invgamma(0.50, 2.00)


# ── Spectral helpers ──────────────────────────────────────────────────────────
def spectral_variance_se(alpha, length, omega):
    return alpha**2 * jnp.sqrt(2.0*jnp.pi) * length * jnp.exp(-0.5*(length*omega)**2)

def fourier_variance_harmonics(alpha, length, j_start, j_end):
    q     = 1.0 / length**2
    j_arr = jnp.arange(j_start, j_end + 1, dtype=jnp.float32)
    log_lam = (2.0*jnp.log(alpha) + jnp.log(2.0) - q
               + j_arr * jnp.log(q / 2.0)
               - jax.scipy.special.gammaln(j_arr + 1.0))
    return jnp.exp(log_lam)

def fourier_scale_harmonics(alpha, length, j_start, j_end):
    v = fourier_variance_harmonics(alpha, length, j_start, j_end)
    return jnp.sqrt(jnp.maximum(v, 1e-30))


# ── Data loading ──────────────────────────────────────────────────────────────
def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df = df.sort_values("day_id").reset_index(drop=True)
    T  = len(df)

    x       = df["day_id"].to_numpy(dtype=float)
    x_std   = (x - x.mean()) / x.std()
    boundary= BOUNDARY_MUL * float(np.max(np.abs(x_std)))
    m       = np.arange(1, TREND_BASIS + 1)
    Phi_tr  = (np.sin(np.pi * (x_std[:, None] + boundary) * m[None, :]
                      / (2.0 * boundary)) / np.sqrt(boundary))
    Phi_tr_c = (Phi_tr - Phi_tr.mean(0, keepdims=True)).astype(np.float32)
    omega_tr = (np.pi * m / (2.0 * boundary)).astype(np.float32)

    tau = df["day_of_year"].to_numpy(dtype=np.float32)
    j_h = np.arange(1, SEASON_FOURIER + 1, dtype=np.float32)
    phases_h = 2.0 * np.pi * j_h[None, :] * tau[:, None] / 366.0
    Phi_sea  = np.concatenate([np.cos(phases_h), np.sin(phases_h)],
                               axis=1).astype(np.float32)

    dow = df["day_of_week"].to_numpy()
    W   = np.zeros((T, 7), dtype=np.float32)
    for k in range(7):
        W[dow == k, k] = 1.0

    y = df["log_relative_births"].to_numpy(dtype=np.float32)
    return df, y, Phi_tr_c, omega_tr, Phi_sea, W, T


# ── Models with component deterministics ──────────────────────────────────────

def hsgp_ncp_model(y, Phi_trend, omega_trend, Phi_season, W_weekday,
                   trend_ig_a, trend_ig_b, season_ig_a, season_ig_b):
    beta0         = numpyro.sample("beta0",         dist.Normal(0.0, 1.0))
    sigma         = numpyro.sample("sigma",         dist.HalfNormal(1.0))
    w_raw         = numpyro.sample("w_raw",         dist.Normal(0.0, 1.0).expand([6]))
    trend_alpha   = numpyro.sample("trend_alpha",   dist.HalfNormal(1.0))
    trend_length  = numpyro.sample("trend_length",  dist.InverseGamma(trend_ig_a, trend_ig_b))
    season_alpha  = numpyro.sample("season_alpha",  dist.HalfNormal(1.0))
    season_length = numpyro.sample("season_length", dist.InverseGamma(season_ig_a, season_ig_b))

    z_trend  = numpyro.sample("z_trend",  dist.Normal(0.0, 1.0).expand([TREND_BASIS]))
    z_season = numpyro.sample("z_season", dist.Normal(0.0, 1.0).expand([2*SEASON_FOURIER]))

    D_tr     = spectral_variance_se(trend_alpha, trend_length, omega_trend)**0.5
    f_tr_raw = Phi_trend @ (D_tr * z_trend)
    f_trend  = f_tr_raw - jnp.mean(f_tr_raw)

    D_sea      = fourier_scale_harmonics(season_alpha, season_length, 1, SEASON_FOURIER)
    D_sea_full = jnp.concatenate([D_sea, D_sea])
    f_season   = Phi_season @ (D_sea_full * z_season)

    w_last   = -jnp.sum(w_raw)
    w_coef   = jnp.concatenate([w_raw, jnp.array([w_last])])
    w_effect = W_weekday @ w_coef

    eta = beta0 + f_trend + f_season + w_effect
    numpyro.sample("obs", dist.Normal(eta, sigma), obs=y)
    numpyro.deterministic("eta",      eta)
    numpyro.deterministic("f_trend",  f_trend)
    numpyro.deterministic("f_season", f_season)
    numpyro.deterministic("w_effect", w_effect)


def make_deeprv_ncp_model(model_obj, params, z_dim, trend_alpha_fixed, trend_length_fixed):
    _af = float(trend_alpha_fixed)
    _lf = float(trend_length_fixed)

    def model(y, Phi_season, W_weekday, season_ig_a, season_ig_b, **_kw):
        beta0         = numpyro.sample("beta0",         dist.Normal(0.0, 1.0))
        sigma         = numpyro.sample("sigma",         dist.HalfNormal(1.0))
        w_raw         = numpyro.sample("w_raw",         dist.Normal(0.0, 1.0).expand([6]))
        season_alpha  = numpyro.sample("season_alpha",  dist.HalfNormal(1.0))
        season_length = numpyro.sample("season_length", dist.InverseGamma(season_ig_a, season_ig_b))

        u        = numpyro.sample("deeprv_z", dist.Normal(0.0, 1.0).expand([z_dim]))
        z_season = numpyro.sample("z_season", dist.Normal(0.0, 1.0).expand([2*SEASON_FOURIER]))

        cond      = jnp.array([_af, _lf])
        trend_raw = model_obj.apply({"params": params}, u[None, :], cond,
                                    method="decode")[0]
        f_trend   = trend_raw - jnp.mean(trend_raw)

        D_sea      = fourier_scale_harmonics(season_alpha, season_length, 1, SEASON_FOURIER)
        D_sea_full = jnp.concatenate([D_sea, D_sea])
        f_season   = Phi_season @ (D_sea_full * z_season)

        w_last   = -jnp.sum(w_raw)
        w_coef   = jnp.concatenate([w_raw, jnp.array([w_last])])
        w_effect = W_weekday @ w_coef

        eta = beta0 + f_trend + f_season + w_effect
        numpyro.sample("obs", dist.Normal(eta, sigma), obs=y)
        numpyro.deterministic("eta",      eta)
        numpyro.deterministic("f_trend",  f_trend)
        numpyro.deterministic("f_season", f_season)
        numpyro.deterministic("w_effect", w_effect)

    return model


# ── DeepRV param loader ───────────────────────────────────────────────────────
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


# ── MCMC runner ───────────────────────────────────────────────────────────────
def run_mcmc(model_fn, model_data, seed, init_vals=None):
    numpyro.set_host_device_count(1)
    kw = {}
    if init_vals is not None:
        kw["init_strategy"] = init_to_value(values=init_vals)
    kernel = NUTS(model_fn, target_accept_prob=TARGET_ACCEPT,
                  max_tree_depth=MAX_TREE_DEPTH, **kw)
    mcmc = MCMC(kernel, num_warmup=NUM_WARMUP, num_samples=NUM_SAMPLES,
                num_chains=1, chain_method="sequential", progress_bar=True)
    t0 = time.perf_counter()
    mcmc.run(jax.random.PRNGKey(seed), **model_data)
    jax.block_until_ready(mcmc.get_samples())
    return mcmc, time.perf_counter() - t0


# ── Extract component summaries ───────────────────────────────────────────────
def extract_components(flat, df, y_obs):
    """Return arrays needed for plotting (all in log-relative space)."""
    S = flat["eta"].shape[0]
    T = flat["eta"].shape[1]

    # Full fit
    eta_s   = np.asarray(flat["eta"])       # (S, T)
    beta0_s = np.asarray(flat["beta0"])     # (S,)
    f_tr_s  = np.asarray(flat["f_trend"])   # (S, T)
    f_se_s  = np.asarray(flat["f_season"])  # (S, T)
    w_eff_s = np.asarray(flat["w_effect"])  # (S, T)

    # Trend = beta0 + f_trend
    trend_s = beta0_s[:, None] + f_tr_s   # (S, T)

    # Seasonal component (in log-relative, centered at 0)
    season_s = f_se_s   # (S, T)

    # Weekday component (in log-relative, centered at 0)
    weekday_s = w_eff_s  # (S, T)

    # Summarize to mean + 90% CI
    def ci(arr, lo=0.05, hi=0.95):
        return (arr.mean(0),
                np.quantile(arr, lo, axis=0),
                np.quantile(arr, hi, axis=0))

    eta_m, eta_lo, eta_hi     = ci(eta_s)
    trend_m, trend_lo, trend_hi = ci(trend_s)

    # Seasonal: group by doy — take posterior mean per-sample then group
    doy  = df["day_of_year"].to_numpy()
    # doy grid: 1..366, sorted
    doy_u = np.arange(1, 367)
    season_by_doy = np.zeros((S, 366))
    for j in range(366):
        mask = (doy == j + 1)
        if mask.sum() > 0:
            season_by_doy[:, j] = season_s[:, mask].mean(axis=1)
    # interpolate missing doys
    present = np.array([(doy == j+1).sum() > 0 for j in range(366)])
    if not present.all():
        present_idx = np.where(present)[0]
        for j in range(366):
            if not present[j]:
                nearest = present_idx[np.argmin(np.abs(present_idx - j))]
                season_by_doy[:, j] = season_by_doy[:, nearest]
    sea_m, sea_lo, sea_hi = ci(season_by_doy)

    # Weekday: extract one representative value per weekday (0=Mon..6=Sun)
    dow = df["day_of_week"].to_numpy()
    w_per_day = np.zeros((S, 7))
    for k in range(7):
        mask = (dow == k)
        w_per_day[:, k] = weekday_s[:, mask].mean(axis=1)
    wday_m, wday_lo, wday_hi = ci(w_per_day)

    # Empirical relative births (100 scale) by doy for scatter
    df_tmp = df.copy()
    df_tmp["rel100"] = 100.0 * np.exp(y_obs)
    doy_scatter = (df_tmp.groupby("day_of_year")["rel100"].mean()
                   .reindex(doy_u).values)

    return {
        "eta_m": eta_m, "eta_lo": eta_lo, "eta_hi": eta_hi,
        "trend_m": trend_m, "trend_lo": trend_lo, "trend_hi": trend_hi,
        "sea_m": sea_m, "sea_lo": sea_lo, "sea_hi": sea_hi,
        "wday_m": wday_m, "wday_lo": wday_lo, "wday_hi": wday_hi,
        "doy_scatter": doy_scatter,
    }


# ── Plotting ──────────────────────────────────────────────────────────────────
BLUE = "#2c7fb8"
RED  = "#e41a1c"
GREY = "#b6b6b6"
RNG  = np.random.default_rng(42)

WEEKDAY_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _doy_dates(doy_arr):
    """Convert 1-366 doy to plot dates using year 2000 as reference."""
    base = pd.Timestamp("2000-01-01")
    return [base + pd.Timedelta(days=int(d) - 1) for d in doy_arr]


def _setup_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)
    ax.axhline(100.0, color=GREY, linewidth=0.9, zorder=0)
    ax.set_ylabel("Relative births", fontsize=9)


def plot_row(axes, df, y_obs, comp, method_label):
    """Fill one row (4 axes) for a single method."""
    rel100  = 100.0 * np.exp(y_obs)
    dates   = df["date"].to_numpy()
    dow     = df["day_of_week"].to_numpy()
    doy_u   = np.arange(1, 367)
    doy_dates = _doy_dates(doy_u)

    # ── 0: Fitted mean ─────────────────────────────────────────────────────
    ax = axes[0]
    ax.scatter(dates, rel100, s=6, color=BLUE, alpha=0.12, rasterized=True)
    ax.fill_between(dates,
                    100*np.exp(comp["eta_lo"]), 100*np.exp(comp["eta_hi"]),
                    color=RED, alpha=0.35, linewidth=0)
    ax.plot(dates, 100*np.exp(comp["eta_m"]), color=RED, linewidth=0.8, alpha=0.9)
    ax.set_title(f"Fitted mean — {method_label}", fontsize=10)
    ax.set_xlabel("Date", fontsize=9)
    _setup_ax(ax)
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # ── 1: Slow trend ──────────────────────────────────────────────────────
    ax = axes[1]
    ax.scatter(dates, rel100, s=6, color=BLUE, alpha=0.12, rasterized=True)
    ax.fill_between(dates,
                    100*np.exp(comp["trend_lo"]), 100*np.exp(comp["trend_hi"]),
                    color=RED, alpha=0.30, linewidth=0)
    ax.plot(dates, 100*np.exp(comp["trend_m"]), color=RED, linewidth=1.6)
    ax.set_title(f"Slow trend — {method_label}", fontsize=10)
    ax.set_xlabel("Date", fontsize=9)
    _setup_ax(ax)
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # ── 2: Yearly seasonal ─────────────────────────────────────────────────
    ax = axes[2]
    ax.scatter(doy_dates, comp["doy_scatter"], s=18, color=BLUE,
               alpha=0.5, zorder=2)
    ax.fill_between(doy_dates,
                    100*np.exp(comp["sea_lo"]), 100*np.exp(comp["sea_hi"]),
                    color=RED, alpha=0.30, linewidth=0)
    ax.plot(doy_dates, 100*np.exp(comp["sea_m"]), color=RED, linewidth=1.6)
    ax.set_title(f"Yearly seasonal — {method_label}", fontsize=10)
    ax.set_xlabel("Month", fontsize=9)
    _setup_ax(ax)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)

    # ── 3: Day of week ─────────────────────────────────────────────────────
    ax = axes[3]
    jitter = RNG.normal(0.0, 0.08, len(dow))
    ax.scatter(dow + jitter, rel100, s=6, color=BLUE, alpha=0.10, rasterized=True)
    ax.fill_between(np.arange(7),
                    100*np.exp(comp["wday_lo"]), 100*np.exp(comp["wday_hi"]),
                    color=RED, alpha=0.40, linewidth=0)
    ax.plot(np.arange(7), 100*np.exp(comp["wday_m"]),
            color=RED, linewidth=1.8, marker="o", markersize=4)
    ax.set_title(f"Day of week — {method_label}", fontsize=10)
    ax.set_xlabel("Day", fontsize=9)
    _setup_ax(ax)
    ax.set_xticks(np.arange(7))
    ax.set_xticklabels(WEEKDAY_LABELS)
    ax.set_xlim(-0.5, 6.5)


def make_figure(df, y_obs, hsgp_comp, drv_comp, drv_label, out_path):
    """4-row × 2-col: rows=components, cols=methods."""
    fig, axes = plt.subplots(4, 2, figsize=(13, 16))
    fig.suptitle("Netherlands Daily Births 1995–2024\nModel decomposition: HSGP vs DeepRV",
                 fontsize=12, y=0.995)

    plot_row(axes[:, 0], df, y_obs, hsgp_comp, "HSGP NCP")
    plot_row(axes[:, 1], df, y_obs, drv_comp,  f"DeepRV {drv_label}")

    fig.tight_layout(rect=[0, 0, 1, 0.995])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")


def make_combined_figure(df, y_obs, hsgp_comp, drv_comp, drv_label, out_path):
    """4-row × 3-col: cols = Raw data | HSGP | DeepRV."""
    rel100    = 100.0 * np.exp(y_obs)
    dates     = df["date"].to_numpy()
    dow       = df["day_of_week"].to_numpy()
    doy       = df["day_of_year"].to_numpy()
    doy_u     = np.arange(1, 367)
    doy_dates = _doy_dates(doy_u)

    fig, axes = plt.subplots(4, 3, figsize=(18, 16))
    fig.suptitle("Netherlands Daily Births 1995–2024\nRaw data | HSGP NCP | DeepRV " + drv_label,
                 fontsize=12, y=0.999)

    col_titles = ["Raw data", "HSGP NCP", f"DeepRV {drv_label}"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=11, fontweight="bold", pad=8)

    s = pd.Series(rel100, index=pd.to_datetime(dates))
    Z = 1.96  # 95% CI multiplier

    def _sem_band(series, window, min_p):
        """Rolling mean ± 1.96*SEM."""
        m  = series.rolling(window, center=True, min_periods=min_p).mean()
        sd = series.rolling(window, center=True, min_periods=min_p).std()
        n  = series.rolling(window, center=True, min_periods=min_p).count()
        sem = sd / np.sqrt(n)
        return m, m - Z * sem, m + Z * sem

    # Rolling mean ± SEM (365-day) for row 0
    roll_m, roll_lo, roll_hi = _sem_band(s, 365, 30)

    # 3-year rolling ± SEM for row 1
    roll3_m, roll3_lo, roll3_hi = _sem_band(s, 3*365, 90)

    # Empirical doy mean ± SEM across years for row 2
    doy_df = pd.DataFrame({"rel100": rel100, "doy": doy})
    def _group_sem(g):
        m = g.mean(); sd = g.std(); n = g.count()
        return pd.Series({"m": m, "lo": m - Z*sd/n**0.5, "hi": m + Z*sd/n**0.5})
    doy_stats = doy_df.groupby("doy")["rel100"].apply(_group_sem).unstack().reindex(doy_u)

    # Empirical dow mean ± SEM for row 3
    dow_df = pd.DataFrame({"rel100": rel100, "dow": dow})
    dow_stats = dow_df.groupby("dow")["rel100"].apply(_group_sem).unstack()

    def _ts_ax(ax, x_idx, mean, lo, hi, scatter_x=None, scatter_y=None):
        if scatter_x is not None:
            ax.scatter(scatter_x, scatter_y, s=4, color=BLUE, alpha=0.12, rasterized=True)
        ax.fill_between(x_idx, lo, hi, color=RED, alpha=0.30, linewidth=0)
        ax.plot(x_idx, mean, color=RED, linewidth=1.4)
        _setup_ax(ax); ax.set_xlabel("Date", fontsize=9)
        ax.xaxis.set_major_locator(mdates.YearLocator(5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # ── Row 0: Fitted mean ────────────────────────────────────────────────────
    # Raw col: scatter + direct thin line (same dense oscillating style as model cols)
    ax = axes[0, 0]
    ax.scatter(dates, rel100, s=4, color=BLUE, alpha=0.12, rasterized=True)
    ax.plot(pd.to_datetime(dates), rel100, color=RED, linewidth=0.5, alpha=0.7)
    _setup_ax(ax); ax.set_xlabel("Date", fontsize=9)
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    for col, comp in enumerate([hsgp_comp, drv_comp], start=1):
        _ts_ax(axes[0, col], dates,
               100*np.exp(comp["eta_m"]),
               100*np.exp(comp["eta_lo"]), 100*np.exp(comp["eta_hi"]),
               dates, rel100)

    # ── Row 1: Slow trend ─────────────────────────────────────────────────────
    _ts_ax(axes[1, 0], roll3_m.index, roll3_m.values, roll3_lo.values, roll3_hi.values,
           dates, rel100)
    for col, comp in enumerate([hsgp_comp, drv_comp], start=1):
        _ts_ax(axes[1, col], dates,
               100*np.exp(comp["trend_m"]),
               100*np.exp(comp["trend_lo"]), 100*np.exp(comp["trend_hi"]),
               dates, rel100)

    # Detrend raw data using HSGP posterior means of trend + weekday,
    # then fit 10-harmonic Fourier to doy means → comparable to model seasonal
    wday_by_t    = hsgp_comp["wday_m"][dow]   # (T,) via dow index
    y_detrended  = y_obs - hsgp_comp["trend_m"] - wday_by_t
    rel_det      = 100.0 * np.exp(y_detrended)
    det_doy_mean = (pd.Series(rel_det, index=doy).groupby(level=0)
                      .mean().reindex(doy_u))
    _doy_u_rad   = 2.0 * np.pi * doy_u / 366.0
    _Phi_raw     = np.concatenate(
        [np.ones((366, 1)),
         np.cos(np.outer(_doy_u_rad, np.arange(1, 11))),
         np.sin(np.outer(_doy_u_rad, np.arange(1, 11)))], axis=1)
    _y_doy       = det_doy_mean.fillna(det_doy_mean.mean()).values
    _coef, *_    = np.linalg.lstsq(_Phi_raw, _y_doy, rcond=None)
    doy_smooth   = _Phi_raw @ _coef

    # ── Row 2: Yearly seasonal ────────────────────────────────────────────────
    def _sea_ax(ax, scatter_y, mean, lo, hi):
        ax.scatter(doy_dates, scatter_y, s=14, color=BLUE, alpha=0.5, zorder=2)
        ax.fill_between(doy_dates, lo, hi, color=RED, alpha=0.30, linewidth=0)
        ax.plot(doy_dates, mean, color=RED, linewidth=1.6)
        _setup_ax(ax); ax.set_xlabel("Month", fontsize=9)
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

    # Raw col: detrended+deweekdayed doy scatter + Fourier smooth (no CI)
    _sea_ax(axes[2, 0], det_doy_mean.values,
            doy_smooth, doy_smooth, doy_smooth)
    for col, comp in enumerate([hsgp_comp, drv_comp], start=1):
        _sea_ax(axes[2, col], comp["doy_scatter"],
                100*np.exp(comp["sea_m"]),
                100*np.exp(comp["sea_lo"]), 100*np.exp(comp["sea_hi"]))

    # ── Row 3: Day of week ────────────────────────────────────────────────────
    def _dow_ax(ax, scatter_dow, mean, lo, hi):
        jitter = RNG.normal(0, 0.08, len(scatter_dow))
        ax.scatter(scatter_dow + jitter, rel100, s=4, color=BLUE, alpha=0.08, rasterized=True)
        ax.fill_between(np.arange(7), lo, hi, color=RED, alpha=0.40, linewidth=0)
        ax.plot(np.arange(7), mean, color=RED, linewidth=1.8, marker="o", markersize=4)
        _setup_ax(ax); ax.set_xlabel("Day", fontsize=9)
        ax.set_xticks(np.arange(7)); ax.set_xticklabels(WEEKDAY_LABELS); ax.set_xlim(-0.5, 6.5)

    _dow_ax(axes[3, 0], dow,
            dow_stats["m"].values, dow_stats["lo"].values, dow_stats["hi"].values)
    for col, comp in enumerate([hsgp_comp, drv_comp], start=1):
        _dow_ax(axes[3, col], dow,
                100*np.exp(comp["wday_m"]),
                100*np.exp(comp["wday_lo"]), 100*np.exp(comp["wday_hi"]))

    # Titles
    row_titles = ["Fitted mean", "Slow trend", "Yearly seasonal", "Day of week"]
    raw_sub    = ["Raw data (observed)", "Rolling mean (3yr)",
                  "Day-of-year mean", "Day-of-week mean"]
    for row in range(4):
        axes[row, 0].set_title(raw_sub[row], fontsize=10, pad=4)
        for col in range(1, 3):
            axes[row, col].set_title(row_titles[row], fontsize=10, pad=4)

    fig.tight_layout(rect=[0, 0, 1, 0.997])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    df, y_obs, Phi_tr, omega_tr, Phi_sea, W, T = load_data()
    print(f"Netherlands births T={T}  ({df['year'].min()}–{df['year'].max()})")
    print(f"warmup={NUM_WARMUP}  samples={NUM_SAMPLES}  1 chain")

    mode_trend  = float(TREND_IG_B  / (TREND_IG_A  + 1.0))
    mode_season = float(SEASON_IG_B / (SEASON_IG_A + 1.0))

    # ── HSGP ─────────────────────────────────────────────────────────────────
    hsgp_md = dict(
        y=jnp.asarray(y_obs), Phi_trend=jnp.asarray(Phi_tr),
        omega_trend=jnp.asarray(omega_tr), Phi_season=jnp.asarray(Phi_sea),
        W_weekday=jnp.asarray(W),
        trend_ig_a=float(TREND_IG_A), trend_ig_b=float(TREND_IG_B),
        season_ig_a=float(SEASON_IG_A), season_ig_b=float(SEASON_IG_B),
    )
    hsgp_init = {
        "beta0": jnp.array(0.0), "sigma": jnp.array(0.1),
        "w_raw": jnp.zeros(6),
        "trend_alpha": jnp.array(0.1), "trend_length": jnp.array(mode_trend),
        "season_alpha": jnp.array(0.1), "season_length": jnp.array(mode_season),
        "z_trend": jnp.zeros(TREND_BASIS), "z_season": jnp.zeros(2*SEASON_FOURIER),
    }
    print("\n── HSGP NCP ──")
    hsgp_mcmc, hsgp_t = run_mcmc(hsgp_ncp_model, hsgp_md, SEED, init_vals=hsgp_init)
    hsgp_flat = {k: np.asarray(v) for k, v in hsgp_mcmc.get_samples().items()}
    print(f"  done {hsgp_t:.0f}s")

    # ── DeepRV ───────────────────────────────────────────────────────────────
    drv_z, drv_h, drv_file, drv_af, drv_lf = _DRV_CONFIGS[DRV_MODEL]
    drv_params    = load_deeprv_params(MODEL_DIR / drv_file)
    drv_model_obj = MLPDeepRV(drv_h + [T])
    drv_fn        = make_deeprv_ncp_model(drv_model_obj, drv_params, drv_z, drv_af, drv_lf)

    drv_md = dict(
        y=jnp.asarray(y_obs), Phi_season=jnp.asarray(Phi_sea),
        W_weekday=jnp.asarray(W),
        season_ig_a=float(SEASON_IG_A), season_ig_b=float(SEASON_IG_B),
    )
    drv_init = {
        "beta0": jnp.array(0.0), "sigma": jnp.array(0.1),
        "w_raw": jnp.zeros(6),
        "season_alpha": jnp.array(0.1), "season_length": jnp.array(mode_season),
        "deeprv_z": jnp.zeros(drv_z), "z_season": jnp.zeros(2*SEASON_FOURIER),
    }
    print(f"\n── DeepRV {DRV_MODEL} (α_f={drv_af}, ℓ_f={drv_lf}) ──")
    drv_mcmc, drv_t = run_mcmc(drv_fn, drv_md, SEED + 999, init_vals=drv_init)
    drv_flat = {k: np.asarray(v) for k, v in drv_mcmc.get_samples().items()}
    print(f"  done {drv_t:.0f}s")

    # ── Extract components ────────────────────────────────────────────────────
    print("\n── Extracting components ──")
    hsgp_comp = extract_components(hsgp_flat, df, y_obs)
    drv_comp  = extract_components(drv_flat,  df, y_obs)

    # Save component summaries so re-plotting doesn't need MCMC
    npz_dir = OUT_DIR / "nl_births_hsgp_deeprv_ncp_v3"
    npz_dir.mkdir(parents=True, exist_ok=True)
    np.savez(npz_dir / "hsgp_components.npz", **hsgp_comp)
    np.savez(npz_dir / f"drv_{DRV_MODEL}_components.npz", **drv_comp)
    print(f"  components saved to {npz_dir}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig_path  = FIGURE_DIR / f"nl_births_components_hsgp_vs_deeprv_{DRV_MODEL}.png"
    comb_path = FIGURE_DIR / f"nl_births_combined_raw_hsgp_deeprv_{DRV_MODEL}.png"
    make_figure(df, y_obs, hsgp_comp, drv_comp, DRV_MODEL, fig_path)
    make_combined_figure(df, y_obs, hsgp_comp, drv_comp, DRV_MODEL, comb_path)


if __name__ == "__main__":
    main()
