"""Shared DGP + data prep + truth-metrics for the synthetic exact-GP / optimised
HSGP+DeepRV study. DGP is copied VERBATIM from the original script 106 so the
generated panel is byte-for-byte identical to the baseline runs (same seed ->
same data). Reads the two source CSVs (regions, regimes) copied into ../data/.
Nothing here writes outside this project folder."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

# ─── seasonal constants (identical to script 106) ─────────────────────────────
TRUE_SIGMA_H = 0.12
TRUE_ELL_H   = 0.75
SEASONAL_A      = 0.35
SEASONAL_M_STAR = 2
_m12  = np.arange(12)
_diff = np.abs(_m12[:, None] - _m12[None, :])
_K_h_fixed = TRUE_SIGMA_H**2 * np.exp(-2.0 * np.sin(np.pi * _diff / 12.0) ** 2 / TRUE_ELL_H**2)
_L_h_fixed = np.linalg.cholesky(_K_h_fixed + 1e-6 * np.eye(12))
_b_h = SEASONAL_A * np.cos(2.0 * np.pi * (_m12 - SEASONAL_M_STAR) / 12.0)
_b_h -= _b_h.mean()

TRUE_HP = {
    "space_alpha": 0.30, "space_length": 0.55,
    "time_alpha": 0.20, "time_length": 18.0,
    "interaction_alpha": 0.18, "interaction_space_length": 0.15, "interaction_time_length": 6.0,
    "sigma_h": TRUE_SIGMA_H, "ell_h": TRUE_ELL_H,
}
HP_NAMES = list(TRUE_HP.keys())

# ─── DGP (verbatim from script 106) ───────────────────────────────────────────
def se_cov(x, length_scale, variance):
    d = cdist(x, x)
    return variance * np.exp(-(d**2) / (2.0 * length_scale**2))

def centered_gp_draw(rng, cov):
    draw = rng.multivariate_normal(np.zeros(cov.shape[0]), cov + 1e-6 * np.eye(cov.shape[0]))
    return draw - draw.mean()

def double_center(m):
    return m - m.mean(0, keepdims=True) - m.mean(1, keepdims=True) + m.mean()

def sample_interaction(rng, coords, time, regime):
    cov_s = se_cov(coords, regime["interaction_space_length_scale"], 1.0)
    cov_t = se_cov(time, regime["interaction_time_length_scale"], 1.0)
    Ls = np.linalg.cholesky(cov_s + 1e-6 * np.eye(cov_s.shape[0]))
    Lt = np.linalg.cholesky(cov_t + 1e-6 * np.eye(cov_t.shape[0]))
    noise = rng.normal(size=(coords.shape[0], time.shape[0]))
    inter = double_center(Ls @ noise @ Lt.T)
    return inter / inter.std() * regime["interaction_sd"]

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
    g = centered_gp_draw(rng, cov_s)
    q = centered_gp_draw(rng, cov_t)
    w = sample_interaction(rng, coords, time, regime)
    h_monthly = generate_cyclic_seasonal(rng)
    months = pd.date_range("1961-01-01", periods=n_months, freq="MS")
    alpha = np.log(regime["baseline_rate_per_100k"] / 100_000)
    rows = []
    for _, sr in regions.iterrows():
        s = int(sr["state_index"])
        for t, _ in enumerate(months):
            seff = float(h_monthly[t % 12])
            latent = g[s] + q[t] + w[s, t]
            expected = sr["population"] * np.exp(alpha + seff + latent)
            rows.append({"state_index": s, "time_index": t,
                         "log_population": float(np.log(sr["population"])),
                         "seasonal_h_true": seff, "latent_f_true": float(latent),
                         "g_true": float(g[s]), "q_true": float(q[t]), "w_true": float(w[s, t]),
                         "count": int(rng.poisson(expected))})
    return pd.DataFrame(rows).sort_values(["state_index", "time_index"]).reset_index(drop=True)

def load_regime_regions():
    regimes = pd.read_csv(DATA / "synthetic_measles_interaction_regimes.csv")
    regime = regimes.loc[regimes["name"] == "interaction_smooth"].iloc[0].to_dict()
    regions = pd.read_csv(DATA / "synthetic_measles_regions.csv").sort_values("state_index").reset_index(drop=True)
    return regime, regions
