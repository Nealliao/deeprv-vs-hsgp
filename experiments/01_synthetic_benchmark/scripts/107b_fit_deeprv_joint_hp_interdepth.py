"""
107b_fit_deeprv_joint_hp_interdepth.py
================================================================================
Same as script 107 (DeepRV, all 9 HP jointly sampled) but with a configurable
interaction-decoder DEPTH, to test whether a shallower (more linear) decoder
improves the joint-inference geometry: fewer divergences, shorter runtime.

Only the interaction decoder depth is varied (z=120, output=1440 — the dominant
divergence/runtime contributor). Space/time decoders stay at the production
depth-2 weights. The interaction decoders come from script 108 (same training
setup across depths, so the comparison isolates depth from training noise).

The judge is the JOINT 9-HP inference itself (divergences, runtime, latent-f
recovery, HP recovery) — NOT the fixed-ell prior fidelity from script 108,
because ell is sampled here and NUTS differentiates through the decoder w.r.t.
log_ell.

Env vars:
  CDRV9B_INTER_DEPTH   which inter decoder depth to load (default 1)
  CDRV9B_NUM_WARMUP, CDRV9B_NUM_SAMPLES, CDRV9B_NUM_CHAINS,
  CDRV9B_SEED_BASE, CDRV9B_TARGET_ACCEPT, CDRV9B_SKIP_EXISTING
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
OUTPUT_DIR = DATA_DIR / "formal_synthetic" / "conditional_deeprv_seasonal_depth"

INTER_DEPTH = int(os.environ.get("CDRV9B_INTER_DEPTH", "1"))
NUM_WARMUP  = int(os.environ.get("CDRV9B_NUM_WARMUP",  "1000"))
NUM_SAMPLES = int(os.environ.get("CDRV9B_NUM_SAMPLES", "1000"))
NUM_CHAINS  = int(os.environ.get("CDRV9B_NUM_CHAINS",  "4"))
SEED_BASE   = int(os.environ.get("CDRV9B_SEED_BASE",   "1"))
SKIP_EXISTING = os.environ.get("CDRV9B_SKIP_EXISTING", "1") == "1"
TARGET_ACCEPT  = float(os.environ.get("CDRV9B_TARGET_ACCEPT", "0.95"))
MAX_TREE_DEPTH = 10

SEASONAL_A       = 0.35
SEASONAL_M_STAR  = 2
TRUE_SIGMA_H     = 0.12
TRUE_ELL_H       = 0.75

_m12   = np.arange(12)
_diff  = np.abs(_m12[:, None] - _m12[None, :])
_b_h   = SEASONAL_A * np.cos(2 * np.pi * (_m12 - SEASONAL_M_STAR) / 12)
_b_h  -= _b_h.mean()
_K_h_fixed = TRUE_SIGMA_H**2 * np.exp(-2 * np.sin(np.pi * _diff / 12)**2 / TRUE_ELL_H**2)

TRUE_HP = {
    "space_alpha": 0.30, "space_length": 0.55,
    "time_alpha": 0.20, "time_length": 18.0,
    "interaction_alpha": 0.18,
    "interaction_space_length": 0.15, "interaction_time_length": 6.0,
    "sigma_h": TRUE_SIGMA_H, "ell_h": TRUE_ELL_H,
}
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
_HP_INIT = {
    "space_alpha": 0.30, "space_length": 0.467,
    "time_alpha": 0.20, "time_length": 17.28,
    "interaction_alpha": 0.18,
    "interaction_space_length": 0.057, "interaction_time_length": 3.62,
    "sigma_h": 0.20, "ell_h": 0.40,
}
HP_NAMES = ["space_alpha","space_length","time_alpha","time_length",
            "interaction_alpha","interaction_space_length","interaction_time_length",
            "sigma_h","ell_h"]


# ─── Decoder (configurable depth) ─────────────────────────────────────────────

class DepthDecoder(nn.Module):
    hidden_dim: int
    output_dim: int
    depth: int

    @nn.compact
    def __call__(self, z, log_ell):
        x = jnp.concatenate([z, log_ell], axis=-1)
        for _ in range(self.depth):
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
    depth      = int(bundle["depth"]) if "depth" in bundle.files else 2
    model = DepthDecoder(hidden_dim=hidden_dim, output_dim=output_dim, depth=depth)
    return model, params, z_dim, depth


def _periodic_kernel_jax(sigma_h, ell_h):
    m12 = jnp.arange(12, dtype=jnp.float32)
    diff = jnp.abs(m12[:, None] - m12[None, :])
    return sigma_h**2 * jnp.exp(-2.0 * jnp.sin(jnp.pi * diff / 12.0)**2 / ell_h**2)


# ─── DGP (identical to script 107) ────────────────────────────────────────────

def _se_cov(x, ell, var):
    d = cdist(x, x)
    return var * np.exp(-(d ** 2) / (2 * ell ** 2))


def _double_center(m):
    return (m - m.mean(axis=0, keepdims=True)
              - m.mean(axis=1, keepdims=True) + m.mean())


def generate_cyclic_seasonal(rng):
    delta = rng.multivariate_normal(np.zeros(12), _K_h_fixed + 1e-6 * np.eye(12))
    h_raw = _b_h + delta
    return h_raw - h_raw.mean()


def generate_panel(regime, regions, n_months, dgp_seed):
    rng    = np.random.default_rng(dgp_seed)
    coords = regions[["x_coord", "y_coord"]].to_numpy()
    time   = np.arange(n_months)[:, None]
    cov_s = _se_cov(coords, regime["space_length_scale"], regime["space_sd"] ** 2)
    cov_t = _se_cov(time,   regime["time_length_scale"],  regime["time_sd"] ** 2)
    g  = rng.multivariate_normal(np.zeros(len(coords)), cov_s + 1e-6 * np.eye(len(coords))); g -= g.mean()
    q  = rng.multivariate_normal(np.zeros(n_months), cov_t + 1e-6 * np.eye(n_months)); q -= q.mean()
    cov_ws = _se_cov(coords, regime["interaction_space_length_scale"], 1.0)
    cov_wt = _se_cov(time,   regime["interaction_time_length_scale"],  1.0)
    Ls = np.linalg.cholesky(cov_ws + 1e-6 * np.eye(len(coords)))
    Lt = np.linalg.cholesky(cov_wt + 1e-6 * np.eye(n_months))
    w = Ls @ rng.standard_normal((len(coords), n_months)) @ Lt.T
    w = _double_center(w); w = w / w.std() * regime["interaction_sd"]
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
                "state": sr["state"], "state_index": s, "month": month, "time_index": t,
                "population": int(sr["population"]),
                "log_population": float(np.log(sr["population"])),
                "seasonal_h_true": seasonal, "latent_f_true": float(latent_f),
                "count": int(rng.poisson(sr["population"] * np.exp(log_lam))),
            })
    return pd.DataFrame(rows).sort_values(["state_index","time_index"]).reset_index(drop=True)


def make_data(panel, regions):
    coords = regions[["x_coord", "y_coord"]].to_numpy()
    coords_c = coords - coords.mean(axis=0, keepdims=True)
    n_months = panel["time_index"].nunique()
    time_vals = np.arange(n_months, dtype=float)
    return {
        "count":           jnp.asarray(panel["count"].to_numpy(dtype=np.int32)),
        "log_population":  jnp.asarray(panel["log_population"].to_numpy(dtype=np.float32)),
        "b_h":             jnp.asarray(_b_h.astype(np.float32)),
        "state_index":     jnp.asarray(panel["state_index"].to_numpy(dtype=np.int32)),
        "time_index":      jnp.asarray(panel["time_index"].to_numpy(dtype=np.int32)),
        "coords_c":        jnp.asarray(coords_c.astype(np.float32)),
        "time_vals":       jnp.asarray(time_vals.astype(np.float32)),
        "beta0_prior_mean": float(np.log(1.2e-5)),
    }


def make_model(space_dec, space_params, z_s_dim,
               time_dec,  time_params,  z_t_dim,
               inter_dec, inter_params, z_w_dim):
    def model(count, log_population, b_h,
              state_index, time_index, coords_c, time_vals, beta0_prior_mean):
        beta0 = numpyro.sample("beta0", dist.Normal(beta0_prior_mean, 1.0))
        sigma_h = numpyro.sample("sigma_h", PRIORS["sigma_h"])
        ell_h   = numpyro.sample("ell_h",   PRIORS["ell_h"])
        K_h = _periodic_kernel_jax(sigma_h, ell_h) + 1e-6 * jnp.eye(12)
        L_h = jnp.linalg.cholesky(K_h)
        z_h   = numpyro.sample("z_h", dist.Normal(0.0, 1.0).expand([12]))
        h_raw = b_h + L_h @ z_h
        h_monthly = h_raw - h_raw.mean()

        sigma_s  = numpyro.sample("space_alpha",              PRIORS["space_alpha"])
        ell_s    = numpyro.sample("space_length",             PRIORS["space_length"])
        sigma_t  = numpyro.sample("time_alpha",               PRIORS["time_alpha"])
        ell_t    = numpyro.sample("time_length",              PRIORS["time_length"])
        sigma_w  = numpyro.sample("interaction_alpha",        PRIORS["interaction_alpha"])
        ell_ws   = numpyro.sample("interaction_space_length", PRIORS["interaction_space_length"])
        ell_wt   = numpyro.sample("interaction_time_length",  PRIORS["interaction_time_length"])

        z_s = numpyro.sample("z_space", dist.Normal(0.0, 1.0).expand([z_s_dim]))
        z_t = numpyro.sample("z_time",  dist.Normal(0.0, 1.0).expand([z_t_dim]))
        z_w = numpyro.sample("z_inter", dist.Normal(0.0, 1.0).expand([z_w_dim]))

        log_ell_s = jnp.array([[jnp.log(ell_s)]])
        log_ell_t = jnp.array([[jnp.log(ell_t)]])
        log_ell_w = jnp.array([[jnp.log(ell_ws), jnp.log(ell_wt)]])

        g_norm = space_dec.apply({"params": space_params}, z_s[None, :], log_ell_s)[0]
        q_norm = time_dec.apply({"params": time_params},  z_t[None, :], log_ell_t)[0]
        w_norm = inter_dec.apply({"params": inter_params}, z_w[None, :], log_ell_w)[0]

        g = sigma_s * g_norm; g = g - jnp.mean(g)
        q = sigma_t * q_norm; q = q - jnp.mean(q)
        w = sigma_w * w_norm; w = w - jnp.mean(w)

        S = coords_c.shape[0]; T = time_vals.shape[0]
        w_mat = w.reshape(S, T)
        latent_f = g[state_index] + q[time_index] + w_mat[state_index, time_index]
        h_t = h_monthly[time_index % 12]
        eta = log_population + beta0 + h_t + latent_f
        eta = jnp.clip(eta, -35.0, 20.0)
        numpyro.deterministic("latent_f", latent_f)
        numpyro.sample("obs", dist.Poisson(rate=jnp.exp(eta)), obs=count)
    return model


def run_nuts(model_fn, data, nuts_seed):
    numpyro.set_host_device_count(NUM_CHAINS)
    init_vals = {hp: _HP_INIT[hp] for hp in _HP_INIT}
    kernel = NUTS(model_fn, target_accept_prob=TARGET_ACCEPT,
                  max_tree_depth=MAX_TREE_DEPTH,
                  init_strategy=init_to_value(values=init_vals))
    mcmc = MCMC(kernel, num_warmup=NUM_WARMUP, num_samples=NUM_SAMPLES,
                num_chains=NUM_CHAINS, chain_method="sequential", progress_bar=True)
    t0 = time.perf_counter()
    mcmc.run(jax.random.PRNGKey(nuts_seed), **data,
             extra_fields=("accept_prob", "num_steps", "diverging", "potential_energy"))
    jax.block_until_ready(mcmc.get_samples())
    return mcmc, time.perf_counter() - t0


def compute_metrics(mcmc, panel, elapsed, dgp_seed, inter_depth):
    sbc = mcmc.get_samples(group_by_chain=True)
    flat = {k: np.asarray(v).reshape((-1,) + tuple(np.asarray(v).shape[2:])) for k, v in sbc.items()}
    latent = flat["latent_f"]
    mean_f = latent.mean(axis=0)
    ci_low = np.quantile(latent, 0.025, axis=0); ci_high = np.quantile(latent, 0.975, axis=0)
    f_true = panel["latent_f_true"].to_numpy()
    rmse = float(np.sqrt(np.mean((mean_f - f_true) ** 2)))
    coverage = float(((f_true >= ci_low) & (f_true <= ci_high)).mean())
    diag = numpyro_summary({k: sbc[k] for k in sbc if k != "latent_f"})
    rhats, esss = [], []
    for k in diag:
        rhats.extend(np.asarray(diag[k]["r_hat"]).flatten().tolist())
        esss.extend(np.asarray(diag[k]["n_eff"]).flatten().tolist())
    extra = mcmc.get_extra_fields()
    n_div = int(np.asarray(extra["diverging"]).sum())
    row = {
        "method": "conditional_deeprv_seasonal", "inter_depth": inter_depth,
        "dgp_seed": dgp_seed, "n_free_params": len(HP_NAMES),
        "rmse_f": rmse, "coverage_95": coverage,
        "max_rhat": float(np.nanmax(rhats)), "min_ess": float(np.nanmin(esss)),
        "n_divergences": n_div, "mean_accept_prob": float(np.asarray(extra["accept_prob"]).mean()),
        "runtime_seconds": elapsed, "ess_per_sec": float(np.nanmin(esss)) / elapsed,
        "latent_f_true": f_true.tolist(), "latent_f_post_mean": mean_f.tolist(),
        "latent_f_post_sd": np.std(latent, axis=0).tolist(),
    }
    for hp in HP_NAMES:
        if hp in flat:
            s = flat[hp]
            row[f"{hp}_post_mean"] = float(s.mean()); row[f"{hp}_post_sd"] = float(s.std())
            row[f"{hp}_post_q025"] = float(np.quantile(s, 0.025))
            row[f"{hp}_post_q975"] = float(np.quantile(s, 0.975))
            row[f"{hp}_truth"] = TRUE_HP[hp]; row[f"{hp}_bias"] = float(s.mean()) - TRUE_HP[hp]
    return row


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[load] decoders (inter depth={INTER_DEPTH}) …")
    space_dec, space_params, z_s_dim, _ = load_decoder(MODEL_DIR / "conditional_deeprv_space.npz")
    time_dec,  time_params,  z_t_dim, _ = load_decoder(MODEL_DIR / "conditional_deeprv_time.npz")
    inter_path = MODEL_DIR / f"conditional_deeprv_inter_depth{INTER_DEPTH}.npz"
    inter_dec, inter_params, z_w_dim, idepth = load_decoder(inter_path)
    print(f"  z_s={z_s_dim}  z_t={z_t_dim}  z_w={z_w_dim}  inter_decoder={inter_path.name} (depth={idepth})")

    regimes = pd.read_csv(DATA_DIR / "synthetic_measles_interaction_regimes.csv")
    regime  = regimes.loc[regimes["name"] == "interaction_smooth"].iloc[0].to_dict()
    regions = pd.read_csv(DATA_DIR / "synthetic_measles_regions.csv").sort_values("state_index").reset_index(drop=True)

    model_fn = make_model(space_dec, space_params, z_s_dim,
                          time_dec,  time_params,  z_t_dim,
                          inter_dec, inter_params, z_w_dim)

    dgp_seed = SEED_BASE
    out_path = OUTPUT_DIR / f"cdrv9_interdepth{INTER_DEPTH}_seed{dgp_seed:04d}.json"
    if SKIP_EXISTING and out_path.exists():
        print(f"[seed {dgp_seed}] skipping (exists)")
        return
    print(f"\n[seed {dgp_seed}] generating DGP …")
    panel = generate_panel(regime, regions, 72, dgp_seed)
    data  = make_data(panel, regions)
    nuts_seed = dgp_seed * 10_000 + 777
    print(f"[seed {dgp_seed}] running NUTS (nuts_seed={nuts_seed}) …")
    mcmc, elapsed = run_nuts(model_fn, data, nuts_seed)
    row = compute_metrics(mcmc, panel, elapsed, dgp_seed, INTER_DEPTH)
    print(f"\n  inter_depth={INTER_DEPTH}  RMSE={row['rmse_f']:.4f}  cover={row['coverage_95']:.4f}  "
          f"Rhat={row['max_rhat']:.4f}  ESS={row['min_ess']:.0f}  div={row['n_divergences']}  "
          f"t={elapsed:.1f}s  ESS/s={row['ess_per_sec']:.1f}  accept={row['mean_accept_prob']:.3f}")
    for hp in HP_NAMES:
        print(f"  {hp:28s} truth={TRUE_HP[hp]:7.3f}  post={row.get(f'{hp}_post_mean',float('nan')):7.3f}  "
              f"bias={row.get(f'{hp}_bias',float('nan')):+7.3f}")
    out_path.write_text(json.dumps(row, indent=2))
    print(f"\n[saved] {out_path.name}")


if __name__ == "__main__":
    main()
