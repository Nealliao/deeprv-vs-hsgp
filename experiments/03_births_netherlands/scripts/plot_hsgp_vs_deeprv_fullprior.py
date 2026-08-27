"""Component decomposition: HSGP vs full-prior DeepRV z30h16 (amplitude factoring).
Reuses 118d's extract_components / make_combined_figure / make_figure."""
import importlib.util, os, sys
from pathlib import Path
PR = "/Users/Zhuanz/Documents/Codex/2026-05-19/files-mentioned-by-the-user-research"
sys.path.insert(0, PR)
os.environ.setdefault("MPLCONFIGDIR", str(Path(PR) / ".matplotlib"))
os.environ.setdefault("NL_WARMUP", "1000")
os.environ.setdefault("NL_SAMPLES", "1000")

import numpy as np, jax, jax.numpy as jnp
import numpyro, numpyro.distributions as dist

_spec = importlib.util.spec_from_file_location(
    "d", f"{PR}/scripts/118d_netherlands_births_plot_components.py")
d = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(d)
from dl4bi.vae.deep_rv import MLPDeepRV

MODEL_DIR = Path(PR) / "data/processed/netherlands_births/models"
FIG_DIR   = Path(PR) / "outputs/figures"


def make_fullprior_ampfact_model(model_obj, params, z_dim):
    """Full-prior + amplitude factoring, with component deterministics for plotting."""
    def model(y, Phi_season, W_weekday, trend_ig_a, trend_ig_b,
              season_ig_a, season_ig_b, **_kw):
        beta0         = numpyro.sample("beta0",         dist.Normal(0.0, 1.0))
        sigma         = numpyro.sample("sigma",         dist.HalfNormal(1.0))
        w_raw         = numpyro.sample("w_raw",         dist.Normal(0.0, 1.0).expand([6]))
        trend_alpha   = numpyro.sample("trend_alpha",   dist.HalfNormal(1.0))
        trend_length  = numpyro.sample("trend_length",  dist.InverseGamma(trend_ig_a, trend_ig_b))
        season_alpha  = numpyro.sample("season_alpha",  dist.HalfNormal(1.0))
        season_length = numpyro.sample("season_length", dist.InverseGamma(season_ig_a, season_ig_b))

        u        = numpyro.sample("deeprv_z", dist.Normal(0.0, 1.0).expand([z_dim]))
        z_season = numpyro.sample("z_season", dist.Normal(0.0, 1.0).expand([2*d.SEASON_FOURIER]))

        cond    = jnp.array([jnp.log(trend_length)])
        f_unit  = model_obj.apply({"params": params}, u[None, :], cond, method="decode")[0]
        f_unit  = f_unit - jnp.mean(f_unit)
        f_trend = trend_alpha * f_unit

        D_sea      = d.fourier_scale_harmonics(season_alpha, season_length, 1, d.SEASON_FOURIER)
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


def main():
    df, y_obs, Phi_tr, omega_tr, Phi_sea, W, T = d.load_data()
    mode_trend  = float(d.TREND_IG_B  / (d.TREND_IG_A  + 1.0))
    mode_season = float(d.SEASON_IG_B / (d.SEASON_IG_A + 1.0))

    # ── HSGP ──
    hsgp_md = dict(y=jnp.asarray(y_obs), Phi_trend=jnp.asarray(Phi_tr),
                   omega_trend=jnp.asarray(omega_tr), Phi_season=jnp.asarray(Phi_sea),
                   W_weekday=jnp.asarray(W),
                   trend_ig_a=float(d.TREND_IG_A), trend_ig_b=float(d.TREND_IG_B),
                   season_ig_a=float(d.SEASON_IG_A), season_ig_b=float(d.SEASON_IG_B))
    hsgp_init = {"beta0": jnp.array(0.0), "sigma": jnp.array(0.1), "w_raw": jnp.zeros(6),
                 "trend_alpha": jnp.array(0.1), "trend_length": jnp.array(mode_trend),
                 "season_alpha": jnp.array(0.1), "season_length": jnp.array(mode_season),
                 "z_trend": jnp.zeros(d.TREND_BASIS), "z_season": jnp.zeros(2*d.SEASON_FOURIER)}
    print("── HSGP ──")
    hsgp_mcmc, _ = d.run_mcmc(d.hsgp_ncp_model, hsgp_md, d.SEED, init_vals=hsgp_init)
    hsgp_flat = {k: np.asarray(v) for k, v in hsgp_mcmc.get_samples().items()}

    # ── DeepRV z30h16 ampfact full-prior ──
    params = d.load_deeprv_params(MODEL_DIR / "nl_births_deeprv_z30_h16_ampfact.npz")
    dobj   = MLPDeepRV([16, 16, T])
    drv_fn = make_fullprior_ampfact_model(dobj, params, 30)
    drv_md = dict(y=jnp.asarray(y_obs), Phi_season=jnp.asarray(Phi_sea), W_weekday=jnp.asarray(W),
                  trend_ig_a=float(d.TREND_IG_A), trend_ig_b=float(d.TREND_IG_B),
                  season_ig_a=float(d.SEASON_IG_A), season_ig_b=float(d.SEASON_IG_B))
    drv_init = {"beta0": jnp.array(0.0), "sigma": jnp.array(0.1), "w_raw": jnp.zeros(6),
                "trend_alpha": jnp.array(0.3), "trend_length": jnp.array(0.3),
                "season_alpha": jnp.array(0.1), "season_length": jnp.array(mode_season),
                "deeprv_z": jnp.zeros(30), "z_season": jnp.zeros(2*d.SEASON_FOURIER)}
    print("── DeepRV z30h16 ampfact (full-prior) ──")
    drv_mcmc, _ = d.run_mcmc(drv_fn, drv_md, d.SEED + 999, init_vals=drv_init)
    drv_flat = {k: np.asarray(v) for k, v in drv_mcmc.get_samples().items()}

    print("── extract + plot ──")
    hsgp_comp = d.extract_components(hsgp_flat, df, y_obs)
    drv_comp  = d.extract_components(drv_flat,  df, y_obs)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    label_file  = "z30h16_ampfact_fullprior"   # keeps filenames disambiguated
    label_title = "fullprior"                  # short label shown in figure titles
    d.make_figure(df, y_obs, hsgp_comp, drv_comp, label_title,
                  FIG_DIR / f"nl_births_components_hsgp_vs_deeprv_{label_file}.png")
    d.make_combined_figure(df, y_obs, hsgp_comp, drv_comp, label_title,
                           FIG_DIR / f"nl_births_combined_raw_hsgp_deeprv_{label_file}.png")
    print("DONE")


if __name__ == "__main__":
    main()
