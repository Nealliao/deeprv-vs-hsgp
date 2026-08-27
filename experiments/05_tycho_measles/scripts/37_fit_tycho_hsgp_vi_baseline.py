from pathlib import Path
import os
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.contrib.hsgp.laplacian import eigenfunctions
from numpyro.contrib.hsgp.spectral_densities import (
    diag_spectral_density_squared_exponential,
)
from numpyro.infer import SVI, Trace_ELBO, Predictive
from numpyro.infer.autoguide import AutoDiagonalNormal
from numpyro.infer.initialization import init_to_value
from numpyro.optim import Adam


DATA_DIR = PROJECT_ROOT / "data" / "processed"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"

START_MONTH = os.environ.get("TYCHO_HSGP_START_MONTH", "1935-01-01")
END_MONTH = os.environ.get("TYCHO_HSGP_END_MONTH", "1968-12-01")

TIME_ELL = float(os.environ.get("TYCHO_HSGP_TIME_ELL", "240.0"))
TIME_M = int(os.environ.get("TYCHO_HSGP_TIME_M", "40"))
INTERACTION_TIME_ELL = float(os.environ.get("TYCHO_HSGP_INTERACTION_TIME_ELL", "240.0"))
INTERACTION_TIME_M = int(os.environ.get("TYCHO_HSGP_INTERACTION_TIME_M", "18"))
LIKELIHOOD = os.environ.get("TYCHO_HSGP_LIKELIHOOD", "negative_binomial").lower()


def hsgp_se(name, x, alpha, length, ell, m):
    dim = jnp.shape(x)[-1] if jnp.ndim(x) > 1 else 1
    phi = eigenfunctions(x=x, ell=ell, m=m)
    spd = jnp.sqrt(
        diag_spectral_density_squared_exponential(
            alpha=alpha,
            length=length,
            ell=ell,
            m=m,
            dim=dim,
        )
    )
    beta = numpyro.sample(
        f"{name}_basis_weights",
        dist.Normal(0.0, 1.0).expand([phi.shape[-1]]),
    )
    return phi @ (spd * beta)


def separable_interaction_hsgp(
    state_index,
    time_index,
    coords_centered,
    time_centered,
    alpha,
    space_length,
    time_length,
):
    phi_space = eigenfunctions(x=coords_centered, ell=[0.8, 0.8], m=[5, 5])
    spd_space = jnp.sqrt(
        diag_spectral_density_squared_exponential(
            alpha=1.0,
            length=space_length,
            ell=[0.8, 0.8],
            m=[5, 5],
            dim=2,
        )
    )
    phi_time = eigenfunctions(
        x=time_centered,
        ell=INTERACTION_TIME_ELL,
        m=INTERACTION_TIME_M,
    )
    spd_time = jnp.sqrt(
        diag_spectral_density_squared_exponential(
            alpha=1.0,
            length=time_length,
            ell=INTERACTION_TIME_ELL,
            m=INTERACTION_TIME_M,
            dim=1,
        )
    )
    weighted_space = phi_space[state_index] * spd_space
    weighted_time = phi_time[time_index] * spd_time
    basis_count = weighted_space.shape[-1] * weighted_time.shape[-1]
    beta = numpyro.sample(
        "interaction_basis_weights",
        dist.Normal(0.0, 1.0).expand([basis_count]),
    ).reshape((weighted_space.shape[-1], weighted_time.shape[-1]))
    return alpha * jnp.einsum("ns,nt,st->n", weighted_space, weighted_time, beta)


def model(
    count,
    log_state_exposure,
    season_sin,
    season_cos,
    state_index,
    time_index,
    coords_centered,
    time_centered,
):
    beta0 = numpyro.sample("beta0", dist.Normal(0.0, 0.5))
    beta_sin = numpyro.sample("beta_sin", dist.Normal(0.0, 0.8))
    beta_cos = numpyro.sample("beta_cos", dist.Normal(0.0, 0.8))

    space_raw = hsgp_se(
        "space",
        coords_centered,
        alpha=0.35,
        length=0.35,
        ell=[0.8, 0.8],
        m=[8, 8],
    )
    time_raw = hsgp_se(
        "time",
        time_centered,
        alpha=1.2,
        length=24.0,
        ell=TIME_ELL,
        m=TIME_M,
    )
    interaction_raw = separable_interaction_hsgp(
        state_index,
        time_index,
        coords_centered,
        time_centered,
        alpha=0.35,
        space_length=0.35,
        time_length=30.0,
    )

    space_effect = space_raw - jnp.mean(space_raw)
    time_effect = time_raw - jnp.mean(time_raw)
    interaction_effect = interaction_raw - jnp.mean(interaction_raw)
    latent_f = space_effect[state_index] + time_effect[time_index] + interaction_effect

    eta = (
        log_state_exposure
        + beta0
        + beta_sin * season_sin
        + beta_cos * season_cos
        + latent_f
    )
    eta = jnp.clip(eta, -20.0, 16.0)
    rate = jnp.exp(eta)
    numpyro.deterministic("latent_f", latent_f)
    numpyro.deterministic("rate", rate)
    if LIKELIHOOD == "negative_binomial":
        nb_concentration = numpyro.sample("nb_concentration", dist.LogNormal(jnp.log(20.0), 0.8))
        numpyro.sample(
            "obs",
            dist.NegativeBinomial2(mean=rate, concentration=nb_concentration),
            obs=count,
        )
    elif LIKELIHOOD == "poisson":
        numpyro.sample("obs", dist.Poisson(rate=rate), obs=count)
    else:
        raise ValueError(f"Unknown likelihood: {LIKELIHOOD}")


def prepare_data():
    panel = pd.read_csv(DATA_DIR / "tycho_measles_state_monthly_panel.csv", parse_dates=["month"])
    panel = panel[
        (panel["month"] >= START_MONTH)
        & (panel["month"] <= END_MONTH)
    ].copy()
    panel = panel.sort_values(["state_index", "time_index"]).reset_index(drop=True)
    panel["model_time_index"] = panel.groupby("state_index").cumcount()
    month_angle = 2.0 * np.pi * (panel["month_of_year"].to_numpy() - 1.0) / 12.0
    panel["season_sin"] = np.sin(month_angle)
    panel["season_cos"] = np.cos(month_angle)

    state_mean_cases = panel.groupby("state_index")["cases"].transform("mean")
    panel["state_exposure"] = state_mean_cases + 0.1
    panel["log_state_exposure"] = np.log(panel["state_exposure"])

    regions = panel.drop_duplicates("state_index").sort_values("state_index")
    coords = regions[["x_coord", "y_coord"]].to_numpy()
    coords_centered = coords - 0.5
    n_months = panel["model_time_index"].nunique()
    time_values = np.arange(n_months, dtype=float)[:, None]
    time_centered = time_values - time_values.mean()

    data = {
        "count": jnp.asarray(panel["cases"].to_numpy(dtype=np.int32)),
        "log_state_exposure": jnp.asarray(
            panel["log_state_exposure"].to_numpy(dtype=np.float32)
        ),
        "season_sin": jnp.asarray(panel["season_sin"].to_numpy(dtype=np.float32)),
        "season_cos": jnp.asarray(panel["season_cos"].to_numpy(dtype=np.float32)),
        "state_index": jnp.asarray(panel["state_index"].to_numpy(dtype=np.int32)),
        "time_index": jnp.asarray(panel["model_time_index"].to_numpy(dtype=np.int32)),
        "coords_centered": jnp.asarray(coords_centered.astype(np.float32)),
        "time_centered": jnp.asarray(time_centered.astype(np.float32)),
    }
    return panel, data


def run_vi(data):
    steps = int(os.environ.get("TYCHO_HSGP_VI_STEPS", "12000"))
    learning_rate = float(os.environ.get("TYCHO_HSGP_VI_LR", "0.003"))
    init_values = {
        "beta0": jnp.asarray(0.0),
        "beta_sin": jnp.asarray(0.0),
        "beta_cos": jnp.asarray(0.0),
        "space_basis_weights": jnp.zeros(64),
        "time_basis_weights": jnp.zeros(TIME_M),
        "interaction_basis_weights": jnp.zeros(25 * INTERACTION_TIME_M),
    }
    if LIKELIHOOD == "negative_binomial":
        init_values["nb_concentration"] = jnp.asarray(20.0)
    guide = AutoDiagonalNormal(
        model,
        init_loc_fn=init_to_value(values=init_values),
        init_scale=0.03,
    )
    svi = SVI(model, guide, Adam(learning_rate), Trace_ELBO(num_particles=1))
    start = time.perf_counter()
    result = svi.run(
        jax.random.PRNGKey(20260614),
        steps,
        **data,
        progress_bar=True,
        stable_update=True,
    )
    elapsed = time.perf_counter() - start
    return guide, result, elapsed


def draw_posterior(guide, params, data):
    draws = int(os.environ.get("TYCHO_HSGP_POSTERIOR_DRAWS", "400"))
    predictive = Predictive(
        model,
        guide=guide,
        params=params,
        num_samples=draws,
        return_sites=(
            "latent_f",
            "rate",
            "obs",
            "beta0",
            "beta_sin",
            "beta_cos",
            "nb_concentration",
        ),
    )
    prediction_data = dict(data)
    prediction_data["count"] = None
    return predictive(jax.random.PRNGKey(20260615), **prediction_data)


def summarize(panel, samples, elapsed, losses):
    latent_samples = np.asarray(samples["latent_f"])
    rate_samples = np.asarray(samples["rate"])
    predictive_counts = np.asarray(samples["obs"])
    latent_mean = latent_samples.mean(axis=0)
    latent_sd = latent_samples.std(axis=0)
    rate_mean = rate_samples.mean(axis=0)
    rate_low = np.quantile(rate_samples, 0.025, axis=0)
    rate_high = np.quantile(rate_samples, 0.975, axis=0)
    predictive_low = np.quantile(predictive_counts, 0.025, axis=0)
    predictive_high = np.quantile(predictive_counts, 0.975, axis=0)

    recovery = panel[
        [
            "state",
            "state_name",
            "state_index",
            "month",
            "model_time_index",
            "cases",
            "observed_report",
            "reporting_weeks",
            "log_state_exposure",
            "season_sin",
            "season_cos",
            "longitude",
            "latitude",
            "x_coord",
            "y_coord",
        ]
    ].copy()
    recovery = recovery.rename(columns={"model_time_index": "time_index"})
    recovery["latent_f_hsgp_vi_mean"] = latent_mean
    recovery["latent_f_hsgp_vi_sd"] = latent_sd
    recovery["expected_cases_hsgp_vi_mean"] = rate_mean
    recovery["expected_cases_hsgp_vi_low_95"] = rate_low
    recovery["expected_cases_hsgp_vi_high_95"] = rate_high
    recovery["predictive_cases_hsgp_vi_low_95"] = predictive_low
    recovery["predictive_cases_hsgp_vi_high_95"] = predictive_high
    recovery["pearson_residual"] = (recovery["cases"] - rate_mean) / np.sqrt(rate_mean + 1e-6)

    rmse_count = np.sqrt(np.mean((rate_mean - recovery["cases"].to_numpy()) ** 2))
    mae_count = np.mean(np.abs(rate_mean - recovery["cases"].to_numpy()))
    interval_coverage_count = np.mean(
        (recovery["cases"] >= recovery["expected_cases_hsgp_vi_low_95"])
        & (recovery["cases"] <= recovery["expected_cases_hsgp_vi_high_95"])
    )
    predictive_coverage_count = np.mean(
        (recovery["cases"] >= recovery["predictive_cases_hsgp_vi_low_95"])
        & (recovery["cases"] <= recovery["predictive_cases_hsgp_vi_high_95"])
    )
    metrics = pd.DataFrame(
        [
            {
                "dataset": "tycho_measles",
                "method": "hsgp_vi_baseline",
                "likelihood": LIKELIHOOD,
                "start_month": START_MONTH,
                "end_month": END_MONTH,
                "n_states": recovery["state"].nunique(),
                "n_months": recovery["time_index"].nunique(),
                "n_observations": len(recovery),
                "runtime_seconds": elapsed,
                "vi_steps": len(losses),
                "final_elbo_loss": float(losses[-1]),
                "total_cases": int(recovery["cases"].sum()),
                "mean_monthly_cases": float(recovery["cases"].mean()),
                "zero_fraction": float((recovery["cases"] == 0).mean()),
                "reporting_fraction": float(recovery["observed_report"].mean()),
                "rmse_count_mean": float(rmse_count),
                "mae_count_mean": float(mae_count),
                "expected_count_interval_coverage_95": float(interval_coverage_count),
                "predictive_count_interval_coverage_95": float(predictive_coverage_count),
                "mean_latent_sd": float(latent_sd.mean()),
                "beta0_mean": float(np.asarray(samples["beta0"]).mean()),
                "beta_sin_mean": float(np.asarray(samples["beta_sin"]).mean()),
                "beta_cos_mean": float(np.asarray(samples["beta_cos"]).mean()),
                "nb_concentration_mean": float(
                    np.asarray(samples.get("nb_concentration", np.asarray([np.nan]))).mean()
                ),
                "time_ell": TIME_ELL,
                "time_m": TIME_M,
                "interaction_time_ell": INTERACTION_TIME_ELL,
                "interaction_time_m": INTERACTION_TIME_M,
            }
        ]
    )
    return recovery, metrics


def plot_recovery(recovery, output_path):
    cases = recovery.pivot(index="state_index", columns="time_index", values="cases")
    mean = recovery.pivot(
        index="state_index",
        columns="time_index",
        values="expected_cases_hsgp_vi_mean",
    )
    latent = recovery.pivot(
        index="state_index",
        columns="time_index",
        values="latent_f_hsgp_vi_mean",
    )
    residual = recovery.pivot(index="state_index", columns="time_index", values="pearson_residual")

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    image0 = axes[0, 0].imshow(np.log1p(cases), aspect="auto", cmap="viridis")
    axes[0, 0].set_title("Observed log monthly cases")
    fig.colorbar(image0, ax=axes[0, 0], fraction=0.046)

    image1 = axes[0, 1].imshow(np.log1p(mean), aspect="auto", cmap="viridis")
    axes[0, 1].set_title("Posterior mean log expected cases")
    fig.colorbar(image1, ax=axes[0, 1], fraction=0.046)

    image2 = axes[1, 0].imshow(latent, aspect="auto", cmap="coolwarm")
    axes[1, 0].set_title("HSGP latent residual risk")
    fig.colorbar(image2, ax=axes[1, 0], fraction=0.046)

    bound = np.nanquantile(np.abs(residual.to_numpy()), 0.98)
    image3 = axes[1, 1].imshow(
        residual,
        aspect="auto",
        cmap="PiYG",
        vmin=-bound,
        vmax=bound,
    )
    axes[1, 1].set_title("Pearson residual")
    fig.colorbar(image3, ax=axes[1, 1], fraction=0.046)

    for axis in axes.ravel():
        axis.set_xlabel("Month index")
        axis.set_ylabel("State index")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    panel, data = prepare_data()
    guide, result, elapsed = run_vi(data)
    losses = np.asarray(result.losses)
    samples = draw_posterior(guide, result.params, data)
    recovery, metrics = summarize(panel, samples, elapsed, losses)

    suffix = f"1935_1968_{LIKELIHOOD}"
    recovery_path = DATA_DIR / f"tycho_hsgp_vi_recovery_{suffix}.csv"
    metrics_path = DATA_DIR / f"tycho_hsgp_vi_metrics_{suffix}.csv"
    loss_path = DATA_DIR / f"tycho_hsgp_vi_loss_{suffix}.csv"
    figure_path = FIGURE_DIR / f"tycho_hsgp_vi_recovery_{suffix}.png"

    recovery.to_csv(recovery_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    pd.DataFrame({"step": np.arange(1, len(losses) + 1), "loss": losses}).to_csv(
        loss_path,
        index=False,
    )
    plot_recovery(recovery, figure_path)

    print(metrics.round(4).to_string(index=False))
    print(recovery_path)
    print(metrics_path)
    print(loss_path)
    print(figure_path)


if __name__ == "__main__":
    main()
