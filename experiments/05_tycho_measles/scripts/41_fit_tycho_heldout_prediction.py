from pathlib import Path
import os
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import gammaln, logsumexp

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.contrib.hsgp.laplacian import eigenfunctions
from numpyro.contrib.hsgp.spectral_densities import (
    diag_spectral_density_squared_exponential,
)
from numpyro.infer import Predictive, SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoDiagonalNormal
from numpyro.infer.initialization import init_to_value
from numpyro.optim import Adam


DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = DATA_DIR / "decoder_models"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"

START_MONTH = os.environ.get("TYCHO_HELDOUT_START_MONTH", "1935-01-01")
TRAIN_END_MONTH = os.environ.get("TYCHO_HELDOUT_TRAIN_END_MONTH", "1966-12-01")
TEST_START_MONTH = os.environ.get("TYCHO_HELDOUT_TEST_START_MONTH", "1967-01-01")
END_MONTH = os.environ.get("TYCHO_HELDOUT_END_MONTH", "1968-12-01")
SPLIT_NAME = os.environ.get("TYCHO_HELDOUT_SPLIT_NAME", "transition_1967_1968")

METHOD = os.environ.get("TYCHO_HELDOUT_METHOD", "hsgp").lower()
LATENT_DIM = int(os.environ.get("TYCHO_HELDOUT_LATENT_DIM", "120"))

TIME_ELL = float(os.environ.get("TYCHO_HSGP_TIME_ELL", "240.0"))
TIME_M = int(os.environ.get("TYCHO_HSGP_TIME_M", "40"))
INTERACTION_TIME_ELL = float(os.environ.get("TYCHO_HSGP_INTERACTION_TIME_ELL", "240.0"))
INTERACTION_TIME_M = int(os.environ.get("TYCHO_HSGP_INTERACTION_TIME_M", "18"))


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
    beta = numpyro.sample(
        "interaction_basis_weights",
        dist.Normal(0.0, 1.0).expand([weighted_space.shape[-1] * weighted_time.shape[-1]]),
    ).reshape((weighted_space.shape[-1], weighted_time.shape[-1]))
    return alpha * jnp.einsum("ns,nt,st->n", weighted_space, weighted_time, beta)


def hsgp_latent_f(state_index, time_index, coords_centered, time_centered):
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
    return (
        (space_raw - jnp.mean(space_raw))[state_index]
        + (time_raw - jnp.mean(time_raw))[time_index]
        + (interaction_raw - jnp.mean(interaction_raw))
    )


def model(
    count,
    train_mask,
    log_state_exposure,
    season_sin,
    season_cos,
    state_index=None,
    time_index=None,
    coords_centered=None,
    time_centered=None,
    decoder_weights=None,
    decoder_bias=None,
):
    beta0 = numpyro.sample("beta0", dist.Normal(0.0, 0.5))
    beta_sin = numpyro.sample("beta_sin", dist.Normal(0.0, 0.8))
    beta_cos = numpyro.sample("beta_cos", dist.Normal(0.0, 0.8))
    nb_concentration = numpyro.sample("nb_concentration", dist.LogNormal(jnp.log(20.0), 0.8))

    if METHOD == "hsgp":
        latent_f = hsgp_latent_f(state_index, time_index, coords_centered, time_centered)
    elif METHOD == "compressed":
        z_dim = decoder_weights.shape[0]
        z = numpyro.sample("decoder_z", dist.Normal(0.0, 1.0).expand([z_dim]))
        latent_f = z @ decoder_weights + decoder_bias
        latent_f = latent_f - jnp.mean(latent_f)
    else:
        raise ValueError(f"Unknown method: {METHOD}")

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
    observation = dist.NegativeBinomial2(mean=rate, concentration=nb_concentration)
    if count is None:
        numpyro.sample("obs", observation)
    else:
        with numpyro.handlers.mask(mask=train_mask):
            numpyro.sample("obs", observation, obs=count)


def prepare_panel():
    panel = pd.read_csv(DATA_DIR / "tycho_measles_state_monthly_panel.csv", parse_dates=["month"])
    panel = panel[
        (panel["month"] >= START_MONTH)
        & (panel["month"] <= END_MONTH)
    ].copy()
    panel = panel.sort_values(["state_index", "time_index"]).reset_index(drop=True)
    panel["model_time_index"] = panel.groupby("state_index").cumcount()
    panel["split"] = np.where(panel["month"] <= TRAIN_END_MONTH, "train", "test")
    month_angle = 2.0 * np.pi * (panel["month_of_year"].to_numpy() - 1.0) / 12.0
    panel["season_sin"] = np.sin(month_angle)
    panel["season_cos"] = np.cos(month_angle)

    train_means = (
        panel[panel["split"] == "train"]
        .groupby("state_index")["cases"]
        .mean()
        .rename("state_exposure")
    )
    panel = panel.merge(train_means, on="state_index", how="left")
    panel["state_exposure"] = panel["state_exposure"].fillna(panel["cases"].mean()) + 0.1
    panel["log_state_exposure"] = np.log(panel["state_exposure"])
    return panel


def prepare_data(panel):
    data = {
        "count": jnp.asarray(panel["cases"].to_numpy(dtype=np.int32)),
        "train_mask": jnp.asarray((panel["split"] == "train").to_numpy()),
        "log_state_exposure": jnp.asarray(
            panel["log_state_exposure"].to_numpy(dtype=np.float32)
        ),
        "season_sin": jnp.asarray(panel["season_sin"].to_numpy(dtype=np.float32)),
        "season_cos": jnp.asarray(panel["season_cos"].to_numpy(dtype=np.float32)),
    }
    if METHOD == "hsgp":
        regions = panel.drop_duplicates("state_index").sort_values("state_index")
        coords = regions[["x_coord", "y_coord"]].to_numpy() - 0.5
        n_months = panel["model_time_index"].nunique()
        time_values = np.arange(n_months, dtype=float)[:, None]
        time_centered = time_values - time_values.mean()
        data.update(
            {
                "state_index": jnp.asarray(panel["state_index"].to_numpy(dtype=np.int32)),
                "time_index": jnp.asarray(panel["model_time_index"].to_numpy(dtype=np.int32)),
                "coords_centered": jnp.asarray(coords.astype(np.float32)),
                "time_centered": jnp.asarray(time_centered.astype(np.float32)),
            }
        )
    elif METHOD == "compressed":
        decoder = np.load(MODEL_DIR / f"tycho_compressed_decoder_1935_1968_z{LATENT_DIM}.npz")
        full_n_months = 408
        decoder_indices = (
            panel["state_index"].to_numpy(dtype=int) * full_n_months
            + panel["model_time_index"].to_numpy(dtype=int)
        )
        weights = decoder["weights"].astype(np.float32)[:, decoder_indices]
        bias = decoder["bias"].astype(np.float32)[decoder_indices]
        data.update(
            {
                "decoder_weights": jnp.asarray(weights),
                "decoder_bias": jnp.asarray(bias),
            }
        )
    return data


def run_vi(data):
    steps = int(os.environ.get("TYCHO_HELDOUT_VI_STEPS", "8000"))
    learning_rate = float(os.environ.get("TYCHO_HELDOUT_VI_LR", "0.003"))
    init_values = {
        "beta0": jnp.asarray(0.0),
        "beta_sin": jnp.asarray(0.0),
        "beta_cos": jnp.asarray(0.0),
        "nb_concentration": jnp.asarray(20.0),
    }
    if METHOD == "hsgp":
        init_values.update(
            {
                "space_basis_weights": jnp.zeros(64),
                "time_basis_weights": jnp.zeros(TIME_M),
                "interaction_basis_weights": jnp.zeros(25 * INTERACTION_TIME_M),
            }
        )
    elif METHOD == "compressed":
        init_values["decoder_z"] = jnp.zeros(data["decoder_weights"].shape[0])

    guide = AutoDiagonalNormal(
        model,
        init_loc_fn=init_to_value(values=init_values),
        init_scale=0.03,
    )
    svi = SVI(model, guide, Adam(learning_rate), Trace_ELBO(num_particles=1))
    start = time.perf_counter()
    result = svi.run(
        jax.random.PRNGKey(20260620),
        steps,
        **data,
        progress_bar=True,
        stable_update=True,
    )
    return guide, result, time.perf_counter() - start


def draw_posterior(guide, params, data):
    draws = int(os.environ.get("TYCHO_HELDOUT_POSTERIOR_DRAWS", "400"))
    predictive = Predictive(
        model,
        guide=guide,
        params=params,
        num_samples=draws,
        return_sites=("latent_f", "rate", "obs", "nb_concentration"),
    )
    prediction_data = dict(data)
    prediction_data["count"] = None
    return predictive(jax.random.PRNGKey(20260621), **prediction_data)


def negative_binomial_log_score(counts, rate_samples, concentration_samples):
    y = counts[None, :]
    mu = np.maximum(rate_samples, 1e-8)
    r = np.maximum(concentration_samples[:, None], 1e-8)
    log_prob = (
        gammaln(y + r)
        - gammaln(r)
        - gammaln(y + 1)
        + r * (np.log(r) - np.log(r + mu))
        + y * (np.log(mu) - np.log(r + mu))
    )
    return logsumexp(log_prob, axis=0) - np.log(log_prob.shape[0])


def summarize(panel, samples, elapsed, losses):
    rate_samples = np.asarray(samples["rate"])
    predictive_counts = np.asarray(samples["obs"])
    concentration_samples = np.asarray(samples["nb_concentration"])
    latent_samples = np.asarray(samples["latent_f"])

    rate_mean = rate_samples.mean(axis=0)
    latent_mean = latent_samples.mean(axis=0)
    latent_sd = latent_samples.std(axis=0)
    pred_low = np.quantile(predictive_counts, 0.025, axis=0)
    pred_high = np.quantile(predictive_counts, 0.975, axis=0)

    recovery = panel[
        ["state", "state_name", "state_index", "month", "model_time_index", "cases", "split"]
    ].copy()
    recovery = recovery.rename(columns={"model_time_index": "time_index"})
    recovery["expected_cases_mean"] = rate_mean
    recovery["predictive_low_95"] = pred_low
    recovery["predictive_high_95"] = pred_high
    recovery["latent_mean"] = latent_mean
    recovery["latent_sd"] = latent_sd
    recovery["covered_95"] = (
        (recovery["cases"] >= recovery["predictive_low_95"])
        & (recovery["cases"] <= recovery["predictive_high_95"])
    )

    test = recovery["split"] == "test"
    test_counts = recovery.loc[test, "cases"].to_numpy()
    test_rate_samples = rate_samples[:, test.to_numpy()]
    log_scores = negative_binomial_log_score(
        test_counts,
        test_rate_samples,
        concentration_samples,
    )
    errors = recovery.loc[test, "expected_cases_mean"].to_numpy() - test_counts

    method_label = "hsgp_vi_nb" if METHOD == "hsgp" else f"compressed_vi_nb_z{LATENT_DIM}"
    metrics = pd.DataFrame(
        [
            {
                "dataset": "tycho_measles",
                "method": method_label,
                "train_start": START_MONTH,
                "train_end": TRAIN_END_MONTH,
                "test_start": TEST_START_MONTH,
                "test_end": END_MONTH,
                "n_train": int((recovery["split"] == "train").sum()),
                "n_test": int(test.sum()),
                "runtime_seconds": elapsed,
                "vi_steps": len(losses),
                "final_elbo_loss": float(losses[-1]),
                "test_rmse_count_mean": float(np.sqrt(np.mean(errors**2))),
                "test_mae_count_mean": float(np.mean(np.abs(errors))),
                "test_predictive_coverage_95": float(recovery.loc[test, "covered_95"].mean()),
                "test_mean_log_score": float(np.mean(log_scores)),
                "test_total_log_score": float(np.sum(log_scores)),
                "mean_latent_sd_test": float(recovery.loc[test, "latent_sd"].mean()),
                "nb_concentration_mean": float(concentration_samples.mean()),
            }
        ]
    )
    return recovery, metrics


def plot_recovery(recovery, output_path):
    monthly = (
        recovery.groupby(["month", "split"], as_index=False)
        .agg(
            cases=("cases", "sum"),
            expected=("expected_cases_mean", "sum"),
            covered=("covered_95", "mean"),
        )
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(monthly["month"], monthly["cases"], label="Observed", color="#2b6cb0")
    axes[0].plot(monthly["month"], monthly["expected"], label="Predicted mean", color="#c05621")
    axes[0].axvline(pd.Timestamp(TEST_START_MONTH), color="black", linewidth=1)
    axes[0].set_title("National held out prediction")
    axes[0].set_ylabel("Cases")
    axes[0].legend(frameon=False)

    axes[1].plot(monthly["month"], monthly["covered"], color="#2f855a")
    axes[1].axvline(pd.Timestamp(TEST_START_MONTH), color="black", linewidth=1)
    axes[1].axhline(0.95, color="black", linewidth=1, linestyle=":")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Monthly interval coverage")

    for axis in axes:
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    panel = prepare_panel()
    data = prepare_data(panel)
    guide, result, elapsed = run_vi(data)
    samples = draw_posterior(guide, result.params, data)
    losses = np.asarray(result.losses)
    recovery, metrics = summarize(panel, samples, elapsed, losses)

    method_suffix = "hsgp" if METHOD == "hsgp" else f"compressed_z{LATENT_DIM}"
    suffix = f"{SPLIT_NAME}_{method_suffix}"
    recovery_path = DATA_DIR / f"tycho_heldout_recovery_{suffix}.csv"
    metrics_path = DATA_DIR / f"tycho_heldout_metrics_{suffix}.csv"
    loss_path = DATA_DIR / f"tycho_heldout_loss_{suffix}.csv"
    figure_path = FIGURE_DIR / f"tycho_heldout_prediction_{suffix}.png"

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
