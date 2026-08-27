from pathlib import Path
import os
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(PROJECT_ROOT / ".numba_cache"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import gammaln, logsumexp

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from flax.core import freeze
from numpyro.infer import Predictive, SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoDiagonalNormal
from numpyro.infer.initialization import init_to_value
from numpyro.optim import Adam

from dl4bi.vae.deep_rv import MLPDeepRV


DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = DATA_DIR / "official_deeprv_models"
DECODER_DIR = DATA_DIR / "decoder_models"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"

START_MONTH = os.environ.get("TYCHO_HELDOUT_START_MONTH", "1935-01-01")
TRAIN_END_MONTH = os.environ.get("TYCHO_HELDOUT_TRAIN_END_MONTH", "1958-12-01")
TEST_START_MONTH = os.environ.get("TYCHO_HELDOUT_TEST_START_MONTH", "1959-01-01")
END_MONTH = os.environ.get("TYCHO_HELDOUT_END_MONTH", "1960-12-01")
SPLIT_NAME = os.environ.get("TYCHO_HELDOUT_SPLIT_NAME", "prevaccine_1959_1960")

LATENT_DIM = int(os.environ.get("TYCHO_OFFICIAL_COEFF_LATENT_DIM", "120"))


def unflatten_params(flat):
    root = {}
    for key, value in flat.items():
        parts = key.split("/")
        cursor = root
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
    return freeze(root)


def load_deeprv_bundle():
    path = MODEL_DIR / f"tycho_official_deeprv_coefficient_1935_1968_z{LATENT_DIM}.npz"
    arrays = np.load(path)
    flat = {
        key: jnp.asarray(arrays[key])
        for key in arrays.files
        if "/" in key
    }
    params = unflatten_params(flat)
    metadata = {
        "hidden_dim": int(arrays["hidden_dim"]),
        "output_dim": int(arrays["output_dim"]),
        "conditionals": jnp.asarray(arrays["conditionals"].astype(np.float32)),
    }
    return params, metadata


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
    decoder = np.load(DECODER_DIR / f"tycho_compressed_decoder_1935_1968_z{LATENT_DIM}.npz")
    full_n_months = 408
    decoder_indices = (
        panel["state_index"].to_numpy(dtype=int) * full_n_months
        + panel["model_time_index"].to_numpy(dtype=int)
    )
    weights = decoder["weights"].astype(np.float32)[:, decoder_indices]
    bias = decoder["bias"].astype(np.float32)[decoder_indices]
    return {
        "count": jnp.asarray(panel["cases"].to_numpy(dtype=np.int32)),
        "train_mask": jnp.asarray((panel["split"] == "train").to_numpy()),
        "log_state_exposure": jnp.asarray(panel["log_state_exposure"].to_numpy(dtype=np.float32)),
        "season_sin": jnp.asarray(panel["season_sin"].to_numpy(dtype=np.float32)),
        "season_cos": jnp.asarray(panel["season_cos"].to_numpy(dtype=np.float32)),
        "decoder_weights": jnp.asarray(weights),
        "decoder_bias": jnp.asarray(bias),
    }


def make_model(deeprv_model, deeprv_params, conditionals):
    def model(
        count,
        train_mask,
        log_state_exposure,
        season_sin,
        season_cos,
        decoder_weights,
        decoder_bias,
    ):
        beta0 = numpyro.sample("beta0", dist.Normal(0.0, 0.5))
        beta_sin = numpyro.sample("beta_sin", dist.Normal(0.0, 0.8))
        beta_cos = numpyro.sample("beta_cos", dist.Normal(0.0, 0.8))
        nb_concentration = numpyro.sample("nb_concentration", dist.LogNormal(jnp.log(20.0), 0.8))
        z = numpyro.sample("deeprv_z", dist.Normal(0.0, 1.0).expand([LATENT_DIM]))

        coefficients = deeprv_model.apply(
            {"params": deeprv_params},
            z[None, :],
            conditionals,
            method="decode",
        )[0]
        latent_f = coefficients @ decoder_weights + decoder_bias
        latent_f = latent_f - jnp.mean(latent_f)

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
        numpyro.deterministic("deeprv_coefficients", coefficients)
        observation = dist.NegativeBinomial2(mean=rate, concentration=nb_concentration)
        if count is None:
            numpyro.sample("obs", observation)
        else:
            with numpyro.handlers.mask(mask=train_mask):
                numpyro.sample("obs", observation, obs=count)

    return model


def run_vi(model, data):
    steps = int(os.environ.get("TYCHO_OFFICIAL_COEFF_VI_STEPS", "8000"))
    learning_rate = float(os.environ.get("TYCHO_OFFICIAL_COEFF_VI_LR", "0.003"))
    init_values = {
        "beta0": jnp.asarray(0.0),
        "beta_sin": jnp.asarray(0.0),
        "beta_cos": jnp.asarray(0.0),
        "nb_concentration": jnp.asarray(20.0),
        "deeprv_z": jnp.zeros(LATENT_DIM),
    }
    guide = AutoDiagonalNormal(
        model,
        init_loc_fn=init_to_value(values=init_values),
        init_scale=0.03,
    )
    svi = SVI(model, guide, Adam(learning_rate), Trace_ELBO(num_particles=1))
    start = time.perf_counter()
    result = svi.run(
        jax.random.PRNGKey(20260704),
        steps,
        **data,
        progress_bar=True,
        stable_update=True,
    )
    return guide, result, time.perf_counter() - start


def draw_posterior(model, guide, params, data):
    draws = int(os.environ.get("TYCHO_OFFICIAL_COEFF_POSTERIOR_DRAWS", "400"))
    predictive = Predictive(
        model,
        guide=guide,
        params=params,
        num_samples=draws,
        return_sites=("latent_f", "rate", "obs", "nb_concentration", "deeprv_coefficients"),
    )
    prediction_data = dict(data)
    prediction_data["count"] = None
    return predictive(jax.random.PRNGKey(20260705), **prediction_data)


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
    coefficient_samples = np.asarray(samples["deeprv_coefficients"])

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

    metrics = pd.DataFrame(
        [
            {
                "dataset": "tycho_measles",
                "method": f"official_deeprv_coeff_vi_nb_z{LATENT_DIM}",
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
                "mean_coefficient_sd": float(coefficient_samples.std(axis=0).mean()),
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
    deeprv_params, metadata = load_deeprv_bundle()
    panel = prepare_panel()
    data = prepare_data(panel)
    deeprv_model = MLPDeepRV(
        [metadata["hidden_dim"], metadata["hidden_dim"], metadata["output_dim"]]
    )
    model = make_model(deeprv_model, deeprv_params, metadata["conditionals"])
    guide, result, elapsed = run_vi(model, data)
    samples = draw_posterior(model, guide, result.params, data)
    losses = np.asarray(result.losses)
    recovery, metrics = summarize(panel, samples, elapsed, losses)

    suffix = f"{SPLIT_NAME}_official_deeprv_coeff_z{LATENT_DIM}"
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
