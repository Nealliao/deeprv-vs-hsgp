from pathlib import Path
import importlib.util
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(PROJECT_ROOT / ".numba_cache"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import optax

from dl4bi.core.train import TrainState
from dl4bi.vae.deep_rv import MLPDeepRV
from dl4bi.vae.train_utils import deep_rv_train_step


DATA_DIR = PROJECT_ROOT / "data" / "processed"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
MODEL_DIR = DATA_DIR / "official_deeprv_models"

LATENT_DIM = int(os.environ.get("TYCHO_OFFICIAL_DEEPRV_LATENT_DIM", "120"))
HIDDEN_DIM = int(os.environ.get("TYCHO_OFFICIAL_DEEPRV_HIDDEN_DIM", "512"))
TRAIN_SAMPLES = int(os.environ.get("TYCHO_OFFICIAL_DEEPRV_TRAIN_SAMPLES", "700"))
TEST_SAMPLES = int(os.environ.get("TYCHO_OFFICIAL_DEEPRV_TEST_SAMPLES", "200"))
TRAIN_STEPS = int(os.environ.get("TYCHO_OFFICIAL_DEEPRV_TRAIN_STEPS", "2500"))
BATCH_SIZE = int(os.environ.get("TYCHO_OFFICIAL_DEEPRV_BATCH_SIZE", "64"))
LEARNING_RATE = float(os.environ.get("TYCHO_OFFICIAL_DEEPRV_LR", "0.0007"))


def load_tycho_prior_module():
    path = PROJECT_ROOT / "scripts" / "38_make_tycho_compressed_decoder.py"
    spec = importlib.util.spec_from_file_location("tycho_prior", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_conditionals(n_states, n_months):
    return np.array(
        [
            0.35,
            24.0,
            0.35,
            1.2,
            0.35,
            30.0,
            0.35,
            float(n_states),
            float(n_months),
        ],
        dtype=np.float32,
    )


def pca_coordinates(train_samples, test_samples, latent_dim):
    bias = train_samples.mean(axis=0)
    centered_train = train_samples - bias
    scaled_train = centered_train / np.sqrt(train_samples.shape[0] - 1)
    _, singular_values, vt = np.linalg.svd(scaled_train, full_matrices=False)
    components = vt[:latent_dim]
    scales = singular_values[:latent_dim]
    z_train = centered_train @ components.T / scales
    z_test = (test_samples - bias) @ components.T / scales
    weights = scales[:, None] * components
    linear_test_prediction = z_test @ weights + bias
    explained = float(np.sum(scales**2) / np.sum(singular_values**2))
    return (
        z_train.astype(np.float32),
        z_test.astype(np.float32),
        bias.astype(np.float32),
        components.astype(np.float32),
        scales.astype(np.float32),
        float(np.sqrt(np.mean((linear_test_prediction - test_samples) ** 2))),
        explained,
    )


def make_batch(z, f, conditionals, rng):
    indices = rng.choice(len(z), size=BATCH_SIZE, replace=False)
    return {
        "z": jnp.asarray(z[indices]),
        "conditionals": jnp.asarray(conditionals),
        "f": jnp.asarray(f[indices]),
    }


def flatten_params(params, prefix=""):
    flat = {}
    for key, value in params.items():
        path = f"{prefix}/{key}" if prefix else str(key)
        if hasattr(value, "items"):
            flat.update(flatten_params(value, path))
        else:
            flat[path] = value
    return flat


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    prior_module = load_tycho_prior_module()
    panel, regions = prior_module.load_panel_and_regions()
    n_months = panel["model_time_index"].nunique()
    n_states = len(regions)
    train_prior = prior_module.generate_prior_samples(
        regions,
        n_months=n_months,
        n_samples=TRAIN_SAMPLES,
        seed=20260630,
    )
    test_prior = prior_module.generate_prior_samples(
        regions,
        n_months=n_months,
        n_samples=TEST_SAMPLES,
        seed=20270630,
    )
    z_train, z_test, bias, components, scales, linear_test_rmse, explained = pca_coordinates(
        train_prior,
        test_prior,
        LATENT_DIM,
    )
    conditionals = make_conditionals(n_states, n_months)

    model = MLPDeepRV([HIDDEN_DIM, HIDDEN_DIM, train_prior.shape[1]])
    variables = model.init(
        jax.random.PRNGKey(0),
        z=jnp.asarray(z_train[:4]),
        conditionals=jnp.asarray(conditionals),
    )
    state = TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=optax.adamw(learning_rate=LEARNING_RATE, weight_decay=1e-5),
    )

    rng = np.random.default_rng(20260630)
    key = jax.random.PRNGKey(20260630)
    losses = []
    for step in range(TRAIN_STEPS):
        key, subkey = jax.random.split(key)
        batch = make_batch(z_train, train_prior, conditionals, rng)
        state, loss = deep_rv_train_step(subkey, state, batch)
        losses.append(float(loss))
        if (step + 1) % 500 == 0:
            print(f"step={step + 1} loss={losses[-1]:.6f}")

    neural_test_prediction = np.asarray(
        model.apply(
            {"params": state.params},
            jnp.asarray(z_test),
            jnp.asarray(conditionals),
            method="decode",
        )
    )
    neural_test_rmse = float(np.sqrt(np.mean((neural_test_prediction - test_prior) ** 2)))
    neural_test_mae = float(np.mean(np.abs(neural_test_prediction - test_prior)))
    variance_ratio = float(neural_test_prediction.var() / test_prior.var())

    model_path = MODEL_DIR / f"tycho_official_deeprv_1935_1968_z{LATENT_DIM}.npz"
    loss_path = DATA_DIR / f"tycho_official_deeprv_training_loss_1935_1968_z{LATENT_DIM}.csv"
    metrics_path = DATA_DIR / f"tycho_official_deeprv_prior_metrics_1935_1968_z{LATENT_DIM}.csv"
    figure_path = FIGURE_DIR / f"tycho_official_deeprv_training_loss_1935_1968_z{LATENT_DIM}.png"

    flat_params = {key: np.asarray(value) for key, value in flatten_params(state.params).items()}
    np.savez_compressed(
        model_path,
        **flat_params,
        latent_dim=np.asarray(LATENT_DIM),
        hidden_dim=np.asarray(HIDDEN_DIM),
        output_dim=np.asarray(train_prior.shape[1]),
        n_states=np.asarray(n_states),
        n_months=np.asarray(n_months),
        bias=bias,
        pca_components=components,
        pca_scales=scales,
        conditionals=conditionals,
    )
    pd.DataFrame({"step": np.arange(1, len(losses) + 1), "loss": losses}).to_csv(
        loss_path,
        index=False,
    )
    pd.DataFrame(
        [
            {
                "dataset": "tycho_measles",
                "method": "official_deeprv_decoder",
                "latent_dim": LATENT_DIM,
                "hidden_dim": HIDDEN_DIM,
                "train_samples": TRAIN_SAMPLES,
                "test_samples": TEST_SAMPLES,
                "train_steps": TRAIN_STEPS,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "output_dim": train_prior.shape[1],
                "explained_prior_variance": explained,
                "linear_decoder_test_rmse": linear_test_rmse,
                "neural_decoder_test_rmse": neural_test_rmse,
                "neural_decoder_test_mae": neural_test_mae,
                "neural_decoder_variance_ratio": variance_ratio,
                "final_train_loss": losses[-1],
                "model_path": str(model_path),
            }
        ]
    ).to_csv(metrics_path, index=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.arange(1, len(losses) + 1), losses, linewidth=1.2)
    ax.set_title("Tycho official DeepRV decoder training loss")
    ax.set_xlabel("Step")
    ax.set_ylabel("MSE loss")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    print(pd.read_csv(metrics_path).round(4).to_string(index=False))
    print(metrics_path)
    print(loss_path)
    print(figure_path)
    print(model_path)


if __name__ == "__main__":
    main()
