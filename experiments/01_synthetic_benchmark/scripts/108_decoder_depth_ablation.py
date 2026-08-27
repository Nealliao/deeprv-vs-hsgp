"""
108_decoder_depth_ablation.py
================================================================================
Decoder-depth ablation for the conditional DeepRV interaction decoder.

Motivation: the true z->f map is (per fixed ell) a LINEAR PCA reconstruction
  f ~= bias + (z * score_scale) @ components.
The MLP's two tanh layers exist only to let that linear map vary smoothly with
log_ell. If the nonlinearity is excessive, it adds curvature that hurts NUTS
geometry (more divergences, longer runtime) without improving fidelity.

This script retrains ONLY the interaction decoder (z=120, output=1440 — the
large component and the main divergence/runtime contributor) at three depths:
  depth 0  pure linear:   [z, log_ell] -> Dense(output)
  depth 1  one hidden:    [z, log_ell] -> Dense(H) -> tanh -> Dense(output)
  depth 2  current:       [z, log_ell] -> Dense(H) -> tanh -> Dense(H) -> tanh -> Dense(output)

Compares prior fidelity (prior_rmse, var_ratio) on a held-out test set. If a
shallower decoder matches depth-2 fidelity, it is saved for an inference
comparison via script 107.

Space/time decoders are NOT touched (z=15/20, already tiny).

Env vars (same defaults as script 103):
  CDRV_N_TRAIN (6000)  CDRV_N_TEST (1000)  CDRV_TRAIN_STEPS (5000)
  CDRV_BATCH_SIZE (128)  CDRV_LR (1e-3)  CDRV_HIDDEN_DIM (512)  CDRV_Z_W (120)
  ABL_DEPTHS (default "0,1,2")  ABL_SAVE_DEPTH (default "1" — which depth to save)
"""

from pathlib import Path
import os
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import invgamma as sp_invgamma

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training import train_state as flax_ts

DATA_DIR  = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = DATA_DIR / "official_deeprv_models"

N_TRAIN     = int(os.environ.get("CDRV_N_TRAIN",     "6000"))
N_TEST      = int(os.environ.get("CDRV_N_TEST",      "1000"))
TRAIN_STEPS = int(os.environ.get("CDRV_TRAIN_STEPS", "5000"))
BATCH_SIZE  = int(os.environ.get("CDRV_BATCH_SIZE",  "128"))
LR          = float(os.environ.get("CDRV_LR",        "1e-3"))
HIDDEN_DIM  = int(os.environ.get("CDRV_HIDDEN_DIM",  "512"))
Z_W         = int(os.environ.get("CDRV_Z_W",         "120"))

ABL_DEPTHS     = [int(d) for d in os.environ.get("ABL_DEPTHS", "0,1,2").split(",") if d.strip()]
ABL_SAVE_DEPTH = int(os.environ.get("ABL_SAVE_DEPTH", "1"))

PRIOR = {
    "interaction_space_length": (5.3661, 0.3648),
    "interaction_time_length":  (7.3012, 30.009),
}


def _se_cov(x, ell):
    d2 = cdist(x, x, "sqeuclidean")
    return np.exp(-d2 / (2.0 * ell ** 2))


def _double_center(m):
    return (m - m.mean(axis=0, keepdims=True)
              - m.mean(axis=1, keepdims=True) + m.mean())


def _sample_invgamma(a, b, n, rng):
    return sp_invgamma.rvs(a=a, scale=b, size=n, random_state=rng)


def generate_interaction_data(coords, time_pts, n, seed):
    rng = np.random.default_rng(seed)
    S = coords.shape[0]; T = len(time_pts)
    t2d = time_pts.reshape(-1, 1).astype(float)
    a_s, b_s = PRIOR["interaction_space_length"]
    a_t, b_t = PRIOR["interaction_time_length"]
    ells_s = _sample_invgamma(a_s, b_s, n, rng)
    ells_t = _sample_invgamma(a_t, b_t, n, rng)
    f_out  = np.empty((n, S * T), dtype=np.float32)
    le_out = np.empty((n, 2),     dtype=np.float32)
    for i, (es, et) in enumerate(zip(ells_s, ells_t)):
        Ks = _se_cov(coords, es) + 1e-6 * np.eye(S)
        Kt = _se_cov(t2d,   et) + 1e-6 * np.eye(T)
        Ls = np.linalg.cholesky(Ks); Lt = np.linalg.cholesky(Kt)
        noise = rng.standard_normal((S, T))
        w = Ls @ noise @ Lt.T
        w = _double_center(w)
        std = w.std()
        if std > 1e-8:
            w /= std
        f_out[i] = w.flatten()
        le_out[i, 0] = np.log(es); le_out[i, 1] = np.log(et)
    return f_out, le_out


def compute_pca_scores(f, z_dim):
    bias = f.mean(axis=0)
    fc = f - bias
    scaled = fc / np.sqrt(len(f) - 1)
    _, sv, vt = np.linalg.svd(scaled, full_matrices=False)
    components  = vt[:z_dim]
    score_scale = sv[:z_dim] + 1e-8
    z = (fc @ components.T) / score_scale
    return z.astype(np.float32), components, score_scale, bias


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


class TrainState(flax_ts.TrainState):
    pass


@jax.jit
def _train_step(state, z_batch, ell_batch, f_batch):
    def loss_fn(params):
        f_pred = state.apply_fn({"params": params}, z_batch, ell_batch)
        return jnp.mean((f_pred - f_batch) ** 2)
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads), loss


def _prior_rmse(state, f_test, z_test, le_test):
    f_hat = np.asarray(state.apply_fn({"params": state.params},
                                      jnp.asarray(z_test), jnp.asarray(le_test)))
    f_hat_c = f_hat - f_hat.mean(axis=1, keepdims=True)
    f_c     = f_test - f_test.mean(axis=1, keepdims=True)
    rmse = float(np.sqrt(np.mean((f_hat_c - f_c) ** 2)))
    var_ratio = float(f_hat_c.var() / max(f_c.var(), 1e-12))
    return rmse, var_ratio


def _flatten_params(params, prefix=""):
    flat = {}
    for k, v in params.items():
        path = f"{prefix}/{k}" if prefix else str(k)
        if hasattr(v, "items"):
            flat.update(_flatten_params(v, path))
        else:
            flat[path] = np.asarray(v)
    return flat


def train_one(depth, z_dim, f_train, le_train, f_test, le_test, z_train, z_test):
    out_dim = f_train.shape[1]; n_ell = le_train.shape[1]
    model = DepthDecoder(hidden_dim=HIDDEN_DIM, output_dim=out_dim, depth=depth)
    params = model.init(jax.random.PRNGKey(0),
                        jnp.zeros((1, z_dim)), jnp.zeros((1, n_ell)))["params"]
    state = TrainState.create(apply_fn=model.apply, params=params,
                              tx=optax.adamw(learning_rate=LR, weight_decay=1e-5))
    n_params = sum(int(np.prod(p.shape)) for p in jax.tree_util.tree_leaves(params))
    rng_np = np.random.default_rng(42 + z_dim + depth)
    n = len(f_train)
    t0 = time.perf_counter()
    last = 0.0
    for step in range(TRAIN_STEPS):
        idx = rng_np.choice(n, size=BATCH_SIZE, replace=False)
        state, loss = _train_step(state, jnp.asarray(z_train[idx]),
                                  jnp.asarray(le_train[idx]), jnp.asarray(f_train[idx]))
        last = float(loss)
    elapsed = time.perf_counter() - t0
    rmse, vr = _prior_rmse(state, f_test, z_test, le_test)
    return state, {
        "depth": depth, "z_dim": z_dim, "n_ell": n_ell,
        "hidden_dim": HIDDEN_DIM, "output_dim": out_dim,
        "n_params": n_params, "train_steps": TRAIN_STEPS, "train_seconds": elapsed,
        "prior_rmse": rmse, "var_ratio": vr, "final_loss": last,
    }


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    regions  = pd.read_csv(DATA_DIR / "synthetic_measles_regions.csv") \
                 .sort_values("state_index").reset_index(drop=True)
    coords   = regions[["x_coord", "y_coord"]].to_numpy(dtype=np.float64)
    coords_c = coords - coords.mean(axis=0, keepdims=True)
    time_pts = np.arange(72, dtype=np.float64)

    print(f"[config] N_TRAIN={N_TRAIN} steps={TRAIN_STEPS} hidden={HIDDEN_DIM} "
          f"z_w={Z_W} depths={ABL_DEPTHS}")
    print("[data] generating interaction GP samples …")
    f_tr, le_tr = generate_interaction_data(coords_c, time_pts, N_TRAIN, seed=3001)
    f_te, le_te = generate_interaction_data(coords_c, time_pts, N_TEST,  seed=3002)

    # PCA encoder (shared across depths — same z targets, fair comparison)
    z_train, components, score_scale, bias = compute_pca_scores(f_tr, Z_W)
    z_test = ((f_te - bias) @ components.T / score_scale).astype(np.float32)
    print(f"  PCA z: train std={z_train.std():.3f}  shape={z_train.shape}")

    records = []
    saved_states = {}
    for depth in ABL_DEPTHS:
        print(f"\n[train] inter decoder depth={depth} …")
        state, meta = train_one(depth, Z_W, f_tr, le_tr, f_te, le_te, z_train, z_test)
        print(f"  depth={depth}  n_params={meta['n_params']:,}  "
              f"prior_rmse={meta['prior_rmse']:.4f}  var_ratio={meta['var_ratio']:.4f}  "
              f"t={meta['train_seconds']:.1f}s")
        records.append(meta)
        saved_states[depth] = state

    df = pd.DataFrame(records)
    print("\n" + "=" * 64)
    print("DEPTH ABLATION — interaction decoder prior fidelity")
    print("=" * 64)
    print(df[["depth","n_params","prior_rmse","var_ratio","train_seconds"]].round(4).to_string(index=False))
    # reference: current production decoder
    ref = MODEL_DIR / "conditional_deeprv_inter.npz"
    if ref.exists():
        b = np.load(ref)
        print(f"\n[ref] current production inter decoder: prior_rmse={float(b['prior_rmse']):.4f}  "
              f"var_ratio={float(b['var_ratio']):.4f}  z_dim={int(b['z_dim'])}")

    # Save every trained depth (same training setup) for a fair inference
    # comparison via script 107b — isolates the depth effect from training noise.
    for depth, state in saved_states.items():
        meta = next(r for r in records if r["depth"] == depth)
        flat = _flatten_params(state.params)
        extra = {k: np.asarray(v) for k, v in meta.items()}
        extra["pca_components"]  = np.asarray(components)
        extra["pca_score_scale"] = np.asarray(score_scale)
        extra["pca_bias"]        = np.asarray(bias)
        out = MODEL_DIR / f"conditional_deeprv_inter_depth{depth}.npz"
        np.savez_compressed(out, **flat, **extra)
        print(f"[saved] {out.name}  (depth={depth} inter decoder)")

    df.to_csv(DATA_DIR / "conditional_deeprv_depth_ablation.csv", index=False)
    print(f"[saved] conditional_deeprv_depth_ablation.csv")


if __name__ == "__main__":
    main()
