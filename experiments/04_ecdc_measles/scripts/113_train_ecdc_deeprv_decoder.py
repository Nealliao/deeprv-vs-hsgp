"""
113_train_ecdc_deeprv_decoder.py
================================================================================
Train conditional DeepRV decoders for the ECDC measles geometry (29 countries
x 120 months). Counterpart of script 103 (synthetic 20x72).

z dims chosen by PCA effective-dimension analysis (matching the synthetic
fidelity standard): z_s=15 (~99.99% var), z_t=12 (~99.9%), z_w=220 (~83%, same
PCA fidelity as the synthetic z_w=120). ECDC interaction is high-frequency
(short inter_space lengthscale 0.078) so it needs a much larger z_w than
synthetic — DeepRV's compression advantage is weaker on real data.

Decoders: decode(z, log_ell) -> f_normalised, frozen at inference.
Geometry baked in: trained on ECDC's standardized coords + arange(120).

Output: data/processed/official_deeprv_models/ecdc_deeprv_{space,time,inter}.npz

Env: ECDC_DRV_N_TRAIN(6000) ECDC_DRV_TRAIN_STEPS(6000) ECDC_DRV_HIDDEN(512)
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
PANEL     = DATA_DIR / "ecdc_measles_panel.csv"

N_TRAIN     = int(os.environ.get("ECDC_DRV_N_TRAIN",     "6000"))
N_TEST      = int(os.environ.get("ECDC_DRV_N_TEST",      "1000"))
TRAIN_STEPS = int(os.environ.get("ECDC_DRV_TRAIN_STEPS", "6000"))
BATCH_SIZE  = int(os.environ.get("ECDC_DRV_BATCH",       "128"))
LR          = float(os.environ.get("ECDC_DRV_LR",        "1e-3"))
HIDDEN_DIM  = int(os.environ.get("ECDC_DRV_HIDDEN",      "512"))
Z_S = int(os.environ.get("ECDC_DRV_Z_S", "15"))
Z_T = int(os.environ.get("ECDC_DRV_Z_T", "12"))
Z_W = int(os.environ.get("ECDC_DRV_Z_W", "220"))

# ECDC lengthscale priors (same as HSGP script 111)
PRIOR = {
    "space_length":             (6.1091, 3.3175),
    "time_length":              (6.5707, 167.3694),
    "interaction_space_length": (5.3661, 0.3648),
    "interaction_time_length":  (6.2718, 27.0192),
}


def _se_cov(x, ell):
    return np.exp(-cdist(x, x, "sqeuclidean") / (2.0 * ell ** 2))


def _dc(m):
    return m - m.mean(0, keepdims=True) - m.mean(1, keepdims=True) + m.mean()


def _ig(a, b, n, rng):
    return sp_invgamma.rvs(a=a, scale=b, size=n, random_state=rng)


def gen_spatial(coords, n, seed):
    rng = np.random.default_rng(seed); S = len(coords)
    a, b = PRIOR["space_length"]; ells = _ig(a, b, n, rng)
    f = np.empty((n, S), np.float32); le = np.empty((n, 1), np.float32)
    for i, ell in enumerate(ells):
        L = np.linalg.cholesky(_se_cov(coords, ell) + 1e-6 * np.eye(S))
        g = L @ rng.standard_normal(S); f[i] = g - g.mean(); le[i] = np.log(ell)
    return f, le


def gen_temporal(tpts, n, seed):
    rng = np.random.default_rng(seed); T = len(tpts); t = tpts.reshape(-1, 1).astype(float)
    a, b = PRIOR["time_length"]; ells = _ig(a, b, n, rng)
    f = np.empty((n, T), np.float32); le = np.empty((n, 1), np.float32)
    for i, ell in enumerate(ells):
        L = np.linalg.cholesky(_se_cov(t, ell) + 1e-6 * np.eye(T))
        q = L @ rng.standard_normal(T); f[i] = q - q.mean(); le[i] = np.log(ell)
    return f, le


def gen_interaction(coords, tpts, n, seed):
    rng = np.random.default_rng(seed); S = len(coords); T = len(tpts); t = tpts.reshape(-1, 1).astype(float)
    a_s, b_s = PRIOR["interaction_space_length"]; a_t, b_t = PRIOR["interaction_time_length"]
    es_ = _ig(a_s, b_s, n, rng); et_ = _ig(a_t, b_t, n, rng)
    f = np.empty((n, S * T), np.float32); le = np.empty((n, 2), np.float32)
    for i, (es, et) in enumerate(zip(es_, et_)):
        Ls = np.linalg.cholesky(_se_cov(coords, es) + 1e-6 * np.eye(S))
        Lt = np.linalg.cholesky(_se_cov(t, et) + 1e-6 * np.eye(T))
        w = Ls @ rng.standard_normal((S, T)) @ Lt.T
        w = _dc(w); s = w.std(); w = w / s if s > 1e-8 else w
        f[i] = w.flatten(); le[i, 0] = np.log(es); le[i, 1] = np.log(et)
    return f, le


def compute_pca_scores(f, z_dim):
    bias = f.mean(0); fc = f - bias
    _, sv, vt = np.linalg.svd(fc / np.sqrt(len(f) - 1), full_matrices=False)
    comp = vt[:z_dim]; ss = sv[:z_dim] + 1e-8
    z = (fc @ comp.T) / ss
    return z.astype(np.float32), comp, ss, bias


class ConditionalDecoder(nn.Module):
    hidden_dim: int
    output_dim: int

    @nn.compact
    def __call__(self, z, log_ell):
        x = jnp.concatenate([z, log_ell], axis=-1)
        x = nn.tanh(nn.Dense(self.hidden_dim)(x))
        x = nn.tanh(nn.Dense(self.hidden_dim)(x))
        return nn.Dense(self.output_dim)(x)


class TrainState(flax_ts.TrainState):
    pass


@jax.jit
def _step(state, z, le, f):
    def loss_fn(p):
        return jnp.mean((state.apply_fn({"params": p}, z, le) - f) ** 2)
    l, g = jax.value_and_grad(loss_fn)(state.params)
    return state.apply_gradients(grads=g), l


def _flatten(params, prefix=""):
    flat = {}
    for k, v in params.items():
        path = f"{prefix}/{k}" if prefix else str(k)
        if hasattr(v, "items"):
            flat.update(_flatten(v, path))
        else:
            flat[path] = np.asarray(v)
    return flat


def train_one(name, z_dim, f_tr, le_tr, f_te, le_te):
    out_dim = f_tr.shape[1]; n_ell = le_tr.shape[1]
    z_tr, comp, ss, bias = compute_pca_scores(f_tr, z_dim)
    z_te = ((f_te - bias) @ comp.T / ss).astype(np.float32)
    model = ConditionalDecoder(hidden_dim=HIDDEN_DIM, output_dim=out_dim)
    params = model.init(jax.random.PRNGKey(0), jnp.zeros((1, z_dim)), jnp.zeros((1, n_ell)))["params"]
    state = TrainState.create(apply_fn=model.apply, params=params,
                              tx=optax.adamw(learning_rate=LR, weight_decay=1e-5))
    rng = np.random.default_rng(42 + z_dim); n = len(f_tr); t0 = time.perf_counter(); last = 0.0
    for step in range(TRAIN_STEPS):
        idx = rng.choice(n, BATCH_SIZE, replace=False)
        state, last = _step(state, jnp.asarray(z_tr[idx]), jnp.asarray(le_tr[idx]), jnp.asarray(f_tr[idx]))
        last = float(last)
        if (step + 1) % 2000 == 0:
            print(f"  [{name}] step {step+1} loss {last:.5f}")
    elapsed = time.perf_counter() - t0
    fhat = np.asarray(state.apply_fn({"params": state.params}, jnp.asarray(z_te), jnp.asarray(le_te)))
    fhc = fhat - fhat.mean(1, keepdims=True); fc = f_te - f_te.mean(1, keepdims=True)
    rmse = float(np.sqrt(np.mean((fhc - fc) ** 2))); vr = float(fhc.var() / max(fc.var(), 1e-12))
    print(f"  [{name}] done {elapsed:.1f}s  prior_rmse={rmse:.4f}  var_ratio={vr:.4f}")
    meta = {"z_dim": z_dim, "n_ell": n_ell, "hidden_dim": HIDDEN_DIM, "output_dim": out_dim,
            "prior_rmse": rmse, "var_ratio": vr}
    flat = _flatten(state.params)
    extra = {k: np.asarray(v) for k, v in meta.items()}
    extra["pca_components"] = np.asarray(comp); extra["pca_score_scale"] = np.asarray(ss); extra["pca_bias"] = np.asarray(bias)
    return flat, extra, meta


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    ec = pd.read_csv(PANEL).drop_duplicates("state_index").sort_values("state_index")
    coords = ec[["x_coord", "y_coord"]].to_numpy(dtype=np.float64)
    tpts = np.arange(120, dtype=np.float64)
    print(f"[config] S={len(coords)} T={len(tpts)}  z_s={Z_S} z_t={Z_T} z_w={Z_W}  "
          f"N_train={N_TRAIN} steps={TRAIN_STEPS}")

    print("[data] generating ECDC-geometry GP samples ...")
    fs, les = gen_spatial(coords, N_TRAIN, 1001); fsT, lesT = gen_spatial(coords, N_TEST, 1002)
    ft, let = gen_temporal(tpts, N_TRAIN, 2001);  ftT, letT = gen_temporal(tpts, N_TEST, 2002)
    fw, lew = gen_interaction(coords, tpts, N_TRAIN, 3001); fwT, lewT = gen_interaction(coords, tpts, N_TEST, 3002)

    recs = []
    for name, zdim, (ftr, ltr, fte, lte) in [
        ("space", Z_S, (fs, les, fsT, lesT)),
        ("time",  Z_T, (ft, let, ftT, letT)),
        ("inter", Z_W, (fw, lew, fwT, lewT))]:
        print(f"\n[train] {name} decoder (z={zdim}) ...")
        flat, extra, meta = train_one(name, zdim, ftr, ltr, fte, lte)
        out = MODEL_DIR / f"ecdc_deeprv_{name}.npz"
        np.savez_compressed(out, **flat, **extra)
        print(f"  saved -> {out.name}")
        recs.append({"name": name, **meta})

    print("\n" + "=" * 56)
    print(pd.DataFrame(recs).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
