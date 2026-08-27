"""Conditional DeepRV decoder with AMPLITUDE FACTORING (measles-style).

f = alpha_f * decoder(z, log_ell)   -- amplitude is an OUTSIDE multiplier, only
the log lengthscale is fed to the decoder. The decoder learns a UNIT-amplitude
field (alpha=1) whose shape varies with ell. This is the key trick from the
measles joint-9HP work: keeping alpha out of the MLP turns the alpha*z funnel
into the SAME benign geometry HSGP has (alpha * basis@z), instead of letting
the MLP's nonlinearity amplify it.

Contrast: the earlier z30h256_conditional fed [alpha, ell] both into the decoder
→ tree_frac 0.271. This version tests whether factoring removes that.
"""
import importlib.util, sys, time
from pathlib import Path
PR = "/Users/Zhuanz/Documents/Codex/2026-05-19/files-mentioned-by-the-user-research"
sys.path.insert(0, PR)
import os
os.environ.setdefault("MPLCONFIGDIR", str(Path(PR) / ".matplotlib"))

import numpy as np
import jax, jax.numpy as jnp, optax

spec = importlib.util.spec_from_file_location("nl118b", f"{PR}/scripts/118b_netherlands_births_deeprv_ablation.py")
b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)

from dl4bi.core.train import TrainState
from dl4bi.vae.deep_rv import MLPDeepRV

Z_DIM, H_DIM = 30, 256
TRAIN_STEPS  = 10000
BATCH        = 128
LR, LR_END   = 3e-4, 1e-5
SEED         = 42
# only lengthscale is conditioned on now; amplitude is factored out (alpha=1 in training)
L_LO, L_HI   = 0.08, 1.50

df, md, meta = b.load_data()
T   = meta["T"]
omega = np.asarray(md["omega_trend"])          # (30,)
Phi   = np.asarray(md["phi_trend_c"])          # (T, 30)
print(f"T={T}  basis={omega.shape[0]}  (amplitude factoring, cond=[log_ell])")

def spec_scale_unit(length):
    # unit amplitude (alpha=1)
    v = 1.0 * np.sqrt(2*np.pi) * length * np.exp(-0.5*(length*omega)**2)
    return np.sqrt(np.maximum(v, 1e-30)).astype(np.float32)   # (30,)

model = MLPDeepRV([H_DIM, H_DIM, T])
init_z = jnp.zeros((8, Z_DIM), jnp.float32)
init_c = jnp.asarray([np.log(0.3)], jnp.float32)   # 1-dim cond: log_ell
variables = model.init(jax.random.PRNGKey(SEED+1), z=init_z, conditionals=init_c)
sched = optax.warmup_cosine_decay_schedule(0.0, LR, 400, TRAIN_STEPS, LR_END)
state = TrainState.create(apply_fn=model.apply, params=variables["params"],
                          tx=optax.chain(optax.adamw(sched, weight_decay=1e-5)))

def _loss(params, z, cond, f, var):
    f_hat = model.apply({"params": params}, z, cond, method="decode")
    return jnp.mean((f_hat - f)**2) / var

@jax.jit
def train_step(state, z, cond, f, var):
    l, g = jax.value_and_grad(_loss)(state.params, z, cond, f, var)
    return state.apply_gradients(grads=g), l

rng = np.random.default_rng(SEED+2)
t0 = time.perf_counter()
for step in range(TRAIN_STEPS):
    l_b = float(np.exp(rng.uniform(np.log(L_LO), np.log(L_HI))))
    sc  = spec_scale_unit(l_b)
    z_b = rng.normal(size=(BATCH, Z_DIM)).astype(np.float32)
    f_b = (z_b * sc[None, :]) @ Phi.T          # unit-amplitude field
    f_b -= f_b.mean(1, keepdims=True)
    var_b = float(f_b.var()) + 1e-12
    state, loss = train_step(state, jnp.asarray(z_b),
                             jnp.asarray([np.log(l_b)], jnp.float32),
                             jnp.asarray(f_b), jnp.asarray(var_b))
    if (step+1) % 1000 == 0:
        print(f"  step {step+1}/{TRAIN_STEPS}  norm_loss={float(loss):.6e}")
train_t = time.perf_counter() - t0
print(f"train={train_t:.0f}s")

# fidelity: only varies with ell now (amplitude is exact by construction)
def decode(z, cond):
    return model.apply({"params": state.params}, z, cond, method="decode")
print("\nprior fidelity (unit-amplitude field, var_ratio→1, rmse→0):")
print(f"{'ell':>7} {'var_ratio':>10} {'rmse':>9}")
rng2 = np.random.default_rng(SEED+9)
for l in [0.10, 0.12, 0.17, 0.25, 0.40, 0.70, 1.00, 1.40]:
    sc = spec_scale_unit(l)
    zt = rng2.normal(size=(400, Z_DIM)).astype(np.float32)
    ftrue = (zt * sc[None, :]) @ Phi.T; ftrue -= ftrue.mean(1, keepdims=True)
    fpred = np.array(decode(jnp.asarray(zt), jnp.asarray([np.log(l)], jnp.float32)))
    fpred = fpred - fpred.mean(1, keepdims=True)
    vr = fpred.var()/ftrue.var()
    rm = np.sqrt(np.mean((fpred-ftrue)**2))
    print(f"{l:7.2f} {vr:10.3f} {rm:9.4f}")

def flatten(d, p=""):
    o={}
    for k,v in d.items():
        f=f"{p}/{k}" if p else str(k)
        if hasattr(v,"items"): o.update(flatten(v,f))
        else: o[f]=np.asarray(v)
    return o
path = Path(PR)/"data/processed/netherlands_births/models/nl_births_deeprv_z30_h256_ampfact.npz"
np.savez_compressed(path, **flatten(state.params),
    latent_dim=np.asarray(Z_DIM), hidden_dim=np.asarray(H_DIM),
    output_dim=np.asarray(T), conditional_dim=np.asarray(1),
    train_l_lo=L_LO, train_l_hi=L_HI, amplitude_factored=np.asarray(1))
print(f"\nsaved -> {path.name}")
