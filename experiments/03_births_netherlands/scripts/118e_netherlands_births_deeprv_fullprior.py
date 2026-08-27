"""
118e_netherlands_births_deeprv_fullprior.py
===========================================
Full-prior DeepRV: sample trend (alpha_f, ell_f) inside NUTS and feed them to a
CONDITIONAL decoder — no HSGP posterior, no fixed hyperparameters. Tests whether
DeepRV can stand on its own, and how badly the alpha_f * z bilinear funnel bites.

Compares against the HSGP NCP baseline (which is itself full-prior, linear basis)
in the same run.

Requires: nl_births_deeprv_z30_h256_conditional.npz  (train_conditional.py)

Run:
  DRV_COND=z30h256_conditional NL_CHAINS=1 NL_WARMUP=1000 NL_SAMPLES=1000 \
      python scripts/118e_netherlands_births_deeprv_fullprior.py
"""
import importlib.util, os, sys, time, json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import jax, jax.numpy as jnp
import numpyro, numpyro.distributions as dist
from numpyro.infer import init_to_value

# Import 118c as a module to reuse model/runner/metrics (env vars read at import).
_spec = importlib.util.spec_from_file_location(
    "v3", str(PROJECT_ROOT / "scripts" / "118c_netherlands_births_hsgp_deeprv_ncp_v3.py"))
v3 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(v3)

from dl4bi.vae.deep_rv import MLPDeepRV

MODEL_DIR = PROJECT_ROOT / "data" / "processed" / "netherlands_births" / "models"
OUT_DIR   = PROJECT_ROOT / "data" / "processed" / "netherlands_births" / "nl_births_deeprv_fullprior"

COND_FILE = os.environ.get("DRV_COND", "z30h256_conditional")
# (z_dim, hidden, file, amplitude_factored)
_COND = {
    "z30h256_conditional": (30, [256, 256], "nl_births_deeprv_z30_h256_conditional.npz", False),
    "z12h36_conditional":  (12, [36,  36],  "nl_births_deeprv_z12_h36_conditional.npz",  False),
    "z12h128_conditional": (12, [128, 128], "nl_births_deeprv_z12_h128_conditional.npz", False),
    "z6h128_conditional":  (6,  [128, 128], "nl_births_deeprv_z6_h128_conditional.npz",  False),
    "z30h128_conditional": (30, [128, 128], "nl_births_deeprv_z30_h128_conditional.npz", False),
    # amplitude-factoring (measles-style): f = alpha * decoder(z, log_ell)
    "z30h256_ampfact":     (30, [256, 256], "nl_births_deeprv_z30_h256_ampfact.npz",     True),
    "z12h128_ampfact":     (12, [128, 128], "nl_births_deeprv_z12_h128_ampfact.npz",     True),
    "z30h128_ampfact":     (30, [128, 128], "nl_births_deeprv_z30_h128_ampfact.npz",     True),
    "z6h128_ampfact":      (6,  [128, 128], "nl_births_deeprv_z6_h128_ampfact.npz",      True),
    "z12h36_ampfact":      (12, [36,  36],  "nl_births_deeprv_z12_h36_ampfact.npz",      True),
    "z30h64_ampfact":      (30, [64,  64],  "nl_births_deeprv_z30_h64_ampfact.npz",      True),
    "z30h32_ampfact":      (30, [32,  32],  "nl_births_deeprv_z30_h32_ampfact.npz",      True),
    "z30h16_ampfact":      (30, [16,  16],  "nl_births_deeprv_z30_h16_ampfact.npz",      True),
}
assert COND_FILE in _COND, f"unknown DRV_COND={COND_FILE!r}"


# ── Full-prior DeepRV model (62 NUTS dims — same as HSGP) ─────────────────────
def make_fullprior_deeprv_model(model_obj, params, z_dim, amp_factored=False):
    """Trend = conditional decoder with SAMPLED (alpha_f, ell_f).
    NUTS dims: u(z_dim) + z_h(20) + {trend_a, trend_l, season_a, season_l,
                                     beta0, sigma, w_raw(6)} = z_dim + 32.

    amp_factored=False: f = decoder(z, [alpha, ell])   (alpha inside the MLP)
    amp_factored=True : f = alpha * decoder(z, [log ell])  (measles-style, alpha
        factored out — turns the alpha*z funnel into HSGP-like benign geometry)."""
    def model(y, Phi_season, W_weekday,
              trend_ig_a, trend_ig_b, season_ig_a, season_ig_b, **_kw):
        beta0         = numpyro.sample("beta0",         dist.Normal(0.0, 1.0))
        sigma         = numpyro.sample("sigma",         dist.HalfNormal(1.0))
        w_raw         = numpyro.sample("w_raw",         dist.Normal(0.0, 1.0).expand([6]))
        trend_alpha   = numpyro.sample("trend_alpha",   dist.HalfNormal(1.0))
        trend_length  = numpyro.sample("trend_length",  dist.InverseGamma(trend_ig_a, trend_ig_b))
        season_alpha  = numpyro.sample("season_alpha",  dist.HalfNormal(1.0))
        season_length = numpyro.sample("season_length", dist.InverseGamma(season_ig_a, season_ig_b))

        u        = numpyro.sample("deeprv_z", dist.Normal(0.0, 1.0).expand([z_dim]))
        z_season = numpyro.sample("z_season", dist.Normal(0.0, 1.0).expand([2*v3.SEASON_FOURIER]))

        if amp_factored:
            # decoder emits a UNIT-amplitude field conditioned on log ell; alpha
            # multiplies it outside the MLP.
            cond      = jnp.array([jnp.log(trend_length)])
            f_unit    = model_obj.apply({"params": params}, u[None, :], cond,
                                        method="decode")[0]
            f_unit    = f_unit - jnp.mean(f_unit)
            f_trend   = trend_alpha * f_unit
        else:
            # alpha and ell both fed into the decoder.
            cond      = jnp.stack([trend_alpha, trend_length])
            trend_raw = model_obj.apply({"params": params}, u[None, :], cond,
                                        method="decode")[0]
            f_trend   = trend_raw - jnp.mean(trend_raw)

        D_sea      = v3.fourier_scale_harmonics(season_alpha, season_length, 1, v3.SEASON_FOURIER)
        D_sea_full = jnp.concatenate([D_sea, D_sea])
        f_season   = Phi_season @ (D_sea_full * z_season)

        w_last = -jnp.sum(w_raw)
        w_coef = jnp.concatenate([w_raw, jnp.array([w_last])])
        eta = beta0 + f_trend + f_season + W_weekday @ w_coef
        numpyro.sample("obs", dist.Normal(eta, sigma), obs=y)
        numpyro.deterministic("eta", eta)
    return model


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df, md, meta = v3.load_data()
    y_obs = df["log_relative_births"].to_numpy()

    print("="*65)
    print(f"Full-prior DeepRV vs HSGP   T={meta['T']}")
    print(f"warmup={v3.NUM_WARMUP} samples={v3.NUM_SAMPLES} chains={v3.NUM_CHAINS}")
    print(f"seasonal: exact Bessel, ρ_h~InvGamma[{v3.SEASON_LO},{v3.SEASON_HI}]")
    print("="*65)

    mode_trend  = float(v3.TREND_IG_B  / (v3.TREND_IG_A  + 1.0))
    mode_season = float(v3.SEASON_IG_B / (v3.SEASON_IG_A + 1.0))

    # ── HSGP baseline (full prior, linear basis) ─────────────────────────────
    hsgp_md = {k: md[k] for k in ("y", "Phi_trend", "omega_trend", "Phi_season",
                                  "W_weekday", "trend_ig_a", "trend_ig_b",
                                  "season_ig_a", "season_ig_b")}
    hsgp_init = {"beta0": jnp.array(0.0), "sigma": jnp.array(0.1),
                 "w_raw": jnp.zeros(6), "trend_alpha": jnp.array(0.1),
                 "trend_length": jnp.array(mode_trend), "season_alpha": jnp.array(0.1),
                 "season_length": jnp.array(mode_season),
                 "z_trend": jnp.zeros(v3.TREND_BASIS),
                 "z_season": jnp.zeros(2*v3.SEASON_FOURIER)}
    print("── HSGP NCP (62 dims, full prior) ──")
    hsgp_mcmc, hsgp_t = v3.run_mcmc(v3.hsgp_ncp_model, hsgp_md, v3.SEED, init_vals=hsgp_init)
    hsgp_row, hsgp_flat = v3.summarize("hsgp_ncp", df, hsgp_mcmc, hsgp_t, 62)
    print(f"  runtime={hsgp_t:.0f}s  rhat={hsgp_row['max_rhat']:.3f}  "
          f"ess={hsgp_row['min_ess']:.0f}  ess/s={hsgp_row['ess_per_sec']:.2f}  "
          f"div={hsgp_row['n_divergences']}  tree_frac={hsgp_row['max_tree_depth_frac']:.3f}")
    print(f"  trend_α={hsgp_row['trend_alpha_post_mean']:.4f}  "
          f"trend_ℓ={hsgp_row['trend_length_post_mean']:.4f}")

    # ── Full-prior DeepRV (conditional decoder) ──────────────────────────────
    z_dim, hid, cond_file, amp_factored = _COND[COND_FILE]
    arrays = np.load(MODEL_DIR / cond_file)
    # this decoder was saved with an extra "params/" prefix — strip it so the
    # rebuilt tree is {"MLP_0": ...} (apply() adds the single "params" layer).
    flat = {}
    for k in arrays.files:
        if "/" not in k:
            continue
        kk = k[len("params/"):] if k.startswith("params/") else k
        flat[kk] = jnp.asarray(arrays[k])
    params = v3.unflatten_params(flat)
    dobj = MLPDeepRV(hid + [meta["T"]])
    fp_model = make_fullprior_deeprv_model(dobj, params, z_dim, amp_factored=amp_factored)

    fp_md = {k: md[k] for k in ("y", "Phi_season", "W_weekday",
                                "trend_ig_a", "trend_ig_b",
                                "season_ig_a", "season_ig_b")}
    # init inside the decoder's trained (alpha,ell) region — a starting point, not a fix
    fp_init = {"beta0": jnp.array(0.0), "sigma": jnp.array(0.1), "w_raw": jnp.zeros(6),
               "trend_alpha": jnp.array(0.3), "trend_length": jnp.array(0.3),
               "season_alpha": jnp.array(0.1), "season_length": jnp.array(mode_season),
               "deeprv_z": jnp.zeros(z_dim), "z_season": jnp.zeros(2*v3.SEASON_FOURIER)}
    fp_dims = z_dim + 32
    print(f"\n── Full-prior DeepRV ({fp_dims} dims, conditional decoder {COND_FILE}) ──")
    fp_mcmc, fp_t = v3.run_mcmc(fp_model, fp_md, v3.SEED + 7, init_vals=fp_init)
    fp_row, fp_flat = v3.summarize(f"deeprv_fullprior_{COND_FILE}", df, fp_mcmc, fp_t, fp_dims)
    print(f"  runtime={fp_t:.0f}s  rhat={fp_row['max_rhat']:.3f}  "
          f"ess={fp_row['min_ess']:.0f}  ess/s={fp_row['ess_per_sec']:.2f}  "
          f"div={fp_row['n_divergences']}  tree_frac={fp_row['max_tree_depth_frac']:.3f}")
    print(f"  trend_α={fp_row['trend_alpha_post_mean']:.4f}  "
          f"trend_ℓ={fp_row['trend_length_post_mean']:.4f}")

    # ── Comparison ────────────────────────────────────────────────────────────
    eta_h = hsgp_flat["eta"].mean(0)
    eta_f = fp_flat["eta"].mean(0)
    corr  = float(np.corrcoef(eta_h, eta_f)[0, 1])
    dmean = float(np.mean(np.abs(eta_h - eta_f)))
    print(f"\n── Comparison ──")
    print(f"  corr(η) = {corr:.4f}   D_mean(η) = {dmean:.4f}")

    import pandas as pd
    pd.DataFrame([hsgp_row, fp_row]).to_csv(OUT_DIR / "metrics_fullprior.csv", index=False)
    (OUT_DIR / "config.json").write_text(json.dumps(
        {**meta, "corr_eta": corr, "D_mean_eta": dmean,
         "cond_decoder": COND_FILE, "amplitude_factored": bool(amp_factored),
         "trend_train_range": {
             "a": ([float(arrays["train_a_lo"]), float(arrays["train_a_hi"])]
                   if "train_a_lo" in arrays.files else "factored (alpha=1)"),
             "l": [float(arrays["train_l_lo"]), float(arrays["train_l_hi"])]}},
        indent=2))
    print(f"\n[saved] {OUT_DIR}")
    print("="*65)


if __name__ == "__main__":
    main()
