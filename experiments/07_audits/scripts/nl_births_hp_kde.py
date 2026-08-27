"""Hyperparameter posterior comparison for the Netherlands births model:
HSGP vs full-prior DeepRV, both sampling every hyperparameter.

Answers the supervisor's question "how do the estimated posterior densities for the
hyperparameters compare?". For each shared hyperparameter we overlay the two
posterior densities and report the 1-Wasserstein distance between the draw sets,
normalised by the pooled posterior sd so the numbers are comparable across
parameters on different scales.
"""
from pathlib import Path
import os, json
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats

CX = Path("/Users/Zhuanz/Documents/Codex/2026-05-19/files-mentioned-by-the-user-research")
DRAWS = CX/"data"/"processed"/"netherlands_births"/"nl_births_deeprv_fullprior"/"hyperparam_draws.npz"
OUT = Path("/Users/Zhuanz/Desktop/thesis_v2")

# (key, display label, whether a reference value exists)
PARAMS = [
    ("trend_length",  r"$\rho_f$  trend lengthscale"),
    ("trend_alpha",   r"$\alpha_f$  trend amplitude"),
    ("season_length", r"$\rho_h$  seasonal lengthscale"),
    ("season_alpha",  r"$\alpha_h$  seasonal amplitude"),
    ("sigma",         r"$\sigma$  observation noise"),
    ("beta0",         r"$\beta_0$  intercept"),
]
C_H, C_D = "#4C72B0", "#C44E52"

def w1(a, b):
    """1-Wasserstein between two empirical samples (equal weights)."""
    n = min(len(a), len(b))
    qs = (np.arange(n) + 0.5) / n
    return float(np.mean(np.abs(np.quantile(a, qs) - np.quantile(b, qs))))

def main():
    d = np.load(DRAWS)
    have = [(k, lab) for k, lab in PARAMS
            if f"hsgp__{k}" in d.files and f"deeprv__{k}" in d.files]
    n = len(have)
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5))
    axes = axes.ravel()
    rows = []

    for ax, (k, lab) in zip(axes, have):
        h = np.asarray(d[f"hsgp__{k}"], float)
        v = np.asarray(d[f"deeprv__{k}"], float)
        lo = min(h.min(), v.min()); hi = max(h.max(), v.max())
        pad = 0.06 * (hi - lo); grid = np.linspace(lo - pad, hi + pad, 400)
        for s, c, nm in ((h, C_H, "HSGP"), (v, C_D, "DeepRV")):
            kde = stats.gaussian_kde(s)
            ax.plot(grid, kde(grid), color=c, lw=2, label=nm)
            ax.fill_between(grid, kde(grid), color=c, alpha=0.18)
        ax.axvline(h.mean(), color=C_H, ls="--", lw=1)
        ax.axvline(v.mean(), color=C_D, ls="--", lw=1)

        pooled_sd = np.sqrt(0.5 * (h.var() + v.var()))
        W = w1(h, v)
        rows.append(dict(parameter=k, hsgp_mean=h.mean(), hsgp_sd=h.std(),
                         deeprv_mean=v.mean(), deeprv_sd=v.std(),
                         w1=W, w1_over_sd=W / pooled_sd,
                         ratio=v.mean() / h.mean() if h.mean() != 0 else np.nan))
        ax.set_title(f"{lab}\n$W_1$ = {W:.4f}  ({W/pooled_sd:.2f} pooled sd)",
                     fontsize=10.5)
        ax.set_yticks([]); ax.legend(fontsize=9, frameon=False)

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle("Netherlands daily births: hyperparameter posteriors, HSGP vs full-prior DeepRV\n"
                 "(both models sample every hyperparameter; dashed lines are posterior means)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    (OUT/"figures").mkdir(exist_ok=True)
    fig.savefig(OUT/"figures"/"nl_births_hp_kde.png", dpi=140)

    (OUT/"results").mkdir(exist_ok=True)
    (OUT/"results"/"nl_births_hp_kde.json").write_text(json.dumps(rows, indent=2, default=float))

    print(f"{'parameter':16s} {'HSGP':>18s} {'DeepRV':>18s} {'W1':>9s} {'W1/sd':>7s} {'ratio':>7s}")
    for r in rows:
        print(f"{r['parameter']:16s} {r['hsgp_mean']:8.4f}±{r['hsgp_sd']:<8.4f} "
              f"{r['deeprv_mean']:8.4f}±{r['deeprv_sd']:<8.4f} "
              f"{r['w1']:9.4f} {r['w1_over_sd']:7.2f} {r['ratio']:7.2f}")
    print("\nsaved figures/nl_births_hp_kde.png")

if __name__ == "__main__":
    main()
