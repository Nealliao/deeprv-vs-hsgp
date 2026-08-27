"""
make_figures.py
================================================================================
Self-contained: regenerates all 5 paper figures for the joint 9-HP
HSGP-vs-DeepRV comparison from the saved result JSONs.

Reads:  ../results/{hsgp,deeprv}_joint9hp_seed{1,2,3}.json
Writes: ../figures/*.png

Only needs numpy, scipy, matplotlib — no project dependencies, no GPU, no MCMC.
Run:  python make_figures.py
"""

from pathlib import Path
import json
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE      = Path(__file__).resolve().parent
RESULTS   = HERE.parent / "results"
FIGURES   = HERE.parent / "figures"
FIGURES.mkdir(exist_ok=True)

SEEDS = [1, 2, 3]
HPS = ["space_alpha", "space_length", "time_alpha", "time_length",
       "interaction_alpha", "interaction_space_length", "interaction_time_length",
       "sigma_h", "ell_h"]
TITLES = [r"$\sigma_g$  (space amp)", r"$\ell_g$  (space len)",
          r"$\sigma_q$  (time amp)", r"$\ell_q$  (time len)",
          r"$\sigma_w$  (inter amp)", r"$\ell_{ws}$  (inter space len)",
          r"$\ell_{wt}$  (inter time len)", r"$\sigma_h$  (seasonal amp)",
          r"$\ell_h$  (seasonal len)"]
GRP  = ["amp", "len", "amp", "len", "amp", "len", "len", "seas", "seas"]
GCOL = {"amp": "#1a7a3a", "len": "#7a4ba0", "seas": "#b5651d"}
HSGP_C, DRV_C, TRUTH_C = "#2b6cb0", "#dd6b20", "#c0392b"


def load(method, seed):
    return json.load(open(RESULTS / f"{method}_joint9hp_seed{seed}.json"))


def _all(method):
    return [load(method, s) for s in SEEDS]


# ─── Figure 1: 9-HP forest plot (3 seeds pooled) ──────────────────────────────

def fig_forest():
    hsgp, deeprv = _all("hsgp"), _all("deeprv")
    def pooled(rows, hp):
        s = np.concatenate([np.array(r[hp + "_samples"]) for r in rows])
        return s.mean(), np.quantile(s, 0.025), np.quantile(s, 0.975)
    plt.rcParams.update({"font.size": 8.5})
    fig, axes = plt.subplots(3, 3, figsize=(10.5, 7.0))
    for ax, hp, title, g in zip(axes.ravel(), HPS, TITLES, GRP):
        truth = hsgp[0][hp + "_truth"]
        for y, rows, col, lab in [(1.0, hsgp, HSGP_C, "HSGP"), (0.0, deeprv, DRV_C, "DeepRV")]:
            m, lo, hi = pooled(rows, hp)
            ax.plot([lo, hi], [y, y], color=col, lw=3.2, alpha=0.55, solid_capstyle="round", zorder=2)
            ax.plot(m, y, "o", color=col, ms=7, zorder=3, label=lab)
        ax.axvline(truth, color=TRUTH_C, ls="--", lw=1.4, zorder=1)
        ax.set_yticks([0, 1]); ax.set_yticklabels(["DeepRV", "HSGP"], fontsize=8)
        ax.set_ylim(-0.7, 1.7)
        ax.set_title(title, fontsize=9.5, fontweight="bold", color=GCOL[g], pad=3)
        ax.grid(axis="x", alpha=0.18); ax.tick_params(labelsize=7.5); ax.margins(x=0.10)
    axes[0, 0].legend(loc="upper right", fontsize=7, framealpha=0.92, handlelength=1.2)
    fig.suptitle("Joint 9-HP recovery: HSGP vs DeepRV  "
                 "(point = pooled posterior mean, bar = 95% CI over 3 seeds; red = truth)",
                 fontsize=10, y=0.995)
    gl = [Line2D([0], [0], color=GCOL[k], lw=3, label=v)
          for k, v in [("amp", "amplitude"), ("len", "lengthscale"), ("seas", "seasonal")]]
    fig.legend(handles=gl, loc="lower center", ncol=3, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=[0, 0.025, 1, 0.97])
    fig.savefig(FIGURES / "joint9hp_forest_3seed.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ─── Figure 2: 9-HP posterior KDE (3 seeds pooled) ────────────────────────────

def fig_kde():
    hsgp, deeprv = _all("hsgp"), _all("deeprv")
    def pool(rows, hp):
        return np.concatenate([np.array(r[hp + "_samples"]) for r in rows])
    plt.rcParams.update({"font.size": 8.5})
    fig, axes = plt.subplots(3, 3, figsize=(10.5, 7.2))
    for ax, hp, title, g in zip(axes.ravel(), HPS, TITLES, GRP):
        truth = hsgp[0][hp + "_truth"]
        sh, sd = pool(hsgp, hp), pool(deeprv, hp)
        lo = min(np.quantile(sh, 0.005), np.quantile(sd, 0.005), truth)
        hi = max(np.quantile(sh, 0.995), np.quantile(sd, 0.995), truth)
        pad = (hi - lo) * 0.06
        xs = np.linspace(lo - pad, hi + pad, 400)
        for s, col in [(sh, HSGP_C), (sd, DRV_C)]:
            ys = gaussian_kde(s)(xs)
            ax.fill_between(xs, ys, color=col, alpha=0.28, zorder=2)
            ax.plot(xs, ys, color=col, lw=1.6, zorder=3)
        ax.axvline(truth, color=TRUTH_C, ls="--", lw=1.4, zorder=4)
        ax.set_title(title, fontsize=9.5, fontweight="bold", color=GCOL[g], pad=3)
        ax.set_yticks([]); ax.tick_params(labelsize=7.5)
        ax.grid(axis="x", alpha=0.15); ax.margins(x=0.02); ax.set_ylim(bottom=0)
    axes[0, 0].plot([], [], color=HSGP_C, lw=2, label="HSGP")
    axes[0, 0].plot([], [], color=DRV_C, lw=2, label="DeepRV")
    axes[0, 0].legend(loc="upper right", fontsize=7, framealpha=0.92, handlelength=1.3)
    fig.suptitle("Joint 9-HP posterior densities: HSGP vs DeepRV  "
                 "(3 seeds pooled, 12000 draws each; red dashed = truth)", fontsize=10, y=0.997)
    gl = [Line2D([0], [0], color=GCOL[k], lw=3, label=v)
          for k, v in [("amp", "amplitude"), ("len", "lengthscale"), ("seas", "seasonal")]]
    fig.legend(handles=gl, loc="lower center", ncol=3, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, -0.004))
    fig.tight_layout(rect=[0, 0.025, 1, 0.965])
    fig.savefig(FIGURES / "joint9hp_kde_3seed.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ─── Figure 3: latent field heatmap (seed=1) ──────────────────────────────────

def _mat(obj, key):
    return np.array(obj[key]).reshape(20, 72)


def fig_heatmap(seed=1):
    h, d = load("hsgp", seed), load("deeprv", seed)
    true = _mat(h, "latent_f_true")
    hm, dm = _mat(h, "latent_f_post_mean"), _mat(d, "latent_f_post_mean")
    hr, dr = hm - true, dm - true
    vmin = min(true.min(), hm.min(), dm.min()); vmax = max(true.max(), hm.max(), dm.max())
    rlim = max(abs(hr).max(), abs(dr).max())
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.4), constrained_layout=True)
    def show(ax, M, title, cmap, vlo, vhi, xlab):
        im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=vlo, vmax=vhi, interpolation="nearest")
        ax.set_title(title, fontsize=9.5, fontweight="bold", pad=3)
        if xlab: ax.set_xlabel("month", fontsize=8)
        ax.set_ylabel("state", fontsize=8); ax.tick_params(labelsize=7); return im
    im0 = show(axes[0, 0], true, "Truth  latent f", "viridis", vmin, vmax, False)
    show(axes[0, 1], hm, "HSGP  posterior mean", "viridis", vmin, vmax, False)
    show(axes[0, 2], dm, "DeepRV  posterior mean", "viridis", vmin, vmax, False)
    fig.colorbar(im0, ax=axes[0, :], fraction=0.025, pad=0.01, label="f")
    axes[1, 0].axis("off")
    axes[1, 0].text(0.0, 0.55,
                    f"RMSE\n  HSGP   {h['rmse_f']:.4f}\n  DeepRV {d['rmse_f']:.4f}\n\n"
                    f"coverage (95%)\n  HSGP   {h['coverage_95']:.3f}\n  DeepRV {d['coverage_95']:.3f}",
                    fontsize=9.5, va="center", family="monospace")
    imr = show(axes[1, 1], hr, "HSGP  residual (mean - truth)", "RdBu_r", -rlim, rlim, True)
    show(axes[1, 2], dr, "DeepRV  residual (mean - truth)", "RdBu_r", -rlim, rlim, True)
    fig.colorbar(imr, ax=axes[1, 1:], fraction=0.025, pad=0.01, label="residual")
    fig.suptitle(f"Latent field recovery (seed={seed}): truth vs HSGP vs DeepRV", fontsize=11)
    fig.savefig(FIGURES / f"joint9hp_latent_heatmap_seed{seed}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ─── Figure 4: seasonal recovery curve (seed=1) ───────────────────────────────

def fig_seasonal(seed=1):
    h, d = load("hsgp", seed), load("deeprv", seed)
    mo = np.arange(1, 13)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(mo, h["h_monthly_true"], "o-", color="#222", lw=1.8, ms=5, label="truth", zorder=5)
    for obj, col, lab in [(h, HSGP_C, "HSGP"), (d, DRV_C, "DeepRV")]:
        ax.plot(mo, obj["h_monthly_post_mean"], color=col, lw=1.8, label=f"{lab} mean")
        ax.fill_between(mo, obj["h_monthly_ci_low"], obj["h_monthly_ci_high"],
                        color=col, alpha=0.18, label=f"{lab} 95% CI")
    ax.set_xticks(mo); ax.set_xlabel("month of year"); ax.set_ylabel(r"seasonal effect $h_m$")
    ax.set_title(f"Seasonal effect recovery (seed={seed})", fontsize=11, fontweight="bold")
    ax.grid(alpha=0.2); ax.legend(fontsize=8, ncol=3, loc="upper center")
    fig.tight_layout()
    fig.savefig(FIGURES / f"joint9hp_seasonal_seed{seed}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ─── Figure 5: posterior SD heatmap (seed=1) ──────────────────────────────────

def fig_posterior_sd(seed=1):
    h, d = load("hsgp", seed), load("deeprv", seed)
    hsd, dsd = _mat(h, "latent_f_post_sd"), _mat(d, "latent_f_post_sd")
    diff = dsd - hsd
    smax = max(hsd.max(), dsd.max()); smin = min(hsd.min(), dsd.min()); dlim = abs(diff).max()
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.2), constrained_layout=True)
    def show(ax, M, title, cmap, vlo, vhi):
        im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=vlo, vmax=vhi, interpolation="nearest")
        ax.set_title(title, fontsize=9.5, fontweight="bold", pad=3)
        ax.set_xlabel("month", fontsize=8); ax.set_ylabel("state", fontsize=8); ax.tick_params(labelsize=7)
        return im
    im0 = show(axes[0], hsd, "HSGP  posterior SD", "magma", smin, smax)
    show(axes[1], dsd, "DeepRV  posterior SD", "magma", smin, smax)
    fig.colorbar(im0, ax=axes[:2], fraction=0.025, pad=0.01, label="posterior SD")
    imd = show(axes[2], diff, "DeepRV SD - HSGP SD", "RdBu_r", -dlim, dlim)
    fig.colorbar(imd, ax=axes[2], fraction=0.05, pad=0.02, label="SD diff")
    fig.suptitle(f"Posterior uncertainty (seed={seed}):  mean SD  "
                 f"HSGP={hsd.mean():.3f}  DeepRV={dsd.mean():.3f}", fontsize=11)
    fig.savefig(FIGURES / f"joint9hp_posterior_sd_seed{seed}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ─── Figure 6: variance decomposition g:q:w (truth vs HSGP vs DeepRV) ──────────

# truth shares per seed, computed from the DGP (draw order g->q->w, matches
# generate_panel in script 106). Hardcoded so this script stays self-contained.
_TRUTH_FRAC = {1: [0.2667, 0.2144, 0.5189],
               2: [0.4223, 0.0172, 0.5605],
               3: [0.5398, 0.2242, 0.2361]}


def fig_variance_decomposition():
    import numpy as np
    cols = {"g": "#1a7a3a", "q": "#2b6cb0", "w": "#dd6b20"}
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = 0.0; xt = []; xl = []
    for s in (1, 2, 3):
        h = load("hsgp", s); d = load("deeprv", s)
        vals = {"TRUTH": _TRUTH_FRAC[s],
                "HSGP": [h["frac_g_contrib"], h["frac_q_contrib"], h["frac_w_contrib"]],
                "DeepRV": [d["frac_g_contrib"], d["frac_q_contrib"], d["frac_w_contrib"]]}
        for name, v in vals.items():
            bot = 0.0
            for i, c in enumerate(["g", "q", "w"]):
                ax.bar(x, v[i], bottom=bot, color=cols[c], edgecolor="white", width=0.8)
                if v[i] > 0.06:
                    ax.text(x, bot + v[i] / 2, f"{v[i]*100:.0f}", ha="center", va="center",
                            fontsize=7, color="white", fontweight="bold")
                bot += v[i]
            xt.append(x); xl.append(name); x += 1
        x += 0.6
    ax.set_xticks(xt); ax.set_xticklabels(xl, rotation=45, ha="right", fontsize=8)
    for i, s in enumerate((1, 2, 3)):
        ax.text(i * 3.6 + 1, 1.04, f"seed {s}", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("variance share of latent f"); ax.set_ylim(0, 1.08)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=cols["g"], label="space g"), Patch(color=cols["q"], label="time q"),
                       Patch(color=cols["w"], label="interaction w")], loc="lower center", ncol=3,
              bbox_to_anchor=(0.5, -0.22), fontsize=9, frameon=False)
    ax.set_title("Synthetic: variance decomposition g:q:w  (truth vs HSGP vs DeepRV)\n"
                 "DeepRV recovers the shares more accurately (mean L1 0.046 vs HSGP 0.178); "
                 "HSGP over-estimates interaction", fontsize=10.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES / "synth_variance_decomposition.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    print("Regenerating figures from", RESULTS)
    fig_forest();       print("  [1/6] forest plot")
    fig_kde();          print("  [2/6] KDE")
    fig_heatmap();      print("  [3/6] latent heatmap")
    fig_seasonal();     print("  [4/6] seasonal curve")
    fig_posterior_sd(); print("  [5/6] posterior SD heatmap")
    fig_variance_decomposition(); print("  [6/6] variance decomposition")
    print("Done →", FIGURES)


if __name__ == "__main__":
    main()
