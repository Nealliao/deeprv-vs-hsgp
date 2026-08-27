"""
112_ecdc_make_figures.py
================================================================================
Regenerate the 4 ECDC real-data figures from the result JSON.

Reads:  data/processed/ecdc/ecdc_hsgp_joint9hp_nb.json
Writes: outputs/figures/ecdc_*.png

  1. ecdc_seasonal_curve.png   data-driven seasonal effect (peak month)
  2. ecdc_heatmap.png          observed / predicted / latent field (29 x 120)
  3. ecdc_9hp_forest.png       joint 9-HP posteriors (no truth -> mean + 95% CI)
  4. ecdc_timeseries_fit.png   observed cases vs posterior rate, 9 top countries

Only numpy/scipy/matplotlib. Run: python 112_ecdc_make_figures.py
"""

from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RES = PROJECT_ROOT / "data" / "processed" / "ecdc" / "ecdc_hsgp_joint9hp_nb.json"
FIG = PROJECT_ROOT / "outputs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

R = json.load(open(RES))
S, T = 29, 120
HSGP_C, DRV_C = "#2b6cb0", "#dd6b20"


def fig_seasonal():
    mo = np.arange(1, 13)
    mean = np.array(R["h_monthly_post_mean"]); lo = np.array(R["h_monthly_post_q025"]); hi = np.array(R["h_monthly_post_q975"])
    peak = mo[np.argmax(mean)]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.axhline(0, color="#999", lw=0.8, ls=":")
    ax.fill_between(mo, lo, hi, color=HSGP_C, alpha=0.18, label="95% CI")
    ax.plot(mo, mean, "o-", color=HSGP_C, lw=2, ms=6, label="posterior mean")
    ax.axvspan(2.5, 5.5, color="#e8c14a", alpha=0.12)
    ax.annotate(f"peak: month {peak}", xy=(peak, mean.max()), xytext=(peak + 0.3, mean.max() + 0.05), fontsize=9, color="#c0392b")
    mname = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    ax.set_xticks(mo); ax.set_xticklabels(mname, fontsize=8)
    ax.set_ylabel(r"seasonal log-risk $h_m$"); ax.set_xlabel("month of year")
    ax.set_title("ECDC measles seasonal effect (data-driven, 2010-2019)\nHSGP joint-9HP, NB likelihood", fontsize=10.5, fontweight="bold")
    ax.grid(alpha=0.15); ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout(); fig.savefig(FIG / "ecdc_seasonal_curve.png", dpi=200, bbox_inches="tight"); plt.close(fig)


def _grids():
    si = np.array(R["state_index"]); ti = np.array(R["time_index"])
    cases = np.array(R["cases"]); rate = np.array(R["rate_post_mean"]); lf = np.array(R["latent_f_post_mean"])
    country = np.array(R["country"])
    codemap = {int(si[i]): country[i] for i in range(len(si))}
    tot = {s: cases[si == s].sum() for s in range(S)}
    order = sorted(range(S), key=lambda s: -tot[s]); pos = {s: i for i, s in enumerate(order)}

    def grid(vals):
        g = np.full((S, T), np.nan)
        for i in range(len(si)):
            g[pos[int(si[i])], int(ti[i])] = vals[i]
        return g
    return grid(cases), grid(rate), grid(lf), [codemap[s] for s in order], codemap, si, ti, cases, rate


def fig_heatmap():
    Gc, Gr, Glf, ylabels, *_ = _grids()
    G_obs = np.log1p(Gc); G_rate = np.log1p(Gr)
    cmap = mpl.cm.viridis.copy(); cmap.set_bad("#dddddd")
    cmap2 = mpl.cm.RdBu_r.copy(); cmap2.set_bad("#dddddd")
    yt = np.arange(0, T, 12); yl = [str(2010 + i) for i in range(10)]
    fig, axes = plt.subplots(1, 3, figsize=(14, 6), constrained_layout=True)
    vmax = np.nanmax([G_obs, G_rate]); rlim = np.nanmax(np.abs(Glf))
    for ax, G, title, cm, vmn, vmx, lab in [
        (axes[0], G_obs, "Observed  log(1+cases)", cmap, 0, vmax, "log(1+cases)"),
        (axes[1], G_rate, "Model  log(1+predicted rate)", cmap, 0, vmax, "log(1+rate)"),
        (axes[2], Glf, "Latent field  f = g+q+w", cmap2, -rlim, rlim, "log-risk")]:
        im = ax.imshow(G, aspect="auto", cmap=cm, vmin=vmn, vmax=vmx, interpolation="nearest")
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_yticks(range(S)); ax.set_yticklabels(ylabels, fontsize=5.5)
        ax.set_xticks(yt); ax.set_xticklabels(yl, fontsize=7, rotation=45)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label=lab)
    axes[0].set_ylabel("country (ordered by total cases)", fontsize=8)
    fig.suptitle("ECDC measles 2010-2019: observed vs HSGP model (countries × months)", fontsize=11, fontweight="bold")
    fig.savefig(FIG / "ecdc_heatmap.png", dpi=200, bbox_inches="tight"); plt.close(fig)


def fig_forest():
    hps = ["space_alpha", "space_length", "time_alpha", "time_length",
           "interaction_alpha", "interaction_space_length", "interaction_time_length", "sigma_h", "ell_h"]
    titles = [r"$\sigma_g$ (space amp)", r"$\ell_g$ (space len)", r"$\sigma_q$ (time amp)",
              r"$\ell_q$ (time len)", r"$\sigma_w$ (inter amp)", r"$\ell_{ws}$ (inter space len)",
              r"$\ell_{wt}$ (inter time len)", r"$\sigma_h$ (seasonal amp)", r"$\ell_h$ (seasonal len)"]
    grp = ["amp", "len", "amp", "len", "amp", "len", "len", "seas", "seas"]
    gcol = {"amp": "#1a7a3a", "len": "#7a4ba0", "seas": "#b5651d"}
    plt.rcParams.update({"font.size": 8.5})
    fig, axes = plt.subplots(3, 3, figsize=(10.5, 7.0))
    for ax, hp, title, g in zip(axes.ravel(), hps, titles, grp):
        s = np.array(R[hp + "_samples"])
        lo, hi = np.quantile(s, 0.005), np.quantile(s, 0.995); pad = (hi - lo) * 0.06
        xs = np.linspace(lo - pad, hi + pad, 400); ys = gaussian_kde(s)(xs)
        ax.fill_between(xs, ys, color=gcol[g], alpha=0.30); ax.plot(xs, ys, color=gcol[g], lw=1.6)
        m = s.mean(); q1, q2 = np.quantile(s, 0.025), np.quantile(s, 0.975)
        ax.axvline(m, color="#222", lw=1.3); ax.axvspan(q1, q2, color="#222", alpha=0.06)
        ax.set_title(title, fontsize=9.5, fontweight="bold", color=gcol[g], pad=3)
        ax.text(0.97, 0.92, f"{m:.2f}\n[{q1:.2f},{q2:.2f}]", transform=ax.transAxes, ha="right", va="top", fontsize=7)
        ax.set_yticks([]); ax.tick_params(labelsize=7.5); ax.grid(axis="x", alpha=0.15); ax.set_ylim(bottom=0)
    fig.suptitle("ECDC measles: joint 9-HP posteriors (HSGP+NB, 2010-2019; black line = mean, band = 95% CI)", fontsize=10, y=0.997)
    gl = [plt.Line2D([0], [0], color=gcol[k], lw=3, label=v) for k, v in [("amp", "amplitude"), ("len", "lengthscale"), ("seas", "seasonal")]]
    fig.legend(handles=gl, loc="lower center", ncol=3, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, -0.004))
    fig.tight_layout(rect=[0, 0.025, 1, 0.965]); fig.savefig(FIG / "ecdc_9hp_forest.png", dpi=200, bbox_inches="tight"); plt.close(fig)


def fig_timeseries():
    _, _, _, _, codemap, si, ti, cases, rate = _grids()
    tot = {s: cases[si == s].sum() for s in range(S)}
    top = sorted(range(S), key=lambda s: -tot[s])[:9]
    fig, axes = plt.subplots(3, 3, figsize=(13, 7.5))
    for ax, s in zip(axes.ravel(), top):
        m = si == s; t = ti[m]; o = np.argsort(t); t = t[o]
        ax.bar(t, cases[m][o], width=1.0, color="#bbb", label="observed", zorder=1)
        ax.plot(t, rate[m][o], color="#c0392b", lw=1.4, label="model rate", zorder=3)
        ax.set_title(codemap[s], fontsize=10, fontweight="bold")
        ax.set_xticks(np.arange(0, 120, 24)); ax.set_xticklabels([str(2010 + i) for i in range(0, 10, 2)], fontsize=7)
        ax.tick_params(labelsize=7); ax.grid(alpha=0.15)
    axes[0, 0].legend(fontsize=7.5, loc="upper right")
    fig.suptitle("ECDC measles: observed cases vs HSGP posterior rate — 9 highest-burden countries\n(does the model capture the outbreaks?)", fontsize=11, fontweight="bold")
    fig.supxlabel("year", fontsize=9); fig.supylabel("monthly cases", fontsize=9)
    fig.tight_layout(rect=[0.01, 0.01, 1, 0.96]); fig.savefig(FIG / "ecdc_timeseries_fit.png", dpi=200, bbox_inches="tight"); plt.close(fig)


def main():
    fig_seasonal();    print("[1/4] seasonal curve")
    fig_heatmap();     print("[2/4] heatmap")
    fig_forest();      print("[3/4] 9-HP forest")
    fig_timeseries();  print("[4/4] timeseries fit")
    print("Done ->", FIG)


if __name__ == "__main__":
    main()
