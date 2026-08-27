"""
make_all_figures.py
================================================================================
Self-contained: regenerates all 8 ECDC figures from the two result JSONs.
Reads ../results/ecdc_{hsgp,deeprv}_joint9hp_nb.json, writes ../figures/*.png.
Only numpy/scipy/matplotlib. Run: python make_all_figures.py
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"
FIG = HERE.parent / "figures"; FIG.mkdir(exist_ok=True)
H = json.load(open(RES / "ecdc_hsgp_joint9hp_nb.json"))
D = json.load(open(RES / "ecdc_deeprv_joint9hp_nb.json"))
S, T = 29, 120
HSGP_C, DRV_C = "#2b6cb0", "#dd6b20"
HPS = ["space_alpha", "space_length", "time_alpha", "time_length",
       "interaction_alpha", "interaction_space_length", "interaction_time_length", "sigma_h", "ell_h"]
TIT = [r"$\sigma_g$ (space amp)", r"$\ell_g$ (space len)", r"$\sigma_q$ (time amp)",
       r"$\ell_q$ (time len)", r"$\sigma_w$ (inter amp)", r"$\ell_{ws}$ (inter space len)",
       r"$\ell_{wt}$ (inter time len)", r"$\sigma_h$ (seasonal amp)", r"$\ell_h$ (seasonal len)"]
GRP = ["amp", "len", "amp", "len", "amp", "len", "len", "seas", "seas"]
GC = {"amp": "#1a7a3a", "len": "#7a4ba0", "seas": "#b5651d"}
MN = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _grids(obj):
    si = np.array(obj["state_index"]); ti = np.array(obj["time_index"]); ct = np.array(obj["country"])
    cases = np.array(obj["cases"]); rate = np.array(obj["rate_post_mean"]); lf = np.array(obj["latent_f_post_mean"])
    cm = {int(si[i]): ct[i] for i in range(len(si))}
    tot = {s: cases[si == s].sum() for s in range(S)}
    order = sorted(range(S), key=lambda s: -tot[s]); pos = {s: i for i, s in enumerate(order)}
    def g(v):
        out = np.full((S, T), np.nan)
        for i in range(len(si)): out[pos[int(si[i])], int(ti[i])] = v[i]
        return out
    return g, cm, order, si, ti, cases, rate, lf


def fig_seasonal():
    mo = np.arange(1, 13); m = np.array(H["h_monthly_post_mean"]); lo = np.array(H["h_monthly_post_q025"]); hi = np.array(H["h_monthly_post_q975"])
    fig, ax = plt.subplots(figsize=(7.5, 4.2)); ax.axhline(0, color="#999", lw=0.8, ls=":")
    ax.fill_between(mo, lo, hi, color=HSGP_C, alpha=0.18, label="95% CI"); ax.plot(mo, m, "o-", color=HSGP_C, lw=2, ms=6, label="posterior mean")
    ax.axvspan(2.5, 5.5, color="#e8c14a", alpha=0.12)
    pk = mo[np.argmax(m)]; ax.annotate(f"peak: month {pk}", xy=(pk, m.max()), xytext=(pk + 0.3, m.max() + 0.05), fontsize=9, color="#c0392b")
    ax.set_xticks(mo); ax.set_xticklabels(MN, fontsize=8); ax.set_ylabel(r"seasonal log-risk $h_m$"); ax.set_xlabel("month of year")
    ax.set_title("ECDC measles seasonal effect (data-driven, 2010-2019)\nHSGP joint-9HP, NB", fontsize=10.5, fontweight="bold")
    ax.grid(alpha=0.15); ax.legend(fontsize=8, loc="upper right"); fig.tight_layout()
    fig.savefig(FIG / "ecdc_seasonal_curve.png", dpi=200, bbox_inches="tight"); plt.close(fig)


def fig_heatmap():
    g, cm, order, *_ = _grids(H)
    Gobs = np.log1p(g(np.array(H["cases"]))); Grate = np.log1p(g(np.array(H["rate_post_mean"]))); Glf = g(np.array(H["latent_f_post_mean"]))
    cv = mpl.cm.viridis.copy(); cv.set_bad("#ddd"); cr = mpl.cm.RdBu_r.copy(); cr.set_bad("#ddd")
    yl = [cm[s] for s in order]; yt = np.arange(0, T, 12); yy = [str(2010 + i) for i in range(10)]
    vmax = np.nanmax([Gobs, Grate]); rl = np.nanmax(np.abs(Glf))
    fig, ax = plt.subplots(1, 3, figsize=(14, 6), constrained_layout=True)
    for a, G, t, c, vm, vx, lb in [(ax[0], Gobs, "Observed log(1+cases)", cv, 0, vmax, "log(1+cases)"),
                                    (ax[1], Grate, "Model log(1+rate)", cv, 0, vmax, "log(1+rate)"),
                                    (ax[2], Glf, "Latent field f=g+q+w", cr, -rl, rl, "log-risk")]:
        im = a.imshow(G, aspect="auto", cmap=c, vmin=vm, vmax=vx, interpolation="nearest"); a.set_title(t, fontsize=10, fontweight="bold")
        a.set_yticks(range(S)); a.set_yticklabels(yl, fontsize=5.5); a.set_xticks(yt); a.set_xticklabels(yy, fontsize=7, rotation=45)
        fig.colorbar(im, ax=a, fraction=0.046, pad=0.02, label=lb)
    ax[0].set_ylabel("country (by total cases)", fontsize=8)
    fig.suptitle("ECDC measles 2010-2019: observed vs HSGP model", fontsize=11, fontweight="bold")
    fig.savefig(FIG / "ecdc_heatmap.png", dpi=200, bbox_inches="tight"); plt.close(fig)


def _forest(objs_cols, fname, suptitle, boxed=()):
    plt.rcParams.update({"font.size": 8.5}); fig, axes = plt.subplots(3, 3, figsize=(11, 7.2))
    for ax, hp, t, gp in zip(axes.ravel(), HPS, TIT, GRP):
        for y, obj, col in objs_cols:
            m = obj[hp + "_post_mean"]; lo = obj[hp + "_post_q025"]; hi = obj[hp + "_post_q975"]
            ax.plot([lo, hi], [y, y], color=col, lw=3.4, alpha=0.55, solid_capstyle="round", zorder=2); ax.plot(m, y, "o", color=col, ms=8, zorder=3)
        ax.set_yticks([c[0] for c in objs_cols]); ax.set_yticklabels([("HSGP" if c[0] == 1 else "DeepRV") for c in objs_cols], fontsize=8)
        ax.set_ylim(-0.7, 1.7) if len(objs_cols) == 2 else None
        ax.set_title(t, fontsize=9.5, fontweight="bold", color=GC[gp], pad=3); ax.grid(axis="x", alpha=0.18); ax.tick_params(labelsize=7.5); ax.margins(x=0.10)
    for r, c in boxed:
        for sp in axes[r, c].spines.values(): sp.set_edgecolor("#c0392b"); sp.set_linewidth(2.0)
    fig.suptitle(suptitle, fontsize=10, y=1.0); fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(FIG / fname, dpi=200, bbox_inches="tight"); plt.close(fig)


def fig_forest_hsgp():
    _forest([(1.0, H, HSGP_C)], "ecdc_9hp_forest.png", "ECDC measles: HSGP joint-9HP posteriors (point=mean, bar=95% CI)")


def fig_cmp_forest():
    _forest([(1.0, H, HSGP_C), (0.0, D, DRV_C)], "ecdc_hsgp_vs_deeprv_9hp.png",
            "ECDC: HSGP vs DeepRV joint 9-HP posteriors (HSGP top / DeepRV bottom; red box = the components that differ most)", boxed=[(0, 0), (1, 1)])


def fig_cmp_kde():
    plt.rcParams.update({"font.size": 8.5}); fig, axes = plt.subplots(3, 3, figsize=(11, 7.2))
    for ax, hp, t, gp in zip(axes.ravel(), HPS, TIT, GRP):
        sh = np.array(H[hp + "_samples"]); sd = np.array(D[hp + "_samples"])
        lo = min(np.quantile(sh, 0.002), np.quantile(sd, 0.002)); hi = max(np.quantile(sh, 0.998), np.quantile(sd, 0.998)); pad = (hi - lo) * 0.05
        xs = np.linspace(lo - pad, hi + pad, 400)
        for s, col in [(sh, HSGP_C), (sd, DRV_C)]:
            ys = gaussian_kde(s)(xs); ax.fill_between(xs, ys, color=col, alpha=0.28); ax.plot(xs, ys, color=col, lw=1.6)
        ax.set_title(t, fontsize=9.5, fontweight="bold", color=GC[gp], pad=3); ax.set_yticks([]); ax.tick_params(labelsize=7.5); ax.grid(axis="x", alpha=0.15); ax.set_ylim(bottom=0)
    for r, c in [(0, 0), (1, 1)]:
        for sp in axes[r, c].spines.values(): sp.set_edgecolor("#c0392b"); sp.set_linewidth(2.0)
    fig.suptitle("ECDC: HSGP (blue) vs DeepRV (orange) 9-HP posterior densities", fontsize=10, y=0.998); fig.tight_layout(rect=[0, 0.02, 1, 0.965])
    fig.savefig(FIG / "ecdc_hsgp_vs_deeprv_kde.png", dpi=200, bbox_inches="tight"); plt.close(fig)


def fig_cmp_recovery():
    si = np.array(H["state_index"]); ti = np.array(H["time_index"]); ct = np.array(H["country"])
    cases = np.array(H["cases"]); rh = np.array(H["rate_post_mean"]); rd = np.array(D["rate_post_mean"])
    cm = {int(si[i]): ct[i] for i in range(len(si))}; tot = {s: cases[si == s].sum() for s in range(S)}; top = sorted(range(S), key=lambda s: -tot[s])[:9]
    fig, axes = plt.subplots(3, 3, figsize=(13, 7.5))
    for ax, s in zip(axes.ravel(), top):
        m = si == s; t = ti[m]; o = np.argsort(t); t = t[o]
        ax.bar(t, cases[m][o], width=1.0, color="#ccc", label="observed", zorder=1)
        ax.plot(t, rh[m][o], color=HSGP_C, lw=1.5, label="HSGP", zorder=3); ax.plot(t, rd[m][o], color=DRV_C, lw=1.5, label="DeepRV", zorder=3, alpha=0.85)
        ax.set_title(cm[s], fontsize=10, fontweight="bold"); ax.set_xticks(np.arange(0, 120, 24)); ax.set_xticklabels([str(2010 + i) for i in range(0, 10, 2)], fontsize=7); ax.tick_params(labelsize=7); ax.grid(alpha=0.15)
    axes[0, 0].legend(fontsize=7, loc="upper right")
    fig.suptitle("ECDC: data recovery — observed vs HSGP vs DeepRV rate (9 top-burden countries)", fontsize=11, fontweight="bold")
    fig.supxlabel("year", fontsize=9); fig.supylabel("monthly cases", fontsize=9); fig.tight_layout(rect=[0.01, 0.01, 1, 0.97])
    fig.savefig(FIG / "ecdc_data_recovery_compare.png", dpi=200, bbox_inches="tight"); plt.close(fig)


def fig_cmp_heatmap():
    g, cm, order, *_ = _grids(H)
    Gobs = np.log1p(g(np.array(H["cases"]))); Grh = np.log1p(g(np.array(H["rate_post_mean"]))); Grd = np.log1p(g(np.array(D["rate_post_mean"])))
    Glh = g(np.array(H["latent_f_post_mean"])); Gld = g(np.array(D["latent_f_post_mean"])); Gdf = Gld - Glh
    cv = mpl.cm.viridis.copy(); cv.set_bad("#ddd"); cr = mpl.cm.RdBu_r.copy(); cr.set_bad("#ddd")
    yl = [cm[s] for s in order]; yt = np.arange(0, T, 12); yy = [str(2010 + i) for i in range(10)]
    vmax = np.nanmax([Gobs, Grh, Grd]); ll = np.nanmax(np.abs([Glh, Gld])); dl = np.nanmax(np.abs(Gdf))
    fig, axes = plt.subplots(2, 3, figsize=(14, 9), constrained_layout=True)
    P = [(0, 0, Gobs, "Observed log(1+cases)", cv, 0, vmax, "log(1+cases)"), (0, 1, Grh, "HSGP predicted", cv, 0, vmax, "log(1+rate)"),
         (0, 2, Grd, "DeepRV predicted", cv, 0, vmax, "log(1+rate)"), (1, 0, Glh, "HSGP latent f", cr, -ll, ll, "log-risk"),
         (1, 1, Gld, "DeepRV latent f", cr, -ll, ll, "log-risk"), (1, 2, Gdf, "DeepRV f - HSGP f", cr, -dl, dl, "diff")]
    for r, c, G, t, cmp, vm, vx, lb in P:
        a = axes[r, c]; im = a.imshow(G, aspect="auto", cmap=cmp, vmin=vm, vmax=vx, interpolation="nearest"); a.set_title(t, fontsize=9.5, fontweight="bold")
        a.set_yticks(range(S)); a.set_yticklabels(yl, fontsize=5); a.set_xticks(yt); a.set_xticklabels(yy, fontsize=6.5, rotation=45); fig.colorbar(im, ax=a, fraction=0.046, pad=0.02, label=lb)
    fig.suptitle("ECDC: HSGP vs DeepRV — data recovery (top) and latent-field decomposition (bottom)", fontsize=11, fontweight="bold")
    fig.savefig(FIG / "ecdc_hsgp_vs_deeprv_heatmap.png", dpi=190, bbox_inches="tight"); plt.close(fig)


def main():
    for fn, name in [(fig_seasonal, "seasonal"), (fig_heatmap, "HSGP heatmap"), (fig_forest_hsgp, "HSGP forest"),
                     (fig_cmp_forest, "cmp forest"), (fig_cmp_kde, "cmp KDE"), (fig_cmp_recovery, "cmp recovery"),
                     (fig_cmp_heatmap, "cmp heatmap")]:
        fn(); print(f"  {name}")
    print("Done ->", FIG)


if __name__ == "__main__":
    main()
