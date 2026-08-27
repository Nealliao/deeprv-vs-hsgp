"""ECDC HSGP vs DeepRV figures matching the Tycho suite: convergence+LOO, traces, PIT calibration, field recovery.
Loads data/processed/ecdc/ecdc_{hsgp,deeprv}_enhanced.npz (per-param rhat/ess, traces, loglik, pp_draws, eta)."""
import os, glob, numpy as np
os.environ.setdefault("MPLCONFIGDIR","/Users/Zhuanz/Desktop/ECDC_measles_HSGP_vs_DeepRV/.matplotlib")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.special import logsumexp
ROOT=os.path.expanduser("~/Desktop/ECDC_measles_HSGP_vs_DeepRV"); ED=f"{ROOT}/data/processed/ecdc"; FIG=f"{ROOT}/figures"
HS=np.load(f"{ED}/ecdc_hsgp_enhanced.npz",allow_pickle=True); DR=np.load(f"{ED}/ecdc_deeprv_enhanced.npz",allow_pickle=True)
Y0=2010; cnt=np.asarray(HS["cases"]); st=np.asarray(HS["state_index"]); ti=np.asarray(HS["time_index"])
ctry=[str(c) for c in HS["country"]]; NS=int(st.max()+1); T=int(ti.max()+1)
C_HS="#4C72B0"; C_DR="#C44E52"; C_OBS="#B0B0B0"
# country label per state_index
lab={}; [lab.setdefault(st[i],ctry[i]) for i in range(len(st))]; names=[lab.get(s,str(s)) for s in range(NS)]

# ===== FIG A: convergence + WAIC/ELPD =====
def waic(ll):
    S=ll.shape[0]; lppd=logsumexp(ll,0)-np.log(S); pw=ll.var(0); ei=lppd-pw
    return ei.sum(), np.sqrt(len(ei)*ei.var()), ei
eH,seH,eiH=waic(np.asarray(HS["loglik"])); eD,seD,eiD=waic(np.asarray(DR["loglik"]))
dW=(eiD-eiH).sum(); dWse=np.sqrt(len(eiH)*(eiD-eiH).var())
rng=np.random.default_rng(0)
fig=plt.figure(figsize=(16,5.5)); gs=fig.add_gridspec(1,3,width_ratios=[1.1,1.1,1.4])
ax=fig.add_subplot(gs[0]); rhH=np.asarray(HS["hp_rhat"]); rhD=np.asarray(DR["hp_rhat"])
ax.scatter(rhH,1+0.11*rng.standard_normal(len(rhH)),color=C_HS,s=45,label="HSGP")
ax.scatter(rhD,0+0.11*rng.standard_normal(len(rhD)),color=C_DR,marker="x",s=45,label="DeepRV")
ax.axvline(1.01,color="k",ls="--",lw=1); ax.set_yticks([0,1]); ax.set_yticklabels(["DeepRV","HSGP"]); ax.set_ylim(-0.6,1.6)
ax.set_title("per-parameter R-hat (<1.01)"); ax.legend(fontsize=9); ax.set_xlim(0.999,max(1.012,rhH.max(),rhD.max())+0.001)
ax2=fig.add_subplot(gs[1]); esH=np.asarray(HS["hp_ess"]); esD=np.asarray(DR["hp_ess"])
ax2.scatter(esH,1+0.11*rng.standard_normal(len(esH)),color=C_HS,s=45); ax2.scatter(esD,0+0.11*rng.standard_normal(len(esD)),color=C_DR,marker="x",s=45)
ax2.axvline(400,color="k",ls="--",lw=1); ax2.set_yticks([0,1]); ax2.set_yticklabels(["DeepRV","HSGP"]); ax2.set_ylim(-0.6,1.6); ax2.set_title("per-parameter ESS (>400)")
ax3=fig.add_subplot(gs[2]); ax3.axis("off")
tbl=[f"ECDC measles 2010-2019 (n={len(cnt)}, S={NS}, T={T})","",
     f"{'':12s}{'HSGP':>12s}{'DeepRV':>12s}",
     f"{'max R-hat':12s}{float(HS['max_rhat']):>12.4f}{float(DR['max_rhat']):>12.4f}",
     f"{'min ESS':12s}{float(HS['min_ess']):>12.0f}{float(DR['min_ess']):>12.0f}",
     f"{'div':12s}{int(HS['div']):>12d}{int(DR['div']):>12d}",
     f"{'runtime(s)':12s}{float(HS['runtime']):>12.0f}{float(DR['runtime']):>12.0f}",
     f"{'cov95':12s}{float(HS['pp_cov95']):>12.3f}{float(DR['pp_cov95']):>12.3f}",
     f"{'RMSE':12s}{float(HS['pp_rmse']):>12.1f}{float(DR['pp_rmse']):>12.1f}","","--- out-of-sample ELPD (WAIC) ---",
     f"HSGP {eH:,.0f} ± {seH:.0f}   DeepRV {eD:,.0f} ± {seD:.0f}",
     f"diff (DeepRV-HSGP) = {dW:,.0f} ± {dWse:.0f}  ({'DeepRV' if dW>0 else 'HSGP'} better, {abs(dW)/dWse:.1f} SE)"]
ax3.text(0,1,"\n".join(tbl),va="top",ha="left",family="monospace",fontsize=10.5)
fig.suptitle("ECDC: convergence + ELPD — HSGP vs DeepRV (4 chains)",fontsize=13,fontweight="bold")
fig.tight_layout(rect=[0,0,1,0.94]); fig.savefig(f"{FIG}/ecdc_convergence_loo.png",dpi=130); plt.close(fig); print("saved ecdc_convergence_loo.png")

# ===== FIG B: traces =====
tr_keys=[("interaction amplitude","trace_interaction_alpha"),("inter-time lengthscale","trace_interaction_time_length"),
         ("NB 1/sqrt(kappa)","trace_inv_sqrt_kappa"),("seasonal sigma_h","trace_sigma_h")]
fig,axes=plt.subplots(len(tr_keys),2,figsize=(15,10))
for i,(ttl,k) in enumerate(tr_keys):
    for j,(D,c,nm) in enumerate([(HS,C_HS,"HSGP"),(DR,C_DR,"DeepRV")]):
        a=axes[i,j]
        if k in D.files:
            tr=np.asarray(D[k])
            for c2 in range(tr.shape[0]): a.plot(tr[c2],lw=0.6,alpha=0.8)
        a.set_title(f"{nm}: {ttl}",fontsize=10)
fig.suptitle("ECDC trace plots (4 chains) — HSGP (left) vs DeepRV (right)",fontsize=13,fontweight="bold")
fig.tight_layout(rect=[0,0,1,0.96]); fig.savefig(f"{FIG}/ecdc_traces.png",dpi=130); plt.close(fig); print("saved ecdc_traces.png")

# ===== FIG C: predictive calibration =====
ppH=np.asarray(HS["pp_draws"]).astype(float); ppD=np.asarray(DR["pp_draws"]).astype(float)
noms=np.array([0.5,0.6,0.7,0.8,0.9,0.95,0.99])
def cov(pp,q): lo=np.quantile(pp,(1-q)/2,0); hi=np.quantile(pp,(1+q)/2,0); return ((cnt>=lo)&(cnt<=hi)).mean()
covH=[cov(ppH,q) for q in noms]; covD=[cov(ppD,q) for q in noms]
def pit(pp): lt=(pp<cnt[None,:]).mean(0); eq=(pp==cnt[None,:]).mean(0); return lt+np.random.default_rng(0).random(len(cnt))*eq
fig,ax=plt.subplots(1,3,figsize=(18,5.5))
a=ax[0]; a.plot([0,1],[0,1],"k--",lw=1,label="perfect"); a.plot(noms,covH,"o-",color=C_HS,lw=2,label="HSGP"); a.plot(noms,covD,"s-",color=C_DR,lw=2,label="DeepRV")
a.set_xlabel("nominal"); a.set_ylabel("empirical coverage"); a.set_title("Calibration curve"); a.legend(); a.grid(alpha=.3); a.set_xlim(.45,1.02); a.set_ylim(.45,1.02)
for a,p,c,nm in [(ax[1],pit(ppH),C_HS,"HSGP"),(ax[2],pit(ppD),C_DR,"DeepRV")]:
    a.hist(p,bins=20,range=(0,1),color=c,alpha=0.8,density=True,edgecolor="white"); a.axhline(1,color="k",ls="--"); a.set_title(f"PIT — {nm}"); a.set_xlabel("PIT"); a.set_ylim(0,2)
fig.suptitle(f"ECDC: posterior predictive calibration — HSGP vs DeepRV (n={len(cnt)})",fontsize=13,fontweight="bold")
fig.tight_layout(rect=[0,0,1,0.93]); fig.savefig(f"{FIG}/ecdc_predictive_calibration.png",dpi=130); plt.close(fig)
print("saved ecdc_predictive_calibration.png  cov95 HSGP %.3f DeepRV %.3f"%(covH[5],covD[5]))

# ===== FIG D: field recovery (viridis + RdBu residual) =====
def grid(v): g=np.full((NS,T),np.nan,np.float32); g[st,ti]=v; return g
obs=grid(np.log1p(cnt)); fH=grid(np.asarray(HS["eta_mean"])); fD=grid(np.asarray(DR["eta_mean"]))
order=np.argsort([-cnt[st==s].sum() for s in range(NS)]); nm=[names[i] for i in order]
obs,fH,fD=obs[order],fH[order],fD[order]; rH=fH-obs; rD=fD-obs
vmax=np.nanpercentile(obs,99.5); rmax=np.nanpercentile(np.abs(np.concatenate([rH,rD])),99); ext=[Y0,Y0+T/12.,NS-.5,-.5]
fig,ax=plt.subplots(2,3,figsize=(19,10))
def hm(a,g,t,cm,vn,vx,yl=False):
    im=a.imshow(g,aspect="auto",cmap=cm,vmin=vn,vmax=vx,extent=ext,interpolation="nearest"); a.set_title(t,fontsize=11,fontweight="bold")
    a.set_yticks(range(NS)); a.set_yticklabels(nm,fontsize=7);
    if yl: a.set_ylabel("country (high→low burden)")
    return im
i1=hm(ax[0,0],obs,"OBSERVED log(count)","viridis",0,vmax,True); hm(ax[0,1],fH,"HSGP posterior mean","viridis",0,vmax); hm(ax[0,2],fD,"DeepRV posterior mean","viridis",0,vmax)
fig.colorbar(i1,ax=ax[0,:],shrink=.7,label="log(count)"); ax[1,0].axis("off")
ax[1,0].text(0,1,("RMSE  HSGP %.1f  DeepRV %.1f\ncov95 HSGP %.3f  DeepRV %.3f\nWAIC diff %.0f ± %.0f (%s better)"
    %(float(HS['pp_rmse']),float(DR['pp_rmse']),float(HS['pp_cov95']),float(DR['pp_cov95']),dW,dWse,'DeepRV' if dW>0 else 'HSGP')),va="top",family="monospace",fontsize=12,transform=ax[1,0].transAxes)
i2=hm(ax[1,1],rH,"HSGP residual (fit-obs)","RdBu_r",-rmax,rmax); hm(ax[1,2],rD,"DeepRV residual (fit-obs)","RdBu_r",-rmax,rmax)
for a in (ax[1,1],ax[1,2]): a.set_xlabel("year")
fig.colorbar(i2,ax=ax[1,1:],shrink=.7,label="residual")
fig.suptitle("ECDC field recovery — observed vs HSGP vs DeepRV (2010-2019 monthly)",fontsize=14,fontweight="bold")
fig.savefig(f"{FIG}/ecdc_field_recovery.png",dpi=130,bbox_inches="tight"); plt.close(fig); print("saved ecdc_field_recovery.png")

# ===== FIG E: top-9 highest-burden countries observed vs rate (HSGP vs DeepRV) =====
tot=np.array([cnt[st==s].sum() for s in range(NS)]); top=np.argsort(tot)[::-1][:9]
etaH=np.asarray(HS["eta_thin"]); etaD=np.asarray(DR["eta_thin"])
muH=np.exp(etaH).mean(0); muD=np.exp(etaD).mean(0)
fig,axes=plt.subplots(3,3,figsize=(18,11)); axes=axes.ravel()
for ax,s in zip(axes,top):
    m=st==s; oy=ti[m]; order=np.argsort(oy); yy=Y0+oy[order]/12.
    ax.bar(yy,cnt[m][order],width=1/12.,color=C_OBS,label="observed")
    ax.plot(yy,muH[m][order],color=C_HS,lw=1.7,label="HSGP rate")
    ax.plot(yy,muD[m][order],color=C_DR,lw=1.7,label="DeepRV rate")
    ax.set_title(names[s],fontsize=12,fontweight="bold"); ax.margins(x=0.01)
axes[0].legend(fontsize=10,framealpha=0.9)
fig.suptitle("ECDC: observed cases vs posterior model rate — 9 highest-burden countries\n(HSGP vs DeepRV; does each capture the outbreaks?)",fontsize=14,fontweight="bold")
fig.supxlabel("year"); fig.supylabel("monthly cases")
fig.tight_layout(rect=[0,0,1,0.95]); fig.savefig(f"{FIG}/ecdc_key_countries_obs_vs_rate.png",dpi=130); plt.close(fig); print("saved ecdc_key_countries_obs_vs_rate.png")
print(f"\nECDC figures -> {FIG}/")
