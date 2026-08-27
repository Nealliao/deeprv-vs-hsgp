"""ECDC prior + HSGP-basis audit: Betancourt containment + §4.1 separation + Riutort-Mayol (SE) basis dims."""
import numpy as np, pandas as pd, json
from scipy.spatial.distance import pdist
from scipy.spatial import cKDTree
from scipy.stats import invgamma
ROOT="/Users/Zhuanz/Desktop/ECDC_measles_HSGP_vs_DeepRV"
p=pd.read_csv(f"{ROOT}/data/processed/ecdc_measles_panel.csv")
coords=p.drop_duplicates("state_index").sort_values("state_index")[["x_coord","y_coord"]].to_numpy()
S=len(coords); T=int(p["time_index"].nunique())
dd,_=cKDTree(coords).query(coords,k=2); nn=np.median(dd[:,1]); minnn=dd[:,1].min()
Sx=np.max(np.abs(coords[:,0])); Sy=np.max(np.abs(coords[:,1])); Ss=max(Sx,Sy)   # spatial half-range
span_s=pdist(coords).max()
tc=np.arange(T)-np.arange(T).mean(); St=np.max(np.abs(tc))                       # time half-range (months)
print("="*74); print(f"ECDC geometry: S={S} countries, T={T} months")
print(f"  spatial: median-NN={nn:.3f} min-NN={minnn:.3f} half-range S={Ss:.3f} max-pairwise={span_s:.3f}")
print(f"  temporal: min-dist=1mo half-range S={St:.1f}mo span={2*St:.0f}mo")
print("="*74)
# priors (from script 111)
PR={"space":(6.1091,3.3175,"[0.25,1.80]"),"inter_space":(5.3661,0.3648,"[0.03,0.25]"),
    "trend":(6.5707,167.37,"[12,80]mo"),"inter_time":(6.2718,27.02,"[2,14]mo")}
def band(a,b): return invgamma.ppf(0.01,a,scale=b),invgamma.ppf(0.99,a,scale=b)
def ovl(a1,b1,a2,b2,hi):
    x=np.linspace(1e-4,hi,300000); f1=invgamma.pdf(x,a1,scale=b1); f2=invgamma.pdf(x,a2,scale=b2)
    return float(np.trapezoid(np.minimum(f1,f2),x))
print("\n--- Betancourt CONTAINMENT (prior 98% band vs [min-dist, span]) ---")
print("  spatial min-NN=%.3f span=%.2f | temporal min=1 span=%.0f"%(minnn,span_s,2*St))
for nm,(a,b,tgt) in PR.items():
    lo,hi=band(a,b); print(f"  {nm:12s} IG({a:.2f},{b:.2f}) 98%-band=[{lo:.3f},{hi:.3f}]  (target {tgt})")
print("\n--- Betancourt §4.1 SEPARATION (overlap coefficient, want <=~2%) ---")
o_s=ovl(*PR['space'][:2],*PR['inter_space'][:2],3.0)
o_t=ovl(*PR['trend'][:2],*PR['inter_time'][:2],120.0)
print(f"  space g  vs inter_space  (spatial): OVL={100*o_s:5.2f}%   {'OK' if o_s<=0.02 else 'overlap>2%'}")
print(f"  trend q  vs inter_time   (time)   : OVL={100*o_t:5.2f}%   {'OK' if o_t<=0.02 else 'overlap>2%'}")
print(f"  NOTE season period=12mo; inter_time 98%-upper={band(*PR['inter_time'][:2])[1]:.1f}mo (aliases season if >~12)")
print("\n--- Riutort-Mayol (SE: m>=1.75*c/(ell/S), boundary c>=3.2*(ell/S)) ---")
d=json.load(open(f"{ROOT}/results/ecdc_hsgp_joint9hp_nb.json"))
# component: (posterior ell, half-range S, boundary L, used m)
comps=[("space g",   d["space_length_post_mean"],            Ss, 1.4, 8),
       ("trend q",   d["time_length_post_mean"],             St,  75., 35),
       ("inter-space",d["interaction_space_length_post_mean"],Ss, 1.0, 8),
       ("inter-time",d["interaction_time_length_post_mean"], St,  75., 15)]
for nm,ell,Sh,L,m in comps:
    c=L/Sh; r=ell/Sh; mreq=1.75*c/r; creq=max(3.2*r,1.2)
    okm="OK" if m>=mreq else "UNDER"; okc="OK" if c>=creq else "c-small"
    print(f"  {nm:12s} ell={ell:7.3f} S={Sh:6.2f} ell/S={r:.3f} c=L/S={c:.2f} | m_req={mreq:5.0f} (used {m:2d}) {okm} | c_req={creq:.2f} {okc}")
print("\n  interaction full-resolution basis = M_WS^2 * M_WT (if inter-space were resolved):")
r_ws=d["interaction_space_length_post_mean"]/Ss; r_wt=d["interaction_time_length_post_mean"]/St
print(f"    = {1.75*(1.0/Ss)/r_ws:.0f}^2 * {1.75*(75./St)/r_wt:.0f} = {(1.75*(1.0/Ss)/r_ws)**2*(1.75*(75./St)/r_wt):.0f}  (vs used 8^2*15={64*15})")
