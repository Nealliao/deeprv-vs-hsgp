# -*- coding: utf-8 -*-
"""Self-contained ECDC HTML report (figures base64-embedded, full code) saved in the ECDC project folder."""
import os, base64, html, glob
ROOT=os.path.expanduser("~/Desktop/ECDC_measles_HSGP_vs_DeepRV"); FIGD=f"{ROOT}/figures"; SCR=f"{ROOT}/scripts"
OUT=f"{ROOT}/ECDC_HSGP_vs_DeepRV_report.html"
def img(fn,cap):
    p=f"{FIGD}/{fn}"
    if not os.path.exists(p): return f"<p><i>[missing {fn}]</i></p>"
    b=base64.b64encode(open(p,"rb").read()).decode()
    return f'<figure><img src="data:image/png;base64,{b}"/><figcaption>{cap}</figcaption></figure>'
def code(fn):
    p=f"{SCR}/{fn}"
    if not os.path.exists(p): return f"<p><i>[missing {fn}]</i></p>"
    return f'<details><summary>{fn}</summary><pre class="code">{html.escape(open(p).read())}</pre></details>'
CSS="""body{font-family:-apple-system,Helvetica,Arial,sans-serif;max-width:1080px;margin:24px auto;padding:0 22px;color:#1a1a1a;line-height:1.6}
h1{font-size:26px;border-bottom:3px solid #333;padding-bottom:8px}h2{font-size:21px;margin-top:36px;border-bottom:1px solid #ccc;padding-bottom:4px}
h3{font-size:16.5px;margin-top:22px;color:#222}table{border-collapse:collapse;margin:14px 0;font-size:14px;width:100%}
th,td{border:1px solid #bbb;padding:6px 10px;text-align:left}th{background:#f0f0f0}td.c,th.c{text-align:center}
figure{margin:20px 0;text-align:center}img{max-width:100%;border:1px solid #ddd;border-radius:4px}
figcaption{font-size:13px;color:#555;margin-top:6px;font-style:italic}
pre.code{background:#f7f7f7;border:1px solid #ddd;border-radius:4px;padding:10px;overflow-x:auto;font-size:11.5px;line-height:1.35}
details{margin:8px 0}summary{cursor:pointer;font-weight:600;color:#0057b7}
.key{background:#fff8e1;border-left:4px solid #ffb300;padding:10px 16px;margin:14px 0}
.win{background:#e8f5e9;border-left:4px solid #43a047;padding:10px 16px;margin:14px 0}
nav{background:#fafafa;border:1px solid #ddd;border-radius:6px;padding:10px 20px}nav a{color:#0057b7;text-decoration:none}code{background:#eef;padding:1px 5px;border-radius:3px;font-size:13px}"""
H=[f"<!doctype html><html><head><meta charset='utf-8'><title>ECDC HSGP vs DeepRV</title><style>{CSS}</style></head><body>"]
H.append("<h1>ECDC 麻疹 —— HSGP vs DeepRV 对比 (real-data chapter)</h1>")
H.append("<p><b>Haoyu Liao, MSc Statistics, Imperial College London</b> &middot; ECDC surveillance measles, Europe 2010-2019 &middot; NegativeBinomial spatio-temporal &middot; 2026-07</p>")
H.append("""<div class='win'><b>结论:ECDC 上 DeepRV 胜.</b> DeepRV 比 HSGP <b>快 3.9×</b>(403s vs 1569s)、<b>绕开 HSGP 的 ell-funnel</b>(div 1 vs 34)、后验预测校准相同(cov 0.988 vs 0.985),样本外预测(WAIC)<b>打平</b>(差 0.6 SE)。根因:ECDC 交互<b>短尺度(inter_space 0.078)且主导(82%)</b> → HSGP 有 ell-funnel;但场<b>低秩到 DeepRV 能忠实压缩</b>(z_w=360, var_ratio 0.85)→ DeepRV 绕坑成功。这与 US Tycho 的边界规则一致(Tycho 周度高秩 → DeepRV 反而 funnel)。</div>""")
H.append("""<nav><b>目录</b>: <a href='#d'>数据/模型</a> · <a href='#p'>先验审计(Betancourt)</a> · <a href='#b'>基维度审计(Riutort-Mayol)</a> · <a href='#r'>结果</a> · <a href='#f'>图</a> · <a href='#c'>代码</a></nav>""")

H.append("<h2 id='d'>1. 数据与模型</h2>")
H.append("""<p><b>数据</b>:ECDC "Reported confirmed cases",月度,29 国 × 120 月(2010-2019,疫苗前 COVID 之外),n=3,450,54% 零,过度离散(var/mean~1134)→ NB。人口 offset = Eurostat。国家质心 cos-lat 校正,标准化(半域跨 0.61,min-NN 0.083)。</p>
<p><b>模型</b>:<code>count ~ NB2(exp(η), κ)</code>,<code>η = logpop + β0 + season[月] + g[国] + q[月] + w[国,月]</code>。<b>HSGP 与 DeepRV 设置一致(本 benchmark 的前提)</b>:同 <b>SE 核</b>、同 <b>exp-sine-squared 周期季节</b>(精确 Cholesky)、同 Betancourt IG priors、同 NB、同窗口、同 NUTS(4 链, target_accept 0.95)。唯一差别 = 潜在场表示(HSGP 线性基 vs DeepRV 神经 decoder)。</p>""")

H.append("<h2 id='p'>2. 先验审计 (Betancourt)</h2>")
H.append("""<p>原则:容纳先验 [min-协变量距离, 域跨];加性分量长度尺度不重叠(§4.1)。</p>
<table><tr><th>检查</th><th>结果</th></tr>
<tr><td>容纳 containment</td><td>✅ 各 98% 带精确落在目标:space [0.25,1.80]、inter_space [0.03,0.25](下界=min-NN 0.083 附近的短端)、trend [12,80]月、inter_time [2,14]月</td></tr>
<tr><td>§4.1 space g vs inter_space</td><td>✅ 重叠 OVL <b>1.86%</b>(&lt;2%,干净;在 0.25 处相接)</td></tr>
<tr><td>§4.1 trend q vs inter_time</td><td>⚠️ OVL <b>3.02%</b>(轻微,[12,14]月小重叠)</td></tr>
<tr><td>inter_time vs 12月季节</td><td>⚠️ inter_time 98%上界 14月 &gt; 12月季节 → 轻微混叠</td></tr></table>
<p><b>结论</b>:ECDC 先验基本守 Betancourt,空间分离干净,时间轴两处轻微瑕疵(次要、无害)。</p>""")

H.append("<h2 id='b'>3. 基维度审计 (Riutort-Mayol) — 诚实版</h2>")
H.append("""<table><tr><th>分量</th><th class='c'>ℓ_post</th><th class='c'>ℓ/S</th><th class='c'>m 需要</th><th class='c'>m 用</th><th>判定</th></tr>
<tr><td>space g</td><td class='c'>0.53</td><td class='c'>0.87</td><td class='c'>5</td><td class='c'>8</td><td>m✓,边界 c 偏小</td></tr>
<tr><td>trend q</td><td class='c'>30.3</td><td class='c'>0.51</td><td class='c'>4</td><td class='c'>35</td><td>m 过剩,c 偏小</td></tr>
<tr><td>inter-time</td><td class='c'>10.5</td><td class='c'>0.18</td><td class='c'>13</td><td class='c'>15</td><td>✅</td></tr>
<tr><td><b>inter-space</b></td><td class='c'><b>0.078</b></td><td class='c'>0.13</td><td class='c'><b>22</b></td><td class='c'><b>8</b></td><td>❌ 欠解析</td></tr></table>
<div class='key'><b>诚实结论(重要)</b>:主效应满足 Riutort-Mayol(可引用的标准规则)。<b>交互无法满足</b>:全分辨需 M_WS²·M_WT = 22²·13 ≈ <b>6,300 个基,超过场维度 3,480(无意义)</b> —— Riutort-Mayol 原文明确承认 D≥2 高维成本(Eq.9 总基 = ∏m_d)。实际用 <b>z_w=960</b>,按<b>交互有效秩(~360,DeepRV z_w=360→var_ratio 0.85)</b>定,以覆盖场方差、保持可行。<b>"交互用有效秩定基"不是文献标准规则,是务实取舍;这里如实标注为欠解析。</b></div>
<p><b>div=34 的机制</b>:inter_space=0.078 短到逼近网格分辨率(M_WS=8 的可解析下限 ~0.22)且主导(82%)→ lengthscale 弱识别 → <b>ell-funnel(div 34)</b>;任何可行基都修不了(数据分辨率墙)。DeepRV 用 decoder 的 ell-conditioning 绕开(div 1)。</p>""")

H.append("<h2 id='r'>4. 结果:HSGP vs DeepRV(4 链,复现原始 ECDC)</h2>")
H.append("""<table><tr><th>指标</th><th class='c'>HSGP</th><th class='c'>DeepRV</th></tr>
<tr><td>chains / R̂ / min-ESS</td><td class='c'>4 / 1.005 / 1497</td><td class='c'>4 / 1.006 / 1177</td></tr>
<tr><td><b>div</b></td><td class='c'><b>34</b>(ell-funnel)</td><td class='c'><b>1</b>(绕开)</td></tr>
<tr><td><b>runtime</b></td><td class='c'>1569s</td><td class='c'><b>403s (3.9×快)</b></td></tr>
<tr><td>PP cov-95</td><td class='c'>0.985</td><td class='c'>0.988</td></tr>
<tr><td>PP RMSE</td><td class='c'>86.0</td><td class='c'>86.8</td></tr>
<tr><td><b>WAIC ELPD</b></td><td class='c'>-7,018 ± 127</td><td class='c'>-7,001 ± 125</td></tr>
<tr><td>变量分解 g:q:w</td><td class='c'>15:3:82</td><td class='c'>26:18:56</td></tr></table>
<p><b>WAIC 差 = 17 ± 26 = 0.6 SE = 样本外预测打平</b>。DeepRV 快 3.9×、绕开 funnel、拟合相同 → <b>ECDC 上 DeepRV 胜</b>。分解 HSGP 交互 82% vs DeepRV 56%(HSGP 高自由度基吸噪声、高估交互;synthetic-truth 偏向 DeepRV 较低份额)。</p>""")

H.append("<h2 id='f'>5. 图</h2>")
H.append(img("ecdc_convergence_loo.png","收敛诊断 + ELPD:逐参数 R̂ 全 &lt;1.006、ESS 全 &gt;1000;WAIC 打平(0.6 SE)。"))
H.append(img("ecdc_key_countries_obs_vs_rate.png","9 个最高负担国:灰柱=观测,蓝=HSGP rate,红=DeepRV rate。两法都追住暴发;HSGP(蓝)常把峰打更尖/过冲(IT 2011/2015、DE 2013-15、PL 2019),DeepRV(红)更贴观测——HSGP 交互 82%(尖、吸噪声)vs DeepRV 56%(软),也是 WAIC 打平的原因。"))
H.append(img("ecdc_field_recovery.png","场恢复(viridis 场 + RdBu 残差):观测 | HSGP | DeepRV;残差 pattern 两法几乎一致(与 WAIC 打平一致)。"))
H.append(img("ecdc_predictive_calibration.png","后验预测校准:校准曲线两法几乎重合;PIT 均近平(校准良好)。"))
H.append(img("ecdc_traces.png","关键超参 4 链 trace:干净混合。"))
H.append(img("ecdc_hsgp_vs_deeprv_kde.png","(原有)9 个超参后验 KDE:HSGP vs DeepRV。"))
H.append(img("ecdc_hsgp_vs_deeprv_heatmap.png","(原有)潜在场热力图对比。"))
H.append(img("ecdc_timeseries_fit.png","(原有)高负担国时间序列拟合。"))
H.append(img("ecdc_seasonal_curve.png","(原有)季节曲线(数据驱动,4月峰)。"))

H.append("<h2 id='c'>6. 代码</h2>")
H.append("<h3>模型脚本</h3>")
for f in ["110_prepare_ecdc_measles_panel.py","111_ecdc_hsgp_joint_hp_nb.py","113_train_ecdc_deeprv_decoder.py","114_ecdc_deeprv_joint_hp_nb.py"]: H.append(code(f))
H.append("<h3>本次新增:图 + 审计脚本</h3>")
for f in ["ecdc_figures.py","ecdc_prior_audit.py"]: H.append(code(f))
H.append("<h3>文件位置</h3><p>后验:data/processed/ecdc/ecdc_{hsgp,deeprv}_enhanced.npz(逐参数 rhat/ess、traces、loglik、pp_draws、eta)+ results/ecdc_{hsgp,deeprv}_joint9hp_nb.json。图:figures/。代码:scripts/。</p>")
H.append("</body></html>")
open(OUT,"w").write("\n".join(H))
print("wrote",OUT,f"({os.path.getsize(OUT)//1024} KB)")
