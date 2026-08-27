# DeepRV vs HSGP — Joint 9-Hyperparameter Benchmark

Spatio-temporal measles model `f_{s,t} = g_s + q_t + w_{s,t}` (S=20 states, T=72 months)
with a fixed-shape seasonal cyclic GP. All **9 hyperparameters sampled jointly** by NUTS:

| group | HPs |
|---|---|
| amplitude | `space_alpha (σ_g)`, `time_alpha (σ_q)`, `interaction_alpha (σ_w)` |
| lengthscale | `space_length`, `time_length`, `interaction_space_length`, `interaction_time_length` |
| seasonal | `sigma_h (σ_h)`, `ell_h (ℓ_h)` |

Compares **HSGP** (Hilbert-space GP basis) vs **DeepRV** (conditional neural decoder) as the
latent prior. 3 DGP seeds. Betancourt (2020) containment priors throughout.

## Folder structure

```
DeepRV_vs_HSGP_joint9HP/
├── README.md
├── DGP_SPECIFICATION.md              # how the synthetic data is generated (the "truth")
├── PRIOR_SPECIFICATION.md            # Betancourt prior rules + how each prior was chosen
├── VARIANCE_DECOMPOSITION.md         # who recovers g:q:w shares correctly (DeepRV closer!)
├── figures/                          # 6 paper figures (200 dpi)
│   ├── joint9hp_forest_3seed.png        9-HP recovery: post mean ± 95% CI vs truth
│   ├── joint9hp_kde_3seed.png           9-HP posterior densities (3 seeds pooled)
│   ├── joint9hp_latent_heatmap_seed1.png  latent field: truth/mean/residual
│   ├── joint9hp_seasonal_seed1.png      12-month seasonal curve recovery
│   ├── joint9hp_posterior_sd_seed1.png  posterior SD (uncertainty) maps
│   └── synth_variance_decomposition.png g:q:w shares truth vs HSGP vs DeepRV
├── results/                          # raw posterior summaries
│   ├── hsgp_joint9hp_seed{1,2,3}.json    HSGP per-seed (metrics + posterior arrays + HP samples)
│   ├── deeprv_joint9hp_seed{1,2,3}.json  DeepRV per-seed (same schema)
│   ├── summary_3seed.csv                 all scalar metrics + 9-HP stats, 6 rows
│   └── depth_ablation.csv                decoder-depth ablation (interaction decoder)
└── scripts/
    ├── make_figures.py               # regenerates all 5 figures from results/  ← run this
    ├── verify_priors.py              # recomputes every number in PRIOR_SPECIFICATION.md
    ├── 106_hsgp_seasonal_hp_benchmark.py        HSGP joint-9HP benchmark
    ├── 107_fit_deeprv_joint_hp_with_seasonal.py DeepRV joint-9HP benchmark
    ├── 107b_fit_deeprv_joint_hp_interdepth.py   decoder-depth inference (exploration)
    └── 108_decoder_depth_ablation.py            decoder-depth fidelity ablation
```

See **`DGP_SPECIFICATION.md`** for how the data is generated and **`PRIOR_SPECIFICATION.md`**
for how every prior was chosen (with `scripts/verify_priors.py` to recompute the numbers).

## Regenerate the figures

Self-contained — only needs `numpy`, `scipy`, `matplotlib` (no GPU, no MCMC):

```bash
cd scripts
python make_figures.py
```

Reads `../results/*.json`, writes `../figures/*.png`. Edit the plotting functions in
`make_figures.py` to restyle.

## Rerun the experiments (needs the full project)

`106/107/107b/108` depend on the synthetic data, region geometry, and trained decoder
weights in the main project (`data/processed/...`), so they will **not** run from this
folder alone — they are included for reference. In the project root:

```bash
SHP_N_SEEDS=3 SHP_STAGES=3 python scripts/106_hsgp_seasonal_hp_benchmark.py        # HSGP
CDRV9_N_SEEDS=3            python scripts/107_fit_deeprv_joint_hp_with_seasonal.py  # DeepRV
```

## Key results (3 seeds, mean ± sd)

| metric | HSGP | DeepRV |
|---|---|---|
| RMSE_f | **0.059 ± 0.003** | 0.075 ± 0.004 |
| coverage (95%) | **0.946 ± 0.009** | 0.857 ± 0.012 |
| divergences | 1.3 ± 0.9 | 2.3 ± 1.9 |
| runtime (s) | 309 ± 32 | **215 ± 50** |
| ESS/sec | 3.28 | **4.03** |

**Trade-off (not a clean win for either):**
- **DeepRV** — amplitudes (σ_g, σ_q, σ_w) nearly unbiased (decoder anchors scale); ~30% faster.
- **HSGP** — better latent-field recovery & coverage; interaction lengthscales more accurate and stable.
- **Seasonal σ_h, ℓ_h posteriors are identical across methods** — the seasonal cyclic GP bypasses
  the surrogate, so the seasonal funnel is surrogate-invariant. σ_h is funnel-biased high
  (0.154 vs truth 0.120) under both → justifies fixing σ_h in the main benchmark.

All 9 HP sample jointly with 0–5 divergences and R-hat < 1.01: the historical
"joint-θ funnel collapse" was a weak-prior artefact, removed by Betancourt containment priors.
