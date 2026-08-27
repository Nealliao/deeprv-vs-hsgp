# ECDC Measles — HSGP vs DeepRV (joint 9-HP, real data)

Joint 9-hyperparameter Bayesian inference on **real ECDC measles surveillance**
(Europe, 2010-2019), comparing **HSGP** (Hilbert-space GP basis) vs **DeepRV**
(conditional neural decoder) as the latent prior. Negative-Binomial likelihood,
data-driven seasonal, Eurostat population exposure.

This is the real-data counterpart of the synthetic benchmark (folder
`DeepRV_vs_HSGP_joint9HP`). On synthetic data we have ground truth; here we do
not — so this folder is about what real data **can and cannot** tell us.

## Model

```
Y_{c,t} ~ NegBin2( E_c * exp(beta0 + h_t + f_{c,t}), kappa )
f_{c,t} = g_c (space) + q_t (time) + w_{c,t} (interaction)      [HSGP basis OR DeepRV decoder]
h_t     = cyclic-GP seasonal, data-driven (no fixed mean)
```
9 HPs sampled jointly: space/time/interaction × (amplitude, lengthscale) + seasonal (sigma_h, ell_h),
plus beta0 and the NB concentration kappa.

## Folder

```
ECDC_measles_HSGP_vs_DeepRV/
├── README.md
├── DATA_SOURCES.md          where the data comes from (ECDC + Eurostat, citable)
├── PRIOR_SPECIFICATION.md   how every prior + the DeepRV latent dim z were chosen
├── FINDINGS.md              the analysis & conclusions (incl. the corrected ones)
├── data/
│   ├── ecdc_measles_panel.csv             29 countries × 120 months, cleaned
│   └── eurostat_population_2010_2019.csv  yearly population (exposure)
├── decoders/                ecdc_deeprv_{space,time,inter}.npz  (z=15/12/360, trained)
├── results/
│   ├── ecdc_hsgp_joint9hp_nb.json         HSGP result (+ variance decomposition)
│   └── ecdc_deeprv_joint9hp_nb.json       DeepRV z=360 result (+ var decomposition)
├── figures/                 8 PNGs (see below)
└── scripts/
    ├── 110_prepare_ecdc_measles_panel.py  build panel from ECDC csv + Eurostat
    ├── 111_ecdc_hsgp_joint_hp_nb.py       HSGP inference
    ├── 112_ecdc_make_figures.py           single-method (HSGP) figures
    ├── 113_train_ecdc_deeprv_decoder.py   train DeepRV decoders (PCA-calibrated z)
    ├── 114_ecdc_deeprv_joint_hp_nb.py     DeepRV inference
    └── make_all_figures.py                regenerates all figures from results/ (self-contained)
```

## Headline results (2010-2019, 29 countries)

| metric | HSGP | DeepRV (z=360) |
|---|---|---|
| latent dim | ~1080 basis | 387 |
| runtime | 1601 s | **410 s** (4× faster) |
| divergences | 34 | **1** |
| ESS/sec | 0.94 | **2.87** |
| PP coverage 95% | 0.985 | 0.988 |
| PP RMSE (counts) | 86.0 | 83.8 |
| **variance decomposition g:q:w** | **15 : 3 : 82** | **26 : 18 : 56** |
| seasonal peak | April | April |

**Both methods are interaction-dominated** (w largest) and agree on the seasonal
(April peak) — consistent with measles being outbreak-driven in the vaccine era.
DeepRV is 4× faster with equal predictive fit, but its decomposition is *softer*
(interaction 56% vs 82%; some variance leaks into the smooth main effects).
See FINDINGS.md for the full story, including why the HP posteriors differ a lot
while predictions barely do.

## Reproduce

Figures (self-contained, needs numpy/scipy/matplotlib only — reads results/, writes figures/):
```bash
cd scripts
python make_all_figures.py
```
(`112_ecdc_make_figures.py` is the original single-method script kept as a
record; its paths point to the original project tree.)

Re-run inference (needs the full modelling stack: jax, numpyro, flax; the panel
in data/ and decoders in decoders/ are included). Paths in the scripts point to
the original project tree (`data/processed/...`), so to run them standalone you
must adjust the DATA_DIR/MODEL_DIR constants to this folder. They are included
primarily as the exact record of what was run.

```bash
python 111_ecdc_hsgp_joint_hp_nb.py      # HSGP, ~30 min
python 114_ecdc_deeprv_joint_hp_nb.py    # DeepRV z=360, ~7 min
```

## Figures

| file | what |
|---|---|
| ecdc_seasonal_curve.png | data-driven seasonal effect (April peak, 6.6× amplitude) |
| ecdc_heatmap.png | HSGP: observed / predicted / latent field (29×120) |
| ecdc_9hp_forest.png | HSGP single-method 9-HP posteriors |
| ecdc_timeseries_fit.png | HSGP: observed vs rate, 9 top-burden countries |
| ecdc_hsgp_vs_deeprv_9hp.png | 9-HP forest, HSGP vs DeepRV |
| ecdc_hsgp_vs_deeprv_kde.png | 9-HP posterior densities, HSGP vs DeepRV |
| ecdc_data_recovery_compare.png | observed vs HSGP vs DeepRV rate, per country |
| ecdc_hsgp_vs_deeprv_heatmap.png | recovery (top) + latent decomposition (bottom) |
