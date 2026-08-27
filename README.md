# Neural surrogate priors for Bayesian hierarchical models: what does DeepRV preserve, and what does it distort?

Code accompanying the MSc thesis (Haoyu Liao, MSc Statistics, Imperial College London,
supervised by Dr Oliver Ratmann).

The thesis compares two approximations that replace the **representation** of a latent Gaussian
process while leaving the sampler and the inferential target unchanged:

- **HSGP** — the Hilbert-space approximate GP, a fixed basis of Laplacian eigenfunctions.
- **DeepRV** — a neural decoder trained offline on exact GP draws.

Everything but the latent representation is matched: kernel, seasonal component, containment priors,
likelihood, data window, and NUTS settings.

📄 [Read the thesis](thesis/Liao_MSc_thesis_DeepRV_vs_HSGP.pdf) ·
🌐 [Project site](https://USERNAME.github.io/deeprv-vs-hsgp/)

---

## Headline result

The comparison separates two properties that are usually reported together.

| | preserved? | evidence |
|---|---|---|
| Latent **fields** and their predictive consequences | ✅ | posterior mean trends correlate at 0.9998 / 0.9999 on the two birth series; held-out predictive density on ECDC is tied at 0.6 SE |
| **Hyperparameter** posteriors | ❌ | Dutch trend lengthscale displaced by 3.0 pooled posterior SD, while every parameter bypassing the decoder agrees to 4 dp |
| **Uncertainty** width | ❌ | DeepRV's 95% bands cover the synthetic truth 85.7% of the time against HSGP's 94.6% |

Which method wins is decided by the complexity of the space–time interaction. Where its lengthscale
approaches the resolution floor of the basis (ECDC), HSGP funnels and DeepRV is ~4× faster at equal
predictive density. Where the interaction is well identified (monthly Tycho), the complete linear
basis fits better in sample by 33 SE of WAIC.

---

## Repository layout

```
experiments/
├── 01_synthetic_benchmark/    joint 9-hyperparameter benchmark, known ground truth
├── 02_synthetic_exact_gp/     exact full-rank GP reference + HSGP basis sweep
├── 03_births_netherlands/     daily births 1995–2024, full-prior HSGP vs DeepRV
├── 04_ecdc_measles/           ECDC 2010–2019, 29 countries × 120 months (+ held-out CV)
├── 05_tycho_measles/          Project Tycho 1928–1962, 49 states × 420 months
├── 06_interaction_ablation/   refits with the space–time interaction removed
└── 07_audits/                 prior-predictive audit, decoder fidelity, hyperparameter KDEs
docs/                          GitHub Pages site
thesis/                        the submitted thesis PDF
```

## Where each thesis result comes from

| Thesis section | Result | Script |
|---|---|---|
| §5.1 US births | positive control, trend corr 0.9998 | `experiments/03_births_netherlands/scripts/` (US variant) |
| §5.2 NL births | hyperparameter displacement | `03_births_netherlands/scripts/118e_netherlands_births_deeprv_fullprior.py` |
| §5.2 Fig. 7 | hyperparameter KDEs | `07_audits/scripts/nl_births_hp_kde.py` |
| §5.3 synthetic | RMSE / coverage over 3 seeds | `01_synthetic_benchmark/scripts/106_hsgp_seasonal_hp_benchmark.py`, `107_fit_deeprv_joint_hp_with_seasonal.py` |
| §5.3 exact GP | untruncated reference, basis sweep | `02_synthetic_exact_gp/scripts/run_exactgp.py`, `run_hsgp.py` |
| §5.4 ECDC | HSGP fit (R̂ 1.005, 34 div) | `04_ecdc_measles/scripts/111_ecdc_hsgp_joint_hp_nb.py` |
| §5.4 ECDC | DeepRV fit (R̂ 1.006, 1 div) | `04_ecdc_measles/scripts/114_ecdc_deeprv_joint_hp_nb.py` |
| §5.4 held-out CV | the tie, 8.1 ± 13.5 (0.6 SE) | `04_ecdc_measles/holdout_cv/holdout_{hsgp,deeprv}.py` |
| §4.4 prior matching | induced variance / correlation | `07_audits/scripts/prior_predictive_audit.py` |
| §4.5 decoder audit | fidelity vs lengthscale, training-range check | `07_audits/scripts/decoder_fidelity_audit.py` |
| App. B refit sequence | bounded-ℓ_g sensitivity | `04_ecdc_measles/scripts/113{b,c,d}_retrain_*.py`, `111e_hsgp_v5.py`, `114e_deeprv_v5.py` |
| §5.6 Tycho | monthly HSGP / DeepRV | `05_tycho_measles/scripts/37_*.py`, `47_*.py` |
| §5.7 ablation | interaction removed | `06_interaction_ablation/` |

## Reproducing

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Each experiment folder runs independently. Typical order within one:

1. `*prepare*.py` / `*generate*.py` — build the panel
2. `*train*decoder*.py` — train the DeepRV decoders (offline, one-off, < 1 min on CPU)
3. `*hsgp*.py` / `*deeprv*.py` — fit under NUTS
4. `*figures*.py` — regenerate the figures

**Not tracked in git:** posterior draw archives and trained decoder weights (`*.npz`, 100 MB+ each).
Regenerate them with the training and fitting scripts above.

## Data

Small processed panels are included. Raw sources are public but must be downloaded separately —
see [DATA.md](DATA.md).

## Citation

```bibtex
@mastersthesis{liao2026deeprv,
  author = {Haoyu Liao},
  title  = {Neural surrogate priors for Bayesian hierarchical models:
            what does DeepRV preserve, and what does it distort?},
  school = {Imperial College London},
  year   = {2026}
}
```

DeepRV is due to Navott et al. (2025); the original methodology and codebase come from the
Machine Learning & Global Health group. HSGP follows Solin & Särkkä (2020) and
Riutort-Mayol et al. (2023).

## Licence

MIT — see [LICENSE](LICENSE).
