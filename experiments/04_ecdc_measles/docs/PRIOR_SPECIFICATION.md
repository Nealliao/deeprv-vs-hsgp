# Prior & Latent-Dimension Choices (ECDC)

How every prior was chosen, and how the DeepRV latent dimension was set.
Principles: Betancourt (2020) containment for GP lengthscales; Stan / Simpson
et al. (2017) for the NB overdispersion; PCA effective-dimension for the
DeepRV latent size.

---

## 1. GP lengthscale priors — Betancourt InverseGamma containment

Choose (a, b) so that `P[l < rho < u] = 0.98`, with bounds from the covariate
geometry (lower ≈ min covariate distance, upper ≈ max span). Spatial geometry
is comparable to the synthetic benchmark (span 1.33, min-NN 0.083), so the
spatial priors are reused; the **time priors are recomputed for 120 months**.

| HP | prior | containment | note |
|---|---|---|---|
| space_length | InvGamma(6.1091, 3.3175) | P[0.25 < l < 1.80] | reused (geometry comparable) |
| time_length | InvGamma(6.5707, 167.369) | P[12 < l < 80] | recomputed for 120 months |
| interaction_space_length | InvGamma(5.3661, 0.3648) | P[0.03 < l < 0.25] | reused (short scale) |
| interaction_time_length | InvGamma(6.2718, 27.019) | P[2 < l < 14] | recomputed for 120 months |

---

## 2. Amplitude priors — HalfNormal, widened for real data

```
space_alpha, time_alpha, interaction_alpha  ~  HalfNormal(1.0)
```
Widened from the synthetic HalfNormal(0.5): real measles log-risk spans
countries far more (per-capita incidence ranges over orders of magnitude), so
the amplitudes need more room.

---

## 3. Seasonal priors — periodic kernel, data-driven

```
sigma_h ~ HalfNormal(0.45)              amplitude (Tycho-calibrated value)
ell_h   ~ InverseGamma(3.9335, 1.9878) P[0.2 < ell_h < 2.5]=0.98 (periodic)
```
The seasonal mean is **NOT a fixed sinusoid** (unlike the synthetic DGP): the
periodic GP learns the shape from the data, because the real peak month is
unknown. (It recovered an April peak — see FINDINGS.) The seasonal block is an
independent cyclic GP and does NOT pass through the HSGP basis or the DeepRV
decoder — which is why the two methods agree exactly on it.

---

## 4. NB overdispersion kappa — reciprocal parameterization (literature)

**Started** with `kappa ~ LogNormal(log 10, 1)` but the posterior (0.84) fell
in its 2.5% left tail → prior centre too high, not defensible.

**Switched** to the literature-standard reciprocal parameterization:
```
1/sqrt(kappa) ~ HalfNormal(1)        kappa = (1/sqrt(kappa))^(-2)
```
- **Stan Prior Choice Recommendations**: set the prior on `1/kappa` or
  `1/sqrt(kappa)`, because kappa→∞ is Poisson (the base model) and corresponds
  to `1/sqrt(kappa)→0`; the reciprocal scale is interpretable, kappa is not.
- **Simpson et al. (2017), Stat. Sci. 32(1)** PC priors: same idea — penalise
  departure from the Poisson base.

**Sensitivity check** (the reason this matters): the two kappa priors give
**identical 9-HP posteriors** (robust). So the kappa prior does not drive the
conclusions; reciprocal is used because it is more defensible and lands the
posterior (0.84) at a healthy interior percentile.

---

## 5. DeepRV latent dimension z — PCA effective dimension

Neural surrogates have **no analytic rule** for the latent size (unlike HSGP's
basis-count rule, Riutort-Mayol et al. 2023). So z is set per component by the
**PCA effective dimension** — the number of principal components needed to
capture a target fraction of the GP prior variance, calibrated to match the
synthetic benchmark's fidelity (var_ratio ≈ 0.85).

| component | output dim | z chosen | captured var / var_ratio |
|---|---|---|---|
| space g | 29 | **15** | ~99.99% |
| time q | 120 | **12** | ~99.9% |
| interaction w | 3480 | **360** | 94% PCA → trained var_ratio 0.847 |

Key point: ECDC interaction is **high-frequency** (inter_space lengthscale
0.078, very short → local outbreaks), so its effective dimension is high —
z_w=120 captures only 64%, z_w=360 reaches the synthetic 0.85 fidelity.
**DeepRV's compression advantage is weaker on high-frequency real data.**

A z-sweep (z_w = 220 / 300 / 360 / 480) is documented in FINDINGS: increasing z
raised var_ratio (0.768 → 0.878) but did NOT change the decomposition and added
divergences — the residual difference vs HSGP is structural, not a capacity
issue.

---

## Verifying the numbers

All InverseGamma containments solve `P[rho<l]=P[rho>u]=0.01` numerically
(scipy). The synthetic-folder script `verify_priors.py` reproduces the shared
priors; the ECDC time priors above were solved the same way (Betancourt
moment-based initial guess for fsolve).
