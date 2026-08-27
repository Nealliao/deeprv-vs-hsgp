# Findings & Analysis (ECDC real data)

The substantive conclusions, including the ones that were corrected along the
way (kept, because the correction itself is a methodological lesson).

---

## 1. Convergence & speed: DeepRV wins clearly

| | HSGP | DeepRV (z=360) |
|---|---|---|
| latent dim | ~1080 basis | 387 |
| runtime | 1601 s | 410 s (**4× faster**) |
| divergences | 34 | 1 |
| min ESS | 1497 | 1177 |
| ESS/sec | 0.94 | 2.87 |

DeepRV's smaller, compressed latent space samples much faster and cleaner.

## 2. Predictive fit: essentially identical

PP coverage 0.985 vs 0.988; PP RMSE 86.0 vs 83.8 (DeepRV marginally better);
per-country recovery curves overlap; the observed/predicted heatmaps are nearly
indistinguishable. **Predictive performance cannot separate the two methods.**

## 3. Variance decomposition (the comparable quantity)

Comparing raw amplitude HPs is invalid — HSGP's alpha multiplies a basis
expansion, DeepRV's sigma_w multiplies a normalized decoder output, so the
numbers aren't on the same scale. The **comparable** quantity is the variance
each component contributes to the latent field, var(g):var(q):var(w):

| component | HSGP | DeepRV |
|---|---|---|
| space g | 15% | 26% |
| time q | 3% | 18% |
| **interaction w** | **82%** | **56%** |

**Both methods are interaction-dominated** (w is the largest component in both).
This matches measles being outbreak-driven in the vaccine era (variation is
"specific country × specific month outbreaks", i.e. interaction; baseline
between-country and shared-time effects are small). DeepRV's decomposition is
**softer**: ~26 percentage points leak from interaction into the smooth
main effects, because a finite nonlinear decoder cannot concentrate
high-frequency variance as tightly as HSGP's complete linear basis.

### Correction recorded
An earlier reading (from raw alpha: "DeepRV space-dominated 7.67, interaction
collapsed 1.73 → decomposition flip") was an **overclaim**. The big space_alpha
sits in front of a near-constant pattern (space_length 3.72 ≫ span 1.33) that
cancels it — space is only 26% of the variance, not dominant. Lesson:
**compare components by variance, not by raw amplitude hyperparameters.**

## 4. Adding latent dimension does NOT fix the softening

z-sweep on the interaction decoder:

| z_w | var_ratio | space_alpha | interaction_alpha | div |
|---|---|---|---|---|
| 360 | 0.847 | 7.67 | 1.73 | 1 |
| 480 | 0.878 | 8.03 | 1.82 | 15 |

More z raised decoder fidelity (var_ratio 0.85→0.88) but **did not change the
decomposition** and **added divergences**. So the difference vs HSGP is a
**structural property** of the nonlinear decoder's inductive bias, not a
capacity/fidelity problem you can tune away. (Decision: keep z=360.)

## 5. Why HP posteriors differ a lot but predictions barely do

This is weak identifiability of the decomposition — a textbook phenomenon.

- The likelihood only "sees" the **total field f = g+q+w** (through the counts),
  not how f is split into components.
- f is pinned down well by the data (→ good, near-identical predictions).
- The split into (g,q,w) is **weakly identified**: many decompositions give
  almost the same f. Along those "reallocate g/q/w but keep f" directions the
  likelihood is nearly flat, so the posterior there is set by the
  **prior/representation**, not the data.
- HSGP (basis) and DeepRV (decoder) have different inductive biases → they land
  at different points along the flat directions → different HPs, same f, same
  predictions.

**Analogy: collinear regression.** With correlated x1,x2 the coefficients
(β1,β2) are not separately identified (large trade-off, very different
posteriors) but the fitted values ŷ are stable. Here HPs = coefficients,
f = fitted values, predictions = ŷ. Different HPs + equal predictions is
exactly collinearity, not a contradiction.

Components are interchangeable because, e.g., "many countries rise the same
year" can be attributed to time-q OR synchronous interaction-w; "a country is
persistently high" to space-g OR always-high interaction-w.

## 6. Real data cannot adjudicate the decomposition

There is **no ground truth** here, and predictive performance can't tell the
methods apart (§2). So we **cannot prove** either decomposition "correct". What
we can say:
- both agree qualitatively (interaction-dominated, April seasonal peak) and
  both match measles epidemiology;
- they differ only in degree, along a weakly-identified direction set by the
  representation;
- to decide which representation recovers decompositions more faithfully, you
  need the **synthetic benchmark** (where truth is known) and/or external
  epidemiological knowledge. This is exactly why synthetic + real are
  complementary, not redundant.

## 7. The seasonal: a clean internal control

sigma_h, ell_h and the seasonal curve (April peak, ~6.6× amplitude) are
**identical across HSGP and DeepRV** — because the seasonal is an independent
cyclic GP that bypasses both the basis and the decoder. Same model, the part
through the surrogate differs (in degree), the part bypassing it is identical.
This localises any representation effect to the surrogate, as expected.

---

## Thesis takeaways

1. **Compare latent components by variance, not raw amplitude HPs** (scale
   non-comparability creates false "flips").
2. **Equal predictive performance ≠ identical inference** — but here the methods
   actually agree qualitatively; the HP differences are weak-identifiability +
   representation, not "one is wrong".
3. **Real data cannot validate the decomposition** (no truth, predictions tie)
   → the synthetic benchmark is essential to transfer credibility.
4. **DeepRV: 4× faster, equal prediction, qualitatively equal decomposition,
   quantitatively softer** on high-frequency interaction; its compression edge
   shrinks as the latent field's effective dimension rises.
5. **Interaction-dominance is the correct signal** for vaccine-era measles
   (outbreak-driven), so both methods give the epidemiologically right picture.
