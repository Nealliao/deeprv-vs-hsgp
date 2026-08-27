"""
verify_priors.py
================================================================================
Recomputes every number in PRIOR_SPECIFICATION.md:
  - InverseGamma containment probabilities P[l < rho < u]
  - truth percentile of each HP under its prior
  - HalfNormal amplitude percentiles
  - additive-separation PDF overlap between component lengthscale priors
  - the periodic-kernel adjacent-month correlation at the ell_h bounds

Only needs numpy + scipy.  Run:  python verify_priors.py
"""

import numpy as np
from scipy.stats import invgamma, halfnorm
from scipy.optimize import fsolve

# Prior parameters used in the benchmark (scripts 100/106/107)
INVGAMMA = {  # name: (a, b, lower, upper, truth)
    "space_length":             (6.1091, 3.3175, 0.25, 1.80, 0.55),
    "time_length":              (9.3607, 179.029, 10.0, 48.0, 18.0),
    "interaction_space_length": (5.3661, 0.3648, 0.03, 0.25, 0.15),
    "interaction_time_length":  (7.3012, 30.009, 2.0, 12.0, 6.0),
    "ell_h":                    (3.9335, 1.9878, 0.20, 2.5, 0.75),
}
HALFNORMAL = {  # name: (scale, truth)
    "space_alpha":       (0.5, 0.30),
    "time_alpha":        (0.5, 0.20),
    "interaction_alpha": (0.5, 0.18),
    "sigma_h":           (0.45, 0.12),
}


def solve_invgamma(l, u, prob=0.98):
    """Solve (a,b) s.t. P[rho<l]=P[rho>u]=(1-prob)/2 — Betancourt's tuning.

    Uses Betancourt's moment-based initial guess (prior_tune.stan) so fsolve
    converges across all scales, not just small ones.
    """
    tail = (1 - prob) / 2
    ratio = (u + l) / (u - l)               # delta=1
    a0 = ratio ** 2 + 2
    b0 = (u + l) / 2 * (ratio ** 2 + 1)
    def eqs(p):
        a, b = p
        if a <= 0 or b <= 0:
            return [1e9, 1e9]
        return [invgamma.cdf(l, a, scale=b) - tail,
                1 - invgamma.cdf(u, a, scale=b) - tail]
    return fsolve(eqs, [a0, b0])


def pdf_overlap(a1, b1, a2, b2, xmax, n=200000):
    x = np.linspace(1e-4, xmax, n)
    return np.trapezoid(np.minimum(invgamma.pdf(x, a1, scale=b1),
                                   invgamma.pdf(x, a2, scale=b2)), x)


print("=" * 70)
print("1. InverseGamma lengthscale containment (target P[l<rho<u] = 0.98)")
print("=" * 70)
for name, (a, b, l, u, truth) in INVGAMMA.items():
    contain = invgamma.cdf(u, a, scale=b) - invgamma.cdf(l, a, scale=b)
    pct = invgamma.cdf(truth, a, scale=b)
    mode = b / (a + 1)
    a_chk, b_chk = solve_invgamma(l, u)
    print(f"  {name:26s} InvGamma({a},{b})")
    print(f"    P[{l} < rho < {u}] = {contain:.4f}   mode={mode:.4f}   "
          f"truth={truth} at {pct*100:.0f} pct")
    print(f"    re-solved (a,b) = ({a_chk:.4f}, {b_chk:.4f})  [should match]")

print()
print("=" * 70)
print("2. HalfNormal amplitude priors")
print("=" * 70)
for name, (scale, truth) in HALFNORMAL.items():
    pct = halfnorm.cdf(truth, scale=scale)
    print(f"  {name:20s} HalfNormal({scale})  median={halfnorm.ppf(0.5, scale=scale):.3f}  "
          f"truth={truth} at {pct*100:.0f} pct")

print()
print("=" * 70)
print("3. Additive separation — PDF overlap between component priors (Sec 4.1)")
print("=" * 70)
ov_s = pdf_overlap(*INVGAMMA["space_length"][:2], *INVGAMMA["interaction_space_length"][:2], xmax=5)
ov_t = pdf_overlap(*INVGAMMA["time_length"][:2], *INVGAMMA["interaction_time_length"][:2], xmax=100)
print(f"  space  <-> inter_space : overlap = {ov_s*100:.2f}%")
print(f"  time   <-> inter_time  : overlap = {ov_t*100:.2f}%")

print()
print("=" * 70)
print("4. Periodic kernel: ell_h -> adjacent-month correlation")
print("   k = exp(-2 sin^2(pi*1/12) / ell_h^2)")
print("=" * 70)
for ell, tag in [(0.20, "prior lower (near white noise)"),
                 (0.75, "truth"),
                 (2.50, "prior upper (near flat -> confounds q_t)")]:
    c = np.exp(-2 * np.sin(np.pi * 1 / 12) ** 2 / ell ** 2)
    print(f"  ell_h={ell:.2f}: corr(adjacent month)={c:.3f}   [{tag}]")

print()
print("=" * 70)
print("5. Frequency separation: time main q (SE, ell_q=18) vs seasonal h (periodic)")
print("   q and h are both functions of time; they separate by FREQUENCY band,")
print("   not by a lengthscale gap.")
print("=" * 70)
rng = np.random.default_rng(0)
T = 72
t = np.arange(T)
# seasonal h: periodic kernel, sigma_h=0.12, ell_h=0.75, sinusoid mean A=0.35
m = np.arange(12)
b = 0.35 * np.cos(2 * np.pi * (m - 2) / 12); b -= b.mean()
dm = np.abs(m[:, None] - m[None, :])
Kh = 0.12 ** 2 * np.exp(-2 * np.sin(np.pi * dm / 12) ** 2 / 0.75 ** 2)
Lh = np.linalg.cholesky(Kh + 1e-9 * np.eye(12))
# time main q: SE kernel, sigma_q=0.20, ell_q=18
dq = np.abs(t[:, None] - t[None, :])
Kq = 0.20 ** 2 * np.exp(-dq ** 2 / (2 * 18.0 ** 2))
Lq = np.linalg.cholesky(Kq + 1e-9 * np.eye(T))
w_eig, V = np.linalg.eigh(Kq)

corrs, var_expl = [], []
for _ in range(2000):
    h = (b + Lh @ rng.standard_normal(12)); h -= h.mean(); h = h[t % 12]
    q = Lq @ rng.standard_normal(T); q -= q.mean()
    corrs.append(np.corrcoef(h, q)[0, 1])
    beta = np.dot(h, q) / np.dot(q, q)
    var_expl.append(1 - (h - beta * q).var() / h.var())
corrs = np.abs(np.array(corrs))
print(f"  |corr(h, q)|: mean={corrs.mean():.3f}  95th pct={np.quantile(corrs, 0.95):.3f}")
print(f"  variance of h explained by a single q draw: {np.mean(var_expl):.3f}")
# fraction of seasonal energy in the top-3 smoothest time-GP modes
B3 = V[:, -3:]
fr = []
for _ in range(500):
    h = (b + Lh @ rng.standard_normal(12)); h -= h.mean(); h = h[t % 12]
    fr.append((B3 @ (B3.T @ h)).var() / h.var())
print(f"  seasonal energy capturable by top-3 time-GP modes: {np.mean(fr):.3f}")
print(f"  effective DOF of the time-GP (ell_q=18): {(w_eig.sum() ** 2 / (w_eig ** 2).sum()):.1f}")
print("  -> near-orthogonal: q sits in ~2-3 low-freq modes, seasonal is higher-freq")

print()
print("Covariate geometry (sets the bounds above):")
print("  space: min nearest-neighbour ~0.13, span ~1.03")
print("  time:  grid spacing 1, span 71")
