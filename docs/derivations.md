# Derivations: the symbolic reference layer

This file records every derivation behind `src/kerrray/geometry/symbolic.py`
and `src/kerrray/photons/orbits.py`: the equations, the SymPy verification
that was actually run (all of it lives in `tests/test_symbolic_geometry.py`,
`tests/test_orbits.py` (null orbits, shadow) and
`tests/test_orbits_timelike.py` (Omega, E, L, ISCO) and is re-executed by
`pytest`), the published
forms each result was matched against, and the implementation mapping. The
conventions are those of `docs/architecture.md` section 1: G = c = 1,
signature (−, +, +, +), Boyer–Lindquist coordinates (t, r, θ, φ) with index
order 0…3, Σ = r² + a² cos²θ, Δ = r² − 2Mr + a², a = spin · M with |a| < M,
and *prograde* ⇔ a L_z > 0.

Every number quoted below was measured on the development machine while
other test suites were running concurrently (timings are indicative only;
residuals are not).

## 1. Kerr metric in Boyer–Lindquist coordinates

Covariant components (all others zero):

```
g_tt   = −(1 − 2Mr/Σ)
g_tφ   = −2Mar sin²θ / Σ
g_rr   = Σ/Δ
g_θθ   = Σ
g_φφ   = (r² + a² + 2Ma²r sin²θ/Σ) sin²θ
```

References checked: Bardeen, Press & Teukolsky 1972 (ApJ 178, 347) eq. (2.1)
(the line element's cross term −4Mar sin²θ/Σ dt dφ gives g_tφ as above);
Misner, Thorne & Wheeler 1973 eq. (33.2); Chandrasekhar 1983 ch. 6
eq. (57); Visser 2007 (arXiv:0706.0622) section 2.

Contravariant components, with A = (r² + a²)² − a²Δ sin²θ:

```
g^tt   = −A/(ΣΔ)
g^tφ   = −2Mar/(ΣΔ)
g^rr   = Δ/Σ
g^θθ   = 1/Σ
g^φφ   = (Δ − a² sin²θ)/(ΣΔ sin²θ)
```

Reference: Chandrasekhar 1983 ch. 6 eq. (60); architecture section 1.1.

SymPy verification performed:

| Check | Where | Result |
|---|---|---|
| `simplify(g · g⁻¹) == I` and `simplify(g⁻¹ · g) == I` | `symbolic_inverse_metric` (raises otherwise), `test_metric_times_inverse_is_identity` | identity |
| a → 0 gives diag(−(1−2M/r), 1/(1−2M/r), r², r² sin²θ) (MTW eq. 23.1), and its inverse | `test_schwarzschild_limit` | exact |
| det g = −Σ² sin²θ | `test_metric_determinant` | exact |
| ω = −g_tφ/g_φφ > 0 for a > 0 (drag towards +φ) | `test_frame_dragging_angular_velocity_sign` | holds |
| R_μν = 0 numerically (section 3) | `test_ricci_tensor_vanishes` | worst |R|/scale = 1.6e−16 |

Implementation: `symbolic_metric()` → `(ImmutableMatrix, KerrSymbols)`;
`symbolic_inverse_metric()`; `lambdified_metric()`,
`lambdified_inverse_metric()`; `lambdified_inverse_metric_derivatives()`
returns (∂_r g^μν, ∂_θ g^μν) as two 4×4 arrays. The numerical layer
(`kerrray.geometry.metric`, geometry role) is cross-checked against these at
random points for spins {0, 0.3, 0.7, 0.95, −0.6} in
`tests/test_symbolic_geometry.py` part 2 (metric, inverse, the five-component
named tuples, ∂_r and ∂_θ of the inverse, and broadcasting over array
inputs), all to rtol 1e−11.

## 2. Christoffel symbols

Definition (Carroll 2004 eq. 3.27; MTW eq. 8.24b):

```
Γ^μ_{αβ} = ½ g^{μν} (∂_α g_{νβ} + ∂_β g_{να} − ∂_ν g_{αβ})
```

computed component by component from the symbolic inverse metric and
`sympy.diff` of the covariant metric; 32 of the 64 components are non-zero.

Simplification policy (measured): `sympy.simplify` on all components took
552 s; `sympy.factor` takes about 1–8 s (load dependent) and halves the
operation count, so `factor` is used. Declaring `M` and `r` positive
slowed `factor` to about 250 s, so all symbols are declared merely real.

Verification:

| Check | Where |
|---|---|
| Γ^μ_{αβ} = Γ^μ_{βα} numerically (rtol 1e−13) | `test_christoffel_symmetric_in_lower_indices` |
| metric compatibility ∂_α g_{μν} = Γ^λ_{αμ} g_{λν} + Γ^λ_{αν} g_{μλ} at random points, a ∈ {0, 0.7} (pins the index order [μ, α, β] and the sign) | `test_christoffel_metric_compatibility` |
| a = 0 reproduces the Schwarzschild list of Carroll 2004 eq. (5.53): Γ^t_{tr} = M/(r(r−2M)), Γ^r_{tt} = M(r−2M)/r³, Γ^r_{rr} = −M/(r(r−2M)), Γ^r_{θθ} = −(r−2M), Γ^r_{φφ} = −(r−2M) sin²θ, Γ^θ_{rθ} = Γ^φ_{rφ} = 1/r, Γ^θ_{φφ} = −sinθ cosθ, Γ^φ_{θφ} = cotθ, all others zero | `test_schwarzschild_christoffels_match_closed_form` |
| numerical `kerrray.geometry.christoffel.christoffel` matches at random points, five spins, rtol 1e−11 | `test_numeric_christoffel_matches_symbolic` |

Implementation: `symbolic_christoffel()` → immutable array (4, 4, 4);
`lambdified_christoffel()` → `(M, a, r, θ) ↦ ndarray(4, 4, 4)`.

## 3. Ricci tensor and the vacuum check

Definition (Carroll 2004 eq. 3.113 contracted on the first and third index;
MTW eq. 8.51):

```
R_{μν} = ∂_λ Γ^λ_{μν} − ∂_ν Γ^λ_{μλ} + Γ^λ_{λσ} Γ^σ_{μν} − Γ^λ_{νσ} Γ^σ_{μλ}
```

The expression is built from the factored Christoffels and deliberately left
unsimplified (architecture section 1.1); it is lambdified with common
subexpression elimination and evaluated at random points. Since the
non-zero result would be caused by floating-point cancellation among terms
of size Γ², the tolerance is relative to `christoffel_scale` =
(Σ_{μαβ} |Γ^μ_{αβ}|)². Measured over 20 random points for each of the spins
{0, 0.3, 0.7, 0.95, −0.6} (r ∈ [r₊ + 0.25M, 12M], θ ∈ [0.15, π − 0.15]): the
worst |R_{μν}|/scale is 1.6e−16; the test tolerance is 1e−12, four orders
of magnitude of margin. Because Kerr is vacuum, this check is independent
of the sign convention of the Riemann tensor.

Implementation: `symbolic_ricci()`, `lambdified_ricci()`,
`ricci_tensor_numeric(M, a, r, θ)`, `christoffel_scale(M, a, r, θ)`.

## 4. Spherical photon orbits: ξ(r), η(r)

Carter's radial potential for null geodesics with ξ = L_z/E, η = Q/E²
(Carter 1968; BPT 1972 eq. 2.9 with μ = 0):

```
R(r) = [(r² + a²) − aξ]² − Δ [η + (ξ − a)²]
```

Spherical orbits satisfy R = 0 and dR/dr = 0. `sympy.solve` of that system
for (ξ, η) (`test_spherical_orbit_constants_are_the_physical_branch`,
0.5 s) returns exactly two branches:

```
(i)  ξ = (r² + a²)/a,                          η = −r⁴/a²                    (η < 0: discarded)
(ii) ξ = −[r²(r − 3M) + a²(r + M)] / [a(r − M)],  η = r³[4Ma² − r(r − 3M)²] / [a²(r − M)²]
```

Branch (ii) was compared symbolically (`simplify(difference) == 0`) with the
published forms, and **all three match**
(`test_spherical_orbit_constants_match_published_forms`):

* Bardeen 1973 (Les Houches) / Chandrasekhar 1983 ch. 7:
  ξ = [M(r² − a²) − rΔ]/[a(r − M)], η = r³[4MΔ − r(r − M)²]/[a²(r − M)²];
* Johannsen & Psaltis 2010 (ApJ 718, 446) eqs. (8)–(9):
  ξ = −(r³ − 3Mr² + a²r + a²M)/[a(r − M)],
  η = −r³(r³ − 6Mr² + 9M²r − 4a²M)/[a²(r − M)²];
* Cunha & Herdeiro 2018 (GRG 50, 42) eqs. (10)–(11):
  ξ = [r²(3M − r) − a²(r + M)]/[a(r − M)], η = r³[4Ma² − r(r − 3M)²]/[a²(r − M)²].

The implementation uses the Cunha & Herdeiro grouping because the explicit
factor (r − 3M) avoids cancellation between r³ and 3Mr² when |a| is small.
Two further identities are verified symbolically
(`test_spherical_orbit_identities`) and used later:

```
η + (ξ − a)² = 4r²Δ/(r − M)²                      (from R = 0 with X ≡ (r² + a²) − aξ = 2rΔ/(r − M))
ξ − a       = −r(r² − 3Mr + 2a²)/[a(r − M)]
η = 0  ⇔  r(r − 3M)² = 4Ma²
```

Numerically, `spherical_orbit_constants` satisfies |R| and |dR/dr| < 1e−12 r⁴
on 25 points of [r_min, r_max] for every non-zero spin tested.

**a = 0.** The parametrisation by r degenerates: every Schwarzschild spherical
photon orbit sits at r = 3M and only ξ² + η = 27M² is determined, so
`spherical_orbit_constants` raises `ValueError` for a = 0 and the callers
below use explicit branches.

Implementation: `radial_potential`, `radial_potential_derivative`,
`theta_potential`, `spherical_orbit_constants` (vectorised in r).

## 5. Equatorial circular photon orbits

η = 0 is the cubic r(r − 3M)² = 4Ma² (the form behind BPT 1972 eq. 2.17). At
a = 0 it has a **double** root at r = 3M, which bracketing root finders
resolve only to about √ε ≈ 1e−8 and which `numpy.roots` handles no better.
The implementation therefore solves the equivalent equation obtained by
taking the square root with the branch sign fixed,

```
r − 3M = ∓ 2|a| √(M/r)      (upper sign prograde, r ≤ 3M; lower sign retrograde, r ≥ 3M),
```

whose left-minus-right side is strictly monotone in r on [M, 3.5M]
(prograde; derivative 1 − |a|√M r^{−3/2} > 0 for r > M when |a| < M) and on
[2.5M, 4M] (retrograde), so `scipy.optimize.brentq` (xtol 1e−14 M,
rtol 1e−14) finds the unique simple root, also at a = 0. This is a
deliberate deviation from the "polynomial root finding" wording of
architecture section 1.2, for the reason above.

Reference: BPT 1972 eq. (2.18), r_ph = 2M{1 + cos[(2/3) arccos(∓|a|/M)]},
implemented as `equatorial_photon_orbit_radius_reference` and never used to
produce a result. Measured |computed − reference| ≤ 4.4e−16 M for spins
{0, 0.3, 0.7, 0.95, −0.6, 0.999}, both senses (test tolerance 1e−12). For
a ≠ 0 the test also checks η(r_ph) ≈ 0 (< 1e−10 r_ph⁴).

## 6. Critical impact parameters

b = |ξ| on the equatorial photon orbit. Using the identity of section 4 with
η = 0, (ξ − a)² = 4r²Δ/(r − M)², and the sign of ξ − a from
ξ − a = −r(r² − 3Mr + 2a²)/[a(r − M)] (negative factor r² − 3Mr + 2a² at the
prograde radius, positive at the retrograde radius),

```
ξ_pro   = sign(a) [ |a| + 2 r_pro √Δ(r_pro) / (r_pro − M) ]
ξ_retro = sign(a) [ |a| − 2 r_retro √Δ(r_retro) / (r_retro − M) ]
```

which are regular at a = 0. For a ≠ 0 the tests check |ξ| against
`spherical_orbit_constants` at r_ph (rtol 1e−10) and the sign convention
a ξ_pro > 0. For a = 0 the computed value is compared with the reference
3√3 M: measured difference 0.0 (test tolerance 1e−12). Monotonicity in |a|
(b_pro decreasing, b_retro increasing) is also tested.

Implementation: `critical_impact_parameters(st) → (b_pro, b_retro)`,
`spherical_orbit_radius_range(st) → (r_pro, r_retro)` (Teo 2003: spherical
photon orbits exist exactly for η ≥ 0, i.e. between the equatorial radii).

## 7. Shadow curve (Bardeen 1973)

For an observer at infinity at inclination i (Boyer–Lindquist θ of the
observer), architecture section 1.3:

```
α = −ξ/sin i,   β = ± √(η + a² cos²i − ξ² cot²i)
```

along the spherical photon orbits. The radicand Θ(r) is positive at the
polar orbit r₀ (ξ(r₀) = 0, found by `brentq` on ξ, which changes sign
between r_pro and r_retro) and negative at r_pro and r_retro unless
cos i = 0, so the admissible interval [r_a, r_b] is found by `brentq` on
Θ over [r_pro, r₀] and [r₀, r_retro] (endpoints kept when Θ ≥ 0 there). This
resolves arbitrarily small inclinations (tested at i = 10⁻² and 10⁻³ deg,
where the curve tends to the axis-view circle of radius √(η(r₀) + a²)
linearly in i; measured relative deviation 7e−6 at 10⁻³ deg). The interval is
sampled with cosine clustering at the turning points, the ±β branches are
joined, and the points are sorted by atan2(β, α) about the origin, which
lies inside every Kerr shadow (the central ray ξ = 0, η = −a² cos²i has
R > 0 for all r and is captured). Returned: 2·(n // 2) points. The two turning points r_a, r_b
are roots of Θ by construction; `brentq` leaves a residual of order
xtol · |Θ'| ≈ 1e−14 whose square root (≈ 1e−7) would be pure noise, so β is
set to exactly 0 there. (Before this rule the curve failed its own closure
and mirror-symmetry tests by ≈ 1e−7.)

Explicit branches: a = 0 returns the circle of radius
`critical_impact_parameters(st)[0]` (computed by root finding, section 6;
the test checks it against 3√3 M to 1e−12 and runs the branch with NumPy
floating-point errors and warnings raised as exceptions, so no 0/0 occurs);
i = 90° sets cos i = 0 exactly.

Tests: closure, ordering, β-symmetry, α² + β² = ξ² + η + a² cos²i, every
sampled point lies on the ξ(r), η(r) family, α extent at i = 90° equals
(−b_pro, +b_retro) to 1e−9 for a = 0.9, mirror symmetry under a → −a, and
identity of the curves for i and 180° − i.

Approximation label: WHAT — the curve is the exact r_o → ∞ limit; WHY — it is
the standard analytic reference for the numerically traced shadow;
LIMITATION — a camera at finite r_o sees O(M/r_o) corrections (architecture
section 1.3), and the r-sampling is not uniform in polar angle.

Implementation: `shadow_curve(st, inclination_deg, n) → (α, β)`.

## 8. Keplerian angular velocity

For a circular equatorial orbit u^μ = u^t(1, 0, 0, Ω) the geodesic
condition is ∂_r(g_tt + 2Ω g_tφ + Ω² g_φφ) = 0 at θ = π/2 (radial component
of the geodesic equation with u^r = u^θ = 0). Solving it with SymPy for Ω,
using the symbolic metric of section 1
(`test_keplerian_angular_velocity_from_circular_orbit_condition`), gives two
roots, (Ma ∓ √(Mr³))/(Ma² − r³), which simplify (with M, r positive) to

```
Ω = ± √M / (r^{3/2} ± a√M)      (BPT 1972 eq. 2.16, upper sign prograde)
```

The implementation writes this with |a| and multiplies by sign(a) so that
`prograde` means a L_z > 0 for either sign of a; the test checks the
implemented value against the SymPy root of the same sense at r ∈ {2.5, 4,
9} for all six spins (rel 1e−13). For a = 0 both senses give ± √(M/r³).

Implementation: `keplerian_angular_velocity(st, r, prograde)`. (Architecture
section 5 lists this function in `physics/redshift.py`; the rendering role
can import or re-export it from here.)

## 9. Circular-orbit energy and angular momentum

With Ω from section 8, u^t = [−(g_tt + 2Ω g_tφ + Ω² g_φφ)]^{−1/2},
E = −(g_tt + Ω g_tφ) u^t and L = (g_tφ + Ω g_φφ) u^t. SymPy shows
(`test_circular_orbit_energy_and_momentum_from_metric`) that E² and L² so
derived equal the squares of

```
E = (r^{3/2} − 2M r^{1/2} ± a M^{1/2}) / [r^{3/4} (r^{3/2} − 3M r^{1/2} ± 2a M^{1/2})^{1/2}]
L = ± M^{1/2} (r² ∓ 2a M^{1/2} r^{1/2} + a²) / [r^{3/4} (r^{3/2} − 3M r^{1/2} ± 2a M^{1/2})^{1/2}]
```

(BPT 1972 eqs. 2.12 and 2.13, upper sign prograde), that the numerical values
agree to 1e−13, and that these E(r), L(r) satisfy R = 0 and dR/dr = 0 for
the timelike radial potential R = [E(r² + a²) − aL]² − Δ[r² + (L − aE)²]
(BPT eq. 2.9, μ = 1, Q = 0) identically.

Implementation: `circular_orbit_energy`, `circular_orbit_angular_momentum`,
`timelike_radial_potential`.

## 10. ISCO: closed form and independent numerical location

Closed form (BPT 1972 eq. 2.21), with a → |a| and the upper sign prograde:

```
Z₁ = 1 + (1 − a²/M²)^{1/3} [(1 + a/M)^{1/3} + (1 − a/M)^{1/3}]
Z₂ = (3a²/M² + Z₁²)^{1/2}
r_ms = M {3 + Z₂ ∓ [(3 − Z₁)(3 + Z₁ + 2Z₂)]^{1/2}}
```

Independent numerical route: the marginally stable orbit is where the second
derivative of the radial potential along the circular family vanishes.
SymPy gives (`test_circular_orbit_energy_and_momentum_from_metric`)

```
d²R/dr² = 2E²a² + 12E²r² − 2L² + 12Mr − 2a² − 12r²
```

so `marginal_stability_function` = R''/E² = 12r² + 2a² − 2(L/E)²
+ (12Mr − 12r² − 2a²)/E², which stays finite at the photon orbit (E → ∞,
L/E finite, 1/E² → 0), is positive there (unstable orbits) and negative at
12M (stable; the ISCO never exceeds 9M). `brentq` on [r_ph, 12M] with the
sign change asserted gives `isco_radius_numeric`.

Measured |closed − numeric| ≤ 2.6e−14 M for all six spins and both senses
(test tolerance 1e−10); at a = 0 both routes give 6M (closed form exactly,
Z₁ = Z₂ = 3; numeric within 1.1e−14). The tests also check that E(r) is
minimal at the ISCO (E(r_isco ± 10⁻³M) > E(r_isco)), that the ISCO lies
outside the photon orbit, and the monotonic ordering in |a|.

Implementation: `isco_radius`, `isco_radius_numeric`,
`marginal_stability_function`. (Architecture section 5 lists `isco_radius`
in `physics/accretion.py`; the rendering role can import it from here.)

## 11. Sign conventions for a < 0

Architecture section 1 defines prograde as a L_z > 0. All radii
(photon orbits, ISCO, spherical-orbit range) are even in a, so they are
computed from |a|; ξ, L and Ω are odd in a and carry sign(a). Consequences
tested (`test_negative_spin_mirrors_positive_spin`): r_ph and r_isco are
identical for ±a, Ω and L flip sign, b_pro and b_retro are identical, and
the shadow curve for −a is the mirror image α → −α. For a = 0 the two
senses are degenerate and sign(0) is taken as +1.

## 12. Implementation map

| Quantity | Function | Verified by |
|---|---|---|
| g_μν, g^μν | `symbolic.symbolic_metric`, `symbolic_inverse_metric`, `lambdified_*` | section 1 |
| Γ^μ_{αβ} | `symbolic.symbolic_christoffel`, `lambdified_christoffel` | section 2 |
| ∂_r g^μν, ∂_θ g^μν | `symbolic.lambdified_inverse_metric_derivatives` | section 1, geometry cross-check |
| R_μν | `symbolic.symbolic_ricci`, `ricci_tensor_numeric`, `christoffel_scale` | section 3 |
| R(r), R'(r), Θ(θ) | `orbits.radial_potential`, `radial_potential_derivative`, `theta_potential` | section 4 |
| ξ(r), η(r) | `orbits.spherical_orbit_constants` | section 4 |
| r_ph | `orbits.equatorial_photon_orbit_radius` (+ `_reference`) | section 5 |
| b_pro, b_retro | `orbits.critical_impact_parameters` | section 6 |
| [r_min, r_max] | `orbits.spherical_orbit_radius_range` | section 6 |
| shadow (α, β) | `orbits.shadow_curve` | section 7 |
| Ω | `orbits.keplerian_angular_velocity` | section 8 |
| E, L, R_timelike | `orbits.circular_orbit_energy`, `circular_orbit_angular_momentum`, `timelike_radial_potential` | section 9 |
| r_isco | `orbits.isco_radius`, `isco_radius_numeric`, `marginal_stability_function` | section 10 |
