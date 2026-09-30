# Equations and Conventions

The single place where every equation KerrRay implements is stated with its
definition, coordinate convention, unit convention, reference and mapping to
the implementing function (PROJECT.md section 43). Every formula below was
verified before it entered the code: symbolically with SymPy in the test
suite (PROJECT.md section 32) and numerically against the lambdified symbolic
expressions at random points. Nothing was copied from memory into the code
without such a check. Exact general relativity is marked *exact*; the one
approximation in this chapter (the plotting embedding) carries a
WHAT / WHY / LIMITATION label.

References used for the geometry chapters:

- BPT: Bardeen, Press and Teukolsky 1972, ApJ 178, 347, eq. 2.1 and the
  definitions that follow it.
- MTW: Misner, Thorne and Wheeler 1973, *Gravitation*, section 33.2
  (eq. 33.2 with the Kerr-Newman charge set to zero) and the horizon and
  ergosphere discussion of chapter 33; chapter 8, eq. 8.24b (Christoffel
  definition).
- Chandrasekhar 1983, *The Mathematical Theory of Black Holes*, chapter 6.
- Visser 2007, arXiv:0706.0622, *The Kerr spacetime: a brief introduction*.
- Carroll 2004, *Spacetime and Geometry*, section 5.4 (Schwarzschild
  Christoffel symbols used as a limiting-case reference).

## 1. Conventions

| Item | Convention |
|---|---|
| Units | Geometric units, `G = c = 1`. The mass `M` is a parameter (`Spacetime.mass`, default `1.0`); lengths, times and the affine parameter are in units of `M`. |
| Signature | `(-, +, +, +)`. |
| Coordinates | Boyer-Lindquist `(t, r, theta, phi)`, index order `0, 1, 2, 3`. `theta` is the polar angle from the spin axis, `phi` the azimuth. Angles are radians inside the engine; degrees only at the configuration and CLI boundary. |
| Spin | `Spacetime.spin` is the dimensionless `a* = a / M` with `abs(a*) < 1` (PROJECT.md section 4.1); `a = spin * M`. `a > 0`: rotation towards `+phi`. Negative spins are accepted (retrograde convention of docs/architecture.md). |
| Auxiliary functions | `Sigma = r^2 + a^2 cos^2(theta)`, `Delta = r^2 - 2 M r + a^2`, `A = (r^2 + a^2)^2 - a^2 Delta sin^2(theta)`. |
| Index order of `Gamma` | `christoffel(...)[..., mu, alpha, beta] = Gamma^mu_{alpha beta}`. |

Implementation: `kerrray.geometry.metric.Spacetime` (frozen dataclass,
validates `mass > 0` and `abs(spin) < 1`), `sigma`, `delta`.

## 2. Kerr metric (exact)

Covariant components in Boyer-Lindquist coordinates, all others zero
(MTW eq. 33.2 with Q = 0; identical in Chandrasekhar ch. 6 and Visser 2007):

```
g_tt     = -(1 - 2 M r / Sigma)
g_tphi   = -2 M a r sin^2(theta) / Sigma           (= g_phit)
g_rr     = Sigma / Delta
g_thth   = Sigma
g_phph   = (r^2 + a^2 + 2 M a^2 r sin^2(theta) / Sigma) sin^2(theta)
```

BPT eq. 2.1 writes the same line element as
`ds^2 = -e^{2nu} dt^2 + e^{2psi} (dphi - omega dt)^2 + e^{2mu1} dr^2 + e^{2mu2} dtheta^2`
with `e^{2nu} = Sigma Delta / A`, `e^{2psi} = A sin^2(theta) / Sigma`,
`e^{2mu1} = Sigma / Delta`, `e^{2mu2} = Sigma`, `omega = 2 M a r / A`.
`tests/test_metric.py::test_symbolic_mtw_and_bpt_forms_agree` shows with
`sympy.simplify` that the two published forms are the same matrix; this is
the cross-reference check required by CLAUDE.md.

Schwarzschild limit `a = 0` (exact): the metric becomes diagonal with
`g_tt = -(1 - 2M/r)`, `g_rr = 1/(1 - 2M/r)`, `g_thth = r^2`,
`g_phph = r^2 sin^2(theta)`; for small `a` the difference from Schwarzschild
is `O(a)` (linear through `g_tphi`, quadratic in the diagonal).

Determinant identity (verified symbolically first, then asserted):
`det g = -Sigma^2 sin^2(theta)`, equivalently `g_tt g_phph - g_tphi^2 =
-Delta sin^2(theta)`.

Implementation: `metric_components` (five components), `metric` (array of
shape `(..., 4, 4)`), `schwarzschild`, `kerr`.

## 3. Inverse metric (exact)

Contravariant components (BPT definitions; derived by inverting the
`(t, phi)` block with the determinant identity above):

```
g^tt     = -A / (Sigma Delta)
g^tphi   = -2 M a r / (Sigma Delta)
g^rr     = Delta / Sigma
g^thth   = 1 / Sigma
g^phph   = (Delta - a^2 sin^2(theta)) / (Sigma Delta sin^2(theta))
         = 1 / (Sigma sin^2(theta)) - a^2 / (Sigma Delta)
```

`g . g^-1 = 1` holds symbolically (`sympy.simplify`) and numerically at
random points for spins `{0, 0.3, -0.5, 0.9, 0.998}`.

Implementation: `inverse_metric_components`, `inverse_metric`.

## 4. Derivatives of the metric (exact, closed form)

Only `d/dr` and `d/dtheta` are non-zero (stationarity and axisymmetry).
With `Sigma_r = 2 r`, `Sigma_theta = -a^2 sin(2 theta)`, `Delta_r = 2 (r - M)`,
`A_r = 4 r (r^2 + a^2) - 2 a^2 (r - M) sin^2(theta)`,
`A_theta = -a^2 Delta sin(2 theta)`, `S = Sigma Delta`,
`S_r = Sigma_r Delta + Sigma Delta_r`, `S_theta = Sigma_theta Delta`:

Covariant (`metric_derivatives`, used to assemble the Christoffel symbols):

```
d_r g_tt      = 2 M (Sigma - 2 r^2) / Sigma^2
d_th g_tt     = -2 M r Sigma_theta / Sigma^2
d_r g_tphi    = -2 M a sin^2(theta) (Sigma - 2 r^2) / Sigma^2
d_th g_tphi   = -2 M a r (sin(2 theta) Sigma - sin^2(theta) Sigma_theta) / Sigma^2
d_r g_rr      = (Sigma_r Delta - Sigma Delta_r) / Delta^2
d_th g_rr     = Sigma_theta / Delta
d_r g_thth    = Sigma_r,      d_th g_thth = Sigma_theta
d_r g_phph    = 2 r sin^2(theta) + 2 M a^2 sin^4(theta) (Sigma - 2 r^2) / Sigma^2
d_th g_phph   = (r^2 + a^2) sin(2 theta)
                + 2 M a^2 r (2 sin^2(theta) sin(2 theta) Sigma - sin^4(theta) Sigma_theta) / Sigma^2
```

Contravariant (`inverse_metric_derivatives`, the input of the Hamiltonian
geodesic equations `dp_mu/dlambda = -(1/2) (d_mu g^{ab}) p_a p_b`):

```
d_x g^tt      = -(A_x S - A S_x) / S^2                        (x = r, theta)
d_r g^tphi    = -2 M a (S - r S_r) / S^2
d_th g^tphi   = 2 M a r S_theta / S^2
d_r g^rr      = (Delta_r Sigma - Delta Sigma_r) / Sigma^2
d_th g^rr     = -Delta Sigma_theta / Sigma^2
d_x g^thth    = -Sigma_x / Sigma^2
d_r g^phph    = -Sigma_r / (Sigma^2 sin^2(theta)) + a^2 S_r / S^2
d_th g^phph   = -(Sigma_theta sin^2(theta) + Sigma sin(2 theta)) / (Sigma^2 sin^4(theta))
                + a^2 S_theta / S^2
```

Both sets were derived by hand from sections 2 and 3 and are verified against
`sympy.diff` of the symbolic metric and inverse metric to `1e-10` relative
(`tests/test_metric.py`, `tests/test_christoffel.py`).

## 5. Christoffel symbols (exact)

Definition (MTW eq. 8.24b):

```
Gamma^mu_{alpha beta} = (1/2) g^{mu nu} (d_alpha g_{nu beta} + d_beta g_{nu alpha} - d_nu g_{alpha beta})
```

Implementation choice: `christoffel` evaluates this definition with the
closed-form derivatives of section 4 and the closed-form inverse of section
3, contracting with `numpy.einsum`. This is exact (no finite differences, no
SymPy at runtime) and vectorised; the 20 independent non-zero components
are not typed out individually, which removes the transcription risk of a
hand-copied table. Verification (`tests/test_christoffel.py`):

- agreement with the definition evaluated symbolically by SymPy (lambdified,
  unsimplified) at random points for spins `{0, 0.3, -0.5, 0.9, 0.998}`,
  `1e-10` relative;
- symmetry `Gamma^mu_{alpha beta} = Gamma^mu_{beta alpha}` (exact equality);
- the zero pattern implied by the Killing symmetries (only the 20 index
  triples with an even number of `t`/`phi` indices, counting the upper index,
  are non-zero);
- metric compatibility, `d_c g^{ab} = -(Gamma^a_{cd} g^{db} + Gamma^b_{cd} g^{ad})`,
  which ties the Christoffel symbols to the independent closed-form
  inverse-metric derivatives (`1e-10` relative);
- the Schwarzschild limit against the textbook forms
  `Gamma^t_{tr} = M / (r (r - 2M))`, `Gamma^r_{tt} = M (r - 2M) / r^3`,
  `Gamma^r_{rr} = -M / (r (r - 2M))`, `Gamma^r_{thth} = -(r - 2M)`,
  `Gamma^r_{phph} = -(r - 2M) sin^2(theta)`, `Gamma^th_{rth} = 1/r`,
  `Gamma^th_{phph} = -sin(theta) cos(theta)`, `Gamma^ph_{rph} = 1/r`,
  `Gamma^ph_{thph} = cot(theta)` (Carroll 2004 section 5.4), `1e-13` relative.

Vacuum check: the Ricci tensor
`R_ab = d_c Gamma^c_ab - d_b Gamma^c_ac + Gamma^c_cd Gamma^d_ab - Gamma^c_bd Gamma^d_ac`
is built symbolically from the same metric (no simplification), lambdified,
and evaluated at 10 random points for spins `{0, 0.6, 0.95}`; every
`abs(R_ab)` is below `1e-8` times the largest `Gamma * Gamma` term at that
point (measured maximum of the ratio over the 30 test points: `1.1e-15`,
i.e. pure round-off). This is the check that the implemented components are
the vacuum Kerr solution.

## 6. Horizons and ergosphere (exact)

Horizons are the roots of `Delta = 0`, where `g_rr` diverges and `g^rr`
vanishes (MTW section 33.2; PROJECT.md section 6):

```
r_plus/minus = M +- sqrt(M^2 - a^2)
```

with `r_plus + r_minus = 2M`, `r_plus r_minus = a^2`. Photon capture uses the
outer horizon `r_plus` (`inside_horizon(st, r, epsilon)` is `r <= r_plus +
epsilon`, the CAPTURED rule of docs/architecture.md).

The ergosphere (outer stationary-limit surface) is where the Killing vector
`d/dt` becomes null, `g_tt = 0`, i.e. `Sigma = 2 M r` (MTW chapter 33;
PROJECT.md section 7):

```
r_E(theta) = M + sqrt(M^2 - a^2 cos^2(theta))
```

Verified (`tests/test_horizon.py`): `Delta(r_plus) = Delta(r_minus) = 0`
(`1e-12 M^2`), `g_tt(r_E(theta), theta) = 0` (`1e-12`) with `g_tt < 0` just
outside and `g_tt > 0` just inside, `r_E(0) = r_E(pi) = r_plus`,
`r_E(pi/2) = 2M`, `r_plus <= r_E(theta) <= 2M` and monotone from pole to
equator, `a -> 0` gives `r_plus -> 2M` and `r_E -> 2M`, `a -> M` gives
`r_plus -> M` monotonically with `r_plus - r_minus -> 0`, linear scaling with
`M`, evenness in the sign of `a`, and the boolean logic of `inside_horizon`
with and without `epsilon`.

Implementation: `horizon_radii`, `outer_horizon`, `ergosphere_radius`,
`inside_horizon` in `kerrray.geometry.horizons`.

## 7. Cartesian embedding for plots (approximation)

```
x = sqrt(r^2 + a^2) sin(theta) cos(phi)
y = sqrt(r^2 + a^2) sin(theta) sin(phi)
z = r cos(theta)
```

- WHAT: the flat-space oblate-spheroidal relation between Boyer-Lindquist
  `(r, theta, phi)` and Cartesian axes; it is the `r`-`theta` part of the
  Kerr-Schild transformation `x + i y = (r + i a) sin(theta) exp(i phi_KS)`
  (Visser 2007) without the azimuthal shift
  `phi_KS - phi = int a / Delta dr`.
- WHY: reduces to spherical polar coordinates for `a = 0`, draws the horizon
  `r = r_plus` as the oblate spheroid it is in Kerr-Schild coordinates
  (equatorial radius `sqrt(2 M r_plus)`, polar radius `r_plus`), and avoids
  the azimuthal integral that diverges at the horizon.
- LIMITATION: not an isometric embedding and not the Kerr-Schild chart;
  plot distances are not proper distances and the azimuth shown is the
  Boyer-Lindquist `phi`. No physics is computed from these coordinates.

Angle units: `deg_to_rad`, `rad_to_deg` wrap `numpy.deg2rad` / `rad2deg` in
float64.

Verified (`tests/test_coordinates.py`): spherical limit for `a = 0`, the
confocal-ellipsoid identity `(x^2 + y^2)/(r^2 + a^2) + z^2/r^2 = 1`,
axis and equator, the oblate horizon, broadcasting, and degree/radian
round trips.

## 8. Implementation mapping

| Equation | Function (`kerrray.geometry`) | Module |
|---|---|---|
| `Sigma`, `Delta` | `sigma(st, r, theta)`, `delta(st, r)` | `metric.py` |
| `g_{mu nu}` | `metric_components`, `metric` | `metric.py` |
| `g^{mu nu}` | `inverse_metric_components`, `inverse_metric` | `metric.py` |
| `d_r g_{mu nu}`, `d_theta g_{mu nu}` | `metric_derivatives` | `metric.py` |
| `d_r g^{mu nu}`, `d_theta g^{mu nu}` | `inverse_metric_derivatives` | `metric.py` |
| `Gamma^mu_{alpha beta}` | `christoffel` (shape `(..., 4, 4, 4)`) | `christoffel.py` |
| `r_plus`, `r_minus` | `horizon_radii`, `outer_horizon` | `horizons.py` |
| `r_E(theta)` | `ergosphere_radius` | `horizons.py` |
| `r <= r_plus + epsilon` | `inside_horizon` | `horizons.py` |
| Plot embedding | `bl_to_cartesian` | `coordinates.py` |
| Angle units | `deg_to_rad`, `rad_to_deg` | `coordinates.py` |

All functions accept scalars or broadcastable NumPy arrays for `r`, `theta`
(and `phi`) and return float64. SymPy is used only in the tests
(`tests/symbolic_kerr.py`) and in the symbolic helper module; it is never
imported on the numerical path (`tests/test_christoffel.py` checks this in a
subprocess).

## 9. Verification summary (Phase 1)

| Check | Method | Tolerance | Test |
|---|---|---|---|
| MTW 33.2 form = BPT 2.1 form | `sympy.simplify` | exact | `test_metric.py` |
| `g . g^-1 = 1` | `sympy.simplify`; numeric at random points, 5 spins | exact; `1e-11` abs | `test_metric.py` |
| `det g = -Sigma^2 sin^2(theta)` | `sympy.simplify`; `numpy.linalg.det` | exact; `1e-10` rel | `test_metric.py` |
| Components vs symbolic metric | lambdified, random points | `1e-12` rel | `test_metric.py` |
| `d g^{mu nu}` vs `sympy.diff` | lambdified, random points | `1e-10` rel | `test_metric.py` |
| Schwarzschild limit | closed-form diagonal | `1e-15` rel | `test_metric.py` |
| Small-`a` scaling | `max abs(g(a) - g(0)) / a` constant | `1e-2` on the ratio | `test_metric.py` |
| Symmetry, signature, zero pattern | exact equality; `eigvalsh` | exact | `test_metric.py` |
| `d g_{mu nu}` vs `sympy.diff` | lambdified, random points | `1e-10` rel | `test_christoffel.py` |
| `Gamma` vs SymPy definition | lambdified, 5 spins x 5 points | `1e-10` rel | `test_christoffel.py` |
| `Gamma` lower-index symmetry | exact equality | exact | `test_christoffel.py` |
| Metric compatibility | `Gamma` vs `inverse_metric_derivatives` | `1e-10` rel | `test_christoffel.py` |
| Schwarzschild `Gamma` | textbook closed forms | `1e-13` rel | `test_christoffel.py` |
| Ricci tensor vanishes | symbolic, lambdified, 3 spins x 10 points | `1e-8` x `Gamma^2` scale | `test_christoffel.py` |
| `Delta(r_pm) = 0`, `g_tt(r_E) = 0`, limits | numeric | `1e-12` | `test_horizon.py` |
| Embedding identities | numeric | `1e-13` rel | `test_coordinates.py` |

## Geodesic equations

The geodesic chapters (Hamiltonian and Christoffel forms, conserved
quantities `E`, `L_z`, Carter `Q`, the null constraint) are written in
`docs/equations_geodesics.md` by the geodesics role and are merged here at
integration.

## Later chapters

Image-plane to photon-momentum mapping (PROJECT.md section 16, Phase 5) and
the lensing, redshift and disk-emission relations (sections 22 to 24,
Phase 7) are added by the phases that implement them.
