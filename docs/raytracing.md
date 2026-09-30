# Ray tracing: observer, camera and initial conditions

This chapter derives and documents the camera of PROJECT.md sections 15 and 16
as implemented in `src/kerrray/raytracing/camera.py` (docs/architecture.md
sections 1.3 and 5): the observer frame, the mapping from a pixel to the
initial photon four-momentum, its sign conventions, the finite-distance
correction with respect to the textbook celestial coordinates, and the
limitations. Everything in sections 2 to 6 is exact general relativity; the
approximations are collected in section 7 with WHAT / WHY / LIMITATION labels.
The tetrad and metric formulas were verified numerically (section 8) rather
than copied; the references give the forms they were checked against.

References:

- BPT: Bardeen, Press and Teukolsky 1972, ApJ 178, 347: eq. 2.1 (metric in the
  form `-e^{2 nu} dt^2 + e^{2 psi} (dphi - omega dt)^2 + e^{2 mu_1} dr^2 +
  e^{2 mu_2} dtheta^2`) and section III (the locally non-rotating frame, LNRF,
  called ZAMO below).
- Frolov and Novikov 1998, *Black Hole Physics*, section 3.3 (ZAMO / locally
  non-rotating observers in Kerr).
- Bardeen 1973, "Timelike and null geodesics in the Kerr metric", in *Black
  Holes (Les Houches 1972)*: celestial coordinates of a distant observer
  `alpha = -xi / sin(theta_o)`, `beta = +-sqrt(eta + a^2 cos^2(theta_o) - xi^2
  cot^2(theta_o))`.
- Cunha and Herdeiro 2018, Gen. Rel. Grav. 50, 42: image-plane coordinates
  defined from the local tetrad components at a finite observer radius, the
  convention adopted in docs/architecture.md section 1.3.
- Press, Teukolsky, Vetterling and Flannery, *Numerical Recipes*, section 5.6
  (cancellation-free quadratic roots).
- docs/equations.md sections 1 to 3 (metric and inverse metric, verified in
  Phase 1) and docs/decisions.md D-001, D-002, D-008.

## 1. Conventions and the observer's position

The conventions of docs/equations.md section 1 apply: `G = c = 1`, lengths in
units of the mass `M`, signature `(-, +, +, +)`, Boyer-Lindquist `(t, r,
theta, phi)` with index order `0, 1, 2, 3`, `a = spin * M`, `Sigma = r^2 + a^2
cos^2(theta)`, `Delta = r^2 - 2 M r + a^2`, `A = (r^2 + a^2)^2 - a^2 Delta
sin^2(theta)`. The momentum is stored covariantly, `p_mu`, so that `E = -p_t`
and `L_z = p_phi` are read off directly; the null Carter constant is `Q =
p_theta^2 + cos^2(theta) (L_z^2 / sin^2(theta) - a^2 E^2)`.

The observer sits at Boyer-Lindquist `(r_o, theta_o, phi_o)`:

| `Camera` field | Meaning |
|---|---|
| `radius` | `r_o` in units of `M`; must exceed the outer horizon `r_plus`. |
| `inclination_deg` | `theta_o` in degrees (D-001): 0 on the spin axis, 90 in the equatorial plane; clamped to `[1e-3, 180 - 1e-3]` degrees (D-008, section 7). |
| `phi_deg` | `phi_o` in degrees; by axisymmetry it only fixes the azimuthal origin of the image. |
| `fov` | Half-width of the image plane in units of `M`: `abs(alpha)` and `abs(beta)` at the outer edge of the outermost pixels. |
| `resolution` | Pixels per side of the square image; an even integer `>= 2`. |

`Camera.from_config(observer, raytrace)` builds the camera from the
`observer` and `raytrace` configuration blocks of docs/architecture.md
section 7. The camera is a pinhole: every ray starts at the observer's event
`(0, r_o, theta_o, phi_o)` with a different direction.

## 2. The ZAMO tetrad (exact)

The zero-angular-momentum observer is the observer whose four-velocity has
`u_phi = 0`. It is the natural rest frame of a distant camera in Kerr because
it is the frame in which the metric is locally Minkowskian *and* the
frame-dragging is subtracted out (BPT section III; Frolov and Novikov section
3.3). Expanding the BPT form of the `(t, phi)` block,

```
-e^{2 nu} dt^2 + e^{2 psi} (dphi - omega dt)^2
    = (-e^{2 nu} + omega^2 e^{2 psi}) dt^2 - 2 omega e^{2 psi} dt dphi + e^{2 psi} dphi^2
```

and matching the Boyer-Lindquist components `g_tt`, `g_tphi`, `g_phph` gives

```
g_phph  = e^{2 psi}
g_tphi  = -omega g_phph              =>  omega   = -g_tphi / g_phph
g_tt    = -e^{2 nu} + omega^2 g_phph =>  lapse^2 = e^{2 nu} = -(g_tt - g_tphi^2 / g_phph)
```

With the components of docs/equations.md section 2 these are the familiar
`omega = 2 M a r / A` and `lapse^2 = Sigma Delta / A`; equivalently, from the
inverse metric of section 3, `lapse^2 = -1 / g^tt` and `omega = g^tphi /
g^tt`. Both equivalences are checked numerically (section 8). The lapse is
positive for every `r > r_plus` (`Delta > 0`), including inside the
ergosphere, so the ZAMO exists everywhere outside the horizon.

The orthonormal tetrad `e_(a)`, `a = t, r, theta, phi`, has the contravariant
components (rows of `zamo_tetrad(st, r, theta)`)

```
e_(t)     = (1 / lapse) (1, 0, 0, omega)
e_(r)     = (0, 1 / sqrt(g_rr), 0, 0)
e_(theta) = (0, 0, 1 / sqrt(g_thth), 0)
e_(phi)   = (0, 0, 0, 1 / sqrt(g_phph))
```

and the dual co-frame (one-forms) is

```
e^(t) = lapse dt,   e^(phi) = sqrt(g_phph) (dphi - omega dt),
e^(r) = sqrt(g_rr) dr,   e^(theta) = sqrt(g_thth) dtheta.
```

Orthonormality `e_(a)^mu e_(b)^nu g_mu_nu = eta_ab`, `eta = diag(-1, 1, 1,
1)`: the three spatial rows are trivially unit and mutually orthogonal
because the metric is diagonal in `(r, theta)`; for the time row

```
g(e_(t), e_(t))   = (g_tt + 2 omega g_tphi + omega^2 g_phph) / lapse^2
                  = (g_tt - g_tphi^2 / g_phph) / lapse^2 = -1,
g(e_(t), e_(phi)) = (g_tphi + omega g_phph) / (lapse sqrt(g_phph)) = 0,
```

both by `omega g_phph = -g_tphi`. The second line is the statement `u_phi = 0`
for `u = e_(t)`: the observer has zero angular momentum. The completeness
relation `eta^{ab} e_(a)^mu e_(b)^nu = g^{mu nu}` follows and is also tested.

Limits and singular loci. For `a = 0`, `omega = 0` and the tetrad is the
static observer's frame `diag(1/sqrt(1 - 2M/r), sqrt(1 - 2M/r), 1/r, 1/(r
sin(theta)))`. On the axis `g_phph = sin^2(theta) (...) -> 0`, so `e_(phi)`
diverges: the tetrad is singular there (the azimuth `phi` is undefined on the
axis), which is why the observer inclination is clamped away from it (section
7). On the horizon `Delta = 0` makes `lapse -> 0` and `g_rr -> infinity`.
`zamo_tetrad` raises `ValueError` for `r <= r_plus`, `theta <= 0`, `theta >=
pi` or non-finite inputs instead of returning infinities.

## 3. The image plane: pixels and celestial coordinates

The image is a square of `n x n` pixels (`n = resolution`, `H = W = n`) with
pixel side `d = 2 fov / n`. The pixel centres carry half-pixel offsets:

```
alpha_j = -fov + (j + 1/2) d = fov (2 j + 1 - n) / n      column j = 0 .. n-1
beta_i  =  fov - (i + 1/2) d = -fov (2 i + 1 - n) / n     row    i = 0 .. n-1
```

`Camera.pixel_coordinates()` returns `alpha` and `beta` as `(H, W)` arrays
with `alpha[i, j] = alpha_j` and `beta[i, j] = beta_i`, so

- `alpha` increases to the right (with the column index),
- `beta` decreases down the rows: row 0 is the top of the image (`beta > 0`),
- the outer edges of the outermost pixels lie exactly at `+-fov`,
- the arrays display in the correct orientation with
  `matplotlib.pyplot.imshow(img, origin="upper")`,
- the grid is exactly reflection symmetric: `alpha[:, ::-1] == -alpha` and
  `beta[::-1, :] == -beta` hold to the last bit because the integer factor
  `2 j + 1 - n` is negated exactly.

`initial_states` flattens the grid in C (row-major) order: ray `k = i W + j`
belongs to pixel `(i, j)`, and `Y.reshape(H, W, 8)` restores the image.

Why `alpha = 0` is never sampled. With even `n` the factor `2 j + 1 - n` is
odd, so no centre has `alpha = 0` or `beta = 0`; the smallest magnitude is
`d / 2`. Three reasons: (i) pixel centres with half-pixel offsets are the
standard area-sampling convention, each pixel representing the square around
its centre, so the edge of the image is at `+-fov` rather than half a pixel
beyond; (ii) an even grid is symmetric under both reflections, so the image of
a reflection-symmetric configuration is symmetric pixel by pixel, which makes
the mirror tests of section 8 exact and the sub-pixel boundary extraction of
the shadow unbiased about the image centre; (iii) `alpha = 0` is the ray with
`L_z = 0` exactly and `beta = 0` for an equatorial observer is the ray with
`p_theta = 0` exactly (the ray that stays in the equatorial plane), while
`alpha = beta = 0` is the exactly radial ray with `b = 0`. These are the
degenerate members of the geodesic families: an `L_z = 0` ray crosses the
axis where `phi` jumps by `pi` and `g^phph` is singular, and an exactly
equatorial ray sits on a boundary of the `theta` motion. Excluding them from
the launch grid keeps the integrator away from those loci at launch without
changing the physics, since they form a set of measure zero in the image.

## 4. From a pixel to the photon momentum (exact)

The observer measures the direction of an arriving photon with the tetrad of
section 2. The tetrad components of the photon momentum are `p^(a) =
eta^{ab} e_(b)^mu p_mu`, i.e. `p^(t) = -e_(t)^mu p_mu`, `p^(i) = e_(i)^mu
p_mu`. For a photon that *arrives* at the observer from the hole side its
spatial direction in the ZAMO frame is `n = (p^(r), p^(theta), p^(phi)) /
p^(t)` with `p^(r) > 0` (moving outward when it reaches the camera) and `n .
n = 1` because the momentum is null. The celestial coordinates of
docs/architecture.md section 1.3 (Cunha and Herdeiro 2018 style) are

```
alpha = -r_o p^(phi) / p^(t),      beta = r_o p^(theta) / p^(t),
```

and the camera inverts this: for the pixel `(alpha, beta)` it sets

```
p^(t) = 1,   p^(phi) = -alpha / r_o,   p^(theta) = beta / r_o,
p^(r) = +sqrt(1 - (alpha^2 + beta^2) / r_o^2).
```

The scaling `p^(t) = 1` is free (the affine parameter is rescaled) and the
square root requires `alpha^2 + beta^2 < r_o^2`: pixels violating this are not
directions on the observer's sky and `initial_states` raises (`fov < r_o /
sqrt(2)` guarantees it). Geometrically `(alpha, beta) / r_o` are the two
transverse direction cosines of the arriving photon: `(alpha, beta)` is the
orthographic projection of the observer's local sky onto a plane at distance
`r_o`, the natural "image plane" of a pinhole camera.

Orientation and signs. The observer looks towards the hole, i.e. along
`-e_(r)`. "Up" in the image is the direction of the spin axis projected on the
sky, which is `-e_(theta)` (decreasing `theta` points towards the north pole).
For a right-handed screen `(right, up, out of the screen towards the viewer)`
with `up = -e_(theta)` and `out = +e_(r)` one has `right = up x out =
-e_(theta) x e_(r) = +e_(phi)` (using `e_(r) x e_(theta) = e_(phi)`,
`e_(theta) x e_(phi) = e_(r)`, `e_(phi) x e_(r) = e_(theta)` for the
right-handed orthonormal triad). A photon seen on the right of the image
therefore comes *from* the `+phi` side, so its momentum at the camera points
towards `-phi`: `p^(phi) < 0`. Hence `alpha = -r_o p^(phi) / p^(t) > 0` on
the right, and likewise a photon seen above the centre comes from the north
side and moves towards `+theta`, `beta = r_o p^(theta) / p^(t) > 0` at the
top. For the observer at `phi_o = 0` (Cartesian `x > 0`) the right of the
image is the `+y` side; matter co-rotating with a hole of `a > 0` on the near
side moves towards `+y`, so the *left* side of the image is the approaching
(prograde) side. Prograde photons (`a L_z > 0`) have `p^(phi) > 0` with `a >
0`, i.e. `alpha < 0`: the flattened side of the Kerr shadow appears on the
left, exactly as in Bardeen's `alpha = -xi / sin(theta_o)`.

Coordinate momentum. `p^mu = sum_a p^(a) e_(a)^mu` and `p_mu = g_mu_nu p^nu`;
because the tetrad is orthonormal, `g_mu_nu p^mu p^nu = eta_ab p^(a) p^(b) =
-1 + p^(r)^2 + p^(theta)^2 + p^(phi)^2 = 0` identically, so the state is null
up to round-off. In closed form (co-frame of section 2, `p_mu = eta_ab p^(b)
e^(a)_mu`),

```
p_t     = -lapse p^(t) - omega sqrt(g_phph) p^(phi)
p_r     =  sqrt(g_rr) p^(r)
p_theta =  sqrt(g_thth) p^(theta)
p_phi   =  sqrt(g_phph) p^(phi)
```

which the tests compare against the tetrad-projected result. Two immediate
consequences: mirroring `alpha` flips `L_z = p_phi` and leaves `p_r`,
`p_theta` and the ZAMO energy `lapse p^(t) = -(p_t + omega p_phi)` unchanged
(while `E = -p_t` itself changes by `2 omega L_z` through frame dragging), and
mirroring `beta` flips `p_theta` only. At `theta_o = pi/2`, `sqrt(g_thth) =
r_o` so `p_theta = beta` exactly.

## 5. Bardeen's limit and the finite-distance correction (exact)

Substituting the closed forms into the definitions of `alpha`, `beta` with
`E = -p_t`, `L_z = p_phi` gives the exact relation between a pixel and the
constants of motion of the ray for an observer at *finite* `r_o`:

```
alpha = -r_o lapse L_z / [sqrt(g_phph) (E - omega L_z)]
beta  =  r_o lapse p_theta / [sqrt(g_thth) (E - omega L_z)]
```

with `lapse`, `omega`, `g_phph = A sin^2(theta_o) / Sigma`, `g_thth = Sigma`
evaluated at `(r_o, theta_o)`. As `r_o -> infinity`: `lapse -> 1`, `omega ->
0`, `sqrt(g_phph) / r_o -> sin(theta_o)`, `sqrt(g_thth) / r_o -> 1`, hence

```
alpha -> -xi / sin(theta_o),    beta -> p_theta / E,
beta^2 -> eta + a^2 cos^2(theta_o) - xi^2 cot^2(theta_o),
```

Bardeen's celestial coordinates (`xi = L_z / E`, `eta = Q / E^2`; the last
line uses the definition of `Q`). The corrections are `O(M / r_o)`:
`lapse = 1 - M / r_o + ...`, `omega = O(a M / r_o^3)`.

Schwarzschild, exactly. For `a = 0` the total angular momentum is `L^2 = Q +
L_z^2 = p_theta^2 + p_phi^2 / sin^2(theta)` and the impact parameter is `b =
L / E`. With `E = lapse p^(t) = sqrt(1 - 2M/r_o) p^(t)` and `L^2 = r_o^2
[p^(theta)^2 + p^(phi)^2] = (alpha^2 + beta^2) p^(t)^2`,

```
b = sqrt(alpha^2 + beta^2) / sqrt(1 - 2 M / r_o)
  = sqrt(alpha^2 + beta^2) [1 + M / r_o + (3/2) (M / r_o)^2 + ...].
```

The pixel radius `sqrt(alpha^2 + beta^2)` underestimates the impact
parameter by the factor `sqrt(1 - 2M/r_o)`: the local sky of an observer at
finite radius is "magnified" by the gravitational blue-shift of the local
frame. Measured with the implementation (`fov = 12`, `8 x 8`, `theta_o = 60`
degrees; `tests/test_camera.py` and the values recorded here were produced by
the same code):

| `r_o / M` | `max abs(b / sqrt(alpha^2 + beta^2) - 1)` | `M / r_o` | residual of the exact relation |
|---|---|---|---|
| 1e2 | 1.0153e-2 | 1e-2 | 4.4e-16 |
| 1e3 | 1.0015e-3 | 1e-3 | 4.4e-16 |
| 1e4 | 1.00015e-4 | 1e-4 | 4.4e-16 |
| 1e6 | 1.000002e-6 | 1e-6 | 2.2e-16 |

The uncorrected discrepancy is `M / r_o (1 + 3/2 M / r_o)` to the digits shown
and scales as `1 / r_o`; the exact relation holds to round-off. For Kerr
(`a = 0.9`, `theta_o = 60` degrees) the measured deviations of `alpha` from
`-xi / sin(theta_o)` and of `beta` from `p_theta / E` are `1.001e-3`,
`1.000e-4` and `1.000e-6` at `r_o = 1e3, 1e4, 1e6`, again `M / r_o` to
leading order.

Consequence for the shadow (Phase 5 and later): a numerical shadow boundary
extracted in pixel coordinates `(alpha, beta)` from a camera at `r_o = 1000`
is smaller than Bardeen's analytic curve (which lives at `r_o = infinity`) by
a relative `M / r_o = 1e-3`, i.e. about `5e-3 M` on a shadow of radius `~5 M`.
This is far below one pixel at any resolution used here (`d = 0.375 M` at
`64^2`, `0.047 M` at `512^2`) but it is a systematic, not a random, offset;
the comparison in `raytracing/boundary.py` should either quote it as part of
the error budget or rescale the numerical boundary by `1 / sqrt(1 - 2M/r_o)`
(exact for `a = 0`, leading order otherwise).

## 6. Backward tracing: reversal and re-normalisation (exact)

The ray tracer integrates *backwards* from the camera: it needs the geodesic
that arrives at the observer with momentum `p_mu`, followed into the past.
The geodesic equation in Hamiltonian form (`dx/dlambda = g^{mu nu} p_nu`,
`dp_mu/dlambda = -(1/2) d_mu g^{ab} p_a p_b`) is invariant under `(lambda,
p) -> (-lambda, -p)`, so the past-directed ray is obtained exactly by
reversing the covariant momentum and integrating forwards in `lambda`
(docs/architecture.md section 1). `initial_states(cam, st)` therefore returns
`p_mu -> -p_mu` (the physical photon is available with `reverse=False`).

Consequences that every consumer must respect:

- The reversed ray has `E = -p_t < 0` and `L_z -> -L_z`. The statement "a
  future-directed photon has `E > 0`" applies to the physical photon
  (`reverse=False`, camera outside the ergosphere), not to the integrated
  state. `Q`, `xi = L_z / E` and `eta = Q / E^2` are invariant under the
  reversal, so *ratios* of constants are the safe quantities to attach to a
  pixel; the prograde/retrograde split of a pixel is `a xi > 0` / `a xi < 0`
  (equivalent to `a L_z > 0` for the physical photon, and to `a L_z < 0` for
  the reversed state).
- The reversed ray moves inward at launch: `dr/dlambda = g^rr p_r < 0`.
  Classification uses the integration direction: `ESCAPED` needs `r >=
  escape_radius` *and* `dr/dlambda > 0` in the integration direction (D-002),
  so a ray launched from `r_o = escape_radius` is never escaped at its first
  step, and `CAPTURED` (`r <= r_plus + horizon_epsilon`) is direction
  independent.
- The time coordinate of the reversed ray runs backwards (`dt/dlambda < 0`);
  elapsed coordinate time along the trajectory is `-t`.

Re-normalisation. The tetrad projection is null to round-off (measured
residual `abs(g^{mu nu} p_mu p_nu) < 7e-16` over all tested spins and
inclinations; `NULL_CHECK_TOLERANCE = 1e-10` guards against a broken tetrad).
After the reversal the null condition is re-imposed exactly by solving

```
g^tt p_t^2 + 2 g^tphi p_phi p_t + (g^rr p_r^2 + g^thth p_theta^2 + g^phph p_phi^2) = 0
```

for `p_t` with the cancellation-free quadratic formula and keeping the root
nearest the current `p_t`. Outside the ergosphere `g^tt < 0` and `g^phph > 0`,
so the two roots have opposite signs and "nearest" coincides with "same sign
as the current value" (negative for the physical photon, positive for the
reversed ray). Inside the ergosphere `g^phph` can be negative and both roots
can be positive (negative-energy photons); the nearest-root rule still picks
the branch the tetrad constructed. The change made by the re-normalisation is
of the order of the round-off it removes (`abs(delta p_t / p_t) ~ 1e-16`), so
the reversed state equals the negated physical state to `1e-13` relative in
the tests.

## 7. Approximations and limitations

**Axis clamp (D-008).** WHAT: `inclination_deg` within `1e-3` degrees of
either pole is moved to `1e-3` degrees (or `180 - 1e-3`) with a logged
warning, in `Camera.__post_init__` as in the configuration schema. WHY: the
ZAMO tetrad is singular on the axis (`g_phph = 0`, `e_(phi)` undefined)
because the azimuth is not a coordinate there; the image orientation
(section 4) needs a preferred `phi` direction. LIMITATION: an "on-axis" image
is the image seen from `theta_o = 1e-3` degrees; the spin-dependent
asymmetry of the image vanishes with `sin(theta_o)` (section 5: `alpha`
depends on `xi / sin(theta_o)` and the frame-dragging terms carry `a
sin^2(theta_o)`), so the clamped image is axisymmetric only up to terms of
that order, which are not zero. The magnitude of the residual asymmetry is a
property of the photon orbits and is measured by the inclination sweep
(EXP-004), not asserted here.

**Observer at finite radius.** WHAT: the physics is exact, but the pixel
coordinates `(alpha, beta)` are the orthographic sky coordinates of a ZAMO at
`r_o`, not the asymptotic celestial coordinates of Bardeen. WHY: a numerical
ray must start at a finite radius, and the ZAMO at `r_o` is the observer that
actually measures the direction. LIMITATION: `(alpha, beta)` differ from the
`r_o -> infinity` values by a relative `M / r_o` (section 5, exact factor
`(1 - 2M/r_o)^(-1/2)` for `a = 0`); comparisons with analytic curves must
account for this, and the image radius `sqrt(alpha^2 + beta^2)` must stay
below `r_o` (`fov < r_o / sqrt(2)` is enforced). The default `r_o = 1000 M`
makes the offset `1e-3`.

**Camera inside the ergosphere.** WHAT: `initial_states` accepts any `r_o >
r_plus` and logs a warning when `r_o <= r_E(theta_o)`. WHY: the ZAMO frame
exists down to the horizon and a near-hole observer is a legitimate
diagnostic. LIMITATION: inside the ergosphere the physical photon can have
`E < 0` (its `p^(t)` in the ZAMO frame is still `+1`), so no consumer may use
the sign of `E` to tell the physical photon from its reversal; the frame
components are the invariant statement. Measured example: `a = 0.9`, `r_o =
1.6`, `theta_o = 90` degrees, `fov = 0.8`, `4 x 4` pixels: `g^phph = -1.47`,
4 of the 16 physical photons have `E < 0` (minimum `-4.6e-3`, maximum
`0.398`), every state is null to `4.4e-16`, `p^(t) = 1` to `1e-12` and the
reversed states are the exact negatives of the physical ones.

**Pinhole camera, no aberration model.** WHAT: all rays start from one event;
a pixel is one direction, with no finite aperture or exposure. WHY: the shadow
and lensing observables of PROJECT.md are direction maps. LIMITATION: no
depth of field, and the image is that of an observer at rest in the ZAMO
frame; an observer with a different velocity would see the same rays with an
aberrated `(alpha, beta)` map (not implemented).

## 8. Verification summary (tests/test_camera.py)

| Check | Method | Tolerance | Measured |
|---|---|---|---|
| Tetrad orthonormality `e g e^T = eta` | 4 spins x 5 points incl. `r_plus + 0.1`, `theta = 0.05` | `1e-12` abs | worst `7.4e-15` |
| Completeness `e^T eta e = g^-1` | same points | `1e-12` rel | passes |
| `lapse^2 = -1/g^tt`, `omega = g^tphi/g^tt = -g_tphi/g_phph`, `u_phi = 0`, `u.u = -1` | closed-form inverse metric | `1e-12` | passes |
| Schwarzschild limit of the tetrad | static-observer diagonal frame | `1e-14` rel | passes |
| Small-spin behaviour | `max abs(e(a) - e(0)) / a` constant at `a = 1e-3, 1e-4` | `1e-2` | passes |
| Broadcasting, rejection of horizon / axis / non-finite | shapes, `ValueError` | exact | passes |
| Pixel grid: layout, half-pixel offsets, edges at `+-fov`, no zero, exact mirror symmetry | 4 resolutions | `1e-13` / exact | passes |
| Null condition of every pixel, reversed and physical | full `g^{mu nu}` contraction, 3 spins x 5 inclinations incl. the clamp | `1e-12` (abs and `/E^2`) | worst `6.9e-16` |
| Reversed `E < 0`, `dr/dlambda < 0`; physical `E > 0`, `dr/dlambda > 0` | signs | exact | passes |
| Reversal = exact sign flip; `xi`, `eta` invariant | compare `reverse=True/False` | `1e-13` rel | passes |
| Co-frame closed form of `p_mu`; row-major layout | section 4 formulas | `1e-12` rel | passes |
| `b = sqrt(L_z^2 + Q)/E` vs `sqrt(alpha^2 + beta^2)` (a = 0) | `r_o = 1e4`; `r_o = 1e6` | `1e-3`; `2e-6` | `1.00015e-4`; `1.000002e-6` |
| Exact finite-distance relation `b = b_0 / sqrt(1 - 2M/r_o)` | same | `1e-12` rel | `4.4e-16` |
| Discrepancy `= M/r_o (1 + ...)`, scales as `1/r_o` | `r_o = 1e3, 1e4, 1e5` | `2e-3` on the ratio | passes |
| Bardeen limit `alpha -> -xi/sin(theta_o)`, `beta -> p_theta/E`, `beta^2` relation | `a = 0.9, -0.5`, `r_o = 1e6` | `1e-5` rel | `1.0e-6` deviation |
| Prograde side is `alpha < 0` | sign of physical `a L_z` | exact | passes |
| Mirror `alpha`: `L_z` flips, `p_r`, `p_theta`, ZAMO energy unchanged, `delta p_t = 2 omega L_z` | 3 spins x 2 inclinations | `1e-12` | passes |
| Mirror `beta`: `p_theta` flips only | same | `1e-12` | passes |
| Equatorial observer: no `beta = 0` row; nearest row has `abs(p_theta) = d/2`; opposite `L_z` for opposite `alpha` | 3 spins | `1e-12` | passes |
| Axis clamp: warning, finite null states at `0`, `1e-4`, `180` degrees | `caplog` | `1e-12` | passes |
| Camera inside the ergosphere: warning, null, `p^(t) = 1`, exact reversal | `a = 0.9`, `r_o = 1.6` | `1e-12` | passes |

The tolerance `2e-6` at `r_o = 1e6` is deliberate: the leading correction is
exactly `M / r_o = 1e-6`, so a `1e-6` bound on the *uncorrected* relation
would sit on the knife edge; the exact relation is checked to `1e-12`
instead.

## 9. Implementation mapping

| Quantity | Function | Module |
|---|---|---|
| Observer, image plane, `(alpha, beta)` grid | `Camera`, `Camera.pixel_coordinates`, `Camera.from_config` | `raytracing/camera.py` |
| `omega`, `lapse`, tetrad `e_(a)^mu` (rows) | `zamo_tetrad(st, r, theta)` -> `(..., 4, 4)` | `raytracing/camera.py` |
| Pixel -> `p^(a)` -> `p^mu` -> `p_mu`, reversal, re-normalisation | `initial_states(cam, st, *, reverse=True)` -> `(N, 8)` | `raytracing/camera.py` |
| Null re-normalisation quadratic | `_null_pt` (private; the geodesics role's `null_momentum_pt` is the public equivalent) | `raytracing/camera.py` |
| Metric, inverse metric, horizon, ergosphere | `kerrray.geometry` (Phase 1) | `geometry/` |
| Inclination clamp constant | `INCLINATION_EPSILON_DEG` | `utils/config.py` |

The state layout is `[t, r, theta, phi, p_t, p_r, p_theta, p_phi]` (indices
`0` to `7`), float64, one row per pixel in row-major image order.
