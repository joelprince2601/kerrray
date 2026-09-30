# Strong-field lensing

PROJECT.md section 22. Implementation: `src/kerrray/physics/lensing.py`,
experiment `src/kerrray/experiments/lensing.py`, command `kerrray lens`
(`src/kerrray/commands/lens.py`), figures `reporting/figures.lensing_figure`.
Conventions: G = c = 1, Boyer-Lindquist coordinates, equatorial photons
(`theta = pi/2`), impact parameter `b = |L_z| / E`, and deflection in
radians with windings included (a ray that circles the hole `n` times has
`alpha > 2 pi n`).

## 1. Quantities

For each impact parameter the experiment records:

* the deflection angle `alpha`;
* the closest approach `r_min` (`photons.trajectories.closest_approach`);
* the number of orbital turns `Delta phi / (2 pi)`
  (`photons.trajectories.azimuthal_winding`).

Three deflection values are compared:

| function | kind | domain |
|---|---|---|
| `deflection_from_trajectory(traj)` | numerical (integrated geodesic) | Schwarzschild and Kerr, escaped equatorial rays |
| `schwarzschild_deflection_exact(st, b)` | exact integral by quadrature (reference) | Schwarzschild, `b > b_c` |
| `weak_field_deflection(st, b) = 4 M / b` | **approximation** (section 6) | `b >> M` |

## 2. Deflection from an integrated ray

In the asymptotically flat region, a point `(r, phi)` with coordinate
velocity `(dr/dlambda, dphi/dlambda)` moves in the direction

```
chi = phi + atan2(r dphi/dlambda, dr/dlambda).
```

Here `atan2(...)` is the angle between the velocity and the radial
direction. For a straight line `chi` is constant. The deflection is
`|chi_exit - chi_launch|`, with `phi` continuous so windings are included.

**Finite launch radius (derived here, leading order).** A Schwarzschild
photon with `E = 1` obeys

```
dphi/dr = (b / r^2) / sqrt(1 - b^2 (1 - 2M/r) / r^2).
```

Write `s = b / r` and expand in `M`. The perturbation of `dphi/dr` is
`-M b^3 / r^5`, and that of `tan(atan2(...)) = r dphi/dr` is `-M b^3 / r^4`.
Between a finite `r_0` and infinity, `chi` therefore still changes by

```
Delta chi = int_{r_0}^inf (-M b^3 / r^5) dr + M b^3 / r_0^4 = (3/4) M b^3 / r_0^4
```

per leg, so the measured deflection is short of the `r_0 -> infinity` value
by `1.5 M b^3 / r_0^4`, with corrections of order `M^2 b / r_0^3`.
Schwarzschild coordinates keep the flat part of the orbit an exact straight
line (`u = sin(phi) / b` solves the orbit equation without its `2 M u^3`
term), which is why the residual starts at `b^3 / r_0^4`.

In Kerr, `dphi/dlambda` gains `g^{t phi} p_t = 2 M a E / r^3 + ...`, which
adds `M a / r_0^2` per leg: the finite-`r_0` value differs by
`+-2 M |a| / r_0^2`. It is too large for prograde rays (`a L_z > 0`) and too
small for retrograde rays.

Measured (numerical minus exact integral, Schwarzschild, rk45 with rtol
`1e-10`, atol `1e-12`, launch and escape radius `r_0`):

| b [M] | r_0 [M] | numerical - exact [rad] | -1.5 M b^3 / r_0^4 [rad] |
|---|---|---|---|
| 6 | 200 | -2.007e-7 | -2.025e-7 |
| 10 | 200 | -8.231e-7 | -9.375e-7 |
| 20 | 200 | -7.465e-6 | -7.500e-6 |
| 6 | 1000 | +6.5e-10 | -3.2e-10 |
| 10 | 1000 | -3.5e-10 | -1.5e-9 |
| 20 | 1000 | -1.055e-8 | -1.200e-8 |

At `r_0 = 1000 M` the residual is at the level of the integration error. In
Kerr (`a = 0.9`, rtol `1e-11`) the change in deflection from `r_0 = 1000` to
`r_0 = 4000` is `-1.617e-6` (prograde, `b = 10`), `+1.654e-6` (retrograde,
`b = 10`) and `-1.606e-6` (prograde, `b = 20`). The leading-order prediction
is `2 M |a| (1/1000^2 - 1/4000^2) = 1.6875e-6`.

`photons.trajectories.deflection_angle` (geodesics role) uses the
straight-line correction `|Delta phi| - pi + arcsin(b/r_0) + arcsin(b/r_1)`.
The experiment records both.

## 3. The exact Schwarzschild integral (reference)

The deflection of a photon with closest approach `r_0` is (Darwin 1959,
Proc. R. Soc. A 249, 180; Weinberg 1972, *Gravitation and Cosmology*,
section 8.5)

```
alpha(b) = 2 int_{r_0}^inf dr / (r^2 sqrt(1/b^2 - (1 - 2M/r) / r^2)) - pi.
```

**Closest approach.** `r_0` is the largest root of `f(r) = r^3 - b^2 (r - 2M)`,
from the turning-point condition `b^2 = r_0^3 / (r_0 - 2M)`.

* For `b > b_c = 3 sqrt(3) M` it is bracketed by `[3M, b]`:
  `f(3M) = M (27 M^2 - b^2) < 0` and `f(b) = 2 M b^2 > 0`.
* It is found with `brentq`.
* It agrees with the trigonometric closed form
  `r_0 = (2b / sqrt 3) cos[(1/3) arccos(-3 sqrt(3) M / b)]` to at most
  `2.1e-13 M` for `b` from `5.3` to `100`.
* `b_c` itself comes from `photons.orbits.critical_impact_parameters` (root
  finding). The reference value `3 sqrt 3` is only used to check it.

**Removing the end-point singularity.**

1. With `u = 1/r` the integral is `2 int_0^{u_0} du / sqrt(P(u))`, where
   `P(u) = 1/b^2 - u^2 + 2 M u^3` and `P(u_0) = 0`.
2. Polynomial division gives
   `P(u) = (u_0 - u) Q(u)` with `Q(u) = (u + u_0) - 2M (u^2 + u u_0 + u_0^2)`.
3. The substitution `u = u_0 (1 - t^2)` (so `du = -2 u_0 t dt` and
   `u_0 - u = u_0 t^2`) turns the integral into one with a smooth integrand:

```
alpha = 4 sqrt(u_0) int_0^1 dt / sqrt(Q(u_0 (1 - t^2))) - pi.
```

**Quadrature settings.** QUADPACK `quad` is called with `epsrel = 1e-12`,
`epsabs = 0` and at most 500 sub-intervals. The largest error estimate of
the deflection over `b / b_c - 1 in [1e-8, 1e3]` (23 points) is
`1.6e-11 rad`.

## 4. Strong-field limit (Bozza 2002)

Near the photon sphere the deflection diverges logarithmically (Bozza 2002,
Phys. Rev. D 66, 103001):

```
alpha = -a_bar log(b / b_c - 1) + b_bar + O((b/b_c - 1) log(b/b_c - 1)).
```

For Schwarzschild, `a_bar = 1` and `b_bar = log[216 (7 - 4 sqrt 3)] - pi =
-0.40023` (Darwin 1959, as quoted by Bozza 2002). These are *comparison*
values (`darwin_strong_field_coefficients`).

`strong_field_fit(b, alpha, b_c=...)` is a linear least-squares fit of
`(a_bar, b_bar)`. Fitted to the exact integral over 12 points with
`b / b_c - 1` from `1e-6` to `1e-3`, it gives `a_bar = 0.99966` and
`b_bar = -0.39616`, with an rms residual of `5.5e-4 rad`. The difference
from `(1, -0.40023)` is the neglected `(b/b_c - 1) log(b/b_c - 1)` term. The
experiment fits the numerical rays with `b / b_c - 1 <= 0.1`
(`STRONG_FIELD_OFFSET_MAX`), so its coefficients are reported, not asserted
tightly.

## 5. Experiment and command

`experiments.lensing.run(cfg)` reads the `lensing` sub-block of
`experiment.parameters`: `impact_min`, `impact_max`, `n_rays` and
`launch_radius`, with defaults 3, 20, 40 and 1000.

* **Sampling.** Scattering rays are spaced geometrically in `b / b_c - 1`,
  from `1e-3` (`CRITICAL_OFFSET_MIN`) when the range reaches below `b_c`.
  One ray in eight is placed below `b_c` to record capture.
* **Integration.** Each ray is launched inward in the equatorial plane from
  `launch_radius` and integrated with the scalar integrator, which records
  the path. The escape radius equals the launch radius.
* **Scans.** Kerr gets a prograde and a retrograde scan. Schwarzschild gets
  one scan compared with the exact integral.
* **Outputs.** Per ray: deflection (both estimates), exact value (spin 0),
  `4M/b`, closest approach, turns and state. Per scan: `b_c`, counts, max
  and rms deviation from the exact integral, max deviation from `4M/b`, the
  strong-field fit and the maximum null error. Figures: deflection against
  `b`, and deflection against `b / b_c - 1` on a log axis. The run also
  writes `report.md` with the nine section 36 headings.

`kerrray lens [--spin] [--impact-range MIN,MAX] [--n] [--launch-radius] [--config]`
prints the computed values through the Rich helpers of
`reporting/console.py`: `b_c`, counts, deviations, the fit, the null error,
the runtime and the paths. The default config is `configs/lensing.yaml`.
Invalid ranges exit with code 1.

## 6. Approximations and limitations

**Weak-field deflection `4 M / b`.**

* **WHAT:** the leading term of the expansion of the exact deflection in
  `M / b` (Weinberg 1972, section 8.5). The next term is `15 pi M^2 / (4 b^2)`.
* **WHY:** it is the textbook reference against which strong-field bending
  is contrasted.
* **LIMITATION:** it is exact only as `b -> infinity`. Measured relative
  error against the exact integral: `-0.322` at `b = 10 M`, `-0.0297` at
  `100 M` and `-0.00295` at `1000 M`. With the second-order term included
  these become `-0.123`, `-1.08e-3` and `-1.07e-5`. The formula is
  meaningless near `b_c`. It does not depend on spin; frame dragging enters
  at order `a M / b^2`.

**Other limitations.**

* Equatorial rays only.
* The exact integral is implemented for spin 0 only.
* Rays with `b / b_c - 1 < 1e-3` are left to the near-critical experiment
  (PROJECT.md section 21).
* The finite launch radius leaves the residual of section 2.

## 7. Tests

`tests/test_lensing.py`:

* `b_c` computed and checked against `3 sqrt 3 M`; closest approach against
  the trigonometric form;
* the exact integral recovers `4M/b` for large `b`, grows monotonically
  towards `b_c` and winds;
* numerical deflection against the exact integral at `b = 10`, and
  convergence with the launch radius;
* in Kerr, prograde rays are less deflected, and the frame-dragging
  residual is checked;
* strong-field fit on synthetic data and on the exact integral against
  Darwin/Bozza;
* impact-parameter sampling;
* the Schwarzschild experiment run, and `kerrray lens` on a fresh Typer app.
