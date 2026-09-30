# Kerr experiments: EXP-005 frame dragging and EXP-003 spin sweep

This document describes the two Kerr experiments of PROJECT.md section 37
that are driven by `configs/kerr.yaml`: their setup, why the frame-dragging
setup isolates the effect it claims to show (PROJECT.md section 14), the
leading-order results they are compared with, and the numbers of the first
recorded runs. Every number below was produced by the code and is quoted from
the run's `summary.json`; nothing was typed in.

| Experiment | Driver | Parameter block | Run |
|---|---|---|---|
| EXP-005 frame dragging | `kerrray.experiments.frame_dragging.run` (physics in `kerrray.physics.frame_dragging`) | `experiment.parameters.frame_dragging` | `kerrray experiment --config configs/kerr.yaml --name frame_dragging` |
| EXP-003 spin sweep | `kerrray.experiments.spin_sweep.run` | `experiment.parameters.spin_sweep` | `kerrray experiment --config configs/kerr.yaml --name spin_sweep` |

Both write `runs/<run_id>/manifest.json` and `config.yaml` and
`reports/<run_id>/report.md`, `summary.json` and the figures
(`kerrray.experiments.base.run_experiment`).

## 1. EXP-005 setup

For every spin in `spins` (default `[-0.9, 0.0, 0.9]`, `M = 1`) two photons
are integrated with the configured integrator (RK45, `rtol = 1e-9`,
`atol = 1e-11`) and termination (`escape_radius = 1000 M`):

* **Equatorial family.** `theta = pi/2`, `E = 1`, `L_z = +b E` with
  `b = impact_parameter` (default 6 M), `Q = 0`, launched inward from
  `(r0, pi/2, phi = 0)` with `r0 = launch_radius` (1000 M). The constants of
  motion, the launch point and the sign of `p_r` are the same for every spin;
  only the spacetime changes. With `L_z > 0` the photon is prograde
  (`a L_z > 0`) for `a > 0` and retrograde for `a < 0`.
* **Polar (`L_z = 0`) family.** `E = 1`, `L_z = 0`, `Q = b_p^2` launched from
  `r0` at `theta_0 = pi - arcsin(b_p / r0)`, i.e. parallel to the spin axis at
  distance `b_p` from it, moving towards the north pole. The default
  `b_p = 1.25 * 2 sqrt(M r_escape) = 79.06 M` keeps the axis crossing of the
  bent ray (at `~b_p^2 / 4M`) beyond the escape radius, because the
  Boyer-Lindquist `theta` leaves `[0, pi]` at an axis crossing
  (docs/numerical_methods.md section 4).
* **Scaling.** The equatorial family is repeated for
  `scaling_impact_parameters` (default `[10, 20, 40]` M).

Measured per ray: termination state, azimuth swept `Delta phi = phi_end -
phi_0` (continuous, never wrapped), turns, closest approach, deflection (for
escaped equatorial rays, with the finite-radius straight-line correction of
`kerrray.photons.trajectories.deflection_angle`), the difference to the exact
quadrature of the separated equations (below), and the drifts of `E`, `L_z`,
`Q` and the null constraint. Per `|a|`: the odd part
`Delta phi(+a) - Delta phi(-a)`, the even part
`[Delta phi(+a) + Delta phi(-a)]/2 - Delta phi(0)`, and the polar azimuths.

## 2. Why this isolates frame dragging

1. **Only the spacetime differs.** Every ray of a family has the same
   `(E, L_z, Q)`, the same launch point and the same initial direction of
   radial motion. Any difference in the outcome is caused by the metric.
2. **Parity of the Kerr metric in `a`.** In Boyer-Lindquist coordinates
   (docs/architecture.md section 1.1) `g_tt`, `g_rr`, `g_thth`, `g_phph` and
   the inverse components `g^tt`, `g^rr`, `g^thth`, `g^phph` depend on `a`
   only through `a^2` (via `Sigma` and `Delta`), while `g_tphi = -2 M a r
   sin^2(theta)/Sigma` and `g^tphi = -2 M a r/(Sigma Delta)` are odd in `a`.
   The Hamiltonian `H = (1/2) g^{mu nu} p_mu p_nu` is therefore the sum of an
   even part and the single odd term `g^tphi p_t p_phi`. For the same initial
   data, any observable splits into a part even in `a` and a part odd in `a`,
   and the odd part vanishes when `g^tphi` is switched off: it is produced by
   the `t`-`phi` coupling, i.e. by the dragging of inertial frames, and by
   nothing else. The odd part `Delta phi(+a) - Delta phi(-a)` is what the
   experiment reports as the frame-dragging signal; the even part collects the
   `a^2` changes of the other components (oblateness of `Sigma`, the shifted
   horizon) and is reported separately, not attributed to frame dragging.
   By the mirror symmetry `phi -> -phi`, `Delta phi(-a; L_z) = -Delta phi(+a;
   -L_z)`, so the odd part is also the difference between the prograde and
   the retrograde photon in the same hole.
3. **The `L_z = 0` family has no other source of azimuthal motion.** With
   `p_phi = 0`, `dphi/dlambda = g^{tphi} p_t = 2 M a r E / (Sigma Delta)` and
   `dphi/dt = g^{tphi}/g^{tt} = 2 M a r / A = omega(r, theta)`, the angular
   velocity of the zero-angular-momentum observers (Bardeen, Press and
   Teukolsky 1972, ApJ 178, 347, section II). At `a = 0` the ray stays in its
   meridional plane (`Delta phi = 0` exactly); for `a != 0` it is carried
   round with the local inertial frames. Its `(r, theta)` motion is identical
   for `+a` and `-a` (the potentials contain `a` only as `a^2` when
   `L_z = 0`), so `Delta phi(-a) = -Delta phi(+a)` must hold to round-off.
4. **What is not claimed.** The total `Delta phi` and the total deflection are
   dominated by the mass (`4M/b`) and are not frame dragging. A captured ray
   is not compared: the Boyer-Lindquist `phi` diverges at the horizon for
   `a != 0` (`dphi/dlambda` contains `a P / Delta`), so its `Delta phi` is a
   coordinate artefact of where the integration stopped; pairs with a
   captured ray report `nan` asymmetries.

### Leading-order references (approximations)

From the separated equations (Carter 1968, Phys. Rev. 174, 1559; BPT 1972
eqs. 2.9-2.10 with `mu = 0`), for an equatorial photon with `E = 1` and
`xi = L_z`:

```
Sigma dr/dlambda   = -+ sqrt(R),   R = P^2 - Delta (xi - a)^2,   P = r^2 + a^2 - a xi
Sigma dphi/dlambda = -(a - xi) + a P / Delta
=> dphi/dr = [xi - a + a P/Delta] / (-+ sqrt(R))
```

To first order in `a` and in the weak field, with `u = 1/r`:
`a P/Delta = a + 2 a M u + ...` and
`R/r^4 = 1 - xi^2 u^2 + 2 M xi^2 u^3 (1 - 2a/xi) + ...`, so

```
Delta phi = 2 int_0^{u_t} (xi + 2 a M u) du / sqrt(1 - xi^2 u^2 + 2 M' xi^2 u^3),   M' = M (1 - 2a/xi)
          = pi + 4M'/xi + 4 a M/xi^2 + ... = pi + 4M/b - 4aM/b^2 + O(M^2/b^2, a^2, a M^2/b^3)
Delta phi(+a) - Delta phi(-a) = -8 a M / b^2           (b = L_z/E > 0)
```

which agrees with the weak-deflection Kerr expansion `alpha = 4M/b -+ 4aM/b^2
+ ...` (upper sign prograde; Sereno and De Luca 2006, Phys. Rev. D 74,
123009; Edery and Godin 2006, Gen. Rel. Grav. 38, 1715). For the `L_z = 0`
ray, `omega ~ 2 M a / r^3` integrated along the straight line at distance `b`
gives `Delta phi_polar = 4 a M / b^2`.

Approximation label (PROJECT.md section 43). WHAT: first-order weak-field
azimuths. WHY: they fix the sign and the `1/b^2` scaling that the measured odd
part must approach. LIMITATION: valid for `b >> M` and an infinitely distant
launch; at finite `b` the higher orders (`a M^2/b^3`, ...) are large, and the
experiment reports the measured ratio instead of assuming agreement. The
*exact* reference is `azimuth_quadrature`: the quadrature of `dphi/dr` above
from `r0` to the turning point (largest real root of the quartic `R`) and
back, with the substitution `r = r_t + s^2`, `scipy.integrate.quad` at
`1e-13`.

## 3. EXP-005 results

Run `20260929T053452Z-866a0d` (`runs/` and `reports/`), git commit `a9c864d5`
(working tree with uncommitted Phase 4 files), defaults of `configs/kerr.yaml`
(`kerrray.experiments.frame_dragging.run`); runtime 8.7 s.

Main family (`b = 6 M`):

| family | a | state | Delta phi [rad] | r_min [M] | deflection [rad] | integrated - quadrature [rad] | steps |
|---|---:|---|---:|---:|---:|---:|---:|
| equatorial | -0.9 | CAPTURED | -10.407678 (not comparable) | 1.435891 | - | - | 370 |
| equatorial | 0 | ESCAPED | 4.849672590 | 4.453409 | 1.719388 | 8.7e-09 | 173 |
| equatorial | +0.9 | ESCAPED | 4.155960860 | 4.972953 | 1.026208 | 7.4e-09 | 175 |
| polar (b_p = 79.057 M) | -0.9 | ESCAPED | -6.043730e-04 | 78.038673 | - | -6.2e-13 | 119 |
| polar | 0 | ESCAPED | 0.0 | 78.038660 | - | 0.0 | 119 |
| polar | +0.9 | ESCAPED | +6.043730e-04 | 78.038673 | - | 6.2e-13 | 119 |

* At `b = 6 M` the capture thresholds are `b_c = 2.844421 M` (prograde,
  `a = 0.9`), `5.196152 M` (`a = 0`) and `6.832319 M` (retrograde,
  `|a| = 0.9`), so the same photon escapes around the co-rotating hole and
  the Schwarzschild hole but is captured by the counter-rotating one. The
  prograde photon passes farther out (`r_min` 4.973 against 4.453 M) and is
  deflected less (1.026 against 1.719 rad).
* Polar family: `Delta phi(+0.9) = +6.043730e-04 rad`, `Delta phi(0) = 0`
  exactly, antisymmetry residual `Delta phi(+a) + Delta phi(-a) = 0.0`; the
  ratio to the leading-order `4aM/b_p^2 = 5.760e-04` is 1.049. The photon is
  dragged in the sense of rotation of the hole.
* Drifts on escaped rays: `Q` below `3.2e-13`, null constraint below
  `2.1e-09`; `E` and `L_z` are conserved exactly by the Hamiltonian form. The
  captured ray reaches a null error of `0.135` at the horizon margin
  (`1/Delta` in Boyer-Lindquist coordinates), which is why captured rays are
  excluded from the comparison.

Odd part against impact parameter (`|a| = 0.9`, all six rays escaped):

| b [M] | odd part [rad] | -8aM/b^2 [rad] | ratio | even part [rad] |
|---:|---:|---:|---:|---:|
| 10 | -0.21495384 | -0.072 | 2.985 | 0.021623 |
| 20 | -0.02819153 | -0.018 | 1.566 | 8.89e-04 |
| 40 | -0.00549918 | -0.0045 | 1.222 | 6.77e-05 |

The odd part is negative at every `b` (the prograde photon sweeps less
azimuth), and its ratio to the leading-order value decreases towards 1 as `b`
grows, as the higher-order terms fall off. The fitted exponent of `|odd
part|` against `b` over 10-40 M is `-2.644` (the asymptotic value is `-2`;
the steeper local slope is the contribution of the higher orders at these
moderate `b`). The integrated azimuths agree with the exact quadrature to
better than `1e-8` rad for every escaped ray.

Figures in the report directory: `frame_dragging_trajectories.png`
(equatorial paths with the horizons in the `bl_to_cartesian` embedding; the
`L_z = 0` paths as `y` against `z`, showing the sideways drag),
`frame_dragging_azimuth.png` (azimuth against the affine parameter, the
accumulated shift relative to `a = 0`, and the `L_z = 0` azimuth), and
`frame_dragging_scaling.png` (odd part and leading-order line, log-log).

## 4. EXP-003 spin sweep

Run `20260929T053501Z-04c5bf`, git commit `a9c864d5`, runtime 1.1 s. All
values from `kerrray.geometry.horizon_radii` and `kerrray.photons.orbits`
(closed forms and bracketed roots validated to `1e-12` in
`tests/test_orbits.py`); the shadow columns are the extent of the analytic
Bardeen (1973) curve for an observer at infinity at the configured
inclination (60 deg), not a traced image.

| spin | r_+ | r_- | r_ph pro | r_ph retro | b_c pro | b_c retro | ISCO pro | ISCO retro | shadow width | shadow centroid alpha |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2.000000 | 0.000000 | 3.000000 | 3.000000 | 5.196152 | 5.196152 | 6.000000 | 6.000000 | 10.392305 | 0.000000 |
| 0.25 | 1.968246 | 0.031754 | 2.695453 | 3.276237 | 4.675351 | 5.680114 | 5.155537 | 6.794850 | 10.355486 | 0.435333 |
| 0.5 | 1.866025 | 0.133975 | 2.347296 | 3.532089 | 4.096267 | 6.138156 | 4.233003 | 7.554585 | 10.234823 | 0.886413 |
| 0.75 | 1.661438 | 0.338562 | 1.916472 | 3.772303 | 3.403101 | 6.576726 | 3.158039 | 8.287778 | 9.982596 | 1.383388 |
| 0.9 | 1.435890 | 0.564110 | 1.557855 | 3.910268 | 2.844421 | 6.832319 | 2.320883 | 8.717352 | 9.685317 | 1.746751 |
| 0.99 | 1.141067 | 0.858933 | 1.167642 | 3.991103 | 2.251724 | 6.983323 | 1.454498 | 8.971861 | 9.259618 | 2.085984 |

(lengths in units of M). The bisected, integrated `b_c` for spins 0.5, 0.9
and 0.99 agree with this table to a relative error of order `1e-9`
(`kerrray validate kerr`, docs/validation.md section 3). Figures:
`spin_sweep_radii.png`, `spin_sweep_critical_b.png`,
`spin_sweep_shadows.png`.

## 5. Limitations

* Boyer-Lindquist coordinates: `phi` diverges at the horizon for `a != 0`
  and `theta` is singular on the axis, hence the escaped-only comparison and
  the polar-family bound on `b_p`.
* The azimuth accumulated beyond the launch radius `r0 = 1000 M` is not
  included in either the integration or the quadrature (they are compared
  over the same radial range, so the comparison is exact; the comparison with
  the infinite-line leading-order formulas is not).
* EXP-003 tabulates closed forms; the traced shadow for each spin comes from
  `kerrray shadow`.
