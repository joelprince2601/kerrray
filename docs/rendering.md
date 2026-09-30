# Redshift and the simplified accretion-disk renderer

PROJECT.md sections 23, 24 and 43. Implementation:
`src/kerrray/physics/redshift.py`, `src/kerrray/physics/accretion.py`,
`src/kerrray/raytracing/renderer.py`, `src/kerrray/reporting/figures.py`
(`disk_figure`), command `kerrray render` (`src/kerrray/commands/render.py`).
Conventions: G = c = 1, lengths in units of M, Boyer-Lindquist coordinates,
signature (-, +, +, +), covariant photon momentum `p_mu`, and `prograde`
meaning `a L_z > 0` (docs/architecture.md section 1).

## 1. Pipeline

```
camera pixel -> initial_states (backward, null)
  -> trace_rays with EventOptions(disk_plane=True, r_in, r_out)
  -> DISK_HIT: interpolated theta = pi/2 crossing state
  -> g = redshift_factor(st, y_cross)
  -> I_obs = g^n I_em(r_cross)
```

Every other pixel gets `I_obs = 0` and `g = NaN`.

## 2. Redshift factor g = nu_obs / nu_em

**Definition.** An observer with four-velocity `u^mu` measures the photon
frequency `nu = -p_mu u^mu`. Hence

```
g = nu_obs / nu_em = (-p_mu u^mu_obs) / (-p_mu u^mu_em)
```

(Cunningham 1975, ApJ 202, 788; Luminet 1979, A&A 75, 228).

* **Observer at infinity** (static): `u_obs = (1, 0, 0, 0)` and
  `nu_obs = E = -p_t`.
* **Emitter:** gas on a circular equatorial Keplerian orbit,
  `u^mu = u^t (1, 0, 0, Omega)`.
  * `Omega = +- M^{1/2} / (r^{3/2} +- a M^{1/2})`, upper sign prograde
    (Bardeen, Press & Teukolsky 1972, ApJ 178, 347, eq. 2.16;
    `photons.orbits.keplerian_angular_velocity`).
  * `u^t = 1 / sqrt(-(g_tt + 2 Omega g_tphi + Omega^2 g_phph))` at
    `theta = pi/2`, from `u.u = -1`.
* **Closed form.** Since `-p_mu u^mu_em = u^t (E - Omega L_z)`,

```
g = 1 / [u^t (1 - Omega L_z / E)].
```

**Implementation mapping.** `redshift_factor(st, y_hit, prograde=...)`
computes the direct contraction `(-p_t) / (-p_mu u^mu)` with
`emitter_four_velocity`. The tests compare it with the closed form above.
The largest measured relative difference is `3.3e-16`, over 384 random
equatorial states for `a = 0, 0.9, -0.5` and both senses; the test threshold
is `1e-12`.

**Checks.**

* **Schwarzschild:** `u^t = (1 - 3M/r)^{-1/2}`, and a photon with
  `L_z = 0` has `g = sqrt(1 - 3M/r)` (tested to `1e-14`).
* **Reversal invariance:** the backward tracer stores the reversed momentum
  (`p -> -p`, docs/raytracing.md section 6). Numerator and denominator both
  change sign, so `g` is unchanged. The tests require bit-equality.
* **Validity:** a non-positive `g` (negative photon energy in the emitter
  frame) raises `ValueError`. So does a radius at or below the circular
  photon orbit, where no timelike circular orbit exists.
* **Static emitter:** `gravitational_redshift_factor` gives
  `g = sqrt(-g_tt)` (outside the ergosphere). It tends to `1 - M/r` at large
  `r` (tested at `r = 1e6 M`).

## 3. The simplified disk model (approximation)

`physics.accretion.DiskModel(r_in, r_out, emissivity_index, intensity_law, prograde)`:

* `r_in` defaults to the ISCO (BPT 1972 eq. 2.21, `photons.orbits.isco_radius`,
  verified numerically against the marginal-stability condition in the
  orbits tests). It must exceed the circular photon orbit radius.
* Emissivity: `I_em(r) = (r / r_in)^(-p)` on `r_in <= r <= r_out` and 0
  elsewhere. It is normalised to 1 at `r_in` and is in arbitrary units.
* Observed intensity: `I_obs = g^3 I_em` (`intensity_law: g3`) or
  `I_obs = g^4 I_em` (`g4`, default).
  * `I_nu / nu^3` is invariant along a ray (Liouville; Misner, Thorne &
    Wheeler 1973, chapter 22). With a frequency-independent `I_em` this
    gives `g^3` for the specific intensity at a fixed observed frequency.
  * The frequency-integrated intensity picks up one more `g` from
    `d nu_obs = g d nu_em`, giving `g^4` (Luminet 1979; Cunningham 1975).

Approximation statement (PROJECT.md section 43):

* **WHAT:** simplified optically thin, geometrically thin, single-surface
  emission in the equatorial plane, with a power-law emissivity in the gas
  rest frame. The gas moves on circular Keplerian orbits.
* **WHY:** to isolate geodesic propagation (lensing, gravitational and
  Doppler shift) from radiative physics. PROJECT.md section 23 calls this a
  visualisation/physics extension, not a full GRMHD simulation.
* **LIMITATION:** it is not equivalent to GRMHD or radiative-transfer
  modelling. It has no absorption or emission along the ray and no
  returning radiation: only the first crossing of the disk annulus counts,
  so the disk is opaque. There is no thickness, no vertical structure, no
  spectrum and no time dependence. The power-law emissivity is a modelling
  choice, not derived from physics. Gas inside the ISCO is not modelled
  unless `r_in` is set lower explicitly.

## 4. Renderer

`raytracing.renderer.render_disk(st, cam, integ, term, disk, backend=..., progress=...)`
runs four steps:

1. `initial_states(cam, st)` builds the backward, null camera rays.
2. `integrate_rays` (`rays.trace_rays`) runs them in chunks with
   `EventOptions(disk_plane=True, r_in, r_out)` from
   `accretion.event_options(disk)`. The `numpy` and `numba` backends are
   supported, and `cuda` raises `BackendUnavailable` (D-007).
3. On `DISK_HIT` pixels it computes `g` and `I_obs` at the crossing state.
4. It returns a `DiskImage`: `intensity`, `g`, `hit`, `r_hit`, `state`,
   `alpha`, `beta`, the crossing states (reversed momentum), the full
   `BatchResult`, the inputs and the backend name.

`save_disk_image` and `load_disk_image_arrays` round-trip everything through
`.npz`.

**Disk-plane event (approximation, from `geodesics.integrators_batch`).**

* **WHAT:** a `theta = pi/2` crossing between two accepted steps is located
  by linear interpolation in `lambda`. If the interpolated radius lies in
  `[r_in, r_out]` the ray stops as `DISK_HIT`.
* **WHY:** the renderer only needs the crossing state.
* **LIMITATION:** `O(h^2)` error in the crossing point, and the interpolated
  state is not re-projected onto the null cone. Because `p_t` and `p_phi`
  are exact constants of the Hamiltonian integration and `u^r = u^theta = 0`
  for the emitter, `g` depends only on `(p_t, p_phi, r_cross)`, so only the
  radius interpolation error enters it. The measured null error of the
  disk rays in section 6 is `5.0e-9`.

## 5. Image orientation and the Doppler side

The camera sets `alpha = -r_o p^(phi) / p^(t)` (docs/raytracing.md
section 3). Take an observer at `phi = 0` and gas moving towards `+phi`
(prograde with `a >= 0`):

* The gas at `phi = -pi/2` moves towards the observer.
* A photon from there arrives moving in `+phi` at the observer
  (`p^(phi) > 0`), so it lands at `alpha < 0`.

**The approaching, blue-shifted side is therefore at `alpha < 0`.** It is
at `alpha > 0` when the gas moves towards `-phi`: prograde with `a < 0`, or
retrograde with `a > 0`. The render report does not assert the effect. It
states the computed mean `g` on each side and whether they agree with this
expectation (`commands.render._interpretation`). The tests check it for
`(a, prograde) = (0, True)` and `(0.9, False)`.

Figures (`reporting.figures.disk_figure`):

* the intensity with a linear stretch and with a log stretch floored at
  `1e-4` of the maximum (single-hue sequential map);
* the redshift map with the diverging `RdBu` map centred on `g = 1`: blue
  for `g > 1` and red for `g < 1`, which is regression-tested.

## 6. `kerrray render` and a measured run

```
kerrray render [--spin] [--inclination] [--resolution] [--fov] [--backend]
               [--r-out] [--r-in] [--emissivity-index] [--intensity-law] [--config]
```

The default config is `configs/shadow.yaml` (resolved from the working
directory, then the repository root). Disk settings live in the `disk`
sub-block of `experiment.parameters`. A warning is logged when
`raytrace.fov < r_out`, because the image edge would cut off the outer
disk. Output goes through `experiments.base.run_experiment` under the name
`render`:

* `runs/<id>/`: `manifest.json`, `config.yaml` and `disk_image.npz`;
* `reports/<id>/`: `disk_linear.png`, `disk_log.png`, `disk_redshift.png`,
  `report.md` (nine section 36 headings) and `summary.json`.

**Measured run** (2026-09-29, numpy backend, 8 logical CPUs; `rk45` with
rtol `1e-9`, atol `1e-11`, `r_o = 1000 M`, `fov = 25 M`, disk defaults):

```
kerrray render --spin 0.9 --inclination 75 --resolution 64 --fov 25
```

Disk defaults: `r_in = ISCO = 2.32088 M`, `r_out = 20 M`, `p = 3`, law
`g4`, prograde.

| quantity | value |
|---|---|
| pixels ESCAPED / CAPTURED / DISK_HIT | 3182 / 54 / 860 |
| g min / max / mean over disk pixels | 0.238877 / 1.34898 / 0.88578 |
| mean g, approaching side (alpha < 0) / receding side (alpha > 0) | 1.0634 / 0.7162 |
| r_hit min / max | 2.34755 / 19.9911 M |
| I_obs max (I_em(r_in) = 1) | 2.18889 |
| max null error, disk rays / all rays | 5.0e-9 / 6.6e-2 (captured rays near the horizon) |
| max energy drift | 0 (p_t is exactly constant in the Hamiltonian form) |
| runtime (experiment, including figures and report) | 12.58 s (10.66 s in an earlier identical run) |

The lensed image of the far side of the disk arches over the shadow, and a
secondary image appears below it. The approaching side is boosted to
`g = 1.35`, and the smallest `g` lies at the inner edge on the receding
side.

## 7. Tests

* `tests/test_redshift.py`, `tests/test_accretion.py`: the relations of
  sections 2 and 3.
* `tests/test_renderer.py`:
  * Schwarzschild image properties, the approaching side is brighter, a
    retrograde disk flips the bright side, and face-on mirror symmetry up
    to the derived Doppler term;
  * `.npz` round trip, chunked rendering equal to a single chunk, and the
    numba kernel equal to numpy;
  * figures and colour-map direction;
  * `kerrray render` on a fresh Typer app, including `--fov`,
    `--backend cuda` (exit 1) and the computed interpretation.
