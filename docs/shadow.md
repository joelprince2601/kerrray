# Shadow reconstruction and boundary comparison

PROJECT.md sections 16 to 18. Implementation: `src/kerrray/raytracing/`
(`rays.py`, `shadow.py`, `boundary.py`), commands `kerrray trace` and
`kerrray shadow` (`src/kerrray/commands/trace.py`, `shadow.py`), experiment
driver `src/kerrray/experiments/shadow.py`. Conventions: G = c = 1, lengths
in units of M, Boyer-Lindquist coordinates, `a = spin * M`
(docs/architecture.md section 1).

## 1. The shadow is a classification result

The shadow is never drawn analytically (PROJECT.md section 17). For every
pixel of the camera image:

1. `raytracing.camera.initial_states(cam, st)` builds the past-directed,
   null-normalised photon state of the pixel (ZAMO tetrad; derivation in
   docs/raytracing.md sections 2 to 6);
2. `raytracing.rays.trace_rays` integrates all pixels in chunks of at most
   `DEFAULT_CHUNK_SIZE = 16384` rays on the selected backend (docs/performance.md);
3. the termination state of each ray (`photons.TerminationState`) is
   reshaped to the image (`Y.reshape(H, W, 8)`, row-major, row 0 at the top,
   `alpha` increasing to the right);
4. the **shadow mask** is `state == CAPTURED`; `ESCAPED` pixels are the
   background; `NUMERICAL_FAILURE` and `OUT_OF_DOMAIN` are counted as
   failures (`ShadowImage.n_failed`) and every other outcome as `n_other`.
   Nothing is reclassified.

`compute_shadow` returns a `ShadowImage` (pixel coordinates, state map,
mask, per-state counts, full `BatchResult`, camera and spacetime) that
round-trips through a compressed `.npz` file (`ShadowImage.save/load`,
format version 1).

Chunking does not change the result: every ray carries its own adaptive step
size and error control inside the batched integrator, so a chunked run is
bit-identical to an unchunked one (`tests/test_backends.py`,
`tests/test_shadow.py`).

## 2. Boundary extraction (sub-pixel)

`boundary.extract_boundary(img, n_angles=360)`:

* The 0/1 mask sampled at the pixel centres is interpolated **bilinearly**
  (Press et al., *Numerical Recipes*, 3rd ed., section 3.6), with value 0
  outside the outermost centres. The boundary is the 0.5 level set of the
  interpolant.
* From the centroid of the captured pixel centres (or a given centre) a ray
  is marched along each polar angle `phi_k = 2 pi k / n_angles` in steps of
  `STEP_FRACTION = 0.125` pixel. The first sample below 0.5 brackets the
  crossing, which is located by linear interpolation between the two
  bracketing samples.
* Output: `(angles, radii)`, radii in units of M from the centre.
* Error bound: the 0.5 level set of the bilinear interpolant of a 0/1 mask
  lies within half a pixel of the true edge
  (`tests/test_boundary.py::test_disk_mask_boundary_is_within_half_a_pixel_of_the_true_radius`).
* Refusals (`ValueError`): empty mask; mask touching the image border (the
  boundary would leave the image, so increase `raytrace.fov`); centre outside
  the captured region.

## 3. Analytic comparison curve (Bardeen 1973)

`boundary.analytic_boundary(st, inclination_deg)` returns
`photons.orbits.shadow_curve`, the `r_o -> infinity` curve of Bardeen (1973,
in *Black Holes*, Les Houches) built from the spherical photon orbits
(Johannsen & Psaltis 2010, ApJ 718, 446; Cunha & Herdeiro 2018, Gen. Rel.
Grav. 50, 42):

```
alpha = -xi(r) / sin(i),   beta = +- sqrt(eta(r) + a^2 cos^2(i) - xi(r)^2 cot^2(i))
```

with `xi = L_z / E`, `eta = Q / E^2` of the spherical orbit at radius `r`
(derived with SymPy from `R(r) = R'(r) = 0`; docs/derivations.md). For
`a = 0` it is the circle of radius `b_c` computed by
`photons.orbits.critical_impact_parameters`.

`boundary.boundary_error(numeric, analytic, centre)` measures the analytic
radius along the *same* polar angles about the *same* centre by exact
ray-segment intersection with the closed polygon (`polygon_radii`; requires
the curve to be star-shaped about the centre, true for the convex Kerr
shadow). It reports, in M and (with `pixel_size`) in pixels:

* `max`, `rms`, signed `mean` of `r_numeric - r_analytic`, `mean_abs`;
* for each curve: mean, min and max radius, asymmetry `max - min`, and area
  centroid (shoelace formula
  `A = 1/2 sum (x_i y_{i+1} - x_{i+1} y_i)`,
  `C_x = 1/(6A) sum (x_i + x_{i+1})(x_i y_{i+1} - x_{i+1} y_i)`, likewise `C_y`);
* the centroid shift numeric minus analytic.

`boundary.summarise_shadow(img)` runs the extraction and both comparisons
(raw and finite-distance corrected, section 4).

## 4. Finite observer distance (approximation)

* **WHAT.** The analytic curve is Bardeen's `r_o -> infinity` limit, while
  the image is recorded by a ZAMO at finite `r_o`. For a static observer in
  Schwarzschild the tetrad components give
  `p^(phi) / p^(t) = (L / r_o) / (E / sqrt(1 - 2M/r_o))`, so with the camera
  convention `alpha = -r_o p^(phi) / p^(t)` a ray of impact parameter `b`
  lands at `|alpha| = b sqrt(1 - 2M / r_o)`: the image is smaller by that
  factor, exactly for `a = 0` and to leading order `1 - M / r_o` for any
  spin (docs/raytracing.md section 5).
  `boundary.finite_distance_scale(st, r_o) = 1 / sqrt(1 - 2M / r_o)` undoes
  it; `boundary_error(..., scale=...)` applies it to the numeric radii and
  centre.
* **WHY.** Rays have to start at a finite radius.
* **LIMITATION.** Without the factor the comparison carries a systematic
  relative offset of about `M / r_o` (`1e-3` at `r_o = 1000 M`, i.e.
  `0.005 M` on a radius of `5.2 M`). At 64 x 64 pixels with `fov = 12 M`
  (pixel `0.375 M`) this is 1.3 % of a pixel, far below the resolution error
  (measured in section 5). The factor matters only for resolution studies
  that approach `M / r_o` accuracy.

## 5. Measured results (64 x 64)

Command: the `summarise_shadow` pipeline on `compute_shadow` output, numpy
backend, `rk45` with rtol `1e-9`, atol `1e-11`, `r_o = 1000 M`,
`i = 60 deg`, `fov = 12 M` (pixel size `0.375 M`), `horizon_epsilon = 1e-6`,
`escape_radius = 1000 M`, 8 logical CPUs, measured 2026-09-29.

| quantity | a = 0 | a = 0.9 |
|---|---|---|
| captured / escaped / other pixels | 608 / 3488 / 0 | 558 / 3538 / 0 |
| wall-clock runtime (numpy) | 7.39 s | 9.39 s |
| rms error vs analytic, raw | 0.0733 M = 0.195 px | 0.0872 M = 0.233 px |
| max error vs analytic, raw | 0.147 M = 0.391 px | 0.182 M = 0.484 px |
| signed mean error, raw | +0.0187 M | -0.0056 M |
| signed mean error, finite-distance corrected | +0.0239 M | -0.0006 M |
| mean radius numeric / analytic | 5.21482 / 5.19615 M | 4.99377 / 4.99942 M |
| asymmetry (max - min radius) numeric / analytic | 0.281 / 0 M | 0.609 / 0.357 M |
| centroid alpha numeric / analytic | 0.0000 / 0.0000 M | 1.6928 / 1.7008 M |
| centroid shift | 0 px | 0.021 px |
| max Carter-constant drift | 2.2e-8 | 2.3e-8 |
| max null-constraint error (all rays) | 9.3e-3 | 1.1e-1 |
| mean accepted steps per ray | 224 | 234 |

Reading the table:

* Every pixel settled as CAPTURED or ESCAPED; no ray failed.
* All boundary errors are below half a pixel, the bound of section 2. The
  numeric asymmetry of the `a = 0` circle (0.28 M) is pixelation: the
  0.5 level set of a bilinearly interpolated mask is a slightly wavy curve.
* The Kerr shadow is displaced to `alpha > 0` by 1.69 M. The numeric
  centroid matches the analytic one to 0.021 px.
* The largest null-constraint errors belong to captured rays near the
  horizon, where `p_r ~ 1/Delta` grows without bound in Boyer-Lindquist
  coordinates (docs/numerical_methods.md section 4). The `kerrray trace`
  panel reports the escaped-ray maximum separately.
* The `a = 0` mean radius is 0.0187 M = 0.05 px above `b_c`. The
  finite-distance factor moves the numeric curve outward by 0.005 M, which
  is below this resolution bias, so it does not reduce the error at 64 x 64.

The tests re-check the same properties at 16 to 24 pixels per side, with
tolerances in pixels (`tests/test_shadow.py`).

## 6. Commands and experiments

* `kerrray trace [--spin --inclination --resolution --fov --backend --method --rtol --atol --set key=value]`:
  prints the Spacetime, Observer and Ray Trace panels, shows a progress
  bar, and prints the Results panel: counts, the maximum null error over all
  rays and over escaped rays, the E, L_z and Carter drifts, the runtime and
  rays per second. It writes `runs/<id>/shadow.npz` with its manifest and
  config, plus `reports/<id>/shadow.png` and `summary.json`. The default
  config is `configs/shadow.yaml`. Relative to the working directory it
  falls back to the repository root.
* `kerrray shadow` does the same, then extracts the boundary, compares it
  with the Bardeen curve (raw and corrected), prints the Shadow boundary
  panel and writes `shadow_boundary.png` with the analytic curve overlaid.
* `kerrray.experiments.shadow.run(cfg)` dispatches on
  `experiment.parameters.mode`. `spin_sweep` is EXP-003,
  `inclination_sweep` is EXP-004 (0 deg is clamped to 1e-3 deg, D-008) and
  `convergence` is EXP-008 plus the tolerance series. The convergence mode
  compares every run with the analytic curve and with the finest run of its
  series, fits log-log slopes, and skips any run whose estimated duration
  exceeds `max_seconds_per_run`. Skips are logged and listed in the report,
  never dropped silently.

## 7. Tests

`tests/test_shadow.py`, `tests/test_boundary.py`, `tests/test_backends.py`,
`tests/test_commands_shadow.py`.
