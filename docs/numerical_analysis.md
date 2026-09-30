# Numerical analysis: solvers, step sizes, near-critical rays, CPU performance and precision

This document describes the Phase 6 numerical studies and the Phase 8 CPU
performance and precision studies (docs/architecture.md section 9), and
records their **measured** results on the development machine. Every number
below was produced by the command quoted next to it; nothing is estimated or
copied from elsewhere. Run directories (`runs/<run_id>/manifest.json`,
`config.yaml`) and reports (`reports/<run_id>/report.md`, `summary.json`,
figures) hold the full data.

| Study | PROJECT.md | Code | Command |
|---|---|---|---|
| Solver comparison | section 19, EXP-006 | `benchmarks/solver.py`, `solver_config.py`, `rayset.py`, `solver_report.py` | `kerrray benchmark solver` |
| Step-size / tolerance study | section 20, EXP-006 | `experiments/step_size.py` | `kerrray experiment --config configs/convergence.yaml --name step_size` |
| Near-critical rays | section 21, EXP-007 | `experiments/near_critical.py` | `kerrray experiment --config configs/near_critical.yaml` |
| CPU performance | section 26 (as amended), EXP-009 | `benchmarks/cpu.py`, `memory.py` | `kerrray benchmark cpu` |
| float32 vs float64 | section 27, EXP-011 | `benchmarks/precision.py` | `kerrray benchmark precision` |
| GPU | section 26, EXP-010 | `benchmarks/gpu.py` (stub) | `kerrray benchmark gpu` |

## 1. Hardware and software of the measurements

Collected by `kerrray.utils.manifest.collect_environment` and stored in every
manifest:

* CPU: `Intel64 Family 6 Model 142 Stepping 12, GenuineIntel`, 8 logical CPUs, no CUDA device
* OS: `Windows-11-10.0.26200-SP0` (AMD64)
* Python 3.13.13 (CPython); numpy 2.5.3, scipy 1.18.1, matplotlib 3.11.2
* Numba 0.67.0 (installed in the virtual environment during this work; the
  solver comparison does not use it). When Numba cannot be imported the
  `numba` backend reports itself unavailable and every numba run is recorded
  as skipped with that reason (never silently omitted).
* Git: the commit of each run is listed with its table; all runs were made
  on a working tree with uncommitted Phase 6/8 files (`git_dirty: true` in
  the manifests).
* Other processes (parallel development agents running test suites) shared
  the machine during the measurements, so wall-clock times carry that noise;
  best-of-repeats timing reduces but does not remove it.

## 2. Benchmark ray set, reference and trajectory error

**Ray set** (`kerrray.benchmarks.rayset.build_ray_set`, seeded by
`experiment.seed`): ray `k` cycles through four kinds (`capture`, `escape`,
`near_critical_out`, `near_critical_in`), the spins `[0.0, 0.9]` and an
equatorial / off-equatorial family, so every block of eight rays covers all
categories. Every ray is built from exact constants of motion relative to the
*computed* critical curve, so its true fate does not depend on the launch
radius:

* equatorial rays have `b = f b_c` with `b_c` the computed prograde or
  retrograde critical impact parameter (`photons.orbits.critical_impact_parameters`);
* off-equatorial rays are launched at the observer inclination (60 deg) with
  `(alpha, beta) = f (alpha_c, beta_c)`, a random point of the analytic shadow
  curve scaled by `f`, and constants `xi = -alpha sin(theta_o)`,
  `eta = beta^2 - a^2 cos^2(theta_o) + xi^2 cot^2(theta_o)` (Bardeen 1973;
  docs/architecture.md section 1.3), which makes `Theta(theta_o) = beta^2 >= 0`
  exactly;
* `f` is uniform in `[0.2, 0.9]` (capture), `[1.1, 3.0]` (escape) or
  `1 +- 10^U(-4,-2)` (near-critical).

**Launch scale.** Rays start at, and escape at, `solver.launch_radius`
(default 50 M), not at the observer radius. WHAT: the benchmark geodesics
are the strong-field part of the observer's rays truncated at 50 M. WHY:
a fixed step `h` costs about `2 r_0 / h` steps per escaping ray; from 1000 M
only `h = 0.1` fits `integration.max_steps = 100000`, while the truncation
error is generated where the curvature is large (`r < 10 M`), which the
truncated rays contain in full (PROJECT.md section 20 allows "an equivalent
physically appropriate scale"). LIMITATION: the weak-field legs out to the
observer are not included; for a pure phase error the absolute position
error at 1000 M would be larger by up to the ratio of the radii.

**Reference**: SciPy `solve_ivp(method="DOP853")` with dense output at
`rtol = 1e-13`, `atol = 1e-15` and the same terminal events as the production
termination rules (capture at `r_+ + horizon_epsilon`, escape at the escape
radius with outward motion, the polar-angle domain).

**Trajectory error**: Euclidean distance (units of M) between the Cartesian
embeddings of the solver end state and the reference state *at the same
affine parameter*. Captured rays are compared at `r = r_+ + 0.1 M` (the
solver is re-run with that capture radius). WHAT: the last 0.1 M of the
plunge is excluded. WHY: in Boyer-Lindquist coordinates `phi` diverges
logarithmically at the horizon for `a != 0`, so a tiny radial error at
`r - r_+ = 1e-6` becomes a large azimuthal error that is a property of the
coordinates, not of the solver. LIMITATION: the plunge is still covered by
the null error, the drifts and the failure rate of the production run.

**Failure rate**: fraction of rays ending `NUMERICAL_FAILURE` or
`OUT_OF_DOMAIN`. **Mismatch**: rays whose termination state differs from the
reference's. Null error and drifts in the tables are maxima over the rays
that did not fail; "Null Error (escaped)" excludes captured rays, whose null
error is dominated by the `1/Delta` cancellation at the horizon
(docs/numerical_methods.md section 4).

## 3. Solver comparison (EXP-006, PROJECT.md section 19)

Command: `kerrray benchmark solver` (default `configs/convergence.yaml`:
40 rays, 3 repeats, step sizes `[0.1, 0.01, 0.001, 0.0001]`, tolerances
`[1e-6, 1e-8, 1e-10, 1e-12]`, `atol = rtol / 100`, `max_steps = 100000`,
`horizon_epsilon = 1e-6`, seed 0). Run `20260929T051346Z-f444b5`, git commit
`3aba8b45` (dirty tree), 695 s wall time in total. Reference: 20 rays
ESCAPED, 20 CAPTURED; 9712 accepted DOP853 steps in 46.3 s.

Skipped (logged and recorded in `summary.json`): `h = 0.0001` (about
1,000,000 steps per escaping ray exceed `max_steps = 100000`) and `h = 0.001`
(predicted 2750 s from the measured 275.0 s at `h = 0.01`, above
`solver.max_seconds_per_config = 600 s`).

| Solver | Setting | Runtime [s] | Steps/ray | Null Error | Null Error (escaped) | Energy Drift | Lz Drift | Carter Drift | Traj. Error max [M] | Traj. Error median [M] | Failure Rate | Mismatch |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RK4 | h = 0.1 | 5.828 | 793.4 | 63.48 | 63.48 | 0 | 0 | 54.22 | 50.05 | 0.0005477 | 0.5 | 20 |
| RK4 | h = 0.01 | 67.67 | 8076 | 0.0291 | 0.0291 | 0 | 0 | 0.0209 | 41.89 | 1.826e-07 | 0.5 | 20 |
| RK45 | rtol = 1e-06 | 2.326 | 80.05 | 99.73 | 9.686e-06 | 0 | 0 | 1.224e-05 | 0.2392 | 0.0001822 | 0 | 0 |
| RK45 | rtol = 1e-08 | 2.71 | 174.4 | 2.288 | 2.557e-07 | 0 | 0 | 2.393e-07 | 0.002356 | 2.676e-06 | 0 | 0 |
| RK45 | rtol = 1e-10 | 5.189 | 436.5 | 0.04014 | 3.507e-09 | 0 | 0 | 3.298e-09 | 2.392e-05 | 2.353e-08 | 0 | 0 |
| RK45 | rtol = 1e-12 | 10.09 | 1100 | 0.0372 | 3.911e-11 | 0 | 0 | 3.074e-10 | 1.875e-07 | 2.07e-10 | 0 | 0 |
| DOP853 | rtol = 1e-06 | 9.592 | 39.85 | 11.98 | 0.000517 | 0 | 0 | 0.0005113 | 0.3697 | 0.000121 | 0 | 0 |
| DOP853 | rtol = 1e-08 | 12.67 | 64.42 | 0.3257 | 3.868e-08 | 0 | 0 | 4.646e-08 | 0.001252 | 1.016e-06 | 0 | 0 |
| DOP853 | rtol = 1e-10 | 17.71 | 109.3 | 0.02592 | 1.613e-10 | 0 | 0 | 2.548e-10 | 1.417e-05 | 6.125e-09 | 0 | 0 |
| DOP853 | rtol = 1e-12 | 23.89 | 181 | 0.02087 | 1.259e-12 | 0 | 0 | 1.707e-10 | 1.102e-07 | 4.005e-11 | 0 | 0 |

Runtime is the best of 3 repeats over all 40 rays; `rk4` and `rk45` run
through the batched integrator (one batch per spin), `dop853` through SciPy
one ray at a time, so runtimes compare within a path and only indicatively
across paths. Figures: `reports/20260929T051346Z-f444b5/runtime_vs_error.png`,
`runtime_vs_null_error.png`.

Observations (all from the table and `summary.json`):

* **Energy and L_z drift are exactly 0** for every solver: in the Hamiltonian
  form `dp_t/dlambda = dp_phi/dlambda = 0` identically, so these columns test
  the implementation, not the accuracy. The Carter drift and the null error
  carry the truncation error.
* **Fixed-step RK4 fails every plunge.** All 20 captured rays end
  `OUT_OF_DOMAIN` at both step sizes (failure rate 0.5, 20 mismatches): the
  last fixed step overshoots into `Delta <= 0`, as documented in
  docs/numerical_methods.md section 4. Shadow extraction with `rk4` must
  treat these rays as non-escaping.
* **RK4 converges at order 4 on the typical ray.** Over the 34 rays compared
  at both step sizes, the median trajectory error falls from 5.477e-04 M to
  5.345e-08 M between `h = 0.1` and `h = 0.01`: fitted order **4.01**
  (`validation.convergence.loglog_order`). The maximum is dominated by one
  ray (index 30: a = 0.9, off-equatorial, `f = 1.0011`, `L_z = 0.041 M`,
  passing within `sin(theta) = 0.0083` of the spin axis): its error is 50.0 M
  at `h = 0.1` and still 41.9 M at `h = 0.01`, because `dphi/dlambda ~
  L_z / sin^2(theta)` near the axis is not resolved by a fixed step and the
  near-critical orbit amplifies the error; at `h = 0.001` a separate check
  gave the correct end polar angle (1.1064 rad against 1.1054 rad for RK45 at
  rtol 1e-12, both escaping near r = 50 M), versus 0.283 rad at `h = 0.01`.
* **Adaptive errors are proportional to rtol.** Over all 40 rays, the
  maximum trajectory error of RK45 scales as `rtol^1.016` (successive orders
  1.003, 0.997, 1.053) and its median as `rtol^0.994`; DOP853 gives
  `rtol^1.076` (maximum) and `rtol^1.083` (median). The escaped-ray null
  error follows the tolerance as well (RK45 exponent 0.90, DOP853 1.41).
* **Cost versus accuracy.** At equal accuracy the adaptive schemes are far
  cheaper than RK4: RK45 at rtol 1e-10 reaches a median error of 2.4e-08 M in
  5.2 s with 437 steps per ray, while RK4 at `h = 0.01` needs 8076 steps per
  ray and 67.7 s for a median of 1.8e-07 M (and still fails every plunge).
  DOP853 takes 2 to 6 times fewer steps than RK45 at the same rtol and is
  more accurate per rtol, but its scalar SciPy path is slower in wall time
  here.
* **Near-critical rays set the maximum error** of every adaptive setting:
  ray 7 (a = 0.9, off-equatorial, retrograde, `f = 0.999886`, captured) for
  all four RK45 tolerances and ray 19 (a = 0.9, equatorial, retrograde,
  `f = 0.999699`, captured) for all four DOP853 tolerances. The error of a
  ray orbiting near the unstable photon orbit grows exponentially with the
  number of turns (section 5), which is why the maxima are 880 to 3055 times
  the medians.

## 4. Step-size and tolerance study (EXP-006, PROJECT.md section 20)

`kerrray.experiments.step_size.run` uses the same ray set, launch radius,
reference, skip rules and per-configuration measurements as the solver
comparison, restricted to RK4 (step sizes) and RK45 (tolerances), and fits
the convergence order of the maximum and the median trajectory error over the
*common* rays (rays compared at every setting of that solver), plus the null
error and Carter drift, with `kerrray.validation.convergence.loglog_order`
and `successive_orders`. The orders quoted in section 3 were computed with
exactly that estimator from the per-ray errors of the solver run above;
a dedicated step-size run repeats the same computations without the DOP853
rows.

## 5. Near-critical rays (EXP-007, PROJECT.md section 21)

**Design.** Equatorial photons are launched inward from
`near_critical.launch_radius` with `b = b_c (1 +- delta)` for every
configured relative offset `delta` (`configs/near_critical.yaml`:
`1e-1 ... 1e-8`). Families: Schwarzschild, with `b_c` computed by bisection
on the integrator's own capture/escape outcome in the generic bracket
`[3, 8] M` to `bisection_tol` (the orbit-equation value is reported next to it
for comparison only), and Kerr prograde and retrograde for every non-zero
spin, with `b_c` from `photons.orbits.critical_impact_parameters`. Every ray
is integrated at every tolerance of `near_critical.tolerances` (default
`[1e-8, 1e-10, 1e-12]`), recording the outcome, the orbital turns, the affine
length, the steps, the conservation diagnostics and the runtime.

**Turns** are `|Delta phi| / (2 pi)` up to the last point with
`r >= r_+ + 0.1 M` (WHAT: the plunge is excluded; WHY: Boyer-Lindquist `phi`
diverges at the horizon for `a != 0`; LIMITATION: captured-ray turns count
the photon-orbit passages only; `turns_total` keeps the full `Delta phi`).

**Sensitivity.** For every family, branch (`+delta` escaping, `-delta`
captured) and tolerance, the least-squares slope of turns against
`log10(delta)` is fitted over the offsets with the expected outcome
(`turns_per_decade` is minus that slope). For every `(family, delta, sign)`
a *flip* is recorded when the outcome differs between tolerances, and the
smallest offset above which every outcome is the expected one is reported.

**Comparison value (Schwarzschild).** Linearising the orbit equation
`(du/dphi)^2 = 1/b^2 - u^2 + 2 M u^3` (u = 1/r) about the photon orbit
`u_c = 1/(3M)`: with `x = u - u_c`, `u^2 - 2Mu^3 = 1/(27 M^2) - x^2 + O(x^3)`,
so `(du/dphi)^2 ~ x^2 + epsilon` with `epsilon = 1/b^2 - 1/b_c^2 ~ -2 delta /
b_c^2`. For `delta > 0` (escaping) the passage through `|x| < X` costs
`2 arccosh(X / sqrt|epsilon|) = ln(1/delta) + O(1)` radians; for `delta < 0`
(captured) it costs `2 arcsinh(X / sqrt(epsilon)) = ln(1/|delta|) + O(1)`.
Both branches therefore wind `ln(10) / (2 pi) = 0.3665` turns per decade of
`delta` at leading order. The escaping-branch value agrees with the strong
deflection limit of Bozza 2002 (Phys. Rev. D 66, 103001; coefficient
`a_bar = 1` for Schwarzschild); the captured-branch statement is our own
linearisation, not a literature value. The experiment reports the fitted
slopes; this value is only used to discuss them. At test scale (offsets
`1e-1, 1e-2, 1e-3` from 30 M, rtol 1e-8) the fitted slopes are within 25 %
of it (asserted in `tests/test_near_critical.py`).

**Measured** (command `kerrray experiment --config configs/near_critical.yaml`;
run `20260929T061117Z-c0e234`, git commit `a9c864d5` (dirty tree), 324 s;
defaults: spins `[0.0, 0.9]`, tolerances `[1e-8, 1e-10, 1e-12]`, offsets
`1e-1 ... 1e-8`, launch and escape radius 1000 M, `rk45`, `atol = rtol / 100`,
bisection tolerance 1e-10 M at the configured rtol 1e-12; 144 rays):

* Schwarzschild `b_c` by bisection: **5.196152422729938 M** (36 iterations,
  half-width 3.6e-11 M); the orbit-equation value is 5.196152422706632 M
  (relative difference 4.5e-12), and `3 sqrt(3) M = 5.196152422706632 M`.
  Kerr a = 0.9: prograde `b_c = 2.8444214034761637 M`, retrograde
  `6.832319230446666 M` (orbit equations).
* **Every one of the 144 rays has the expected outcome and there is no
  classification flip between tolerances**, down to `delta = 1e-8` at
  rtol 1e-8: the smallest reliable offset is 1e-8 (the smallest tested) for
  every family, branch and tolerance.
* Fitted turns per decade of `delta` (minus the slope of turns against
  `log10(delta)`, 8 points each):

| family | branch | rtol 1e-8 | rtol 1e-10 | rtol 1e-12 |
|---|---|---:|---:|---:|
| Schwarzschild | escaping (`+delta`) | 0.3655 | 0.3639 | 0.3639 |
| Schwarzschild | captured (`-delta`) | 0.3674 | 0.3687 | 0.3687 |
| Kerr a = 0.9 prograde | escaping | 0.9385 | 0.9398 | 0.9398 |
| Kerr a = 0.9 prograde | captured | 0.8512 | 0.8503 | 0.8525 |
| Kerr a = 0.9 retrograde | escaping | 0.2876 | 0.2853 | 0.2853 |
| Kerr a = 0.9 retrograde | captured | 0.2883 | 0.2899 | 0.2901 |

  The Schwarzschild slopes are within 0.7 % (escaping, 0.3639) and 0.6 %
  (captured, 0.3687) of the comparison value 0.3665 at rtol 1e-12. For Kerr
  no comparison value is claimed here; the measured slopes show that a
  prograde a = 0.9 photon winds about 2.6 times as much per decade as a
  Schwarzschild one and a retrograde one about 0.78 times as much.
* Schwarzschild escaping branch at rtol 1e-12: turns grow from 0.829
  (`delta = 1e-1`) to 3.366 (`1e-8`); affine length 2034 to 2071 M; accepted
  steps 666 to 1313; maximum null error 1.30e-12 to 1.64e-12. The turn count
  depends on the tolerance by at most 0.019 turns at `delta = 1e-8` on this
  branch (3.385 at rtol 1e-8 against 3.366 at 1e-12), the largest spread being
  on the Kerr prograde captured branch (6.540, 6.515 and 6.527 at
  `delta = 1e-8`).
* Figures: `reports/20260929T061117Z-c0e234/turns_vs_offset.png`,
  `affine_length_vs_offset.png`, `turns_by_tolerance.png`, `trajectory.png`.

## 6. CPU performance (EXP-009; PROJECT.md section 26 as amended by D-007)

**Workload.** `N` backward rays at seeded uniform random image-plane points
`(alpha, beta)` in `[-12, 12]^2` M of the configured camera
(`configs/benchmark.yaml`: a = 0.9, observer at 1000 M, inclination 60 deg),
built with the same ZAMO-tetrad construction as the camera (agreement with
`raytracing.camera.initial_states` to 1e-12 is tested); the sets are nested
(the rays of a smaller `N` are the first rays of a larger one). Integrator:
`rk45`, rtol 1e-9, atol 1e-11, float64, `horizon_epsilon = 1e-6`, escape at
1000 M. Wall time is the best of 3 repeats. Peak memory is the `tracemalloc`
peak of a separate (untimed) run; it traces NumPy buffers but not
allocations inside Numba-compiled code, so for `numba` it covers the wrapper
arrays only. "Array estimate" is `N x 8 x itemsize x working arrays` plus
per-ray scalars (`benchmarks/memory.py`: 24 working arrays for numpy, 2 for
numba). Trajectory error: the first 200 rays integrated to `lambda = 2000 M`
by each backend at the benchmark tolerance, against a numpy float64
rtol 1e-12 reference, maximum over the 8 state components of
`|y - y_ref| / max(1, |y_ref|)`, on the 161 rays that reached `lambda = 2000 M`
in both runs (39 are captured before and excluded).

Command: `kerrray benchmark cpu --set "experiment.parameters.benchmark.ray_counts=[1000,10000,100000,1000000]"
--set experiment.parameters.benchmark.max_seconds_per_run=2000`. Run
`20260929T052912Z-f23a49`, git commit `a55ea1cb` (dirty tree), 2367 s in
total. Numba 0.67.0 became importable in the virtual environment while this
run was in progress and was used for the `numba` rows (compile/warm-up time
0.634 s, recorded as `compile_time_s`).

| backend | rays | status | wall_time_s | rays_per_s | peak_traced_MB | array_estimate_MB | max_null_error | max_null_error_escaped | max_energy_drift | max_carter_drift | trajectory_error_max | captured | escaped | other |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| numpy | 1,000 | ok | 2.463 | 406 | 1.506 | 1.656 | 0.08465 | 8.893e-09 | 0 | 2.216e-08 | 3.987e-08 | 135 | 865 | 0 |
| numpy | 10,000 | ok | 12.36 | 808.8 | 14.95 | 16.56 | 0.1215 | 2.016e-08 | 0 | 2.837e-08 | 3.987e-08 | 1,325 | 8,675 | 0 |
| numpy | 100,000 | ok | 172.2 | 580.8 | 149.4 | 165.6 | 0.1228 | 2.92e-08 | 0 | 4.411e-08 | 3.987e-08 | 13,555 | 86,445 | 0 |
| numpy | 1,000,000 | skipped |  |  |  |  |  |  |  |  |  |  |  |  |
| numba | 1,000 | ok | 0.1119 | 8937 | 0.3338 | 0.248 | 0.08465 | 8.893e-09 | 0 | 2.216e-08 | 3.987e-08 | 135 | 865 | 0 |
| numba | 10,000 | ok | 1.185 | 8436 | 3.284 | 2.48 | 0.1215 | 2.016e-08 | 0 | 2.837e-08 | 3.987e-08 | 1,325 | 8,675 | 0 |
| numba | 100,000 | ok | 11.68 | 8561 | 32.8 | 24.8 | 0.1228 | 2.92e-08 | 0 | 4.411e-08 | 3.987e-08 | 13,555 | 86,445 | 0 |
| numba | 1,000,000 | ok | 125.8 | 7949 | 328 | 248 | 0.1313 | 5.45e-08 | 0 | 4.894e-08 | 3.987e-08 | 136,488 | 863,512 | 0 |

Skipped (logged in the run and listed in `summary.json` and the report):
numpy, 1,000,000 rays: "predicted 6886.5 s exceeds max_seconds_per_run =
2000 s at 580.8 rays/s" (prediction = (repeats + 1) x N / measured
throughput of the previous count).

Observations:

* **Scaling.** Fitted log-log exponent of wall time against `N`: numpy
  **0.922** (1e3 to 1e5), numba **1.015** (1e3 to 1e6). The numba backend is
  linear in `N` over three decades (7949 to 8937 rays/s). The numpy backend
  amortises its per-iteration Python overhead between 1e3 and 1e4 rays
  (406 to 809 rays/s) and then loses throughput at 1e5 (581 rays/s), where
  its working set (149 MB traced) no longer fits the caches; the machine was
  shared with other processes, which also contributes (the three 1e5
  repeats took 206.5, 172.2 and 174.8 s).
* **Speed-up of numba over numpy** (measured wall times): 22.0x at 1e3,
  10.4x at 1e4, 14.7x at 1e5 rays, on 8 logical CPUs.
* **Identical physics.** Both backends give the same termination counts,
  the same maximum null error, Carter drift and trajectory error at every
  `N` (docs/performance.md documents the agreement test). Energy and L_z
  drift are exactly 0 (Hamiltonian form).
* **Accuracy at the default tolerance.** Escaped-ray null error stays below
  5.5e-08 up to 1e6 rays; the fixed-lambda trajectory error of the 161
  compared rays is 3.99e-08 (maximum) and 2.03e-09 (median) against the
  rtol 1e-12 reference. The all-ray maximum null error (0.08 to 0.13) comes
  from captured rays at the horizon (`1/Delta` cancellation,
  docs/numerical_methods.md section 4).
* **Memory.** numpy: 149 MB traced peak at 1e5 rays (1.49 kB per ray),
  consistent with the 24-working-array estimate (165.6 MB). numba: 328 MB at
  1e6 rays of wrapper-level arrays (0.33 kB per ray; kernel-internal
  allocations are not traced).
* About 13.6 % of the random image-plane points in `[-12, 12]^2` M are
  captured (135 / 1000, 13,555 / 100,000, 136,488 / 1,000,000).

## 7. float32 versus float64 (EXP-011, PROJECT.md section 27)

**Method.** The configured camera image (64 x 64 = 4096 rays, a = 0.9,
inclination 60 deg, observer at 1000 M, field of view +-12 M, pixel 0.375 M)
is traced with `rk45` on both CPU backends at rtol 1e-5, 1e-6 and 1e-7
(`atol = rtol / 100`) in float32 and float64 at the same tolerance, so that
a difference is due to the arithmetic. **float32 horizon margin:** float32
cannot resolve `horizon_epsilon = 1e-6` next to `r_+ = 1.436 M` (about 17
float32 ulps at r ~ 1; docs/numerical_methods.md section 7), so float32 runs
use `horizon_epsilon_float32 = 1e-3`. A float64 **control** run at the same
1e-3 margin isolates the precision effect (comparison "float64_control vs
float32"), and "float64 vs float64_control" measures the effect of the
margin alone. Wall time is the best of 3 traces of the whole image.

Command: `kerrray benchmark precision` (default `configs/benchmark.yaml`,
64 x 64). Run `20260929T060841Z-f365e3`, git commit `a9c864d5` (dirty tree),
128 s in total.

| backend | dtype | role | rtol | horizon_epsilon | runtime_s | max_null_error_escaped | median_null_error | max_carter_drift_escaped | max_null_error | captured | escaped | other |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| numpy | float32 | main | 1e-05 | 0.001 | 1.353 | 7.871e-05 | 1.239e-05 | 0.0001148 | 5.468 | 558 | 3,538 | 0 |
| numpy | float64 | main | 1e-05 | 1e-06 | 2.586 | 7.992e-05 | 1.24e-05 | 0.000101 | 283.4 | 558 | 3,538 | 0 |
| numpy | float64 | control | 1e-05 | 0.001 | 2.607 | 7.992e-05 | 1.24e-05 | 0.000101 | 0.2668 | 558 | 3,538 | 0 |
| numpy | float32 | main | 1e-06 | 0.001 | 2.412 | 4.342e-05 | 1.268e-06 | 6.751e-05 | 4.585 | 558 | 3,538 | 0 |
| numpy | float64 | main | 1e-06 | 1e-06 | 2.43 | 4.649e-06 | 1.213e-06 | 1.503e-05 | 69.78 | 558 | 3,538 | 0 |
| numpy | float64 | control | 1e-06 | 0.001 | 3.429 | 4.649e-06 | 1.213e-06 | 1.503e-05 | 0.05728 | 558 | 3,538 | 0 |
| numpy | float32 | main | 1e-07 | 0.001 | 8.882 | 8.225e-05 | 5.049e-07 | 0.0001261 | 17.69 | 558 | 3,538 | 0 |
| numpy | float64 | main | 1e-07 | 1e-06 | 4.056 | 6.992e-07 | 1.395e-07 | 1.583e-06 | 8.752 | 558 | 3,538 | 0 |
| numpy | float64 | control | 1e-07 | 0.001 | 4.386 | 6.992e-07 | 1.395e-07 | 1.583e-06 | 0.00706 | 558 | 3,538 | 0 |
| numba | float32 | main | 1e-05 | 0.001 | 0.1827 | 7.871e-05 | 1.239e-05 | 0.0001148 | 5.468 | 558 | 3,538 | 0 |
| numba | float64 | main | 1e-05 | 1e-06 | 0.1717 | 7.992e-05 | 1.24e-05 | 0.000101 | 283.4 | 558 | 3,538 | 0 |
| numba | float64 | control | 1e-05 | 0.001 | 0.1583 | 7.992e-05 | 1.24e-05 | 0.000101 | 0.2668 | 558 | 3,538 | 0 |
| numba | float32 | main | 1e-06 | 0.001 | 0.203 | 4.342e-05 | 1.268e-06 | 6.751e-05 | 4.585 | 558 | 3,538 | 0 |
| numba | float64 | main | 1e-06 | 1e-06 | 0.31 | 4.649e-06 | 1.213e-06 | 1.503e-05 | 69.78 | 558 | 3,538 | 0 |
| numba | float64 | control | 1e-06 | 0.001 | 0.2368 | 4.649e-06 | 1.213e-06 | 1.503e-05 | 0.05728 | 558 | 3,538 | 0 |
| numba | float32 | main | 1e-07 | 0.001 | 0.3083 | 8.225e-05 | 5.049e-07 | 0.0001261 | 17.69 | 558 | 3,538 | 0 |
| numba | float64 | main | 1e-07 | 1e-06 | 0.357 | 6.992e-07 | 1.395e-07 | 1.583e-06 | 8.752 | 558 | 3,538 | 0 |
| numba | float64 | control | 1e-07 | 0.001 | 0.2815 | 6.992e-07 | 1.395e-07 | 1.583e-06 | 0.00706 | 558 | 3,538 | 0 |

Comparisons (every row of both backends and all three tolerances gave the
same values): 0 differing pixels, 0 rays with a different termination state,
mean boundary-radius difference 0 M, equivalent-radius difference 0 M and
sub-pixel boundary difference 0 M, for both "float64_control vs float32"
and "float64 vs float64_control". The captured mask has 558 pixels
(equivalent radius 4.998 M, mean boundary-pixel radius 4.825 M about the
float64 centroid). Runtime ratio float32 / float64 (control): numpy 0.519,
0.704, 2.025 and numba 1.154, 0.858, 1.095 at rtol 1e-5, 1e-6, 1e-7.

Observations:

* **The 64 x 64 shadow is insensitive to the precision and to the horizon
  margin**: identical masks and identical per-ray outcomes at every
  tolerance on both backends. At this resolution a precision effect would
  have to move the boundary by a sizeable fraction of a 0.375 M pixel to
  show; it does not.
* **float32 saturates the accuracy near 1e-5 to 1e-4.** The escaped-ray null
  error of float64 follows the tolerance (7.99e-05, 4.65e-06, 6.99e-07 at
  rtol 1e-5, 1e-6, 1e-7) while float32 stays at 7.9e-05, 4.3e-05, 8.2e-05; the
  escaped-ray Carter drift behaves the same way (float64 1.0e-04, 1.5e-05,
  1.6e-06; float32 1.1e-04, 6.8e-05, 1.3e-04). The two precisions are
  indistinguishable at rtol 1e-5 and diverge below it, consistent with the
  crossover near rtol 1e-6 reported in docs/numerical_methods.md section 7.
  Median null errors stay close (1.24e-05 vs 1.24e-05, 1.27e-06 vs 1.21e-06,
  5.0e-07 vs 1.4e-07), so the saturation affects the worst rays first.
* **float32 is not faster on the CPU at useful tolerances.** numpy float32
  is faster at rtol 1e-5 (ratio 0.52) and 1e-6 (0.70) but twice as slow at
  1e-7 (2.03), where the tolerance approaches float32 round-off and the
  step control works harder; numba shows no systematic gain (0.86 to 1.15).
* **The horizon margin, not the precision, sets the captured-ray null
  error**: float64 at `horizon_epsilon = 1e-6` has an all-ray maximum of
  283.4 / 69.8 / 8.75 against 0.267 / 0.057 / 0.0071 for the float64 control
  at 1e-3, with the same shadow. This supports the recommendation of
  docs/numerical_methods.md section 8 to consider a larger `horizon_epsilon`
  for shadow runs.
* Both backends give identical diagnostics and masks in both precisions.

## 8. GPU (EXP-010)

`kerrray benchmark gpu` probes `numba.cuda.is_available()` inside a guarded
import and records `gpu_available: false` with the message "GPU acceleration
is optional future work (docs/decisions.md D-007); no CUDA device available;
no GPU numbers are reported". On this machine (run
`20260929T061225Z-7ed426`) the probe reports `numba.cuda.is_available()
returned False`: there is no CUDA device. No runtime, speed-up or memory
figure is produced.

## 9. Reproducing the measurements

```text
kerrray benchmark solver                                   # section 3 (about 12 min here)
kerrray experiment --config configs/convergence.yaml --name step_size
kerrray experiment --config configs/near_critical.yaml     # section 5 (about 5 min here)
kerrray benchmark cpu --set "experiment.parameters.benchmark.ray_counts=[1000,10000,100000,1000000]" \
                      --set experiment.parameters.benchmark.max_seconds_per_run=2000
kerrray benchmark precision                                # 64 x 64, section 7
kerrray benchmark gpu
```

Every run writes `runs/<run_id>/manifest.json` (configuration, seed,
environment, git commit, results) and `reports/<run_id>/report.md` with the
nine PROJECT.md section 36 sections, `summary.json` and the figures. Skipped
settings are always logged and listed under `skipped` in `summary.json` and
in the report.
