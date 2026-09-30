# CPU backends and performance

docs/architecture.md section 9 and docs/decisions.md D-007 (CPU first, GPU
as optional future work). Implementation: `src/kerrray/raytracing/backends/`
(`base.py`, `numpy_backend.py`, `numba_backend.py`, `numba_kernel.py`) and
`src/kerrray/raytracing/rays.py`.

## 1. Backends

`get_backend(name)` returns an object with
`integrate_batch(st, Y0, integ, term, events=None) -> BatchResult`.

| name | implementation | availability |
|---|---|---|
| `numpy` | `NumpyBackend`: `geodesics.integrate_batch`, the vectorised reference (per-ray adaptive Dormand-Prince 5(4) or RK4, an active mask, no Python loop over rays) | always |
| `numba` | `NumbaBackend`: the scalar per-ray kernel of `numba_kernel.py` under `numba.njit(parallel=True, fastmath=False, cache=True)`, one ray per `prange` iteration | only when Numba can be imported; otherwise `get_backend("numba")` raises `BackendUnavailable` with the reason |
| `cuda` | none | never; always raises `BackendUnavailable` ("GPU acceleration is optional future work (docs/decisions.md D-007); no CUDA device is used"). No GPU number is ever reported. |

`available_backends()` returns `{"numpy": True, "numba": <importable>, "cuda": False}`.
The package imports cleanly without Numba.

`rays.trace_rays` runs any backend (by name or instance) in chunks of at
most `DEFAULT_CHUNK_SIZE = 16384` rays and calls a progress hook after each
chunk. The NumPy right-hand side allocates a few dozen `(n_active, 8)`
temporaries, so a chunk of 16384 float64 states (1 MB) keeps its working
set near 60 MB. Every ray carries its own step size and error control, so
chunked results are bit-identical to unchunked ones (tested).

## 2. Agreement by construction

The Numba kernel mirrors the NumPy reference operation by operation:

* **Metric.** `inverse_metric_terms` evaluates the five contravariant Kerr
  components and their `r` and `theta` derivatives with the closed forms of
  `geometry.inverse_metric_components` and `inverse_metric_derivatives`
  (BPT 1972 eq. 2.1 definitions, verified against SymPy in the geometry
  tests). It uses the same operation order, once per right-hand-side call.
* **Right-hand side.** `geodesic_rhs_scalar` is the Hamiltonian right-hand
  side with the exact `L_z = 0` polar-axis guard. Metric values are cast to
  the working dtype before they meet the momenta, as in
  `geodesics.equations.geodesic_rhs`.
* **Stepping.** The tableaus are imported from `geodesics.tableaus`, never
  retyped. The error norm is the Hairer-Norsett-Wanner scaled RMS, with
  NumPy's pairwise summation order over the eight components, and the
  step-size controller, termination rules and priority are those of the
  reference.
* **Floating point.** `fastmath` is off, so there is no reassociation or
  FMA contraction. Constants that meet working-dtype values come from a
  working-dtype array, so a Python literal never promotes float32 to
  float64. The affine parameter, the conservation diagnostics and the
  disk-crossing interpolant are float64, as in the reference.

Required agreement (`tests/test_numba_backend.py`):

* identical termination states, step counts and rejected-step counts;
* final states and `lambda` to `1e-9` relative in float64;
* conservation maxima to `1e-9`;
* the same agreement for RK4 and budget-limited runs, for the disk-plane
  event (float64 and float32) and in the Schwarzschild limit;
* a float32 variant that matches NumPy float32, with a larger Carter drift
  than float64;
* the kernel metric and right-hand side against the geometry and geodesics
  modules (RHS bit-equal; metric to `1e-12` relative).

The metric tolerance is `1e-12`, not tighter, because `d_r g^tt` subtracts
two nearly equal products. A one-ulp operand difference between scalar and
vectorised evaluation shows up as `1.7e-14` relative at `a = -0.7`.

## 3. Numba availability in this environment

> **Update, 2026-09-30.** Sections 3 and 4 describe the state early on
> 2026-09-29 and are out of date. Numba (0.67.0) was installed later that
> day and is now the optional `fast` extra in `pyproject.toml`. Compiled
> Numba run times were measured in the CPU benchmark, run
> 20260929T052912Z-f23a49 (Table 9 of `research.md`), and the Numba backend
> is used for the image studies of the paper. The 64 × 64 run times in
> section 4 have no run record; the recorded a = 0.9 run
> (20260929T054034Z-de5a37) took 32.4 s.

**Numba is not installed in the project virtual environment** (checked
2026-09-29 with `pip list`, and no other interpreter on the machine has it).
It is also not declared in `pyproject.toml`. Therefore:

* `get_backend("numba")` raises `BackendUnavailable`, and
  `available_backends()["numba"]` is `False`;
* the kernel is verified in **interpreted mode**
  (`NumbaBackend(interpreted=True)`), which runs the same kernel source in
  plain Python. This checks the kernel's logic and floating-point order
  against NumPy, but it is a verification mode, not a performance mode;
* no compiled-Numba runtime has been measured, and none is reported
  anywhere.

When Numba is installed (it is the one extra dependency architecture
section 8 allows), the same tests exercise the compiled kernel with no code
change. The interpreted-mode subset in
`test_agrees_with_numpy_on_camera_rays_float64` (56 of the first 500
pixels) then becomes the full 500.

## 4. Measurements

All runs on 2026-09-29, 8 logical CPUs, `rk45` with rtol `1e-9`, atol
`1e-11`, float64, `r_o = 1000 M`, `i = 60 deg`, `fov = 12 M`.

| run | rays | wall-clock |
|---|---|---|
| numpy, 64 x 64 shadow, a = 0 | 4096 | 7.39 s (554 rays/s) |
| numpy, 64 x 64 shadow, a = 0.9 | 4096 | 9.39 s (436 rays/s) |
| numpy, 4 x 4 shadow, a = 0.9 | 16 | 1.67 s |
| numba kernel **interpreted**, same 4 x 4 shadow | 16 | 2.11 s (0.132 s per ray); states identical to numpy, max relative state difference 0 |
| numba **compiled**, 64 x 64 shadow | 4096 | not measurable here (Numba not installed) |

The NumPy cost per ray is much higher for 16 rays than for 4096, because
the batched loop's per-step overhead is shared by fewer rays. The CPU
benchmark (`benchmarks/cpu.py`, EXP-009, performance role) is the place for
ray-count scaling.

## 5. Precision

`integ.dtype = "float32"` runs states, derivatives, step sizes and error
control in float32. The geometry is evaluated in float64 and cast, and
`lambda` and the diagnostics stay in float64. float32 cannot resolve
`horizon_epsilon = 1e-6` next to `r_plus ~ 1`, so use `>= 1e-3` and
`rtol >= 1e-6` (docs/numerical_methods.md section 7). Both backends support
it identically (tested).
