# Decisions

Settled specification ambiguities and engineering choices, each recorded as
Decision / Reason / Consequence. The first three are the ambiguities listed in
CLAUDE.md ("Spec ambiguities to settle early"); PROJECT.md wins on any
conflict with this file.

## D-001: Observer angle configuration key

**Decision.** `inclination_deg` (PROJECT.md section 34) is the single
configuration key for the observer's polar position. It is the
Boyer-Lindquist polar angle `theta` of the observer in degrees: `0` places the
observer on the spin axis, `90` in the equatorial plane. The key `theta_deg`
from section 15 is not accepted and raises a configuration error. The
observer azimuth key is `phi_deg` (degrees) with default `0`.

**Reason.** Sections 15 and 34 name the same quantity differently
(`theta_deg` and `inclination_deg`). One key avoids silent divergence between
CLI flags (`--inclination`, sections 17 and 28) and YAML files, and the
explicit `_deg` suffix keeps the unit visible.

**Consequence.** `src/kerrray/utils/config.py` validates
`0 <= inclination_deg <= 180` and rejects unknown keys, so an old-style
`theta_deg` file fails loudly instead of being ignored. Phase 5 (camera and
observer) must map `inclination_deg` to the Boyer-Lindquist `theta` exactly
as stated here and document the mapping in `docs/equations.md`. See D-008 for
the treatment of the axis itself.

## D-002: Escape radius equal to the observer radius

**Decision.** A ray is classified `ESCAPED` only when `r >= escape_radius`
AND it moves radially outward along the integration direction. A ray launched
inward from an observer at `r = escape_radius` is therefore never classified
as escaped at launch. Configuration validation emits a `logger.warning` when
`termination.escape_radius` is smaller than `observer.radius`.

**Reason.** The section 34 example sets both radii to `1000.0`. Section 11
already requires "outward motion" for `ESCAPED`; making that condition
explicit prevents every backward-traced ray from terminating at its first
step. An escape radius below the observer radius is legal but unusual, hence
a warning rather than an error.

**Consequence.** The classifier implemented in Phase 2 must test the sign of
the radial motion in addition to the radius. The decision is recorded now so
that Phase 2 and Phase 5 implement the same rule; the configuration warning
is live from Phase 0.

## D-003: Pandas is not a dependency

**Decision.** Pandas is not in `pyproject.toml`. It is added only if a later
phase needs DataFrames (for example for benchmark tables in Phase 6 or
Phase 9).

**Reason.** Section 31 lists Pandas but the Phase 0 prompt in section 50 does
not, and sections 31, 48 and 50 all ask for a small dependency tree.

**Consequence.** Tables are built with Rich and plain Python structures
(`src/kerrray/reporting/tables.py`) until a phase demonstrates a concrete
need; adding Pandas then requires a note in this file.

## D-004: Configuration keys are explicit; the `experiment` block is optional

**Decision.** Every key of the five physics blocks of PROJECT.md section 34
(`black_hole`, `observer`, `integration`, `raytrace`, `termination`) is
required in the YAML file; the only default among them is
`observer.phi_deg = 0` (D-001). The section 34 example therefore loads
verbatim. The `experiment` block (`name`, `description`, `parameters`) is a
KerrRay addition and is optional: when it or its `name` key is absent,
`load_config` names the experiment after the file stem
(`configs/kerr.yaml` -> `kerr`) and `KerrRayConfig.from_yaml` /
`from_mapping` use `"unnamed"` unless the caller passes `default_name`.
When the block is present it is validated like any other (unknown keys are
errors). Experiment-specific settings live under `experiment.parameters`, a
mapping owned by the experiment driver; the schema does not interpret it.
`parameters` is exposed as a read-only mapping (`types.MappingProxyType`
over a private deep copy): keys cannot be added, replaced or removed, and
the nested lists and mappings inside it must be treated as read-only.
Drivers that need to modify values copy first (`KerrRayConfig.to_mapping()`
or `kerrray.utils.config_parsing.to_plain`).

*Amendment (2026-09-28, docs/architecture.md section 7).* The keys added by
the architecture contract (`integration.step_size`, `integration.lambda_max`,
`raytrace.fov`, `raytrace.backend`, `raytrace.dtype`, `experiment.seed`,
`experiment.output_dir`, `experiment.report_dir`) carry documented defaults
so that the section 34 example and older files still load; every shipped
`configs/*.yaml` nevertheless states them explicitly. The section 34 keys
themselves remain required. The structure of `experiment.parameters` is
fixed by D-009.

**Reason.** Sections 34, 47 and 48 forbid hard-coded experiment parameters
and hidden constants, so the physics inputs stay explicit. Section 34 does
not define an `experiment` block, and the CLI commands of section 28
(`kerrray shadow --spin 0.9 ...`, `kerrray experiment --config ...`) must be
able to start from any section 34 file without inventing a name. The
read-only mapping keeps a frozen configuration from being mutated in place
by one driver and silently reused by another in the same process.

**Consequence.** Command-line flags in later phases start from a
configuration file and apply dotted-key overrides
(`kerrray.utils.config.apply_overrides`) rather than from built-in defaults.
Every `configs/*.yaml` still states `experiment.name` explicitly so that run
directories and reports are named deliberately.

## D-005: Integration method names

**Decision.** `integration.method` must be one of `rk4` (fixed-step
classical Runge-Kutta), `rk45` (adaptive embedded Dormand-Prince 5(4)) or
`dop853` (SciPy's `solve_ivp` method `DOP853`, the eighth-order
Dormand-Prince scheme, used as an independent reference for cross-checks and
never as the default); the set is `SUPPORTED_INTEGRATION_METHODS` in
`src/kerrray/utils/config_blocks.py`, re-exported by `config.py`.
*Updated 2026-09-28: `dop853` added, following docs/architecture.md
section 4.*

**Reason.** `rk4` and `rk45` are the two schemes section 19 names for the
solver comparison. The trajectory-error measurements of EXP-006 and EXP-009
need a reference trajectory of much higher accuracy than the scheme under
test; delegating that reference to SciPy's DOP853 avoids hand-typing a third
tableau while keeping it independent of KerrRay's own integrators.
Validating the name at load time makes configuration errors visible before
any physics runs.

**Consequence.** `rk4` and `rk45` are implemented in
`src/kerrray/geodesics/integrators.py`; `dop853` delegates to SciPy's
`solve_ivp` and is intended for single-ray reference integrations, while the
batched CPU backends share one tableau and right-hand side for `rk4` and
`rk45` (docs/architecture.md section 9). Adding another scheme extends the
set and this entry.

## D-006: Run identifier format

**Decision.** `run_id` is `YYYYMMDDTHHMMSSZ-xxxxxx`: a UTC timestamp followed
by six random hexadecimal characters from `secrets`, independent of the
experiment seed.

**Reason.** Section 35 requires a run id and a timestamp; the random suffix
guarantees uniqueness when identical configurations run within one second
or on several machines, and keeping it independent of `random_seed`
preserves the meaning of the seed as a physics-only input.

**Consequence.** Run directories sort chronologically by name, and the seed
recorded in the manifest is the only random input that influences results.

## D-007: GPU acceleration is optional future work; Phase 8 is CPU performance and precision

**Decision.** PROJECT.md sections 25 to 27, Phase 8 of section 49 and EXP-010
treat GPU acceleration as a core deliverable. For this build it is
**optional future work**. Phase 8 is redefined as *CPU performance and
precision*: the `numpy` backend (vectorised batched integrator, the
reference) and the `numba` backend (CPU-parallel, same tableau and
right-hand side, required to agree with `numpy` to floating-point round-off)
are the two supported backends; `cuda` is a named but unavailable backend
(`raytrace.backend: cuda` validates, `get_backend("cuda")` raises
`BackendUnavailable`). `kerrray benchmark gpu` prints that GPU acceleration
is optional future work and exits 0 without inventing a number. EXP-009
becomes the CPU benchmark (1,000 to 1,000,000 rays; runtime, peak memory via
`tracemalloc` plus array accounting, max null-constraint error, max energy
drift, trajectory error against a tight-tolerance reference, shadow-boundary
convergence per resolution) and EXP-011 the float32-versus-float64 study on
the CPU backends. The guiding research question becomes: *How do numerical
integration tolerances and ray density affect the accuracy of Kerr shadow
reconstruction on CPU-based systems?* (docs/scientific_background.md).
`configs/gpu.yaml` keeps its name for section 30 compatibility but holds the
CPU benchmark; `configs/benchmark.yaml` is the primary file with the same
content.

**Reason.** (docs/architecture.md section 9.) The development machine has no
CUDA device, so any GPU number would be unverifiable and PROJECT.md
sections 26 and 47 forbid reporting unmeasured results; the scientific
content of the project (photon dynamics, validation, error analysis) does not
depend on a GPU; and numerical efficiency on commodity hardware is the
better research angle for a reproducible study.

**Consequence.** `CLAUDE.md` Phase 8 reads "CPU performance and precision
(GPU optional future work)"; the README roadmap lists GPU as an optional
branch; PROJECT.md is not rewritten. A future GPU backend must reuse the
`Backend` protocol of `raytracing/backends/base.py` and satisfy the same
agreement-with-`numpy` test before any speed-up is reported.

## D-008: Observer inclination is clamped away from the spin axis

**Decision.** `observer.inclination_deg` is validated in `[0, 180]` and then
clamped to `[1e-3, 180 - 1e-3]` degrees with a `logger.warning`
(`INCLINATION_EPSILON_DEG` in `src/kerrray/utils/config_blocks.py`);
`ObserverConfig.inclination_rad` exposes the clamped angle in radians. The
clamp is a documented approximation:

- **WHAT.** An observer requested on the spin axis (`0` or `180` degrees, or
  within `1e-3` degrees of it) is placed at polar angle `1e-3` degrees
  (`179.999` degrees) instead.
- **WHY.** The camera is built from the zero-angular-momentum-observer
  (ZAMO) orthonormal tetrad, whose `phi` leg is proportional to
  `1 / sin(theta)` and is undefined at `sin(theta) = 0`; the Boyer-Lindquist
  `phi` coordinate itself is degenerate on the axis. Clamping keeps the
  tetrad finite without a second coordinate chart.
- **LIMITATION.** An "on-axis" image is really the image seen at
  `theta = 1e-3` degrees; the shadow is circularly symmetric only up to the
  asymmetry introduced by that offset, which vanishes with `sin(theta)`
  (`sin(1e-3 deg) = 1.7e-5`) and is to be quantified, not assumed, by the
  inclination sweep EXP-004. Validation of the face-on limit must quote the
  clamped angle, not `0`.

**Reason.** EXP-004 (PROJECT.md section 37) requires an inclination of `0`
degrees, and rejecting it would make the required sweep impossible; a
silent shift would violate CLAUDE.md's rule that approximations are stated.
`1e-3` degrees is far below any angular resolution the experiments probe
(a `64` to `1024` pixel image plane of half-width `12 M` resolves angles of
order `1e-2` to `1e-1` in `alpha / r_o`) while keeping `sin(theta)` at
`~1.7e-5`, well away from floating-point underflow.

**Consequence.** `configs/shadow.yaml` lists `0` in the inclination sweep and
the run logs the clamp; the camera role (docs/raytracing.md) implements the
tetrad assuming `sin(theta) > 0` and may rely on this guarantee.

## D-009: Experiment parameter sub-blocks

**Decision.** `experiment.parameters` is a mapping of *named sub-blocks*,
each owned and parsed by one experiment driver into a frozen dataclass with
`kerrray.utils.config.experiment_block(cfg, name, BlockType)` (built on
`config_parsing.parse_block`; unknown keys inside a sub-block are errors; a
missing sub-block yields the dataclass defaults; converters exist for
`float`, `int`, `str`, `bool`, `float | None`, `list[float]`, `list[int]`,
`list[str]` and the tuple equivalents). The sub-block names and defaults are
those of docs/architecture.md section 7:

| Sub-block | Keys (defaults) | Used by |
|---|---|---|
| `lensing` | `impact_min 3.0, impact_max 20.0, n_rays 40, launch_radius 1000.0` | `configs/lensing.yaml` |
| `convergence` | `resolutions [64,128,256,512], tolerances [1e-6,1e-8,1e-10,1e-12], step_sizes [0.1,0.01,0.001,0.0001]` | `configs/convergence.yaml`, `configs/shadow.yaml` |
| `near_critical` | `offsets [1e-1 ... 1e-8]` (relative: b = b_c(1 ± δ)), `launch_radius 1000.0`, `tolerances`, `spins`, `bisection_tol`, `bisection_max_iterations` | `configs/near_critical.yaml` |
| `frame_dragging` | `spins [-0.9,0.0,0.9], impact_parameter 6.0, launch_radius 1000.0`, `scaling_impact_parameters`, `polar_impact_parameter` | `configs/kerr.yaml` |
| `spin_sweep` | `spins [0.0,0.25,0.5,0.75,0.9,0.99]` | `configs/shadow.yaml`, `configs/kerr.yaml` |
| `inclination_sweep` | `inclinations_deg [0,30,45,60,75,90]` | `configs/shadow.yaml` |
| `disk` | `enabled false, r_in null (ISCO), r_out 20.0, emissivity_index 3.0, intensity_law g4, prograde true` | absent from the shipped files (defaults apply) |
| `benchmark` | `ray_counts [1000,10000,100000,1000000], backends [numpy,numba], dtypes [float32,float64], repeats 3`, `max_seconds_per_run`, `reference_rays`, `reference_rtol`, `horizon_epsilon_float32 1e-3`, `precision_tolerances` | `configs/benchmark.yaml`, `configs/gpu.yaml` |
| `solver` | `n_rays, repeats, spins, launch_radius 50.0, reference_rtol, reference_atol, max_seconds_per_config 600` | `configs/benchmark.yaml`, `configs/convergence.yaml` |
| `validation` | Schwarzschild: `photon_sphere_tol 1e-6, critical_b_tol 1e-6, bisection_iterations 60`; Kerr: `horizon_tol, small_spins, schwarzschild_limit_tol, conservation_tol, critical_b_tol, bisection_iterations, near_extremal_spins` (provisional values in the file); extension keys listed in docs/validation.md section 7 | `configs/schwarzschild.yaml`, `configs/kerr.yaml` |

`raytrace.resolution` is `64` in every shipped file (the development default
of docs/architecture.md section 8); resolution studies are driven by
`convergence.resolutions`, and `1024` is opt-in through a CLI override.

**Reason.** D-004 left `experiment.parameters` free-form, which would let
each driver invent its own key names and let a typo in a YAML file pass
unnoticed. Fixing the names in one place lets the YAML files, the drivers and
the tests (`tests/test_config.py` parses every shipped sub-block against the
section 7 shapes) agree, while still keeping every experiment input in the
configuration file rather than in code (PROJECT.md section 34).

**Consequence.** A driver that needs a new key adds it to its dataclass with
a default and to this table; a driver that receives an unknown key fails
loudly. The `validation` tolerances in `configs/kerr.yaml` are provisional
Phase 4 inputs (PROJECT.md section 13 names the checks but no thresholds)
and must be confirmed by the Kerr validation work.
