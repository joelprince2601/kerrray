# KerrRay architecture and interface contract

This document is the binding interface contract between the KerrRay subsystems.
It exists so that the subsystems can be developed in parallel against a single
set of conventions and function signatures. `PROJECT.md` is the specification
and wins on any conflict of intent; this document only fixes *how* the
specification is realised in code. Every equation quoted here is a **statement
of the convention to implement**, not a substitute for verification: the
implementer of each module must re-derive or symbolically verify every formula
(see `docs/equations.md`) and, if a verified derivation disagrees with this
document, the derivation wins and the disagreement must be flagged in
`docs/decisions.md`.

## 1. Conventions

| Item | Convention |
|---|---|
| Units | Geometric units, G = c = 1. Mass `M` is a parameter (default 1.0). Lengths, times and the affine parameter are in units of M. |
| Signature | (-, +, +, +) |
| Coordinates | Boyer-Lindquist (t, r, theta, phi), index order 0, 1, 2, 3. |
| Spin | `spin` is the dimensionless a* = a/M with abs(a*) < 1; `a = spin * mass`. `a > 0` means the hole rotates towards +phi. A photon is *prograde* when `a * L_z > 0`. |
| Sigma, Delta | Sigma = r^2 + a^2 cos^2(theta), Delta = r^2 - 2 M r + a^2. |
| Horizons | r_pm = M pm sqrt(M^2 - a^2). Capture uses the outer horizon r_plus. Ergosphere: r_E(theta) = M + sqrt(M^2 - a^2 cos^2(theta)). |
| Momentum | Covariant components p_mu. Conserved: E = -p_t, L_z = p_phi, Carter Q = p_theta^2 + cos^2(theta) (L_z^2 / sin^2(theta) - a^2 E^2) (null case). |
| Future-directed photon | E > 0, i.e. p_t < 0. |
| Photon state | `y = [t, r, theta, phi, p_t, p_r, p_theta, p_phi]`, a float64 array of shape (8,). A batch is `Y` of shape (N, 8). Angles in radians internally; degrees only at the config/CLI boundary. |
| Affine parameter | lambda, integrated forwards (d lambda > 0). Backward ray tracing reverses the momentum (`p_mu -> -p_mu`) and integrates forwards in lambda; this is exact because the geodesic equation is invariant under (lambda, p) -> (-lambda, -p). |
| Formulation | Primary: first-order Hamiltonian form, H = (1/2) g^{mu nu} p_mu p_nu, with dx^mu/dlambda = g^{mu nu} p_nu and dp_mu/dlambda = -(1/2) (d_mu g^{alpha beta}) p_alpha p_beta. Only d_r and d_theta are non-zero, so p_t and p_phi are exactly constant in the ODE. Secondary (cross-check only): second-order Christoffel form with state `z = [x^mu, u^mu]`. |
| Termination states | `TerminationState` IntEnum: `RUNNING = 0` (internal), `ESCAPED = 1`, `CAPTURED = 2`, `MAX_AFFINE_PARAMETER = 3`, `NUMERICAL_FAILURE = 4`, `OUT_OF_DOMAIN = 5`, `DISK_HIT = 6` (only when a disk event is enabled; documented extension of PROJECT.md section 11). Exhausting `max_steps` maps to `MAX_AFFINE_PARAMETER`. |
| CAPTURED | r <= r_plus + `horizon_epsilon`. |
| ESCAPED | r >= `escape_radius` **and** dr/dlambda > 0 in the integration direction (never at launch, see `docs/decisions.md`). |
| OUT_OF_DOMAIN | r < 0, theta outside [0, pi] after the integrator's own handling, or non-finite state that is not a solver failure. |
| NUMERICAL_FAILURE | NaN/Inf produced, or adaptive step underflow (h < `h_min`). |
| Observer angle | Config key `observer.inclination_deg` = Boyer-Lindquist theta of the observer in degrees. 0 deg = on the spin axis. Values are clamped to [`inclination_epsilon_deg`, 180 - `inclination_epsilon_deg`] with `inclination_epsilon_deg = 1e-3` and a logged warning, because the ZAMO tetrad is singular on the axis (documented approximation: WHAT/WHY/LIMITATION). |
| Random seed | `experiment.seed` (default 0) recorded in every manifest. |

### 1.1 Kerr metric in Boyer-Lindquist coordinates (to be verified, not copied)

Covariant components (all other components zero):

```
g_tt     = -(1 - 2 M r / Sigma)
g_tphi   = -2 M a r sin^2(theta) / Sigma
g_rr     = Sigma / Delta
g_thth   = Sigma
g_phph   = (r^2 + a^2 + 2 M a^2 r sin^2(theta) / Sigma) sin^2(theta)
```

Contravariant components:

```
g^tt     = -[(r^2 + a^2)^2 - a^2 Delta sin^2(theta)] / (Sigma Delta)
g^tphi   = -2 M a r / (Sigma Delta)
g^rr     = Delta / Sigma
g^thth   = 1 / Sigma
g^phph   = (Delta - a^2 sin^2(theta)) / (Sigma Delta sin^2(theta))
```

References to check against (the implementer must cite the ones actually used):
Bardeen, Press & Teukolsky 1972, ApJ 178, 347 (eq. 2.1 and the separated
potentials); Carter 1968, Phys. Rev. 174, 1559; Misner, Thorne & Wheeler 1973,
section 33.2; Chandrasekhar 1983, *The Mathematical Theory of Black Holes*,
ch. 6; Visser 2007, arXiv:0706.0622 ("The Kerr spacetime: a brief
introduction"). Required symbolic checks (SymPy): g times g^-1 = identity;
Schwarzschild limit a -> 0; Ricci tensor vanishes (evaluate the symbolic
Ricci tensor numerically at random points, abs(R_mu_nu) < 1e-8; no need to
simplify symbolically); Christoffel symbols computed from the metric
definition match the closed-form implementation at random points.

### 1.2 Carter potentials (null case)

With xi = L_z / E and eta = Q / E^2:

```
R(r)       = [(r^2 + a^2) - a xi]^2 - Delta [eta + (xi - a)^2]      (times E^2)
Theta(th)  = eta + a^2 cos^2(th) - xi^2 cot^2(th)                    (times E^2)
```

Spherical photon orbits satisfy R(r) = R'(r) = 0; solving gives the standard
parametrisation xi(r), eta(r) used for the analytic shadow curve (Bardeen
1973; see also Johannsen & Psaltis 2010, ApJ 718, 446, and Cunha & Herdeiro
2018, Gen. Rel. Grav. 50, 42). Implementers derive xi(r), eta(r) with SymPy
from R = R' = 0 rather than typing them in, then cite the reference they
match. Equatorial circular photon orbits: solve R = R' = 0 with eta = 0
numerically (polynomial root finding) and compare with Bardeen, Press &
Teukolsky 1972 eq. 2.18, r_ph = 2M {1 + cos[(2/3) arccos(-+ a/M)]} (upper
sign prograde).

### 1.3 Celestial coordinates and the camera

The observer sits at (r_o, theta_o, phi_o) and uses the zero-angular-momentum
observer (ZAMO) orthonormal tetrad. For a photon arriving with local tetrad
components p^(a), the image-plane (celestial) coordinates are

```
alpha = -r_o p^(phi) / p^(t)        beta = r_o p^(theta) / p^(t)
```

in units of M (Cunha & Herdeiro 2018 style; for r_o -> infinity these reduce
to Bardeen's alpha = -xi / sin(theta_o), beta = +-sqrt(eta + a^2 cos^2(theta_o)
- xi^2 cot^2(theta_o))). The camera inverts this: for pixel (alpha, beta) it
sets p^(t) = 1, p^(phi) = -alpha / r_o, p^(theta) = beta / r_o,
p^(r) = +sqrt(1 - (alpha^2 + beta^2) / r_o^2) (photon arriving from the hole
side, moving outward), converts to Boyer-Lindquist covariant p_mu via the
tetrad, reverses the sign for backward tracing, and re-normalises the null
condition numerically. The implementer must write the derivation in
`docs/raytracing.md`, including the sign conventions, and test that (i) the
initial state is null to 1e-12, (ii) E > 0 for the physical photon, (iii) for
a = 0 the impact parameter b = sqrt(alpha^2 + beta^2) (1 + O(M/r_o)) is
recovered from L_z/E and Q/E^2. Pixel centres are offset by half a pixel and
resolutions are even, so alpha = 0 (L_z = 0 exactly) is never sampled.
`raytrace.fov` is the half-width of the image plane in units of M (alpha at
the outer pixel edge).

## 2. Package layout and file ownership

Each subsystem owns the files listed for it and must not edit files owned by
another subsystem. Shared files (`cli.py`, `utils/config.py`, `CLAUDE.md`,
`README.md`, `pyproject.toml`, the five spec doc files) are edited only by the
`infra` and `integration` roles.

```
src/kerrray/
  __init__.py, __main__.py, cli.py                     infra / integration
  commands/                                            one module per CLI verb (see section 6)
  geometry/   metric.py christoffel.py coordinates.py horizons.py symbolic.py    geometry role (symbolic.py: symbolic role)
  geodesics/  state.py equations.py integrators.py initial_conditions.py         geodesics role
  photons/    classification.py constants.py trajectories.py orbits.py           geodesics role (orbits.py: kerr role)
  raytracing/ camera.py rays.py renderer.py shadow.py boundary.py                camera role (camera.py) / raytracing role
  raytracing/backends/ base.py numpy_backend.py numba_backend.py                 raytracing role (base, numpy), performance role (numba)
  physics/    frame_dragging.py lensing.py redshift.py accretion.py              kerr role (frame_dragging), rendering role (rest)
  validation/ schwarzschild.py kerr.py conservation.py convergence.py            schwarzschild role (kerr.py: kerr role)
  benchmarks/ solver.py cpu.py precision.py memory.py gpu.py                     numerical role (solver), performance role (cpu, precision, memory, gpu stub)
  experiments/ base.py shadow.py near_critical.py lensing.py spin_sweep.py
               frame_dragging.py step_size.py                                    infra (base.py); others as named in section 6
  reporting/  console.py plots.py tables.py report.py                            infra role
  utils/      config.py logging.py seeds.py manifest.py units.py                 infra role
tests/        one or more test files per module, named test_<module>.py          each owner
docs/         equations.md numerical_methods.md validation.md experiments.md
              scientific_background.md raytracing.md decisions.md architecture.md
```

## 3. Geometry API (`kerrray.geometry`)

All functions are NumPy-vectorised: `r` and `theta` may be scalars or arrays of
any common broadcastable shape; outputs broadcast accordingly.

```python
# metric.py
@dataclass(frozen=True)
class Spacetime:
    mass: float = 1.0
    spin: float = 0.0            # dimensionless a* = a / M, validated abs(spin) < 1
    @property
    def a(self) -> float: ...    # spin * mass
    @property
    def is_schwarzschild(self) -> bool: ...

class MetricComponents(NamedTuple):        # covariant g_{mu nu}
    g_tt: Array; g_tphi: Array; g_rr: Array; g_thth: Array; g_phph: Array

class InverseMetricComponents(NamedTuple): # contravariant g^{mu nu}
    gtt: Array; gtphi: Array; grr: Array; gthth: Array; gphph: Array

def sigma(st: Spacetime, r, theta) -> Array
def delta(st: Spacetime, r) -> Array
def metric_components(st, r, theta) -> MetricComponents
def inverse_metric_components(st, r, theta) -> InverseMetricComponents
def metric(st, r, theta) -> Array              # shape (..., 4, 4)
def inverse_metric(st, r, theta) -> Array      # shape (..., 4, 4)
def inverse_metric_derivatives(st, r, theta) -> tuple[InverseMetricComponents, InverseMetricComponents]
    # (d/dr of the five g^{mu nu}, d/dtheta of the five g^{mu nu}); closed form, verified against SymPy
def schwarzschild(mass: float = 1.0) -> Spacetime
def kerr(mass: float, spin: float) -> Spacetime

# christoffel.py
def christoffel(st, r, theta) -> Array         # shape (..., 4, 4, 4), index order [mu, alpha, beta] for Gamma^mu_{alpha beta}

# horizons.py
def horizon_radii(st) -> tuple[float, float]   # (r_plus, r_minus)
def outer_horizon(st) -> float
def ergosphere_radius(st, theta) -> Array
def inside_horizon(st, r, epsilon: float = 0.0) -> Array[bool]

# coordinates.py
def bl_to_cartesian(st, r, theta, phi) -> tuple[Array, Array, Array]
    # x = sqrt(r^2 + a^2) sin(theta) cos(phi), y = sqrt(r^2 + a^2) sin(theta) sin(phi), z = r cos(theta) (document the choice)
def deg_to_rad(x) / rad_to_deg(x)

# symbolic.py (SymPy, import lazily; never on the hot path)
def symbolic_metric() -> (sympy.Matrix, symbols)      # g_{mu nu}(t, r, theta, phi; M, a)
def symbolic_inverse_metric() -> sympy.Matrix
def symbolic_christoffel() -> nested list / sympy.Array
def lambdified_christoffel() -> Callable[(M, a, r, theta) -> ndarray(4, 4, 4)]
def ricci_tensor_numeric(M, a, r, theta) -> ndarray(4, 4)   # symbolic Ricci evaluated at a point
```

`utils/units.py` (infra): `GeometricUnits(mass_solar: float)` with conversions
`length_m(x_M)`, `time_s(t_M)`, `mass_kg`, `frequency_hz`, using CODATA/IAU
constants with cited values; the internal engine never uses SI.

## 4. Geodesic API (`kerrray.geodesics`, `kerrray.photons`)

```python
# geodesics/state.py
IDX_T, IDX_R, IDX_TH, IDX_PH, IDX_PT, IDX_PR, IDX_PTH, IDX_PPH = range(8)
@dataclass(frozen=True)
class PhotonState:
    x: ndarray(4); p: ndarray(4)
    def to_array(self) -> ndarray(8); @classmethod from_array(cls, y)
def pack(x, p) -> ndarray(..., 8); def unpack(y) -> (x, p)

# geodesics/equations.py
def hamiltonian(st, y) -> Array                 # (1/2) g^{mu nu} p_mu p_nu, shape (...)
def geodesic_rhs(st, y) -> Array                # dy/dlambda, shape (..., 8), Hamiltonian form
def geodesic_rhs_christoffel(st, z) -> Array    # second-order form, z = [x, u], for cross-checks
def null_momentum_pt(st, x, p_r, p_theta, p_phi, *, future_directed=True) -> Array
    # solves g^{tt} p_t^2 + 2 g^{tphi} p_t p_phi + (g^{rr} p_r^2 + g^{thth} p_theta^2 + g^{phph} p_phi^2) = 0 for p_t;
    # future_directed=True picks the root with E = -p_t > 0 and raises if none exists

# geodesics/integrators.py
@dataclass(frozen=True)
class IntegratorOptions:
    method: str = "rk45"          # "rk4" (fixed step), "rk45" (Dormand-Prince 5(4), adaptive), "dop853" (SciPy reference)
    rtol: float = 1e-9
    atol: float = 1e-11
    step_size: float = 0.01       # fixed step in lambda (units of M) for rk4; initial step for rk45
    max_steps: int = 100_000
    lambda_max: float = 1.0e4     # affine-parameter budget (units of M)
    h_min: float = 1e-12          # below this the ray is NUMERICAL_FAILURE
    dtype: str = "float64"        # "float32" | "float64" (batched path only)
@dataclass(frozen=True)
class TerminationOptions:
    horizon_epsilon: float = 1e-6
    escape_radius: float = 1000.0
@dataclass(frozen=True)
class EventOptions:               # optional; used by the accretion renderer
    disk_plane: bool = False      # stop at theta = pi/2 crossings with r_in <= r <= r_out
    r_in: float = 6.0; r_out: float = 20.0

def integrate(st, y0, integ: IntegratorOptions, term: TerminationOptions, *, record: bool = True) -> Trajectory
def integrate_batch(st, Y0, integ, term, *, events: EventOptions | None = None) -> BatchResult
    # vectorised over rays with per-ray adaptive step size and an active mask; float32 supported via integ.dtype
def make_step_rk4(rhs) / make_step_rk45(rhs)   # tableau helpers (Dormand & Prince 1980; Hairer, Norsett & Wanner 1993 Table 5.2)

# photons/classification.py
class TerminationState(IntEnum): RUNNING=0, ESCAPED=1, CAPTURED=2, MAX_AFFINE_PARAMETER=3, NUMERICAL_FAILURE=4, OUT_OF_DOMAIN=5, DISK_HIT=6
def classify_state(st, y, dydl, term: TerminationOptions, *, lam, integ) -> TerminationState        # single ray
def classify_batch(st, Y, dYdl, term, *, lam, integ) -> ndarray[int]                                 # vectorised

# photons/constants.py
def energy(y) -> Array; def angular_momentum(y) -> Array
def carter_constant(st, y) -> Array; def null_constraint(st, y) -> Array   # = hamiltonian (should be 0)
def relative_drift(values, reference) -> Array                              # abs(v - v0) / max(abs(v0), floor), floor documented
@dataclass class ConservationDiagnostics: energy, lz, carter, null: arrays over the trajectory; *_drift arrays; max_* scalars

# photons/trajectories.py
@dataclass
class Trajectory:
    spacetime: Spacetime; lam: ndarray(n); y: ndarray(n, 8); state: TerminationState
    n_steps: int; n_rejected: int; runtime_s: float; diagnostics: ConservationDiagnostics
def closest_approach(traj) -> float
def azimuthal_winding(traj) -> float            # total delta phi (radians); turns = delta phi / (2 pi)
def deflection_angle(traj) -> float             # equatorial escaped rays; corrected for finite launch/exit radius; derivation documented
def to_cartesian(traj) -> (x, y, z)

@dataclass
class BatchResult:
    Y: ndarray(N, 8) final states; state: ndarray(N) int; n_steps: ndarray(N); lam: ndarray(N)
    max_null_error: ndarray(N); max_energy_drift: ndarray(N); max_lz_drift: ndarray(N); max_carter_drift: ndarray(N)
    runtime_s: float; event_Y: ndarray(N, 8) | None; event_hit: ndarray(N, bool) | None

# geodesics/initial_conditions.py
def photon_from_constants(st, r, theta, E, Lz, Q, *, sign_r: int, sign_theta: int, t=0.0, phi=0.0) -> ndarray(8)
def equatorial_photon(st, r0, b, *, inward=True, prograde=True, E=1.0) -> ndarray(8)   # impact parameter b = abs(L_z)/E
def tangential_photon(st, r0, *, prograde=True, E=1.0) -> ndarray(8)                    # p_r = 0 launch in the equatorial plane
def from_local_direction(st, x, n_local: ndarray(3), *, frame="zamo", E_local=1.0) -> ndarray(8)  # used by the camera
```

## 5. Ray tracing, backends, physics, validation, experiments

```python
# raytracing/camera.py
@dataclass(frozen=True)
class Camera:
    radius: float; inclination_deg: float; phi_deg: float = 0.0; fov: float = 12.0; resolution: int = 64
    @property n_rays; def pixel_coordinates(self) -> (alpha (H, W), beta (H, W))
def zamo_tetrad(st, r, theta) -> ndarray(4, 4)            # e_(a)^mu rows, documented
def initial_states(cam, st) -> ndarray(N, 8)               # past-directed, null-normalised, row-major over (H, W)

# raytracing/backends/base.py
class Backend(Protocol):
    name: str
    def integrate_batch(self, st, Y0, integ, term, events=None) -> BatchResult
class BackendUnavailable(RuntimeError)
def get_backend(name: str) -> Backend                     # "numpy" | "numba"; "cuda" always raises BackendUnavailable
                                                          # with the message that GPU acceleration is optional future work (section 9)
def available_backends() -> dict[str, bool]               # {"numpy": True, "numba": <importable>, "cuda": False}

# raytracing/rays.py
def trace_rays(st, Y0, integ, term, *, backend="numpy", events=None, progress: Callable | None = None) -> BatchResult
# raytracing/shadow.py
@dataclass class ShadowImage: alpha, beta (H, W); state (H, W) int; captured (H, W) bool; result: BatchResult; camera; spacetime
def compute_shadow(st, cam, integ, term, *, backend="numpy", progress=None) -> ShadowImage
# raytracing/boundary.py
def extract_boundary(img: ShadowImage, n_angles=360) -> (angles, radii)     # sub-pixel boundary in the (alpha, beta) plane
def analytic_boundary(st, inclination_deg, n=720) -> (alpha, beta)        # from photons.orbits (Bardeen curve)
def boundary_error(numeric, analytic) -> dict(max, rms, mean)
# raytracing/renderer.py
def render_disk(st, cam, integ, term, disk: DiskModel, *, backend="numpy") -> ndarray(H, W) intensity

# physics/lensing.py
def deflection_from_trajectory(traj) -> float
def schwarzschild_deflection_exact(st, b) -> float        # numerical quadrature of the exact integral (Darwin 1959 form), reference only
def weak_field_deflection(st, b) -> float                 # 4M/b, labelled approximation
# physics/redshift.py
def keplerian_angular_velocity(st, r, *, prograde=True) -> float          # BPT 1972 eq. 2.16
def emitter_four_velocity(st, r) -> ndarray(4)
def redshift_factor(st, y_hit, *, prograde=True) -> float                 # g = nu_obs / nu_em with observer at infinity
# physics/accretion.py
@dataclass class DiskModel: r_in (default: ISCO), r_out, emissivity_index p, intensity_law ("g3" | "g4"), prograde
def isco_radius(st, *, prograde=True) -> float                             # BPT 1972 eq. 2.21, and verified numerically
def emissivity(disk, r) -> Array
# physics/frame_dragging.py
def frame_dragging_experiment(spins, b, *, integ, term, r0) -> FrameDraggingResult   # same abs(b), launched identically for a<0, 0, >0

# validation/schwarzschild.py
def photon_sphere_experiment(integ, term, *, r_lo, r_hi, tol) -> PhotonSphereResult   # bisection on tangential launch radius
def critical_impact_experiment(integ, term, *, b_lo, b_hi, tol, r0) -> CriticalImpactResult   # bisection on b
def run_schwarzschild_validation(cfg) -> ValidationReport   # computed values, references 3M and 3 sqrt(3) M, relative errors
# validation/kerr.py
def run_kerr_validation(cfg) -> ValidationReport      # horizon, a->0 convergence, conservation, prograde/retrograde b_c vs computed reference, near-extremal
# validation/conservation.py, validation/convergence.py: reusable drift summaries and Richardson-style convergence-rate estimates

# experiments/base.py (infra)
@dataclass class ExperimentContext: cfg, run_dir, report_dir, logger, rng
def run_experiment(name, cfg, fn: Callable[[ExperimentContext], dict]) -> RunRecord   # writes runs/<id>/manifest.json + config.yaml, then reports/<id>/summary.json
# reporting/report.py (infra)
@dataclass class ReportSections: objective, model, method, parameters, results, error_analysis, interpretation, limitations, reproducibility
def write_report(report_dir, sections: ReportSections, figures: list[Path], tables: list[str]) -> Path
```

## 6. CLI

`cli.py` owns the Typer app and the banner. Every command module in
`src/kerrray/commands/` exposes `def register(app: typer.Typer) -> None` that
adds its command or sub-app; `cli.py` calls each `register` in this order:
`blackhole`, `geodesic`, `trace`, `shadow`, `lens`, `validate`, `benchmark`,
`experiment`, `report`. Owners: `blackhole`, `geodesic` -- kerr role;
`trace`, `shadow` -- raytracing role; `lens` -- rendering role; `validate` --
schwarzschild role (its `kerr` sub-command calls
`validation.kerr.run_kerr_validation`); `benchmark` -- numerical role (its
`solver` sub-command is the numerical role's own; its `cpu`, `precision` and
`gpu` sub-commands call `benchmarks.cpu.run_cpu_benchmark`,
`benchmarks.precision.run_precision_study` and `benchmarks.gpu.run_gpu_benchmark`
from the performance role, the last of which only reports that GPU
acceleration is optional future work and no device is used); `experiment` --
numerical role (dispatches on `experiment.name` to
`kerrray.experiments.<name>.run`); `report` -- infra role. Every command loads
a config (`--config`, defaulting to the matching file in `configs/`), applies
CLI overrides (`--spin`, `--inclination`, `--resolution`,
`--impact-parameter`, ...), prints section-29-style panels via
`reporting/console.py`, and prints only computed numbers.

## 7. Configuration blocks

`utils/config.py` (Phase 0, extended only by the infra role) keeps the six
fixed blocks of PROJECT.md section 34 with these additions, each with a
documented default so that existing YAML files still load (the infra role
also states them explicitly in every `configs/*.yaml`):

```yaml
black_hole:   {mass: 1.0, spin: 0.0}
observer:     {radius: 1000.0, inclination_deg: 60.0, phi_deg: 0.0}
integration:  {method: rk45, rtol: 1e-9, atol: 1e-11, max_steps: 100000,
               step_size: 0.01, lambda_max: 10000.0}          # method also accepts dop853
raytrace:     {resolution: 64, fov: 12.0, backend: numpy, dtype: float64}
termination:  {horizon_epsilon: 1e-6, escape_radius: 1000.0}
experiment:   {name: shadow, description: "...", seed: 0, output_dir: runs, report_dir: reports,
               parameters: {...}}
```

Experiment-specific settings live under `experiment.parameters` (decision
D-004): a free-form mapping that the *owning experiment module* parses into
its own frozen dataclass with `kerrray.utils.config_parsing.parse_block`
(unknown keys inside a sub-block are errors). The sub-block names and their
defaults are fixed here so that the YAML files and the experiment drivers
agree:

```yaml
experiment:
  parameters:
    lensing:          {impact_min: 3.0, impact_max: 20.0, n_rays: 40, launch_radius: 1000.0}
    convergence:      {resolutions: [64, 128, 256, 512], tolerances: [1e-6, 1e-8, 1e-10, 1e-12],
                       step_sizes: [0.1, 0.01, 0.001, 0.0001]}    # 1024 is opt-in via CLI override; log what was skipped
    near_critical:    {offsets: [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8], launch_radius: 1000.0}
    frame_dragging:   {spins: [-0.9, 0.0, 0.9], impact_parameter: 6.0, launch_radius: 1000.0}
    spin_sweep:       {spins: [0.0, 0.25, 0.5, 0.75, 0.9, 0.99]}
    inclination_sweep: {inclinations_deg: [0, 30, 45, 60, 75, 90]}
    disk:             {enabled: false, r_in: null, r_out: 20.0, emissivity_index: 3.0, intensity_law: g4, prograde: true}
    benchmark:        {ray_counts: [1000, 10000, 100000, 1000000], backends: [numpy, numba],
                       dtypes: [float32, float64], repeats: 3}
```

Helper: `kerrray.utils.config.experiment_block(cfg, "lensing", LensingParams)`
(infra role) returns the parsed dataclass or its defaults when the sub-block
is absent.

## 8. Testing and performance rules for parallel work

- Every module ships tests in `tests/test_<module>.py`; the full suite must run
  in under about 5 minutes on 8 CPU cores, so tests use small resolutions
  (<= 32x32), short rays, and loose but meaningful tolerances. Long runs
  belong in experiments, not tests.
- No test may hard-code an expected physical result except the documented
  reference values it *checks against* (3M, 3 sqrt(3) M, BPT closed forms),
  and those must be computed by the code under test, never printed from the
  reference.
- Tests must not depend on the working directory or on `runs/` contents; use
  `tmp_path`.
- Do not import SymPy on any hot path; symbolic checks live in `symbolic.py`
  and in tests.
- Numba is installed and is the only additional dependency allowed beyond the
  Phase 0 set; it is optional at import time (`raytracing/backends` must
  import cleanly without it, reporting the backend as unavailable).
- No file over about 400 lines; split instead.
- Development default resolution is 64x64. Shadows at 256 and 512 are
  experiment runs, not defaults; 1024 is opt-in.

## 9. Scope amendment (2026-09-28): CPU first, GPU optional future work

PROJECT.md sections 25 to 27, Phase 8 of section 49 and EXP-010 treat GPU
acceleration as a core deliverable. For this build GPU acceleration is
**optional future work**, for three reasons: the development machine has no
CUDA device, so any GPU numbers would be unverifiable; the scientific content
of the project does not depend on it; and commodity-hardware numerical
efficiency is the better research angle. Phase 8 is therefore redefined as
**CPU performance and precision**:

- Backends: `numpy` (vectorised batched integrator, the reference) and
  `numba` (CPU-parallel, `numba.njit(parallel=True)`, same tableau and RHS,
  must agree with `numpy` to floating-point round-off). `cuda` is a named but
  unavailable backend; `kerrray benchmark gpu` prints that GPU acceleration is
  optional future work and exits 0 without inventing any number.
- CPU benchmark (`benchmarks/cpu.py`, EXP-009): 1,000, 10,000, 100,000 and
  1,000,000 rays; measure wall-clock runtime, peak memory (tracemalloc plus
  array accounting; no psutil), max null-constraint error, max energy drift,
  trajectory error against a tight-tolerance reference, and, per resolution,
  shadow-boundary convergence.
- Precision study (`benchmarks/precision.py`, EXP-011): float32 vs float64 on
  the CPU backends: null error, conserved-quantity drift, shadow-boundary
  difference, runtime.
- Guiding research question recorded in `docs/scientific_background.md` and
  the README: *How do numerical integration tolerances and ray density affect
  the accuracy of Kerr shadow reconstruction on CPU-based systems?* The
  tolerance x resolution grid of EXP-008/EXP-009 is the experiment that
  addresses it.
- `CLAUDE.md` Phase 8 line reads "Phase 8: CPU performance and precision (GPU
  optional future work)". PROJECT.md is not rewritten; this amendment is
  recorded in `docs/decisions.md` and referenced from the README roadmap:
  Kerr physics -> photon geodesics -> numerical validation -> CPU ray tracing
  -> shadow reconstruction -> numerical error analysis -> strong-field
  lensing -> scientific experiments -> research, with GPU as an optional
  branch.
