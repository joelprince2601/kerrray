# Numerical Methods

## Purpose

Describes how KerrRay integrates null geodesics numerically and how it
measures numerical error: the integration schemes, step-size and tolerance
control, termination conditions, conservation diagnostics and the convergence
methodology behind the numerical-analysis experiments (PROJECT.md sections 9,
11, 19 to 21 and 27). It serves the project's primary scientific question: how
numerical error affects observable features such as photon capture boundaries,
lensing and black-hole shadow structure (section 2). Items 1 to 5 and 9 are
written in Phase 2 (this version); items 6 to 8 are added in Phase 6.

## 1. Formulation: Hamiltonian versus Christoffel (PROJECT.md section 9)

Both formulations of `docs/equations_geodesics.md` are implemented in
`src/kerrray/geodesics/equations.py`. The first-order Hamiltonian form
(state `y = [x^μ, p_μ]`) is the production formulation; the second-order
Christoffel form (state `z = [x^μ, u^μ]`) is kept as an independent
cross-check. The choice rests on numerical grounds, not convenience:

| Criterion | Hamiltonian `dx = g^{μν} p_ν`, `dp = −½ ∂g^{αβ} p_α p_β` | Christoffel `du = −Γ u u` |
|---|---|---|
| Exact conservation in the ODE | `dp_t/dλ = dp_φ/dλ = 0` identically: E and L_z are constant to round-off (drift 0 measured; asserted ≤ 10⁻¹³) | E = −g_{tμ} u^μ and L_z = g_{φμ} u^μ are conserved only up to the truncation error of the scheme |
| Right-hand side cost | 5 inverse-metric components and 10 derivatives (closed forms), ~45 multiply-adds | 64 Christoffel components (40 independent) contracted with u u: ~130 multiply-adds plus the (4,4,4) assembly |
| Constraint monitoring | H = ½ p_μ dx^μ/dλ is available from the state and the derivative already computed, no extra metric evaluation | g_{μν} u^μ u^ν needs a metric evaluation |
| Order of the system | First order by construction, 8 equations | Second order rewritten as 8 first-order equations; same size |
| Stiffness near the horizon | p_r ~ 1/Δ diverges (section 4) | u^r stays finite but Γ^r_{rr} ~ 1/Δ; both forms need small steps there |

The two forms are checked against each other by integrating the same Kerr
ray (a = 0.9, off-equatorial) with SciPy's DOP853 at rtol = 10⁻¹² in both
formulations and converting `p_μ = g_{μν} u^ν` at the end point: the
positions and momenta agree to 9 × 10⁻¹² (`tests/test_geodesics.py`), well
inside the 10⁻⁷ target. PROJECT.md asks for the choice to be recorded in
`docs/scientific_background.md` as well; that file is owned by the
integration role and should cite this section.

The Hamiltonian right-hand side is verified against a SymPy differentiation
of H at random states for spins 0, 0.9 and −0.6 (relative agreement
5 × 10⁻¹⁶, asserted 10⁻¹⁰).

## 2. State vector, affine parameter and integration schemes

State `y = [t, r, θ, φ, p_t, p_r, p_θ, p_φ]` (`docs/architecture.md` section
1), affine parameter λ in units of M starting at 0 and increasing. Backward
ray tracing reverses `p_μ` and still integrates forward in λ
(`docs/equations_geodesics.md` section 6).

Three schemes are selected by `integration.method`
(`src/kerrray/geodesics/integrators.py`, tableaus in `tableaus.py`):

* `rk4`: classical fourth-order Runge-Kutta with the fixed step
  `integration.step_size` (Kutta 1901; Hairer, Norsett and Wanner 1993,
  *Solving Ordinary Differential Equations I*, Table 1.2). Four right-hand-
  side evaluations per step; the evaluation at the new state is reused as
  the first stage of the next step, so the classifier and the null-constraint
  monitor cost nothing extra.
* `rk45`: the embedded Dormand-Prince RK5(4)7M pair (Dormand and Prince
  1980, J. Comput. Appl. Math. 6, 19; Hairer, Norsett and Wanner 1993, Table
  5.2). The fifth-order solution is propagated (local extrapolation), the
  fourth-order one gives the error estimate, and the first-same-as-last
  property makes a step cost six evaluations. The tableau (nodes `C`,
  matrix `A`, weights `B`, error weights `E = b̂ − b`) is compared exactly
  with `scipy.integrate.RK45.C/A/B/E` in `tests/test_integrators.py`;
  additionally Σb = 1, ΣE = 0, row sums of `A` equal the nodes and the FSAL
  row equals `b`. `integration.step_size` is the initial step.
* `dop853`: `scipy.integrate.solve_ivp(method="DOP853")` (Hairer's 8(5,3)
  Dormand-Prince code) with terminal events for the termination rules.
  This is the reference solver for convergence studies and is scalar only
  (`integrate_batch` raises for it). SciPy has no step cap, so `max_steps`
  is not enforced on this path (`lambda_max` is its budget) and the
  rejected-step count is reported as 0 because SciPy does not expose it.

Measured behaviour (`tests/test_integrators.py`): the RK4 end-point error
against a DOP853 reference (rtol 10⁻¹³) on a Schwarzschild ray from r = 20
M to λ = 30 M decreases with slope 4.008 in log(error) versus log(h) over
h = 0.5 … 0.03125 (asserted in [3.5, 4.5]); the RK45 error and step count
are monotone in rtol from 10⁻⁵ to 10⁻¹¹.

## 3. Error control (`rk45`)

Hairer, Norsett and Wanner 1993, section II.4, eqs. (4.10)-(4.13):

    sc_i = atol + rtol · max(|y_i|, |y_new,i|)
    err  = sqrt( (1/8) Σ_i (e_i / sc_i)² )          (e = h Σ_k E_k k_k, the embedded estimate)
    accept if err ≤ 1
    h_new = h · min(facmax, max(facmin, fac · err^(−1/5)))     fac = 0.9, facmin = 0.2, facmax = 5

The exponent is −1/(q+1) with q = 4 the order of the error-estimating
method. After a rejected step the factor is capped at 1 (no growth), as the
same section recommends. `err = 0` gives `facmax`. A step whose result or
error estimate is non-finite fails the ray immediately
(`NUMERICAL_FAILURE`), as does a step size below `h_min` after a rejection.
The last step is clipped to `lambda_max − λ` so that the budget is met
exactly, and the clipped step does not feed the controller.

Because `dp_t/dλ = dp_φ/dλ = 0` exactly, the p_t and p_φ components never
contribute to the error norm; the cyclic coordinates t and φ do (their scale
`rtol·|t|` is large at t ~ 10³, which mostly loosens their own control).

## 4. Termination and classification (PROJECT.md section 11)

`src/kerrray/photons/classification.py` applies the rules of
`docs/architecture.md` section 1 to the initial state and after every
accepted step, with derivative `dY/dλ` taken from the FSAL stage:

| State | Rule |
|---|---|
| `NUMERICAL_FAILURE` | any non-finite component of the state or of dY/dλ; adaptive step below `h_min` |
| `OUT_OF_DOMAIN` | r < 0; θ outside [0, π] by more than 10⁻⁹ rad; `L_z ≠ 0` with `|sin θ| < 10⁻¹²` |
| `CAPTURED` | r ≤ r₊ + `horizon_epsilon` |
| `ESCAPED` | r ≥ `escape_radius` **and** dr/dλ > 0 (D-002: never at an inward launch, tested) |
| `MAX_AFFINE_PARAMETER` | λ ≥ `lambda_max` or accepted steps ≥ `max_steps` |
| `DISK_HIT` | batched disk-plane event only (section 6) |

When several rules hold the list order is the priority (a non-finite state
is a failure whatever its radius; a state outside the coordinate domain is
reported as such even if r ≤ r₊ + ε, so that a fixed-step overshoot through
the horizon is visible as a numerical event rather than silently counted as
a capture; see the limitation below). The `dop853` path realises the same
rules as SciPy terminal events with directions: `r − (r₊ + ε)` downward,
`r − escape_radius` upward (which is exactly "outward motion"), and the two
θ bounds.

**Polar axis.** g^{φφ} ~ 1/sin²θ. For rays with exactly L_z = 0 every
occurrence of g^{φφ} multiplies p_φ = 0 and the equations set those products
to zero, so such rays (e.g. launched on the axis) integrate without 0·∞;
if they cross the axis, θ leaves [0, π] and the ray ends `OUT_OF_DOMAIN`
(the continuation through the axis would be θ → −θ, φ → φ + π; not
implemented because the camera never samples L_z = 0 exactly). Rays with
L_z ≠ 0 cannot reach the axis analytically (Θ(θ) < 0 there), so a state
within |sin θ| < 10⁻¹² of the axis is a numerical artefact and is
`OUT_OF_DOMAIN`.

**Limitation: Boyer-Lindquist coordinates near the horizon.** As r → r₊,
Δ → 0 and the ingoing photon's p_r = Σ (dr/dλ)/Δ diverges like 1/Δ while
dp_r/dλ ~ 1/Δ²; the coordinate time t also diverges logarithmically. With
`horizon_epsilon = 10⁻⁶` the ray is followed to Δ ≈ 2 × 10⁻⁶ (M = 1, a = 0),
where p_r ~ 5 × 10⁵. The adaptive scheme resolves this by shrinking the
step in proportion to Δ (about 150 extra accepted steps for a Schwarzschild
plunge at rtol 10⁻⁹, i.e. roughly doubling the cost of a captured ray
relative to an escaping one), and the null error |H|/E² of a captured ray
is dominated by the cancellation between the ~1/Δ terms of H at the last
steps (7 × 10⁻³ at rtol 10⁻⁹ for b = 4 from r₀ = 1000, versus 10⁻⁹ along
escaping rays). A larger `horizon_epsilon` (10⁻⁴ … 10⁻³) removes most of
that cost and error at no physical price, because no turning point exists
near the horizon (`docs/equations_geodesics.md` section 4): every inward-
moving ray that close is captured. The fixed-step `rk4` cannot resolve the
approach: its last step overshoots into Δ ≤ 0, where the stage evaluations
are unphysical, and the ray typically ends `OUT_OF_DOMAIN` (r < 0) or
`NUMERICAL_FAILURE` rather than `CAPTURED`; with h = 0.05 … 0.5 this was
observed on every plunging ray. Consumers comparing solvers should count
these outcomes as the failure rate of the fixed-step scheme (PROJECT.md
section 19), and shadow extraction with `rk4` should treat them as
non-escaping. Horizon-penetrating coordinates (Kerr-Schild) would remove
the singularity but are outside the scope fixed by `docs/architecture.md`.

## 5. Conservation diagnostics (PROJECT.md section 10)

At every accepted step the integrators evaluate E = −p_t, L_z = p_φ, the
Carter constant Q and the null constraint H = ½ p_μ dx^μ/dλ and keep the
running maxima of

    ε_E = |E − E₀| / max(|E₀|, floor_E),   ε_L = |L_z − L_z,0| / max(|L_z,0|, floor_L),
    ε_Q = |Q − Q₀| / max(|Q₀|, floor_Q),   ε_H = |H| / E₀²

with floors `floor_E = |E₀|`, `floor_L = |E₀| M`, `floor_Q = E₀² M²` (the
scales of the constants of an E₀ photon; a reference below its floor is
compared absolutely in units of the floor: equatorial rays have Q₀ = 0
exactly and would otherwise divide by zero). `relative_drift(values,
reference, floor=1.0)` is the generic helper. For the scalar integrator the
full arrays along the recorded trajectory are also available
(`ConservationDiagnostics`); the batched integrator returns the per-ray
maxima (`BatchResult`). Measured on the test rays (rtol 10⁻⁹, atol 10⁻¹¹):
ε_E = ε_L = 0 exactly; ε_Q = 8.8 × 10⁻¹⁰ and ε_H = 7.5 × 10⁻¹⁰ for an
off-equatorial a = 0.9 ray from r₀ = 30 M (asserted < 10⁻⁷ and < 10⁻⁸);
ε_H = 1.3 × 10⁻⁹ along the b = 6 Schwarzschild ray from r₀ = 1000 M.

## 6. Batched integration design

`integrate_batch` (`src/kerrray/geodesics/integrators_batch.py`) advances
all active rays by one step attempt per loop iteration; every array
operation is over the rays and there is no Python loop over rays. Per
iteration: gather the active rays; clip their step to the affine budget;
take one step with the same tableau code as the scalar path (`h` broadcast
as a column); evaluate the error norm per ray; accept or reject each ray
individually and update its own step size and its own "rejected before"
flag; scatter the accepted states, derivatives, λ and step counts back;
update the running diagnostics; classify the accepted rays
(`classify_batch`); apply the optional disk event; drop terminated rays
from the active mask. Fixed-step `rk4` uses the same loop with the error
test disabled. The loop is bounded by a proven attempt count
(`max_attempts`): each rejection multiplies h by at most 0.9, each accepted
step by at most 5, and h ≤ max(lambda_max, step_size), so the number of
attempts before h < h_min is finite.

Agreement with the scalar integrator: identical inputs give identical step
sequences up to the last-ulp differences of NumPy's SIMD transcendental
kernels on long arrays versus scalars; on rays that are not near-critical
the final states agree to < 10⁻⁹ relative (asserted), and to 0 on most.
Near-critical rays (b within a few per cent of the capture threshold,
circling near the photon sphere) amplify ulp differences exponentially, a
property of the physics that Phase 6 (near-critical sensitivity) studies.

**Disk-plane event** (`EventOptions(disk_plane=True, r_in, r_out)`): a
θ = π/2 crossing between two accepted states is located by linear
interpolation in λ (approximation: WHAT, linear interpolant of the crossing
state; WHY, the renderer needs only the crossing and the interpolant is
consistent with small steps; LIMITATION, O(h²) error in the crossing state,
not re-projected onto the null cone, and a ray lying exactly in the plane
never triggers). If the interpolated radius lies in [r_in, r_out] the ray
stops as `DISK_HIT` with the crossing state as its final state; otherwise
it continues. A crossing detected in the same step as a capture or escape
wins, because the crossing happened earlier along the step.

Throughput measured on the development machine (8 logical CPUs, NumPy
2.5, float64, 10 000 Kerr a = 0.9 rays with random off-equatorial null
momenta from r₀ = 100 M, 200 accepted steps each, best of 3 runs):
`rk45` 1.56 × 10⁵ accepted ray-steps per second (1.75 × 10⁵ attempted,
11 % rejected at rtol 10⁻⁸), `rk4` 2.72 × 10⁵ ray-steps per second;
float32 `rk4` 3.23 × 10⁵. The right-hand side (two geometry evaluations in
closed form plus the products) dominates: 0.62 µs per ray per evaluation
at N = 10⁴, six evaluations per RK45 step. These are the numbers of the
NumPy reference backend before any optimisation (PROJECT.md section 42);
the numba backend of Phase 8 targets the same loop.

## 7. Precision (float32 versus float64; PROJECT.md section 27)

`IntegratorOptions.dtype` selects the working precision of the batched
path: the states, the derivatives, the Runge-Kutta arithmetic, the step
sizes and the error control are all float32 or all float64. The affine
parameter and the diagnostics are accumulated in float64 (the float32
spacing at λ ~ 10³ is 6 × 10⁻⁵, larger than typical steps near the hole).
The geometry functions of `kerrray.geometry` evaluate in float64 by
construction and their outputs are cast to the working dtype *before* they
are combined with the momenta, so the float32 path is float32 everywhere
except inside the metric closed forms (integration request: dtype-
preserving geometry functions would make it float32 throughout).

Measured (a = 0.9 off-equatorial rays from r₀ = 200 M to λ = 300 M,
rtol 10⁻⁷, atol 10⁻⁹, against a float64 rtol 10⁻¹² reference): float64
end-point errors 1 … 3 × 10⁻⁷ relative, float32 3 … 40 × 10⁻⁷; null error
1.3 × 10⁻⁷ versus 3 … 6 × 10⁻⁷; Carter drift 1.5 × 10⁻⁷ versus 5 … 13 × 10⁻⁷.
At rtol 10⁻⁵ the two precisions are indistinguishable (truncation error
dominates); the crossover is near rtol 10⁻⁶.

Limitations of float32: (i) `horizon_epsilon = 10⁻⁶` relative to r₊ ~ 1 is
about 17 ulps of float32, and the ~1/Δ growth of p_r near the horizon
overflows or loses all digits well before that: in the tests, plunging
rays at rtol ≤ 10⁻⁶ end `NUMERICAL_FAILURE` (step underflow after
thousands of rejections) or, at rtol 10⁻⁵, `CAPTURED` with |H|/E² ~ 10⁶.
float32 runs should use `horizon_epsilon ≳ 10⁻³` and rtol ≳ 10⁻⁵ (Phase 8
precision study). (ii) Tolerances below the float32 round-off (~10⁻⁷ per
operation) cannot be met: the error estimate saturates at round-off, every
step is rejected and the ray fails at h < h_min.

## 8. Recommendations for the later phases

* Default: `rk45`, rtol 10⁻⁹, atol 10⁻¹¹ (configs), `horizon_epsilon`
  10⁻⁶ as specified; consider 10⁻⁴ for large shadow runs (section 4).
* Solver comparison (section 19): use `integrate` for trajectory errors
  against `dop853` and `integrate_batch` for runtime; count `OUT_OF_DOMAIN`
  and `NUMERICAL_FAILURE` as failures.
* Step-size study (section 20): a fixed step h needs ≈ λ_total/h steps;
  rays from r₀ = 1000 M need λ ≈ 1000 M just to reach the hole, so
  `max_steps` must be raised accordingly (h = 10⁻⁴ means > 10⁷ steps).

## References

* Dormand, J. R. and Prince, P. J. 1980, J. Comput. Appl. Math. 6, 19.
* Hairer, E., Norsett, S. P. and Wanner, G. 1993, *Solving Ordinary
  Differential Equations I: Nonstiff Problems*, 2nd ed., Springer:
  Table 1.2 (RK4), Table 5.2 (Dormand-Prince 5(4)), section II.4
  (step-size control, eqs. 4.10-4.13), section II.5 (DOP853 is described in
  the same book's DOP853 code, used through SciPy).
* Kutta, W. 1901, Z. Math. Phys. 46, 435 (classical RK4).
* SciPy 1.18 `scipy.integrate.solve_ivp`, `RK45`, `DOP853`.

Section and table numbers were not re-read from the printed books in this
offline environment; the tableau itself is verified against SciPy.
