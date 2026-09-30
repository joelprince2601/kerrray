# Scientific Background

## Purpose

KerrRay numerically integrates photon null geodesics through the curved
spacetime of a rotating Kerr black hole (PROJECT.md section 0). This document
records the scientific questions the project addresses, the roadmap that
follows from them, the physical setting the code assumes (the Kerr solution,
the coordinate and unit conventions, the horizon and ergosphere structure,
the photon equations of motion) and the formulation chosen for numerical
integration (PROJECT.md sections 2, 4 to 10). It exists so that every
equation in the code can be traced to a documented convention and a trusted
reference, never written from memory (PROJECT.md section 47). The equations
themselves, with their references and implementation mapping, live in
`docs/equations.md`; the numerical schemes and their measured behaviour in
`docs/numerical_methods.md`.

## Primary scientific question

PROJECT.md section 2 states the core question:

> How accurately and efficiently can photon null geodesics be numerically
> integrated in rotating Kerr spacetime, and how does numerical error affect
> observable features such as photon capture boundaries, lensing and
> black-hole shadow structure?

Secondary questions (section 2): how Kerr spin alters photon trajectories;
how frame dragging affects photon motion; how observer inclination affects
the apparent shadow; how adaptive integration compares with fixed-step
integration; how numerical precision affects near-critical trajectories; how
parallelisation affects ray-tracing performance; and how the numerical
shadow boundary converges as resolution and solver tolerance change.

## CPU research question

GPU acceleration is optional future work for this build
(`docs/decisions.md`, D-007; `docs/architecture.md` section 9). The
performance part of the primary question is therefore posed for commodity
hardware:

> How do numerical integration tolerances and ray density affect the
> accuracy of Kerr shadow reconstruction on CPU-based systems?

The tolerance x resolution grid of EXP-008 and EXP-009
(`configs/convergence.yaml`, `configs/benchmark.yaml`) is the experiment
that addresses it: for each adaptive tolerance and each image resolution the
shadow boundary is extracted and compared with the analytic Bardeen curve and
with a tight-tolerance reference, while runtime, peak memory, null-constraint
error and conserved-quantity drift are measured on the `numpy` and `numba`
CPU backends (EXP-011 repeats the measurements in float32 and float64). Every
number reported for these questions is computed by a run whose manifest
records the configuration, seed and environment (PROJECT.md section 35).

## Roadmap

The phases of PROJECT.md section 49 are followed in order, with Phase 8
redefined (D-007):

1. Kerr physics: metric, inverse metric, horizons, ergosphere, symbolic
   verification (Phase 1).
2. Photon geodesics: state vector, null condition, Hamiltonian equations of
   motion, RK4 / RK45 integrators, termination, conservation diagnostics
   (Phase 2).
3. Numerical validation: Schwarzschild photon sphere and critical impact
   parameter, then Kerr limits, frame dragging and near-extremal behaviour
   (Phases 3 and 4).
4. CPU ray tracing: camera, ZAMO tetrad, batched integration on the `numpy`
   and `numba` backends (Phase 5).
5. Shadow reconstruction: capture/escape classification, sub-pixel boundary
   extraction, comparison with the analytic curve (Phase 5).
6. Numerical error analysis: solver comparison, tolerance and step-size
   sweeps, resolution convergence, near-critical sensitivity, float32 versus
   float64 (Phases 6 and 8).
7. Strong-field lensing, redshift and the optional thin disk (Phase 7).
8. Scientific experiments EXP-001 to EXP-009 and EXP-011 with reports
   (Phase 9).
9. Research: the CPU research question above and the future questions of
   PROJECT.md section 38 (Phase 10).

GPU acceleration (EXP-010, PROJECT.md sections 25 and 26) is an optional
branch off step 4 that reuses the backend protocol; it is attempted only when
a CUDA device is available and every claimed speed-up is measured.

## Physical setting (outline)

The following items are written by the geometry and geodesics work as the
corresponding equations are verified; each entry in `docs/equations.md`
records definition, coordinate convention, unit convention, reference and
implementation mapping (PROJECT.md section 43).

1. Kerr spacetime and the dimensionless spin `a_* = a / M`, `|a_*| < 1`
   (section 4.1). `a > 0` means rotation towards `+phi`; a photon is
   prograde when `a L_z > 0`.
2. Units: `G = c = 1`, lengths and times in units of `M` with `M = 1` by
   default (section 33). SI conversions for reporting only:
   `kerrray.utils.units.GeometricUnits(mass_solar)` with CODATA 2018 `G`,
   the exact SI `c` and the IAU 2015 nominal `GM_sun`; the engine never uses
   SI values.
3. Boyer-Lindquist coordinates `(t, r, theta, phi)`, signature
   `(-, +, +, +)`, `Sigma = r^2 + a^2 cos^2(theta)`,
   `Delta = r^2 - 2 M r + a^2` (section 5).
4. Event horizons `r_+- = M +- sqrt(M^2 - a^2)`; capture at
   `r <= r_+ + horizon_epsilon` (section 6).
5. Ergosphere `r_E(theta) = M + sqrt(M^2 - a^2 cos^2(theta))` (section 7).
6. Photon dynamics: null geodesics parametrised by an affine parameter
   `lambda`, integrated forwards; backward ray tracing reverses the momentum
   (section 8).
7. Integration formulation: see *Formulation choice* below (section 9).
8. Conserved quantities `E = -p_t`, `L_z = p_phi`, the Carter constant `Q`
   and the null constraint `H = 0`; drift diagnostics defined as relative
   deviations from the launch values (section 10).
9. Approximations, each stated as WHAT / WHY / LIMITATION (section 43): the
   observer-axis clamp (D-008); finite observer and escape radii in place of
   infinity (D-002, `docs/raytracing.md`); the weak-field deflection `4M/b`
   used only as a labelled comparison (`physics/lensing.py`).
10. References: the sources each equation is verified against (below).

## Formulation choice

PROJECT.md section 9 asks for an evaluation of two formulations of the
photon equations of motion and forbids choosing purely for convenience.

**Option A: second-order geodesic equation.** State
`z = [x^mu, u^mu]` with `u^mu = dx^mu / dlambda`, evolved by
`du^mu / dlambda = -Gamma^mu_{alpha beta} u^alpha u^beta`. It needs the
Christoffel symbols (40 independent components in Kerr, verified
symbolically in `geometry/symbolic.py`), and the constants of motion `E`,
`L_z` and `Q` are *derived* quantities that the integrator does not preserve
by construction, so their drift measures the total integration error.

**Option B: first-order Hamiltonian form.** State
`y = [x^mu, p_mu]` with `H = (1/2) g^{mu nu} p_mu p_nu`, evolved by
`dx^mu / dlambda = g^{mu nu} p_nu` and
`dp_mu / dlambda = -(1/2) (d_mu g^{alpha beta}) p_alpha p_beta`. Because the
Kerr metric depends on neither `t` nor `phi`, `dp_t / dlambda` and
`dp_phi / dlambda` vanish identically: `E = -p_t` and `L_z = p_phi` are
constant in the discrete equations as well (up to round-off), which is the
statement that the Killing vectors `d_t` and `d_phi` yield conserved
momenta (Misner, Thorne & Wheeler 1973, section 25.2; Carter 1968). Only the
`r` and `theta` derivatives of the five non-zero `g^{mu nu}` are needed. The
null condition `H = 0` and the Carter constant remain non-trivial checks of
the integration.

**Choice.** KerrRay integrates Option B as its primary formulation and keeps
Option A as a cross-check (`geodesics/equations.py`:
`geodesic_rhs` and `geodesic_rhs_christoffel`). The reasons that do not
depend on measurements: two of the four conserved quantities are exact in
Option B, so the remaining diagnostics (`Q` and `H`) isolate the error in
the `(r, theta)` sub-system; the right-hand side needs 10 metric derivatives
instead of 40 Christoffel components, which matters for the batched CPU
backends; and the covariant momentum `p_mu` is what the camera and the
redshift calculation use directly (`docs/raytracing.md`). The reasons that
do depend on measurements, namely stability and error growth per unit cost
on the same trajectories with both formulations, are recorded with the
numbers that support them in `docs/numerical_methods.md` (numerical-methods
section "Formulation comparison"); the choice stands only if those
measurements do not contradict it, and the cross-check between the two
formulations is a permanent test of the Phase 2 suite.

## References

Sources the geometry and geodesic equations are verified against (each
implementation cites the one it used):

- Bardeen, J. M., Press, W. H. & Teukolsky, S. A. 1972, ApJ 178, 347
  (Kerr metric, circular photon orbits, ISCO).
- Bardeen, J. M. 1973, in *Black Holes (Les Houches 1972)*, eds. DeWitt &
  DeWitt (shadow of a Kerr black hole).
- Carter, B. 1968, Phys. Rev. 174, 1559 (separability, the fourth constant).
- Chandrasekhar, S. 1983, *The Mathematical Theory of Black Holes*, ch. 6
  and 7.
- Cunha, P. V. P. & Herdeiro, C. A. R. 2018, Gen. Rel. Grav. 50, 42 (shadow
  review, observer conventions).
- Dormand, J. R. & Prince, P. J. 1980, J. Comput. Appl. Math. 6, 19 (RK5(4)
  tableau); Hairer, Norsett & Wanner 1993, *Solving Ordinary Differential
  Equations I*, Table 5.2.
- Misner, C. W., Thorne, K. S. & Wheeler, J. A. 1973, *Gravitation*,
  sections 25.2 and 33.2.
- Visser, M. 2007, arXiv:0706.0622, "The Kerr spacetime: a brief
  introduction".
- Tiesinga et al. 2021, Rev. Mod. Phys. 93, 025010 (CODATA 2018 constants);
  Prsa et al. 2016, AJ 152, 41 (IAU 2015 nominal solar values).

## Status

Scientific questions, roadmap and formulation framing recorded (2026-09-28).
Items 1 to 8 of the physical setting are filled in as the geometry and
geodesics work verifies each equation; the measured formulation comparison
is written in `docs/numerical_methods.md`.
