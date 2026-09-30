"""Kerr validation (PROJECT.md section 13; docs/validation.md section 3).

``run_kerr_validation(cfg)`` runs six checks through
:func:`kerrray.experiments.base.run_experiment` (experiment name
``validate_kerr``) and returns a :class:`~kerrray.validation.ValidationReport`:

1. **horizon**: for the configured spin and every spin used below,
   ``Delta(r_+) = Delta(r_-) = 0``, ``r_+`` against the larger root of the
   quadratic ``r^2 - 2 M r + a^2`` found independently with ``numpy.roots``,
   ``g_tt = 0`` on the ergosphere ``r_E(theta)``, ``r_E(0) = r_+`` and
   ``r_E(pi/2) = 2M`` (Visser 2007, arXiv:0706.0622, sections 2-3).
2. **schwarzschild_limit**: the critical impact parameter ``b_c(a)`` found by
   bracketed classification of integrated rays (:mod:`kerr_bisection`) for
   small spins approaches the Schwarzschild reference ``3 sqrt(3) M``
   (:data:`kerrray.validation.schwarzschild.CRITICAL_IMPACT_REFERENCE`):
   ``|b(a) - 3 sqrt(3) M|`` scales linearly in ``a`` for both senses
   (fitted exponent), the odd part ``[b_pro - b_ret]/2`` is linear, the even
   part ``[b_pro + b_ret]/2 - 3 sqrt(3) M`` is ``O(a^2)`` and must vanish at
   the smallest spin within ``schwarzschild_limit_tol``, and the metric
   difference ``max |g(a) - g(0)|`` scales linearly.
3. **critical_impact**: prograde and retrograde ``b_c`` by the same search
   for the configured spins against
   :func:`kerrray.photons.orbits.critical_impact_parameters` (derived with
   SymPy from ``R = R' = 0`` and matched to Bardeen, Press and Teukolsky
   1972, eq. 2.18; docs/derivations.md section 6); relative error below
   ``critical_b_tol``.
4. **conservation**: drift of ``E``, ``L_z``, ``Q`` and the null constraint
   along long off-equatorial escaping rays (``Q > 0``) at high spins.
5. **near_extremal**: a prograde ray with ``b = b_c (1 + offset)`` at
   near-extremal spins for each ``horizon_epsilon``: outcome, steps, drift,
   runtime, closest approach, turns, and the bound ``r_turn - r_+`` above
   which a capture margin would misclassify the ray.
6. **reversibility**: integrate forward, reverse ``p_mu``, integrate back
   (docs/architecture.md section 1) and measure the distance to the start.

Every reference value except ``3 sqrt(3) M`` is computed at run time by the
geometry and ``photons.orbits`` code (never stored); tolerances come from
``experiment.parameters.validation`` (:class:`KerrValidationParams`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Final

import numpy as np

from kerrray.experiments.base import ExperimentContext, run_experiment
from kerrray.geodesics import IntegratorOptions, TerminationOptions, integrate, photon_from_constants
from kerrray.geometry import Spacetime, delta, ergosphere_radius, horizon_radii, kerr, metric_components, schwarzschild
from kerrray.photons import TerminationState, Trajectory
from kerrray.photons.orbits import critical_impact_parameters, equatorial_photon_orbit_radius
from kerrray.physics.frame_dragging import turning_point_radius
from kerrray.utils.config import KerrRayConfig
from kerrray.utils.manifest import EnvironmentInfo
from kerrray.validation import (
    ValidationCheck,
    ValidationReport,
    integrator_options_from_config,
    termination_options_from_config,
)
from kerrray.validation.conservation import (
    drift_summary,
    off_equatorial_ray,
    reversibility_roundtrip,
    trajectory_row,
    worst_drift,
)
from kerrray.validation.convergence import loglog_order
from kerrray.validation.kerr_bisection import CriticalImpactBisection, bisect_critical_impacts
from kerrray.validation.schwarzschild import CRITICAL_IMPACT_REFERENCE

__all__ = ["BRACKET_FRACTION", "EXPERIMENT_NAME", "KerrValidationParams", "check_conservation",
           "check_critical_impact", "check_horizon", "check_near_extremal", "check_reversibility",
           "check_schwarzschild_limit", "run_kerr_validation"]

EXPERIMENT_NAME: Final[str] = "validate_kerr"
BRACKET_FRACTION: Final[float] = 0.1
"""The bracket is shrunk to ``BRACKET_FRACTION * critical_b_tol * M`` so that the
midpoint error (half the width) stays well below the pass tolerance."""


@dataclass(frozen=True)
class KerrValidationParams:
    """``experiment.parameters.validation`` of ``configs/kerr.yaml`` plus documented extensions.

    The first seven keys are the D-009 block (defaults equal to the shipped
    file); the others are documented in docs/validation.md section 7.
    """

    horizon_tol: float = 1.0e-12
    small_spins: list[float] = field(default_factory=lambda: [1e-2, 1e-3, 1e-4])
    schwarzschild_limit_tol: float = 1.0e-6
    conservation_tol: float = 1.0e-8
    critical_b_tol: float = 1.0e-6
    bisection_iterations: int = 60
    near_extremal_spins: list[float] = field(default_factory=lambda: [0.99, 0.999])
    critical_spins: list[float] = field(default_factory=lambda: [0.5, 0.9, 0.99])
    conservation_spins: list[float] = field(default_factory=lambda: [0.9, 0.99, 0.999])
    launch_radius: float = 1000.0
    impact_bracket: list[float] = field(default_factory=lambda: [1.0, 10.0])
    probes_per_iteration: int = 8
    scaling_exponent_tol: float = 0.1
    conservation_xi: float = 2.0
    conservation_eta: float = 32.0
    conservation_theta_deg: float = 60.0
    near_extremal_offset: float = 1.0e-3
    horizon_epsilons: list[float] = field(default_factory=lambda: [1e-3, 1e-6])
    reversibility_spin: float = 0.9
    reversibility_affine_fraction: float = 1.9
    reversibility_tol: float = 1.0e-6


def check_horizon(mass: float, spins: list[float], params: KerrValidationParams) -> ValidationCheck:
    """Check 1: horizon radii and ergosphere for each spin (module docstring)."""
    rows, worst = [], 0.0
    theta = np.linspace(0.05, math.pi - 0.05, 9)
    for spin in spins:
        st = kerr(mass, float(spin))
        r_plus, r_minus = horizon_radii(st)
        root_plus = float(np.max(np.roots([1.0, -2.0 * mass, st.a * st.a]).real))
        with np.errstate(divide="ignore", invalid="ignore"):
            g_tt = np.abs(metric_components(st, ergosphere_radius(st, theta), theta).g_tt)
        errors = {
            "delta_at_r_plus": abs(float(delta(st, r_plus))) / mass**2,
            "delta_at_r_minus": abs(float(delta(st, r_minus))) / mass**2,
            "r_plus_vs_quadratic_root": abs(r_plus - root_plus) / root_plus,
            "g_tt_on_ergosphere": float(np.max(g_tt)),
            "r_E_pole_vs_r_plus": abs(float(ergosphere_radius(st, 0.0)) - r_plus) / mass,
            "r_E_equator_vs_2M": abs(float(ergosphere_radius(st, 0.5 * math.pi)) - 2.0 * mass) / mass,
        }
        worst = max(worst, *errors.values())
        rows.append({"spin": float(spin), "r_plus": r_plus, "r_minus": r_minus, "r_plus_quadratic_root": root_plus,
                     "errors": errors})
    return ValidationCheck(
        name="horizon",
        description="Delta(r_+-) = 0, r_+ against the numpy.roots quadratic root, g_tt = 0 on the ergosphere",
        computed={"rows": rows},
        reference={"closed_form": "r_+- = M +- sqrt(M^2 - a^2); r_E = M + sqrt(M^2 - a^2 cos^2 theta)"},
        error={"max": worst},
        tolerance={"horizon_tol": params.horizon_tol},
        passed=worst <= params.horizon_tol,
    )


def _bisect_pair(st: Spacetime, params: KerrValidationParams, integ: IntegratorOptions,
                 term: TerminationOptions) -> tuple[CriticalImpactBisection, ...]:
    lo, hi = params.impact_bracket
    return bisect_critical_impacts(
        st, integ, term, r0=params.launch_radius * st.mass, b_lo=lo * st.mass, b_hi=hi * st.mass,
        tol=BRACKET_FRACTION * params.critical_b_tol * st.mass,
        max_iterations=params.bisection_iterations, n_probe=params.probes_per_iteration,
    )


def _metric_norm(st: Spacetime, st0: Spacetime) -> float:
    r = np.linspace(3.0 * st.mass, 50.0 * st.mass, 48)[:, None]
    th = np.linspace(0.1, math.pi - 0.1, 24)[None, :]
    g, g0 = metric_components(st, r, th), metric_components(st0, r, th)
    return max(float(np.max(np.abs(x - y))) for x, y in zip(g, g0, strict=True))


def check_schwarzschild_limit(mass: float, params: KerrValidationParams, integ: IntegratorOptions,
                              term: TerminationOptions) -> ValidationCheck:
    """Check 2: bisected ``b_c(a)`` for small spins converges to ``3 sqrt(3) M`` linearly in ``a``."""
    b0 = CRITICAL_IMPACT_REFERENCE * mass
    st0 = schwarzschild(mass)
    spins = np.array(sorted(params.small_spins, reverse=True), dtype=np.float64)
    pro, ret, d_pro, d_ret, norms, stats = [], [], [], [], [], []
    for a in spins:
        st = kerr(mass, float(a))
        res_p, res_r = _bisect_pair(st, params, integ, term)
        pro.append(res_p.b_c)
        ret.append(res_r.b_c)
        dp, dr = critical_impact_parameters(st)
        d_pro.append(dp)
        d_ret.append(dr)
        norms.append(_metric_norm(st, st0))
        stats.append({"spin": float(a), "iterations": max(res_p.iterations, res_r.iterations), "n_rays": res_p.n_rays,
                      "width": max(res_p.width, res_r.width), "runtime_s": res_p.runtime_s})
    pro_a, ret_a, d_pro_a, d_ret_a, norms_a = map(np.array, (pro, ret, d_pro, d_ret, norms))
    odd, even = 0.5 * (pro_a - ret_a), 0.5 * (pro_a + ret_a) - b0
    rel_err = np.maximum(np.abs(pro_a - d_pro_a) / d_pro_a, np.abs(ret_a - d_ret_a) / d_ret_a)
    exponents = {"prograde": loglog_order(spins, np.abs(pro_a - b0)),
                 "retrograde": loglog_order(spins, np.abs(ret_a - b0)),
                 "metric_norm": loglog_order(spins, norms_a)}
    two = spins.size >= 2
    slope = float(np.polyfit(spins, odd, 1)[0]) if two else math.nan
    slope_derived = float(np.polyfit(spins, 0.5 * (d_pro_a - d_ret_a), 1)[0]) if two else math.nan
    monotone = bool(np.all(np.diff(norms_a) < 0)) and bool(np.all(np.diff(np.abs(pro_a - b0)) < 0))
    errors = {
        "max_relative_error_vs_derived": float(np.max(rel_err)),
        "even_part_at_smallest_spin": abs(float(even[int(np.argmin(spins))])),
        "exponent_deviation": {k: (abs(v - 1.0) if math.isfinite(v) else math.nan) for k, v in exponents.items()},
        "monotone_decrease": monotone,
    }
    fits_ok = all((not math.isfinite(v)) or v <= params.scaling_exponent_tol
                  for v in errors["exponent_deviation"].values())
    passed = (errors["max_relative_error_vs_derived"] <= params.critical_b_tol
              and errors["even_part_at_smallest_spin"] <= params.schwarzschild_limit_tol and fits_ok and monotone)
    return ValidationCheck(
        name="schwarzschild_limit",
        description="bisected b_c(a) -> 3 sqrt(3) M as a -> 0: linear deviation, O(a^2) even part, metric norm -> 0",
        computed={
            "spins": spins.tolist(), "b_prograde": pro, "b_retrograde": ret,
            "deviation_over_spin_prograde": ((pro_a - b0) / spins).tolist(),
            "deviation_over_spin_retrograde": ((ret_a - b0) / spins).tolist(),
            "odd_part": odd.tolist(), "even_part": even.tolist(), "metric_norm": norms,
            "exponents": exponents, "odd_part_slope": slope, "search": stats,
        },
        reference={"b_c_schwarzschild": b0, "b_prograde_derived": d_pro, "b_retrograde_derived": d_ret,
                   "odd_part_slope_derived": slope_derived, "expected_exponent": 1.0},
        error=errors,
        tolerance={"critical_b_tol": params.critical_b_tol, "schwarzschild_limit_tol": params.schwarzschild_limit_tol,
                   "scaling_exponent_tol": params.scaling_exponent_tol},
        passed=passed,
        notes="exponent checks are skipped (nan) when fewer than two small spins are configured",
    )


def check_critical_impact(mass: float, params: KerrValidationParams, integ: IntegratorOptions,
                          term: TerminationOptions) -> ValidationCheck:
    """Check 3: prograde/retrograde ``b_c`` by bracketed search versus the derived closed forms."""
    rows, worst = [], 0.0
    for spin in params.critical_spins:
        st = kerr(mass, float(spin))
        res_p, res_r = _bisect_pair(st, params, integ, term)
        d_p, d_r = critical_impact_parameters(st)
        e_p, e_r = abs(res_p.b_c - d_p) / d_p, abs(res_r.b_c - d_r) / d_r
        worst = max(worst, e_p, e_r)
        rows.append({
            "spin": float(spin), "b_prograde": res_p.b_c, "b_prograde_derived": d_p, "rel_err_prograde": e_p,
            "b_retrograde": res_r.b_c, "b_retrograde_derived": d_r, "rel_err_retrograde": e_r,
            "bracket_width": max(res_p.width, res_r.width), "iterations": max(res_p.iterations, res_r.iterations),
            "n_rays": res_p.n_rays, "max_steps": res_p.max_steps, "runtime_s": res_p.runtime_s,
            "r_ph_prograde": equatorial_photon_orbit_radius(st, True),
            "r_ph_retrograde": equatorial_photon_orbit_radius(st, False),
        })
    return ValidationCheck(
        name="critical_impact",
        description="prograde and retrograde b_c by bracketed classification versus photons.orbits",
        computed={"rows": rows},
        reference={"source": "kerrray.photons.orbits.critical_impact_parameters (BPT 1972 eq. 2.18 radius)"},
        error={"max_relative_error": worst},
        tolerance={"critical_b_tol": params.critical_b_tol},
        passed=worst <= params.critical_b_tol,
    )


def check_conservation(mass: float, params: KerrValidationParams, integ: IntegratorOptions,
                       term: TerminationOptions) -> tuple[ValidationCheck, Trajectory | None]:
    """Check 4: drifts along long off-equatorial (``Q > 0``) escaping rays at high spins."""
    rows, summaries, escaped, last = [], [], True, None
    for spin in params.conservation_spins:
        st = kerr(mass, float(spin))
        y0 = off_equatorial_ray(st, params.launch_radius * mass, params.conservation_theta_deg,
                                params.conservation_xi * mass, params.conservation_eta * mass**2)
        traj = integrate(st, y0, integ, term, record=True)
        summaries.append(drift_summary(traj))
        escaped = escaped and traj.state == TerminationState.ESCAPED
        rows.append(trajectory_row(traj, spin=float(spin)))
        last = traj
    worst = worst_drift(summaries) if summaries else None
    return ValidationCheck(
        name="conservation",
        description="max drift of E, L_z, Q and |H|/E^2 along off-equatorial (Q > 0) escaping rays",
        computed={"rows": rows, "xi": params.conservation_xi, "eta": params.conservation_eta,
                  "theta_deg": params.conservation_theta_deg},
        reference={"drift": 0.0},
        error={"max_drift": worst.as_dict() if worst else {}, "all_escaped": escaped},
        tolerance={"conservation_tol": params.conservation_tol},
        passed=escaped and worst is not None and worst.within(params.conservation_tol),
    ), last


def check_near_extremal(mass: float, params: KerrValidationParams, integ: IntegratorOptions,
                        term: TerminationOptions) -> tuple[ValidationCheck, Trajectory | None]:
    """Check 5: near-extremal prograde rays just above ``b_c`` and their ``horizon_epsilon`` sensitivity.

    Pass: at the smallest margin every ray escapes, no ray fails or leaves the
    domain, and the ``Q`` and null drifts stay within ``conservation_tol``.
    """
    eps_list = sorted(params.horizon_epsilons, reverse=True)
    rows, passed, worst, last = [], True, 0.0, None
    for spin in params.near_extremal_spins:
        st = kerr(mass, float(spin))
        b_c = critical_impact_parameters(st)[0]
        b = b_c * (1.0 + params.near_extremal_offset)
        r_plus = horizon_radii(st)[0]
        r_turn = turning_point_radius(st, b, 0.0)
        runs = []
        for eps in eps_list:
            y0 = photon_from_constants(st, params.launch_radius * mass, 0.5 * math.pi, 1.0, b, 0.0,
                                       sign_r=-1, sign_theta=1)
            traj = integrate(st, y0, integ, replace(term, horizon_epsilon=eps), record=True)
            d = drift_summary(traj)
            worst = max(worst, d.carter, d.null)
            runs.append(trajectory_row(traj, horizon_epsilon=eps))
            last = traj
        states = {r["state"] for r in runs}
        bad = states & {TerminationState.NUMERICAL_FAILURE.name, TerminationState.OUT_OF_DOMAIN.name}
        passed = passed and runs[-1]["state"] == TerminationState.ESCAPED.name and not bad
        rows.append({"spin": float(spin), "b": b, "b_c_derived": b_c, "r_plus": r_plus,
                     "r_ph_prograde": equatorial_photon_orbit_radius(st, True), "r_turn_analytic": r_turn,
                     "outcome_flip_epsilon": (r_turn - r_plus) if r_turn is not None else math.nan,
                     "outcome_changed": len(states) > 1, "runs": runs})
    return ValidationCheck(
        name="near_extremal",
        description="prograde ray at b = b_c (1 + offset) for near-extremal spins and several horizon margins",
        computed={"rows": rows, "offset": params.near_extremal_offset},
        reference={"expected_state_at_smallest_epsilon": TerminationState.ESCAPED.name},
        error={"max_carter_or_null_drift": worst},
        tolerance={"conservation_tol": params.conservation_tol},
        passed=passed and worst <= params.conservation_tol,
        notes="outcome_flip_epsilon = r_turn - r_+: a horizon_epsilon at or above it reports CAPTURED",
    ), last


def check_reversibility(mass: float, params: KerrValidationParams, integ: IntegratorOptions,
                        term: TerminationOptions) -> ValidationCheck:
    """Check 6: forward integration, momentum reversal, backward integration, distance to the start."""
    st = kerr(mass, params.reversibility_spin)
    r0 = params.launch_radius * mass
    y0 = off_equatorial_ray(st, r0, params.conservation_theta_deg, params.conservation_xi * mass,
                            params.conservation_eta * mass**2)
    trip = reversibility_roundtrip(st, y0, integ, term, params.reversibility_affine_fraction * r0)
    errors = trip.pop("errors")
    budget = TerminationState.MAX_AFFINE_PARAMETER.name
    both_budget = trip["forward_state"] == trip["backward_state"] == budget
    return ValidationCheck(
        name="reversibility",
        description="forward to lambda = fraction * r0, reverse p_mu, integrate back, compare with the start",
        computed={"spin": params.reversibility_spin, "affine_length": params.reversibility_affine_fraction * r0, **trip},
        reference={"start_state": trip["start"]},
        error=errors,
        tolerance={"reversibility_tol": params.reversibility_tol},
        passed=both_budget and all(v <= params.reversibility_tol for v in errors.values()),
        notes="both legs must end MAX_AFFINE_PARAMETER (escape disabled, capture unchanged)",
    )


def _body(cfg: KerrRayConfig, ctx: ExperimentContext, holder: dict[str, ValidationReport]) -> dict[str, Any]:
    from kerrray.validation.kerr_report import write_validation_outputs

    params = ctx.parameters("validation", KerrValidationParams)
    mass, spin = cfg.black_hole.mass, cfg.black_hole.spin
    integ, term = integrator_options_from_config(cfg), termination_options_from_config(cfg)
    horizon_spins = sorted({float(spin), *params.critical_spins, *params.near_extremal_spins,
                            *params.conservation_spins, *params.small_spins})
    checks: list[ValidationCheck] = []
    trajectories: dict[str, Trajectory | None] = {}

    def log(check: ValidationCheck) -> None:
        checks.append(check)
        ctx.logger.info("%s: %s %s (%.1f s)", EXPERIMENT_NAME, check.name, "PASS" if check.passed else "FAIL",
                        ctx.timer.lap(check.name))

    log(check_horizon(mass, horizon_spins, params))
    log(check_schwarzschild_limit(mass, params, integ, term))
    log(check_critical_impact(mass, params, integ, term))
    cons, trajectories["conservation"] = check_conservation(mass, params, integ, term)
    log(cons)
    near, trajectories["near_extremal"] = check_near_extremal(mass, params, integ, term)
    log(near)
    log(check_reversibility(mass, params, integ, term))
    report = ValidationReport.from_checks("kerr", spin, checks)
    holder["report"] = report
    write_validation_outputs(ctx, report, params, trajectories["conservation"], trajectories["near_extremal"])
    return {**report.to_dict(), "stage_runtimes_s": dict(ctx.timer.laps)}


def run_kerr_validation(cfg: KerrRayConfig, *, environment: EnvironmentInfo | None = None) -> ValidationReport:
    """Run the six Kerr checks as the ``validate_kerr`` experiment and return the report.

    The run directory receives ``manifest.json`` and ``config.yaml``; the
    report directory receives ``summary.json``, the figures and ``report.md``
    (:mod:`kerrray.validation.kerr_report`).
    """
    holder: dict[str, ValidationReport] = {}
    record = run_experiment(EXPERIMENT_NAME, cfg, lambda ctx: _body(cfg, ctx, holder), environment=environment)
    return replace(holder["report"], run=record)
