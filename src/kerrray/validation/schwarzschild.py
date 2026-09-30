"""Schwarzschild validation: EXP-001 photon sphere and EXP-002 critical impact
parameter (PROJECT.md sections 12 and 37; docs/validation.md section 2).

Both quantities are *measured* from integrated rays by a bracketed capture
threshold search; the reference values are used only to compute the error.

* **Photon sphere (EXP-001).** An equatorial photon launched tangentially
  (``p_r = 0``) at ``r0`` sits at a turning point of the radial motion, and
  the sign of ``dp_r/dlambda`` decides its fate: below the unstable circular
  orbit it falls into the hole, above it it escapes. The search keeps
  ``r_lo`` (``CAPTURED``) and ``r_hi`` (``ESCAPED``) and shrinks the bracket
  with ``n_probe`` rays per iteration (``n_probe = 1`` is classical
  bisection). The discriminant is linear in ``r0 - r_ph``, so the bracket
  resolves ``r_ph`` to the tolerance.
* **Critical impact parameter (EXP-002).** Photons launched inward from
  ``r0`` in the equatorial plane with impact parameter ``b = |L_z|/E`` are
  captured for ``b < b_c`` and escape for ``b > b_c``; the same search on
  ``b`` (:func:`kerrray.validation.kerr_bisection.bisect_critical_impacts`).
* **Convergence of b_c.** The search is repeated for the adaptive
  tolerances ``rtol_levels`` and the fixed RK4 steps ``step_sizes`` with a
  bracket width far below the numerical error, and the error against the
  reference is fitted with :func:`kerrray.validation.convergence.loglog_order`
  plus a reference-free Richardson estimate. These rays start at
  ``convergence_launch_radius`` and escape at the same radius: an outbound
  photon beyond the single maximum of the effective potential
  ``(1 - 2M/r)/r^2`` never turns again (Misner, Thorne and Wheeler 1973,
  section 25.6), so the outcome equals that of a launch from far away while
  the fixed-step runs stay affordable.

References (comparison only): photon sphere ``r = 3M`` and critical impact
parameter ``b_c = 3 sqrt(3) M`` (Misner, Thorne and Wheeler 1973, section
25.6; Chandrasekhar 1983, *The Mathematical Theory of Black Holes*, chapter 3).
They are defined once below and enter nothing but the error columns. The
fitted fixed-step order is compared with the formal order 4 of the classical
Runge-Kutta scheme (Hairer, Norsett and Wanner 1993, section II.1).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace
from typing import Any, Final

import numpy as np

from kerrray.experiments.base import ExperimentContext, run_experiment
from kerrray.geodesics import IntegratorOptions, TerminationOptions, integrate_batch, tangential_photon
from kerrray.geometry import schwarzschild
from kerrray.photons import TerminationState
from kerrray.reporting.plots import plot_convergence
from kerrray.reporting.report import ReportSections, write_report
from kerrray.reporting.tables import markdown_table
from kerrray.utils.config import KerrRayConfig
from kerrray.utils.manifest import EnvironmentInfo
from kerrray.validation import (
    ValidationCheck,
    ValidationReport,
    integrator_options_from_config,
    termination_options_from_config,
)
from kerrray.validation.convergence import ConvergenceStudy, convergence_study
from kerrray.validation.kerr_bisection import bisect_critical_impacts

__all__ = [
    "CRITICAL_IMPACT_REFERENCE",
    "PHOTON_SPHERE_REFERENCE",
    "RK4_FORMAL_ORDER",
    "CriticalImpactResult",
    "PhotonSphereResult",
    "SchwarzschildValidationParams",
    "critical_impact_experiment",
    "photon_sphere_experiment",
    "rk4_step_convergence",
    "rtol_convergence",
    "run_schwarzschild_validation",
]

PHOTON_SPHERE_REFERENCE: Final[float] = 3.0
"""Schwarzschild photon-sphere radius in units of M (MTW 1973 section 25.6). Comparison only."""
CRITICAL_IMPACT_REFERENCE: Final[float] = 3.0 * math.sqrt(3.0)
"""Schwarzschild critical impact parameter in units of M (MTW 1973 section 25.6). Comparison only."""
RK4_FORMAL_ORDER: Final[int] = 4
"""Formal global order of the classical Runge-Kutta scheme (Hairer, Norsett and Wanner 1993, II.1)."""
EXPERIMENT_NAME: Final[str] = "validate_schwarzschild"


@dataclass(frozen=True)
class SchwarzschildValidationParams:
    """``experiment.parameters.validation`` of ``configs/schwarzschild.yaml``.

    The first three keys are the D-009 block; the others are documented
    extensions with defaults (docs/validation.md section 7). Brackets are
    experiment inputs (wide intervals checked at run time), never references.
    """

    photon_sphere_tol: float = 1.0e-6
    critical_b_tol: float = 1.0e-6
    bisection_iterations: int = 60
    photon_sphere_bracket: list[float] = field(default_factory=lambda: [2.5, 4.0])
    impact_bracket: list[float] = field(default_factory=lambda: [4.0, 7.0])
    launch_radius: float = 1000.0
    probes_per_iteration: int = 15
    run_convergence: bool = True
    rtol_levels: list[float] = field(default_factory=lambda: [1e-6, 1e-8, 1e-10, 1e-12])
    step_sizes: list[float] = field(default_factory=lambda: [0.4, 0.2, 0.1])
    convergence_launch_radius: float = 20.0
    convergence_halfwidth: float = 1.0e-2
    convergence_tol: float = 1.0e-11
    order_tolerance: float = 0.5


@dataclass(frozen=True)
class PhotonSphereResult:
    """Outcome of :func:`photon_sphere_experiment` (radii in units of M)."""

    radius: float
    r_lo: float
    r_hi: float
    width: float
    iterations: int
    n_rays: int
    max_steps: int
    runtime_s: float


@dataclass(frozen=True)
class CriticalImpactResult:
    """Outcome of :func:`critical_impact_experiment` (impact parameters in units of M)."""

    b_c: float
    b_lo: float
    b_hi: float
    width: float
    iterations: int
    n_rays: int
    max_steps: int
    runtime_s: float
    launch_radius: float
    horizon_crossings: int = 0


def _require(states: np.ndarray, values: np.ndarray, what: str) -> None:
    ok = (states == TerminationState.CAPTURED) | (states == TerminationState.ESCAPED)
    if not np.all(ok):
        bad = [(float(v), TerminationState(int(s)).name) for v, s in zip(values, states, strict=True) if not
               (s == TerminationState.CAPTURED or s == TerminationState.ESCAPED)]
        raise RuntimeError(f"{what}: rays ended neither CAPTURED nor ESCAPED: {bad}")


def photon_sphere_experiment(
    integ: IntegratorOptions,
    term: TerminationOptions,
    *,
    r_lo: float,
    r_hi: float,
    tol: float,
    max_iterations: int = 60,
    n_probe: int = 1,
    mass: float = 1.0,
) -> PhotonSphereResult:
    """Locate the unstable circular photon orbit by bracketing tangential launches (module docstring).

    Raises:
        ValueError: On an invalid bracket, tolerance or probe count.
        RuntimeError: If ``r_lo`` is not captured, ``r_hi`` does not escape,
            a ray ends in another state, or the outcome is not monotone.
    """
    if not (0.0 < r_lo < r_hi) or not tol > 0.0 or n_probe < 1 or max_iterations < 1:
        raise ValueError(f"invalid search: r_lo={r_lo}, r_hi={r_hi}, tol={tol}, n_probe={n_probe}")
    st = schwarzschild(mass)
    t0 = time.perf_counter()
    n_rays = max_steps = iterations = 0

    def classify(radii: np.ndarray) -> np.ndarray:
        nonlocal n_rays, max_steps
        res = integrate_batch(st, np.stack([tangential_photon(st, float(r)) for r in radii]), integ, term)
        n_rays += radii.size
        max_steps = max(max_steps, int(np.max(res.n_steps)))
        _require(res.state, radii, "photon sphere")
        return np.asarray(res.state)

    ends = classify(np.array([r_lo, r_hi]))
    if ends[0] != TerminationState.CAPTURED or ends[1] != TerminationState.ESCAPED:
        raise RuntimeError(f"bracket [{r_lo}, {r_hi}] invalid: {TerminationState(int(ends[0])).name} at r_lo, "
                           f"{TerminationState(int(ends[1])).name} at r_hi")
    lo, hi = r_lo, r_hi
    fractions = np.arange(1, n_probe + 1) / (n_probe + 1)
    while hi - lo > tol and iterations < max_iterations:
        radii = lo + (hi - lo) * fractions
        states = classify(radii)
        captured = radii[states == TerminationState.CAPTURED]
        escaped = radii[states == TerminationState.ESCAPED]
        if captured.size and escaped.size and captured.max() > escaped.min():
            raise RuntimeError(f"non-monotone outcome in r0: captured at {captured.max()!r}, escaped at {escaped.min()!r}")
        lo = max(lo, float(captured.max())) if captured.size else lo
        hi = min(hi, float(escaped.min())) if escaped.size else hi
        iterations += 1
    return PhotonSphereResult(0.5 * (lo + hi), lo, hi, hi - lo, iterations, n_rays, max_steps,
                              time.perf_counter() - t0)


def critical_impact_experiment(
    integ: IntegratorOptions,
    term: TerminationOptions,
    *,
    b_lo: float,
    b_hi: float,
    tol: float,
    r0: float,
    max_iterations: int = 60,
    n_probe: int = 1,
    mass: float = 1.0,
    horizon_crossing_as_capture: bool = False,
) -> CriticalImpactResult:
    """Bracket the capture threshold in ``b`` for inward equatorial rays from ``r0`` (module docstring)."""
    (res,) = bisect_critical_impacts(
        schwarzschild(mass), integ, term, r0=r0, b_lo=b_lo, b_hi=b_hi, tol=tol,
        max_iterations=max_iterations, n_probe=n_probe, directions=(True,),
        horizon_crossing_as_capture=horizon_crossing_as_capture,
    )
    return CriticalImpactResult(res.b_c, res.b_lo, res.b_hi, res.width, res.iterations, res.n_rays,
                                res.max_steps, res.runtime_s, r0, res.horizon_crossings)


def _convergence(parameter: str, runs: list[tuple[float, IntegratorOptions]], params: SchwarzschildValidationParams,
                 term: TerminationOptions, center: float, mass: float, crossing: bool) -> ConvergenceStudy:
    values, extra = [], []
    for level, integ in runs:
        res = critical_impact_experiment(
            integ, term, b_lo=center - params.convergence_halfwidth, b_hi=center + params.convergence_halfwidth,
            tol=params.convergence_tol, r0=params.convergence_launch_radius * mass, max_iterations=params.bisection_iterations,
            n_probe=params.probes_per_iteration, mass=mass, horizon_crossing_as_capture=crossing,
        )
        values.append(res.b_c)
        extra.append({parameter: level, "b_c": res.b_c, "width": res.width, "iterations": res.iterations,
                      "n_rays": res.n_rays, "max_steps": res.max_steps, "runtime_s": res.runtime_s,
                      "horizon_crossings": res.horizon_crossings})
    return convergence_study(parameter, [lv for lv, _ in runs], values, CRITICAL_IMPACT_REFERENCE * mass,
                             noise_floor=params.convergence_tol, extra=extra)


def rtol_convergence(params: SchwarzschildValidationParams, integ: IntegratorOptions, term: TerminationOptions,
                     center: float, mass: float = 1.0) -> ConvergenceStudy:
    """``b_c`` against the adaptive ``rtol`` (``atol`` keeps the configured ``atol/rtol`` ratio).

    ``center`` is the centre of the search bracket (normally the ``b_c``
    measured by :func:`critical_impact_experiment`), half-width
    ``params.convergence_halfwidth``.
    """
    ratio = integ.atol / integ.rtol
    runs = [(r, replace(integ, method="rk45", rtol=r, atol=r * ratio)) for r in params.rtol_levels]
    term_c = replace(term, escape_radius=params.convergence_launch_radius * mass)
    return _convergence("rtol", runs, params, term_c, center, mass, False)


def rk4_step_convergence(params: SchwarzschildValidationParams, integ: IntegratorOptions, term: TerminationOptions,
                         center: float, mass: float = 1.0) -> ConvergenceStudy:
    """``b_c`` against the fixed RK4 step (horizon crossings relabelled, see kerr_bisection)."""
    runs = []
    for h in params.step_sizes:
        budget = int(math.ceil(integ.lambda_max / h)) + 1
        runs.append((h, replace(integ, method="rk4", step_size=h, max_steps=max(integ.max_steps, budget))))
    term_c = replace(term, escape_radius=params.convergence_launch_radius * mass)
    return _convergence("step_size", runs, params, term_c, center, mass, True)


def _threshold_check(name: str, what: str, value: float, reference: float, tol: float,
                     detail: dict[str, Any]) -> ValidationCheck:
    err = abs(value - reference)
    return ValidationCheck(
        name=name,
        description=f"{what} from a bracketed capture/escape search of integrated rays",
        computed={"value": value, **detail},
        reference={"value": reference},
        error={"absolute": err, "relative": err / reference},
        tolerance={"absolute": tol},
        passed=err <= tol,
    )


def _study_check(name: str, study: ConvergenceStudy, passed: bool, tolerance: dict[str, Any], notes: str) -> ValidationCheck:
    return ValidationCheck(
        name=name,
        description=f"convergence of the bisected b_c with {study.parameter}",
        computed=study.as_dict(),
        reference={"value": study.reference},
        error={"finest": study.errors[-1], "fitted_order": study.fitted_order},
        tolerance=tolerance,
        passed=passed,
        notes=notes,
    )


def _body(cfg: KerrRayConfig, ctx: ExperimentContext, holder: dict[str, ValidationReport]) -> dict[str, Any]:
    params = ctx.parameters("validation", SchwarzschildValidationParams)
    mass = cfg.black_hole.mass
    integ, term = integrator_options_from_config(cfg), termination_options_from_config(cfg)
    n_probe, iters = params.probes_per_iteration, params.bisection_iterations
    ps = photon_sphere_experiment(integ, term, r_lo=params.photon_sphere_bracket[0] * mass,
                                  r_hi=params.photon_sphere_bracket[1] * mass, tol=params.photon_sphere_tol * mass,
                                  max_iterations=iters, n_probe=n_probe, mass=mass)
    ctx.logger.info("photon sphere %.9f M (%.1f s)", ps.radius, ps.runtime_s)
    cb = critical_impact_experiment(integ, term, b_lo=params.impact_bracket[0] * mass,
                                    b_hi=params.impact_bracket[1] * mass, tol=params.critical_b_tol * mass,
                                    r0=params.launch_radius * mass, max_iterations=iters, n_probe=n_probe, mass=mass)
    ctx.logger.info("critical impact parameter %.9f M (%.1f s)", cb.b_c, cb.runtime_s)
    checks = [
        _threshold_check("photon_sphere", "photon-sphere radius", ps.radius, PHOTON_SPHERE_REFERENCE * mass,
                         params.photon_sphere_tol * mass, {k: v for k, v in ps.__dict__.items() if k != "radius"}),
        _threshold_check("critical_impact", "critical impact parameter", cb.b_c, CRITICAL_IMPACT_REFERENCE * mass,
                         params.critical_b_tol * mass, {k: v for k, v in cb.__dict__.items() if k != "b_c"}),
    ]
    if params.run_convergence:
        rt = rtol_convergence(params, integ, term, cb.b_c, mass)
        ctx.logger.info("rtol convergence: fitted order %.3f (%.1f s)", rt.fitted_order, ctx.timer.lap("rtol"))
        resolved = [e for e in rt.errors if e > rt.noise_floor]
        rt_ok = bool(np.all(np.diff(resolved) <= 0.0)) and rt.errors[-1] <= params.critical_b_tol * mass
        checks.append(_study_check("rtol_convergence", rt, rt_ok, {"finest_absolute": params.critical_b_tol * mass},
                                   "pass: errors above the bracket width do not grow as rtol shrinks and the finest is within critical_b_tol"))
        rk = rk4_step_convergence(params, integ, term, cb.b_c, mass)
        ctx.logger.info("RK4 convergence: fitted order %.3f (%.1f s)", rk.fitted_order, ctx.timer.lap("rk4"))
        rk_ok = math.isfinite(rk.fitted_order) and abs(rk.fitted_order - RK4_FORMAL_ORDER) <= params.order_tolerance
        checks.append(_study_check("rk4_convergence", rk, rk_ok,
                                   {"formal_order": RK4_FORMAL_ORDER, "order_tolerance": params.order_tolerance},
                                   "pass: fitted order within order_tolerance of the formal RK4 order"))
    report = ValidationReport.from_checks("schwarzschild", cfg.black_hole.spin, checks)
    holder["report"] = report
    _write_report(ctx, report, params)
    return report.to_dict()


def _write_report(ctx: ExperimentContext, report: ValidationReport, params: SchwarzschildValidationParams) -> None:
    """``report.md`` with the verdict table, the convergence tables and figures (values from ``report`` only)."""
    figures, tables = [], [markdown_table([
        {"check": c.name, "verdict": "PASS" if c.passed else "FAIL",
         **{f"error {k}": v for k, v in c.error.items()}} for c in report.checks], precision=6)]
    for c in report.checks:
        if c.name.endswith("_convergence"):
            comp = c.computed
            tables.append(markdown_table([{comp["parameter"]: lv, "b_c": v, "error": e}
                                          for lv, v, e in zip(comp["levels"], comp["values"], comp["errors"],
                                                              strict=True)], precision=14))
            figures.append(plot_convergence(ctx.report_dir / f"{c.name}.png", comp["levels"], comp["errors"],
                                            comp["parameter"], "|b_c - 3 sqrt(3) M| [M]",
                                            title=f"Critical impact parameter against {comp['parameter']}"))
    lines = [f"- {c.name}: {'PASS' if c.passed else 'FAIL'}; computed {c.computed.get('value', '')}; "
             + ", ".join(f"{k} {v}" for k, v in c.error.items()) for c in report.checks]
    write_report(ctx.report_dir, ReportSections(
        objective="EXP-001 and EXP-002 (PROJECT.md sections 12 and 37): recover the Schwarzschild photon sphere and "
                  "critical impact parameter from integrated rays and measure the error against 3M and 3 sqrt(3) M.",
        mathematical_model="Schwarzschild metric (a = 0 limit of Kerr) in Boyer-Lindquist coordinates; null "
                           "geodesics in Hamiltonian form (docs/equations_geodesics.md).",
        numerical_method="Bracketed capture/escape search (kerrray.validation.schwarzschild, kerr_bisection) with the "
                         "configured integrator; convergence of b_c with rtol and with the fixed RK4 step.",
        parameters="\n".join(f"- `{k}`: {v}" for k, v in params.__dict__.items()),
        results="\n".join(lines),
        error_analysis="Errors include the bisection half-width (at most half the bracket tolerance) and the "
                       "integration error; convergence errors below the bracket tolerance are excluded from the fits.",
        interpretation="See the verdicts: each value is compared with its reference within the configured tolerance.",
        limitations="Fixed-step RK4 captured rays cross the horizon in one step and are relabelled CAPTURED "
                    "(documented in kerrray.validation.kerr_bisection).",
        reproducibility=f"Run {ctx.run_id}; configuration in {ctx.run_dir / 'config.yaml'}.",
    ), figures, tables, title="Schwarzschild validation (validate_schwarzschild)")


def run_schwarzschild_validation(cfg: KerrRayConfig, *, environment: EnvironmentInfo | None = None) -> ValidationReport:
    """Run EXP-001/EXP-002 (and the b_c convergence studies) as ``validate_schwarzschild``.

    Raises:
        ValueError: If the configuration has a non-zero spin.
    """
    if cfg.black_hole.spin != 0.0:
        raise ValueError(f"Schwarzschild validation needs black_hole.spin = 0, got {cfg.black_hole.spin!r}")
    holder: dict[str, ValidationReport] = {}
    record = run_experiment(EXPERIMENT_NAME, cfg, lambda ctx: _body(cfg, ctx, holder), environment=environment)
    return replace(holder["report"], run=record)
