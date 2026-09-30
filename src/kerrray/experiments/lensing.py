"""Strong-field lensing experiment: deflection angle, closest approach and
orbital turns versus impact parameter (PROJECT.md section 22; docs/lensing.md
section 5).

Equatorial photons are launched inward from ``launch_radius`` with impact
parameters sampled between ``impact_min`` and ``impact_max`` (sub-block
``lensing`` of ``experiment.parameters``, D-009), integrated one by one with
the scalar integrator (which records the trajectory, so the closest approach
and the azimuthal winding are available), and their deflection is measured
with :func:`kerrray.physics.lensing.deflection_from_trajectory`. For Kerr the
prograde (``a L_z > 0``) and retrograde senses are scanned separately; for
Schwarzschild one scan is compared with the exact integral
:func:`kerrray.physics.lensing.schwarzschild_deflection_exact` and every scan
with the weak-field value ``4 M / b``. Near the critical impact parameter the
Bozza 2002 form ``alpha = -a_bar log(b/b_c - 1) + b_bar`` is fitted.

The escape radius of the run is the launch radius (the deflection is
measured between the launch and the exit at the same radius), recorded in
the results; ``lambda_max`` must cover about ``2 launch_radius`` plus the
orbit, and a warning is logged when it does not.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from kerrray.experiments.base import ExperimentContext, RunRecord, run_experiment
from kerrray.geodesics.initial_conditions import equatorial_photon
from kerrray.geodesics.integrators import IntegratorOptions, TerminationOptions, integrate
from kerrray.geometry import Spacetime
from kerrray.photons.classification import TerminationState
from kerrray.photons.orbits import critical_impact_parameters
from kerrray.photons.trajectories import azimuthal_winding, closest_approach, deflection_angle
from kerrray.physics.lensing import (
    StrongFieldFit,
    darwin_strong_field_coefficients,
    deflection_from_trajectory,
    schwarzschild_deflection_exact,
    strong_field_fit,
    weak_field_deflection,
)
from kerrray.raytracing.renderer import integrator_options_from_config
from kerrray.reporting.figures import lensing_figure
from kerrray.reporting.report import ReportSections, write_report
from kerrray.reporting.tables import markdown_table
from kerrray.utils.config import ConfigError, KerrRayConfig

__all__ = [
    "CRITICAL_OFFSET_MIN",
    "EXPERIMENT_NAME",
    "LENSING_BLOCK",
    "STRONG_FIELD_OFFSET_MAX",
    "LensingParams",
    "LensingScan",
    "lensing_scan",
    "run",
    "sample_impact_parameters",
]

EXPERIMENT_NAME: Final[str] = "lensing"
LENSING_BLOCK: Final[str] = "lensing"

CRITICAL_OFFSET_MIN: Final[float] = 1e-3
"""Smallest ``b / b_c - 1`` sampled when the requested range reaches below ``b_c``.

Closer rays circle the hole several times and their deflection is dominated
by the exponential sensitivity near the photon orbit (PROJECT.md section
21), which is the subject of the near-critical experiment, not of this one.
"""

STRONG_FIELD_OFFSET_MAX: Final[float] = 1e-1
"""Rays with ``b / b_c - 1`` at most this large enter the Bozza-form fit."""

SUBCRITICAL_FRACTION: Final[int] = 8
"""When the range crosses ``b_c``, one ray in this many is placed below ``b_c`` (captured)."""


@dataclass(frozen=True)
class LensingParams:
    """The ``experiment.parameters.lensing`` sub-block (docs/architecture.md section 7)."""

    impact_min: float = 3.0
    impact_max: float = 20.0
    n_rays: int = 40
    launch_radius: float = 1000.0

    def __post_init__(self) -> None:
        for name in ("impact_min", "impact_max", "launch_radius"):
            value = float(getattr(self, name))
            if not (math.isfinite(value) and value > 0.0):
                raise ConfigError(f"'lensing.{name}' must be a finite number > 0, got {value!r}")
        if not self.impact_max > self.impact_min:
            raise ConfigError("'lensing.impact_max' must exceed 'lensing.impact_min'")
        if self.n_rays < 2:
            raise ConfigError(f"'lensing.n_rays' must be >= 2, got {self.n_rays!r}")
        if not self.launch_radius > self.impact_max:
            raise ConfigError("'lensing.launch_radius' must exceed 'lensing.impact_max'")


def sample_impact_parameters(params: LensingParams, b_c: float) -> NDArray[np.float64]:
    """Impact parameters of the scan for a sense of rotation with critical value ``b_c``.

    Scattering rays (``b > b_c``) are spaced geometrically in ``b / b_c - 1``
    so that the logarithmic divergence at ``b_c`` is resolved; the smallest
    offset is :data:`CRITICAL_OFFSET_MIN` when the requested range reaches
    below ``b_c``, in which case one ray in :data:`SUBCRITICAL_FRACTION` is
    placed linearly in ``[impact_min, b_c)`` to record capture. A range
    entirely below ``b_c`` is sampled linearly (all rays are captured).
    """
    lo, hi, n = params.impact_min, params.impact_max, params.n_rays
    if hi <= b_c:
        return np.linspace(lo, hi, n)
    x_hi = hi / b_c - 1.0
    if lo < b_c:
        n_sub = max(1, n // SUBCRITICAL_FRACTION)
        sub = np.linspace(lo, b_c, n_sub, endpoint=False)
        x_lo = CRITICAL_OFFSET_MIN
    else:
        n_sub = 0
        sub = np.empty(0)
        x_lo = lo / b_c - 1.0
    n_scat = n - n_sub
    x_lo = min(x_lo, x_hi)
    scat = b_c * (1.0 + np.geomspace(x_lo, x_hi, n_scat)) if n_scat > 1 else np.array([hi])
    return np.concatenate([sub, scat])


@dataclass
class LensingScan:
    """Per-ray results of one scan (one sense of rotation)."""

    prograde: bool
    b_critical: float
    b: NDArray[np.float64]
    deflection: NDArray[np.float64]
    deflection_straight_line: NDArray[np.float64]
    closest_approach: NDArray[np.float64]
    turns: NDArray[np.float64]
    state: list[str] = field(default_factory=list)
    n_steps: NDArray[np.int64] = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    runtime_s: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))
    max_null_error: NDArray[np.float64] = field(default_factory=lambda: np.zeros(0))

    def counts(self) -> dict[str, int]:
        """Rays per termination state name."""
        out: dict[str, int] = {}
        for name in self.state:
            out[name] = out.get(name, 0) + 1
        return out


def lensing_scan(
    st: Spacetime,
    b_values: NDArray[np.float64],
    integ: IntegratorOptions,
    term: TerminationOptions,
    *,
    prograde: bool = True,
) -> LensingScan:
    """Integrate one equatorial ray per impact parameter and measure its deflection.

    Rays start at ``term.escape_radius`` (the launch radius) moving inward.
    Escaped rays get both deflection estimates (end-point directions and the
    straight-line correction of :func:`kerrray.photons.trajectories.deflection_angle`);
    captured or failed rays get ``NaN`` deflections, ``0`` turns after capture
    is not meaningful and is reported as recorded.
    """
    b_c = critical_impact_parameters(st)[0 if prograde else 1]
    r_0 = term.escape_radius
    n = len(b_values)
    defl = np.full(n, np.nan)
    defl_straight = np.full(n, np.nan)
    r_min = np.full(n, np.nan)
    turns = np.full(n, np.nan)
    n_steps = np.zeros(n, dtype=np.int64)
    runtime = np.zeros(n)
    null_err = np.zeros(n)
    states: list[str] = []
    for k, b in enumerate(b_values):
        y0 = equatorial_photon(st, r_0, float(b), inward=True, prograde=prograde)
        traj = integrate(st, y0, integ, term, record=True)
        states.append(traj.state.name)
        r_min[k] = closest_approach(traj)
        turns[k] = azimuthal_winding(traj) / (2.0 * math.pi)
        n_steps[k] = traj.n_steps
        runtime[k] = traj.runtime_s
        null_err[k] = traj.diagnostics.max_null_error
        if traj.state == TerminationState.ESCAPED:
            defl[k] = deflection_from_trajectory(traj)
            defl_straight[k] = deflection_angle(traj)
    return LensingScan(
        prograde=prograde,
        b_critical=float(b_c),
        b=np.asarray(b_values, dtype=np.float64),
        deflection=defl,
        deflection_straight_line=defl_straight,
        closest_approach=r_min,
        turns=turns,
        state=states,
        n_steps=n_steps,
        runtime_s=runtime,
        max_null_error=null_err,
    )


def _fit_or_none(b: NDArray[np.float64], alpha: NDArray[np.float64], b_c: float) -> StrongFieldFit | None:
    near = (b > b_c) & (b / b_c - 1.0 <= STRONG_FIELD_OFFSET_MAX) & np.isfinite(alpha)
    if np.count_nonzero(near) < 3:
        return None
    return strong_field_fit(b[near], alpha[near], b_c=b_c)


def _fit_dict(fit: StrongFieldFit | None) -> dict[str, Any] | None:
    if fit is None:
        return None
    return {
        "a_bar": fit.a_bar,
        "b_bar": fit.b_bar,
        "n_points": fit.n_points,
        "rms_residual": fit.rms_residual,
        "offset_max": STRONG_FIELD_OFFSET_MAX,
    }


def _scan_results(scan: LensingScan, st: Spacetime) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    weak = weak_field_deflection(st, scan.b)
    escaped = np.isfinite(scan.deflection)
    result: dict[str, Any] = {
        "prograde": scan.prograde,
        "b_critical": scan.b_critical,
        "b": scan.b,
        "deflection": scan.deflection,
        "deflection_straight_line": scan.deflection_straight_line,
        "closest_approach": scan.closest_approach,
        "turns": scan.turns,
        "state": scan.state,
        "n_steps": scan.n_steps,
        "runtime_s": scan.runtime_s,
        "max_null_error": float(np.max(scan.max_null_error)) if len(scan.max_null_error) else 0.0,
        "weak_field": weak,
        "counts": scan.counts(),
        "max_abs_dev_weak_field": float(np.max(np.abs(scan.deflection[escaped] - weak[escaped]))) if np.any(escaped) else None,
        "strong_field_fit": _fit_dict(_fit_or_none(scan.b, scan.deflection, scan.b_critical)),
    }
    rows: list[dict[str, Any]] = []
    exact = None
    if st.is_schwarzschild:
        exact = np.full(len(scan.b), np.nan)
        above = scan.b > scan.b_critical
        if np.any(above):
            exact[above] = schwarzschild_deflection_exact(st, scan.b[above])
        both = escaped & np.isfinite(exact)
        dev = scan.deflection[both] - exact[both]
        result["exact"] = exact
        result["max_abs_dev_exact"] = float(np.max(np.abs(dev))) if both.any() else None
        result["rms_dev_exact"] = float(np.sqrt(np.mean(dev**2))) if both.any() else None
        result["strong_field_fit_exact"] = _fit_dict(_fit_or_none(scan.b, exact, scan.b_critical))
    for k in range(len(scan.b)):
        row: dict[str, Any] = {
            "b [M]": float(scan.b[k]),
            "b/b_c - 1": float(scan.b[k] / scan.b_critical - 1.0),
            "deflection [rad]": None if not escaped[k] else float(scan.deflection[k]),
        }
        if exact is not None:
            row["exact [rad]"] = None if not np.isfinite(exact[k]) else float(exact[k])
        row["4M/b [rad]"] = float(weak[k])
        row["closest approach [M]"] = float(scan.closest_approach[k])
        row["turns"] = float(scan.turns[k])
        row["state"] = scan.state[k]
        rows.append(row)
    return result, rows


def _figures(ctx: ExperimentContext, scans: dict[str, LensingScan], results: dict[str, Any], st: Spacetime) -> list[Path]:
    figures: list[Path] = []
    for label, scan in scans.items():
        res = results[label]
        curves: dict[str, Any] = {"numerical (RK, end-point directions)": scan.deflection}
        if "exact" in res:
            curves["exact integral (Darwin 1959)"] = res["exact"]
        curves["weak field 4M/b"] = res["weak_field"]
        figures.append(
            lensing_figure(
                ctx.report_dir / f"lensing_deflection_{label}.png",
                scan.b,
                curves,
                title=f"Deflection vs impact parameter ({label}, spin {st.spin:g})",
            )
        )
        x = scan.b / scan.b_critical - 1.0
        above = x > 0
        curves_sf: dict[str, Any] = {"numerical": np.where(above, scan.deflection, np.nan)}
        if "exact" in res:
            curves_sf["exact integral"] = np.where(above, res["exact"], np.nan)
        fit = res.get("strong_field_fit")
        if fit is not None:
            curves_sf[f"fit -{fit['a_bar']:.3f} log(b/b_c-1) + {fit['b_bar']:+.3f}"] = np.where(
                above, -fit["a_bar"] * np.log(np.where(above, x, 1.0)) + fit["b_bar"], np.nan
            )
        figures.append(
            lensing_figure(
                ctx.report_dir / f"lensing_strong_field_{label}.png",
                np.where(above, x, np.nan),
                curves_sf,
                xlabel="b / b_c - 1",
                logx=True,
                title=f"Strong-field divergence ({label}, b_c = {scan.b_critical:.4f} M)",
            )
        )
    return figures


def _sections(cfg: KerrRayConfig, params: LensingParams, results: dict[str, Any], st: Spacetime) -> ReportSections:
    lines = []
    for label, res in results["scans"].items():
        lines.append(f"- {label}: b_c = {res['b_critical']:.6g} M; rays {res['counts']}")
        if res.get("max_abs_dev_exact") is not None:
            lines.append(f"  max |numerical - exact| = {res['max_abs_dev_exact']:.3e} rad, rms {res['rms_dev_exact']:.3e} rad")
        if res.get("max_abs_dev_weak_field") is not None:
            lines.append(f"  max |numerical - 4M/b| = {res['max_abs_dev_weak_field']:.3e} rad")
        fit = res.get("strong_field_fit")
        if fit is not None:
            lines.append(f"  Bozza fit on {fit['n_points']} rays with b/b_c - 1 <= {fit['offset_max']:g}: a_bar = {fit['a_bar']:.4f}, b_bar = {fit['b_bar']:.4f}, rms residual {fit['rms_residual']:.2e} rad")
    darwin = results.get("darwin_coefficients")
    darwin_text = "" if darwin is None else f" Darwin 1959 / Bozza 2002 reference for Schwarzschild: a_bar = {darwin['a_bar']:g}, b_bar = {darwin['b_bar']:.4f}."
    return ReportSections(
        objective="Deflection angle, closest approach and number of orbital turns of equatorial photons as a function of the impact parameter, from far weak-field rays to the logarithmic divergence at the critical impact parameter (PROJECT.md section 22).",
        mathematical_model=f"Null geodesics of the Kerr metric (mass {st.mass:g}, spin {st.spin:g}) in Boyer-Lindquist coordinates, Hamiltonian form (docs/equations.md). The deflection is |chi_exit - chi_launch| with chi = phi + atan2(r dphi/dlambda, dr/dlambda) at the end points (docs/lensing.md section 2). Reference: the exact Schwarzschild integral of Darwin 1959 (docs/lensing.md section 3) and the weak-field value 4M/b.",
        numerical_method=f"Scalar {cfg.integration.method} integration (rtol {cfg.integration.rtol:g}, atol {cfg.integration.atol:g}), one ray per impact parameter launched inward from r = {params.launch_radius:g} M; the escape radius equals the launch radius. Impact parameters spaced geometrically in b/b_c - 1 (docs/lensing.md section 5).",
        parameters=f"impact range [{params.impact_min:g}, {params.impact_max:g}] M, {params.n_rays} rays per sense of rotation, launch radius {params.launch_radius:g} M, horizon epsilon {cfg.termination.horizon_epsilon:g}, lambda_max {cfg.integration.lambda_max:g} M.",
        results="\n".join(lines),
        error_analysis="Max null-constraint error |H|/E^2 per scan: " + ", ".join(f"{label} {res['max_null_error']:.2e}" for label, res in results["scans"].items()) + ". The finite launch radius leaves 1.5 M b^3 / r_0^4 (Schwarzschild) plus 2 M |a| / r_0^2 (frame dragging) in the deflection (docs/lensing.md section 2).",
        interpretation="The deflection exceeds 4M/b increasingly as b decreases and diverges logarithmically at b_c, where photons circle the hole before escaping (turns > 1/2); for Kerr the prograde sense has the smaller b_c and the smaller deflection at equal b." + darwin_text,
        limitations="Equatorial rays only; the exact integral is available for spin 0 only; rays with b/b_c - 1 below 1e-3 are not sampled (near-critical experiment); the weak-field formula is an approximation valid for b >> M (docs/lensing.md section 6).",
        reproducibility=f"Configuration and manifest under the run directory; seed {cfg.experiment.seed}. Figures and this report are written next to summary.json.",
    )


def _body(ctx: ExperimentContext) -> dict[str, Any]:
    cfg = ctx.cfg
    params = ctx.parameters(LENSING_BLOCK, LensingParams)
    st = Spacetime(mass=cfg.black_hole.mass, spin=cfg.black_hole.spin)
    integ = integrator_options_from_config(cfg)
    term = TerminationOptions(horizon_epsilon=cfg.termination.horizon_epsilon, escape_radius=params.launch_radius)
    if integ.lambda_max < 3.0 * params.launch_radius:
        ctx.logger.warning(
            "integration.lambda_max = %g may not cover a round trip from launch_radius = %g "
            "(about 2 launch_radius plus the orbit); rays may end MAX_AFFINE_PARAMETER",
            integ.lambda_max,
            params.launch_radius,
        )
    b_c = critical_impact_parameters(st)
    senses = {"schwarzschild": True} if st.is_schwarzschild else {"prograde": True, "retrograde": False}
    scans: dict[str, LensingScan] = {}
    results: dict[str, Any] = {
        "spacetime": {"mass": st.mass, "spin": st.spin},
        "launch_radius": params.launch_radius,
        "escape_radius": term.escape_radius,
        "n_rays": params.n_rays,
        "b_critical": {"prograde": b_c[0], "retrograde": b_c[1]},
        "scans": {},
    }
    tables: list[str] = []
    for label, prograde in senses.items():
        b_values = sample_impact_parameters(params, b_c[0 if prograde else 1])
        ctx.logger.info("lensing scan %s: %d rays, b in [%g, %g]", label, len(b_values), b_values.min(), b_values.max())
        scan = lensing_scan(st, b_values, integ, term, prograde=prograde)
        scans[label] = scan
        res, rows = _scan_results(scan, st)
        results["scans"][label] = res
        tables.append(f"**{label}** (b_c = {scan.b_critical:.6g} M)\n\n" + markdown_table(rows))
        ctx.timer.lap(label)
    if st.is_schwarzschild:
        a_bar, b_bar = darwin_strong_field_coefficients()
        results["darwin_coefficients"] = {"a_bar": a_bar, "b_bar": b_bar}
    figures = _figures(ctx, scans, results["scans"], st)
    report = write_report(
        ctx.report_dir,
        _sections(cfg, params, results, st),
        figures=figures,
        tables=tables,
        title=f"KerrRay lensing experiment (spin {st.spin:g})",
    )
    results["figures"] = [str(p) for p in figures]
    results["report"] = str(report)
    results["stage_runtime_s"] = dict(ctx.timer.laps)
    return results


def run(cfg: KerrRayConfig) -> RunRecord:
    """Run the lensing experiment for ``cfg`` (docs/architecture.md section 6)."""
    return run_experiment(EXPERIMENT_NAME, cfg, _body)
