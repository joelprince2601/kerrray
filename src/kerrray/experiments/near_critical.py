"""Near-critical photon trajectories (EXP-007, PROJECT.md section 21).

Equatorial photons are launched inward from ``launch_radius`` with impact
parameters ``b = b_c (1 +- delta)`` for every ``delta`` in
``experiment.parameters.near_critical.offsets`` and every family:

* Schwarzschild (spin ``0`` in ``spins``): ``b_c`` is *computed by bisection*
  on the integrator's own capture/escape outcome between the generic bracket
  ``[BISECTION_BRACKET[0] M, BISECTION_BRACKET[1] M]`` to the absolute
  tolerance ``bisection_tol`` (units of ``M``), with the configured
  integration block; a ray that does not escape within the budget (captured,
  failed or budget exhausted) counts as the captured side. The independent
  root-finding value of
  :func:`kerrray.photons.orbits.critical_impact_parameters` (which the
  symbolic role's tests compare with ``3 sqrt(3) M``) is reported next to it
  for comparison, never used to produce the bisected value.
* Kerr (each non-zero spin): prograde and retrograde families with ``b_c``
  from :func:`kerrray.photons.orbits.critical_impact_parameters`.

For each ray and each relative tolerance in ``tolerances`` (``atol`` keeps the
configured ``atol / rtol`` ratio) the run records the outcome, the number of
orbital turns, the affine length, the accepted and rejected steps, the
conservation diagnostics and the runtime. Turns are ``|Delta phi| / (2 pi)``
accumulated over the recorded points with ``r >= r_+ + PLUNGE_MARGIN``
(:data:`kerrray.benchmarks.rayset.PLUNGE_MARGIN`); WHAT, the plunge below
``r_+ + 0.1 M`` is excluded from the count; WHY, in Boyer-Lindquist
coordinates ``phi`` of an infalling photon diverges logarithmically at the
horizon for ``a != 0`` (frame dragging), which would add a coordinate
winding of about ``a ln(1/horizon_epsilon) / (r_+ - r_-)`` radians to every
captured Kerr ray; LIMITATION, ``turns`` of captured rays counts the
photon-sphere passages only. ``turns_total`` keeps the full ``Delta phi``.

Sensitivity: the least-squares slope of ``turns`` against ``log10(delta)``
is fitted separately for the escaping branch (``b > b_c``) and the captured
branch (``b < b_c``) of every family at every tolerance, over the offsets
whose outcome is the expected one; ``turns_per_decade`` is minus that slope
(turns gained per decade of decrease of ``delta``). The fitted slopes are
reported, never assumed. For comparison only, the Schwarzschild strong
deflection limit (Bozza 2002, Phys. Rev. D 66, 103001, coefficient
``a_bar = 1``) gives ``Delta phi = -ln(delta) + const`` on the escaping branch,
i.e. ``ln(10) / (2 pi)`` turns per decade; the same linearisation about the
photon orbit gives the same leading coefficient on the captured branch
(docs/numerical_analysis.md section 5; our derivation, not a literature
value). Numerical sensitivity:
for every ``(family, delta, sign)`` the outcomes at the different tolerances
are compared and a *flip* is recorded when they disagree; per family, branch
and tolerance the smallest offset above which every outcome is the expected
one is reported.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np

from kerrray.benchmarks.rayset import PLUNGE_MARGIN
from kerrray.benchmarks.solver import integrator_options, termination_options
from kerrray.experiments.base import ExperimentContext, RunRecord, run_experiment
from kerrray.geodesics import IDX_PH, IDX_R, IntegratorOptions, TerminationOptions, equatorial_photon, integrate
from kerrray.geometry import Spacetime, outer_horizon
from kerrray.photons import TerminationState, Trajectory, closest_approach, to_cartesian
from kerrray.photons.orbits import critical_impact_parameters
from kerrray.reporting.plots import plot_lines, plot_trajectory_xy
from kerrray.reporting.report import ReportSections, write_report
from kerrray.reporting.tables import markdown_table
from kerrray.utils.config import KerrRayConfig
from kerrray.utils.config_parsing import ConfigError

__all__ = [
    "BISECTION_BRACKET",
    "EXPERIMENT_NAME",
    "Family",
    "NearCriticalParams",
    "SCHWARZSCHILD_TURNS_PER_DECADE",
    "bisect_critical_impact",
    "build_families",
    "measure_ray",
    "run",
    "turns_outside_plunge",
    "turns_per_decade",
]

EXPERIMENT_NAME: Final[str] = "near_critical"
BISECTION_BRACKET: Final[tuple[float, float]] = (3.0, 8.0)
"""Generic bracket (units of M) for the Schwarzschild bisection: b = 3 M is captured, b = 8 M escapes."""
SIGNS: Final[tuple[int, int]] = (1, -1)
EXPECTED: Final[dict[int, TerminationState]] = {1: TerminationState.ESCAPED, -1: TerminationState.CAPTURED}
SCHWARZSCHILD_TURNS_PER_DECADE: Final[float] = math.log(10.0) / (2.0 * math.pi)
"""Comparison value only: Schwarzschild strong-deflection-limit winding per decade of delta (module docstring)."""


@dataclass(frozen=True)
class NearCriticalParams:
    """``experiment.parameters.near_critical`` (D-009 keys plus the numerical role's additions)."""

    offsets: list[float] = field(default_factory=lambda: [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8])
    launch_radius: float = 1000.0
    tolerances: list[float] = field(default_factory=lambda: [1e-8, 1e-10, 1e-12])
    spins: list[float] = field(default_factory=lambda: [0.0, 0.9])
    bisection_tol: float = 1e-10
    bisection_max_iterations: int = 60

    def __post_init__(self) -> None:
        for key in ("offsets", "tolerances"):
            values = getattr(self, key)
            if not values or any(not (math.isfinite(v) and 0.0 < v < 1.0) for v in values):
                raise ConfigError(f"'near_critical.{key}' must be a non-empty list of numbers in (0, 1)")
        if not self.launch_radius > BISECTION_BRACKET[1]:
            raise ConfigError(f"'near_critical.launch_radius' must exceed {BISECTION_BRACKET[1]} M")
        if not self.spins or any(not abs(s) < 1.0 for s in self.spins):
            raise ConfigError("'near_critical.spins' must be a non-empty list with |spin| < 1")
        if not (math.isfinite(self.bisection_tol) and self.bisection_tol > 0.0):
            raise ConfigError("'near_critical.bisection_tol' must be > 0")
        if self.bisection_max_iterations < 1:
            raise ConfigError("'near_critical.bisection_max_iterations' must be >= 1")


@dataclass(frozen=True)
class Family:
    """One (spacetime, sense) family with its critical impact parameter."""

    name: str
    spacetime: Spacetime
    prograde: bool
    b_c: float
    b_c_source: str
    b_c_reference: float
    bisection: dict[str, Any] | None = None


def _escapes(st: Spacetime, r0: float, b: float, integ: IntegratorOptions, term: TerminationOptions) -> bool:
    traj = integrate(st, equatorial_photon(st, r0, b, inward=True, prograde=True), integ, term, record=False)
    return traj.state == TerminationState.ESCAPED


def bisect_critical_impact(
    st: Spacetime, r0: float, integ: IntegratorOptions, term: TerminationOptions, *, b_lo: float, b_hi: float,
    tol: float, max_iterations: int,
) -> dict[str, Any]:
    """Bisect the capture/escape threshold in ``b`` (module docstring); returns the bracket and counts."""
    if _escapes(st, r0, b_lo, integ, term) or not _escapes(st, r0, b_hi, integ, term):
        raise RuntimeError(f"bracket [{b_lo}, {b_hi}] does not straddle the capture threshold")
    iterations = 0
    while b_hi - b_lo > tol and iterations < max_iterations:
        mid = 0.5 * (b_lo + b_hi)
        if _escapes(st, r0, mid, integ, term):
            b_hi = mid
        else:
            b_lo = mid
        iterations += 1
    return {"b_c": 0.5 * (b_lo + b_hi), "b_lo": b_lo, "b_hi": b_hi, "half_width": 0.5 * (b_hi - b_lo),
            "iterations": iterations, "n_integrations": iterations + 2, "tol": tol, "rtol": integ.rtol, "method": integ.method}


def build_families(params: NearCriticalParams, cfg: KerrRayConfig, integ: IntegratorOptions, term: TerminationOptions) -> list[Family]:
    """Families of the run: Schwarzschild by bisection, Kerr prograde/retrograde from the orbit equations."""
    families: list[Family] = []
    for spin in params.spins:
        st = Spacetime(mass=cfg.black_hole.mass, spin=float(spin))
        b_pro, b_ret = critical_impact_parameters(st)
        if spin == 0.0:
            m = cfg.black_hole.mass
            bis = bisect_critical_impact(st, params.launch_radius, integ, term, b_lo=BISECTION_BRACKET[0] * m,
                                         b_hi=BISECTION_BRACKET[1] * m, tol=params.bisection_tol,
                                         max_iterations=params.bisection_max_iterations)
            families.append(Family("schwarzschild", st, True, bis["b_c"], "bisection", b_pro, bis))
        else:
            families.append(Family(f"kerr a={spin:g} prograde", st, True, b_pro, "critical_impact_parameters", b_pro))
            families.append(Family(f"kerr a={spin:g} retrograde", st, False, b_ret, "critical_impact_parameters", b_ret))
    return families


def turns_outside_plunge(traj: Trajectory, margin: float = PLUNGE_MARGIN) -> float:
    """``|Delta phi| / 2 pi`` up to the last recorded point with ``r >= r_+ + margin``."""
    r = traj.y[:, IDX_R]
    outside = np.flatnonzero(r >= outer_horizon(traj.spacetime) + margin)
    last = int(outside[-1]) if outside.size else 0
    return float(abs(traj.y[last, IDX_PH] - traj.y[0, IDX_PH]) / (2.0 * math.pi))


def measure_ray(st: Spacetime, r0: float, b: float, prograde: bool, integ: IntegratorOptions, term: TerminationOptions) -> tuple[dict[str, Any], Trajectory]:
    """Integrate one equatorial ray and return its measurements and the recorded trajectory."""
    traj = integrate(st, equatorial_photon(st, r0, b, inward=True, prograde=prograde), integ, term, record=True)
    d = traj.diagnostics
    row = {
        "state": traj.state.name, "turns": turns_outside_plunge(traj),
        "turns_total": float(abs(traj.y[-1, IDX_PH] - traj.y[0, IDX_PH]) / (2.0 * math.pi)),
        "affine_length": float(traj.lam[-1]), "n_steps": traj.n_steps, "n_rejected": traj.n_rejected,
        "runtime_s": traj.runtime_s, "closest_approach": closest_approach(traj),
        "max_null_error": d.max_null_error, "max_energy_drift": d.max_energy_drift,
        "max_lz_drift": d.max_lz_drift, "max_carter_drift": d.max_carter_drift,
    }
    return row, traj


def turns_per_decade(offsets: Sequence[float], turns: Sequence[float]) -> float:
    """Least-squares slope of ``turns`` against ``log10(1/delta)`` (``nan`` with fewer than two distinct offsets)."""
    if len(set(offsets)) < 2:
        return math.nan
    return float(np.polyfit(-np.log10(np.asarray(offsets, dtype=float)), np.asarray(turns, dtype=float), 1)[0])


def _analyse(rows: list[dict[str, Any]], families: Sequence[Family], params: NearCriticalParams) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Sensitivity fits per (family, sign, rtol) and the classification flips per (family, delta, sign)."""
    fits, flips = [], []
    for fam in families:
        for sign in SIGNS:
            expected = EXPECTED[sign].name
            for rtol in params.tolerances:
                sel = [r for r in rows if r["family"] == fam.name and r["sign"] == sign and r["rtol"] == rtol]
                good = [r for r in sel if r["state"] == expected]
                reliable = [r["offset"] for r in sel if all(s["state"] == expected for s in sel if s["offset"] >= r["offset"])]
                per_decade = turns_per_decade([r["offset"] for r in good], [r["turns"] for r in good])
                fits.append({
                    "family": fam.name, "sign": sign, "rtol": rtol, "expected": expected,
                    "n_points": len(good), "offsets": [r["offset"] for r in good],
                    "turns": [r["turns"] for r in good],
                    "slope_turns_vs_log10_delta": -per_decade,
                    "turns_per_decade": per_decade,
                    "smallest_reliable_offset": min(reliable) if reliable else math.nan,
                })
            for delta in params.offsets:
                sel = {r["rtol"]: r["state"] for r in rows if r["family"] == fam.name and r["sign"] == sign and r["offset"] == delta}
                flips.append({"family": fam.name, "sign": sign, "offset": delta, "outcomes": sel,
                              "flip": len(set(sel.values())) > 1, "expected": expected,
                              "agrees_with_expected": all(s == expected for s in sel.values())})
    return fits, flips


def _figures(ctx: ExperimentContext, rows: list[dict[str, Any]], families: Sequence[Family], params: NearCriticalParams,
             sample: Trajectory | None) -> list[Any]:
    tight = min(params.tolerances)
    turns_series: dict[str, tuple[list[float], list[float]]] = {}
    length_series: dict[str, tuple[list[float], list[float]]] = {}
    for fam in families:
        for sign in SIGNS:
            sel = sorted((r for r in rows if r["family"] == fam.name and r["sign"] == sign and r["rtol"] == tight), key=lambda r: r["offset"])
            label = f"{fam.name} b_c(1{'+' if sign > 0 else '-'}delta)"
            turns_series[label] = ([r["offset"] for r in sel], [r["turns"] for r in sel])
            length_series[label] = ([r["offset"] for r in sel], [r["affine_length"] for r in sel])
    figures = [
        plot_lines(ctx.report_dir / "turns_vs_offset.png", turns_series, xlabel="relative offset delta", ylabel="orbital turns",
                   logx=True, title=f"Turns near the critical impact parameter (rtol {tight:g})"),
        plot_lines(ctx.report_dir / "affine_length_vs_offset.png", length_series, xlabel="relative offset delta",
                   ylabel="affine length [M]", logx=True, title="Affine length"),
    ]
    fam = families[0]
    by_tol = {}
    for rtol in params.tolerances:
        sel = sorted((r for r in rows if r["family"] == fam.name and r["sign"] == 1 and r["rtol"] == rtol), key=lambda r: r["offset"])
        by_tol[f"rtol {rtol:g}"] = ([r["offset"] for r in sel], [r["turns"] for r in sel])
    figures.append(plot_lines(ctx.report_dir / "turns_by_tolerance.png", by_tol, xlabel="relative offset delta", ylabel="orbital turns",
                              logx=True, title=f"{fam.name}, b = b_c (1 + delta): tolerance dependence"))
    if sample is not None:
        x, y, _ = to_cartesian(sample)
        figures.append(plot_trajectory_xy(ctx.report_dir / "trajectory.png", x, y, outer_horizon(sample.spacetime),
                                          title="Closest escaping near-critical ray"))
    return figures


def _sections(ctx: ExperimentContext, params: NearCriticalParams, families: Sequence[Family], fits: list[dict[str, Any]],
              flips: list[dict[str, Any]]) -> ReportSections:
    cfg = ctx.cfg
    fam_lines = []
    for fam in families:
        text = f"{fam.name}: b_c = {fam.b_c:.12g} M ({fam.b_c_source}), orbit-equation value {fam.b_c_reference:.12g} M, relative difference {abs(fam.b_c - fam.b_c_reference) / fam.b_c_reference:.3e}"
        if fam.bisection:
            text += f"; bisection: {fam.bisection['iterations']} iterations, half-width {fam.bisection['half_width']:.3e} M, {fam.bisection['n_integrations']} integrations at rtol {fam.bisection['rtol']:g}"
        fam_lines.append(text + ".")
    fit_lines = [
        f"{f['family']}, {'escaping' if f['sign'] > 0 else 'captured'} branch, rtol {f['rtol']:g}: {f['turns_per_decade']:.4f} turns per decade from {f['n_points']} points; smallest offset with reliable outcomes {f['smallest_reliable_offset']:g}."
        for f in fits
    ]
    flipped = [f for f in flips if f["flip"]]
    flip_text = ("Classification flips between tolerances: " + "; ".join(
        f"{f['family']} delta={f['offset']:g} {'+' if f['sign'] > 0 else '-'}: " + ", ".join(f"rtol {k:g} {v}" for k, v in f["outcomes"].items()) for f in flipped
    ) + ".") if flipped else "No classification flip between tolerances."
    return ReportSections(
        objective="Quantify how photon trajectories launched near the critical impact parameter depend on the offset delta and on the integration tolerance: orbital turns, affine length, steps, conservation drift, runtime and the capture/escape classification (PROJECT.md section 21, EXP-007).",
        mathematical_model="Equatorial null geodesics of Schwarzschild and Kerr in the Hamiltonian form; b_c is the impact parameter of the unstable circular photon orbit (docs/derivations.md), the separatrix between capture and escape.",
        numerical_method="Bisection on the integrator's own outcome for Schwarzschild (generic bracket [3, 8] M); prograde/retrograde Kerr values from the orbit equations. Rays integrated with the configured method at each tolerance; turns counted outside r_+ + 0.1 M (module docstring).",
        parameters=f"offsets {params.offsets}, tolerances {params.tolerances}, spins {params.spins}, launch radius {params.launch_radius:g} M, escape radius {cfg.termination.escape_radius:g} M, horizon_epsilon {cfg.termination.horizon_epsilon:g}, method {cfg.integration.method}, bisection tolerance {params.bisection_tol:g} M, seed {cfg.experiment.seed}.",
        results="\n".join(fam_lines) + "\n\nSensitivity fits (turns against log10(1/delta)):\n" + "\n".join(fit_lines) + "\n\n" + flip_text,
        error_analysis="Near the unstable orbit a perturbation grows exponentially with the azimuth, so the number of turns grows linearly with log(1/delta) while the truncation error is amplified by the same factor; the smallest offset whose classification is stable across tolerances measures the effective precision of the integrator on this problem.",
        interpretation=(f"The fitted turns-per-decade slopes are measured values. For comparison only, the Schwarzschild strong deflection limit (Bozza 2002) gives ln(10)/(2 pi) = {SCHWARZSCHILD_TURNS_PER_DECADE:.4f} turns per decade on the escaping branch, and the linearisation about the photon orbit gives the same leading coefficient on the captured branch (docs/numerical_analysis.md section 5); at large delta the pre-asymptotic terms bend the fit. "
                        "Flips at small delta mark the tolerance below which the integrator can no longer resolve the separatrix."),
        limitations="Turns of captured rays exclude the plunge below r_+ + 0.1 M (Boyer-Lindquist frame-dragging winding). The bisected b_c inherits the classification error of the integrator at its tolerance; offsets below that error are not resolved by construction. Runtimes are scalar (single-ray) timings.",
        reproducibility=f"Run {ctx.run_id}; seed {cfg.experiment.seed}; manifest in {ctx.run_dir}; results in summary.json.",
    )


def run(cfg: KerrRayConfig) -> RunRecord:
    """Run the near-critical experiment (module docstring)."""

    def body(ctx: ExperimentContext) -> dict[str, Any]:
        params = ctx.parameters("near_critical", NearCriticalParams)
        if cfg.integration.method == "rk4":
            raise ConfigError("near_critical needs an adaptive integration.method (rk45 or dop853) for its tolerance sweep")
        term = termination_options(cfg)
        base = integrator_options(cfg, cfg.integration.method)
        families = build_families(params, cfg, base, term)
        ctx.timer.lap("critical_impact_parameters")
        rows: list[dict[str, Any]] = []
        sample: Trajectory | None = None
        for fam in families:
            for delta in params.offsets:
                for sign in SIGNS:
                    b = fam.b_c * (1.0 + sign * delta)
                    for rtol in params.tolerances:
                        row, traj = measure_ray(fam.spacetime, params.launch_radius, b, fam.prograde,
                                                integrator_options(cfg, cfg.integration.method, rtol=rtol), term)
                        rows.append({"family": fam.name, "spin": fam.spacetime.spin, "prograde": fam.prograde, "b_c": fam.b_c,
                                     "offset": delta, "sign": sign, "b": b, "rtol": rtol, **row})
                        if fam is families[0] and sign == 1 and rtol == min(params.tolerances) and traj.state == TerminationState.ESCAPED:
                            sample = traj
            ctx.logger.info("%s: %d rays done", fam.name, len(rows))
        ctx.timer.lap("rays")
        fits, flips = _analyse(rows, families, params)
        figures = _figures(ctx, rows, families, params, sample)
        table_cols = ["family", "offset", "sign", "rtol", "state", "turns", "affine_length", "n_steps", "max_null_error", "max_carter_drift", "runtime_s"]
        write_report(ctx.report_dir, _sections(ctx, params, families, fits, flips), figures,
                     [markdown_table(rows, columns=table_cols, precision=4)], title="EXP-007 near-critical rays")
        return {
            "families": [{"name": f.name, "spin": f.spacetime.spin, "prograde": f.prograde, "b_c": f.b_c, "b_c_source": f.b_c_source,
                          "b_c_reference": f.b_c_reference, "bisection": f.bisection} for f in families],
            "rows": rows, "fits": fits, "flips": flips,
            "n_flips": sum(1 for f in flips if f["flip"]),
            "schwarzschild_turns_per_decade_comparison": SCHWARZSCHILD_TURNS_PER_DECADE,
            "figures": [str(f) for f in figures],
            "runtime_laps_s": dict(ctx.timer.laps),
        }

    return run_experiment(EXPERIMENT_NAME, cfg, body)
