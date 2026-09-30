"""Figures, tables and ``report.md`` of the Kerr validation run (PROJECT.md section 36).

Everything written here comes from the :class:`~kerrray.validation.kerr.ValidationReport`
built by the checks; no number is typed in.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from kerrray.experiments.base import ExperimentContext
from kerrray.geometry import ergosphere_radius, outer_horizon
from kerrray.photons import Trajectory, to_cartesian
from kerrray.reporting.plots import plot_conservation, plot_lines, plot_trajectory_xy
from kerrray.reporting.report import ReportSections, write_report
from kerrray.reporting.tables import markdown_table
from kerrray.validation import worst_error

if TYPE_CHECKING:  # pragma: no cover
    from kerrray.validation.kerr import KerrValidationParams, ValidationReport

__all__ = ["write_validation_outputs"]


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _figures(report_dir: Path, report: ValidationReport, cons: Trajectory | None, near: Trajectory | None) -> list[Path]:
    figures: list[Path] = []
    limit = report.computed_values["schwarzschild_limit"]
    ref = report.references["schwarzschild_limit"]
    spins = np.asarray(limit["spins"])
    b0 = ref["b_c_schwarzschild"]
    if spins.size >= 2:
        series = {
            "|b_pro(a) - b_c(0)|": (spins, np.abs(np.asarray(limit["b_prograde"]) - b0)),
            "|b_ret(a) - b_c(0)|": (spins, np.abs(np.asarray(limit["b_retrograde"]) - b0)),
            "|even part|": (spins, np.abs(np.asarray(limit["even_part"]))),
            "max |g(a) - g(0)|": (spins, np.asarray(limit["metric_norm"])),
        }
        figures.append(plot_lines(report_dir / "schwarzschild_limit.png", series, xlabel="spin a/M",
                                  ylabel="difference to Schwarzschild [M]", logx=True, logy=True,
                                  title="Schwarzschild limit of the bisected critical impact parameter"))
    rows = report.computed_values["critical_impact"]["rows"]
    if rows:
        s = np.array([r["spin"] for r in rows])
        series = {
            "prograde (bisection)": (s, np.array([r["b_prograde"] for r in rows])),
            "prograde (derived)": (s, np.array([r["b_prograde_derived"] for r in rows])),
            "retrograde (bisection)": (s, np.array([r["b_retrograde"] for r in rows])),
            "retrograde (derived)": (s, np.array([r["b_retrograde_derived"] for r in rows])),
        }
        figures.append(plot_lines(report_dir / "critical_impact.png", series, xlabel="spin a/M",
                                  ylabel="b_c [M]", title="Critical impact parameters"))
    if cons is not None:
        d = cons.diagnostics
        figures.append(plot_conservation(report_dir / "conservation.png", cons.lam,
                                         {"E": d.energy_drift, "L_z": d.lz_drift, "Q": d.carter_drift, "|H|/E^2": d.null_error},
                                         title=f"Conservation along an off-equatorial ray, a = {cons.spacetime.spin:g}"))
    if near is not None:
        x, y, _ = to_cartesian(near)
        st = near.spacetime
        phi = np.linspace(0.0, 2.0 * math.pi, 361)
        r_e = float(ergosphere_radius(st, 0.5 * math.pi))
        figures.append(plot_trajectory_xy(report_dir / "near_extremal.png", x, y, outer_horizon(st),
                                          (r_e * np.cos(phi), r_e * np.sin(phi)),
                                          title=f"Near-extremal prograde ray, a = {st.spin:g}"))
    return figures


def _tables(report: ValidationReport) -> list[str]:
    tables = [markdown_table([
        {"check": c.name, "verdict": "PASS" if c.passed else "FAIL",
         "worst error": _fmt(worst_error(c.error)), "tolerance": ", ".join(f"{k} = {v:g}" for k, v in c.tolerance.items())}
        for c in report.checks
    ])]
    limit, ref = report.computed_values["schwarzschild_limit"], report.references["schwarzschild_limit"]
    tables.append(markdown_table([
        {"spin": a, "b_pro (bisection)": bp, "b_pro (derived)": dp, "b_ret (bisection)": br, "b_ret (derived)": dr,
         "odd part": o, "even part": e, "max |g(a)-g(0)|": n}
        for a, bp, dp, br, dr, o, e, n in zip(
            limit["spins"], limit["b_prograde"], ref["b_prograde_derived"], limit["b_retrograde"],
            ref["b_retrograde_derived"], limit["odd_part"], limit["even_part"], limit["metric_norm"], strict=True)
    ], precision=10))
    tables.append(markdown_table([
        {k: v for k, v in row.items() if k not in ("n_rays", "max_steps", "runtime_s", "iterations", "bracket_width")}
        for row in report.computed_values["critical_impact"]["rows"]
    ], precision=10))
    tables.append(markdown_table([
        {"spin": r["spin"], "state": r["state"], "steps": r["n_steps"], "r_min": r["closest_approach"],
         "E drift": r["max_drift"]["energy"], "L_z drift": r["max_drift"]["lz"], "Q drift": r["max_drift"]["carter"],
         "null error": r["max_drift"]["null"]}
        for r in report.computed_values["conservation"]["rows"]
    ], precision=3))
    tables.append(markdown_table([
        {"spin": r["spin"], "b": r["b"], "r_turn": r["r_turn_analytic"], "r_+": r["r_plus"],
         "flip epsilon": r["outcome_flip_epsilon"], "horizon_epsilon": run["horizon_epsilon"], "state": run["state"],
         "steps": run["n_steps"], "runtime [s]": run["runtime_s"], "r_min": run["closest_approach"],
         "turns": run["turns"], "Q drift": run["max_drift"]["carter"], "null error": run["max_drift"]["null"]}
        for r in report.computed_values["near_extremal"]["rows"] for run in r["runs"]
    ], precision=6))
    return tables


def _interpretation(report: ValidationReport) -> str:
    """Interpretation text built only from the computed values of ``report``."""
    lines = []
    crit = report.check("critical_impact")
    lines.append(f"Bisected critical impact parameters agree with the closed forms to a worst relative error of "
                 f"{_fmt(crit.error['max_relative_error'])} ({'PASS' if crit.passed else 'FAIL'}).")
    lim = report.check("schwarzschild_limit")
    exps = lim.computed["exponents"]
    lines.append("Schwarzschild limit: fitted exponents of |b(a) - 3 sqrt(3) M| against a: prograde "
                 f"{_fmt(exps['prograde'])}, retrograde {_fmt(exps['retrograde'])}; even part at the smallest spin "
                 f"{_fmt(lim.error['even_part_at_smallest_spin'])} M.")
    near = report.check("near_extremal")
    for row in near.computed["rows"]:
        run = row["runs"][-1]
        lines.append(f"Near-extremal a = {row['spin']:g}: {run['state']} after {run['n_steps']} steps, "
                     f"{_fmt(run['turns'])} turns, closest approach {_fmt(run['closest_approach'])} M "
                     f"(r_+ = {_fmt(row['r_plus'])} M, flip epsilon {_fmt(row['outcome_flip_epsilon'])} M).")
    rev = report.check("reversibility")
    lines.append("Reversibility round trip: " + ", ".join(f"{k} {_fmt(v)}" for k, v in rev.error.items()) + ".")
    return "\n".join(f"- {line}" for line in lines)


def _sections(ctx: ExperimentContext, report: ValidationReport, params: KerrValidationParams) -> ReportSections:
    cfg = ctx.cfg
    rev = report.computed_values["reversibility"]
    return ReportSections(
        objective=(
            "Validate the Kerr photon dynamics of PROJECT.md section 13: horizon radius, convergence to "
            "Schwarzschild as a -> 0, conserved quantities, prograde/retrograde critical impact parameters, "
            "near-extremal spins and forward/backward consistency. Every reference is computed at run time."
        ),
        mathematical_model=(
            "Kerr metric in Boyer-Lindquist coordinates (docs/equations.md), null geodesics in the Hamiltonian "
            "form (docs/equations_geodesics.md), separated Carter potentials for the launch states, and the "
            "closed-form photon-orbit quantities of docs/derivations.md as references."
        ),
        numerical_method=(
            f"Integrator {cfg.integration.method} with rtol = {cfg.integration.rtol:g}, atol = {cfg.integration.atol:g}; "
            f"critical impact parameters by bracketed classification of equatorial rays launched inward from "
            f"r0 = {params.launch_radius:g} M with {params.probes_per_iteration} probes per iteration "
            f"(kerrray.validation.kerr_bisection); metric norm on a (r, theta) grid; drifts from the integrator diagnostics."
        ),
        parameters="\n".join(f"- `{k}`: {v}" for k, v in params.__dict__.items()),
        results="\n".join(
            f"- **{c.name}**: {'PASS' if c.passed else 'FAIL'} ({c.description})" for c in report.checks
        ) + "\n\nTables: verdicts; Schwarzschild limit per small spin; critical impact parameters; conservation; near-extremal runs.",
        error_analysis="\n".join(
            f"- {c.name}: " + ", ".join(f"{k} = {_fmt(v)}" for k, v in c.error.items() if not isinstance(v, dict))
            + "".join(f", {k}.{kk} = {_fmt(vv)}" for k, v in c.error.items() if isinstance(v, dict) for kk, vv in v.items())
            for c in report.checks
        ),
        interpretation=_interpretation(report),
        limitations=(
            "Boyer-Lindquist coordinates: captured rays accumulate a 1/Delta null error near the horizon, so the "
            "conservation check uses escaping rays; the near-extremal check reports, not hides, the dependence on "
            "horizon_epsilon. The metric norm is a max over a finite (r, theta) grid. The tolerances are the "
            "provisional D-009 inputs of configs/kerr.yaml (docs/validation.md section 7)."
        ),
        reproducibility=(
            f"Run {ctx.run_id}; seed {cfg.experiment.seed}; configuration copied to {ctx.run_dir / 'config.yaml'}; "
            f"reversibility forward state {rev['forward_state']}, backward state {rev['backward_state']}."
        ),
    )


def write_validation_outputs(
    ctx: ExperimentContext,
    report: ValidationReport,
    params: KerrValidationParams,
    cons_traj: Trajectory | None,
    near_traj: Trajectory | None,
) -> Path:
    """Write the figures, tables and ``report.md`` into ``ctx.report_dir``; return the report path."""
    figures = _figures(ctx.report_dir, report, cons_traj, near_traj)
    tables = _tables(report)
    return write_report(ctx.report_dir, _sections(ctx, report, params), figures, tables,
                        title="Kerr validation (validate_kerr)")
