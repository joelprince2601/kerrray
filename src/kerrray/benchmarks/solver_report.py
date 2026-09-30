"""Tables, figures, report text and the results mapping of the solver benchmark.

Split from :mod:`kerrray.benchmarks.solver` to keep both modules short. All
text here is filled from measured values; the mathematical and numerical
descriptions cite docs/numerical_methods.md and docs/numerical_analysis.md.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from kerrray.benchmarks.rayset import PLUNGE_MARGIN, BenchmarkRay, ReferenceSolution
from kerrray.benchmarks.solver import ConfigurationResult, ConvergenceParams, SolverParams
from kerrray.experiments.base import ExperimentContext
from kerrray.photons import TerminationState
from kerrray.reporting.plots import plot_lines
from kerrray.reporting.report import ReportSections, write_report
from kerrray.reporting.tables import markdown_table

__all__ = ["ray_rows", "runtime_error_series", "table_rows", "write_solver_outputs"]


def ray_rows(rays: Sequence[BenchmarkRay], refs: Sequence[ReferenceSolution]) -> list[dict[str, Any]]:
    """One row per ray: construction parameters and the reference outcome."""
    return [
        {
            "index": ray.index,
            "label": ray.label,
            "spin": ray.spin,
            "kind": ray.kind,
            "equatorial": ray.equatorial,
            "scale": ray.scale,
            "prograde": ray.prograde,
            "reference_state": ref.state.name,
            "reference_lambda_end": ref.lam_end,
            "reference_steps": ref.n_steps,
        }
        for ray, ref in zip(rays, refs, strict=True)
    ]


def table_rows(results: Sequence[ConfigurationResult]) -> list[dict[str, Any]]:
    """Rows of the PROJECT.md section 19 table, one per configuration.

    Null error and drifts are maxima over the rays that did not fail
    (``NUMERICAL_FAILURE``/``OUT_OF_DOMAIN`` rays are counted in
    ``Failure Rate`` instead); ``Null Error (escaped)`` excludes the captured
    rays as well. ``Mismatch`` counts rays whose termination state differs
    from the reference's.
    """
    rows = []
    for res in results:
        s = res.summary()
        rows.append(
            {
                "Solver": res.configuration.name,
                "Setting": res.configuration.setting_label,
                "Runtime [s]": s["runtime_s"],
                "Steps/ray": s["mean_steps"],
                "Null Error": s["max_null_error_ok"],
                "Null Error (escaped)": s["max_null_error_escaped"],
                "Energy Drift": s["max_energy_drift_ok"],
                "Lz Drift": s["max_lz_drift_ok"],
                "Carter Drift": s["max_carter_drift_ok"],
                "Traj. Error max [M]": s["max_trajectory_error"],
                "Traj. Error median [M]": s["median_trajectory_error"],
                "Failure Rate": s["failure_rate"],
                "Mismatch": s["n_mismatch"],
            }
        )
    return rows


def runtime_error_series(results: Sequence[ConfigurationResult], key: str = "max_trajectory_error") -> dict[str, tuple[list[float], list[float]]]:
    """``{solver name: (runtimes, errors)}`` over the configurations with a finite error."""
    series: dict[str, tuple[list[float], list[float]]] = {}
    for res in results:
        s = res.summary()
        value = s[key]
        if not (isinstance(value, float) and math.isfinite(value) and value > 0.0):
            continue
        xs, ys = series.setdefault(res.configuration.name, ([], []))
        xs.append(s["runtime_s"])
        ys.append(value)
    return series


def _outcome_text(counts: dict[str, int]) -> str:
    return ", ".join(f"{name} {n}" for name, n in counts.items() if n)


def _reference_counts(refs: Sequence[ReferenceSolution]) -> dict[str, int]:
    return {s.name: sum(1 for r in refs if r.state == s) for s in TerminationState if s != TerminationState.RUNNING}


def _sections(
    ctx: ExperimentContext,
    params: SolverParams,
    conv: ConvergenceParams,
    rays: Sequence[BenchmarkRay],
    refs: Sequence[ReferenceSolution],
    results: Sequence[ConfigurationResult],
    skipped: Sequence[dict[str, Any]],
) -> ReportSections:
    cfg = ctx.cfg
    ref_counts = _reference_counts(refs)
    ref_runtime = sum(r.runtime_s for r in refs)
    summaries = [r.summary() for r in results]
    best_err = min((s for s in summaries if math.isfinite(s["max_trajectory_error"])), key=lambda s: s["max_trajectory_error"], default=None)
    fastest = min(summaries, key=lambda s: s["runtime_s"])
    worst_fail = max(summaries, key=lambda s: s["failure_rate"])
    lines_results = [
        f"{len(rays)} rays launched inward from r0 = {params.launch_radius:g} M, escaping at the same radius (observer inclination "
        f"{cfg.observer.inclination_deg:g} deg for the off-equatorial family), spins {params.spins}; "
        f"reference outcomes: {_outcome_text(ref_counts)}; reference cost {ref_runtime:.2f} s "
        f"({sum(r.n_steps for r in refs)} accepted DOP853 steps).",
        "",
        "Per configuration (production termination rules): " + "; ".join(
            f"{s['label']}: {_outcome_text(s['outcomes'])}" for s in summaries
        ) + ".",
    ]
    if skipped:
        lines_results += ["", "Skipped fixed-step settings: " + "; ".join(
            f"h = {item['setting']:g}: {item['reason']}" for item in skipped) + "."]
    else:
        lines_results += ["", "No configuration was skipped."]
    interpretation = []
    if best_err is not None:
        interpretation.append(f"Smallest maximum trajectory error: {best_err['label']} ({best_err['max_trajectory_error']:.3e} M in {best_err['runtime_s']:.3f} s).")
    interpretation.append(f"Fastest configuration: {fastest['label']} ({fastest['runtime_s']:.3f} s, max trajectory error {fastest['max_trajectory_error']:.3e} M).")
    interpretation.append(f"Highest failure rate: {worst_fail['label']} ({worst_fail['failure_rate']:.2f}, outcomes {_outcome_text(worst_fail['outcomes'])}).")
    interpretation.append(
        "Energy and L_z drifts are exactly zero in the Hamiltonian formulation whenever the reported value is 0 "
        "(dp_t/dlambda = dp_phi/dlambda = 0 identically); the Carter drift and the null error measure the truncation error."
    )
    return ReportSections(
        objective=(
            "Compare fixed-step RK4, adaptive Dormand-Prince RK45 and SciPy DOP853 on a fixed set of Schwarzschild and Kerr "
            "photon geodesics: runtime, accepted steps, null-constraint error, conserved-quantity drift, trajectory error "
            "against a tight-tolerance reference, failure rate and outcome mismatches (PROJECT.md section 19, EXP-006)."
        ),
        mathematical_model=(
            "Null geodesics of the Kerr metric in Boyer-Lindquist coordinates, first-order Hamiltonian form "
            "dx/dlambda = g^{mu nu} p_nu, dp_mu/dlambda = -(1/2) d_mu g^{ab} p_a p_b (docs/equations_geodesics.md). "
            "Rays are built from exact constants of motion (E, L_z, Q) relative to the computed critical impact "
            "parameters and the analytic shadow curve (docs/numerical_analysis.md section 2)."
        ),
        numerical_method=(
            f"Reference: DOP853 with dense output at rtol = {params.reference_rtol:g}, atol = {params.reference_atol:g}. "
            "Trajectory error = Euclidean distance between the Cartesian embeddings of the solver end state and the reference "
            f"state at the same affine parameter; captured rays are compared at r = r_+ + {PLUNGE_MARGIN:g} M "
            "(docs/numerical_analysis.md section 2). Runtime is the best of "
            f"{params.repeats} repeats on integrate_batch (rk4, rk45) or integrate (dop853)."
        ),
        parameters=(
            f"n_rays = {params.n_rays}, repeats = {params.repeats}, launch and escape radius {params.launch_radius:g} M, "
            f"step sizes {conv.step_sizes}, tolerances {conv.tolerances} "
            f"(atol = rtol x {cfg.integration.atol / cfg.integration.rtol:g}), max_steps = {cfg.integration.max_steps}, "
            f"lambda_max = {cfg.integration.lambda_max:g}, horizon_epsilon = {cfg.termination.horizon_epsilon:g}, "
            f"time budget per fixed-step configuration {params.max_seconds_per_config:g} s, seed = {cfg.experiment.seed}."
        ),
        results="\n".join(lines_results),
        error_analysis=(
            "Trajectory errors are absolute distances in units of M at the solver's own final affine parameter; "
            "a NaN means the ray failed or its outcome differed from the reference (no matched point exists). "
            "Null error is max |H| / E_0^2 over all accepted steps; for captured rays it is dominated by the 1/Delta "
            "cancellation of the Boyer-Lindquist horizon approach (docs/numerical_methods.md section 4), so the value "
            "over escaped rays only is also recorded in summary.json (max_null_error_escaped)."
        ),
        interpretation="\n".join(interpretation),
        limitations=(
            "Runtimes of the batched (rk4, rk45) and scalar SciPy (dop853) paths carry different constant overheads and "
            "are only indicative across paths. The plunge below r_+ + 0.1 M is excluded from the trajectory error. "
            f"Rays start and escape at {params.launch_radius:g} M rather than at the observer radius "
            f"{cfg.observer.radius:g} M (module docstring of kerrray.benchmarks.solver: the weak-field legs are not included). "
            "The maximum trajectory error is dominated by the near-critical rays, whose errors are amplified "
            "exponentially by the unstable photon orbit; the median is the typical ray. "
            "Fixed-step RK4 cannot resolve the horizon approach, so its plunging rays end OUT_OF_DOMAIN or "
            "NUMERICAL_FAILURE and are counted as failures. The reference itself carries an error of order its tolerance."
        ),
        reproducibility=(
            f"Run {ctx.run_id}; configuration and manifest in {ctx.run_dir}; results in summary.json; "
            f"random seed {cfg.experiment.seed} fixes the ray set."
        ),
    )


def write_solver_outputs(
    ctx: ExperimentContext,
    params: SolverParams,
    conv: ConvergenceParams,
    rays: Sequence[BenchmarkRay],
    refs: Sequence[ReferenceSolution],
    results: Sequence[ConfigurationResult],
    skipped: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Write the figures and the report into ``ctx.report_dir`` and return the results mapping."""
    rows = table_rows(results)
    table = markdown_table(rows, precision=4)
    figures = []
    series = runtime_error_series(results)
    if series:
        figures.append(
            plot_lines(ctx.report_dir / "runtime_vs_error.png", series, xlabel="runtime [s] (best of repeats)",
                       ylabel="max trajectory error [M]", logx=True, logy=True, title="Solver comparison")
        )
    null_series = runtime_error_series(results, "max_null_error")
    if null_series:
        figures.append(
            plot_lines(ctx.report_dir / "runtime_vs_null_error.png", null_series, xlabel="runtime [s] (best of repeats)",
                       ylabel="max |H| / E0^2", logx=True, logy=True, title="Null-constraint error")
        )
    sections = _sections(ctx, params, conv, rays, refs, results, skipped)
    write_report(ctx.report_dir, sections, figures, [table], title="EXP-006 solver comparison")
    return {
        "n_rays": len(rays),
        "spins": list(params.spins),
        "launch_radius": params.launch_radius,
        "escape_radius": params.launch_radius,
        "repeats": params.repeats,
        "reference": {
            "rtol": params.reference_rtol,
            "atol": params.reference_atol,
            "outcomes": _reference_counts(refs),
            "runtime_s": float(sum(r.runtime_s for r in refs)),
            "total_steps": int(sum(r.n_steps for r in refs)),
        },
        "configurations": [r.summary() for r in results],
        "skipped": list(skipped),
        "rays": ray_rows(rays, refs),
        "table": rows,
        "figures": [str(f) for f in figures],
        "runtime_laps_s": dict(ctx.timer.laps),
        "best_max_trajectory_error": _best_error(results),
    }


def _best_error(results: Sequence[ConfigurationResult]) -> float:
    finite = [e for e in (r.summary()["max_trajectory_error"] for r in results) if math.isfinite(e)]
    return min(finite) if finite else math.nan
