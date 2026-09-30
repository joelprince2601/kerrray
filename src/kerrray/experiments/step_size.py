"""Step-size and tolerance study (EXP-006, PROJECT.md section 20).

``run(cfg)`` integrates the fixed ray set of :mod:`kerrray.benchmarks.rayset`
with fixed-step RK4 at every ``experiment.parameters.convergence.step_sizes``
entry and adaptive RK45 at every ``tolerances`` entry, measuring per setting
the trajectory deviation from the DOP853 reference, the null-constraint error,
the conserved-quantity drifts and the runtime exactly as the solver
comparison does (:mod:`kerrray.benchmarks.solver`, same ``solver`` sub-block,
same launch radius, same skip rules). It then fits the empirical convergence
orders from the data: the slope of ``log(error)`` against ``log(h)`` for RK4
and against ``log(rtol)`` for RK45.

Which error is fitted: for each solver, the maximum and the median
trajectory error over the *common* rays, i.e. the rays that have a finite
trajectory error at every setting of that solver (a ray that fails or changes
outcome at one setting would otherwise enter and leave the statistic and bend
the slope). The maximum follows the worst ray of the set (near-critical
rays amplify the local error exponentially), the median the typical ray.
The number of common rays is reported; the all-ray maxima are reported as
well.

Order estimator: :func:`kerrray.validation.convergence.loglog_order` and
:func:`~kerrray.validation.convergence.successive_orders` (schwarzschild
role), imported lazily; if that module is not importable the least-squares
slope :func:`kerrray.reporting.plots.loglog_slope` is used and the estimator
name recorded says so. Nothing about the values is assumed; the fitted
numbers are reported with the points they were fitted to.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Final

import numpy as np

from kerrray.benchmarks.solver import (
    ConfigurationResult,
    ConvergenceParams,
    SolverParams,
    build_configurations,
    evaluate_all,
    prepare,
)
from kerrray.benchmarks.solver_report import ray_rows, runtime_error_series, table_rows
from kerrray.experiments.base import ExperimentContext, RunRecord, run_experiment
from kerrray.reporting.plots import loglog_slope, plot_convergence, plot_lines
from kerrray.reporting.report import ReportSections, write_report
from kerrray.reporting.tables import markdown_table
from kerrray.utils.config import KerrRayConfig

__all__ = ["EXPERIMENT_NAME", "common_ray_errors", "estimate_order", "fit_series", "run"]

EXPERIMENT_NAME: Final[str] = "step_size"
LOCAL_ESTIMATOR: Final[str] = "kerrray.reporting.plots.loglog_slope"
VALIDATION_ESTIMATOR: Final[str] = "kerrray.validation.convergence.loglog_order"


def estimate_order(x: Sequence[float], y: Sequence[float]) -> dict[str, Any]:
    """Fitted order of ``y ~ C x^p`` over the positive finite pairs.

    Returns ``{"order", "successive", "estimator"}``; ``order`` is ``nan``
    with fewer than two usable points (never a made-up number).
    """
    xs = np.asarray(x, dtype=float)
    ys = np.asarray(y, dtype=float)
    usable = np.isfinite(xs) & np.isfinite(ys) & (xs > 0) & (ys > 0)
    try:
        from kerrray.validation.convergence import loglog_order, successive_orders
    except ImportError:
        if np.count_nonzero(usable) < 2 or np.unique(xs[usable]).size < 2:
            return {"order": math.nan, "successive": [], "estimator": LOCAL_ESTIMATOR}
        return {"order": loglog_slope(xs[usable], ys[usable]), "successive": [], "estimator": LOCAL_ESTIMATOR}
    return {"order": float(loglog_order(xs, ys)), "successive": [float(v) for v in successive_orders(xs, ys)],
            "estimator": VALIDATION_ESTIMATOR}


def _of_solver(results: Sequence[ConfigurationResult], solver: str) -> list[ConfigurationResult]:
    return [r for r in results if r.configuration.solver == solver]


def common_ray_errors(
    results: Sequence[ConfigurationResult], solver: str, stat: str = "max"
) -> tuple[list[float], list[float], int]:
    """``(settings, max or median error over the common rays, number of common rays)`` for one solver."""
    reduce = {"max": np.max, "median": np.median}[stat]
    sel = _of_solver(results, solver)
    if not sel:
        return [], [], 0
    common = np.logical_and.reduce([np.isfinite(r.trajectory_error) for r in sel])
    n_common = int(np.count_nonzero(common))
    values = [float(reduce(r.trajectory_error[common])) if n_common else math.nan for r in sel]
    return [r.configuration.setting for r in sel], values, n_common


def fit_series(results: Sequence[ConfigurationResult], solver: str, key: str) -> dict[str, Any]:
    """Settings, values and fitted order of ``key`` for one solver.

    ``key`` is a :meth:`ConfigurationResult.summary` field, or
    ``"common_trajectory_error"`` / ``"common_trajectory_error_median"`` for
    :func:`common_ray_errors` with the maximum / the median.
    """
    n_common = None
    if key.startswith("common_trajectory_error"):
        stat = "median" if key.endswith("median") else "max"
        xs, ys, n_common = common_ray_errors(results, solver, stat)
    else:
        sel = _of_solver(results, solver)
        xs = [r.configuration.setting for r in sel]
        ys = [float(r.summary()[key]) for r in sel]
    fit = estimate_order(xs, ys)
    return {"settings": xs, "values": ys, "n_common_rays": n_common, **fit}


FITS: Final[tuple[tuple[str, str, str], ...]] = (
    ("rk4_trajectory_error", "rk4", "common_trajectory_error"),
    ("rk4_trajectory_error_median", "rk4", "common_trajectory_error_median"),
    ("rk4_trajectory_error_all", "rk4", "max_trajectory_error"),
    ("rk4_null_error", "rk4", "max_null_error_ok"),
    ("rk4_carter_drift", "rk4", "max_carter_drift_ok"),
    ("rk45_trajectory_error", "rk45", "common_trajectory_error"),
    ("rk45_trajectory_error_median", "rk45", "common_trajectory_error_median"),
    ("rk45_trajectory_error_all", "rk45", "max_trajectory_error"),
    ("rk45_null_error", "rk45", "max_null_error_ok"),
)


def _figures(ctx: ExperimentContext, results: Sequence[ConfigurationResult], fits: dict[str, dict[str, Any]]) -> list[Any]:
    figures = []
    for key, name, xlabel in (("rk4_trajectory_error", "rk4", "step size h [M]"),
                              ("rk45_trajectory_error", "rk45", "relative tolerance rtol")):
        f = fits[key]
        pts = [(s, v) for s, v in zip(f["settings"], f["values"], strict=True) if math.isfinite(v) and v > 0]
        if pts:
            figures.append(plot_convergence(
                ctx.report_dir / f"convergence_{name}.png", [p[0] for p in pts], [p[1] for p in pts], xlabel,
                "max trajectory error, common rays [M]", fit_slope=len(pts) >= 2,
                title=f"{name.upper()} convergence ({f['n_common_rays']} common rays)"))
    diag = {}
    for key, label in (("rk4_null_error", "RK4 null error"), ("rk4_carter_drift", "RK4 Carter drift")):
        f = fits[key]
        ok = [(s, v) for s, v in zip(f["settings"], f["values"], strict=True) if math.isfinite(v) and v > 0]
        if ok:
            diag[label] = ([s for s, _ in ok], [v for _, v in ok])
    if diag:
        figures.append(plot_lines(ctx.report_dir / "diagnostics_rk4.png", diag, xlabel="step size h [M]",
                                  ylabel="max over non-failed rays", logx=True, logy=True,
                                  title="RK4 conservation diagnostics"))
    series = runtime_error_series(results)
    if series:
        figures.append(plot_lines(ctx.report_dir / "runtime_vs_error.png", series, xlabel="runtime [s] (best of repeats)",
                                  ylabel="max trajectory error [M]", logx=True, logy=True, title="Cost versus accuracy"))
    return figures


def _order_text(name: str, fit: dict[str, Any]) -> str:
    pts = ", ".join(f"({s:g}, {v:.3e})" for s, v in zip(fit["settings"], fit["values"], strict=True))
    common = f", {fit['n_common_rays']} common rays" if fit.get("n_common_rays") is not None else ""
    if math.isnan(fit["order"]):
        return f"{name}: fewer than two usable points, no fit ({pts}{common})."
    succ = ", ".join(f"{v:.3f}" for v in fit["successive"]) if fit["successive"] else "n/a"
    return (f"{name}: fitted order {fit['order']:.3f} ({fit['estimator']}; successive orders {succ}) "
            f"from the points {pts}{common}.")


def _sections(ctx: ExperimentContext, params: SolverParams, conv: ConvergenceParams, fits: dict[str, dict[str, Any]],
              skipped: Sequence[dict[str, Any]]) -> ReportSections:
    cfg = ctx.cfg
    order_lines = [
        _order_text("RK4 trajectory error versus h (maximum over common rays)", fits["rk4_trajectory_error"]),
        _order_text("RK4 trajectory error versus h (median over common rays)", fits["rk4_trajectory_error_median"]),
        _order_text("RK4 trajectory error versus h (all compared rays)", fits["rk4_trajectory_error_all"]),
        _order_text("RK4 null error versus h", fits["rk4_null_error"]),
        _order_text("RK4 Carter drift versus h", fits["rk4_carter_drift"]),
        _order_text("RK45 trajectory error versus rtol (maximum over common rays)", fits["rk45_trajectory_error"]),
        _order_text("RK45 trajectory error versus rtol (median over common rays)", fits["rk45_trajectory_error_median"]),
        _order_text("RK45 trajectory error versus rtol (all compared rays)", fits["rk45_trajectory_error_all"]),
        _order_text("RK45 null error versus rtol", fits["rk45_null_error"]),
    ]
    skipped_text = ("Skipped: " + "; ".join(f"rk4 h = {s['setting']:g}: {s['reason']}" for s in skipped) + ".") \
        if skipped else "No setting was skipped."
    return ReportSections(
        objective=("Measure how the trajectory deviation, null-constraint error, conservation drift and runtime of "
                   "fixed-step RK4 and adaptive RK45 scale with the step size and the tolerance, and fit the empirical "
                   "convergence orders (PROJECT.md section 20, EXP-006)."),
        mathematical_model=("Null geodesics in the first-order Hamiltonian form (docs/equations_geodesics.md) on the "
                            "fixed ray set of docs/numerical_analysis.md section 2."),
        numerical_method=(
            f"Same measurements as the solver comparison (reference DOP853 at rtol {params.reference_rtol:g}, "
            f"atol {params.reference_atol:g}; trajectory error at matched affine parameter; captured rays compared at "
            "r_+ + 0.1 M). Orders are least-squares slopes of log(error) versus log(h) or log(rtol) over the settings "
            "with a finite, positive error, on the rays compared at every setting."
        ),
        parameters=(
            f"n_rays = {params.n_rays}, repeats = {params.repeats}, step sizes {conv.step_sizes}, tolerances "
            f"{conv.tolerances}, launch and escape radius {params.launch_radius:g} M, max_steps "
            f"{cfg.integration.max_steps}, seed {cfg.experiment.seed}."
        ),
        results="\n".join(order_lines) + "\n\n" + skipped_text,
        error_analysis=(
            "A fixed-step scheme of order p has a global error C h^p once h resolves the strong-field passage; at "
            "coarse steps the pre-asymptotic regime flattens or steepens the slope, and the near-critical rays of the "
            "set amplify the local error exponentially, so the fitted value is an effective order on this ray set. "
            "For RK45 the global error is proportional to rtol only while the local error estimate is representative "
            "of the accumulated error; the fitted exponent measures that proportionality."
        ),
        interpretation=("The fitted orders above are measured values on this ray set; docs/numerical_analysis.md "
                        "compares them with the nominal order 4 of RK4 and with tolerance proportionality for RK45."),
        limitations=("Fixed-step RK4 fails in the horizon plunge (docs/numerical_methods.md section 4), so its failure "
                     "rate is high by construction and the trajectory error is taken over the rays it follows to "
                     "r_+ + 0.1 M. Runtimes are batched NumPy timings. Rays start and escape at the solver launch "
                     "radius, not at the observer radius."),
        reproducibility=f"Run {ctx.run_id}; seed {cfg.experiment.seed}; manifest in {ctx.run_dir}; results in summary.json.",
    )


def run(cfg: KerrRayConfig) -> RunRecord:
    """Run the step-size and tolerance study (module docstring)."""

    def body(ctx: ExperimentContext) -> dict[str, Any]:
        params = ctx.parameters("solver", SolverParams)
        conv = ctx.parameters("convergence", ConvergenceParams)
        rays, refs, term = prepare(ctx, params)
        configurations, skipped = build_configurations(cfg, conv, launch_radius=params.launch_radius,
                                                       include_dop853=False)
        for item in skipped:
            ctx.logger.warning("skipping rk4 h = %g: %s", item["setting"], item["reason"])
        results, budget_skips = evaluate_all(ctx, rays, refs, configurations, term, params)
        skipped = skipped + budget_skips
        ctx.timer.lap("configurations")
        fits = {name: fit_series(results, solver, key) for name, solver, key in FITS}
        for name, fit in fits.items():
            ctx.logger.info("%s: fitted order %s (%s)", name, fit["order"], fit["estimator"])
        rows = table_rows(results)
        figures = _figures(ctx, results, fits)
        write_report(ctx.report_dir, _sections(ctx, params, conv, fits, skipped), figures,
                     [markdown_table(rows, precision=4)], title="EXP-006 step-size study")
        return {
            "n_rays": len(rays),
            "launch_radius": params.launch_radius,
            "fits": fits,
            "fitted_order_rk4": fits["rk4_trajectory_error"]["order"],
            "fitted_order_rk45": fits["rk45_trajectory_error"]["order"],
            "fitted_order_rk4_median": fits["rk4_trajectory_error_median"]["order"],
            "fitted_order_rk45_median": fits["rk45_trajectory_error_median"]["order"],
            "configurations": [r.summary() for r in results],
            "skipped": list(skipped),
            "rays": ray_rows(rays, refs),
            "table": rows,
            "figures": [str(f) for f in figures],
            "runtime_laps_s": dict(ctx.timer.laps),
        }

    return run_experiment(EXPERIMENT_NAME, cfg, body)
