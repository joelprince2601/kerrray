"""Solver comparison benchmark: fixed-step RK4, adaptive RK45 and DOP853 (EXP-006, PROJECT.md section 19).

:func:`run_solver_benchmark` integrates the fixed ray set of
:mod:`kerrray.benchmarks.rayset` with every solver configuration derived from
``experiment.parameters.convergence`` (``rk4`` at each ``step_sizes`` entry,
``rk45`` and ``dop853`` at each ``tolerances`` entry) and reports, per
configuration, the wall-clock runtime (best of ``repeats``), the accepted
steps per ray, the maximum null-constraint error ``|H| / E_0^2``, the maximum
relative drifts of ``E``, ``L_z`` and the Carter constant, the trajectory
error against the DOP853 reference at ``rtol = 1e-13``, ``atol = 1e-15``
(:func:`kerrray.benchmarks.rayset.trajectory_error`), the failure rate
(``NUMERICAL_FAILURE`` or ``OUT_OF_DOMAIN``) and the number of rays whose
termination state differs from the reference's. Every number is measured.

Runtime is the time of the integration path each solver is provided on:
``rk4`` and ``rk45`` run through :func:`kerrray.geodesics.integrate_batch`
(vectorised over the ray set, one batch per spacetime), ``dop853`` through
:func:`kerrray.geodesics.integrate` (SciPy, one ray at a time). The two paths
have different constant overheads (docs/numerical_methods.md section 6), so
runtimes are comparable within a path and only indicative across paths; the
step counts are path independent.

Adaptive tolerances: ``atol = rtol * (integration.atol / integration.rtol)``
keeps the configured ratio (``1e-2`` in the shipped files).

Launch scale (PROJECT.md section 20, "an equivalent physically appropriate
scale"): the rays start at ``solver.launch_radius`` (default
:data:`DEFAULT_LAUNCH_RADIUS` = 50 M) and escape at the same radius, not at
``observer.radius``. WHAT, the benchmark geodesics are the strong-field
part of the observer's rays, truncated at 50 M; WHY, a fixed step ``h``
costs about ``2 r_0 / h`` steps per escaping ray, so from 1000 M only
``h = 0.1`` fits ``integration.max_steps = 100000`` and the step-size
sequence could not be compared, while the truncation error is generated
where the curvature is large (``r < 10 M``, ``2M/r > 0.2``), which the
truncated rays contain in full; LIMITATION, the weak-field legs between
50 M and the observer, where the error of the adaptive schemes is small but
the far-field phase error ``r delta_phi`` grows with ``r``, are not
included, so absolute trajectory errors at 1000 M would be larger by up to
the ratio of the radii for a pure phase error.

Skipped configurations (never silent, logged and listed in the results and
the report): a fixed step whose estimated step count for an escaping ray,
``2 * launch_radius / h``, exceeds ``integration.max_steps``, and a fixed
step whose predicted cost exceeds ``solver.max_seconds_per_config``. The
prediction scales the measured cost of the previous (larger) fixed step by
the step ratio (the cost of a fixed-step pass is proportional to the number
of steps).

Configuration sub-blocks (docs/decisions.md D-009): ``convergence`` (the
fixed D-009 shape) and ``solver`` (numerical role): ``n_rays`` (default 40),
``repeats`` (3), ``spins`` (``[0.0, 0.9]``), ``launch_radius`` (50.0),
``reference_rtol`` (1e-13), ``reference_atol`` (1e-15) and
``max_seconds_per_config`` (600.0).
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from kerrray.benchmarks.rayset import (
    FAILURE_STATES,
    BenchmarkRay,
    ReferenceSolution,
    build_ray_set,
    comparison_termination,
    reference_solution,
    trajectory_error,
)
from kerrray.benchmarks.solver_config import (
    DEFAULT_LAUNCH_RADIUS,
    DEFAULT_N_RAYS,
    DEFAULT_REPEATS,
    EXPERIMENT_NAME,
    ConvergenceParams,
    SolverConfiguration,
    SolverParams,
    benchmark_termination,
    build_configurations,
    integrator_options,
    minimum_launch_radius,
    termination_options,
)
from kerrray.experiments.base import ExperimentContext, RunRecord, run_experiment
from kerrray.geodesics import IntegratorOptions, TerminationOptions, integrate, integrate_batch
from kerrray.photons import TerminationState
from kerrray.utils.config import KerrRayConfig
from kerrray.utils.config_parsing import ConfigError

__all__ = [
    "DEFAULT_LAUNCH_RADIUS",
    "DEFAULT_N_RAYS",
    "DEFAULT_REPEATS",
    "EXPERIMENT_NAME",
    "ConfigurationResult",
    "ConvergenceParams",
    "RunOutcome",
    "SolverConfiguration",
    "SolverParams",
    "benchmark_termination",
    "build_configurations",
    "build_references",
    "evaluate_all",
    "evaluate_configuration",
    "integrator_options",
    "minimum_launch_radius",
    "prepare",
    "run_ray_set",
    "run_solver_benchmark",
    "termination_options",
]


@dataclass
class RunOutcome:
    """Per-ray results of one pass over the ray set with one solver configuration."""

    state: NDArray[np.int64]
    n_steps: NDArray[np.int64]
    lam: NDArray[np.float64]
    Y: NDArray[np.float64]
    max_null_error: NDArray[np.float64]
    max_energy_drift: NDArray[np.float64]
    max_lz_drift: NDArray[np.float64]
    max_carter_drift: NDArray[np.float64]
    runtime_s: float


def run_ray_set(rays: Sequence[BenchmarkRay], integ: IntegratorOptions, term: TerminationOptions) -> RunOutcome:
    """Integrate the whole ray set once: batched per spacetime for rk4/rk45, scalar for dop853."""
    n = len(rays)
    out = RunOutcome(
        state=np.zeros(n, dtype=np.int64),
        n_steps=np.zeros(n, dtype=np.int64),
        lam=np.zeros(n),
        Y=np.zeros((n, 8)),
        max_null_error=np.zeros(n),
        max_energy_drift=np.zeros(n),
        max_lz_drift=np.zeros(n),
        max_carter_drift=np.zeros(n),
        runtime_s=0.0,
    )
    if integ.method == "dop853":
        for k, ray in enumerate(rays):
            traj = integrate(ray.spacetime, ray.y0, integ, term, record=False)
            out.state[k], out.n_steps[k], out.lam[k], out.Y[k] = int(traj.state), traj.n_steps, traj.lam[-1], traj.y[-1]
            d = traj.diagnostics
            out.max_null_error[k], out.max_energy_drift[k] = d.max_null_error, d.max_energy_drift
            out.max_lz_drift[k], out.max_carter_drift[k] = d.max_lz_drift, d.max_carter_drift
            out.runtime_s += traj.runtime_s
        return out
    groups: dict[float, list[int]] = {}
    for k, ray in enumerate(rays):
        groups.setdefault(ray.spin, []).append(k)
    for indices in groups.values():
        idx = np.asarray(indices)
        st = rays[indices[0]].spacetime
        batch = integrate_batch(st, np.stack([rays[k].y0 for k in indices]), integ, term)
        out.state[idx], out.n_steps[idx], out.lam[idx], out.Y[idx] = batch.state, batch.n_steps, batch.lam, batch.Y
        out.max_null_error[idx], out.max_energy_drift[idx] = batch.max_null_error, batch.max_energy_drift
        out.max_lz_drift[idx], out.max_carter_drift[idx] = batch.max_lz_drift, batch.max_carter_drift
        out.runtime_s += batch.runtime_s
    return out


@dataclass
class ConfigurationResult:
    """Everything measured for one solver configuration."""

    configuration: SolverConfiguration
    production: RunOutcome
    comparison: RunOutcome
    runtime_repeats: list[float]
    trajectory_error: NDArray[np.float64]
    reference_state: NDArray[np.int64]

    @property
    def runtime_s(self) -> float:
        """Best (minimum) wall-clock time over the repeats."""
        return float(min(self.runtime_repeats))

    @property
    def n_rays(self) -> int:
        return int(self.production.state.size)

    @property
    def failures(self) -> NDArray[np.bool_]:
        return np.isin(self.production.state, list(FAILURE_STATES))

    @property
    def mismatches(self) -> NDArray[np.bool_]:
        return self.production.state != self.reference_state

    def outcome_counts(self) -> dict[str, int]:
        """Number of rays per termination state name."""
        return {s.name: int(np.count_nonzero(self.production.state == int(s))) for s in TerminationState if s != TerminationState.RUNNING}

    def summary(self) -> dict[str, Any]:
        """JSON-ready summary of this configuration (``nan`` where nothing was comparable).

        ``*_ok`` maxima are taken over the rays that did not fail (the
        failures are counted by ``failure_rate``; their diagnostics describe
        the breakdown, not the accuracy), ``*_escaped`` over the escaped rays
        (free of the Boyer-Lindquist horizon cancellation,
        docs/numerical_methods.md section 4).
        """
        prod = self.production
        errs = self.trajectory_error
        ok = np.isfinite(errs)
        esc = prod.state == int(TerminationState.ESCAPED)
        good = ~self.failures

        def max_over(values: NDArray[np.float64], sel: NDArray[np.bool_]) -> float:
            return float(np.max(values[sel])) if np.any(sel) else math.nan

        return {
            "solver": self.configuration.solver,
            "setting": self.configuration.setting,
            "label": self.configuration.label,
            "runtime_s": self.runtime_s,
            "runtime_repeats_s": list(self.runtime_repeats),
            "mean_steps": float(np.mean(prod.n_steps)),
            "total_steps": int(np.sum(prod.n_steps)),
            "max_null_error": float(np.max(prod.max_null_error)),
            "max_null_error_ok": max_over(prod.max_null_error, good),
            "max_null_error_escaped": max_over(prod.max_null_error, esc),
            "max_energy_drift": float(np.max(prod.max_energy_drift)),
            "max_energy_drift_ok": max_over(prod.max_energy_drift, good),
            "max_lz_drift": float(np.max(prod.max_lz_drift)),
            "max_lz_drift_ok": max_over(prod.max_lz_drift, good),
            "max_carter_drift": float(np.max(prod.max_carter_drift)),
            "max_carter_drift_ok": max_over(prod.max_carter_drift, good),
            "max_trajectory_error": float(np.max(errs[ok])) if np.any(ok) else math.nan,
            "median_trajectory_error": float(np.median(errs[ok])) if np.any(ok) else math.nan,
            "n_compared": int(np.count_nonzero(ok)),
            "n_failures": int(np.count_nonzero(self.failures)),
            "failure_rate": float(np.count_nonzero(self.failures) / self.n_rays),
            "n_mismatch": int(np.count_nonzero(self.mismatches)),
            "outcomes": self.outcome_counts(),
            "trajectory_error": [float(e) for e in errs],
            "state": [int(s) for s in prod.state],
            "n_steps": [int(s) for s in prod.n_steps],
        }


def build_references(
    rays: Sequence[BenchmarkRay], cfg: KerrRayConfig, params: SolverParams, term: TerminationOptions
) -> list[ReferenceSolution]:
    """DOP853 dense-output reference for every ray under the benchmark termination rules ``term``."""
    return [
        reference_solution(
            ray.spacetime, ray.y0, term, rtol=params.reference_rtol, atol=params.reference_atol,
            lambda_max=cfg.integration.lambda_max, first_step=cfg.integration.step_size,
        )
        for ray in rays
    ]


def evaluate_configuration(
    rays: Sequence[BenchmarkRay],
    refs: Sequence[ReferenceSolution],
    conf: SolverConfiguration,
    term: TerminationOptions,
    *,
    repeats: int,
) -> ConfigurationResult:
    """Run one configuration: ``repeats`` production passes (statistics, best runtime) and one comparison pass."""
    if repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {repeats!r}")
    runs = [run_ray_set(rays, conf.integ, term) for _ in range(repeats)]
    production = min(runs, key=lambda r: r.runtime_s)
    comparison = run_ray_set(rays, conf.integ, comparison_termination(term))
    errors = np.array(
        [trajectory_error(ref, float(comparison.lam[k]), comparison.Y[k], int(comparison.state[k]))
         for k, ref in enumerate(refs)]
    )
    return ConfigurationResult(
        configuration=conf,
        production=production,
        comparison=comparison,
        runtime_repeats=[r.runtime_s for r in runs],
        trajectory_error=errors,
        reference_state=np.array([int(ref.state) for ref in refs], dtype=np.int64),
    )


def evaluate_all(
    ctx: ExperimentContext,
    rays: Sequence[BenchmarkRay],
    refs: Sequence[ReferenceSolution],
    configurations: Sequence[SolverConfiguration],
    term: TerminationOptions,
    params: SolverParams,
) -> tuple[list[ConfigurationResult], list[dict[str, Any]]]:
    """Evaluate every configuration, skipping (and logging) fixed steps predicted to exceed the time budget.

    Returns the results and the list of budget skips. The prediction for a
    fixed step ``h`` is the measured wall time of the previous fixed step
    ``h_prev`` (all passes) times ``h_prev / h``.
    """
    results: list[ConfigurationResult] = []
    skipped: list[dict[str, Any]] = []
    last_fixed: tuple[float, float] | None = None
    for conf in configurations:
        if conf.solver == "rk4" and last_fixed is not None:
            predicted = last_fixed[1] * last_fixed[0] / conf.setting
            if predicted > params.max_seconds_per_config:
                reason = (f"predicted {predicted:.0f} s (measured {last_fixed[1]:.1f} s at h = {last_fixed[0]:g}, "
                          f"scaled by the step ratio) exceeds solver.max_seconds_per_config = "
                          f"{params.max_seconds_per_config:g} s")
                ctx.logger.warning("skipping rk4 h = %g: %s", conf.setting, reason)
                skipped.append({"solver": "rk4", "setting": conf.setting, "reason": reason,
                                "predicted_seconds": predicted})
                continue
        t0 = time.perf_counter()
        res = evaluate_configuration(rays, refs, conf, term, repeats=params.repeats)
        elapsed = time.perf_counter() - t0
        if conf.solver == "rk4":
            last_fixed = (conf.setting, elapsed)
        results.append(res)
        ctx.logger.info("%s: runtime %.3f s, max trajectory error %s", conf.label, res.runtime_s,
                        res.summary()["max_trajectory_error"])
    return results, skipped


def prepare(
    ctx: ExperimentContext, params: SolverParams
) -> tuple[list[BenchmarkRay], list[ReferenceSolution], TerminationOptions]:
    """Build the seeded ray set, the benchmark termination rules and the DOP853 references."""
    cfg = ctx.cfg
    r_min = minimum_launch_radius(params.spins, cfg.black_hole.mass)
    if params.launch_radius < r_min:
        raise ConfigError(
            f"'solver.launch_radius' = {params.launch_radius:g} M is below {r_min:.3g} M, the smallest radius "
            "from which the escaping rays of the set can be launched inward"
        )
    rays = build_ray_set(
        params.n_rays, params.spins, mass=cfg.black_hole.mass, launch_radius=params.launch_radius,
        inclination_deg=cfg.observer.inclination_deg, rng=ctx.rng,
    )
    term = benchmark_termination(cfg, params.launch_radius)
    ctx.logger.info("%d rays from r0 = %g M, computing DOP853 references", len(rays), params.launch_radius)
    refs = build_references(rays, cfg, params, term)
    ctx.timer.lap("references")
    return rays, refs, term


def run_solver_benchmark(cfg: KerrRayConfig, *, n_rays: int | None = None, repeats: int | None = None) -> RunRecord:
    """Run the solver comparison (module docstring) and write manifest, summary, figures and report.

    ``n_rays`` and ``repeats`` override ``experiment.parameters.solver`` (CLI
    options ``--n-rays`` and ``--repeats``).
    """
    from kerrray.benchmarks.solver_report import write_solver_outputs

    def body(ctx: ExperimentContext) -> dict[str, Any]:
        params = ctx.parameters("solver", SolverParams).replace(n_rays=n_rays, repeats=repeats)
        conv = ctx.parameters("convergence", ConvergenceParams)
        rays, refs, term = prepare(ctx, params)
        configurations, skipped = build_configurations(cfg, conv, launch_radius=params.launch_radius)
        for item in skipped:
            ctx.logger.warning("skipping rk4 h = %g: %s", item["setting"], item["reason"])
        results, budget_skips = evaluate_all(ctx, rays, refs, configurations, term, params)
        ctx.timer.lap("configurations")
        return write_solver_outputs(ctx, params, conv, rays, refs, results, skipped + budget_skips)

    return run_experiment(EXPERIMENT_NAME, cfg, body)
