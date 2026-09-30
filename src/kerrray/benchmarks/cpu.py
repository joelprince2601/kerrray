"""CPU performance benchmark, EXP-009 (PROJECT.md section 26 as redefined by docs/decisions.md D-007).

:func:`run_cpu_benchmark` traces ``N`` backward rays for every ray count of
``experiment.parameters.benchmark.ray_counts`` on every listed backend that
is available and records, per run: the wall-clock time (best of ``repeats``),
the throughput in rays per second, the peak traced memory
(:func:`kerrray.benchmarks.memory.peak_memory`) and the analytic array
estimate (:func:`kerrray.benchmarks.memory.array_bytes`), the maximum null
error and conserved-quantity drifts, the trajectory error against a
tight-tolerance reference, and the termination counts. A log-log plot of
runtime against ray count with the fitted exponent and a Markdown table go
into the report.

Rays: ``N`` seeded random image-plane points ``(alpha, beta)``, uniform in
the square ``[-fov, fov]^2`` of the configured camera, mapped to
past-directed null states with the same tetrad construction as
:func:`kerrray.raytracing.camera.initial_states` (:func:`image_plane_states`),
traced in the configured spacetime from the configured observer. The sets
are nested: the points are drawn as ``(alpha, beta)`` pairs from a generator
re-seeded for every ``N`` with one seed taken from ``experiment.seed``, so
the rays of a smaller ``N`` are the first ``N`` rays of a larger one, and a
count is only sampled when it is actually run (a skipped count costs nothing).

Trajectory error: the first ``reference_rays`` rays are integrated to a
fixed affine parameter ``lambda = 2 r_o`` (no escape, capture only) with the
NumPy backend at ``rtol = reference_rtol`` (float64), and every run integrates
the same rays with its own backend and tolerance; the error is the maximum
over the eight state components of ``|y - y_ref| / max(1, |y_ref|)`` on rays
that reached ``lambda = 2 r_o`` in both runs (captured rays end at different
affine parameters and are excluded, their number is reported). Comparing at
equal ``lambda`` avoids the ill-conditioned comparison of end points on the
horizon or the escape sphere.

Time budget: before each run its cost is predicted from the throughput
measured on the previous run of the same backend (a pilot of
:data:`PILOT_RAYS` rays for the first), as ``(repeats + 1) * N / throughput``
(the timed repeats plus the traced memory run); a run predicted to exceed
``max_seconds_per_run`` is skipped, logged and recorded in the results and
in the report, never silently.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any, Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

from kerrray.benchmarks.memory import WORKING_ARRAYS, array_bytes, format_bytes, peak_memory
from kerrray.experiments.base import ExperimentContext, RunRecord, run_experiment
from kerrray.geodesics import IntegratorOptions, TerminationOptions, null_momentum_pt
from kerrray.geometry import Spacetime, metric
from kerrray.photons import BatchResult, TerminationState
from kerrray.raytracing.backends.base import Backend, BackendUnavailable, get_backend
from kerrray.raytracing.camera import Camera, zamo_tetrad
from kerrray.reporting.plots import loglog_slope, plot_convergence, plot_lines
from kerrray.reporting.report import ReportSections, write_report
from kerrray.reporting.tables import markdown_table
from kerrray.utils.config import SUPPORTED_BACKENDS, SUPPORTED_DTYPES, KerrRayConfig
from kerrray.utils.config_parsing import ConfigError
from kerrray.utils.manifest import collect_environment

__all__ = [
    "EXPERIMENT_NAME", "PILOT_RAYS", "REFERENCE_ESCAPE_RADIUS", "TABLE_COLUMNS", "BenchmarkParams",
    "image_plane_states", "integrator_options", "run_cpu_benchmark", "sample_rays", "termination_counts",
    "time_best_of", "trajectory_error",
]

EXPERIMENT_NAME: Final[str] = "benchmark_cpu"
PILOT_RAYS: Final[int] = 100
"""Rays of the throughput pilot that precedes the first timed run of a backend."""
REFERENCE_ESCAPE_RADIUS: Final[float] = 1.0e9
"""Escape radius of the fixed-``lambda`` trajectory comparison (never reached)."""
TABLE_COLUMNS: Final[tuple[str, ...]] = (
    "backend", "rays", "status", "wall_time_s", "rays_per_s", "peak_traced_MB", "array_estimate_MB",
    "max_null_error", "max_null_error_escaped", "max_energy_drift", "max_carter_drift",
    "trajectory_error_max", "captured", "escaped", "other",
)


@dataclass(frozen=True)
class BenchmarkParams:
    """The ``benchmark`` sub-block of ``experiment.parameters`` (docs/decisions.md D-009).

    ``ray_counts``, ``backends``, ``dtypes`` and ``repeats`` are the keys of
    docs/architecture.md section 7; the others are performance-role
    additions with defaults: the per-run time budget, the size and tolerance
    of the trajectory reference, the float32 capture margin and the
    tolerances of the precision study (:mod:`kerrray.benchmarks.precision`).
    """

    ray_counts: list[int] = field(default_factory=lambda: [1000, 10000, 100000, 1000000])
    backends: list[str] = field(default_factory=lambda: ["numpy", "numba"])
    dtypes: list[str] = field(default_factory=lambda: ["float32", "float64"])
    repeats: int = 3
    max_seconds_per_run: float = 600.0
    reference_rays: int = 200
    reference_rtol: float = 1e-12
    horizon_epsilon_float32: float = 1e-3
    precision_tolerances: list[float] = field(default_factory=lambda: [1e-5, 1e-6, 1e-7])

    def __post_init__(self) -> None:
        if not self.ray_counts or any(n < 1 for n in self.ray_counts):
            raise ConfigError("'benchmark.ray_counts' must be a non-empty list of integers >= 1")
        for name in self.backends:
            if name not in SUPPORTED_BACKENDS:
                raise ConfigError(f"'benchmark.backends' entry {name!r} is not one of {sorted(SUPPORTED_BACKENDS)}")
        for name in self.dtypes:
            if name not in SUPPORTED_DTYPES:
                raise ConfigError(f"'benchmark.dtypes' entry {name!r} is not one of {sorted(SUPPORTED_DTYPES)}")
        if self.repeats < 1 or self.reference_rays < 1:
            raise ConfigError("'benchmark.repeats' and 'benchmark.reference_rays' must be >= 1")
        for key in ("max_seconds_per_run", "reference_rtol", "horizon_epsilon_float32"):
            if not getattr(self, key) > 0.0:
                raise ConfigError(f"'benchmark.{key}' must be > 0")
        if not self.precision_tolerances or any(not tol > 0.0 for tol in self.precision_tolerances):
            raise ConfigError("'benchmark.precision_tolerances' must be a non-empty list of numbers > 0")


def integrator_options(cfg: KerrRayConfig, *, dtype: str | None = None, **overrides: Any) -> IntegratorOptions:
    """Build :class:`IntegratorOptions` from the ``integration`` block (plus keyword overrides)."""
    integ = IntegratorOptions(
        method=cfg.integration.method,
        rtol=cfg.integration.rtol,
        atol=cfg.integration.atol,
        step_size=cfg.integration.step_size,
        max_steps=cfg.integration.max_steps,
        lambda_max=cfg.integration.lambda_max,
        dtype=cfg.raytrace.dtype if dtype is None else dtype,
    )
    return replace(integ, **overrides) if overrides else integ


def image_plane_states(cam: Camera, st: Spacetime, alpha: ArrayLike, beta: ArrayLike) -> NDArray[np.float64]:
    """Past-directed null states (N, 8) for arbitrary celestial coordinates of ``cam``.

    Same construction as :func:`kerrray.raytracing.camera.initial_states`
    (docs/raytracing.md sections 4 and 6): local ZAMO components ``p^(t) = 1``,
    ``p^(r) = sqrt(1 - (alpha^2 + beta^2)/r_o^2)``, ``p^(theta) = beta / r_o``,
    ``p^(phi) = -alpha / r_o``, projected with the tetrad, lowered with the
    metric, reversed for backward tracing and re-normalised by solving the
    null condition for ``p_t`` (the past-directed root, ``dt/dlambda < 0``,
    which is the reversed photon outside the ergosphere). Agreement with
    ``initial_states`` on the pixel grid is tested in ``tests/test_benchmarks_cpu.py``.
    """
    a = np.asarray(alpha, dtype=np.float64).ravel()
    b = np.asarray(beta, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError("alpha and beta must have the same shape")
    r_o, th_o, ph_o = cam.radius, cam.inclination_rad, cam.phi_rad
    rho2 = (a**2 + b**2) / r_o**2
    if np.any(rho2 >= 1.0):
        raise ValueError("alpha^2 + beta^2 must be < radius^2 for every point")
    p_local = np.empty((a.size, 4), dtype=np.float64)
    p_local[:, 0] = 1.0
    p_local[:, 1] = np.sqrt(1.0 - rho2)
    p_local[:, 2] = b / r_o
    p_local[:, 3] = -a / r_o
    p_dn = -((p_local @ zamo_tetrad(st, r_o, th_o)) @ metric(st, r_o, th_o))
    x = np.array([0.0, r_o, th_o, ph_o])
    p_dn[:, 0] = null_momentum_pt(st, x, p_dn[:, 1], p_dn[:, 2], p_dn[:, 3], future_directed=False)
    y = np.empty((a.size, 8), dtype=np.float64)
    y[:, :4] = x
    y[:, 4:] = p_dn
    return y


def sample_rays(cam: Camera, st: Spacetime, n: int, rng: np.random.Generator) -> NDArray[np.float64]:
    """``n`` rays at seeded uniform random image-plane points in ``[-fov, fov]^2`` (module docstring).

    The points are drawn as ``(n, 2)`` pairs, so two fresh generators with the
    same seed give nested sets: ``sample_rays(.., k, g1)`` equals the first
    ``k`` rows of ``sample_rays(.., n, g2)`` for ``k <= n``.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    points = rng.uniform(-cam.fov, cam.fov, size=(n, 2))
    return image_plane_states(cam, st, points[:, 0], points[:, 1])


def termination_counts(result: BatchResult) -> dict[str, int]:
    """Rays per :class:`TerminationState` name (all members, zeros included)."""
    return {member.name: result.count(member) for member in TerminationState}


def time_best_of(fn: Callable[[], BatchResult], repeats: int) -> tuple[BatchResult, float, list[float]]:
    """Call ``fn`` ``repeats`` times; return the last result, the best wall time and every time."""
    times: list[float] = []
    result: BatchResult | None = None
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    assert result is not None
    return result, min(times), times


def trajectory_error(run: BatchResult, ref: BatchResult) -> dict[str, Any]:
    """Fixed-``lambda`` state error of ``run`` against ``ref`` (module docstring)."""
    both = (run.state == int(TerminationState.MAX_AFFINE_PARAMETER)) & (
        ref.state == int(TerminationState.MAX_AFFINE_PARAMETER)
    )
    n_compared = int(np.count_nonzero(both))
    if n_compared == 0:
        return {"max": None, "median": None, "n_compared": 0, "n_excluded": int(both.size)}
    y = run.Y[both].astype(np.float64)
    y_ref = ref.Y[both].astype(np.float64)
    per_ray = np.max(np.abs(y - y_ref) / np.maximum(1.0, np.abs(y_ref)), axis=1)
    return {
        "max": float(np.max(per_ray)),
        "median": float(np.median(per_ray)),
        "n_compared": n_compared,
        "n_excluded": int(both.size - n_compared),
    }


def _null_error_escaped(result: BatchResult) -> float | None:
    escaped = result.state == int(TerminationState.ESCAPED)
    return float(np.max(result.max_null_error[escaped])) if np.any(escaped) else None


def _run_one(
    engine: Backend, st: Spacetime, Y0: NDArray, integ: IntegratorOptions, term: TerminationOptions,
    params: BenchmarkParams, ref: BatchResult, integ_ref: IntegratorOptions, term_ref: TerminationOptions,
) -> dict[str, Any]:
    n = int(Y0.shape[0])
    result, best, times = time_best_of(lambda: engine.integrate_batch(st, Y0, integ, term), params.repeats)
    _, peak = peak_memory(lambda: engine.integrate_batch(st, Y0, integ, term))
    n_ref = ref.n_rays
    traj = trajectory_error(engine.integrate_batch(st, Y0[:n_ref], replace(integ, lambda_max=integ_ref.lambda_max), term_ref), ref)
    counts = termination_counts(result)
    settled = counts["CAPTURED"] + counts["ESCAPED"]
    estimate = array_bytes(n, integ.np_dtype, WORKING_ARRAYS.get(engine.name, WORKING_ARRAYS["numpy"]))
    return {
        "backend": engine.name,
        "rays": n,
        "status": "ok",
        "wall_time_s": best,
        "wall_times_s": times,
        "rays_per_s": n / best,
        "peak_traced_MB": peak / 1e6,
        "array_estimate_MB": estimate / 1e6,
        "max_null_error": float(np.max(result.max_null_error)),
        "max_null_error_escaped": _null_error_escaped(result),
        "median_null_error": float(np.median(result.max_null_error)),
        "max_energy_drift": float(np.max(result.max_energy_drift)),
        "max_lz_drift": float(np.max(result.max_lz_drift)),
        "max_carter_drift": float(np.max(result.max_carter_drift)),
        "trajectory_error_max": traj["max"],
        "trajectory_error_median": traj["median"],
        "trajectory_rays_compared": traj["n_compared"],
        "mean_steps": float(np.mean(result.n_steps)),
        "captured": counts["CAPTURED"],
        "escaped": counts["ESCAPED"],
        "other": n - settled,
        "counts": counts,
    }


def _skip(backend: str, n: int, reason: str, predicted: float | None = None) -> dict[str, Any]:
    return {"backend": backend, "rays": n, "status": "skipped", "reason": reason, "predicted_seconds": predicted}


def _body(ctx: ExperimentContext) -> dict[str, Any]:
    cfg = ctx.cfg
    params = ctx.parameters("benchmark", BenchmarkParams)
    st = Spacetime(mass=cfg.black_hole.mass, spin=cfg.black_hole.spin)
    cam = Camera.from_config(cfg.observer, cfg.raytrace)
    integ = integrator_options(cfg)
    term = TerminationOptions(cfg.termination.horizon_epsilon, cfg.termination.escape_radius)
    n_max = max(params.ray_counts)
    ray_seed = int(ctx.rng.integers(2**63 - 1))
    cache: dict[int, NDArray[np.float64]] = {}

    def rays(n: int) -> NDArray[np.float64]:
        if n not in cache:
            cache.clear()
            cache[n] = sample_rays(cam, st, n, np.random.default_rng(ray_seed))
        return cache[n]

    n_ref = min(params.reference_rays, n_max)
    lambda_ref = min(2.0 * cam.radius, integ.lambda_max)
    atol_ratio = cfg.integration.atol / cfg.integration.rtol
    integ_ref = replace(integ, rtol=params.reference_rtol, atol=params.reference_rtol * atol_ratio,
                        lambda_max=lambda_ref, dtype="float64")
    term_ref = TerminationOptions(cfg.termination.horizon_epsilon, REFERENCE_ESCAPE_RADIUS)
    ref = get_backend("numpy").integrate_batch(st, rays(n_ref), integ_ref, term_ref)
    ctx.timer.lap("reference")
    rows: list[dict[str, Any]] = []
    unavailable: dict[str, str] = {}
    compile_s: dict[str, float] = {}
    for name in params.backends:
        try:
            engine = get_backend(name)
        except (BackendUnavailable, ValueError) as exc:
            unavailable[name] = str(exc)
            ctx.logger.warning("backend %s skipped: %s", name, exc)
            rows.extend(_skip(name, n, f"backend unavailable: {exc}") for n in params.ray_counts)
            continue
        if hasattr(engine, "warm_up"):
            compile_s[name] = float(engine.warm_up(st, integ, term))
        throughput: float | None = None
        for n in params.ray_counts:
            if throughput is None:
                n_pilot = min(n, PILOT_RAYS)
                _, pilot_s, _ = time_best_of(lambda: engine.integrate_batch(st, rays(n_pilot), integ, term), 1)
                throughput = n_pilot / pilot_s
            predicted = (params.repeats + 1) * n / throughput
            if predicted > params.max_seconds_per_run:
                reason = (f"predicted {predicted:.1f} s exceeds max_seconds_per_run = "
                          f"{params.max_seconds_per_run:g} s at {throughput:.1f} rays/s")
                ctx.logger.warning("backend %s, %d rays skipped: %s", name, n, reason)
                rows.append(_skip(name, n, reason, predicted))
                continue
            row = _run_one(engine, st, rays(n), integ, term, params, ref, integ_ref, term_ref)
            throughput = row["rays_per_s"]
            rows.append(row)
            ctx.logger.info("backend %s, %d rays: %.3f s (%.1f rays/s)", name, n, row["wall_time_s"], throughput)
    figures, exponents = _figures(ctx, rows)
    table = markdown_table([{k: r.get(k) for k in TABLE_COLUMNS} for r in rows], columns=list(TABLE_COLUMNS), precision=4)
    env = collect_environment()
    results = {
        "parameters": {k: getattr(params, k) for k in BenchmarkParams.__dataclass_fields__},
        "spacetime": {"mass": st.mass, "spin": st.spin},
        "camera": {"radius": cam.radius, "inclination_deg": cam.inclination_deg, "fov": cam.fov},
        "integration": {"method": integ.method, "rtol": integ.rtol, "atol": integ.atol, "dtype": integ.dtype},
        "ray_seed": ray_seed,
        "reference": {"rays": n_ref, "rtol": integ_ref.rtol, "atol": integ_ref.atol, "lambda_max": lambda_ref,
                      "counts": termination_counts(ref)},
        "runs": rows,
        "skipped": [r for r in rows if r["status"] == "skipped"],
        "backends_unavailable": unavailable,
        "compile_time_s": compile_s,
        "runtime_exponents": exponents,
        "hardware": env.hardware(),
        "git_commit": env.git_commit,
        "figures": [str(p) for p in figures],
    }
    _write(ctx, results, table, figures, env)
    return results


def _figures(ctx: ExperimentContext, rows: list[dict[str, Any]]) -> tuple[list, dict[str, float | None]]:
    figures = []
    exponents: dict[str, float | None] = {}
    series: dict[str, tuple[list[int], list[float]]] = {}
    for row in rows:
        if row["status"] == "ok":
            series.setdefault(row["backend"], ([], []))
            series[row["backend"]][0].append(row["rays"])
            series[row["backend"]][1].append(row["wall_time_s"])
    for name, (ns, ts) in series.items():
        exponents[name] = loglog_slope(ns, ts) if len(ns) >= 2 else None
        figures.append(plot_convergence(ctx.report_dir / f"runtime_vs_rays_{name}.png", ns, ts, "rays N",
                                        "wall time [s]", title=f"EXP-009 runtime, {name} backend"))
    if len(series) >= 2:
        figures.append(plot_lines(ctx.report_dir / "runtime_vs_rays_all.png", {k: v for k, v in series.items()},
                                  xlabel="rays N", ylabel="wall time [s]", logx=True, logy=True, title="EXP-009 runtime"))
    return figures, exponents


def _write(ctx: ExperimentContext, res: dict[str, Any], table: str, figures: list, env: Any) -> None:
    skipped = res["skipped"]
    skip_text = "\n".join(f"- {s['backend']}, {s['rays']} rays: {s['reason']}" for s in skipped) or "- none"
    expo = ", ".join(f"{k}: {v:.3f}" if v is not None else f"{k}: n/a" for k, v in res["runtime_exponents"].items()) or "n/a"
    hw = res["hardware"]
    sections = ReportSections(
        objective="EXP-009: wall-clock runtime, throughput, memory and accuracy of the CPU backends versus ray count.",
        mathematical_model="Null geodesics of the Kerr metric in the Hamiltonian form (docs/equations.md); backward rays from the ZAMO camera.",
        numerical_method=(f"{res['integration']['method']} with rtol {res['integration']['rtol']:g}, atol {res['integration']['atol']:g}, "
                          f"{res['integration']['dtype']}; reference: numpy float64 at rtol {res['reference']['rtol']:g} to lambda = {res['reference']['lambda_max']:g} M "
                          f"on {res['reference']['rays']} rays; peak memory by tracemalloc; array estimate N x 8 x itemsize x working arrays."),
        parameters=f"spin {res['spacetime']['spin']}, inclination {res['camera']['inclination_deg']} deg, r_o {res['camera']['radius']}, fov {res['camera']['fov']}; "
                   f"ray counts {res['parameters']['ray_counts']}, backends {res['parameters']['backends']}, repeats {res['parameters']['repeats']}, "
                   f"budget {res['parameters']['max_seconds_per_run']} s per run.",
        results=f"Fitted runtime exponents (log-log slope of wall time vs N): {expo}.\n\nSkipped runs:\n{skip_text}\n\n"
                f"Backends unavailable: {res['backends_unavailable'] or 'none'}. Compile/warm-up times: {res['compile_time_s'] or 'n/a'}.",
        error_analysis="max_null_error is dominated by captured rays near the horizon (Boyer-Lindquist 1/Delta); "
                       "max_null_error_escaped and the fixed-lambda trajectory error are the accuracy metrics of the run.",
        interpretation="See docs/numerical_analysis.md (CPU benchmark) for the discussion of these measurements.",
        limitations="tracemalloc traces Python-level allocations only (Numba runtime allocations are not counted); "
                    "the trajectory error excludes captured rays; predicted-cost skips are recorded above.",
        reproducibility=f"Run {ctx.run_id}; hardware {hw['processor']} ({hw['cpu_count']} CPUs, {hw['platform']}); git commit {res['git_commit']}.",
    )
    write_report(ctx.report_dir, sections, figures=figures, tables=[table], title="KerrRay EXP-009: CPU benchmark", environment=env)
    (ctx.report_dir / "benchmark_table.md").write_text(table + "\n", encoding="utf-8", newline="\n")


def run_cpu_benchmark(cfg: KerrRayConfig) -> RunRecord:
    """Run EXP-009 for ``cfg`` and return the :class:`RunRecord` (module docstring)."""
    return run_experiment(EXPERIMENT_NAME, cfg, _body)
