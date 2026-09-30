"""Floating-point precision study, EXP-011 (PROJECT.md section 27; docs/decisions.md D-007).

:func:`run_precision_study` traces the configured camera image (default
64 x 64, ``a = 0.9``, ``i = 60`` deg in ``configs/benchmark.yaml``) in float32
and float64 on every available CPU backend, for each tolerance of
``experiment.parameters.benchmark.precision_tolerances`` (both precisions at
the same ``rtol``, ``atol = rtol * (atol / rtol)_config``, so that a
difference is attributable to the arithmetic, not to the tolerance), and
records per run the runtime, the termination counts, the maximum and median
null error, the conserved-quantity drifts and the shadow mask. For each
backend and tolerance the two precisions are compared: number of pixels that
differ, difference of the mean boundary radius and of the equivalent radius
of the captured mask in units of ``M`` and in pixels (:func:`mask_metrics`),
and, when :mod:`kerrray.raytracing.boundary` is importable and accepts the
traced image, the sub-pixel boundary radii difference.

float32 horizon caveat (docs/numerical_methods.md section 7): float32 cannot
resolve ``horizon_epsilon = 1e-6`` next to ``r_plus ~ 1 M`` (about 17 ulps),
and plunging rays then fail or lose every digit of the null constraint. The
float32 runs therefore use ``horizon_epsilon_float32`` (default ``1e-3``)
while float64 uses the configured value; the report states this. So that
the float32/float64 comparison isolates the arithmetic, a float64 *control*
run at ``horizon_epsilon_float32`` is traced as well (``dtype`` recorded as
``float64``, ``role`` ``"control"``): the comparison rows are float32 against
that control, and a second comparison (float64 configured margin against
float64 control) measures the effect of the margin alone. The escaped-ray
null error and Carter drift are the precision metrics; captured-ray maxima
depend on how close to the horizon a ray is followed.

Runtime: best of ``repeats`` traces of the whole image (the first trace of a
process also pays one-time import costs, which the best-of removes).
"""

from __future__ import annotations

from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from kerrray.benchmarks.cpu import BenchmarkParams, integrator_options, termination_counts, time_best_of
from kerrray.experiments.base import ExperimentContext, RunRecord, run_experiment
from kerrray.geodesics import IntegratorOptions, TerminationOptions
from kerrray.geometry import Spacetime
from kerrray.photons import BatchResult, TerminationState
from kerrray.raytracing.backends.base import Backend, BackendUnavailable, get_backend
from kerrray.raytracing.camera import Camera, initial_states
from kerrray.reporting.plots import plot_lines, plot_shadow
from kerrray.reporting.report import ReportSections, write_report
from kerrray.reporting.tables import markdown_table
from kerrray.utils.config import KerrRayConfig
from kerrray.utils.manifest import collect_environment

__all__ = ["EXPERIMENT_NAME", "boundary_radii", "mask_metrics", "run_metrics", "run_precision_study", "trace_image"]

EXPERIMENT_NAME: Final[str] = "benchmark_precision"
RUN_COLUMNS: Final[tuple[str, ...]] = (
    "backend", "dtype", "role", "rtol", "horizon_epsilon", "runtime_s", "max_null_error_escaped", "median_null_error",
    "max_carter_drift_escaped", "max_null_error", "captured", "escaped", "other",
)
CMP_COLUMNS: Final[tuple[str, ...]] = (
    "backend", "rtol", "compared", "n_differing_pixels", "n_state_mismatch", "boundary_radius_diff_M", "boundary_radius_diff_px",
    "equivalent_radius_diff_M", "subpixel_boundary_max_diff_M", "runtime_ratio_f32_over_f64",
)


def trace_image(engine: Backend, st: Spacetime, cam: Camera, integ: IntegratorOptions, term: TerminationOptions
                ) -> tuple[NDArray[np.bool_], BatchResult, Any]:
    """Captured mask ``(H, W)``, the batch result and the ``ShadowImage`` (or ``None``).

    Uses :func:`kerrray.raytracing.shadow.compute_shadow` when it is importable
    and accepts the backend; otherwise traces the pixels directly with
    ``engine`` and builds the mask from the termination codes.
    """
    try:
        from kerrray.raytracing.shadow import compute_shadow

        img = compute_shadow(st, cam, integ, term, backend=engine)
        return np.asarray(img.captured, dtype=bool), img.result, img
    except Exception:  # noqa: BLE001 - concurrent module; any incompatibility means "fall back"
        result = engine.integrate_batch(st, initial_states(cam, st), integ, term)
        mask = (np.asarray(result.state) == int(TerminationState.CAPTURED)).reshape(cam.shape)
        return mask, result, None


def boundary_radii(img: Any, n_angles: int = 360) -> NDArray[np.float64] | None:
    """Sub-pixel boundary radii from :mod:`kerrray.raytracing.boundary` when available, else ``None``."""
    if img is None:
        return None
    try:
        from kerrray.raytracing.boundary import extract_boundary

        _, radii = extract_boundary(img, n_angles=n_angles)
        return np.asarray(radii, dtype=np.float64)
    except Exception:  # noqa: BLE001 - optional concurrent module
        return None


def _boundary_pixels(mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
    padded = np.pad(mask, 1, constant_values=False)
    inner = padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:]
    return mask & ~inner


def mask_metrics(mask_a: NDArray, mask_b: NDArray, alpha: NDArray, beta: NDArray, pixel_size: float) -> dict[str, Any]:
    """Compare two captured masks on the same pixel grid (``a`` is the reference, e.g. float64).

    Returns the captured counts, the equivalent radii ``sqrt(area / pi)``, the
    mean radius of the boundary pixels about the centroid of ``mask_a``, the
    number of differing pixels and the differences ``a - b`` in ``M`` and pixels.
    """
    a = np.asarray(mask_a, dtype=bool)
    b = np.asarray(mask_b, dtype=bool)
    if a.shape != b.shape or a.shape != np.shape(alpha):
        raise ValueError("masks and pixel coordinates must share one shape")
    n_a, n_b = int(np.count_nonzero(a)), int(np.count_nonzero(b))
    eq_a = float(np.sqrt(n_a * pixel_size**2 / np.pi))
    eq_b = float(np.sqrt(n_b * pixel_size**2 / np.pi))
    if n_a > 0:
        ca, cb = float(np.mean(alpha[a])), float(np.mean(beta[a]))
    else:
        ca = cb = 0.0
    rad = np.sqrt((np.asarray(alpha) - ca) ** 2 + (np.asarray(beta) - cb) ** 2)

    def mean_boundary(mask: NDArray[np.bool_]) -> float | None:
        edge = _boundary_pixels(mask)
        return float(np.mean(rad[edge])) if np.any(edge) else None

    r_a, r_b = mean_boundary(a), mean_boundary(b)
    diff_r = None if r_a is None or r_b is None else r_a - r_b
    return {
        "n_captured_a": n_a,
        "n_captured_b": n_b,
        "n_differing_pixels": int(np.count_nonzero(a ^ b)),
        "equivalent_radius_a_M": eq_a,
        "equivalent_radius_b_M": eq_b,
        "equivalent_radius_diff_M": eq_a - eq_b,
        "equivalent_radius_diff_px": (eq_a - eq_b) / pixel_size,
        "boundary_radius_a_M": r_a,
        "boundary_radius_b_M": r_b,
        "boundary_radius_diff_M": diff_r,
        "boundary_radius_diff_px": None if diff_r is None else diff_r / pixel_size,
        "centroid_a": [ca, cb],
    }


def run_metrics(result: BatchResult) -> dict[str, Any]:
    """Accuracy summaries of one traced image (all rays, escaped rays, captured rays)."""
    escaped = np.asarray(result.state) == int(TerminationState.ESCAPED)
    captured = np.asarray(result.state) == int(TerminationState.CAPTURED)

    def max_over(values: NDArray, sel: NDArray[np.bool_]) -> float | None:
        return float(np.max(values[sel])) if np.any(sel) else None

    counts = termination_counts(result)
    return {
        "counts": counts,
        "captured": counts["CAPTURED"],
        "escaped": counts["ESCAPED"],
        "other": result.n_rays - counts["CAPTURED"] - counts["ESCAPED"],
        "max_null_error": float(np.max(result.max_null_error)),
        "median_null_error": float(np.median(result.max_null_error)),
        "max_null_error_escaped": max_over(result.max_null_error, escaped),
        "max_null_error_captured": max_over(result.max_null_error, captured),
        "max_energy_drift": float(np.max(result.max_energy_drift)),
        "max_lz_drift": float(np.max(result.max_lz_drift)),
        "max_carter_drift": float(np.max(result.max_carter_drift)),
        "max_carter_drift_escaped": max_over(result.max_carter_drift, escaped),
        "mean_steps": float(np.mean(result.n_steps)),
        "mean_rejected": None if result.n_rejected is None else float(np.mean(result.n_rejected)),
    }


def _trace_timed(engine: Backend, st: Spacetime, cam: Camera, integ: IntegratorOptions, term: TerminationOptions,
                 repeats: int) -> tuple[NDArray[np.bool_], BatchResult, Any, float, list[float]]:
    """:func:`trace_image` ``repeats`` times; returns the last mask/result/image, the best and all wall times."""
    out: list[Any] = []

    def once() -> BatchResult:
        out[:] = list(trace_image(engine, st, cam, integ, term))
        return out[1]

    _, best, times = time_best_of(once, repeats)
    return out[0], out[1], out[2], best, times


def _compare(name: str, rtol: float, label: str, a: tuple, b: tuple, alpha: NDArray, beta: NDArray,
             pixel_size: float) -> dict[str, Any]:
    """Compare run ``a`` (reference) with run ``b``; each is ``(mask, result, image, runtime)``."""
    m_a, r_a, i_a, t_a = a
    m_b, r_b, i_b, t_b = b
    cmp = {"backend": name, "rtol": rtol, "compared": label, **mask_metrics(m_a, m_b, alpha, beta, pixel_size),
           "n_state_mismatch": int(np.count_nonzero(np.asarray(r_a.state) != np.asarray(r_b.state))),
           "runtime_ratio_f32_over_f64": t_b / t_a if t_a > 0 else None}
    b_a, b_b = boundary_radii(i_a), boundary_radii(i_b)
    if b_a is not None and b_b is not None and b_a.shape == b_b.shape and np.all(np.isfinite(b_a) & np.isfinite(b_b)):
        d = b_a - b_b
        cmp["subpixel_boundary_max_diff_M"] = float(np.max(np.abs(d)))
        cmp["subpixel_boundary_rms_diff_M"] = float(np.sqrt(np.mean(d**2)))
        cmp["subpixel_boundary_max_diff_px"] = float(np.max(np.abs(d)) / pixel_size)
    else:
        cmp["subpixel_boundary_max_diff_M"] = None
    return cmp


def _body(ctx: ExperimentContext) -> dict[str, Any]:
    cfg = ctx.cfg
    params = ctx.parameters("benchmark", BenchmarkParams)
    st = Spacetime(mass=cfg.black_hole.mass, spin=cfg.black_hole.spin)
    cam = Camera.from_config(cfg.observer, cfg.raytrace)
    alpha, beta = cam.pixel_coordinates()
    atol_ratio = cfg.integration.atol / cfg.integration.rtol
    eps = {"float64": cfg.termination.horizon_epsilon, "float32": params.horizon_epsilon_float32}
    # (label, dtype, horizon_epsilon): the float64 control shares the float32 margin.
    plan = [(dtype, dtype, eps[dtype]) for dtype in params.dtypes]
    if "float32" in params.dtypes and "float64" in params.dtypes and eps["float32"] != eps["float64"]:
        plan.append(("float64_control", "float64", eps["float32"]))
    runs: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    unavailable: dict[str, str] = {}
    masks: dict[tuple[str, float, str], NDArray[np.bool_]] = {}
    for name in params.backends:
        try:
            engine = get_backend(name)
        except (BackendUnavailable, ValueError) as exc:
            unavailable[name] = str(exc)
            ctx.logger.warning("backend %s skipped: %s", name, exc)
            continue
        for rtol in params.precision_tolerances:
            traced: dict[str, tuple[NDArray[np.bool_], BatchResult, Any, float]] = {}
            for label, dtype, horizon_eps in plan:
                integ = integrator_options(cfg, dtype=dtype, rtol=rtol, atol=rtol * atol_ratio)
                term = TerminationOptions(horizon_eps, cfg.termination.escape_radius)
                mask, result, img, best, times = _trace_timed(engine, st, cam, integ, term, params.repeats)
                traced[label] = (mask, result, img, best)
                masks[(name, rtol, label)] = mask
                runs.append({"backend": name, "dtype": dtype, "role": "control" if label.endswith("control") else "main",
                             "rtol": rtol, "atol": integ.atol, "horizon_epsilon": horizon_eps, "runtime_s": best,
                             "runtimes_s": times, "mask_source": "shadow" if img is not None else "direct",
                             **run_metrics(result)})
                ctx.logger.info("%s %s rtol %g: %.2f s, %s", name, label, rtol, best, runs[-1]["counts"])
            if "float32" in traced:
                ref_label = "float64_control" if "float64_control" in traced else "float64"
                if ref_label in traced:
                    comparisons.append(_compare(name, rtol, f"{ref_label} vs float32", traced[ref_label],
                                                traced["float32"], alpha, beta, cam.pixel_size))
            if "float64_control" in traced and "float64" in traced:
                cmp = _compare(name, rtol, "float64 vs float64_control (margin only)", traced["float64"],
                               traced["float64_control"], alpha, beta, cam.pixel_size)
                cmp["runtime_ratio_f32_over_f64"] = None
                comparisons.append(cmp)
    figures = _figures(ctx, runs, masks, alpha, beta)
    run_table = markdown_table([{k: r.get(k) for k in RUN_COLUMNS} for r in runs], columns=list(RUN_COLUMNS), precision=4)
    cmp_table = markdown_table([{k: c.get(k) for k in CMP_COLUMNS} for c in comparisons], columns=list(CMP_COLUMNS), precision=4)
    env = collect_environment()
    results = {
        "parameters": {k: getattr(params, k) for k in BenchmarkParams.__dataclass_fields__},
        "spacetime": {"mass": st.mass, "spin": st.spin},
        "camera": {"radius": cam.radius, "inclination_deg": cam.inclination_deg, "fov": cam.fov,
                   "resolution": cam.resolution, "pixel_size_M": cam.pixel_size},
        "integration": {"method": cfg.integration.method, "atol_over_rtol": atol_ratio},
        "horizon_epsilon": eps,
        "runs": runs,
        "comparisons": comparisons,
        "backends_unavailable": unavailable,
        "hardware": env.hardware(),
        "git_commit": env.git_commit,
        "figures": [str(p) for p in figures],
    }
    _write(ctx, results, [run_table, cmp_table], figures, env)
    return results


def _figures(ctx: ExperimentContext, runs: list[dict[str, Any]], masks: dict, alpha: NDArray, beta: NDArray) -> list:
    figures = []
    for key, label in (("max_null_error_escaped", "max null error, escaped rays"), ("max_carter_drift_escaped", "max Carter drift, escaped rays")):
        series: dict[str, tuple[list[float], list[float]]] = {}
        for r in runs:
            if r.get(key) is not None and r[key] > 0:
                tag = f"{r['backend']} {r['dtype']}" + (" control" if r.get('role') == 'control' else "")
                s = series.setdefault(tag, ([], []))
                s[0].append(r["rtol"])
                s[1].append(r[key])
        if series:
            figures.append(plot_lines(ctx.report_dir / f"{key}_vs_rtol.png", series, xlabel="rtol", ylabel=label,
                                      logx=True, logy=True, title="EXP-011 precision study"))
    if masks:
        backend, rtol = min(masks)[0], min(k[1] for k in masks if k[0] == min(masks)[0])
        for dtype in ("float64", "float32"):
            mask = masks.get((backend, rtol, dtype))
            if mask is not None:
                figures.append(plot_shadow(ctx.report_dir / f"shadow_{backend}_{dtype}.png", alpha, beta, mask,
                                           title=f"{backend} {dtype}, rtol {rtol:g}"))
    return figures


def _write(ctx: ExperimentContext, res: dict[str, Any], tables: list[str], figures: list, env: Any) -> None:
    p = res["parameters"]
    hw = res["hardware"]
    caveat = (f"float32 runs use horizon_epsilon = {res['horizon_epsilon']['float32']:g} instead of the configured "
              f"{res['horizon_epsilon']['float64']:g} because float32 cannot resolve a 1e-6 margin next to r_plus ~ 1 M "
              "(docs/numerical_methods.md section 7). A float64 control run at the float32 margin (role 'control') "
              "isolates the arithmetic: the 'float64_control vs float32' rows compare equal margins, and the "
              "'float64 vs float64_control' rows measure the effect of the margin alone. Escaped-ray maxima are the "
              "precision metrics; captured-ray maxima depend on how close to the horizon a ray is followed.")
    sections = ReportSections(
        objective="EXP-011: float32 versus float64 on the CPU backends: null error, conserved-quantity drift, shadow-boundary difference and runtime.",
        mathematical_model="Kerr null geodesics in Hamiltonian form; shadow = captured-pixel mask of the backward-traced image (docs/equations.md).",
        numerical_method=f"{ctx.cfg.integration.method}; both precisions at each rtol of {p['precision_tolerances']} with atol/rtol as configured; "
                         "boundary compared through the captured masks (mean boundary radius about the float64 centroid, equivalent radius) "
                         "and, when available, kerrray.raytracing.boundary.",
        parameters=f"spin {res['spacetime']['spin']}, inclination {res['camera']['inclination_deg']} deg, resolution {res['camera']['resolution']} "
                   f"(pixel {res['camera']['pixel_size_M']:.4g} M), backends {p['backends']}, dtypes {p['dtypes']}.",
        results=f"{caveat}\n\nBackends unavailable: {res['backends_unavailable'] or 'none'}.",
        error_analysis="Differences in pixels are quantised by the grid: a boundary shift below one pixel shows up only through "
                       "the mean boundary radius or the sub-pixel boundary; escaped-ray null error and Carter drift are the precision metrics.",
        interpretation="See docs/numerical_analysis.md (precision study).",
        limitations=caveat,
        reproducibility=f"Run {ctx.run_id}; hardware {hw['processor']} ({hw['cpu_count']} CPUs, {hw['platform']}); git commit {res['git_commit']}.",
    )
    write_report(ctx.report_dir, sections, figures=figures, tables=tables, title="KerrRay EXP-011: float32 versus float64", environment=env)
    (ctx.report_dir / "precision_tables.md").write_text("\n\n".join(tables) + "\n", encoding="utf-8", newline="\n")


def run_precision_study(cfg: KerrRayConfig) -> RunRecord:
    """Run EXP-011 for ``cfg`` and return the :class:`RunRecord` (module docstring)."""
    return run_experiment(EXPERIMENT_NAME, cfg, _body)
