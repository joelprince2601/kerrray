"""Shadow experiments: EXP-003 spin sweep, EXP-004 inclination sweep, EXP-008
resolution convergence (PROJECT.md sections 17, 18 and 37).

``run(cfg)`` dispatches on the ``mode`` key of ``experiment.parameters``
(default ``"spin_sweep"``):

* ``spin_sweep``: one shadow per spin of ``parameters.spin_sweep.spins`` at
  the configured inclination and resolution; a table of boundary metrics
  against spin (mean radius, centroid shift, asymmetry = max radius - min
  radius, error against the analytic curve) and figures.
* ``inclination_sweep``: the same against ``parameters.inclination_sweep.
  inclinations_deg`` at the configured spin (0 degrees is clamped to 1e-3
  degrees by the schema, docs/decisions.md D-008).
* ``convergence``: shadows at ``parameters.convergence.resolutions`` (base
  tolerance) and at ``parameters.convergence.tolerances`` (base
  resolution); the boundary error against the analytic curve and against
  the finest numerical run of each series. A run whose estimated duration,
  ``n_rays / (rays per second of the previous run)``, exceeds
  ``max_seconds_per_run`` is skipped and the skip is recorded in the results
  and the report (never silently).

Every shadow is saved as ``<run_dir>/shadow_<label>.npz`` and drawn with the
analytic curve overlaid; the report goes to ``<report_dir>/report.md``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np

from kerrray.experiments.base import ExperimentContext, RunRecord, run_experiment
from kerrray.geodesics import IntegratorOptions, TerminationOptions
from kerrray.geometry import Spacetime
from kerrray.photons import TerminationState
from kerrray.raytracing.boundary import extract_boundary, summarise_shadow
from kerrray.raytracing.camera import Camera
from kerrray.raytracing.shadow import ShadowImage, compute_shadow
from kerrray.reporting.plots import loglog_slope, plot_convergence, plot_lines, plot_shadow
from kerrray.reporting.report import ReportSections, write_report
from kerrray.reporting.tables import markdown_table
from kerrray.utils.config import ConfigError, KerrRayConfig

__all__ = ["MODES", "ConvergenceParams", "InclinationSweepParams", "SpinSweepParams", "run"]

MODES: Final[tuple[str, ...]] = ("spin_sweep", "inclination_sweep", "convergence")
DEFAULT_MODE: Final[str] = "spin_sweep"


@dataclass(frozen=True)
class SpinSweepParams:
    """``experiment.parameters.spin_sweep`` (docs/decisions.md D-009)."""

    spins: list[float] = field(default_factory=lambda: [0.0, 0.25, 0.5, 0.75, 0.9, 0.99])


@dataclass(frozen=True)
class InclinationSweepParams:
    """``experiment.parameters.inclination_sweep`` (D-009)."""

    inclinations_deg: list[float] = field(default_factory=lambda: [0.0, 30.0, 45.0, 60.0, 75.0, 90.0])


@dataclass(frozen=True)
class ConvergenceParams:
    """``experiment.parameters.convergence`` (D-009) plus the time budget.

    ``step_sizes`` belongs to the step-size study of the numerical role and is
    accepted here only so that the shared sub-block parses; it is not used.
    """

    resolutions: list[int] = field(default_factory=lambda: [64, 128, 256, 512])
    tolerances: list[float] = field(default_factory=lambda: [1e-6, 1e-8, 1e-10, 1e-12])
    step_sizes: list[float] = field(default_factory=lambda: [0.1, 0.01, 0.001, 0.0001])
    max_seconds_per_run: float = 900.0

    def __post_init__(self) -> None:
        if not self.max_seconds_per_run > 0.0:
            raise ConfigError("convergence.max_seconds_per_run must be > 0")


def _options(cfg: KerrRayConfig) -> tuple[IntegratorOptions, TerminationOptions]:
    i, t = cfg.integration, cfg.termination
    return (
        IntegratorOptions(method=i.method, rtol=i.rtol, atol=i.atol, step_size=i.step_size,
                          max_steps=i.max_steps, lambda_max=i.lambda_max, dtype=cfg.raytrace.dtype),
        TerminationOptions(horizon_epsilon=t.horizon_epsilon, escape_radius=t.escape_radius),
    )


def _shadow(ctx: ExperimentContext, cfg: KerrRayConfig, label: str) -> tuple[ShadowImage, dict[str, Any]]:
    """Trace one shadow for ``cfg``, save it, compare its boundary; return image and metrics row."""
    st = Spacetime(mass=cfg.black_hole.mass, spin=cfg.black_hole.spin)
    cam = Camera.from_config(cfg.observer, cfg.raytrace)
    integ, term = _options(cfg)
    img = compute_shadow(st, cam, integ, term, backend=cfg.raytrace.backend)
    img.save(ctx.run_dir / f"shadow_{label}.npz")
    runtime = img.result.runtime_s
    rays_per_s = cam.n_rays / runtime if runtime > 0.0 else math.nan
    row: dict[str, Any] = {
        "label": label, "spin": st.spin, "inclination_deg": cam.inclination_deg,
        "resolution": cam.resolution, "rtol": integ.rtol, "pixel_size": cam.pixel_size,
        "captured": img.count(TerminationState.CAPTURED), "escaped": img.count(TerminationState.ESCAPED),
        "failed": img.n_failed, "other": img.n_other, "runtime_s": runtime, "rays_per_s": rays_per_s,
        "max_null_error": float(img.result.max_null_error.max()),
        "max_energy_drift": float(img.result.max_energy_drift.max()),
    }
    try:
        summary = summarise_shadow(img)
    except ValueError as exc:  # empty mask or shadow touching the border
        ctx.logger.warning("run %s: boundary skipped: %s", label, exc)
        row["boundary_skipped"] = str(exc)
        return img, row
    err, corr = summary["error"], summary["error_corrected"]
    row.update({
        "radius_mean": err["numeric_radius_mean"], "radius_mean_analytic": err["analytic_radius_mean"],
        "asymmetry": err["numeric_asymmetry"], "asymmetry_analytic": err["analytic_asymmetry"],
        "centroid_alpha": err["numeric_centroid_alpha"], "centroid_beta": err["numeric_centroid_beta"],
        "centroid_shift": err["centroid_shift_norm"], "centroid_shift_px": err["centroid_shift_norm_px"],
        "rms_error": err["rms"], "rms_error_px": err["rms_px"], "max_error": err["max"],
        "max_error_px": err["max_px"], "mean_error": err["mean"], "rms_error_corrected": corr["rms"],
    })
    plot_shadow(ctx.report_dir / f"shadow_{label}.png", img.alpha, img.beta, img.captured,
                analytic_curve=summary["analytic"],
                title=f"a = {st.spin:g}, i = {cam.inclination_deg:g} deg, {cam.resolution}^2")
    ctx.logger.info("run %s: %d rays in %.2f s (%.0f rays/s), rms error %.4g M = %.3g px",
                    label, cam.n_rays, runtime, rays_per_s, err["rms"], err["rms_px"])
    return img, row


def _series(rows: Sequence[dict[str, Any]], x: str, y: str) -> tuple[list[float], list[float]]:
    pairs = [(r[x], r[y]) for r in rows if y in r]
    return [p[0] for p in pairs], [p[1] for p in pairs]


def _sweep(ctx: ExperimentContext, mode: str) -> dict[str, Any]:
    """EXP-003 / EXP-004: shadows against spin or inclination."""
    if mode == "spin_sweep":
        values = ctx.parameters("spin_sweep", SpinSweepParams).spins
        key, x_label = "black_hole.spin", "spin"
    else:
        values = ctx.parameters("inclination_sweep", InclinationSweepParams).inclinations_deg
        key, x_label = "observer.inclination_deg", "inclination_deg"
    rows: list[dict[str, Any]] = []
    figures: list[Path] = []
    for value in values:
        label = f"{x_label}_{value:g}"
        _, row = _shadow(ctx, ctx.cfg.with_overrides({key: value}), label)
        rows.append(row)
        if "rms_error" in row:
            figures.append(ctx.report_dir / f"shadow_{label}.png")
    with_boundary = [r for r in rows if "rms_error" in r]
    if len(with_boundary) >= 2:
        figures.append(plot_lines(ctx.report_dir / "radius_vs_parameter.png", {
            "numeric mean radius": _series(with_boundary, x_label, "radius_mean"),
            "analytic mean radius": _series(with_boundary, x_label, "radius_mean_analytic"),
        }, xlabel=x_label, ylabel="mean radius [M]"))
        figures.append(plot_lines(ctx.report_dir / "asymmetry_vs_parameter.png", {
            "numeric max - min radius": _series(with_boundary, x_label, "asymmetry"),
            "analytic max - min radius": _series(with_boundary, x_label, "asymmetry_analytic"),
            "centroid shift vs analytic": _series(with_boundary, x_label, "centroid_shift"),
        }, xlabel=x_label, ylabel="[M]"))
        figures.append(plot_lines(ctx.report_dir / "error_vs_parameter.png", {
            "rms error": _series(with_boundary, x_label, "rms_error"),
            "rms error, finite-distance corrected": _series(with_boundary, x_label, "rms_error_corrected"),
            "max error": _series(with_boundary, x_label, "max_error"),
        }, xlabel=x_label, ylabel="boundary error [M]"))
    columns = [x_label, "captured", "failed", "radius_mean", "radius_mean_analytic", "centroid_alpha",
               "centroid_shift", "asymmetry", "asymmetry_analytic", "rms_error", "rms_error_px",
               "max_error_px", "rms_error_corrected", "rays_per_s"]
    results = {"mode": mode, "parameter": x_label, "values": list(values), "rows": rows,
               "figures": [str(f) for f in figures]}
    _report(ctx, mode, results, [markdown_table(rows, columns=columns, precision=5)], figures)
    return results


def _budget(ctx: ExperimentContext, n_rays: int, rays_per_s: float | None, limit: float, label: str,
            skipped: list[dict[str, Any]]) -> bool:
    """True when the run fits the time budget; otherwise log and record the skip."""
    if rays_per_s is None or not math.isfinite(rays_per_s) or rays_per_s <= 0.0:
        return True
    estimate = n_rays / rays_per_s
    if estimate <= limit:
        return True
    ctx.logger.warning("run %s skipped: estimated %.0f s (%d rays at %.0f rays/s) exceeds "
                       "max_seconds_per_run = %.0f s", label, estimate, n_rays, rays_per_s, limit)
    skipped.append({"label": label, "n_rays": n_rays, "estimated_seconds": estimate,
                    "rays_per_s_previous": rays_per_s, "max_seconds_per_run": limit})
    return False


def _versus_finest(images: list[ShadowImage], rows: list[dict[str, Any]]) -> None:
    """Add the boundary error of every run against the finest run about a common centre."""
    finest = images[-1]
    if not finest.captured.any():
        return
    centre = (float(finest.alpha[finest.captured].mean()), float(finest.beta[finest.captured].mean()))
    _, r_ref = extract_boundary(finest, centre=centre)
    for img, row in zip(images, rows, strict=True):
        try:
            _, r = extract_boundary(img, centre=centre)
        except ValueError:
            continue
        diff = r - r_ref
        row["rms_vs_finest"] = float(np.sqrt(np.mean(diff**2)))
        row["max_vs_finest"] = float(np.max(np.abs(diff)))
        row["rms_vs_finest_px"] = row["rms_vs_finest"] / row["pixel_size"]


def _convergence(ctx: ExperimentContext) -> dict[str, Any]:
    """EXP-008: boundary error against resolution and against tolerance."""
    p = ctx.parameters("convergence", ConvergenceParams)
    cfg = ctx.cfg
    skipped: list[dict[str, Any]] = []
    figures: list[Path] = []
    tables: list[str] = []
    results: dict[str, Any] = {"mode": "convergence", "max_seconds_per_run": p.max_seconds_per_run,
                               "base_rtol": cfg.integration.rtol, "base_resolution": cfg.raytrace.resolution}
    series_specs = [
        ("resolution", "raytrace.resolution", sorted(p.resolutions), "resolution [pixels per side]"),
        ("rtol", "integration.rtol", sorted(p.tolerances, reverse=True), "relative tolerance"),
    ]
    rays_per_s: float | None = None
    for name, key, values, x_label in series_specs:
        images: list[ShadowImage] = []
        rows: list[dict[str, Any]] = []
        for value in values:
            overrides: dict[str, Any] = {key: value}
            if name == "rtol":  # keep the configured atol / rtol ratio
                overrides["integration.atol"] = value * cfg.integration.atol / cfg.integration.rtol
            run_cfg = cfg.with_overrides(overrides)
            label = f"{name}_{value:g}"
            if not _budget(ctx, run_cfg.raytrace.resolution ** 2, rays_per_s, p.max_seconds_per_run, label, skipped):
                continue
            img, row = _shadow(ctx, run_cfg, label)
            rays_per_s = row["rays_per_s"]
            images.append(img)
            rows.append(row)
        if images:
            _versus_finest(images, rows)
        results[f"{name}_rows"] = rows
        columns = [name, "captured", "failed", "pixel_size", "radius_mean", "rms_error", "rms_error_px",
                   "max_error", "rms_error_corrected", "rms_vs_finest", "max_vs_finest", "runtime_s", "rays_per_s"]
        tables.append(markdown_table(rows, columns=columns, precision=5))
        x, y = _series(rows, name, "rms_error")
        if len(x) >= 2:
            results[f"{name}_slope_vs_analytic"] = loglog_slope(x, y)
            figures.append(plot_convergence(ctx.report_dir / f"error_vs_{name}.png", x, y, x_label,
                                            "rms boundary error vs analytic [M]"))
        x2, y2 = _series(rows[:-1], name, "rms_vs_finest")
        if len(x2) >= 2 and all(v > 0 for v in y2):
            results[f"{name}_slope_vs_finest"] = loglog_slope(x2, y2)
            figures.append(plot_convergence(ctx.report_dir / f"error_vs_{name}_finest.png", x2, y2, x_label,
                                            "rms boundary error vs finest run [M]"))
        if images:
            figures.append(ctx.report_dir / f"shadow_{rows[-1]['label']}.png")
    results["skipped"] = skipped
    results["figures"] = [str(f) for f in figures]
    _report(ctx, "convergence", results, tables, figures)
    return results


def _report(ctx: ExperimentContext, mode: str, results: dict[str, Any], tables: list[str],
            figures: list[Path]) -> None:
    """Write ``report.md`` with the nine PROJECT.md section 36 headings."""
    cfg = ctx.cfg
    skipped = results.get("skipped", [])
    skipped_text = ("\n".join(f"- `{s['label']}`: estimated {s['estimated_seconds']:.0f} s for {s['n_rays']} rays "
                              f"at {s['rays_per_s_previous']:.0f} rays/s exceeds the budget of "
                              f"{s['max_seconds_per_run']:.0f} s" for s in skipped) or "No run was skipped.")
    rows = results.get("rows") or results.get("resolution_rows") or []
    failed = sum(int(r.get("failed", 0)) for r in rows)
    sections = ReportSections(
        objective=f"Shadow experiment mode `{mode}` (PROJECT.md sections 17, 18, 37): the shadow is the set "
                  "of pixels whose backward-traced rays are CAPTURED; its boundary is compared with the "
                  "analytic curve of Bardeen (docs/shadow.md).",
        mathematical_model="Kerr null geodesics in Boyer-Lindquist coordinates, Hamiltonian form "
                           "(docs/equations.md); ZAMO camera (docs/raytracing.md); analytic boundary from "
                           "the spherical photon orbits (docs/derivations.md).",
        numerical_method=f"Integrator `{cfg.integration.method}`, rtol {cfg.integration.rtol:g}, atol "
                         f"{cfg.integration.atol:g}, backend `{cfg.raytrace.backend}`, dtype "
                         f"`{cfg.raytrace.dtype}`; bilinear 0.5-level boundary extraction with 360 polar "
                         "rays (docs/shadow.md).",
        parameters=f"Mass {cfg.black_hole.mass:g}, spin {cfg.black_hole.spin:g}, observer radius "
                   f"{cfg.observer.radius:g} M, inclination {cfg.observer.inclination_deg:g} deg, resolution "
                   f"{cfg.raytrace.resolution}, fov {cfg.raytrace.fov:g} M; experiment parameters in "
                   "`config.yaml` of the run.",
        results="Boundary metrics per run (lengths in M unless suffixed `_px`):",
        error_analysis="`rms_error` / `max_error`: radial difference to the analytic (r_o -> infinity) curve; "
                       "`rms_error_corrected`: after the finite-distance factor 1/sqrt(1 - 2M/r_o); "
                       "`rms_vs_finest`: difference to the finest run of the series about a common centre. "
                       + "; ".join(f"log-log slope {k.replace('_slope_', ' ')}: {v:.3f}"
                                   for k, v in results.items() if "_slope_" in k),
        interpretation=f"{failed} of the traced rays failed (NUMERICAL_FAILURE or OUT_OF_DOMAIN). "
                       "The boundary error is bounded by the pixel size; see the tables for the measured values.",
        limitations="Finite observer radius (relative offset M/r_o), pixelated mask (error <= half a pixel), "
                    "clamped on-axis inclination (D-008). Skipped runs:\n" + skipped_text,
        reproducibility=f"Run `{ctx.run_id}`; seed {cfg.experiment.seed}; configuration and manifest in "
                        f"`{ctx.run_dir}`.",
    )
    write_report(ctx.report_dir, sections, figures=figures, tables=tables,
                 title=f"KerrRay shadow experiment: {mode}")


def run(cfg: KerrRayConfig) -> RunRecord:
    """Run the shadow experiment selected by ``experiment.parameters.mode``."""
    mode = str(cfg.experiment.parameters.get("mode", DEFAULT_MODE))
    if mode not in MODES:
        raise ConfigError(f"experiment.parameters.mode must be one of {MODES}, got {mode!r}")

    def body(ctx: ExperimentContext) -> dict[str, Any]:
        return _convergence(ctx) if mode == "convergence" else _sweep(ctx, mode)

    return run_experiment("shadow", cfg, body)
