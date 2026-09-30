"""``kerrray shadow``: trace, extract the shadow boundary and compare with theory.

Everything ``kerrray trace`` does (:mod:`kerrray.commands.trace`), then:
the sub-pixel boundary of the captured region
(:func:`kerrray.raytracing.boundary.extract_boundary`), the analytic
``r_o -> infinity`` curve of Bardeen (:func:`kerrray.raytracing.boundary.analytic_boundary`),
their radial differences in units of ``M`` and in pixels, raw and with the
finite-distance factor ``1 / sqrt(1 - 2M/r_o)`` applied (docs/shadow.md),
the mean radius, asymmetry and centroid shift, and a figure with the
analytic curve overlaid on the captured mask. The default resolution is the
``raytrace.resolution`` of ``configs/shadow.yaml`` (64).
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

import typer
from rich.console import Console

from kerrray.commands.trace import (
    DEFAULT_CONFIG,
    build_overrides,
    execute,
    load_run_config,
)
from kerrray.experiments.base import ExperimentContext
from kerrray.raytracing.boundary import summarise_shadow
from kerrray.raytracing.shadow import ShadowImage
from kerrray.reporting.console import Row, get_console, render_section
from kerrray.reporting.plots import plot_shadow
from kerrray.utils.config import ConfigError
from kerrray.utils.logging import get_logger

__all__ = ["BOUNDARY_FIGURE_FILENAME", "boundary_results", "boundary_rows", "register", "shadow"]

BOUNDARY_FIGURE_FILENAME: Final[str] = "shadow_boundary.png"
logger = get_logger(__name__)


def boundary_results(summary: Mapping[str, Any]) -> dict[str, Any]:
    """JSON-serialisable boundary metrics from :func:`summarise_shadow`."""
    err, corr = summary["error"], summary["error_corrected"]
    return {
        "boundary_centre": list(summary["centre"]),
        "boundary_n_angles": int(len(summary["angles"])),
        "finite_distance_scale": summary["finite_distance_scale"],
        "boundary_error": dict(err),
        "boundary_error_corrected": dict(corr),
    }


def boundary_rows(summary: Mapping[str, Any]) -> list[Row]:
    """*Shadow boundary* section rows."""
    err, corr = summary["error"], summary["error_corrected"]
    ca, cb = summary["centre"]
    return [
        ("Mean radius", f"{err['numeric_radius_mean']:.5g} M (analytic {err['analytic_radius_mean']:.5g} M)"),
        ("Radius range", f"{err['numeric_radius_min']:.5g} .. {err['numeric_radius_max']:.5g} M"),
        ("Asymmetry", f"{err['numeric_asymmetry']:.4g} M (analytic {err['analytic_asymmetry']:.4g} M)"),
        ("Centroid", f"({ca:.4g}, {cb:.4g}) M"),
        ("Centroid shift", f"{err['centroid_shift_norm']:.3g} M = {err['centroid_shift_norm_px']:.3g} px vs analytic"),
        ("Max error", f"{err['max']:.4g} M = {err['max_px']:.3g} px"),
        ("RMS error", f"{err['rms']:.4g} M = {err['rms_px']:.3g} px"),
        ("Mean error", f"{err['mean']:+.4g} M (numeric - analytic)"),
        ("Finite-distance", f"factor {summary['finite_distance_scale']:.6f}: rms {corr['rms']:.4g} M = {corr['rms_px']:.3g} px"),
    ]


def _boundary_stage(ctx: ExperimentContext, img: ShadowImage, console: Console) -> dict[str, Any]:
    """Boundary extraction, comparison and figure; returns the extra results."""
    try:
        summary = summarise_shadow(img)
    except ValueError as exc:
        logger.warning("boundary extraction skipped: %s", exc)
        render_section(console, "Shadow boundary", [("Skipped", str(exc))])
        return {"boundary_skipped": str(exc)}
    cam, st = img.camera, img.spacetime
    figure = plot_shadow(
        ctx.report_dir / BOUNDARY_FIGURE_FILENAME,
        img.alpha,
        img.beta,
        img.captured,
        analytic_curve=summary["analytic"],
        title=f"a = {st.spin:g}, i = {cam.inclination_deg:g} deg, {cam.resolution}^2 with Bardeen curve",
    )
    render_section(console, "Shadow boundary", boundary_rows(summary))
    results = boundary_results(summary)
    results["boundary_figure"] = str(figure)
    return results


def shadow(
    config: Path = typer.Option(DEFAULT_CONFIG, "--config", help="Configuration file."),
    spin: float | None = typer.Option(None, "--spin", help="Dimensionless spin a/M (|a/M| < 1)."),
    inclination: float | None = typer.Option(None, "--inclination", help="Observer inclination in degrees."),
    resolution: int | None = typer.Option(None, "--resolution", help="Pixels per side (even; config default 64)."),
    fov: float | None = typer.Option(None, "--fov", help="Image half-width in units of M."),
    backend: str | None = typer.Option(None, "--backend", help="numpy | numba | cuda (unavailable)."),
    method: str | None = typer.Option(None, "--method", help="rk4 | rk45."),
    rtol: float | None = typer.Option(None, "--rtol", help="Relative tolerance (rk45)."),
    atol: float | None = typer.Option(None, "--atol", help="Absolute tolerance (rk45)."),
    sets: list[str] = typer.Option([], "--set", help="Dotted override key.path=value (repeatable)."),
) -> None:
    """Reconstruct the black-hole shadow numerically and compare its boundary with theory."""
    try:
        overrides = build_overrides(
            spin=spin, inclination=inclination, resolution=resolution, fov=fov, backend=backend,
            method=method, rtol=rtol, atol=atol, sets=sets,
        )
    except ConfigError as exc:
        get_console(stderr=True).print(f"error: {exc}")
        raise typer.Exit(code=1) from None
    cfg = load_run_config(config, overrides)
    execute(cfg, "shadow", extra=_boundary_stage)


def register(app: typer.Typer) -> None:
    """Add the ``shadow`` command to ``app`` (docs/architecture.md section 6)."""
    app.command("shadow")(shadow)
