"""``kerrray render --spin 0.9 --inclination 60 --resolution 64 --r-out 20``:
render the simplified accretion disk (PROJECT.md sections 23 and 24;
docs/rendering.md section 4).

Loads ``configs/shadow.yaml`` (or ``--config``), applies the overrides to
``black_hole.spin``, ``observer.inclination_deg``, ``raytrace.resolution``,
``raytrace.fov``, ``raytrace.backend`` and the ``disk`` sub-block of ``experiment.parameters``, and runs the ``render``
experiment through :func:`kerrray.experiments.base.run_experiment`: the disk
image is rendered with :func:`kerrray.raytracing.renderer.render_disk`, the
arrays are saved as ``disk_image.npz`` in the run directory, and PNGs with a
linear stretch, a log stretch and the redshift map, plus ``report.md``, go to
the report directory. Every printed number is computed by the render.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

import numpy as np
import typer

from kerrray.experiments.base import ExperimentContext, RunRecord, run_experiment
from kerrray.geometry import Spacetime
from kerrray.physics.accretion import DISK_BLOCK, DiskModel, disk_params
from kerrray.raytracing.backends import BackendUnavailable, get_backend
from kerrray.raytracing.camera import Camera
from kerrray.raytracing.renderer import (
    DiskImage,
    integrator_options_from_config,
    render_disk,
    save_disk_image,
    termination_options_from_config,
)
from kerrray.reporting.console import Row, get_console, render_section
from kerrray.reporting.figures import disk_figure
from kerrray.reporting.report import ReportSections, write_report
from kerrray.utils.config import ConfigError, KerrRayConfig, load_config
from kerrray.utils.logging import get_logger

__all__ = ["DEFAULT_CONFIG", "EXPERIMENT_NAME", "NPZ_FILENAME", "register", "render", "run"]

DEFAULT_CONFIG: Final[str] = "configs/shadow.yaml"
EXPERIMENT_NAME: Final[str] = "render"
NPZ_FILENAME: Final[str] = "disk_image.npz"

logger = get_logger(__name__)


def _default_config_path() -> Path:
    local = Path(DEFAULT_CONFIG)
    return local if local.is_file() else Path(__file__).resolve().parents[3] / DEFAULT_CONFIG


def _image_summary(image: DiskImage) -> dict[str, Any]:
    hit = image.hit
    g = image.g[hit]
    summary: dict[str, Any] = {
        "resolution": image.camera.resolution,
        "n_rays": image.camera.n_rays,
        "counts": image.counts(),
        "n_hits": image.n_hits,
        "backend": image.backend,
        "disk": {
            "r_in": image.disk.r_in,
            "r_out": image.disk.r_out,
            "emissivity_index": image.disk.emissivity_index,
            "intensity_law": image.disk.intensity_law,
            "prograde": image.disk.prograde,
        },
        "intensity_max": float(np.max(image.intensity)),
        "intensity_sum": float(np.sum(image.intensity)),
        "max_null_error": float(np.max(image.result.max_null_error)),
        "max_null_error_disk": float(np.max(image.result.max_null_error[hit.ravel()])) if image.n_hits else None,
        "max_energy_drift": float(np.max(image.result.max_energy_drift)),
        "integration_runtime_s": image.result.runtime_s,
    }
    if g.size:
        alpha = image.alpha[hit]
        summary.update(
            {
                "g_min": float(np.min(g)),
                "g_max": float(np.max(g)),
                "g_mean": float(np.mean(g)),
                "g_mean_alpha_negative": float(np.mean(g[alpha < 0])) if np.any(alpha < 0) else None,
                "g_mean_alpha_positive": float(np.mean(g[alpha > 0])) if np.any(alpha > 0) else None,
                "r_hit_min": float(np.min(image.r_hit[hit])),
                "r_hit_max": float(np.max(image.r_hit[hit])),
            }
        )
    return summary


def _interpretation(image: DiskImage, summary: dict[str, Any]) -> str:
    """Interpretation text driven by the computed mean ``g`` on either side of the image.

    For a prograde disk the gas on the ``alpha < 0`` side moves towards the
    observer when ``a >= 0`` (``alpha > 0`` when ``a < 0``; docs/rendering.md
    section 5), so the Doppler effect predicts a larger mean ``g`` there. The
    text states whether the computed means agree with that expectation.
    """
    neg, pos = summary.get("g_mean_alpha_negative"), summary.get("g_mean_alpha_positive")
    if not image.n_hits or neg is None or pos is None:
        return "Too few disk pixels on both sides of the image to compare the approaching and receding sides."
    approaching_negative = (image.spacetime.a >= 0.0) == image.disk.prograde
    g_app, g_rec = (neg, pos) if approaching_negative else (pos, neg)
    side = "alpha < 0" if approaching_negative else "alpha > 0"
    verdict = "consistent with" if g_app > g_rec else "NOT consistent with"
    return (
        f"Mean g on the approaching side ({side}) is {g_app:.4f} and on the receding side {g_rec:.4f}: "
        f"{verdict} the Doppler boost of gas moving towards the observer (docs/rendering.md section 5). "
        f"The smallest g ({summary['g_min']:.4f}) combines the gravitational redshift near r_in with the recession."
    )


def _sections(cfg: KerrRayConfig, image: DiskImage, summary: dict[str, Any]) -> ReportSections:
    st, disk, cam = image.spacetime, image.disk, image.camera
    g_text = (
        f"g in [{summary['g_min']:.4f}, {summary['g_max']:.4f}], mean {summary['g_mean']:.4f}; "
        f"mean g on alpha < 0: {summary['g_mean_alpha_negative']}, on alpha > 0: {summary['g_mean_alpha_positive']}. "
        if image.n_hits else "No ray hit the disk. "
    )
    return ReportSections(
        objective="Image of a simplified accretion disk around the black hole: lensed geometry, gravitational and Doppler shift, and the observed intensity (PROJECT.md sections 23 and 24).",
        mathematical_model=f"Null geodesics of the Kerr metric (mass {st.mass:g}, spin {st.spin:g}) traced backwards from a ZAMO camera (docs/raytracing.md); emitter on circular Keplerian orbits (BPT 1972 eq. 2.16), g = E / (-p.u) (Cunningham 1975); I_obs = g^{disk.redshift_power} (r/r_in)^(-{disk.emissivity_index:g}) (docs/rendering.md).",
        numerical_method=f"Batched {cfg.integration.method} integration (rtol {cfg.integration.rtol:g}, atol {cfg.integration.atol:g}, dtype {cfg.raytrace.dtype}, backend {image.backend}) with the disk-plane event (linear interpolation of the theta = pi/2 crossing).",
        parameters=f"camera r = {cam.radius:g} M, inclination {cam.inclination_deg:g} deg, fov {cam.fov:g} M, {cam.resolution} x {cam.resolution} pixels; disk r_in = {disk.r_in:.6g} M, r_out = {disk.r_out:g} M, p = {disk.emissivity_index:g}, law {disk.intensity_law}, prograde {disk.prograde}.",
        results=f"Termination counts {summary['counts']}; {summary['n_hits']} disk pixels. {g_text}Intensity max {summary['intensity_max']:.6g} (arbitrary units, I_em(r_in) = 1).",
        error_analysis=f"Max null-constraint error |H|/E^2 = {summary['max_null_error']:.3e} over all rays (captured rays approach the horizon, where the Boyer-Lindquist 1/Delta cancellation dominates, docs/numerical_methods.md) and {summary['max_null_error_disk']} over disk pixels; max energy drift {summary['max_energy_drift']:.3e}. The disk crossing is located by linear interpolation between accepted steps (O(h^2)).",
        interpretation=_interpretation(image, summary),
        limitations="Simplified optically thin, geometrically thin, single-surface emission with a power-law emissivity; no radiative transfer, returning radiation, disk thickness or spectral shape (docs/rendering.md section 3). Not a GRMHD model.",
        reproducibility=f"Configuration and manifest in the run directory, arrays in {NPZ_FILENAME}; seed {cfg.experiment.seed}.",
    )


def _body(ctx: ExperimentContext) -> dict[str, Any]:
    cfg = ctx.cfg
    st = Spacetime(mass=cfg.black_hole.mass, spin=cfg.black_hole.spin)
    params = disk_params(cfg)
    disk = DiskModel.from_params(st, params)
    cam = Camera.from_config(cfg.observer, cfg.raytrace)
    image = render_disk(
        st, cam, integrator_options_from_config(cfg), termination_options_from_config(cfg), disk,
        backend=cfg.raytrace.backend,
    )
    ctx.timer.lap("render")
    summary = _image_summary(image)
    summary["disk_enabled_flag"] = params.enabled
    npz = save_disk_image(ctx.run_dir / NPZ_FILENAME, image)
    figures = [
        disk_figure(ctx.report_dir / "disk_linear.png", image, quantity="intensity", stretch="linear",
                    title=f"disk intensity, spin {st.spin:g}, i = {cam.inclination_deg:g} deg"),
        disk_figure(ctx.report_dir / "disk_log.png", image, quantity="intensity", stretch="log",
                    title="disk intensity (log stretch)"),
        disk_figure(ctx.report_dir / "disk_redshift.png", image, quantity="redshift",
                    title="redshift factor g"),
    ]
    report = write_report(ctx.report_dir, _sections(cfg, image, summary), figures=figures,
                          title=f"KerrRay disk render (spin {st.spin:g}, i = {cam.inclination_deg:g} deg)")
    summary["npz"] = str(npz)
    summary["figures"] = [str(p) for p in figures]
    summary["report"] = str(report)
    summary["stage_runtime_s"] = dict(ctx.timer.laps)
    return summary


def run(cfg: KerrRayConfig) -> RunRecord:
    """Run the ``render`` experiment for ``cfg`` (manifest, npz, PNGs, report)."""
    return run_experiment(EXPERIMENT_NAME, cfg, _body)


def render(
    spin: float | None = typer.Option(None, "--spin", help="Dimensionless spin a/M."),
    inclination: float | None = typer.Option(None, "--inclination", help="Observer inclination in degrees."),
    resolution: int | None = typer.Option(None, "--resolution", help="Pixels per side (even)."),
    fov: float | None = typer.Option(None, "--fov", help="Image half-width in units of M (should exceed r_out)."),
    backend: str | None = typer.Option(None, "--backend", help="numpy | numba | cuda (unavailable)."),
    r_out: float | None = typer.Option(None, "--r-out", help="Outer disk radius in units of M."),
    r_in: float | None = typer.Option(None, "--r-in", help="Inner disk radius in units of M (default: ISCO)."),
    emissivity_index: float | None = typer.Option(None, "--emissivity-index", help="Exponent p of (r/r_in)^(-p)."),
    intensity_law: str | None = typer.Option(None, "--intensity-law", help="g3 or g4."),
    config: Path = typer.Option(None, "--config", help=f"Configuration file (default {DEFAULT_CONFIG})."),
) -> None:
    """Render the simplified accretion disk (linear and log PNGs, redshift map, npz)."""
    overrides: dict[str, Any] = {}
    if spin is not None:
        overrides["black_hole.spin"] = spin
    if inclination is not None:
        overrides["observer.inclination_deg"] = inclination
    if resolution is not None:
        overrides["raytrace.resolution"] = resolution
    if fov is not None:
        overrides["raytrace.fov"] = fov
    if backend is not None:
        overrides["raytrace.backend"] = backend
    block = f"experiment.parameters.{DISK_BLOCK}"
    for key, value in (("r_out", r_out), ("r_in", r_in), ("emissivity_index", emissivity_index), ("intensity_law", intensity_law)):
        if value is not None:
            overrides[f"{block}.{key}"] = value
    console = get_console()
    try:
        cfg = load_config(config if config is not None else _default_config_path(), overrides)
        st = Spacetime(mass=cfg.black_hole.mass, spin=cfg.black_hole.spin)
        disk = DiskModel.from_config(st, cfg)
        get_backend(cfg.raytrace.backend)
    except (ConfigError, ValueError, BackendUnavailable) as exc:
        get_console(stderr=True).print(f"error: {exc}")
        raise typer.Exit(code=1) from None
    render_section(console, "Spacetime", [
        ("Metric", "Schwarzschild" if st.is_schwarzschild else "Kerr"),
        ("Mass", st.mass), ("Spin", st.spin), ("Coordinates", "Boyer-Lindquist"),
    ])
    render_section(console, "Observer", [
        ("Radius", cfg.observer.radius), ("Inclination", cfg.observer.inclination_deg),
        ("Resolution", f"{cfg.raytrace.resolution} x {cfg.raytrace.resolution}"), ("FOV", cfg.raytrace.fov),
    ])
    render_section(console, "Disk", [
        ("r_in", disk.r_in), ("r_out", disk.r_out), ("Emissivity index", disk.emissivity_index),
        ("Intensity law", disk.intensity_law), ("Prograde", disk.prograde),
    ])
    if cfg.raytrace.fov < disk.r_out:
        logger.warning(
            "raytrace.fov = %g M is below disk r_out = %g M: the outer disk is cut off by the image edge "
            "(use --fov)", cfg.raytrace.fov, disk.r_out,
        )
    try:
        record = run(cfg)
    except (ConfigError, ValueError, RuntimeError) as exc:
        get_console(stderr=True).print(f"error: {exc}")
        raise typer.Exit(code=1) from None
    res = record.results
    rows: list[Row] = [(name, count) for name, count in res["counts"].items()]
    rows += [("Disk pixels", res["n_hits"])]
    if res["n_hits"]:
        rows += [("g min", res["g_min"]), ("g max", res["g_max"]), ("g mean", res["g_mean"]),
                 ("r_hit min", res["r_hit_min"]), ("r_hit max", res["r_hit_max"])]
    rows += [("Intensity max", res["intensity_max"]), ("Max null error", res["max_null_error"]),
             ("Null error (disk)", res["max_null_error_disk"]),
             ("Energy drift", res["max_energy_drift"]), ("Runtime", record.runtime_s),
             ("Run directory", str(record.run_dir)), ("Report", res["report"])]
    render_section(console, "Results", rows)


def register(app: typer.Typer) -> None:
    """Add the ``render`` command to ``app``."""
    app.command("render")(render)
