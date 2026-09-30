"""``kerrray trace``: trace an image and print the section 29 panels.

The command loads ``configs/shadow.yaml`` (or ``--config``), applies the
command-line overrides as dotted configuration keys, prints the *Spacetime*,
*Observer* and *Ray Trace* sections, traces every pixel with a Rich progress
bar (*Status*) and prints the *Results* section with the computed counts,
conservation diagnostics and runtime. The run goes through
:func:`kerrray.experiments.base.run_experiment` under the name ``"trace"``, so
``runs/<run_id>/`` receives the manifest, the configuration copy and the
state map (``shadow.npz``) and ``reports/<run_id>/`` the figure
(``shadow.png``) and ``summary.json``. ``kerrray shadow``
(:mod:`kerrray.commands.shadow`) reuses :func:`execute` and adds the boundary
comparison.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Final

import numpy as np
import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn

from kerrray.experiments.base import ExperimentContext, RunRecord, run_experiment
from kerrray.geodesics import IntegratorOptions, TerminationOptions
from kerrray.geometry import Spacetime, outer_horizon
from kerrray.photons import TerminationState
from kerrray.raytracing.backends import BackendUnavailable, get_backend
from kerrray.raytracing.camera import Camera
from kerrray.raytracing.rays import DEFAULT_CHUNK_SIZE
from kerrray.raytracing.shadow import ShadowImage, compute_shadow
from kerrray.reporting.console import Row, get_console, render_section
from kerrray.reporting.plots import plot_shadow
from kerrray.utils.config import ConfigError, KerrRayConfig, load_config
from kerrray.utils.logging import get_logger

__all__ = [
    "DEFAULT_CONFIG",
    "STATE_MAP_FILENAME",
    "FIGURE_FILENAME",
    "build_overrides",
    "execute",
    "load_run_config",
    "register",
    "resolve_config_path",
    "trace",
]

DEFAULT_CONFIG: Final[Path] = Path("configs") / "shadow.yaml"
STATE_MAP_FILENAME: Final[str] = "shadow.npz"
FIGURE_FILENAME: Final[str] = "shadow.png"
PROGRESS_CHUNKS: Final[int] = 16
"""The image is traced in about this many chunks so that the progress bar moves."""

logger = get_logger(__name__)

ExtraFunction = Callable[[ExperimentContext, ShadowImage, Console], Mapping[str, Any]]


def build_overrides(
    *,
    spin: float | None = None,
    inclination: float | None = None,
    resolution: int | None = None,
    fov: float | None = None,
    backend: str | None = None,
    method: str | None = None,
    rtol: float | None = None,
    atol: float | None = None,
    sets: list[str] | None = None,
) -> dict[str, Any]:
    """Map the command-line flags (``None`` = not given) to dotted configuration keys.

    ``sets`` holds free-form ``key.path=value`` overrides applied after the
    named flags; a malformed entry raises ``ConfigError``.
    """
    named = {
        "black_hole.spin": spin,
        "observer.inclination_deg": inclination,
        "raytrace.resolution": resolution,
        "raytrace.fov": fov,
        "raytrace.backend": backend,
        "integration.method": method,
        "integration.rtol": rtol,
        "integration.atol": atol,
    }
    overrides: dict[str, Any] = {key: value for key, value in named.items() if value is not None}
    for item in sets or []:
        key, sep, value = item.partition("=")
        if not sep or not key.strip():
            raise ConfigError(f"--set expects key.path=value, got {item!r}")
        overrides[key.strip()] = value.strip()
    return overrides


def resolve_config_path(config: Path) -> Path:
    """``config`` if it exists; else, for a relative path, the same path under the repository root.

    Lets ``kerrray trace`` / ``kerrray shadow`` find the default
    ``configs/shadow.yaml`` from any working directory of a source checkout.
    """
    if config.is_file() or config.is_absolute():
        return config
    candidate = Path(__file__).resolve().parents[3] / config
    return candidate if candidate.is_file() else config


def load_run_config(config: Path, overrides: Mapping[str, Any]) -> KerrRayConfig:
    """Load ``config`` with ``overrides``; a bad file or override exits with code 1."""
    try:
        return load_config(resolve_config_path(Path(config)), overrides)
    except (ConfigError, FileNotFoundError, OSError) as exc:
        get_console(stderr=True).print(f"error: {exc}")
        raise typer.Exit(code=1) from None


def spacetime_rows(st: Spacetime) -> list[Row]:
    """*Spacetime* section rows (PROJECT.md section 29)."""
    return [
        ("Metric", "Schwarzschild" if st.is_schwarzschild else "Kerr"),
        ("Mass", f"{st.mass:g} M"),
        ("Spin", f"{st.spin:.3f}"),
        ("Horizon r+", f"{outer_horizon(st):.6g} M"),
        ("Coordinates", "Boyer-Lindquist"),
    ]


def observer_rows(cam: Camera) -> list[Row]:
    """*Observer* section rows."""
    return [
        ("Radius", f"{cam.radius:g} M"),
        ("Inclination", f"{cam.inclination_deg:g} deg"),
        ("Azimuth", f"{cam.phi_deg:g} deg"),
    ]


def raytrace_rows(cfg: KerrRayConfig, cam: Camera) -> list[Row]:
    """*Ray Trace* section rows."""
    integ = cfg.integration
    return [
        ("Resolution", f"{cam.resolution} x {cam.resolution}"),
        ("Rays", cam.n_rays),
        ("Field of view", f"+-{cam.fov:g} M"),
        ("Pixel size", f"{cam.pixel_size:.4g} M"),
        ("Integrator", integ.method.upper()),
        ("rtol / atol", f"{integ.rtol:g} / {integ.atol:g}"),
        ("Backend", cfg.raytrace.backend),
        ("dtype", cfg.raytrace.dtype),
    ]


def display_path(path: Path | str) -> str:
    """``path`` relative to the working directory when it lies inside it, else unchanged.

    Rich crops a long space-free word such as an absolute path with an
    ellipsis, so the short form is preferred for the console; the results
    mapping keeps the absolute path.
    """
    try:
        return str(Path(path).resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def results_rows(img: ShadowImage, results: Mapping[str, Any]) -> list[Row]:
    """*Results* section rows from the traced image and the results mapping."""
    return [
        ("Captured", img.count(TerminationState.CAPTURED)),
        ("Escaped", img.count(TerminationState.ESCAPED)),
        ("Numerical fail", img.n_failed),
        ("Other", img.n_other),
        ("Max null error", results["max_null_error"]),
        ("  escaped rays", results["max_null_error_escaped"]),
        ("Energy drift", results["max_energy_drift"]),
        ("Lz drift", results["max_lz_drift"]),
        ("Carter drift", results["max_carter_drift"]),
        ("Runtime", f"{results['trace_runtime_s']:.3f} s"),
        ("Rays / s", f"{results['rays_per_second']:.1f}"),
        ("State map", display_path(results["state_map"])),
        ("Figure", display_path(results["figure"])),
    ]


def _options_from_config(cfg: KerrRayConfig) -> tuple[IntegratorOptions, TerminationOptions]:
    integ = cfg.integration
    return (
        IntegratorOptions(
            method=integ.method,
            rtol=integ.rtol,
            atol=integ.atol,
            step_size=integ.step_size,
            max_steps=integ.max_steps,
            lambda_max=integ.lambda_max,
            dtype=cfg.raytrace.dtype,
        ),
        TerminationOptions(
            horizon_epsilon=cfg.termination.horizon_epsilon,
            escape_radius=cfg.termination.escape_radius,
        ),
    )


def _diagnostics(img: ShadowImage) -> dict[str, float]:
    res = img.result
    escaped = res.state == int(TerminationState.ESCAPED)
    null_escaped = float(res.max_null_error[escaped].max()) if np.any(escaped) else math.nan
    return {
        "max_null_error": float(res.max_null_error.max()),
        "max_null_error_escaped": null_escaped,
        "max_energy_drift": float(res.max_energy_drift.max()),
        "max_lz_drift": float(res.max_lz_drift.max()),
        "max_carter_drift": float(res.max_carter_drift.max()),
    }


def trace_image(ctx: ExperimentContext, console: Console) -> tuple[ShadowImage, dict[str, Any]]:
    """Trace the image of ``ctx.cfg`` with a progress bar; save the state map and figure.

    Returns the image and the JSON-serialisable results (counts, diagnostics,
    runtime, rays per second and the file paths).
    """
    cfg = ctx.cfg
    st = Spacetime(mass=cfg.black_hole.mass, spin=cfg.black_hole.spin)
    cam = Camera.from_config(cfg.observer, cfg.raytrace)
    integ, term = _options_from_config(cfg)
    backend = get_backend(cfg.raytrace.backend)
    if integ.method == "rk4" and integ.max_steps * integ.step_size < 2.0 * cam.radius:
        logger.warning(
            "rk4 with max_steps * step_size = %g M is below twice the observer radius (%g M); "
            "rays may end as MAX_AFFINE_PARAMETER before reaching the hole or escaping",
            integ.max_steps * integ.step_size,
            cam.radius,
        )
    chunk = min(DEFAULT_CHUNK_SIZE, max(256, -(-cam.n_rays // PROGRESS_CHUNKS)))
    render_section(console, "Status", [])
    with Progress(
        TextColumn("tracing"),
        BarColumn(bar_width=30),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as bar:
        task = bar.add_task("trace", total=cam.n_rays)
        img = compute_shadow(
            st,
            cam,
            integ,
            term,
            backend=backend,
            progress=lambda fraction: bar.update(task, completed=fraction * cam.n_rays),
            chunk_size=chunk,
        )
    console.print()
    state_map = img.save(ctx.run_dir / STATE_MAP_FILENAME)
    figure = plot_shadow(
        ctx.report_dir / FIGURE_FILENAME,
        img.alpha,
        img.beta,
        img.captured,
        title=f"a = {st.spin:g}, i = {cam.inclination_deg:g} deg, {cam.resolution}^2",
    )
    runtime = img.result.runtime_s
    results: dict[str, Any] = {
        "spin": st.spin,
        "mass": st.mass,
        "inclination_deg": cam.inclination_deg,
        "observer_radius": cam.radius,
        "resolution": cam.resolution,
        "fov": cam.fov,
        "pixel_size": cam.pixel_size,
        "n_rays": cam.n_rays,
        "backend": backend.name,
        "method": integ.method,
        "rtol": integ.rtol,
        "atol": integ.atol,
        "counts": dict(img.counts),
        "n_failed": img.n_failed,
        "n_other": img.n_other,
        **_diagnostics(img),
        "trace_runtime_s": runtime,
        "rays_per_second": cam.n_rays / runtime if runtime > 0.0 else math.nan,
        "state_map": str(state_map),
        "figure": str(figure),
    }
    return img, results


def execute(cfg: KerrRayConfig, name: str, extra: ExtraFunction | None = None) -> RunRecord:
    """Print the sections, run the trace as experiment ``name`` and print the results.

    ``extra`` (used by ``kerrray shadow``) receives the context, the traced
    image and the console after the trace; the mapping it returns is merged
    into the results and it may print further sections. User errors (bad
    camera, unavailable backend) exit with code 1 and a one-line message.
    """
    console = get_console()
    st = Spacetime(mass=cfg.black_hole.mass, spin=cfg.black_hole.spin)
    try:
        cam = Camera.from_config(cfg.observer, cfg.raytrace)
        get_backend(cfg.raytrace.backend)
    except (ValueError, BackendUnavailable) as exc:
        get_console(stderr=True).print(f"error: {exc}")
        raise typer.Exit(code=1) from None
    render_section(console, "Spacetime", spacetime_rows(st))
    render_section(console, "Observer", observer_rows(cam))
    render_section(console, "Ray Trace", raytrace_rows(cfg, cam))

    def body(ctx: ExperimentContext) -> dict[str, Any]:
        img, results = trace_image(ctx, console)
        render_section(console, "Results", results_rows(img, results))
        if extra is not None:
            results.update(extra(ctx, img, console))
        return results

    try:
        record = run_experiment(name, cfg, body)
    except (ValueError, BackendUnavailable) as exc:
        get_console(stderr=True).print(f"error: {exc}")
        raise typer.Exit(code=1) from None
    render_section(
        console,
        "Run",
        [("Run id", record.run_id), ("Run directory", display_path(record.run_dir)),
         ("Report directory", display_path(record.report_dir)),
         ("Total runtime", f"{record.runtime_s:.3f} s")],
    )
    return record


def trace(
    config: Path = typer.Option(DEFAULT_CONFIG, "--config", help="Configuration file."),
    spin: float | None = typer.Option(None, "--spin", help="Dimensionless spin a/M (|a/M| < 1)."),
    inclination: float | None = typer.Option(None, "--inclination", help="Observer inclination in degrees."),
    resolution: int | None = typer.Option(None, "--resolution", help="Pixels per side (even)."),
    fov: float | None = typer.Option(None, "--fov", help="Image half-width in units of M."),
    backend: str | None = typer.Option(None, "--backend", help="numpy | numba | cuda (unavailable)."),
    method: str | None = typer.Option(None, "--method", help="rk4 | rk45."),
    rtol: float | None = typer.Option(None, "--rtol", help="Relative tolerance (rk45)."),
    atol: float | None = typer.Option(None, "--atol", help="Absolute tolerance (rk45)."),
    sets: list[str] = typer.Option([], "--set", help="Dotted override key.path=value (repeatable)."),
) -> None:
    """Trace every pixel of the observer's image and report the termination map."""
    try:
        overrides = build_overrides(
            spin=spin, inclination=inclination, resolution=resolution, fov=fov, backend=backend,
            method=method, rtol=rtol, atol=atol, sets=sets,
        )
    except ConfigError as exc:
        get_console(stderr=True).print(f"error: {exc}")
        raise typer.Exit(code=1) from None
    cfg = load_run_config(config, overrides)
    execute(cfg, "trace")


def register(app: typer.Typer) -> None:
    """Add the ``trace`` command to ``app`` (docs/architecture.md section 6)."""
    app.command("trace")(trace)
