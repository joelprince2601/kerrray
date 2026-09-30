"""``kerrray lens --spin 0.9 --impact-range 3,20``: strong-field lensing scan
(PROJECT.md sections 22 and 28; docs/lensing.md section 5).

Loads ``configs/lensing.yaml`` (or ``--config``), applies the command-line
overrides to ``black_hole.spin`` and to the ``lensing`` sub-block of
``experiment.parameters``, runs :func:`kerrray.experiments.lensing.run` and
prints the computed critical impact parameters, ray counts, deviations from
the exact integral (spin 0) and the strong-field fit, plus the run and report
paths. Configuration errors exit with code 1 and a one-line message.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

import typer

from kerrray.experiments.lensing import LENSING_BLOCK, run
from kerrray.reporting.console import Row, get_console, render_section
from kerrray.utils.config import ConfigError, KerrRayConfig, load_config

__all__ = ["DEFAULT_CONFIG", "default_config_path", "lens", "parse_impact_range", "register"]

DEFAULT_CONFIG: Final[str] = "configs/lensing.yaml"


def default_config_path(name: str = DEFAULT_CONFIG) -> Path:
    """``name`` relative to the working directory if it exists there, else in the repository root."""
    local = Path(name)
    if local.is_file():
        return local
    return Path(__file__).resolve().parents[3] / name


def parse_impact_range(text: str) -> tuple[float, float]:
    """Parse ``"MIN,MAX"`` into two positive floats with ``MIN < MAX``."""
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 2:
        raise ConfigError(f"--impact-range must be MIN,MAX, got {text!r}")
    try:
        lo, hi = float(parts[0]), float(parts[1])
    except ValueError:
        raise ConfigError(f"--impact-range must be two numbers, got {text!r}") from None
    if not (0.0 < lo < hi):
        raise ConfigError(f"--impact-range needs 0 < MIN < MAX, got {text!r}")
    return lo, hi


def _load(config: Path, spin: float | None, impact_range: str | None, n: int | None, launch_radius: float | None) -> KerrRayConfig:
    overrides: dict[str, Any] = {}
    if spin is not None:
        overrides["black_hole.spin"] = spin
    if impact_range is not None:
        lo, hi = parse_impact_range(impact_range)
        overrides[f"experiment.parameters.{LENSING_BLOCK}.impact_min"] = lo
        overrides[f"experiment.parameters.{LENSING_BLOCK}.impact_max"] = hi
    if n is not None:
        overrides[f"experiment.parameters.{LENSING_BLOCK}.n_rays"] = n
    if launch_radius is not None:
        overrides[f"experiment.parameters.{LENSING_BLOCK}.launch_radius"] = launch_radius
    return load_config(config, overrides)


def _scan_rows(label: str, res: dict[str, Any]) -> list[Row]:
    rows: list[Row] = [(f"{label} b_c", res["b_critical"])]
    for state, count in res["counts"].items():
        rows.append((f"{label} {state.lower()}", count))
    if res.get("max_abs_dev_exact") is not None:
        rows.append((f"{label} max |num-exact|", res["max_abs_dev_exact"]))
    if res.get("max_abs_dev_weak_field") is not None:
        rows.append((f"{label} max |num-4M/b|", res["max_abs_dev_weak_field"]))
    fit = res.get("strong_field_fit")
    if fit is not None:
        rows.append((f"{label} fit a_bar", fit["a_bar"]))
        rows.append((f"{label} fit b_bar", fit["b_bar"]))
    rows.append((f"{label} max null err", res["max_null_error"]))
    return rows


def lens(
    spin: float | None = typer.Option(None, "--spin", help="Dimensionless spin a/M (|spin| < 1)."),
    impact_range: str | None = typer.Option(None, "--impact-range", metavar="MIN,MAX", help="Impact parameters in units of M."),
    n: int | None = typer.Option(None, "--n", min=2, help="Rays per sense of rotation."),
    launch_radius: float | None = typer.Option(None, "--launch-radius", help="Launch (and escape) radius in units of M."),
    config: Path = typer.Option(None, "--config", help=f"Configuration file (default {DEFAULT_CONFIG})."),
) -> None:
    """Deflection angle, closest approach and turns versus impact parameter."""
    console = get_console()
    try:
        cfg = _load(config if config is not None else default_config_path(), spin, impact_range, n, launch_radius)
    except ConfigError as exc:
        get_console(stderr=True).print(f"error: {exc}")
        raise typer.Exit(code=1) from None
    params = cfg.experiment.parameters.get(LENSING_BLOCK, {})
    render_section(console, "Spacetime", [
        ("Metric", "Schwarzschild" if cfg.black_hole.spin == 0.0 else "Kerr"),
        ("Mass", cfg.black_hole.mass),
        ("Spin", cfg.black_hole.spin),
        ("Coordinates", "Boyer-Lindquist"),
    ])
    render_section(console, "Lensing", [
        ("Impact min", params.get("impact_min", 3.0)),
        ("Impact max", params.get("impact_max", 20.0)),
        ("Rays", params.get("n_rays", 40)),
        ("Launch radius", params.get("launch_radius", 1000.0)),
        ("Integrator", cfg.integration.method.upper()),
        ("rtol", cfg.integration.rtol),
    ])
    try:
        record = run(cfg)
    except (ConfigError, ValueError) as exc:
        get_console(stderr=True).print(f"error: {exc}")
        raise typer.Exit(code=1) from None
    rows: list[Row] = []
    for label, res in record.results["scans"].items():
        rows.extend(_scan_rows(label, res))
    rows.append(("Runtime", record.runtime_s))
    rows.append(("Run directory", str(record.run_dir)))
    rows.append(("Report", record.results["report"]))
    render_section(console, "Results", rows)


def register(app: typer.Typer) -> None:
    """Add the ``lens`` command to ``app`` (docs/architecture.md section 6)."""
    app.command("lens")(lens)
