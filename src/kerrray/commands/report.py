"""``kerrray report --run runs/<run_id>``: print a stored run (PROJECT.md section 28).

Reads ``manifest.json`` from the run directory (and
``<report_dir>/<run_id>/summary.json`` when the run has one) and prints the
*Run*, *Spacetime*, *Solver* and *Results* sections with the recorded values.
A run that does not exist exits with code 1 and a one-line message.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

import typer

from kerrray.experiments.base import SUMMARY_FILENAME
from kerrray.reporting.console import Row, get_console, render_section
from kerrray.utils.config_blocks import DEFAULT_OUTPUT_DIR, DEFAULT_REPORT_DIR
from kerrray.utils.manifest import MANIFEST_FILENAME

__all__ = ["find_summary", "load_manifest", "register", "resolve_run_dir", "result_rows"]

UNKNOWN: Final[str] = "unknown"
MAX_INLINE_LIST: Final[int] = 6
"""Lists longer than this are summarised as ``list of N values``."""


def resolve_run_dir(run: Path) -> Path:
    """Return the run directory for ``run``: the path itself, or ``runs/<run>`` for a bare id.

    Raises:
        FileNotFoundError: If neither exists.
    """
    if run.is_dir():
        return run
    candidate = Path(DEFAULT_OUTPUT_DIR) / run
    if len(run.parts) == 1 and candidate.is_dir():
        return candidate
    raise FileNotFoundError(
        f"run directory {str(run)!r} does not exist (expected {DEFAULT_OUTPUT_DIR}/<run_id>)"
    )


def load_manifest(run_dir: Path) -> dict[str, Any]:
    """Read ``manifest.json`` from ``run_dir``.

    Raises:
        FileNotFoundError: If the manifest is missing.
        ValueError: If the manifest is not a JSON object.
    """
    path = run_dir / MANIFEST_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found: {run_dir} is not a KerrRay run directory")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return data


def find_summary(run_dir: Path, manifest: Mapping[str, Any]) -> Path | None:
    """Locate ``summary.json`` for the run, or ``None`` if the run has none.

    The report directory recorded in the manifest's configuration is tried
    first (relative to the current directory), then the same directory
    relative to the parent of the runs directory.
    """
    run_id = str(manifest.get("run_id", run_dir.name))
    config = manifest.get("config") if isinstance(manifest.get("config"), Mapping) else {}
    experiment = config.get("experiment") if isinstance(config.get("experiment"), Mapping) else {}
    report_dir = Path(str(experiment.get("report_dir", DEFAULT_REPORT_DIR)))
    candidates = [report_dir / run_id / SUMMARY_FILENAME]
    if not report_dir.is_absolute():
        candidates.append(run_dir.resolve().parent.parent / report_dir / run_id / SUMMARY_FILENAME)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _display_path(path: Path) -> str:
    """``path`` relative to the working directory when it lies inside it, else as given.

    Rich crops a single long "word" such as an absolute path with an ellipsis,
    so the short relative form (``runs/<run_id>``) is preferred when possible.
    """
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def _block(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    block = config.get(name)
    return block if isinstance(block, Mapping) else {}


def _flatten(values: Mapping[str, Any], prefix: str = "") -> list[Row]:
    rows: list[Row] = []
    for key, value in values.items():
        label = f"{prefix}{key}"
        if isinstance(value, Mapping):
            rows.extend(_flatten(value, f"{label}."))
        elif isinstance(value, list):
            if len(value) <= MAX_INLINE_LIST and not any(isinstance(v, (Mapping, list)) for v in value):
                rows.append((label, ", ".join(str(v) for v in value)))
            else:
                rows.append((label, f"list of {len(value)} values"))
        else:
            rows.append((label, value))
    return rows


def result_rows(results: Mapping[str, Any]) -> list[Row]:
    """Section rows for a results mapping: nested keys are dotted, long lists summarised."""
    rows = _flatten(results)
    return rows if rows else [("(none)", "no results recorded")]


def _run_rows(run_dir: Path, manifest: Mapping[str, Any], summary_path: Path | None) -> list[Row]:
    config = _block(manifest, "config")
    experiment = _block(config, "experiment")
    return [
        ("Run id", manifest.get("run_id", UNKNOWN)),
        ("Experiment", manifest.get("experiment", experiment.get("name", UNKNOWN))),
        ("Status", manifest.get("status", UNKNOWN)),
        ("Started", manifest.get("timestamp", UNKNOWN)),
        ("Runtime", manifest.get("runtime_s", UNKNOWN)),
        ("Seed", manifest.get("random_seed", UNKNOWN)),
        ("Git commit", manifest.get("git_commit", UNKNOWN)),
        ("Run directory", _display_path(run_dir)),
        ("Summary", _display_path(summary_path) if summary_path is not None else "not found"),
    ]


def _spacetime_rows(config: Mapping[str, Any]) -> list[Row]:
    hole, observer = _block(config, "black_hole"), _block(config, "observer")
    spin = hole.get("spin")
    metric = UNKNOWN if spin is None else ("Schwarzschild" if float(spin) == 0.0 else "Kerr")
    return [
        ("Metric", metric),
        ("Mass", hole.get("mass", UNKNOWN)),
        ("Spin", spin if spin is not None else UNKNOWN),
        ("Coordinates", "Boyer-Lindquist"),
        ("Observer radius", observer.get("radius", UNKNOWN)),
        ("Inclination", observer.get("inclination_deg", UNKNOWN)),
    ]


def _solver_rows(config: Mapping[str, Any]) -> list[Row]:
    integration, raytrace = _block(config, "integration"), _block(config, "raytrace")
    termination = _block(config, "termination")
    return [
        ("Integrator", str(integration.get("method", UNKNOWN)).upper()),
        ("rtol", integration.get("rtol", UNKNOWN)),
        ("atol", integration.get("atol", UNKNOWN)),
        ("Max steps", integration.get("max_steps", UNKNOWN)),
        ("Step size", integration.get("step_size", UNKNOWN)),
        ("Lambda max", integration.get("lambda_max", UNKNOWN)),
        ("Backend", raytrace.get("backend", UNKNOWN)),
        ("dtype", raytrace.get("dtype", UNKNOWN)),
        ("Resolution", raytrace.get("resolution", UNKNOWN)),
        ("Escape radius", termination.get("escape_radius", UNKNOWN)),
    ]


def report(
    run: Path = typer.Option(
        ...,
        "--run",
        help="Run directory runs/<run_id>, or a bare run id looked up under runs/.",
    ),
) -> None:
    """Print the manifest and summary of a stored run."""
    try:
        run_dir = resolve_run_dir(run)
        manifest = load_manifest(run_dir)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        get_console(stderr=True).print(f"error: {exc}")
        raise typer.Exit(code=1) from None
    summary_path = find_summary(run_dir, manifest)
    results: Mapping[str, Any] = _block(manifest, "results")
    if summary_path is not None:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if isinstance(summary, Mapping) and isinstance(summary.get("results"), Mapping):
            results = summary["results"]
    config = _block(manifest, "config")
    console = get_console()
    render_section(console, "Run", _run_rows(run_dir, manifest, summary_path))
    render_section(console, "Spacetime", _spacetime_rows(config))
    render_section(console, "Solver", _solver_rows(config))
    render_section(console, "Results", result_rows(results))


def register(app: typer.Typer) -> None:
    """Add the ``report`` command to ``app`` (docs/architecture.md section 6)."""
    app.command("report")(report)
