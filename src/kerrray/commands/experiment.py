"""``kerrray experiment --config configs/<name>.yaml [--name NAME] [--set key=value ...]``.

Loads the configuration, applies dotted-key overrides (``--set
black_hole.spin=0.5``; values are parsed as YAML scalars or flow sequences,
so ``--set experiment.parameters.convergence.step_sizes=[0.5,0.25]`` works),
resolves the experiment module ``kerrray.experiments.<experiment.name>`` (or
``--name``) with :mod:`importlib`, calls its ``run(cfg)`` and prints the
run identity, the result keys and values and the run and report paths. An
unknown experiment name lists the available experiment modules and exits
with code 2, as do configuration errors.
"""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import typer
import yaml

import kerrray.experiments
from kerrray.commands.report import result_rows
from kerrray.reporting.console import Row, get_console, render_section
from kerrray.utils.config import ConfigError, KerrRayConfig, load_config

__all__ = ["available_experiments", "load_experiment_module", "parse_overrides", "register", "run_named_experiment"]

EXPERIMENTS_PACKAGE: Final[str] = "kerrray.experiments"
CONFIG_DIR: Final[Path] = Path("configs")
EXIT_USAGE: Final[int] = 2


def available_experiments() -> list[str]:
    """Names of the experiment modules in :mod:`kerrray.experiments` (``base`` excluded)."""
    return sorted(m.name for m in pkgutil.iter_modules(kerrray.experiments.__path__) if m.name != "base")


def _numbers(value: Any) -> Any:
    """Convert numeric-looking strings (YAML 1.1 reads ``1e-6`` as text) to ``float``, recursively in lists."""
    if isinstance(value, list):
        return [_numbers(v) for v in value]
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    return value


def parse_overrides(assignments: Sequence[str] | None) -> dict[str, Any]:
    """Turn ``["a.b=1", "c=[1,2]", "d=1e-6"]`` into ``{"a.b": 1, "c": [1, 2], "d": 1e-06}``.

    Values are parsed as YAML; strings that parse as a number (``1e-6``,
    which YAML 1.1 leaves as text) become ``float``.
    """
    overrides: dict[str, Any] = {}
    for item in assignments or ():
        key, sep, raw = item.partition("=")
        if not sep or not key.strip():
            raise ConfigError(f"override {item!r} must have the form key=value")
        try:
            value = yaml.safe_load(raw)
        except yaml.YAMLError:
            value = raw
        overrides[key.strip()] = raw if value is None and raw.strip() else _numbers(value)
    return overrides


def load_experiment_module(name: str) -> Any:
    """Import ``kerrray.experiments.<name>`` and check that it exposes a callable ``run``.

    Raises:
        LookupError: If no such module exists or it has no ``run``; the message
            lists the available experiments.
    """
    qualified = f"{EXPERIMENTS_PACKAGE}.{name}"
    try:
        module = importlib.import_module(qualified)
    except ModuleNotFoundError as exc:
        if exc.name != qualified:
            raise
        raise LookupError(f"unknown experiment {name!r}; available: {', '.join(available_experiments())}") from None
    run = getattr(module, "run", None)
    if not callable(run):
        raise LookupError(f"{qualified} has no run(cfg) function; available: {', '.join(available_experiments())}")
    return module


def run_named_experiment(name: str, cfg: KerrRayConfig) -> Any:
    """Run experiment ``name`` on ``cfg`` and return its :class:`~kerrray.experiments.base.RunRecord`."""
    return load_experiment_module(name).run(cfg)


def _run_rows(name: str, config_path: Path, record: Any) -> list[Row]:
    return [
        ("Experiment", name),
        ("Config", str(config_path)),
        ("Run id", record.run_id),
        ("Runtime [s]", record.runtime_s),
        ("Run directory", str(record.run_dir)),
        ("Report directory", str(record.report_dir)),
        ("Summary", str(record.summary_path)),
    ]


def experiment(
    config: Path | None = typer.Option(None, "--config", help="Configuration file; defaults to configs/<NAME>.yaml when --name is given."),
    name: str | None = typer.Option(None, "--name", help="Experiment module name (overrides experiment.name of the file)."),
    set_: list[str] | None = typer.Option(None, "--set", metavar="KEY=VALUE", help="Dotted-key override, repeatable."),
) -> None:
    """Run the experiment named by the configuration (or --name) and print its summary."""
    console = get_console()
    errors = get_console(stderr=True)
    if config is None:
        if name is None:
            errors.print("error: give --config or --name (available experiments: " + ", ".join(available_experiments()) + ")")
            raise typer.Exit(code=EXIT_USAGE)
        config = CONFIG_DIR / f"{name}.yaml"
    try:
        cfg = load_config(config, parse_overrides(set_))
    except ConfigError as exc:
        errors.print(f"error: {exc}")
        raise typer.Exit(code=EXIT_USAGE) from None
    experiment_name = name if name is not None else cfg.experiment.name
    try:
        module = load_experiment_module(experiment_name)
    except LookupError as exc:
        errors.print(f"error: {exc}")
        raise typer.Exit(code=EXIT_USAGE) from None
    render_section(console, "Experiment", [("Name", experiment_name), ("Config", str(config)),
                                           ("Seed", cfg.experiment.seed), ("Spin", cfg.black_hole.spin),
                                           ("Integrator", cfg.integration.method.upper())])
    try:
        record = module.run(cfg)
    except ConfigError as exc:
        errors.print(f"error: {exc}")
        raise typer.Exit(code=EXIT_USAGE) from None
    results: Mapping[str, Any] = record.results if isinstance(record.results, Mapping) else {}
    render_section(console, "Results", result_rows(results))
    render_section(console, "Run", _run_rows(experiment_name, config, record))


def register(app: typer.Typer) -> None:
    """Add the ``experiment`` command to ``app`` (docs/architecture.md section 6)."""
    app.command("experiment")(experiment)
