"""``kerrray benchmark`` command group: ``solver``, ``cpu``, ``precision``, ``gpu``.

``solver`` runs :func:`kerrray.benchmarks.solver.run_solver_benchmark`
(EXP-006, PROJECT.md section 19) on ``--config`` (default
``configs/convergence.yaml``) with the optional ``--repeats`` and
``--n-rays`` overrides and prints the section 19 table with the measured
values. ``cpu``, ``precision`` and ``gpu`` import
``kerrray.benchmarks.cpu.run_cpu_benchmark``,
``kerrray.benchmarks.precision.run_precision_study`` and
``kerrray.benchmarks.gpu.run_gpu_benchmark`` lazily (default config
``configs/benchmark.yaml``); when a module cannot be imported the command
says so and exits with code 2, as it does on configuration errors. Skipped
runs are printed with their logged reason. Nothing printed here is invented:
every number comes from a run record.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

import typer

from kerrray.commands.experiment import parse_overrides
from kerrray.commands.report import result_rows
from kerrray.reporting.console import get_console, render_section
from kerrray.reporting.tables import rich_table
from kerrray.utils.config import ConfigError, load_config

__all__ = ["benchmark_app", "register", "run_lazy_benchmark"]

DEFAULT_SOLVER_CONFIG: Final[Path] = Path("configs") / "convergence.yaml"
DEFAULT_BENCHMARK_CONFIG: Final[Path] = Path("configs") / "benchmark.yaml"
EXIT_USAGE: Final[int] = 2
LAZY_BENCHMARKS: Final[dict[str, tuple[str, str]]] = {
    "cpu": ("kerrray.benchmarks.cpu", "run_cpu_benchmark"),
    "precision": ("kerrray.benchmarks.precision", "run_precision_study"),
    "gpu": ("kerrray.benchmarks.gpu", "run_gpu_benchmark"),
}

benchmark_app = typer.Typer(help="Solver, CPU, precision and GPU benchmarks (PROJECT.md sections 19, 26, 27).", no_args_is_help=True)


def _load(config: Path, overrides: list[str] | None) -> Any:
    try:
        return load_config(config, parse_overrides(overrides))
    except ConfigError as exc:
        get_console(stderr=True).print(f"error: {exc}")
        raise typer.Exit(code=EXIT_USAGE) from None


TABLE_SPECS: Final[dict[str, tuple[tuple[str, str | None], ...]]] = {
    "solver": (("table", None),),
    "cpu": (("runs", "TABLE_COLUMNS"),),
    "precision": (("runs", "RUN_COLUMNS"), ("comparisons", "CMP_COLUMNS")),
    "gpu": (),
}
"""Per benchmark: ``(results key, column-tuple attribute of the benchmark module)`` printed as tables."""
SUMMARY_EXCLUDE: Final[frozenset[str]] = frozenset(
    {"table", "runs", "comparisons", "configurations", "rays", "skipped", "figures", "parameters"}
)
"""Result keys shown as tables or kept only in summary.json (too long for the Results section)."""


def _is_rows(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(isinstance(row, Mapping) for row in value)


def _print_record(record: Any, kind: str = "solver", module: Any = None) -> None:
    console = get_console()
    results: Mapping[str, Any] = record.results if isinstance(getattr(record, "results", None), Mapping) else {}
    for key, columns_attr in TABLE_SPECS.get(kind, ()):
        rows = results.get(key)
        if not _is_rows(rows):
            continue
        columns = list(getattr(module, columns_attr, ())) if columns_attr else None
        console.print(rich_table(rows, columns=columns or None, title=key, precision=4))
        console.print()
    skipped = results.get("skipped")
    if _is_rows(skipped):
        render_section(console, "Skipped (logged)", [
            (f"{item.get('backend', item.get('solver', ''))} {item.get('rays', item.get('setting', ''))}",
             item.get("reason", "")) for item in skipped
        ])
    render_section(console, "Results", result_rows({k: v for k, v in results.items() if k not in SUMMARY_EXCLUDE}))
    render_section(console, "Run", [("Run id", record.run_id), ("Runtime [s]", record.runtime_s),
                                    ("Run directory", str(record.run_dir)), ("Report directory", str(record.report_dir))])


@benchmark_app.command("solver")
def solver(
    config: Path = typer.Option(DEFAULT_SOLVER_CONFIG, "--config", help="Configuration file (convergence sub-block)."),
    repeats: int | None = typer.Option(None, "--repeats", min=1, help="Timing repeats per configuration (best is reported)."),
    n_rays: int | None = typer.Option(None, "--n-rays", min=1, help="Size of the fixed ray set."),
    set_: list[str] | None = typer.Option(None, "--set", metavar="KEY=VALUE", help="Dotted-key override, repeatable."),
) -> None:
    """Compare fixed-step RK4, adaptive RK45 and DOP853 on a fixed ray set (EXP-006)."""
    from kerrray.benchmarks.solver import run_solver_benchmark

    cfg = _load(config, set_)
    try:
        record = run_solver_benchmark(cfg, n_rays=n_rays, repeats=repeats)
    except ConfigError as exc:
        get_console(stderr=True).print(f"error: {exc}")
        raise typer.Exit(code=EXIT_USAGE) from None
    _print_record(record, "solver")


def run_lazy_benchmark(kind: str, config: Path, overrides: list[str] | None) -> None:
    """Import and run one of the performance role's benchmarks, or exit 2 if it is not available."""
    module_name, function_name = LAZY_BENCHMARKS[kind]
    try:
        module = importlib.import_module(module_name)
        fn = getattr(module, function_name)
    except (ImportError, AttributeError):
        get_console(stderr=True).print(
            f"error: {module_name}.{function_name} is not available in this build; "
            "the CPU/precision/GPU benchmarks belong to Phase 8 (docs/decisions.md, D-007)."
        )
        raise typer.Exit(code=EXIT_USAGE) from None
    cfg = _load(config, overrides)
    try:
        record = fn(cfg)
    except ConfigError as exc:
        get_console(stderr=True).print(f"error: {exc}")
        raise typer.Exit(code=EXIT_USAGE) from None
    if hasattr(record, "results"):
        _print_record(record, kind, module)
    else:
        get_console().print(str(record))


@benchmark_app.command("cpu")
def cpu(
    config: Path = typer.Option(DEFAULT_BENCHMARK_CONFIG, "--config", help="Configuration file (benchmark sub-block)."),
    set_: list[str] | None = typer.Option(None, "--set", metavar="KEY=VALUE", help="Dotted-key override, repeatable."),
) -> None:
    """CPU performance benchmark: runtime, memory and accuracy versus ray count (EXP-009)."""
    run_lazy_benchmark("cpu", config, set_)


@benchmark_app.command("precision")
def precision(
    config: Path = typer.Option(DEFAULT_BENCHMARK_CONFIG, "--config", help="Configuration file (benchmark sub-block)."),
    set_: list[str] | None = typer.Option(None, "--set", metavar="KEY=VALUE", help="Dotted-key override, repeatable."),
) -> None:
    """float32 versus float64 precision study on the configured camera image (EXP-011)."""
    run_lazy_benchmark("precision", config, set_)


@benchmark_app.command("gpu")
def gpu(
    config: Path = typer.Option(DEFAULT_BENCHMARK_CONFIG, "--config", help="Configuration file (experiment block only)."),
    set_: list[str] | None = typer.Option(None, "--set", metavar="KEY=VALUE", help="Dotted-key override, repeatable."),
) -> None:
    """GPU benchmark stub: reports that GPU acceleration is optional future work (D-007)."""
    run_lazy_benchmark("gpu", config, set_)


def register(app: typer.Typer) -> None:
    """Add the ``benchmark`` sub-app to ``app`` (docs/architecture.md section 6)."""
    app.add_typer(benchmark_app, name="benchmark")
