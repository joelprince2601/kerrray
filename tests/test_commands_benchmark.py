"""Tests for ``kerrray benchmark`` (kerrray.commands.benchmark) on a fresh Typer app."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import typer
from typer.testing import CliRunner

from kerrray.commands import benchmark as benchmark_cmd

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
runner = CliRunner()


def _app() -> typer.Typer:
    app = typer.Typer()

    @app.callback()
    def main() -> None:
        """Test application (forces sub-command mode)."""

    benchmark_cmd.register(app)
    return app


def _text(result: Any) -> str:
    text = result.output
    try:
        text += result.stderr
    except (ValueError, AttributeError):
        pass
    return text


def _dirs(tmp_path: Path) -> list[str]:
    return ["--set", f"experiment.output_dir={tmp_path / 'runs'}", "--set", f"experiment.report_dir={tmp_path / 'reports'}"]


def _summary(tmp_path: Path) -> dict[str, Any]:
    paths = list((tmp_path / "reports").glob("*/summary.json"))
    assert len(paths) == 1
    return json.loads(paths[0].read_text(encoding="utf-8"))


def test_benchmark_group_lists_its_commands() -> None:
    result = runner.invoke(_app(), ["benchmark", "--help"])
    assert result.exit_code == 0
    for name in ("solver", "cpu", "precision", "gpu"):
        assert name in result.output


def test_benchmark_gpu_reports_future_work(tmp_path: Path) -> None:
    result = runner.invoke(_app(), ["benchmark", "gpu", "--config", str(CONFIGS / "gpu.yaml"), *_dirs(tmp_path)])
    assert result.exit_code == 0, _text(result)
    assert "optional future work" in result.output
    results = _summary(tmp_path)["results"]
    assert results["gpu_numbers_reported"] is False and "gpu_available" in results


def test_benchmark_solver_prints_the_measured_table(tmp_path: Path) -> None:
    args = ["benchmark", "solver", "--config", str(CONFIGS / "convergence.yaml"), "--n-rays", "2", "--repeats", "1",
            *_dirs(tmp_path),
            "--set", "experiment.parameters.convergence.step_sizes=[0.2]",
            "--set", "experiment.parameters.convergence.tolerances=[1e-5]",
            "--set", "experiment.parameters.solver.launch_radius=15",
            "--set", "experiment.parameters.solver.reference_rtol=1e-8",
            "--set", "experiment.parameters.solver.reference_atol=1e-10"]
    result = runner.invoke(_app(), args, env={"COLUMNS": "250"})
    assert result.exit_code == 0, _text(result)
    for label in ("RK4", "RK45", "DOP853", "Runtime"):
        assert label in result.output
    results = _summary(tmp_path)["results"]
    assert results["n_rays"] == 2 and results["repeats"] == 1
    printed = f"{results['table'][0]['Runtime [s]']:.4g}"
    assert printed[:4] in result.output  # the printed runtime is the recorded one


def test_benchmark_solver_rejects_a_too_small_launch_radius(tmp_path: Path) -> None:
    result = runner.invoke(_app(), ["benchmark", "solver", "--config", str(CONFIGS / "convergence.yaml"),
                                    *_dirs(tmp_path), "--set", "experiment.parameters.solver.launch_radius=2"])
    assert result.exit_code == 2
    assert "launch_radius" in _text(result)


def test_benchmark_cpu_and_precision_run_tiny_workloads(tmp_path: Path) -> None:
    common = ["--set", "observer.radius=30", "--set", "termination.escape_radius=30",
              "--set", "termination.horizon_epsilon=1e-3", "--set", "integration.rtol=1e-6",
              "--set", "integration.atol=1e-8", "--set", "experiment.parameters.benchmark.repeats=1",
              "--set", "experiment.parameters.benchmark.backends=[numpy]"]
    cpu_dir, prec_dir = tmp_path / "cpu", tmp_path / "precision"
    cpu = runner.invoke(_app(), ["benchmark", "cpu", "--config", str(CONFIGS / "benchmark.yaml"), *_dirs(cpu_dir),
                                 *common, "--set", "experiment.parameters.benchmark.ray_counts=[8]",
                                 "--set", "experiment.parameters.benchmark.reference_rays=4"])
    assert cpu.exit_code == 0, _text(cpu)
    assert "runs" in cpu.output and "numpy" in cpu.output
    assert _summary(cpu_dir)["results"]["runs"][0]["status"] == "ok"
    prec = runner.invoke(_app(), ["benchmark", "precision", "--config", str(CONFIGS / "benchmark.yaml"),
                                  *_dirs(prec_dir), *common, "--set", "raytrace.resolution=4",
                                  "--set", "experiment.parameters.benchmark.precision_tolerances=[1e-5]"])
    assert prec.exit_code == 0, _text(prec)
    assert "comparisons" in prec.output and "float32" in prec.output


def test_missing_benchmark_module_exits_2(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(benchmark_cmd.LAZY_BENCHMARKS, "cpu", ("kerrray.benchmarks.no_such_module", "run"))
    result = runner.invoke(_app(), ["benchmark", "cpu", "--config", str(CONFIGS / "benchmark.yaml")])
    assert result.exit_code == 2
    assert "not available" in _text(result)


def test_benchmark_configuration_error_exits_2() -> None:
    result = runner.invoke(_app(), ["benchmark", "gpu", "--config", str(CONFIGS / "gpu.yaml"),
                                    "--set", "black_hole.spin=1.5"])
    assert result.exit_code == 2
