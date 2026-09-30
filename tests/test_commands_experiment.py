"""Tests for ``kerrray experiment`` (kerrray.commands.experiment) on a fresh Typer app."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import typer
from typer.testing import CliRunner

from kerrray.commands import experiment as experiment_cmd
from kerrray.utils.config_parsing import ConfigError

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
runner = CliRunner()


def _app() -> typer.Typer:
    app = typer.Typer()

    @app.callback()
    def main() -> None:
        """Test application (forces sub-command mode)."""

    experiment_cmd.register(app)
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


def test_parse_overrides_reads_yaml_scalars_and_lists() -> None:
    parsed = experiment_cmd.parse_overrides(["a.b=1", "c=[0.5, 0.25]", "d=text", "e=1e-6"])
    assert parsed == {"a.b": 1, "c": [0.5, 0.25], "d": "text", "e": 1e-6}
    with pytest.raises(ConfigError):
        experiment_cmd.parse_overrides(["no_equals_sign"])


def test_available_experiments_and_unknown_names() -> None:
    names = experiment_cmd.available_experiments()
    assert "near_critical" in names and "step_size" in names and "base" not in names
    with pytest.raises(LookupError, match="available: .*near_critical"):
        experiment_cmd.load_experiment_module("no_such_experiment")


def test_unknown_experiment_lists_the_available_ones(tmp_path: Path) -> None:
    result = runner.invoke(_app(), ["experiment", "--config", str(CONFIGS / "convergence.yaml"),
                                    "--name", "no_such_experiment", *_dirs(tmp_path)])
    assert result.exit_code == 2
    assert "near_critical" in _text(result) and "step_size" in _text(result)


def test_config_name_without_module_is_reported(tmp_path: Path) -> None:
    # experiment.name of convergence.yaml is "convergence", which has no driver module
    result = runner.invoke(_app(), ["experiment", "--config", str(CONFIGS / "convergence.yaml"), *_dirs(tmp_path)])
    assert result.exit_code == 2
    assert "unknown experiment 'convergence'" in _text(result)


def test_usage_and_configuration_errors_exit_2(tmp_path: Path) -> None:
    assert runner.invoke(_app(), ["experiment"]).exit_code == 2
    bad = runner.invoke(_app(), ["experiment", "--config", str(CONFIGS / "near_critical.yaml"),
                                 "--set", "black_hole.spin=2.0"])
    assert bad.exit_code == 2 and "error" in _text(bad)
    missing = runner.invoke(_app(), ["experiment", "--config", str(tmp_path / "missing.yaml")])
    assert missing.exit_code == 2


def test_dispatch_passes_the_overridden_config_to_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def fake_run(cfg: Any) -> SimpleNamespace:
        seen["cfg"] = cfg
        return SimpleNamespace(run_id="RUN-1", runtime_s=0.25, run_dir=tmp_path, report_dir=tmp_path,
                               summary_path=tmp_path / "summary.json", results={"answer": 42.5})

    monkeypatch.setattr(experiment_cmd, "load_experiment_module",
                        lambda name: SimpleNamespace(run=fake_run) if name == "fake" else None)
    result = runner.invoke(_app(), ["experiment", "--config", str(CONFIGS / "near_critical.yaml"), "--name", "fake",
                                    "--set", "black_hole.spin=0.5", *_dirs(tmp_path)])
    assert result.exit_code == 0, _text(result)
    assert seen["cfg"].black_hole.spin == 0.5
    assert "42.5" in result.output and "RUN-1" in result.output


def test_real_dispatch_to_the_step_size_experiment(tmp_path: Path) -> None:
    args = ["experiment", "--config", str(CONFIGS / "convergence.yaml"), "--name", "step_size", *_dirs(tmp_path),
            "--set", "experiment.parameters.convergence.step_sizes=[0.2]",
            "--set", "experiment.parameters.convergence.tolerances=[1e-5]",
            "--set", "experiment.parameters.solver.n_rays=2",
            "--set", "experiment.parameters.solver.repeats=1",
            "--set", "experiment.parameters.solver.launch_radius=15",
            "--set", "experiment.parameters.solver.reference_rtol=1e-8",
            "--set", "experiment.parameters.solver.reference_atol=1e-10"]
    result = runner.invoke(_app(), args)
    assert result.exit_code == 0, _text(result)
    assert "step_size" in result.output
    summaries = list((tmp_path / "reports").glob("*/summary.json"))
    assert len(summaries) == 1
    assert (summaries[0].parent / "report.md").is_file()
