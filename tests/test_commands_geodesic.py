"""Tests for ``kerrray geodesic`` and ``kerrray blackhole create`` on a fresh Typer app (CliRunner)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest
import typer
from typer.testing import CliRunner

from kerrray.commands import blackhole, geodesic
from kerrray.commands.geodesic import launch_constants, launch_state
from kerrray.geodesics import carter_potentials
from kerrray.geometry import kerr
from kerrray.photons.orbits import critical_impact_parameters, isco_radius

REPO = Path(__file__).resolve().parents[1]
CONFIG = str(REPO / "configs" / "kerr.yaml")


def _app() -> typer.Typer:
    app = typer.Typer(add_completion=False)

    @app.callback()
    def _root() -> None:
        """Test application root."""

    blackhole.register(app)
    geodesic.register(app)
    return app


def _dirs(tmp: Path) -> list[str]:
    return ["--set", f"experiment.output_dir={tmp / 'runs'}", "--set", f"experiment.report_dir={tmp / 'reports'}"]


def _value(output: str, label: str) -> str:
    for line in output.splitlines():
        if line.strip().startswith(label):
            return line.strip()[len(label):].strip()
    raise AssertionError(f"{label!r} not in output:\n{output}")


def test_blackhole_create_prints_computed_values() -> None:
    result = CliRunner().invoke(_app(), ["blackhole", "create", "--mass", "1", "--spin", "0.9", "--config", CONFIG])
    assert result.exit_code == 0, result.output
    st = kerr(1.0, 0.9)
    b_pro, b_ret = critical_impact_parameters(st)
    assert float(_value(result.output, "Prograde b_c").split()[0]) == pytest.approx(b_pro, abs=1e-9)
    assert float(_value(result.output, "Retrograde b_c").split()[0]) == pytest.approx(b_ret, abs=1e-9)
    assert float(_value(result.output, "Outer r+").split()[0]) == pytest.approx(1.0 + math.sqrt(1.0 - 0.81), abs=1e-9)
    assert f"{isco_radius(st, True):.9f}" in result.output and f"{isco_radius(st, False):.9f}" in result.output


def test_blackhole_create_rejects_extremal_spin() -> None:
    result = CliRunner().invoke(_app(), ["blackhole", "create", "--spin", "1.0", "--config", CONFIG])
    assert result.exit_code == 1


@pytest.mark.parametrize("theta", [30.0, 60.0, 90.0])
def test_launch_state_is_a_polar_turning_point(theta: float) -> None:
    st = kerr(1.0, 0.9)
    xi, eta = launch_constants(st, theta, 5.2, True)
    assert xi > 0.0
    _, big_theta, _, scale = carter_potentials(st, 1000.0, math.radians(theta), 1.0, xi, eta)
    assert abs(big_theta) <= 1e-12 * max(scale, 1.0)
    y0 = launch_state(st, 1000.0, theta, 5.2, False)
    assert y0[7] < 0.0 and y0[5] < 0.0  # retrograde for a > 0, moving inward
    with pytest.raises(ValueError):
        launch_constants(st, 0.0, 5.2, True)


def test_geodesic_command_reports_diagnostics(tmp_path: Path) -> None:
    args = ["geodesic", "--config", CONFIG, "--spin", "0.9", "--theta", "60", "--impact-parameter", "5.2",
            "--prograde", "--method", "rk45", "--rtol", "1e-8", "--launch-radius", "100", *_dirs(tmp_path)]
    result = CliRunner().invoke(_app(), args)
    assert result.exit_code == 0, result.output
    for label in ("Termination", "Max null error", "Energy drift", "L_z drift", "Carter Q drift", "Closest approach",
                  "Turns", "Steps", "Runtime"):
        _value(result.output, label)
    assert _value(result.output, "Termination") == "ESCAPED"
    assert float(_value(result.output, "Max null error")) < 1e-6
    runs = list((tmp_path / "runs").iterdir())
    assert len(runs) == 1
    manifest = json.loads((runs[0] / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["experiment"] == "geodesic" and manifest["results"]["state"] == "ESCAPED"
    assert (tmp_path / "reports" / runs[0].name / "trajectory.png").is_file()


def test_geodesic_command_capture_without_plot(tmp_path: Path) -> None:
    args = ["geodesic", "--config", CONFIG, "--spin", "0.0", "--theta", "90", "--impact-parameter", "4.0",
            "--retrograde", "--launch-radius", "50", "--no-plot", *_dirs(tmp_path)]
    result = CliRunner().invoke(_app(), args)
    assert result.exit_code == 0, result.output
    assert _value(result.output, "Termination") == "CAPTURED"
    assert not list((tmp_path / "reports").rglob("trajectory.png"))


def test_geodesic_command_rejects_bad_input(tmp_path: Path) -> None:
    result: Any = CliRunner().invoke(_app(), ["geodesic", "--config", CONFIG, "--theta", "0", *_dirs(tmp_path)])
    assert result.exit_code == 1
    result = CliRunner().invoke(_app(), ["geodesic", "--config", CONFIG, "--set", "black_hole.unknown=1"])
    assert result.exit_code == 1
