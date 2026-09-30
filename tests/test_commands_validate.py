"""Tests for ``kerrray validate schwarzschild`` and ``kerrray validate kerr`` on a fresh Typer app.

The runs are shrunk with ``--set`` (launch and escape at 50 M, rtol 1e-8, no
convergence study, one spin per Kerr family) so the module runs in seconds;
the default-configuration outputs are recorded in docs/validation.md.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from kerrray.commands import validate
from kerrray.commands.validate import parse_sets, schwarzschild_rows
from kerrray.utils.config import ConfigError
from kerrray.validation import ValidationCheck, ValidationReport
from kerrray.validation.convergence import convergence_study
from kerrray.validation.schwarzschild import CRITICAL_IMPACT_REFERENCE, PHOTON_SPHERE_REFERENCE

REPO = Path(__file__).resolve().parents[1]
SCHW = str(REPO / "configs" / "schwarzschild.yaml")
KERR = str(REPO / "configs" / "kerr.yaml")
FAST = ["--rtol", "1e-8", "--set", "integration.atol=1e-10", "--set", "termination.escape_radius=50",
        "--set", "observer.radius=50"]


def _app() -> typer.Typer:
    app = typer.Typer(add_completion=False)

    @app.callback()
    def _root() -> None:
        """Test application root."""

    validate.register(app)
    return app


def _dirs(tmp: Path) -> list[str]:
    return ["--set", f"experiment.output_dir={tmp / 'runs'}", "--set", f"experiment.report_dir={tmp / 'reports'}"]


def _value(output: str, label: str) -> str:
    for line in output.splitlines():
        if line.strip().startswith(label):
            return line.strip()[len(label):].strip()
    raise AssertionError(f"{label!r} not in output:\n{output}")


def test_parse_sets() -> None:
    assert parse_sets(["a.b=1e-8", "c=[1, 2]", "d=false", "e=text"]) == {"a.b": 1e-8, "c": [1, 2], "d": False,
                                                                         "e": "text"}
    with pytest.raises(ConfigError):
        parse_sets(["novalue"])


def test_validate_schwarzschild_passes_and_prints_section_12_lines(tmp_path: Path) -> None:
    args = ["validate", "schwarzschild", "--config", SCHW, *FAST, *_dirs(tmp_path),
            "--set", "experiment.parameters.validation.run_convergence=false",
            "--set", "experiment.parameters.validation.launch_radius=50"]
    result = CliRunner().invoke(_app(), args)
    assert result.exit_code == 0, result.output
    assert float(_value(result.output, "Analytical photon sphere").split()[0]) == pytest.approx(PHOTON_SPHERE_REFERENCE)
    numerical = float(_value(result.output, "Numerical photon sphere").split()[0])
    assert abs(numerical - PHOTON_SPHERE_REFERENCE) <= 1e-6
    b_num = float(_value(result.output, "Numerical critical impact b_c").split()[0])
    assert abs(b_num - CRITICAL_IMPACT_REFERENCE) <= 1e-6
    rel = float(_value(result.output, "Relative error"))
    assert rel == pytest.approx(abs(numerical - PHOTON_SPHERE_REFERENCE) / PHOTON_SPHERE_REFERENCE, rel=1e-3)
    assert _value(result.output, "All checks") == "PASS"
    (run,) = (tmp_path / "runs").iterdir()
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["experiment"] == "validate_schwarzschild" and manifest["results"]["all_passed"] is True


def test_validate_schwarzschild_fails_with_coarse_tolerance(tmp_path: Path) -> None:
    args = ["validate", "schwarzschild", "--config", SCHW, "--rtol", "1e-4", "--set", "integration.atol=1e-6",
            "--set", "termination.escape_radius=50", "--set", "observer.radius=50", *_dirs(tmp_path),
            "--set", "experiment.parameters.validation.run_convergence=false",
            "--set", "experiment.parameters.validation.launch_radius=50"]
    result = CliRunner().invoke(_app(), args)
    assert result.exit_code == 1
    assert "FAIL" in result.output


def test_validate_schwarzschild_rejects_spin_and_bad_bracket(tmp_path: Path) -> None:
    result = CliRunner().invoke(_app(), ["validate", "schwarzschild", "--config", SCHW, *_dirs(tmp_path),
                                         "--set", "black_hole.spin=0.5"])
    assert result.exit_code == 1
    result = CliRunner().invoke(_app(), ["validate", "schwarzschild", "--config", SCHW, *FAST, *_dirs(tmp_path),
                                         "--set", "experiment.parameters.validation.photon_sphere_bracket=[3.5, 4.0]"])
    assert result.exit_code == 1


def test_schwarzschild_rows_render_convergence_tables() -> None:
    study = convergence_study("rtol", [1e-6, 1e-8, 1e-10], [5.2 + 1e-7, 5.2 + 1e-9, 5.2 + 1e-11], 5.2,
                              noise_floor=1e-12, extra=[{"runtime_s": 1.0}] * 3)
    base = {"width": 1e-7, "iterations": 5, "n_rays": 80, "max_steps": 300, "runtime_s": 2.0}
    checks = [
        ValidationCheck("photon_sphere", "", {"value": 3.0 + 1e-8, **base}, {"value": 3.0},
                        {"absolute": 1e-8, "relative": 3.3e-9}, {"absolute": 1e-6}, True),
        ValidationCheck("critical_impact", "", {"value": 5.2, **base}, {"value": 5.2},
                        {"absolute": 0.0, "relative": 0.0}, {"absolute": 1e-6}, True),
        ValidationCheck("rtol_convergence", "", study.as_dict(), {"value": 5.2}, {"finest": 1e-11},
                        {"finest_absolute": 1e-6}, True),
    ]
    titles = dict(schwarzschild_rows(ValidationReport.from_checks("schwarzschild", 0.0, checks)))
    conv = dict(titles["Convergence of b_c with rtol (rk45)"])
    assert conv["Fitted order"].startswith("1.0000")
    assert conv["Richardson order"].startswith("1.0000")
    assert dict(titles["Verdict"])["All checks"] == "PASS"


def test_validate_kerr_small_run(tmp_path: Path) -> None:
    args = ["validate", "kerr", "--config", KERR, *FAST, *_dirs(tmp_path),
            "--set", ("experiment.parameters.validation={small_spins: [0.01], critical_spins: [0.5], "
                      "conservation_spins: [0.9], near_extremal_spins: [0.99], critical_b_tol: 1.0e-4, "
                      "schwarzschild_limit_tol: 1.0e-4, conservation_tol: 1.0e-7, launch_radius: 50, impact_bracket: [2.0, 8.0], "
                      "reversibility_tol: 1.0e-5}")]
    result = CliRunner().invoke(_app(), args)
    assert result.exit_code == 0, result.output
    for title in ("Horizon and ergosphere", "Schwarzschild limit", "Critical impact parameters", "Conservation",
                  "Near-extremal", "Forward/backward consistency", "Verdict"):
        assert title in result.output
    assert _value(result.output, "All checks") == "PASS"
