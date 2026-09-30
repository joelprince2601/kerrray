"""Tests for ``kerrray trace``, ``kerrray shadow`` and the shadow experiment driver."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest
import typer
from typer.testing import CliRunner

from kerrray.commands import shadow as shadow_cmd
from kerrray.commands import trace as trace_cmd
from kerrray.commands.trace import build_overrides
from kerrray.experiments import shadow as shadow_exp
from kerrray.raytracing.shadow import ShadowImage
from kerrray.reporting.report import REPORT_HEADINGS
from kerrray.utils.config import ConfigError, load_config
from kerrray.utils.config_blocks import INCLINATION_EPSILON_DEG
from kerrray.utils.manifest import CONFIG_FILENAME, MANIFEST_FILENAME

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "shadow.yaml"
FAST = ["--fov", "8", "--rtol", "1e-7", "--atol", "1e-9"]


def _app() -> typer.Typer:
    app = typer.Typer(add_completion=False)

    @app.callback()
    def _root() -> None:
        """Test application root (forces sub-command mode)."""

    trace_cmd.register(app)
    shadow_cmd.register(app)
    return app


def _dirs(tmp_path: Path) -> list[str]:
    return ["--set", f"experiment.output_dir={tmp_path / 'runs'}",
            "--set", f"experiment.report_dir={tmp_path / 'reports'}"]


def _output(result: Any) -> str:
    text = result.output
    try:
        text += result.stderr
    except ValueError:
        pass
    return text


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _single_run(tmp_path: Path) -> tuple[Path, Path]:
    runs = [p for p in (tmp_path / "runs").iterdir() if p.is_dir()]
    reports = [p for p in (tmp_path / "reports").iterdir() if p.is_dir()]
    assert len(runs) == 1 and len(reports) == 1 and runs[0].name == reports[0].name
    return runs[0], reports[0]


# --- kerrray trace -----------------------------------------------------------------


def test_trace_command_prints_panels_and_writes_files(tmp_path: Path) -> None:
    args = ["trace", "--config", str(CONFIG_PATH), "--spin", "0.9", "--inclination", "60",
            "--resolution", "8", *FAST, *_dirs(tmp_path)]
    result = CliRunner().invoke(_app(), args)
    assert result.exit_code == 0, _output(result)
    out = result.output
    for label in ("Spacetime", "Observer", "Ray Trace", "Status", "Results", "Captured", "Escaped",
                  "Numerical fail", "Max null error", "Energy drift", "Runtime", "8 x 8"):
        assert label in out
    run_dir, report_dir = _single_run(tmp_path)
    assert (run_dir / MANIFEST_FILENAME).is_file() and (run_dir / CONFIG_FILENAME).is_file()
    assert (report_dir / trace_cmd.FIGURE_FILENAME).is_file()
    summary = _read_json(report_dir / "summary.json")
    results = summary["results"]
    assert summary["experiment"] == "trace" and summary["status"] == "completed"
    assert sum(results["counts"].values()) == results["n_rays"] == 64
    assert results["n_failed"] == 0
    assert results["rays_per_second"] > 0.0 and results["trace_runtime_s"] > 0.0
    img = ShadowImage.load(run_dir / trace_cmd.STATE_MAP_FILENAME)
    assert img.counts == results["counts"]
    assert img.spacetime.spin == 0.9 and img.camera.resolution == 8
    assert f"{results['counts']['CAPTURED']:,}" in out
    assert load_config(run_dir / CONFIG_FILENAME).black_hole.spin == 0.9


def test_shadow_command_reports_boundary_metrics(tmp_path: Path) -> None:
    args = ["shadow", "--config", str(CONFIG_PATH), "--spin", "0", "--resolution", "16",
            *FAST, *_dirs(tmp_path)]
    result = CliRunner().invoke(_app(), args)
    assert result.exit_code == 0, _output(result)
    out = result.output
    for label in ("Shadow boundary", "Mean radius", "RMS error", "Max error", "Centroid shift", "Finite-distance"):
        assert label in out
    run_dir, report_dir = _single_run(tmp_path)
    assert (report_dir / shadow_cmd.BOUNDARY_FIGURE_FILENAME).is_file()
    results = _read_json(report_dir / "summary.json")["results"]
    assert results["boundary_error"]["rms_px"] < 1.0
    assert results["boundary_error"]["max_px"] < 1.0
    assert results["boundary_n_angles"] == 360
    assert results["finite_distance_scale"] == pytest.approx(1.0 / (1.0 - 2.0e-3) ** 0.5)
    assert abs(results["boundary_error"]["analytic_asymmetry"]) < 1e-9  # circle for a = 0


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--resolution", "7"], "even integer"),
        (["--backend", "cuda"], "optional future work"),
        (["--set", "novalue"], "key.path=value"),
        (["--set", "black_hole.spin=1.5"], "spin"),
    ],
)
def test_user_errors_exit_with_code_1(tmp_path: Path, extra: list[str], message: str) -> None:
    args = ["trace", "--config", str(CONFIG_PATH), "--resolution", "8", *FAST, *_dirs(tmp_path), *extra]
    result = CliRunner().invoke(_app(), args)
    assert result.exit_code == 1
    assert message in _output(result)
    assert not (tmp_path / "runs").exists()


def test_missing_config_exits_with_code_1(tmp_path: Path) -> None:
    result = CliRunner().invoke(_app(), ["trace", "--config", str(tmp_path / "none.yaml")])
    assert result.exit_code == 1
    assert "error" in _output(result)


def test_build_overrides_maps_flags_to_dotted_keys() -> None:
    assert build_overrides() == {}
    overrides = build_overrides(spin=0.5, resolution=32, method="rk4", sets=["experiment.seed=3", "a.b = c"])
    assert overrides == {"black_hole.spin": 0.5, "raytrace.resolution": 32, "integration.method": "rk4",
                         "experiment.seed": "3", "a.b": "c"}
    with pytest.raises(ConfigError):
        build_overrides(sets=["=x"])


# --- experiments.shadow ------------------------------------------------------------


def _cfg(tmp_path: Path, **params: Any) -> Any:
    overrides: dict[str, Any] = {
        "experiment.output_dir": str(tmp_path / "runs"),
        "experiment.report_dir": str(tmp_path / "reports"),
        "raytrace.resolution": 8, "raytrace.fov": 8.0,
        "integration.rtol": 1e-7, "integration.atol": 1e-9,
    }
    overrides.update({f"experiment.parameters.{key}": value for key, value in params.items()})
    return load_config(CONFIG_PATH, overrides)


def _check_report(report_dir: Path) -> str:
    text = (report_dir / "report.md").read_text(encoding="utf-8")
    for _, heading in REPORT_HEADINGS:
        assert heading in text
    assert (report_dir / "summary.json").is_file() and (report_dir / "metadata.json").is_file()
    return text


def test_spin_sweep_writes_table_figures_and_report(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, **{"mode": "spin_sweep", "spin_sweep.spins": [0.0, 0.9]})
    record = shadow_exp.run(cfg)
    rows = record.results["rows"]
    assert record.results["mode"] == "spin_sweep" and [r["spin"] for r in rows] == [0.0, 0.9]
    for row in rows:
        assert row["captured"] > 0 and row["failed"] == 0
        assert row["rms_error_px"] < 1.0 and row["radius_mean"] > 0.0
    assert rows[1]["centroid_alpha"] > rows[0]["centroid_alpha"]  # a = 0.9 shadow is displaced
    for figure in record.results["figures"]:
        assert Path(figure).is_file()
    assert {Path(f).name for f in record.results["figures"]} >= {
        "shadow_spin_0.png", "shadow_spin_0.9.png", "radius_vs_parameter.png", "error_vs_parameter.png"}
    assert (record.run_dir / "shadow_spin_0.npz").is_file()
    text = _check_report(record.report_dir)
    assert "| spin |" in text and "No run was skipped." in text


def test_inclination_sweep_uses_the_clamped_axis_angle(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, **{"mode": "inclination_sweep", "inclination_sweep.inclinations_deg": [0, 90]})
    record = shadow_exp.run(cfg)
    rows = record.results["rows"]
    assert [r["inclination_deg"] for r in rows] == [INCLINATION_EPSILON_DEG, 90.0]
    assert all(r["failed"] == 0 and "rms_error" in r for r in rows)
    _check_report(record.report_dir)


def test_convergence_skips_over_budget_runs_and_logs_them(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    cfg = _cfg(tmp_path, **{
        "mode": "convergence", "convergence.resolutions": [8, 16], "convergence.tolerances": [1e-6],
        "convergence.max_seconds_per_run": 1e-3,
    })
    with caplog.at_level(logging.WARNING, logger="kerrray"):
        record = shadow_exp.run(cfg)
    results = record.results
    assert [r["resolution"] for r in results["resolution_rows"]] == [8]  # the first run always executes
    assert results["rtol_rows"] == []
    skipped = results["skipped"]
    assert [s["label"] for s in skipped] == ["resolution_16", "rtol_1e-06"]
    assert all(s["estimated_seconds"] > s["max_seconds_per_run"] == 1e-3 for s in skipped)
    assert sum("skipped" in r.getMessage() for r in caplog.records) == 2
    text = _check_report(record.report_dir)
    assert "resolution_16" in text and "rtol_1e-06" in text and "exceeds the budget" in text
    assert results["resolution_rows"][0]["rms_vs_finest"] == 0.0


def test_convergence_compares_against_finest_run(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, **{"mode": "convergence", "convergence.resolutions": [8, 12],
                           "convergence.tolerances": [1e-6, 1e-7]})
    record = shadow_exp.run(cfg)
    res_rows, tol_rows = record.results["resolution_rows"], record.results["rtol_rows"]
    assert [r["resolution"] for r in res_rows] == [8, 12]
    assert [r["rtol"] for r in tol_rows] == [1e-6, 1e-7]
    assert res_rows[0]["rms_vs_finest"] > 0.0 and res_rows[1]["rms_vs_finest"] == 0.0
    assert res_rows[0]["rms_vs_finest_px"] == pytest.approx(res_rows[0]["rms_vs_finest"] / res_rows[0]["pixel_size"])
    assert record.results["skipped"] == []
    assert "resolution_slope_vs_analytic" in record.results
    assert Path(record.report_dir / "error_vs_resolution.png").is_file()


def test_run_rejects_an_unknown_mode(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="mode"):
        shadow_exp.run(_cfg(tmp_path, mode="bogus"))
    with pytest.raises(ConfigError):
        shadow_exp.ConvergenceParams(max_seconds_per_run=0.0)


def test_default_config_path_resolves_from_any_working_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The relative default ``configs/shadow.yaml`` falls back to the repository root."""
    monkeypatch.chdir(tmp_path)
    resolved = trace_cmd.resolve_config_path(trace_cmd.DEFAULT_CONFIG)
    assert resolved.is_file() and resolved == CONFIG_PATH.resolve()
    local = tmp_path / "configs" / "shadow.yaml"
    local.parent.mkdir()
    local.write_text(CONFIG_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    assert trace_cmd.resolve_config_path(trace_cmd.DEFAULT_CONFIG) == trace_cmd.DEFAULT_CONFIG  # a local file wins
    missing = Path("configs") / "no_such_file.yaml"
    assert trace_cmd.resolve_config_path(missing) == missing  # left for load_config to report
