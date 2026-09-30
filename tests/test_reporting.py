"""Tests for plots, tables, the report writer and the ``report`` CLI command."""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pytest
import typer
from matplotlib.axes import Axes
from rich.table import Table
from typer.testing import CliRunner

from kerrray.commands.report import register, resolve_run_dir, result_rows
from kerrray.experiments.base import ExperimentContext, run_experiment
from kerrray.reporting.plots import (
    SERIES_COLORS,
    loglog_slope,
    plot_conservation,
    plot_convergence,
    plot_lines,
    plot_shadow,
    plot_trajectory_xy,
    series_style,
)
from kerrray.reporting.report import (
    METADATA_FILENAME,
    REPORT_FILENAME,
    REPORT_HEADINGS,
    ReportSections,
    write_report,
)
from kerrray.reporting.tables import column_names, format_cell, markdown_table, rich_table
from kerrray.utils.config import KerrRayConfig, load_config
from kerrray.utils.manifest import EnvironmentInfo, collect_environment

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "kerr.yaml"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@pytest.fixture(scope="module")
def env() -> EnvironmentInfo:
    return collect_environment()


def _assert_png(path: Path) -> None:
    assert path.is_file()
    assert path.read_bytes()[:8] == PNG_SIGNATURE


# --- plots ------------------------------------------------------------------------


def test_headless_backend_is_selected() -> None:
    assert matplotlib.get_backend().lower() == "agg"


def test_series_styles_are_fixed_order_and_secondary_encoding() -> None:
    assert len(SERIES_COLORS) == 8
    assert series_style(0)["color"] == SERIES_COLORS[0]
    assert series_style(7)["color"] == SERIES_COLORS[7]
    assert series_style(8)["color"] == SERIES_COLORS[0]
    assert series_style(8)["linestyle"] != series_style(0)["linestyle"]


def test_plot_trajectory_to_file_and_to_axes(tmp_path: Path) -> None:
    lam = np.linspace(0.0, 4 * np.pi, 200)
    x, y = 6.0 * np.cos(lam), 6.0 * np.sin(lam)
    theta = np.linspace(0.0, 2 * np.pi, 90)
    ergo = (2.0 * np.cos(theta), 1.5 * np.sin(theta))
    out = plot_trajectory_xy(tmp_path / "figs" / "trajectory.png", x, y, 1.9, ergo, title="t")
    assert out == tmp_path / "figs" / "trajectory.png"
    _assert_png(out)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    try:
        returned = plot_trajectory_xy(ax, x, y, 1.9)
        assert returned is ax and isinstance(returned, Axes)
        assert ax.get_xlabel() == "x [M]" and ax.get_ylabel() == "y [M]"
        assert len(ax.patches) == 1
    finally:
        plt.close(fig)


def test_plot_conservation_handles_zeros_without_warnings(tmp_path: Path) -> None:
    lam = np.linspace(0.0, 100.0, 50)
    drifts = {
        "energy": np.zeros_like(lam),
        "carter": 1e-12 * (1.0 + lam),
        "null": np.where(lam > 50, np.nan, 1e-14 * lam),
    }
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = plot_conservation(tmp_path / "conservation.png", lam, drifts, title="drift")
    _assert_png(out)


def test_plot_conservation_rejects_bad_input(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one"):
        plot_conservation(tmp_path / "c.png", [0.0, 1.0], {})
    with pytest.raises(ValueError, match="values"):
        plot_conservation(tmp_path / "c.png", [0.0, 1.0], {"e": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="floor"):
        plot_conservation(tmp_path / "c.png", [0.0, 1.0], {"e": [1.0, 2.0]}, floor=0.0)


def test_loglog_slope_recovers_power_law() -> None:
    x = np.array([64.0, 128.0, 256.0, 512.0])
    assert loglog_slope(x, 3.0 * x**-2.0) == pytest.approx(-2.0, abs=1e-10)
    assert loglog_slope(x, 0.5 * x**4.0) == pytest.approx(4.0, abs=1e-10)
    # non-positive and non-finite points are ignored, not fatal
    assert loglog_slope([1.0, 10.0, 100.0, 1000.0], [1.0, 0.1, 0.0, np.nan]) == pytest.approx(-1.0)


def test_loglog_slope_needs_two_usable_points() -> None:
    with pytest.raises(ValueError, match="two positive finite"):
        loglog_slope([1.0, 2.0], [0.0, 1.0])
    with pytest.raises(ValueError, match="same length"):
        loglog_slope([1.0, 2.0], [1.0])


def test_plot_convergence_with_and_without_fit(tmp_path: Path) -> None:
    x = np.array([64, 128, 256, 512])
    y = 2.0 * x**-1.0
    _assert_png(plot_convergence(tmp_path / "conv.png", x, y, "resolution", "error"))
    _assert_png(
        plot_convergence(tmp_path / "conv2.png", x, y, "resolution", "error", fit_slope=False)
    )
    _assert_png(plot_convergence(tmp_path / "conv3.png", [1.0], [1.0], "x", "y"))


def test_plot_shadow_with_mask_and_analytic_curve(tmp_path: Path) -> None:
    axis = np.linspace(-6.0, 6.0, 16)
    alpha, beta = np.meshgrid(axis, axis)
    captured = alpha**2 + beta**2 < 25.0
    theta = np.linspace(0.0, 2 * np.pi, 100)
    curve = (5.0 * np.cos(theta), 5.0 * np.sin(theta))
    _assert_png(plot_shadow(tmp_path / "shadow.png", alpha, beta, captured, curve, title="s"))
    _assert_png(plot_shadow(tmp_path / "shadow1d.png", axis, axis, captured))
    with pytest.raises(ValueError, match="2-D"):
        plot_shadow(tmp_path / "bad.png", axis, axis, captured.ravel())


def test_plot_lines_mapping_and_sequence(tmp_path: Path) -> None:
    x = np.logspace(-6, -12, 4)
    series = {"rk45": (x, x**0.5), "rk4": (x, x**0.8), "dop853": (x, x)}
    _assert_png(
        plot_lines(tmp_path / "lines.png", series, xlabel="rtol", ylabel="error", logx=True, logy=True)
    )
    _assert_png(plot_lines(tmp_path / "one.png", [("only", [0, 1, 2], [1, 2, 3])], markers=False))
    with pytest.raises(ValueError, match="at least one"):
        plot_lines(tmp_path / "none.png", {})


# --- tables -----------------------------------------------------------------------

ROWS: list[dict[str, Any]] = [
    {"resolution": 64, "error": 1 / 3},
    {"resolution": 128, "error": 0.08333333333, "note": "a|b"},
]


def test_column_names_first_seen_order_or_explicit() -> None:
    assert column_names(ROWS) == ["resolution", "error", "note"]
    assert column_names(ROWS, ["note", "error"]) == ["note", "error"]
    assert column_names([]) == []


def test_format_cell_uses_significant_digits() -> None:
    assert format_cell(1 / 3) == "0.333333"
    assert format_cell(1 / 3, precision=3) == "0.333"
    assert format_cell(262144) == "262,144"
    assert format_cell(np.float32(0.5)) == "0.5"
    assert format_cell(None) == ""
    assert format_cell("rk45") == "rk45"


def test_markdown_table_layout() -> None:
    text = markdown_table(ROWS, precision=4)
    lines = text.split("\n")
    assert lines[0] == "| resolution | error | note |"
    assert lines[1] == "| ---: | ---: | --- |"
    assert lines[2] == "| 64 | 0.3333 |  |"
    assert lines[3] == "| 128 | 0.08333 | a\\|b |"
    assert not text.endswith("\n")
    assert markdown_table([]) == ""


def test_rich_table_columns_and_rows() -> None:
    table = rich_table(ROWS, title="convergence")
    assert isinstance(table, Table)
    assert [column.header for column in table.columns] == ["resolution", "error", "note"]
    assert table.row_count == 2
    assert table.columns[0].justify == "right"
    assert table.columns[2].justify == "left"


# --- report -----------------------------------------------------------------------


def test_report_sections_match_headings() -> None:
    names = [name for name, _ in REPORT_HEADINGS]
    assert names == list(ReportSections.__dataclass_fields__)
    assert [heading for _, heading in REPORT_HEADINGS] == [
        "Objective", "Mathematical model", "Numerical method", "Parameters", "Results",
        "Error analysis", "Interpretation", "Limitations", "Reproducibility information",
    ]


def test_write_report_has_nine_headings_figures_tables_and_metadata(
    tmp_path: Path, env: EnvironmentInfo
) -> None:
    report_dir = tmp_path / "reports" / "run-1"
    figure = report_dir / "figures" / "convergence.png"
    figure.parent.mkdir(parents=True)
    figure.write_bytes(PNG_SIGNATURE)
    outside = tmp_path / "shared.png"
    outside.write_bytes(PNG_SIGNATURE)
    sections = ReportSections(
        objective="Measure the photon sphere.",
        results="Computed r_ph = 3.0000012 M.",
        reproducibility="Seed 0.",
    )
    table = markdown_table(ROWS)

    path = write_report(report_dir, sections, [figure, outside], [table], title="EXP-001", environment=env)

    assert path == report_dir / REPORT_FILENAME
    text = path.read_text(encoding="utf-8")
    assert text.startswith("# EXP-001\n")
    headings = re.findall(r"^## (\d+)\. (.+)$", text, re.MULTILINE)
    assert headings == [(str(i), h) for i, (_, h) in enumerate(REPORT_HEADINGS, start=1)]
    assert "Measure the photon sphere." in text
    assert "_Not provided._" in text
    assert "![convergence](figures/convergence.png)" in text
    assert "![shared](../../shared.png)" in text
    assert table in text
    results_pos, error_pos = text.index("## 5. Results"), text.index("## 6. Error analysis")
    assert results_pos < text.index(table) < text.index("![convergence]") < error_pos
    assert env.git_commit in text
    assert "\r\n" not in text

    metadata = json.loads((report_dir / METADATA_FILENAME).read_text(encoding="utf-8"))
    assert metadata["title"] == "EXP-001"
    assert metadata["git_commit"] == env.git_commit
    assert metadata["figures"] == ["figures/convergence.png", "../../shared.png"]
    assert metadata["environment"]["python_version"] == env.python_version
    assert metadata["sections"] == [h for _, h in REPORT_HEADINGS]
    assert "timestamp" in metadata and metadata["n_tables"] == 1


# --- kerrray report ---------------------------------------------------------------


def _app() -> typer.Typer:
    app = typer.Typer(add_completion=False)

    @app.callback()
    def _root() -> None:
        """Test application root (forces sub-command mode)."""

    register(app)
    return app


def _output(result: Any) -> str:
    text = result.output
    try:
        text += result.stderr
    except ValueError:
        pass
    return text


@pytest.fixture
def stored_run(tmp_path: Path, env: EnvironmentInfo) -> Any:
    cfg = load_config(
        CONFIG_PATH,
        {
            "experiment.output_dir": str(tmp_path / "runs"),
            "experiment.report_dir": str(tmp_path / "reports"),
        },
    )

    def body(ctx: ExperimentContext) -> dict[str, Any]:
        return {
            "max_null_error": 1.234e-11,
            "counts": {"captured": 10, "escaped": 54},
            "spins": [0.0, 0.5, 0.9],
            "long": list(range(20)),
        }

    return run_experiment("kerr", cfg, body, environment=env)


def test_report_command_prints_stored_values(stored_run: Any) -> None:
    result = CliRunner().invoke(_app(), ["report", "--run", str(stored_run.run_dir)])
    assert result.exit_code == 0, _output(result)
    out = result.output
    for section in ("Run", "Spacetime", "Solver", "Results"):
        assert section in out
    assert stored_run.run_id in out
    assert "kerr" in out and "completed" in out
    assert "Kerr" in out and "0.9" in out and "Boyer-Lindquist" in out
    assert "RK45" in out and "1e-09" in out
    assert "1.234e-11" in out
    assert "counts.captured" in out and "10" in out
    assert "0.0, 0.5, 0.9" in out
    assert "list of 20 values" in out
    # Rich crops long absolute paths at the console width, so check that a summary was found.
    assert "Summary" in out and "not found" not in out


def test_report_command_reads_bare_run_id_under_runs(
    tmp_path: Path, env: EnvironmentInfo, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    cfg = load_config(CONFIG_PATH)  # relative runs/ and reports/
    record = run_experiment("kerr", cfg, lambda ctx: {"x": 1}, environment=env)
    assert resolve_run_dir(Path(record.run_id)) == Path("runs") / record.run_id
    result = CliRunner().invoke(_app(), ["report", "--run", record.run_id])
    assert result.exit_code == 0, _output(result)
    assert record.run_id in result.output
    assert "not found" not in result.output
    assert f"reports/{record.run_id}/summary.json".replace("/", "\\") in result.output.replace(
        "/", "\\"
    )


def test_report_command_missing_run_exits_1(tmp_path: Path) -> None:
    result = CliRunner().invoke(_app(), ["report", "--run", str(tmp_path / "runs" / "nope")])
    assert result.exit_code == 1
    assert "does not exist" in _output(result)
    empty = tmp_path / "empty-run"
    empty.mkdir()
    result = CliRunner().invoke(_app(), ["report", "--run", str(empty)])
    assert result.exit_code == 1
    assert "manifest.json" in _output(result)


def test_result_rows_flatten_and_summarise() -> None:
    rows = result_rows({"a": {"b": 1}, "c": [1, 2], "d": list(range(7)), "e": [{"f": 1}]})
    assert rows == [("a.b", 1), ("c", "1, 2"), ("d", "list of 7 values"), ("e", "list of 1 values")]
    assert result_rows({}) == [("(none)", "no results recorded")]


def test_report_command_works_without_summary(tmp_path: Path, env: EnvironmentInfo) -> None:
    cfg: KerrRayConfig = load_config(
        CONFIG_PATH,
        {
            "experiment.output_dir": str(tmp_path / "runs"),
            "experiment.report_dir": str(tmp_path / "reports"),
        },
    )
    record = run_experiment("kerr", cfg, lambda ctx: {"y": 2.5}, environment=env)
    record.summary_path.unlink()
    result = CliRunner().invoke(_app(), ["report", "--run", str(record.run_dir)])
    assert result.exit_code == 0, _output(result)
    assert "not found" in result.output and "2.5" in result.output
