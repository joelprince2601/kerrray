"""Tests for the float32/float64 precision study (kerrray.benchmarks.precision, EXP-011) and the GPU stub."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from rich.console import Console

from kerrray.benchmarks.gpu import GPU_UNAVAILABLE_MESSAGE, gpu_status, run_gpu_benchmark
from kerrray.benchmarks.precision import mask_metrics, run_precision_study
from kerrray.utils.config import load_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _grid(n: int = 20, half: float = 10.0) -> tuple[np.ndarray, np.ndarray, float]:
    pixel = 2.0 * half / n
    centres = -half + pixel * (np.arange(n) + 0.5)
    beta, alpha = np.meshgrid(centres[::-1], centres, indexing="ij")
    return alpha, beta, pixel


def test_mask_metrics_of_identical_and_shrunk_disks() -> None:
    alpha, beta, pixel = _grid()
    rad = np.hypot(alpha, beta)
    big, small = rad < 5.0, rad < 4.0
    same = mask_metrics(big, big, alpha, beta, pixel)
    assert same["n_differing_pixels"] == 0
    assert same["equivalent_radius_diff_M"] == 0.0 and same["boundary_radius_diff_M"] == 0.0
    diff = mask_metrics(big, small, alpha, beta, pixel)
    assert diff["n_differing_pixels"] == int(np.count_nonzero(big ^ small))
    assert diff["equivalent_radius_diff_M"] > 0.0 and diff["boundary_radius_diff_M"] > 0.0
    assert diff["equivalent_radius_diff_px"] == pytest.approx(diff["equivalent_radius_diff_M"] / pixel)
    with pytest.raises(ValueError):
        mask_metrics(big, small[:-1], alpha, beta, pixel)


@pytest.fixture(scope="module")
def precision_record(tmp_path_factory: pytest.TempPathFactory):
    tmp = tmp_path_factory.mktemp("precision")
    cfg = load_config(CONFIGS / "benchmark.yaml", {
        "experiment.output_dir": str(tmp / "runs"),
        "experiment.report_dir": str(tmp / "reports"),
        "observer.radius": 50.0,
        "termination.escape_radius": 50.0,
        "raytrace.resolution": 8,
        "experiment.parameters.benchmark": {"backends": ["numpy"], "precision_tolerances": [1e-5], "repeats": 1},
    })
    return run_precision_study(cfg)


def test_precision_runs_use_the_documented_float32_horizon_margin(precision_record) -> None:
    res = precision_record.results
    eps32 = res["parameters"]["horizon_epsilon_float32"]
    assert res["horizon_epsilon"] == {"float64": 1e-6, "float32": eps32}
    runs = {(r["dtype"], r["role"]): r for r in res["runs"]}
    assert set(runs) == {("float32", "main"), ("float64", "main"), ("float64", "control")}
    assert runs[("float32", "main")]["horizon_epsilon"] == eps32
    assert runs[("float64", "control")]["horizon_epsilon"] == eps32
    assert runs[("float64", "main")]["horizon_epsilon"] == 1e-6
    for run in res["runs"]:
        assert run["runtime_s"] > 0
        assert run["captured"] + run["escaped"] + run["other"] == 64
        assert run["max_null_error_escaped"] is not None


def test_precision_comparisons_are_computed(precision_record) -> None:
    comparisons = precision_record.results["comparisons"]
    assert [c["compared"] for c in comparisons] == [
        "float64_control vs float32", "float64 vs float64_control (margin only)"]
    for cmp in comparisons:
        assert cmp["n_differing_pixels"] >= 0 and cmp["n_captured_a"] > 0
    assert comparisons[0]["runtime_ratio_f32_over_f64"] > 0
    report = (precision_record.report_dir / "report.md").read_text(encoding="utf-8")
    assert "horizon_epsilon" in report and "float64_control" in report
    assert (precision_record.report_dir / "precision_tables.md").is_file()


def test_gpu_stub_reports_unavailability_and_no_numbers(tmp_path: Path) -> None:
    status = gpu_status()
    assert isinstance(status.available, bool) and status.detail
    cfg = load_config(CONFIGS / "gpu.yaml", {"experiment.output_dir": str(tmp_path / "runs"),
                                             "experiment.report_dir": str(tmp_path / "reports")})
    console = Console(record=True, width=200)
    rec = run_gpu_benchmark(cfg, console=console)
    assert rec.results["gpu_numbers_reported"] is False
    assert rec.results["gpu_available"] is status.available
    if not status.available:
        assert rec.results["message"] == GPU_UNAVAILABLE_MESSAGE
        assert "optional future work" in console.export_text()
    numeric = [k for k, v in rec.results.items() if isinstance(v, (int, float)) and not isinstance(v, bool)]
    assert numeric == []  # never a timing, speed-up or memory figure
