"""Tests for the step-size and tolerance study (kerrray.experiments.step_size, EXP-006)."""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from kerrray.experiments.step_size import common_ray_errors, estimate_order, fit_series, run
from kerrray.utils.config import load_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_estimate_order_recovers_a_synthetic_power_law() -> None:
    h = np.array([0.4, 0.2, 0.1, 0.05])
    fit = estimate_order(h, 3.0 * h**4)
    assert fit["order"] == pytest.approx(4.0, rel=1e-10)
    assert all(v == pytest.approx(4.0, rel=1e-10) for v in fit["successive"])


def test_estimate_order_is_nan_without_two_usable_points() -> None:
    assert math.isnan(estimate_order([0.1], [1e-3])["order"])
    assert math.isnan(estimate_order([0.1, 0.05], [math.nan, 1e-3])["order"])


def _fake(solver: str, setting: float, errors: list[float]) -> SimpleNamespace:
    return SimpleNamespace(configuration=SimpleNamespace(solver=solver, setting=setting),
                           trajectory_error=np.array(errors))


def test_common_ray_errors_use_only_rays_finite_at_every_setting() -> None:
    results = [
        _fake("rk4", 0.2, [1e-2, 5.0, math.nan]),
        _fake("rk4", 0.1, [1e-3, math.nan, 1e-3]),
        _fake("rk45", 1e-6, [1.0, 1.0, 1.0]),
    ]
    settings, values, n_common = common_ray_errors(results, "rk4")
    assert settings == [0.2, 0.1] and n_common == 1
    assert values == [1e-2, 1e-3]
    fit = fit_series(results, "rk4", "common_trajectory_error")
    assert fit["order"] == pytest.approx(math.log(10.0) / math.log(2.0))
    assert fit["n_common_rays"] == 1
    results.append(_fake("rk4", 0.05, [1e-4, 1.0, 3e-4]))
    _, medians, n_common = common_ray_errors(results, "rk4", "median")
    assert n_common == 1 and medians == [1e-2, 1e-3, 1e-4]


def test_step_size_run_reports_fitted_orders(tmp_path: Path) -> None:
    cfg = load_config(CONFIGS / "convergence.yaml", {
        "experiment.output_dir": str(tmp_path / "runs"),
        "experiment.report_dir": str(tmp_path / "reports"),
        "experiment.parameters.convergence.step_sizes": [0.2, 0.1],
        "experiment.parameters.convergence.tolerances": [1e-5, 1e-7],
        "experiment.parameters.solver": {"n_rays": 4, "repeats": 1, "launch_radius": 15.0,
                                         "reference_rtol": 1e-10, "reference_atol": 1e-12},
    })
    rec = run(cfg)
    res = rec.results
    assert [c["label"] for c in res["configurations"]] == ["RK4 h = 0.2", "RK4 h = 0.1",
                                                            "RK45 rtol = 1e-05", "RK45 rtol = 1e-07"]
    rk4, rk45 = res["fits"]["rk4_trajectory_error"], res["fits"]["rk45_trajectory_error"]
    assert rk4["settings"] == [0.2, 0.1] and rk45["settings"] == [1e-5, 1e-7]
    assert rk4["n_common_rays"] >= 1 and rk45["n_common_rays"] == 4
    # the orders are fitted from the measured points, whatever their value
    assert res["fitted_order_rk4"] == rk4["order"]
    assert res["fitted_order_rk4_median"] == res["fits"]["rk4_trajectory_error_median"]["order"]
    assert rk4["order"] == pytest.approx(math.log(rk4["values"][0] / rk4["values"][1]) / math.log(2.0))
    assert rk45["order"] == pytest.approx(math.log(rk45["values"][0] / rk45["values"][1]) / math.log(100.0))
    assert rk45["order"] > 0.0 and rk4["order"] > 0.0  # errors fall as the step or tolerance is refined
    assert (rec.report_dir / "report.md").is_file()
    assert (rec.report_dir / "convergence_rk4.png").is_file()
    assert (rec.report_dir / "convergence_rk45.png").is_file()
