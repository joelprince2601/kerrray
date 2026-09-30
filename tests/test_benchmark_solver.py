"""Tests for the solver benchmark (kerrray.benchmarks.solver, rayset, solver_config, solver_report)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from kerrray.benchmarks.rayset import (
    PLUNGE_MARGIN,
    RAY_KINDS,
    build_ray_set,
    comparison_termination,
    position_error,
    reference_solution,
    trajectory_error,
)
from kerrray.benchmarks.solver import (
    ConvergenceParams,
    SolverParams,
    build_configurations,
    minimum_launch_radius,
    run_solver_benchmark,
)
from kerrray.benchmarks.solver_report import table_rows
from kerrray.geodesics import TerminationOptions, hamiltonian
from kerrray.photons import TerminationState
from kerrray.utils.config import load_config
from kerrray.utils.config_parsing import ConfigError

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
LAUNCH = 15.0


def _tiny_overrides(tmp_path: Path, **solver: object) -> dict[str, object]:
    params = {"n_rays": 4, "repeats": 1, "launch_radius": LAUNCH, "reference_rtol": 1e-10,
              "reference_atol": 1e-12, **solver}
    return {
        "experiment.output_dir": str(tmp_path / "runs"),
        "experiment.report_dir": str(tmp_path / "reports"),
        "experiment.parameters.convergence.step_sizes": [0.2, 0.1, 1e-5],
        "experiment.parameters.convergence.tolerances": [1e-5, 1e-7],
        "experiment.parameters.solver": params,
    }


# --- ray set -----------------------------------------------------------------


def test_ray_set_is_seeded_and_covers_every_category() -> None:
    rays_a = build_ray_set(16, [0.0, 0.9], mass=1.0, launch_radius=50.0, inclination_deg=60.0,
                           rng=np.random.default_rng(3))
    rays_b = build_ray_set(16, [0.0, 0.9], mass=1.0, launch_radius=50.0, inclination_deg=60.0,
                           rng=np.random.default_rng(3))
    assert all(np.array_equal(a.y0, b.y0) for a, b in zip(rays_a, rays_b, strict=True))
    assert {r.kind for r in rays_a} == set(RAY_KINDS)
    assert {r.spin for r in rays_a} == {0.0, 0.9}
    assert {r.equatorial for r in rays_a} == {True, False}
    for ray in rays_a:
        assert ray.y0[1] == pytest.approx(50.0)
        assert ray.y0[5] < 0.0  # launched inward
        assert abs(float(hamiltonian(ray.spacetime, ray.y0))) < 1e-10
        if ray.kind == "capture":
            assert ray.scale < 1.0
        elif ray.kind in ("escape", "near_critical_out"):
            assert ray.scale > 1.0


def test_ray_set_rejects_too_small_launch_radius() -> None:
    r_min = minimum_launch_radius([0.0, 0.9], 1.0)
    with pytest.raises(ValueError, match="too small"):
        build_ray_set(4, [0.0, 0.9], mass=1.0, launch_radius=0.5 * r_min, inclination_deg=60.0,
                      rng=np.random.default_rng(0))


def test_reference_outcomes_follow_the_critical_curve_and_self_error_is_zero() -> None:
    rays = build_ray_set(8, [0.0, 0.9], mass=1.0, launch_radius=LAUNCH, inclination_deg=60.0,
                         rng=np.random.default_rng(1))
    term = TerminationOptions(horizon_epsilon=1e-6, escape_radius=LAUNCH)
    for ray in rays:
        ref = reference_solution(ray.spacetime, ray.y0, term, rtol=1e-10, atol=1e-12, lambda_max=1e4,
                                 first_step=0.01)
        expected = TerminationState.CAPTURED if ray.scale < 1.0 else TerminationState.ESCAPED
        assert ref.state == expected, ray.label
        assert position_error(ray.spacetime, ref.y_end, ref.y_end) == 0.0
        assert trajectory_error(ref, ref.lam_end, ref.at(ref.lam_end), int(ref.state)) == pytest.approx(0.0, abs=1e-12)
        # a failed solver state is never compared
        assert math.isnan(trajectory_error(ref, ref.lam_end, ref.y_end, int(TerminationState.NUMERICAL_FAILURE)))


def test_comparison_termination_uses_the_plunge_margin() -> None:
    term = comparison_termination(TerminationOptions(horizon_epsilon=1e-6, escape_radius=40.0))
    assert term.horizon_epsilon == PLUNGE_MARGIN
    assert term.escape_radius == 40.0


# --- configuration -----------------------------------------------------------


def test_solver_params_validation_and_replace() -> None:
    with pytest.raises(ConfigError):
        SolverParams(n_rays=0)
    with pytest.raises(ConfigError):
        SolverParams(launch_radius=-1.0)
    with pytest.raises(ConfigError):
        SolverParams(spins=[1.0])
    params = SolverParams().replace(n_rays=7, repeats=None)
    assert params.n_rays == 7 and params.repeats == SolverParams().repeats


def test_fixed_steps_above_max_steps_are_skipped_with_a_reason() -> None:
    cfg = load_config(CONFIGS / "convergence.yaml")
    conv = ConvergenceParams(step_sizes=[0.1, 1e-4], tolerances=[1e-6])
    configurations, skipped = build_configurations(cfg, conv, launch_radius=50.0)
    assert [c.label for c in configurations] == ["RK4 h = 0.1", "RK45 rtol = 1e-06", "DOP853 rtol = 1e-06"]
    assert len(skipped) == 1 and skipped[0]["setting"] == 1e-4
    assert "max_steps" in skipped[0]["reason"]
    assert configurations[1].integ.atol / configurations[1].integ.rtol == pytest.approx(
        cfg.integration.atol / cfg.integration.rtol)


# --- end-to-end --------------------------------------------------------------


@pytest.fixture(scope="module")
def solver_record(tmp_path_factory: pytest.TempPathFactory):
    tmp_path = tmp_path_factory.mktemp("solver")
    cfg = load_config(CONFIGS / "convergence.yaml", _tiny_overrides(tmp_path, max_seconds_per_config=5.0))
    return run_solver_benchmark(cfg)


def test_solver_benchmark_writes_every_output(solver_record) -> None:
    rec = solver_record
    assert rec.summary_path.is_file() and rec.manifest_path.is_file()
    assert (rec.report_dir / "report.md").is_file()
    assert (rec.report_dir / "runtime_vs_error.png").is_file()
    report = (rec.report_dir / "report.md").read_text(encoding="utf-8")
    assert "| Solver |" in report and "RK45" in report and "DOP853" in report


def test_solver_benchmark_measures_every_configuration(solver_record) -> None:
    res = solver_record.results
    labels = [c["label"] for c in res["configurations"]]
    assert labels == ["RK4 h = 0.2", "RK4 h = 0.1", "RK45 rtol = 1e-05", "RK45 rtol = 1e-07",
                      "DOP853 rtol = 1e-05", "DOP853 rtol = 1e-07"]
    # h = 1e-5: 2 * 15 / 1e-5 = 3e6 estimated steps > max_steps, skipped and recorded, never silent
    assert [s["setting"] for s in res["skipped"]] == [1e-5]
    assert res["skipped"][0]["reason"]
    assert res["n_rays"] == 4 and res["launch_radius"] == LAUNCH
    for conf in res["configurations"]:
        assert conf["runtime_s"] > 0.0
        assert conf["runtime_s"] == min(conf["runtime_repeats_s"])
        assert conf["mean_steps"] > 0
        assert 0.0 <= conf["failure_rate"] <= 1.0
        assert len(conf["trajectory_error"]) == 4
    adaptive = [c for c in res["configurations"] if c["solver"] != "rk4"]
    for conf in adaptive:
        assert conf["failure_rate"] == 0.0 and conf["n_mismatch"] == 0
        # Hamiltonian form: p_t and p_phi are exactly constant along the flow
        assert conf["max_energy_drift"] == 0.0 and conf["max_lz_drift"] == 0.0
    by_label = {c["label"]: c for c in res["configurations"]}
    for solver in ("RK45", "DOP853"):
        loose, tight = by_label[f"{solver} rtol = 1e-05"], by_label[f"{solver} rtol = 1e-07"]
        assert tight["median_trajectory_error"] < loose["median_trajectory_error"]
        assert tight["max_null_error_escaped"] < loose["max_null_error_escaped"]
        assert tight["mean_steps"] > loose["mean_steps"]


def test_solver_table_rows_match_the_summaries(solver_record) -> None:
    rows = solver_record.results["table"]
    assert len(rows) == len(solver_record.results["configurations"])
    for row, conf in zip(rows, solver_record.results["configurations"], strict=True):
        assert row["Runtime [s]"] == conf["runtime_s"]
        assert row["Failure Rate"] == conf["failure_rate"]


def test_solver_time_budget_skip_is_logged(tmp_path: Path) -> None:
    overrides = _tiny_overrides(tmp_path, n_rays=2, max_seconds_per_config=1e-9)
    overrides["experiment.parameters.convergence.step_sizes"] = [0.2, 0.1]
    overrides["experiment.parameters.convergence.tolerances"] = [1e-5]
    rec = run_solver_benchmark(load_config(CONFIGS / "convergence.yaml", overrides))
    skipped = rec.results["skipped"]
    assert [s["setting"] for s in skipped] == [0.1]
    assert "max_seconds_per_config" in skipped[0]["reason"] and skipped[0]["predicted_seconds"] > 0
    assert "h = 0.1" in (rec.report_dir / "report.md").read_text(encoding="utf-8")


def test_table_rows_helper_handles_empty_input() -> None:
    assert table_rows([]) == []
