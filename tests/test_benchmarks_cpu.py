"""Tests for the CPU benchmark (kerrray.benchmarks.cpu, EXP-009) and memory accounting (benchmarks.memory)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kerrray.benchmarks.cpu import (
    BenchmarkParams,
    image_plane_states,
    run_cpu_benchmark,
    sample_rays,
    trajectory_error,
)
from kerrray.benchmarks.memory import array_bytes, format_bytes, peak_memory
from kerrray.geodesics import hamiltonian
from kerrray.geometry import Spacetime
from kerrray.photons import BatchResult, TerminationState
from kerrray.raytracing.camera import Camera, initial_states
from kerrray.utils.config import load_config
from kerrray.utils.config_parsing import ConfigError

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_image_plane_states_match_the_camera_on_its_pixel_grid() -> None:
    st = Spacetime(mass=1.0, spin=0.9)
    cam = Camera(radius=100.0, inclination_deg=60.0, fov=8.0, resolution=6)
    alpha, beta = cam.pixel_coordinates()
    ours = image_plane_states(cam, st, alpha, beta)
    ref = initial_states(cam, st)
    np.testing.assert_allclose(ours, ref, rtol=1e-12, atol=1e-12)
    assert np.max(np.abs(hamiltonian(st, ours))) < 1e-12


def test_sample_rays_are_seeded_and_null() -> None:
    st = Spacetime(mass=1.0, spin=0.5)
    cam = Camera(radius=200.0, inclination_deg=45.0, fov=10.0, resolution=4)
    a = sample_rays(cam, st, 50, np.random.default_rng(7))
    b = sample_rays(cam, st, 50, np.random.default_rng(7))
    np.testing.assert_array_equal(a, b)
    assert a.shape == (50, 8)
    np.testing.assert_array_equal(sample_rays(cam, st, 20, np.random.default_rng(7)), a[:20])  # nested sets
    assert np.max(np.abs(hamiltonian(st, a))) < 1e-12
    with pytest.raises(ValueError):
        sample_rays(cam, st, 0, np.random.default_rng(0))


def _batch(Y: np.ndarray, states: list[int]) -> BatchResult:
    n = Y.shape[0]
    zeros = np.zeros(n)
    return BatchResult(Y=Y, state=np.array(states), n_steps=np.ones(n, dtype=np.int64), lam=zeros.copy(),
                       max_null_error=zeros.copy(), max_energy_drift=zeros.copy(), max_lz_drift=zeros.copy(),
                       max_carter_drift=zeros.copy(), runtime_s=0.0)


def test_trajectory_error_compares_only_rays_at_the_fixed_lambda() -> None:
    lam_state = int(TerminationState.MAX_AFFINE_PARAMETER)
    ref = _batch(np.full((3, 8), 10.0), [lam_state, lam_state, int(TerminationState.CAPTURED)])
    Y = np.full((3, 8), 10.0)
    Y[0, 1] += 1e-3
    run = _batch(Y, [lam_state, lam_state, lam_state])
    err = trajectory_error(run, ref)
    assert err["n_compared"] == 2 and err["n_excluded"] == 1
    assert err["max"] == pytest.approx(1e-4)
    assert trajectory_error(_batch(Y, [int(TerminationState.CAPTURED)] * 3), ref)["max"] is None


def test_memory_helpers() -> None:
    assert array_bytes(10, np.float64, 2, per_ray_scalars=0) == 10 * 8 * 8 * 2
    assert array_bytes(10, np.float32, 1, per_ray_scalars=4) == 10 * 8 * 4 + 40
    result, peak = peak_memory(lambda: np.ones(250_000))
    assert result.nbytes == 2_000_000
    assert peak >= 2_000_000
    assert format_bytes(512) == "512 B" and format_bytes(2_500_000) == "2.5 MB"


def test_benchmark_params_validation() -> None:
    with pytest.raises(ConfigError):
        BenchmarkParams(ray_counts=[0])
    with pytest.raises(ConfigError):
        BenchmarkParams(backends=["gpu"])
    with pytest.raises(ConfigError):
        BenchmarkParams(max_seconds_per_run=0.0)


@pytest.fixture(scope="module")
def cpu_record(tmp_path_factory: pytest.TempPathFactory):
    tmp = tmp_path_factory.mktemp("cpu")
    cfg = load_config(CONFIGS / "benchmark.yaml", {
        "experiment.output_dir": str(tmp / "runs"),
        "experiment.report_dir": str(tmp / "reports"),
        "observer.radius": 30.0,
        "termination.escape_radius": 30.0,
        "termination.horizon_epsilon": 1e-3,
        "integration.rtol": 1e-6,
        "integration.atol": 1e-8,
        "experiment.parameters.benchmark": {
            "ray_counts": [16, 64, 10_000_000], "backends": ["numpy", "numba"], "repeats": 1,
            "reference_rays": 4, "reference_rtol": 1e-9, "max_seconds_per_run": 30.0,
        },
    })
    return run_cpu_benchmark(cfg)


def test_cpu_benchmark_measures_or_logs_every_requested_run(cpu_record) -> None:
    res = cpu_record.results
    runs = res["runs"]
    assert {(r["backend"], r["rays"]) for r in runs} == {
        (b, n) for b in ("numpy", "numba") for n in (16, 64, 10_000_000)}
    for row in runs:
        if row["status"] == "skipped":
            assert row["reason"]  # never silent
            assert row in res["skipped"]
        else:
            assert row["status"] == "ok"
            assert row["wall_time_s"] > 0 and row["rays_per_s"] == pytest.approx(row["rays"] / row["wall_time_s"])
            assert row["peak_traced_MB"] > 0
            assert row["captured"] + row["escaped"] + row["other"] == row["rays"]
            assert row["trajectory_rays_compared"] >= 1
    numpy_rows = {r["rays"]: r for r in runs if r["backend"] == "numpy"}
    assert numpy_rows[16]["status"] == "ok" and numpy_rows[64]["status"] == "ok"
    # 10^7 rays cannot fit the 30 s budget on any CPU backend: it is skipped with its predicted cost
    assert numpy_rows[10_000_000]["status"] == "skipped"
    assert numpy_rows[10_000_000]["predicted_seconds"] > 30.0
    if "numba" in res["backends_unavailable"]:
        assert all(r["status"] == "skipped" for r in runs if r["backend"] == "numba")
    assert res["runtime_exponents"]["numpy"] is not None
    assert res["hardware"]["cpu_count"] >= 1 and res["git_commit"]


def test_cpu_benchmark_report_and_figures(cpu_record) -> None:
    report = (cpu_record.report_dir / "report.md").read_text(encoding="utf-8")
    assert "Skipped runs" in report and "10000000" in report
    assert (cpu_record.report_dir / "benchmark_table.md").is_file()
    assert (cpu_record.report_dir / "runtime_vs_rays_numpy.png").is_file()
