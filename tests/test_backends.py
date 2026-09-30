"""Tests for the backend registry and chunked ray tracing (raytracing role)."""

from __future__ import annotations

import numpy as np
import pytest

from kerrray.geodesics import IntegratorOptions, TerminationOptions, integrate_batch
from kerrray.geometry import Spacetime
from kerrray.photons import BatchResult, TerminationState
from kerrray.raytracing.backends import (
    BACKEND_NAMES,
    CUDA_UNAVAILABLE_MESSAGE,
    Backend,
    BackendUnavailable,
    NumpyBackend,
    available_backends,
    get_backend,
)
from kerrray.raytracing.camera import Camera, initial_states
from kerrray.raytracing.rays import concatenate_results, trace_rays

INTEG = IntegratorOptions(method="rk45", rtol=1e-7, atol=1e-9)
TERM = TerminationOptions(horizon_epsilon=1e-6, escape_radius=1000.0)
RESULT_FIELDS = (
    "Y", "state", "n_steps", "lam", "max_null_error", "max_energy_drift",
    "max_lz_drift", "max_carter_drift", "n_rejected",
)


@pytest.fixture(scope="module")
def kerr() -> Spacetime:
    return Spacetime(mass=1.0, spin=0.9)


@pytest.fixture(scope="module")
def rays_8x8(kerr: Spacetime) -> np.ndarray:
    return initial_states(Camera(radius=1000.0, inclination_deg=60.0, fov=8.0, resolution=8), kerr)


@pytest.fixture(scope="module")
def unchunked(kerr: Spacetime, rays_8x8: np.ndarray) -> BatchResult:
    return trace_rays(kerr, rays_8x8, INTEG, TERM)


# --- registry ----------------------------------------------------------------------


def test_available_backends_reports_the_three_names() -> None:
    avail = available_backends()
    assert tuple(avail) == BACKEND_NAMES == ("numpy", "numba", "cuda")
    assert avail["numpy"] is True
    assert avail["cuda"] is False
    assert isinstance(avail["numba"], bool)


def test_cuda_backend_is_unavailable_with_the_d007_message() -> None:
    with pytest.raises(BackendUnavailable) as info:
        get_backend("cuda")
    assert str(info.value) == CUDA_UNAVAILABLE_MESSAGE
    assert "optional future work" in str(info.value) and "D-007" in str(info.value)


def test_numpy_backend_is_a_backend() -> None:
    backend = get_backend("numpy")
    assert isinstance(backend, NumpyBackend)
    assert isinstance(backend, Backend)
    assert backend.name == "numpy"


def test_numba_backend_is_consistent_with_available_backends() -> None:
    try:
        backend = get_backend("numba")
    except BackendUnavailable as exc:
        assert available_backends()["numba"] is False
        assert "numba" in str(exc)
    else:
        assert available_backends()["numba"] is True
        assert backend.name == "numba"


def test_unknown_backend_name_is_a_value_error() -> None:
    with pytest.raises(ValueError, match="unknown backend"):
        get_backend("opencl")


def test_numpy_backend_matches_integrate_batch(kerr: Spacetime, rays_8x8: np.ndarray) -> None:
    subset = rays_8x8[:6]
    direct = integrate_batch(kerr, subset, INTEG, TERM)
    via_backend = NumpyBackend().integrate_batch(kerr, subset, INTEG, TERM)
    for name in RESULT_FIELDS:
        assert np.array_equal(getattr(direct, name), getattr(via_backend, name)), name


# --- trace_rays ---------------------------------------------------------------------


def test_chunked_tracing_equals_unchunked_and_reports_progress(
    kerr: Spacetime, rays_8x8: np.ndarray, unchunked: BatchResult
) -> None:
    fractions: list[float] = []
    chunked = trace_rays(kerr, rays_8x8, INTEG, TERM, chunk_size=20, progress=fractions.append)
    assert fractions == [20 / 64, 40 / 64, 60 / 64, 1.0]
    for name in RESULT_FIELDS:
        assert np.array_equal(getattr(unchunked, name), getattr(chunked, name)), name
    assert chunked.event_Y is None and chunked.event_hit is None
    assert chunked.runtime_s > 0.0 and unchunked.runtime_s > 0.0


def test_trace_rays_accepts_a_backend_instance(
    kerr: Spacetime, rays_8x8: np.ndarray, unchunked: BatchResult
) -> None:
    result = trace_rays(kerr, rays_8x8, INTEG, TERM, backend=NumpyBackend(), chunk_size=64)
    assert np.array_equal(result.state, unchunked.state)
    assert np.array_equal(result.Y, unchunked.Y)


def test_state_counts_sum_to_the_number_of_rays(unchunked: BatchResult) -> None:
    total = sum(unchunked.count(state) for state in TerminationState)
    assert total == unchunked.n_rays == 64
    assert unchunked.count(TerminationState.CAPTURED) > 0
    assert unchunked.count(TerminationState.ESCAPED) > 0
    assert unchunked.count(TerminationState.NUMERICAL_FAILURE) == 0


def test_trace_rays_rejects_bad_input(kerr: Spacetime, rays_8x8: np.ndarray) -> None:
    with pytest.raises(ValueError, match="shape"):
        trace_rays(kerr, rays_8x8[:, :7], INTEG, TERM)
    with pytest.raises(ValueError, match="at least one ray"):
        trace_rays(kerr, rays_8x8[:0], INTEG, TERM)
    with pytest.raises(ValueError, match="chunk_size"):
        trace_rays(kerr, rays_8x8, INTEG, TERM, chunk_size=0)
    with pytest.raises(BackendUnavailable):
        trace_rays(kerr, rays_8x8, INTEG, TERM, backend="cuda")


def test_concatenate_results_single_part_and_optional_fields(unchunked: BatchResult) -> None:
    assert concatenate_results([unchunked]) is unchunked
    joined = concatenate_results([unchunked, unchunked])
    assert joined.n_rays == 2 * unchunked.n_rays
    assert joined.runtime_s == pytest.approx(2.0 * unchunked.runtime_s)
    assert np.array_equal(joined.n_rejected[: unchunked.n_rays], unchunked.n_rejected)
    without = BatchResult(
        Y=unchunked.Y, state=unchunked.state, n_steps=unchunked.n_steps, lam=unchunked.lam,
        max_null_error=unchunked.max_null_error, max_energy_drift=unchunked.max_energy_drift,
        max_lz_drift=unchunked.max_lz_drift, max_carter_drift=unchunked.max_carter_drift,
        runtime_s=0.5,
    )
    assert concatenate_results([unchunked, without]).n_rejected is None
    with pytest.raises(ValueError):
        concatenate_results([])
