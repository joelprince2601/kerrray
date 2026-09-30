"""Optional maximum adaptive step (IntegratorOptions.h_max).

The default (None) leaves the Dormand-Prince controller unbounded, as used in
research.md; a finite h_max caps every step, identically in the scalar,
NumPy batch and Numba integrators.
"""

from __future__ import annotations

import numpy as np
import pytest

from kerrray.geodesics import IntegratorOptions, TerminationOptions, integrate_batch
from kerrray.geodesics.integrators import integrate
from kerrray.geometry import kerr
from kerrray.photons import TerminationState
from kerrray.raytracing.backends.numba_backend import NumbaBackend, numba_available
from kerrray.raytracing.camera import Camera, initial_states

ST = kerr(1.0, 0.9)
TERM = TerminationOptions(escape_radius=1000.0)


def _rays(n: int = 60) -> np.ndarray:
    cam = Camera(radius=1000.0, inclination_deg=60.0, fov=8.0, resolution=16)
    return initial_states(cam, ST)[:n]


def test_h_max_validation() -> None:
    assert IntegratorOptions().h_max is None
    for bad in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            IntegratorOptions(h_max=bad)


def test_none_equals_huge_cap() -> None:
    """h_max = None and a cap no step ever reaches give bitwise-identical results."""
    Y0 = _rays()
    a = integrate_batch(ST, Y0, IntegratorOptions(rtol=1e-6, atol=1e-8), TERM)
    b = integrate_batch(ST, Y0, IntegratorOptions(rtol=1e-6, atol=1e-8, h_max=1e30), TERM)
    assert np.array_equal(a.state, b.state) and np.array_equal(a.Y, b.Y) and np.array_equal(a.n_steps, b.n_steps)


def test_cap_bounds_steps_and_costs_steps() -> None:
    """With h_max = 5 M a ray from 1000 M needs at least 2 * 1000 / 5 steps if it escapes."""
    Y0 = _rays()
    free = integrate_batch(ST, Y0, IntegratorOptions(rtol=1e-6, atol=1e-8, lambda_max=1e4), TERM)
    capped = integrate_batch(ST, Y0, IntegratorOptions(rtol=1e-6, atol=1e-8, lambda_max=1e4, h_max=5.0), TERM)
    esc = capped.state == int(TerminationState.ESCAPED)
    assert esc.any() and np.all(capped.n_steps[esc] >= 2 * 1000 // 5 - 5)
    assert np.all(capped.n_steps >= free.n_steps)
    assert np.array_equal(free.state, capped.state)


def test_scalar_matches_batch_with_cap() -> None:
    Y0 = _rays(8)
    integ = IntegratorOptions(rtol=1e-6, atol=1e-8, lambda_max=1e4, h_max=20.0)
    batch = integrate_batch(ST, Y0, integ, TERM)
    for i in range(len(Y0)):
        one = integrate(ST, Y0[i], integ, TERM)
        assert int(one.state) == int(batch.state[i])
        assert one.n_steps == batch.n_steps[i]
        assert np.max(np.abs(one.y_end - batch.Y[i]) / np.maximum(1.0, np.abs(batch.Y[i]))) < 1e-12


@pytest.mark.skipif(not numba_available(), reason="Numba not installed")
@pytest.mark.parametrize("h_max", [None, 20.0, 3.0])
def test_numba_matches_numpy_with_cap(h_max: float | None) -> None:
    Y0 = _rays()
    integ = IntegratorOptions(rtol=1e-6, atol=1e-8, lambda_max=1e4, h_max=h_max)
    a = NumbaBackend().integrate_batch(ST, Y0, integ, TERM)
    b = integrate_batch(ST, Y0, integ, TERM)
    assert np.array_equal(a.state, b.state)
    assert np.array_equal(a.n_steps, b.n_steps) and np.array_equal(a.n_rejected, b.n_rejected)
    assert np.max(np.abs(a.Y - b.Y) / np.maximum(1.0, np.abs(b.Y))) < 1e-12
