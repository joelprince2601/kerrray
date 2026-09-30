"""Numba backend: kernel geometry and right-hand side against the geometry and geodesics
modules, agreement with the numpy reference backend (float64, float32, rk4, disk event,
termination budgets) and availability handling (docs/architecture.md section 9).

Without Numba the kernel source runs interpreted (``NumbaBackend(interpreted=True)``);
with Numba the compiled kernel is exercised. Either way the comparison is against
:func:`kerrray.geodesics.integrate_batch`, the reference implementation.
"""

from __future__ import annotations

import numpy as np
import pytest

from kerrray.geodesics import (
    EventOptions,
    IntegratorOptions,
    TerminationOptions,
    geodesic_rhs,
    integrate_batch,
    photon_from_constants,
)
from kerrray.geometry import inverse_metric_components, inverse_metric_derivatives, kerr, schwarzschild
from kerrray.photons import TerminationState
from kerrray.raytracing.backends import numba_kernel as K
from kerrray.raytracing.backends.base import BackendUnavailable, available_backends, get_backend
from kerrray.raytracing.backends.numba_backend import (
    NumbaBackend,
    _kernel_constants,
    integrate_batch_numba,
    numba_available,
    numba_threads,
)
from kerrray.raytracing.camera import Camera, initial_states

INTERPRETED = not numba_available()
ST = kerr(1.0, 0.9)
FAR = TerminationOptions(escape_radius=1e6)


def backend() -> NumbaBackend:
    return NumbaBackend(interpreted=INTERPRETED)


def relative_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    return np.max(np.abs(a64 - b64) / np.maximum(1.0, np.abs(b64)), axis=-1)


@pytest.mark.parametrize("spin", [0.0, 0.9, -0.7])
def test_inverse_metric_terms_match_geometry(spin: float) -> None:
    st = kerr(1.0, spin)
    rng = np.random.default_rng(7)
    geom, _, _, _ = _kernel_constants(st, IntegratorOptions(), TerminationOptions(), None, np.dtype(np.float64))
    out = np.empty(K.N_TERMS)
    for _ in range(200):
        r, th = rng.uniform(1.6, 40.0), rng.uniform(0.05, np.pi - 0.05)
        K.inverse_metric_terms(geom, r, th, out)
        g = inverse_metric_components(st, r, th)
        d_r, d_th = inverse_metric_derivatives(st, r, th)
        ref = np.array([*g, *d_r, *d_th])
        # Exact zeros (g^tphi and its derivatives at a = 0) must stay exact zeros, so
        # compare with assert_allclose (0 == 0 passes) instead of dividing by |ref|.
        # rtol 1e-12, not 1e-14: d_r g^tt = -(A_r Sigma Delta - A (Sigma Delta)_r) / (Sigma Delta)^2
        # subtracts two nearly equal products, which amplifies a one-ulp operand
        # difference between the scalar kernel (compiled by LLVM, or the interpreted
        # ``math`` module without Numba) and vectorised NumPy evaluation to ~1e-14 relative
        # (measured 1.7e-14 at a = -0.7); that is round-off, not a formula difference.
        np.testing.assert_allclose(out, ref, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_scalar_rhs_matches_geodesic_rhs(dtype: str) -> None:
    cam = Camera(radius=1000.0, inclination_deg=60.0, fov=12.0, resolution=8)
    Y = initial_states(cam, ST).astype(dtype)
    Y[:, 1] = np.linspace(1.7, 30.0, len(Y)).astype(dtype)  # spread the radii into the strong field
    geom, consts, _, _ = _kernel_constants(ST, IntegratorOptions(dtype=dtype), TerminationOptions(), None, Y.dtype)
    f = np.empty(8, dtype=Y.dtype)
    g64, g = np.empty(K.N_TERMS), np.empty(K.N_TERMS, dtype=Y.dtype)
    ref = geodesic_rhs(ST, Y)
    for k, y in enumerate(Y):
        K.geodesic_rhs_scalar(geom, consts, y, f, g64, g)
        assert f.dtype == Y.dtype
        assert np.allclose(f, ref[k], rtol=0.0, atol=0.0), (dtype, k, f, ref[k])
    # Exact L_z = 0 rays take the finite axis limit (no 0 * inf).
    y0 = Y[0].copy()
    y0[7] = 0
    K.geodesic_rhs_scalar(geom, consts, y0, f, g64, g)
    assert np.all(np.isfinite(f)) and np.array_equal(f, geodesic_rhs(ST, y0))


def test_agrees_with_numpy_on_camera_rays_float64() -> None:
    """Backward camera rays (a = 0.9, i = 60 deg): same termination states, states to ~1e-9.

    Compiled: the first 500 pixels of a 24 x 24 image. Interpreted (no Numba), where the
    kernel costs ~0.1 s per ray in plain Python: every 9th of those pixels (56 rays,
    10 captured and 46 escaped), which still crosses the shadow edge on several rows.
    """
    cam = Camera(radius=1000.0, inclination_deg=60.0, fov=12.0, resolution=24)
    Y0 = initial_states(cam, ST)[:500:9] if INTERPRETED else initial_states(cam, ST)[:500]
    min_captured, min_escaped = (5, 30) if INTERPRETED else (10, 100)
    integ = IntegratorOptions(method="rk45", rtol=1e-8, atol=1e-10)
    term = TerminationOptions()
    mine = backend().integrate_batch(ST, Y0, integ, term)
    ref = integrate_batch(ST, Y0, integ, term)
    assert mine.Y.dtype == np.float64 and mine.n_rejected is not None
    assert np.array_equal(mine.state, ref.state)
    assert {int(s) for s in mine.state} <= {int(TerminationState.CAPTURED), int(TerminationState.ESCAPED)}
    assert mine.count(TerminationState.CAPTURED) > min_captured and mine.count(TerminationState.ESCAPED) > min_escaped
    rel = relative_difference(mine.Y, ref.Y)
    assert np.max(rel) < 1e-9, np.max(rel)
    assert np.max(np.abs(mine.lam - ref.lam) / np.maximum(1.0, ref.lam)) < 1e-9
    assert np.array_equal(mine.n_steps, ref.n_steps) and np.array_equal(mine.n_rejected, ref.n_rejected)
    for name in ("max_null_error", "max_energy_drift", "max_lz_drift", "max_carter_drift"):
        a, b = getattr(mine, name), getattr(ref, name)
        assert np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300)) < 1e-9, name
    assert np.all(mine.max_energy_drift == 0.0) and np.all(mine.max_lz_drift == 0.0)
    assert mine.runtime_s > 0.0 and mine.event_Y is None and mine.event_hit is None


def test_float32_variant_matches_numpy_float32() -> None:
    cam = Camera(radius=1000.0, inclination_deg=60.0, fov=12.0, resolution=6)
    Y0 = initial_states(cam, ST)
    integ = IntegratorOptions(method="rk45", rtol=1e-6, atol=1e-8, dtype="float32")
    term = TerminationOptions(horizon_epsilon=1e-3)  # float32 cannot resolve 1e-6 next to r_plus
    mine = backend().integrate_batch(ST, Y0, integ, term)
    ref = integrate_batch(ST, Y0, integ, term)
    assert mine.Y.dtype == np.float32 and mine.lam.dtype == np.float64
    assert np.all(np.isfinite(mine.Y)) and np.array_equal(mine.state, ref.state)
    assert {int(s) for s in mine.state} <= {int(TerminationState.CAPTURED), int(TerminationState.ESCAPED)}
    assert np.max(relative_difference(mine.Y, ref.Y)) < 1e-3
    assert np.array_equal(mine.n_steps, ref.n_steps)
    assert np.allclose(mine.max_null_error, ref.max_null_error, rtol=1e-6)
    # Precision ordering: float32 is less accurate than float64 on the same rays and tolerance.
    r64 = integrate_batch_numba(ST, Y0, IntegratorOptions(method="rk45", rtol=1e-6, atol=1e-8), term, interpreted=INTERPRETED)
    assert np.mean(mine.max_carter_drift) > np.mean(r64.max_carter_drift)


def test_rk4_and_budgets_match_numpy() -> None:
    y = np.stack([photon_from_constants(ST, 15.0, 1.0, 1.0, lz, 4.0, sign_r=1, sign_theta=-1) for lz in (2.0, -3.0, 0.5)])
    integ4 = IntegratorOptions(method="rk4", step_size=0.05, lambda_max=20.0)
    a, b = backend().integrate_batch(ST, y, integ4, FAR), integrate_batch(ST, y, integ4, FAR)
    assert np.all(a.state == int(TerminationState.MAX_AFFINE_PARAMETER)) and np.all(a.n_steps == 400)
    assert np.max(relative_difference(a.Y, b.Y)) < 1e-12 and np.array_equal(a.lam, b.lam)
    cam = Camera(radius=1000.0, inclination_deg=60.0, fov=12.0, resolution=4)
    Y0 = initial_states(cam, ST)
    integ = IntegratorOptions(method="rk45", rtol=1e-8, atol=1e-10, max_steps=30)
    a, b = backend().integrate_batch(ST, Y0, integ, TerminationOptions()), integrate_batch(ST, Y0, integ, TerminationOptions())
    assert np.all(a.state == int(TerminationState.MAX_AFFINE_PARAMETER)) and np.array_equal(a.state, b.state)
    assert np.all(a.n_steps == 30) and np.max(relative_difference(a.Y, b.Y)) < 1e-12
    integ_lam = IntegratorOptions(method="rk45", rtol=1e-8, atol=1e-10, lambda_max=500.0)
    a, b = backend().integrate_batch(ST, Y0, integ_lam, TerminationOptions()), integrate_batch(ST, Y0, integ_lam, TerminationOptions())
    assert np.all(a.lam == 500.0) and np.array_equal(a.state, b.state) and np.max(relative_difference(a.Y, b.Y)) < 1e-9


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_disk_event_matches_numpy(dtype: str) -> None:
    y = np.stack([photon_from_constants(ST, 200.0, 1.2, 1.0, lz, 15.0, sign_r=-1, sign_theta=1) for lz in (-6.0, 3.0, 6.0, 9.0)])
    events = EventOptions(disk_plane=True, r_in=6.0, r_out=30.0)
    integ = IntegratorOptions(method="rk45", rtol=1e-7, atol=1e-9, lambda_max=400.0, dtype=dtype)
    a = backend().integrate_batch(ST, y, integ, FAR, events=events)
    b = integrate_batch(ST, y, integ, FAR, events=events)
    assert np.all(a.state == int(TerminationState.DISK_HIT)) and np.array_equal(a.state, b.state)
    assert a.event_hit is not None and np.array_equal(a.event_hit, b.event_hit) and np.all(a.event_hit)
    assert np.max(relative_difference(a.event_Y, b.event_Y)) < 1e-6
    assert np.allclose(a.event_Y[:, 2], 0.5 * np.pi, rtol=0.0, atol=1e-6)
    assert np.all((a.event_Y[:, 1] >= 6.0) & (a.event_Y[:, 1] <= 30.0))
    assert np.max(np.abs(a.lam - b.lam)) < 1e-6


def test_schwarzschild_limit_matches_numpy() -> None:
    st = schwarzschild()
    cam = Camera(radius=1000.0, inclination_deg=90.0, fov=8.0, resolution=6)
    Y0 = initial_states(cam, st)
    integ = IntegratorOptions(method="rk45", rtol=1e-9, atol=1e-11)
    a, b = NumbaBackend(interpreted=INTERPRETED).integrate_batch(st, Y0, integ, TerminationOptions()), integrate_batch(st, Y0, integ, TerminationOptions())
    assert np.array_equal(a.state, b.state) and np.max(relative_difference(a.Y, b.Y)) < 1e-9


def test_availability_and_registry() -> None:
    if numba_available():
        assert isinstance(get_backend("numba"), NumbaBackend)
        assert available_backends()["numba"] is True
        assert numba_threads() is not None and numba_threads() >= 1
    else:
        with pytest.raises(BackendUnavailable):
            NumbaBackend()
        with pytest.raises(BackendUnavailable):
            get_backend("numba")
        with pytest.raises(BackendUnavailable):
            integrate_batch_numba(ST, np.zeros((1, 8)) + 1.0, IntegratorOptions(), TerminationOptions())
        assert available_backends()["numba"] is False and numba_threads() is None
    assert available_backends()["cuda"] is False
    assert NumbaBackend(interpreted=True).name == "numba"


def test_input_validation_and_warm_up() -> None:
    engine = backend()
    with pytest.raises(ValueError):
        engine.integrate_batch(ST, np.ones((2, 7)), IntegratorOptions(), TerminationOptions())
    with pytest.raises(ValueError):
        engine.integrate_batch(ST, np.full((1, 8), np.nan), IntegratorOptions(), TerminationOptions())
    with pytest.raises(ValueError):
        engine.integrate_batch(ST, np.ones((1, 8)), IntegratorOptions(method="dop853"), TerminationOptions())
    seconds = engine.warm_up(ST, IntegratorOptions(), TerminationOptions())
    assert seconds > 0.0
