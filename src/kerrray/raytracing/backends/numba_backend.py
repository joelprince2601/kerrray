"""CPU-parallel Numba backend: one ray per ``prange`` iteration (docs/architecture.md section 9).

:class:`NumbaBackend` runs the scalar kernels of
:mod:`kerrray.raytracing.backends.numba_kernel` under
``numba.njit(parallel=True, fastmath=False, cache=True)``: the same
Dormand-Prince 5(4) tableau (imported from :mod:`kerrray.geodesics.tableaus`),
the same Hairer-Norsett-Wanner error norm and controller limits, the same
right-hand side and the same termination rules as the NumPy reference
:func:`kerrray.geodesics.integrate_batch`, so the two backends agree to
floating-point round-off (``tests/test_numba_backend.py``; docs/performance.md).
The result is the same :class:`~kerrray.photons.BatchResult` (final states,
termination codes, step counts, affine parameter, running maxima of the null
error and of the ``E``, ``L_z`` and Carter drifts, rejected-step counts and,
when the disk-plane event is enabled, the interpolated crossing states).

Precision: ``integ.dtype`` selects float32 or float64 exactly as in the
reference (Numba specialises the kernel on the array dtype; every constant
is passed in the working dtype). float32 cannot resolve the default
``horizon_epsilon = 1e-6`` next to ``r_plus ~ 1``; use ``>= 1e-3``
(docs/numerical_methods.md section 7).

Numba is optional. The module imports cleanly without it; the constructor
then raises :class:`~kerrray.raytracing.backends.base.BackendUnavailable`
unless ``interpreted=True`` is requested explicitly, which runs the kernel
source in plain Python (orders of magnitude slower) for verification only.
"""

from __future__ import annotations

import time
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

from kerrray.geodesics import EventOptions, IntegratorOptions, TerminationOptions
from kerrray.geodesics.integrators import max_attempts
from kerrray.geodesics.state import STATE_SIZE
from kerrray.geodesics.tableaus import (
    DP_A,
    DP_E,
    ERROR_EXPONENT,
    FACTOR_MAX,
    FACTOR_MIN,
    RK4_B,
    RK4_C,
    SAFETY,
)
from kerrray.geometry import Spacetime, outer_horizon
from kerrray.photons import BatchResult
from kerrray.photons.classification import AXIS_SIN_TOLERANCE, THETA_DOMAIN_TOLERANCE
from kerrray.photons.constants import angular_momentum, carter_constant, drift_floors, energy
from kerrray.raytracing.backends import numba_kernel as kernel
from kerrray.raytracing.backends.base import BackendUnavailable

__all__ = ["NumbaBackend", "integrate_batch_numba", "numba_available", "numba_threads"]

NUMBA_MISSING_MESSAGE: Final[str] = (
    "numba is not importable in this environment; install numba to use the 'numba' backend "
    "(the numpy backend is the reference and is always available)"
)


def numba_available() -> bool:
    """``True`` when Numba could be imported (the compiled kernel can run)."""
    return kernel.NUMBA_AVAILABLE


def numba_threads() -> int | None:
    """Number of threads Numba's parallel layer uses, or ``None`` without Numba."""
    if not kernel.NUMBA_AVAILABLE:
        return None
    import numba

    return int(numba.get_num_threads())


def _validate_batch(Y0: ArrayLike) -> NDArray[np.float64]:
    arr = np.array(Y0, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != STATE_SIZE:
        raise ValueError(f"Y0 must have shape (N, {STATE_SIZE}), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Y0 must be finite")
    return arr


def _kernel_constants(
    st: Spacetime, integ: IntegratorOptions, term: TerminationOptions, events: EventOptions | None, dtype: np.dtype
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Geometry constants (float64), working-dtype constants, float64 and integer parameters.

    The Python-level products are evaluated in the same order as in
    :mod:`kerrray.geometry.metric` (``-2.0 * m * a`` is ``(-2.0 * m) * a``, ...)
    so that the kernel reproduces its floating-point results exactly.
    """
    m, a = st.mass, st.a
    geom = np.array([m, a, a**2, -2.0 * m * a, 2.0 * m * a, 2.0 * m, 2.0 * a**2], dtype=np.float64)
    r_in = events.r_in if events is not None else 0.0
    r_out = events.r_out if events is not None else 0.0
    consts = np.array(
        [
            0.0, 1.0, 2.0, -0.5, 8.0, SAFETY, FACTOR_MIN, FACTOR_MAX, ERROR_EXPONENT,
            integ.rtol, integ.atol, integ.h_min, outer_horizon(st) + term.horizon_epsilon,
            term.escape_radius, -THETA_DOMAIN_TOLERANCE, np.pi + THETA_DOMAIN_TOLERANCE,
            AXIS_SIN_TOLERANCE, integ.step_size, r_in, r_out, 0.5 * np.pi,
        ],
        dtype=dtype,
    )
    h_max = np.inf if integ.h_max is None else integ.h_max
    params = np.array([integ.lambda_max, 0.5 * np.pi, h_max], dtype=np.float64)
    disk = events is not None and events.disk_plane
    ipar = np.array(
        [integ.max_steps, max_attempts(integ), int(integ.method == "rk45"), int(disk)], dtype=np.int64
    )
    return geom, consts, params, ipar


def _tableaus(dtype: np.dtype) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Dense Dormand-Prince ``A`` (7 x 7) and ``E`` (7,), RK4 ``B`` and ``C``, in the working dtype."""
    tab_a = np.zeros((7, 7), dtype=np.float64)
    for i, row in enumerate(DP_A):
        tab_a[i, : len(row)] = row
    return (
        tab_a.astype(dtype),
        np.asarray(DP_E, dtype=dtype),
        np.asarray(RK4_B, dtype=dtype),
        np.asarray(RK4_C, dtype=dtype),
    )


def integrate_batch_numba(
    st: Spacetime,
    Y0: ArrayLike,
    integ: IntegratorOptions,
    term: TerminationOptions,
    *,
    events: EventOptions | None = None,
    interpreted: bool = False,
) -> BatchResult:
    """Integrate ``N`` photon states ``Y0`` (N, 8) with the scalar per-ray kernel.

    Args:
        st: The spacetime.
        Y0: Initial states (finite, shape ``(N, 8)``).
        integ: ``"rk4"`` or ``"rk45"`` with tolerances, budgets and ``dtype``
            (``"dop853"`` raises ``ValueError`` as in the reference).
        term: Capture margin and escape radius.
        events: Optional disk-plane event (same linear interpolant as the reference).
        interpreted: Run the kernel source in plain Python instead of the
            compiled kernel (verification only; also the only mode that works
            without Numba).

    Returns:
        A :class:`~kerrray.photons.BatchResult` with the same fields as
        :func:`kerrray.geodesics.integrate_batch`.
    """
    if integ.method == "dop853":
        raise ValueError("method 'dop853' is available in integrate() only; the numba backend supports 'rk4' and 'rk45'")
    if not interpreted and not kernel.NUMBA_AVAILABLE:
        raise BackendUnavailable(NUMBA_MISSING_MESSAGE)
    t0 = time.perf_counter()
    y0_64 = _validate_batch(Y0)
    n_rays = y0_64.shape[0]
    dtype = integ.np_dtype
    geom, consts, params, ipar = _kernel_constants(st, integ, term, events, dtype)
    tab_a, tab_e, rk4_b, rk4_c = _tableaus(dtype)
    Y = np.ascontiguousarray(y0_64.astype(dtype))
    F = np.empty_like(Y)
    lam = np.zeros(n_rays, dtype=np.float64)
    n_steps = np.zeros(n_rays, dtype=np.int64)
    n_rejected = np.zeros(n_rays, dtype=np.int64)
    state = np.zeros(n_rays, dtype=np.int64)
    maxima = [np.zeros(n_rays, dtype=np.float64) for _ in range(4)]
    e0 = np.ascontiguousarray(energy(y0_64), dtype=np.float64)
    lz0 = np.ascontiguousarray(angular_momentum(y0_64), dtype=np.float64)
    q0 = np.ascontiguousarray(carter_constant(st, y0_64), dtype=np.float64)
    fe, fl, fq = (np.ascontiguousarray(v, dtype=np.float64) for v in drift_floors(st, e0))
    fe2 = fe**2
    disk = bool(ipar[kernel.I_DISK])
    event_Y = np.zeros((n_rays if disk else 0, STATE_SIZE), dtype=dtype)
    event_hit = np.zeros(n_rays if disk else 0, dtype=np.bool_)
    run = kernel.integrate_rays_kernel
    if interpreted:
        run = getattr(run, "py_func", run)
    args = (
        geom, consts, params, ipar, tab_a, tab_e, rk4_b, rk4_c, Y, F, lam, n_steps, n_rejected, state,
        *maxima, e0, lz0, q0, fe, fl, fq, fe2, event_Y, event_hit,
    )
    if interpreted:
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            run(*args)
    else:
        run(*args)
    return BatchResult(
        Y=Y,
        state=state,
        n_steps=n_steps,
        lam=lam,
        max_null_error=maxima[0],
        max_energy_drift=maxima[1],
        max_lz_drift=maxima[2],
        max_carter_drift=maxima[3],
        runtime_s=time.perf_counter() - t0,
        event_Y=event_Y if disk else None,
        event_hit=event_hit if disk else None,
        n_rejected=n_rejected,
    )


class NumbaBackend:
    """The Numba backend (``name == "numba"``) behind the ``Backend`` protocol.

    Args:
        interpreted: Run the kernel source in plain Python (verification
            mode; works without Numba). The default requires Numba and
            raises :class:`BackendUnavailable` otherwise, which
            :func:`kerrray.raytracing.backends.base.get_backend` reports as
            the backend being unavailable.
    """

    name: Final[str] = "numba"

    def __init__(self, *, interpreted: bool = False) -> None:
        if not interpreted and not kernel.NUMBA_AVAILABLE:
            raise BackendUnavailable(NUMBA_MISSING_MESSAGE)
        self.interpreted = interpreted

    def integrate_batch(
        self,
        st: Spacetime,
        Y0: ArrayLike,
        integ: IntegratorOptions,
        term: TerminationOptions,
        events: EventOptions | None = None,
    ) -> BatchResult:
        """Integrate the states ``Y0`` (N, 8); see :func:`integrate_batch_numba`."""
        return integrate_batch_numba(st, Y0, integ, term, events=events, interpreted=self.interpreted)

    def warm_up(self, st: Spacetime, integ: IntegratorOptions, term: TerminationOptions) -> float:
        """Trigger compilation for the dtype and scheme of ``integ`` and return the seconds it took.

        Integrates one short ray (``lambda_max = step_size``) so that the
        time is dominated by compilation (or by the cache load); report it
        separately from timed runs. Interpreted mode returns the (tiny) run time.
        """
        y0 = np.array([[0.0, 50.0, 0.5 * np.pi, 0.0, -1.0, -1.0, 0.0, 0.0]], dtype=np.float64)
        short = IntegratorOptions(
            method=integ.method, rtol=integ.rtol, atol=integ.atol, step_size=integ.step_size,
            max_steps=integ.max_steps, lambda_max=integ.step_size, h_min=integ.h_min, dtype=integ.dtype,
        )
        t0 = time.perf_counter()
        self.integrate_batch(st, y0, short, term)
        return time.perf_counter() - t0

    def __repr__(self) -> str:
        return f"NumbaBackend(interpreted={self.interpreted})"
