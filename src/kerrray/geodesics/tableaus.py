"""Runge-Kutta tableaus, step helpers and the adaptive step controller.

Fixed-step scheme: the classical fourth-order Runge-Kutta method (Kutta
1901; Hairer, Norsett and Wanner 1993, *Solving Ordinary Differential
Equations I*, Table 1.2 "the" Runge-Kutta method of order 4).

Adaptive scheme: the Dormand-Prince embedded pair RK5(4)7M (Dormand and
Prince 1980, J. Comput. Appl. Math. 6, 19; Hairer, Norsett and Wanner 1993,
Table 5.2 "Dormand-Prince 5(4)"). The seven-stage tableau has the
first-same-as-last (FSAL) property: the last stage of an accepted step is the
first stage of the next one, so a step costs six right-hand-side evaluations.
The fifth-order solution is propagated (local extrapolation) and the fourth-
order embedded solution provides the error estimate. The coefficients are
compared against ``scipy.integrate.RK45`` in ``tests/test_integrators.py``.

Error control follows Hairer, Norsett and Wanner 1993, section II.4, eqs.
(4.10)-(4.13): the scaled RMS error norm ``err = sqrt(mean((e_i / sc_i)^2))``
with ``sc_i = atol + rtol * max(|y0_i|, |y1_i|)``, the step is accepted when
``err <= 1``, and the next step is ``h * min(facmax, max(facmin, fac *
err^(-1/5)))`` with ``fac = 0.9``, ``facmin = 0.2``, ``facmax = 5``; after a
rejected step the factor is capped at 1 (no growth), as the same section
recommends.

The step functions are dtype-generic: with a float32 state and float32 step
sizes every operation stays in float32 (the tableau constants are Python
floats, which NumPy treats as weak scalars).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "DP_A",
    "DP_B",
    "DP_C",
    "DP_E",
    "ERROR_EXPONENT",
    "FACTOR_MAX",
    "FACTOR_MIN",
    "RK4_B",
    "RK4_C",
    "SAFETY",
    "StepFunction",
    "error_norm",
    "make_step_rk4",
    "make_step_rk45",
    "step_factor",
]

FloatArray = NDArray[np.floating]
StepFunction = Callable[[FloatArray, FloatArray, FloatArray | None], tuple[FloatArray, FloatArray | None, FloatArray]]
"""``step(y, h, f0) -> (y_new, error_estimate | None, f(y_new))``; ``h`` broadcasts over the leading axes of ``y``."""

# Classical RK4 (Hairer, Norsett and Wanner 1993, Table 1.2).
RK4_C = (0.0, 0.5, 0.5, 1.0)
RK4_B = (1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0)

# Dormand-Prince 5(4) (Dormand and Prince 1980; Hairer, Norsett and Wanner 1993, Table 5.2).
DP_C = (0.0, 1.0 / 5.0, 3.0 / 10.0, 4.0 / 5.0, 8.0 / 9.0, 1.0, 1.0)
DP_A = (
    (),
    (1.0 / 5.0,),
    (3.0 / 40.0, 9.0 / 40.0),
    (44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0),
    (19372.0 / 6561.0, -25360.0 / 2187.0, 64448.0 / 6561.0, -212.0 / 729.0),
    (9017.0 / 3168.0, -355.0 / 33.0, 46732.0 / 5247.0, 49.0 / 176.0, -5103.0 / 18656.0),
    (35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0),
)
DP_B = (35.0 / 384.0, 0.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0, 0.0)
"""Fifth-order weights (the propagated solution); equal to the last row of ``DP_A`` (FSAL)."""
DP_B_HAT = (
    5179.0 / 57600.0,
    0.0,
    7571.0 / 16695.0,
    393.0 / 640.0,
    -92097.0 / 339200.0,
    187.0 / 2100.0,
    1.0 / 40.0,
)
"""Fourth-order embedded weights."""
DP_E = tuple(bh - b for bh, b in zip(DP_B_HAT, DP_B, strict=True))
"""Error weights ``b_hat - b``: ``y_hat - y_new = h * sum_i E_i k_i`` (same sign convention as SciPy's RK45.E)."""

SAFETY = 0.9
FACTOR_MIN = 0.2
FACTOR_MAX = 5.0
ERROR_EXPONENT = -1.0 / 5.0
"""``-1/(q + 1)`` with ``q = 4`` the order of the embedded (error-estimating) method."""


def _hcol(h: FloatArray) -> FloatArray:
    """Step size(s) as a trailing-axis column so they broadcast over state components."""
    return np.asarray(h)[..., None]


def make_step_rk4(rhs: Callable[[FloatArray], FloatArray]) -> StepFunction:
    """Return a classical RK4 step ``step(y, h, f0) -> (y_new, None, f(y_new))``.

    ``f0`` is ``rhs(y)`` from the previous step (or ``None`` to evaluate it);
    the returned ``f(y_new)`` is the next step's ``f0``. Four evaluations per
    step (the fourth, at ``y_new``, is reused as the next first stage).
    """

    def step(y: FloatArray, h: FloatArray, f0: FloatArray | None = None) -> tuple[FloatArray, None, FloatArray]:
        hc = _hcol(h)
        k1 = rhs(y) if f0 is None else f0
        k2 = rhs(y + hc * (RK4_C[1] * k1))
        k3 = rhs(y + hc * (RK4_C[2] * k2))
        k4 = rhs(y + hc * (RK4_C[3] * k3))
        y_new = y + hc * (RK4_B[0] * k1 + RK4_B[1] * k2 + RK4_B[2] * k3 + RK4_B[3] * k4)
        return y_new, None, rhs(y_new)

    return step


def make_step_rk45(rhs: Callable[[FloatArray], FloatArray]) -> StepFunction:
    """Return a Dormand-Prince 5(4) step ``step(y, h, f0) -> (y_new, err, f(y_new))``.

    ``err = h * sum_i E_i k_i`` is the embedded error estimate (shape of
    ``y``). ``f(y_new)`` is the seventh stage (FSAL) and is the next step's
    ``f0`` when the step is accepted. Six evaluations per step.
    """
    a = DP_A
    e = DP_E

    def step(y: FloatArray, h: FloatArray, f0: FloatArray | None = None) -> tuple[FloatArray, FloatArray, FloatArray]:
        hc = _hcol(h)
        k1 = rhs(y) if f0 is None else f0
        k2 = rhs(y + hc * (a[1][0] * k1))
        k3 = rhs(y + hc * (a[2][0] * k1 + a[2][1] * k2))
        k4 = rhs(y + hc * (a[3][0] * k1 + a[3][1] * k2 + a[3][2] * k3))
        k5 = rhs(y + hc * (a[4][0] * k1 + a[4][1] * k2 + a[4][2] * k3 + a[4][3] * k4))
        k6 = rhs(y + hc * (a[5][0] * k1 + a[5][1] * k2 + a[5][2] * k3 + a[5][3] * k4 + a[5][4] * k5))
        y_new = y + hc * (a[6][0] * k1 + a[6][2] * k3 + a[6][3] * k4 + a[6][4] * k5 + a[6][5] * k6)
        k7 = rhs(y_new)
        err = hc * (e[0] * k1 + e[2] * k3 + e[3] * k4 + e[4] * k5 + e[5] * k6 + e[6] * k7)
        return y_new, err, k7

    return step


def error_norm(err: FloatArray, y: FloatArray, y_new: FloatArray, atol: float, rtol: float) -> FloatArray:
    """Scaled RMS error norm of Hairer, Norsett and Wanner 1993, eq. (4.11).

    ``sqrt(mean_i((err_i / (atol + rtol * max(|y_i|, |y_new_i|)))^2))`` over
    the state components (last axis); returns shape ``y.shape[:-1]``.
    """
    scale = atol + rtol * np.maximum(np.abs(y), np.abs(y_new))
    return np.sqrt(np.mean((err / scale) ** 2, axis=-1))


def step_factor(err: FloatArray, rejected_before: FloatArray | bool = False) -> FloatArray:
    """Step-size multiplier ``min(facmax, max(facmin, fac * err^(-1/5)))`` (HNW eq. 4.13).

    ``err = 0`` gives ``facmax``. Where ``rejected_before`` is true (the
    previous attempt of this step was rejected) the multiplier is capped at 1.
    """
    err_arr = np.asarray(err)
    with np.errstate(divide="ignore"):
        raw = SAFETY * np.power(err_arr, ERROR_EXPONENT)
    raw = np.where(err_arr == 0, FACTOR_MAX, raw)
    factor = np.minimum(FACTOR_MAX, np.maximum(FACTOR_MIN, raw))
    return np.where(rejected_before, np.minimum(factor, 1.0), factor).astype(err_arr.dtype, copy=False)
