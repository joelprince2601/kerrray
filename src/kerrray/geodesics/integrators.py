"""Single-ray null-geodesic integration (PROJECT.md sections 9, 11 and 19).

Three schemes share the Hamiltonian right-hand side
(:func:`kerrray.geodesics.equations.geodesic_rhs`):

* ``"rk4"``: fixed-step classical Runge-Kutta (:mod:`kerrray.geodesics.tableaus`).
* ``"rk45"``: adaptive Dormand-Prince 5(4) with the Hairer-Norsett-Wanner
  step controller (same module).
* ``"dop853"``: ``scipy.integrate.solve_ivp(method="DOP853")`` (Hairer's
  eighth-order Dormand-Prince code) with terminal events for the termination
  conditions; the reference solver for convergence studies.

Termination follows :mod:`kerrray.photons.classification`. The affine
parameter starts at ``0`` and the last step is clipped so that ``lambda``
never exceeds ``lambda_max``. See docs/numerical_methods.md for the design
and its limitations (``max_steps`` is not enforced by the SciPy path, which
has no step cap; ``lambda_max`` is its only budget).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from kerrray.geodesics.equations import geodesic_rhs
from kerrray.geodesics.state import IDX_R, IDX_TH, STATE_SIZE
from kerrray.geodesics.tableaus import (
    FACTOR_MAX,
    SAFETY,
    error_norm,
    make_step_rk4,
    make_step_rk45,
    step_factor,
)
from kerrray.geometry import Spacetime, outer_horizon
from kerrray.photons.classification import (
    THETA_DOMAIN_TOLERANCE,
    TerminationState,
    classify_state,
)
from kerrray.photons.constants import (
    ConservationDiagnostics,
    angular_momentum,
    carter_constant,
    conservation_diagnostics,
    drift_floors,
    energy,
    null_constraint_from_rhs,
    relative_drift,
)
from kerrray.photons.trajectories import Trajectory

__all__ = [
    "EventOptions",
    "IntegratorOptions",
    "SUPPORTED_METHODS",
    "TerminationOptions",
    "integrate",
    "max_attempts",
]

SUPPORTED_METHODS = frozenset({"rk4", "rk45", "dop853"})


@dataclass(frozen=True)
class IntegratorOptions:
    """Integration scheme, tolerances and budgets (docs/architecture.md section 4).

    Attributes:
        method: ``"rk4"`` (fixed step), ``"rk45"`` (adaptive Dormand-Prince
            5(4)) or ``"dop853"`` (SciPy reference; scalar only).
        rtol, atol: Relative and absolute tolerances of the adaptive schemes.
        step_size: Fixed step in ``lambda`` (units of ``M``) for ``rk4``;
            the initial step for the adaptive schemes.
        max_steps: Cap on accepted steps (``MAX_AFFINE_PARAMETER`` when hit).
        lambda_max: Affine-parameter budget (units of ``M``).
        h_min: Adaptive step below which the ray is ``NUMERICAL_FAILURE``.
        h_max: Optional upper bound on the adaptive step (units of ``M``);
            ``None`` (default) leaves the step unbounded.
        dtype: ``"float32"`` or ``"float64"`` working precision (batched path).
    """

    method: str = "rk45"
    rtol: float = 1e-9
    atol: float = 1e-11
    step_size: float = 0.01
    max_steps: int = 100_000
    lambda_max: float = 1.0e4
    h_min: float = 1e-12
    dtype: str = "float64"
    h_max: float | None = None

    def __post_init__(self) -> None:
        if self.method not in SUPPORTED_METHODS:
            raise ValueError(f"method must be one of {sorted(SUPPORTED_METHODS)}, got {self.method!r}")
        for name in ("rtol", "atol", "step_size", "lambda_max", "h_min"):
            value = getattr(self, name)
            if not (math.isfinite(value) and value > 0.0):
                raise ValueError(f"{name} must be a finite number > 0, got {value!r}")
        if self.h_max is not None and not (math.isfinite(self.h_max) and self.h_max > 0.0):
            raise ValueError(f"h_max must be None or a finite number > 0, got {self.h_max!r}")
        if self.max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {self.max_steps!r}")
        if self.dtype not in ("float32", "float64"):
            raise ValueError(f"dtype must be 'float32' or 'float64', got {self.dtype!r}")

    @property
    def np_dtype(self) -> np.dtype:
        """The working NumPy dtype."""
        return np.dtype(self.dtype)


@dataclass(frozen=True)
class TerminationOptions:
    """Capture margin and escape radius (docs/architecture.md section 1)."""

    horizon_epsilon: float = 1e-6
    escape_radius: float = 1000.0

    def __post_init__(self) -> None:
        if not (math.isfinite(self.horizon_epsilon) and self.horizon_epsilon >= 0.0):
            raise ValueError(f"horizon_epsilon must be >= 0, got {self.horizon_epsilon!r}")
        if not (math.isfinite(self.escape_radius) and self.escape_radius > 0.0):
            raise ValueError(f"escape_radius must be > 0, got {self.escape_radius!r}")


@dataclass(frozen=True)
class EventOptions:
    """Optional equatorial disk-plane event for the batched integrator.

    When ``disk_plane`` is set, a ray whose ``theta`` crosses ``pi/2`` between
    two accepted steps is stopped as ``DISK_HIT`` at the crossing if the
    (linearly interpolated) radius satisfies ``r_in <= r <= r_out``;
    otherwise it continues.
    """

    disk_plane: bool = False
    r_in: float = 6.0
    r_out: float = 20.0

    def __post_init__(self) -> None:
        if not (0.0 < self.r_in <= self.r_out):
            raise ValueError(f"need 0 < r_in <= r_out, got r_in={self.r_in!r}, r_out={self.r_out!r}")


def max_attempts(integ: IntegratorOptions) -> int:
    """Upper bound on step attempts (accepted + rejected) before ``h < h_min``.

    Each rejection multiplies ``h`` by at most ``SAFETY`` and each accepted
    step by at most ``FACTOR_MAX``, while ``h`` never exceeds
    ``max(lambda_max, step_size)``, so with ``n_acc <= max_steps`` accepted
    steps the number of rejections is at most ``[ln(h_max / h_min) + n_acc
    ln(FACTOR_MAX)] / ln(1 / SAFETY)``. Used only as a defensive loop guard.
    """
    h_max = max(integ.lambda_max, integ.step_size)
    rej = (math.log(h_max / integ.h_min) + integ.max_steps * math.log(FACTOR_MAX)) / math.log(1.0 / SAFETY)
    return integ.max_steps + int(math.ceil(rej)) + 1


def _validate_y0(y0: ArrayLike) -> NDArray[np.float64]:
    arr = np.array(y0, dtype=np.float64)
    if arr.shape != (STATE_SIZE,):
        raise ValueError(f"y0 must have shape ({STATE_SIZE},), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("y0 must be finite")
    return arr


class _RunningMax:
    """Running maxima of the conservation diagnostics for one ray."""

    def __init__(self, st: Spacetime, y0: NDArray[np.float64]) -> None:
        self.st = st
        self.e0 = float(energy(y0))
        self.lz0 = float(angular_momentum(y0))
        self.q0 = float(carter_constant(st, y0))
        fe, fl, fq = drift_floors(st, self.e0)
        self.floors = (float(fe), float(fl), float(fq))
        self.max_e = self.max_lz = self.max_q = self.max_null = 0.0

    def update(self, y: NDArray[np.float64], f: NDArray[np.float64]) -> None:
        fe, fl, fq = self.floors
        self.max_e = max(self.max_e, float(relative_drift(energy(y), self.e0, floor=fe)))
        self.max_lz = max(self.max_lz, float(relative_drift(angular_momentum(y), self.lz0, floor=fl)))
        self.max_q = max(self.max_q, float(relative_drift(carter_constant(self.st, y), self.q0, floor=fq)))
        self.max_null = max(self.max_null, float(abs(null_constraint_from_rhs(y, f))) / fe**2)

    def finish(self, diag: ConservationDiagnostics) -> ConservationDiagnostics:
        diag.max_energy_drift = max(diag.max_energy_drift, self.max_e)
        diag.max_lz_drift = max(diag.max_lz_drift, self.max_lz)
        diag.max_carter_drift = max(diag.max_carter_drift, self.max_q)
        diag.max_null_error = max(diag.max_null_error, self.max_null)
        return diag


def _integrate_rk(
    st: Spacetime, y0: NDArray[np.float64], integ: IntegratorOptions, term: TerminationOptions, record: bool
) -> tuple[list[float], list[NDArray[np.float64]], TerminationState, int, int, _RunningMax]:
    """Scalar RK4 / RK45 loop; returns recorded points, state, step counts and diagnostics."""

    def rhs(y: NDArray[np.float64]) -> NDArray[np.float64]:
        return geodesic_rhs(st, y)

    adaptive = integ.method == "rk45"
    step = make_step_rk45(rhs) if adaptive else make_step_rk4(rhs)
    y = y0.copy()
    f = rhs(y)
    lam = 0.0
    h = integ.step_size
    if adaptive and integ.h_max is not None:
        h = min(h, integ.h_max)
    n_steps = n_rejected = attempts = 0
    rejected_before = False
    lams, ys = [0.0], [y.copy()]
    running = _RunningMax(st, y)
    running.update(y, f)
    state = classify_state(st, y, f, term, lam=lam, integ=integ, n_steps=0)
    guard = max_attempts(integ)
    while state == TerminationState.RUNNING:
        attempts += 1
        if attempts > guard:  # cannot happen by the bound in max_attempts; defensive
            state = TerminationState.NUMERICAL_FAILURE
            break
        remaining = integ.lambda_max - lam
        last = h >= remaining
        h_eff = remaining if last else h
        y_new, err, f_new = step(y, np.float64(h_eff), f)
        if not (np.all(np.isfinite(y_new)) and np.all(np.isfinite(f_new))):
            state = TerminationState.NUMERICAL_FAILURE
            break
        if adaptive:
            en = float(error_norm(err, y, y_new, integ.atol, integ.rtol))
            if not np.isfinite(en):
                state = TerminationState.NUMERICAL_FAILURE
                break
            if en > 1.0:
                n_rejected += 1
                h = h_eff * float(step_factor(en, rejected_before))
                rejected_before = True
                if h < integ.h_min:
                    state = TerminationState.NUMERICAL_FAILURE
                    break
                continue
            h_next = h if last else h_eff * float(step_factor(en, rejected_before))
            if integ.h_max is not None:
                h_next = min(h_next, integ.h_max)
            rejected_before = False
        else:
            h_next = h
        y, f = y_new, f_new
        lam = integ.lambda_max if last else lam + h_eff
        h = h_next
        n_steps += 1
        running.update(y, f)
        if record:
            lams.append(lam)
            ys.append(y.copy())
        state = classify_state(st, y, f, term, lam=lam, integ=integ, n_steps=n_steps, h=h)
    if not record:
        lams, ys = [0.0, lam], [ys[0], y.copy()]
    return lams, ys, state, n_steps, n_rejected, running


def _integrate_dop853(
    st: Spacetime, y0: NDArray[np.float64], integ: IntegratorOptions, term: TerminationOptions, record: bool
) -> tuple[list[float], list[NDArray[np.float64]], TerminationState, int, int, _RunningMax]:
    """SciPy DOP853 reference path with terminal events for the termination rules."""
    from scipy.integrate import solve_ivp

    r_cap = outer_horizon(st) + term.horizon_epsilon

    def fun(_lam: float, y: NDArray[np.float64]) -> NDArray[np.float64]:
        return geodesic_rhs(st, y)

    def captured(_lam: float, y: NDArray[np.float64]) -> float:
        return float(y[IDX_R] - r_cap)

    def escaped(_lam: float, y: NDArray[np.float64]) -> float:
        return float(y[IDX_R] - term.escape_radius)

    def theta_low(_lam: float, y: NDArray[np.float64]) -> float:
        return float(y[IDX_TH] + THETA_DOMAIN_TOLERANCE)

    def theta_high(_lam: float, y: NDArray[np.float64]) -> float:
        return float(np.pi + THETA_DOMAIN_TOLERANCE - y[IDX_TH])

    events: list[Any] = [captured, escaped, theta_low, theta_high]
    outcomes = [
        TerminationState.CAPTURED,
        TerminationState.ESCAPED,
        TerminationState.OUT_OF_DOMAIN,
        TerminationState.OUT_OF_DOMAIN,
    ]
    for ev, direction in zip(events, (-1, 1, -1, -1), strict=True):
        ev.terminal = True
        ev.direction = direction
    sol = solve_ivp(
        fun,
        (0.0, integ.lambda_max),
        y0,
        method="DOP853",
        rtol=integ.rtol,
        atol=integ.atol,
        first_step=min(integ.step_size, integ.lambda_max),
        events=events,
    )
    if sol.status == 1:
        hit = [k for k, t_ev in enumerate(sol.t_events) if len(t_ev) > 0]
        state = outcomes[hit[0]]
    elif sol.status == 0:
        state = TerminationState.MAX_AFFINE_PARAMETER
    else:
        state = TerminationState.NUMERICAL_FAILURE
    ys_all = sol.y.T
    running = _RunningMax(st, y0)
    for row in ys_all:
        running.update(row, geodesic_rhs(st, row))
    if record:
        lams, ys = [float(v) for v in sol.t], [row.copy() for row in ys_all]
    else:
        lams, ys = [float(sol.t[0]), float(sol.t[-1])], [ys_all[0].copy(), ys_all[-1].copy()]
    return lams, ys, state, int(len(sol.t) - 1), 0, running


def integrate(
    st: Spacetime,
    y0: ArrayLike,
    integ: IntegratorOptions,
    term: TerminationOptions,
    *,
    record: bool = True,
) -> Trajectory:
    """Integrate one photon state ``y0`` (8,) until a termination rule fires.

    Args:
        st: The spacetime.
        y0: Initial state ``[t, r, theta, phi, p_t, p_r, p_theta, p_phi]``.
        integ: Scheme, tolerances and budgets.
        term: Capture margin and escape radius.
        record: Keep every accepted step (default) or only the end points.
            The ``max_*`` diagnostics always cover every accepted step.

    Returns:
        A :class:`Trajectory`. For ``"dop853"`` the rejected-step count is
        not available from SciPy and is reported as ``0``.
    """
    y0_arr = _validate_y0(y0)
    t0 = time.perf_counter()
    if integ.method == "dop853":
        lams, ys, state, n_steps, n_rejected, running = _integrate_dop853(st, y0_arr, integ, term, record)
    else:
        lams, ys, state, n_steps, n_rejected, running = _integrate_rk(st, y0_arr, integ, term, record)
    runtime = time.perf_counter() - t0
    y_arr = np.vstack(ys)
    diag = running.finish(conservation_diagnostics(st, y_arr))
    return Trajectory(
        spacetime=st,
        lam=np.asarray(lams, dtype=np.float64),
        y=y_arr,
        state=state,
        n_steps=n_steps,
        n_rejected=n_rejected,
        runtime_s=runtime,
        diagnostics=diag,
    )
