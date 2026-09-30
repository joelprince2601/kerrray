"""Vectorised integration of many rays at once (docs/architecture.md section 4).

The batch loop advances every *active* ray by one step attempt per iteration
with its own step size; there is no Python loop over rays. Each iteration:

1. gathers the active rays, clips their step to the affine budget and takes
   one Runge-Kutta step (same tableau helpers as the scalar path, with
   first-same-as-last reuse of the derivative);
2. for the adaptive scheme evaluates the Hairer-Norsett-Wanner error norm,
   accepts or rejects each ray individually and updates its step size;
   non-finite results and steps below ``h_min`` are ``NUMERICAL_FAILURE``;
3. scatters accepted states back, updates the running conservation
   diagnostics (``|H| / E_0^2`` from ``p_mu dx^mu/dlambda``, drifts of ``E``,
   ``L_z`` and ``Q``), classifies the accepted rays
   (:func:`kerrray.photons.classification.classify_batch`) and applies the
   optional disk-plane event.

Disk-plane event (approximation, PROJECT.md section 43). WHAT: a ``theta =
pi/2`` crossing between two accepted states is located by linear
interpolation in ``lambda`` and the ray stops as ``DISK_HIT`` when the
interpolated radius lies in ``[r_in, r_out]``. WHY: the accretion renderer
only needs the crossing state, and the linear interpolant is consistent with
the step's own accuracy for small steps. LIMITATION: the interpolated
state has an ``O(h^2)`` error in the crossing point and is not re-projected
onto the null cone; a ray exactly in the plane (``theta = pi/2`` at both
ends) never triggers the event.

Precision: ``integ.dtype`` sets the working dtype of the states, the
derivatives, the step sizes and the error control (float32 or float64). The
affine parameter and the diagnostics are accumulated in float64 (the
float32 spacing at ``lambda ~ 1e3`` is ``6e-5``, larger than typical steps).
The geometry is evaluated in float64 and cast (see
:mod:`kerrray.geodesics.equations`). float32 cannot resolve ``horizon_epsilon
= 1e-6`` relative to ``r_plus ~ 1`` beyond a few ulps and needs tolerances
above its round-off (``rtol >~ 1e-6``); docs/numerical_methods.md.
"""

from __future__ import annotations

from collections.abc import Callable

import time

import numpy as np
from numpy.typing import ArrayLike, NDArray

from kerrray.geodesics.equations import geodesic_rhs
from kerrray.geodesics.integrators import (
    EventOptions,
    IntegratorOptions,
    TerminationOptions,
    max_attempts,
)
from kerrray.geodesics.state import IDX_R, IDX_TH, STATE_SIZE
from kerrray.geodesics.tableaus import error_norm, make_step_rk4, make_step_rk45, step_factor
from kerrray.geometry import Spacetime
from kerrray.photons.classification import TerminationState, classify_batch
from kerrray.photons.constants import (
    angular_momentum,
    carter_constant,
    drift_floors,
    energy,
    null_constraint_from_rhs,
)
from kerrray.photons.trajectories import BatchResult

__all__ = ["integrate_batch"]


class _BatchDiagnostics:
    """Running per-ray maxima of the conservation diagnostics (float64)."""

    def __init__(self, st: Spacetime, Y0: NDArray[np.float64], F0: NDArray[np.floating]) -> None:
        self.st = st
        self.e0 = energy(Y0)
        self.lz0 = angular_momentum(Y0)
        self.q0 = carter_constant(st, Y0)
        self.fe, self.fl, self.fq = drift_floors(st, self.e0)
        n = Y0.shape[0]
        self.max_e = np.zeros(n)
        self.max_lz = np.zeros(n)
        self.max_q = np.zeros(n)
        self.max_null = np.abs(null_constraint_from_rhs(Y0, F0.astype(np.float64))) / self.fe**2

    def update(self, idx: NDArray[np.intp], y: NDArray[np.floating], f: NDArray[np.floating]) -> None:
        """Fold the accepted states ``y`` (rows ``idx``) into the running maxima."""
        y64 = y.astype(np.float64, copy=False)
        f64 = f.astype(np.float64, copy=False)
        self.max_e[idx] = np.maximum(self.max_e[idx], _drift(energy(y64), self.e0[idx], self.fe[idx]))
        self.max_lz[idx] = np.maximum(self.max_lz[idx], _drift(angular_momentum(y64), self.lz0[idx], self.fl[idx]))
        self.max_q[idx] = np.maximum(self.max_q[idx], _drift(carter_constant(self.st, y64), self.q0[idx], self.fq[idx]))
        self.max_null[idx] = np.maximum(self.max_null[idx], np.abs(null_constraint_from_rhs(y64, f64)) / self.fe[idx] ** 2)


def _drift(values: NDArray[np.float64], reference: NDArray[np.float64], floor: NDArray[np.float64]) -> NDArray[np.float64]:
    """Elementwise :func:`kerrray.photons.constants.relative_drift` with per-ray floors."""
    return np.abs(values - reference) / np.maximum(np.abs(reference), floor)


def _validate_batch(Y0: ArrayLike) -> NDArray[np.float64]:
    arr = np.array(Y0, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != STATE_SIZE:
        raise ValueError(f"Y0 must have shape (N, {STATE_SIZE}), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Y0 must be finite")
    return arr


def integrate_batch(
    st: Spacetime,
    Y0: ArrayLike,
    integ: IntegratorOptions,
    term: TerminationOptions,
    *,
    events: EventOptions | None = None,
    on_step: Callable[[NDArray[np.int64], NDArray[np.floating], NDArray[np.int64]], None] | None = None,
) -> BatchResult:
    """Integrate ``N`` photon states ``Y0`` (N, 8) with ``"rk4"`` or ``"rk45"``.

    ``"dop853"`` is the scalar reference solver and raises ``ValueError``
    here. See the module docstring for the algorithm, the event and the
    precision handling.

    ``on_step``, when given, is called after every sweep of accepted steps
    as ``on_step(indices, states, codes)``: the ray indices that advanced,
    their new states (a copy, shape ``(k, 8)``) and their termination codes
    (``RUNNING`` while still active). It observes only; it cannot change the
    integration, though it may raise to abort it. It is used to record paths
    for animation and to report live progress.
    """
    if integ.method == "dop853":
        raise ValueError("method 'dop853' is available in integrate() only; integrate_batch supports 'rk4' and 'rk45'")
    y0_64 = _validate_batch(Y0)
    n_rays = y0_64.shape[0]
    dtype = integ.np_dtype
    adaptive = integ.method == "rk45"
    disk = events is not None and events.disk_plane
    t0 = time.perf_counter()

    def rhs(y: NDArray[np.floating]) -> NDArray[np.floating]:
        return geodesic_rhs(st, y)

    step = make_step_rk45(rhs) if adaptive else make_step_rk4(rhs)
    Y = y0_64.astype(dtype)
    F = rhs(Y)
    lam = np.zeros(n_rays, dtype=np.float64)
    h = np.full(n_rays, integ.step_size, dtype=dtype)
    if adaptive and integ.h_max is not None:
        h = np.minimum(h, np.asarray(integ.h_max, dtype=dtype))
    n_steps = np.zeros(n_rays, dtype=np.int64)
    n_rejected = np.zeros(n_rays, dtype=np.int64)
    rejected_before = np.zeros(n_rays, dtype=bool)
    diag = _BatchDiagnostics(st, y0_64, F)
    state = classify_batch(st, Y, F, term, lam=lam, integ=integ, n_steps=n_steps)
    event_Y = np.zeros((n_rays, STATE_SIZE), dtype=dtype) if disk else None
    event_hit = np.zeros(n_rays, dtype=bool) if disk else None
    half_pi = 0.5 * np.pi
    guard = max_attempts(integ)
    attempts = 0
    active = state == TerminationState.RUNNING
    while np.any(active):
        attempts += 1
        if attempts > guard:  # cannot happen by the bound in max_attempts; defensive
            state[active] = TerminationState.NUMERICAL_FAILURE
            break
        idx = np.flatnonzero(active)
        y = Y[idx]
        lam_i = lam[idx]
        h_i = h[idx]
        remaining = integ.lambda_max - lam_i
        last = h_i.astype(np.float64) >= remaining
        h_eff = np.where(last, remaining, h_i).astype(dtype)
        y_new, err, f_new = step(y, h_eff, F[idx])
        finite = np.isfinite(y_new).all(axis=1) & np.isfinite(f_new).all(axis=1)
        if adaptive:
            with np.errstate(invalid="ignore"):
                en = error_norm(err, y, y_new, integ.atol, integ.rtol)
                finite &= np.isfinite(en)
                accept = finite & (en <= 1.0)
                factor = step_factor(np.where(finite, en, 1.0), rejected_before[idx])
            h_new = np.where(accept & last, h_i, h_eff * factor)
            if integ.h_max is not None:
                h_new = np.minimum(h_new, np.asarray(integ.h_max, dtype=h_new.dtype))
        else:
            accept = finite
            h_new = h_i
        rejected = finite & ~accept
        state[idx[~finite]] = TerminationState.NUMERICAL_FAILURE
        n_rejected[idx[rejected]] += 1
        rejected_before[idx] = rejected
        h[idx] = h_new
        state[idx[rejected & (h_new < integ.h_min)]] = TerminationState.NUMERICAL_FAILURE
        acc = idx[accept]
        if acc.size:
            ya = y_new[accept]
            fa = f_new[accept]
            Y[acc] = ya
            F[acc] = fa
            lam_old = lam_i[accept]
            h_acc = h_eff[accept].astype(np.float64)
            lam[acc] = np.where(last[accept], integ.lambda_max, lam_old + h_acc)
            n_steps[acc] += 1
            diag.update(acc, ya, fa)
            codes = classify_batch(st, ya, fa, term, lam=lam[acc], integ=integ, n_steps=n_steps[acc], h=h[acc])
            if disk:
                y_old = y[accept]
                th_old = y_old[:, IDX_TH].astype(np.float64)
                th_new = ya[:, IDX_TH].astype(np.float64)
                crossed = (th_old - half_pi) * (th_new - half_pi) < 0.0
                if np.any(crossed):
                    with np.errstate(divide="ignore", invalid="ignore"):
                        s = np.where(crossed, (half_pi - th_old) / (th_new - th_old), 0.0)
                    y_cross = y_old + (s[:, None] * (ya - y_old)).astype(dtype)
                    y_cross[:, IDX_TH] = half_pi
                    r_cross = y_cross[:, IDX_R]
                    hit = crossed & (r_cross >= events.r_in) & (r_cross <= events.r_out)
                    if np.any(hit):
                        hit_idx = acc[hit]
                        event_Y[hit_idx] = y_cross[hit]
                        event_hit[hit_idx] = True
                        Y[hit_idx] = y_cross[hit]
                        lam[hit_idx] = lam_old[hit] + s[hit] * h_acc[hit]
                        codes[hit] = TerminationState.DISK_HIT
            state[acc] = codes
            if on_step is not None:
                on_step(acc.copy(), Y[acc].copy(), np.asarray(codes, dtype=np.int64).copy())
        active = state == TerminationState.RUNNING
    return BatchResult(
        Y=Y,
        state=state,
        n_steps=n_steps,
        lam=lam,
        max_null_error=diag.max_null,
        max_energy_drift=diag.max_e,
        max_lz_drift=diag.max_lz,
        max_carter_drift=diag.max_q,
        runtime_s=time.perf_counter() - t0,
        event_Y=event_Y,
        event_hit=event_hit,
        n_rejected=n_rejected,
    )
