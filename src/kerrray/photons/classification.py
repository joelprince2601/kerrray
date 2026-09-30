"""Photon termination states and the classification rules (PROJECT.md section 11).

The rules are those of docs/architecture.md section 1, applied after every
accepted integration step (and to the initial state), in this priority order
when several hold at once:

1. ``NUMERICAL_FAILURE``: any non-finite state or right-hand-side component,
   or an adaptive step size below ``h_min`` (passed by the integrator).
2. ``OUT_OF_DOMAIN``: ``r < 0``; ``theta`` outside ``[0, pi]`` by more than
   :data:`THETA_DOMAIN_TOLERANCE`; or a ray with ``L_z != 0`` within
   :data:`AXIS_SIN_TOLERANCE` of the polar axis (``|sin(theta)| <
   AXIS_SIN_TOLERANCE``). The last case cannot occur for an exact solution:
   the polar potential ``Theta(theta) = Q + a^2 E^2 cos^2(theta) - L_z^2
   cot^2(theta)`` is negative near the axis whenever ``L_z != 0``, so such a
   state is a numerical artefact and the ``1/sin^2(theta)`` terms of the
   equations would overflow (docs/numerical_methods.md, polar axis). Rays
   with exactly ``L_z = 0`` may pass through the axis; in Boyer-Lindquist
   coordinates that continuation leaves ``[0, pi]`` and is reported as
   ``OUT_OF_DOMAIN`` (the camera never samples ``L_z = 0`` exactly).
3. ``CAPTURED``: ``r <= r_plus + horizon_epsilon``.
4. ``ESCAPED``: ``r >= escape_radius`` **and** ``dr/dlambda > 0`` in the
   integration direction (docs/decisions.md D-002: never at an inward launch).
5. ``MAX_AFFINE_PARAMETER``: ``lambda >= lambda_max`` or ``n_steps >=
   max_steps`` (step count passed by the integrator).
6. ``DISK_HIT`` is assigned by the batched integrator's optional disk-plane
   event, not by these rules.

Otherwise the ray is ``RUNNING``.
"""

from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

from kerrray.geodesics.state import IDX_PPH, IDX_R, IDX_TH
from kerrray.geometry import Spacetime, outer_horizon

if TYPE_CHECKING:  # pragma: no cover - typing only (avoids an import cycle)
    from kerrray.geodesics.integrators import IntegratorOptions, TerminationOptions

__all__ = [
    "AXIS_SIN_TOLERANCE",
    "THETA_DOMAIN_TOLERANCE",
    "TerminationState",
    "classify_batch",
    "classify_state",
]

THETA_DOMAIN_TOLERANCE = 1e-9
"""``theta`` may leave ``[0, pi]`` by at most this much (radians) before OUT_OF_DOMAIN."""

AXIS_SIN_TOLERANCE = 1e-12
"""A ray with ``L_z != 0`` and ``|sin(theta)|`` below this is OUT_OF_DOMAIN."""


class TerminationState(IntEnum):
    """Why a ray stopped (``RUNNING`` is the internal 'not yet' value)."""

    RUNNING = 0
    ESCAPED = 1
    CAPTURED = 2
    MAX_AFFINE_PARAMETER = 3
    NUMERICAL_FAILURE = 4
    OUT_OF_DOMAIN = 5
    DISK_HIT = 6


def classify_batch(
    st: Spacetime,
    Y: ArrayLike,
    dYdl: ArrayLike,
    term: TerminationOptions,
    *,
    lam: ArrayLike,
    integ: IntegratorOptions,
    n_steps: ArrayLike | None = None,
    h: ArrayLike | None = None,
) -> NDArray[np.int64]:
    """Vectorised classification of states ``Y`` (N, 8) with derivatives ``dYdl`` (N, 8).

    Args:
        st: The spacetime.
        Y: States after an accepted step.
        dYdl: ``dY/dlambda`` at those states (its ``r`` component is the
            radial motion used by the ESCAPED rule).
        term: Horizon margin and escape radius.
        lam: Affine parameter of each state, shape ``(N,)`` (or scalar).
        integ: Integrator options (``lambda_max``, ``max_steps``, ``h_min``).
        n_steps: Accepted steps taken so far per ray (optional).
        h: Current step size per ray (optional; ``h < h_min`` fails the ray).

    Returns:
        Integer array of shape ``(N,)`` with :class:`TerminationState` values.
    """
    y = np.asarray(Y)
    dy = np.asarray(dYdl)
    if y.shape != dy.shape or y.shape[-1:] != (8,):
        raise ValueError(f"Y and dYdl must both have shape (..., 8), got {y.shape} and {dy.shape}")
    lead = y.shape[:-1]
    lam_arr = np.broadcast_to(np.asarray(lam, dtype=np.float64), lead)
    r = y[..., IDX_R]
    th = y[..., IDX_TH]
    lz = y[..., IDX_PPH]
    dr = dy[..., IDX_R]

    finite = np.isfinite(y).all(axis=-1) & np.isfinite(dy).all(axis=-1)
    failed = ~finite
    if h is not None:
        failed |= np.broadcast_to(np.asarray(h), lead) < integ.h_min
    with np.errstate(invalid="ignore"):
        out_of_domain = (r < 0.0) | (th < -THETA_DOMAIN_TOLERANCE) | (th > np.pi + THETA_DOMAIN_TOLERANCE)
        out_of_domain |= (lz != 0) & (np.abs(np.sin(th)) < AXIS_SIN_TOLERANCE)
        captured = r <= outer_horizon(st) + term.horizon_epsilon
        escaped = (r >= term.escape_radius) & (dr > 0.0)
        exhausted = lam_arr >= integ.lambda_max
    if n_steps is not None:
        exhausted |= np.broadcast_to(np.asarray(n_steps), lead) >= integ.max_steps

    state = np.full(lead, int(TerminationState.RUNNING), dtype=np.int64)
    # Assign in reverse priority so that higher-priority rules overwrite.
    state[exhausted] = TerminationState.MAX_AFFINE_PARAMETER
    state[escaped] = TerminationState.ESCAPED
    state[captured] = TerminationState.CAPTURED
    state[out_of_domain] = TerminationState.OUT_OF_DOMAIN
    state[failed] = TerminationState.NUMERICAL_FAILURE
    return state


def classify_state(
    st: Spacetime,
    y: ArrayLike,
    dydl: ArrayLike,
    term: TerminationOptions,
    *,
    lam: float,
    integ: IntegratorOptions,
    n_steps: int | None = None,
    h: float | None = None,
) -> TerminationState:
    """Classify a single state ``y`` (8,) with derivative ``dydl`` (8,); see :func:`classify_batch`."""
    code = classify_batch(
        st,
        np.asarray(y)[None, :],
        np.asarray(dydl)[None, :],
        term,
        lam=np.asarray([lam], dtype=np.float64),
        integ=integ,
        n_steps=None if n_steps is None else np.asarray([n_steps]),
        h=None if h is None else np.asarray([h]),
    )
    return TerminationState(int(code[0]))
