"""Trajectory containers and geometric summaries of integrated rays.

:class:`Trajectory` holds one recorded ray (affine parameter and state
samples, termination state, step counts, runtime and conservation
diagnostics); :class:`BatchResult` holds the final states and per-ray
summaries of a vectorised run (docs/architecture.md section 4).

Deflection angle (equatorial escaped rays). The azimuth swept between the
launch point at radius ``r_0`` and the exit point at radius ``r_1`` is
compared with the azimuth a straight line of the same impact parameter
``b = |L_z| / E`` would sweep between the same two radii in flat space. Along
the straight line ``y = b`` traversed in ``x`` from ``-sqrt(r_0^2 - b^2)`` to
``+sqrt(r_1^2 - b^2)`` the polar angle changes by ``pi - arcsin(b/r_0) -
arcsin(b/r_1)``, so ::

    delta = |Delta phi| - pi + arcsin(b / r_0) + arcsin(b / r_1)

Approximation (PROJECT.md section 43):

* WHAT: the flat-space straight-line correction for finite launch and exit
  radii.
* WHY: rays start and stop at finite radius; without the correction the
  finite-radius geometry alone contributes ``arcsin(b/r_0) + arcsin(b/r_1)
  ~ b/r_0 + b/r_1`` (``1.2e-2`` rad for ``b = 6``, ``r = 1000``), far larger
  than the numerical error.
* LIMITATION: the bending accumulated beyond the launch/exit radius is
  neglected. In the weak field the deflection per unit ``x`` along the
  straight line is ``2 M b / (x^2 + b^2)^{3/2}`` (Weinberg 1972, section
  8.5), whose integral beyond ``|x| = X`` is ``(2 M / b)(1 - X / sqrt(X^2 +
  b^2)) ~ M b / X^2`` per end, i.e. about ``6e-6`` rad per end for ``b = 6``
  and ``X = 1000``. For ``a != 0`` the same asymptotics apply (frame dragging
  decays as ``1/r^3``), but the interpretation of the total bending is left
  to the lensing module.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from kerrray.geodesics.state import IDX_PH, IDX_R, IDX_TH
from kerrray.geometry import Spacetime, bl_to_cartesian
from kerrray.photons.classification import TerminationState
from kerrray.photons.constants import ConservationDiagnostics, angular_momentum, energy

__all__ = [
    "BatchResult",
    "EQUATORIAL_TOLERANCE",
    "Trajectory",
    "azimuthal_winding",
    "closest_approach",
    "deflection_angle",
    "to_cartesian",
]

FloatArray = NDArray[np.floating]

EQUATORIAL_TOLERANCE = 1e-6
"""``|theta - pi/2|`` (radians) within which a recorded ray counts as equatorial."""


@dataclass
class Trajectory:
    """One integrated ray.

    Attributes:
        spacetime: The spacetime the ray was integrated in.
        lam: Affine parameter at the recorded points, shape ``(n,)``.
        y: Recorded states, shape ``(n, 8)`` (at least the first and last).
        state: Termination state.
        n_steps: Accepted steps.
        n_rejected: Rejected step attempts (adaptive methods).
        runtime_s: Wall-clock seconds of the integration.
        diagnostics: Conserved quantities and drifts along the recorded points.
    """

    spacetime: Spacetime
    lam: FloatArray
    y: FloatArray
    state: TerminationState
    n_steps: int
    n_rejected: int
    runtime_s: float
    diagnostics: ConservationDiagnostics

    @property
    def y0(self) -> FloatArray:
        """The initial state."""
        return self.y[0]

    @property
    def y_end(self) -> FloatArray:
        """The final state."""
        return self.y[-1]


@dataclass
class BatchResult:
    """Final states and per-ray summaries of :func:`kerrray.geodesics.integrate_batch`.

    Attributes:
        Y: Final states, shape ``(N, 8)`` (in the working dtype).
        state: Termination state codes, shape ``(N,)``.
        n_steps: Accepted steps per ray.
        lam: Final affine parameter per ray (float64).
        max_null_error: Max over the ray of ``|H| / E_0^2``.
        max_energy_drift, max_lz_drift, max_carter_drift: Max relative drifts.
        runtime_s: Wall-clock seconds for the whole batch.
        event_Y: Interpolated disk-crossing states (``(N, 8)``) when the disk
            event is enabled, else ``None``.
        event_hit: Per-ray disk-hit flags when enabled, else ``None``.
        n_rejected: Rejected step attempts per ray.
    """

    Y: FloatArray
    state: NDArray[np.int64]
    n_steps: NDArray[np.int64]
    lam: FloatArray
    max_null_error: FloatArray
    max_energy_drift: FloatArray
    max_lz_drift: FloatArray
    max_carter_drift: FloatArray
    runtime_s: float
    event_Y: FloatArray | None = None
    event_hit: NDArray[np.bool_] | None = None
    n_rejected: NDArray[np.int64] | None = None

    @property
    def n_rays(self) -> int:
        """Number of rays in the batch."""
        return int(self.Y.shape[0])

    def count(self, state: TerminationState) -> int:
        """Number of rays that ended in ``state``."""
        return int(np.count_nonzero(self.state == int(state)))


def closest_approach(traj: Trajectory) -> float:
    """Minimum Boyer-Lindquist ``r`` over the recorded points.

    This is the minimum over the *samples*; between samples the true minimum
    differs by ``O(h^2)`` relative for a smooth turning point (docstring of
    :func:`deflection_angle` for the same caveat).
    """
    return float(np.min(traj.y[:, IDX_R]))


def azimuthal_winding(traj: Trajectory) -> float:
    """Total change of azimuth ``phi_end - phi_0`` in radians (turns = value / 2 pi).

    ``phi`` is integrated continuously (never wrapped), so windings beyond
    ``2 pi`` are preserved.
    """
    return float(traj.y[-1, IDX_PH] - traj.y[0, IDX_PH])


def deflection_angle(traj: Trajectory) -> float:
    """Deflection of an equatorial escaped ray, corrected for finite launch/exit radii.

    ``delta = |Delta phi| - pi + arcsin(b / r_0) + arcsin(b / r_1)`` with
    ``b = |L_z| / E`` (module docstring). Raises ``ValueError`` unless the ray
    is ``ESCAPED`` and stays equatorial within :data:`EQUATORIAL_TOLERANCE`.
    """
    if traj.state != TerminationState.ESCAPED:
        raise ValueError(f"deflection_angle needs an ESCAPED ray, got {traj.state.name}")
    th = traj.y[:, IDX_TH]
    if np.max(np.abs(th - 0.5 * np.pi)) > EQUATORIAL_TOLERANCE:
        raise ValueError("deflection_angle is defined for equatorial rays only")
    y0, y1 = traj.y[0], traj.y[-1]
    b = float(abs(angular_momentum(y0)) / energy(y0))
    r0, r1 = float(y0[IDX_R]), float(y1[IDX_R])
    if b > r0 or b > r1:
        raise ValueError("impact parameter exceeds a launch or exit radius; no straight-line reference")
    return abs(azimuthal_winding(traj)) - np.pi + float(np.arcsin(b / r0)) + float(np.arcsin(b / r1))


def to_cartesian(traj: Trajectory) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Plotting Cartesian coordinates of the recorded points (:func:`kerrray.geometry.bl_to_cartesian`)."""
    return bl_to_cartesian(traj.spacetime, traj.y[:, IDX_R], traj.y[:, IDX_TH], traj.y[:, IDX_PH])
