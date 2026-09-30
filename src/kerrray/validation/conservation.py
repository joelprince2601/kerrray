"""Reusable conservation-drift summaries (PROJECT.md section 10; docs/validation.md).

Along an exact null geodesic of Kerr the energy ``E = -p_t``, the axial
angular momentum ``L_z = p_phi``, the Carter constant ``Q`` and the null
constraint ``H = (1/2) g^{mu nu} p_mu p_nu = 0`` are conserved (Carter 1968,
Phys. Rev. 174, 1559; Bardeen, Press and Teukolsky 1972, ApJ 178, 347, eqs.
2.9-2.10). Their drifts are computed by :mod:`kerrray.photons.constants`
(relative to the first point with the ``E_0``-based floors documented there);
this module only condenses them into one record per ray and applies a
tolerance.

Note that in the Hamiltonian formulation ``p_t`` and ``p_phi`` are constant
*by construction* (their right-hand sides vanish identically), so the
``E`` and ``L_z`` drifts are exactly zero up to round-off; the informative
diagnostics are ``Q`` and the null constraint, which the integrator does not
preserve structurally (docs/numerical_methods.md section 5).

The second half of the module holds the trajectory-level runs of the Kerr
validation (docs/validation.md sections 3-4): off-equatorial launch states, a
per-ray summary row and the forward/backward round trip.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from kerrray.geodesics import IntegratorOptions, TerminationOptions, integrate, photon_from_constants
from kerrray.geometry import Spacetime
from kerrray.photons import BatchResult, Trajectory, azimuthal_winding, closest_approach

__all__ = [
    "NO_ESCAPE_RADIUS",
    "DriftSummary",
    "batch_drift_summary",
    "drift_summary",
    "off_equatorial_ray",
    "reversibility_roundtrip",
    "trajectory_row",
    "worst_drift",
]

NO_ESCAPE_RADIUS: Final[float] = 1.0e9
"""Escape radius (units of M) used when a ray must run its full affine budget."""


@dataclass(frozen=True)
class DriftSummary:
    """Maximum drifts of one ray.

    Attributes:
        energy, lz, carter: Maximum relative drifts of ``E``, ``L_z`` and ``Q``.
        null: Maximum of ``|H| / E_0^2``.
    """

    energy: float
    lz: float
    carter: float
    null: float

    @property
    def worst(self) -> float:
        """Largest of the four (``nan`` only if all are ``nan``)."""
        finite = [v for v in (self.energy, self.lz, self.carter, self.null) if math.isfinite(v)]
        return max(finite) if finite else math.nan

    def within(self, tolerance: float) -> bool:
        """True when every drift is finite and ``<= tolerance``."""
        values = (self.energy, self.lz, self.carter, self.null)
        return all(math.isfinite(v) and v <= tolerance for v in values)

    def as_dict(self) -> dict[str, float]:
        """``{"energy", "lz", "carter", "null"}`` mapping."""
        return {"energy": self.energy, "lz": self.lz, "carter": self.carter, "null": self.null}


def drift_summary(traj: Trajectory) -> DriftSummary:
    """Summary of a recorded trajectory (maxima over every accepted step)."""
    d = traj.diagnostics
    return DriftSummary(
        energy=float(d.max_energy_drift),
        lz=float(d.max_lz_drift),
        carter=float(d.max_carter_drift),
        null=float(d.max_null_error),
    )


def batch_drift_summary(result: BatchResult, index: int | None = None) -> DriftSummary:
    """Summary of ray ``index`` of a batch, or the maxima over all rays when ``index`` is ``None``."""
    sel: Any = slice(None) if index is None else index

    def pick(arr: np.ndarray) -> float:
        return float(np.max(np.asarray(arr, dtype=np.float64)[sel]))

    return DriftSummary(
        energy=pick(result.max_energy_drift),
        lz=pick(result.max_lz_drift),
        carter=pick(result.max_carter_drift),
        null=pick(result.max_null_error),
    )


def worst_drift(summaries: Iterable[DriftSummary]) -> DriftSummary:
    """Component-wise maximum of several summaries."""
    items = list(summaries)
    if not items:
        raise ValueError("need at least one DriftSummary")
    return DriftSummary(
        energy=max(s.energy for s in items),
        lz=max(s.lz for s in items),
        carter=max(s.carter for s in items),
        null=max(s.null for s in items),
    )


# --------------------------------------------------------------------------
# Trajectory-level consistency runs used by the Kerr validation
# --------------------------------------------------------------------------


def off_equatorial_ray(st: Spacetime, r0: float, theta_deg: float, xi: float, eta: float) -> NDArray[np.float64]:
    """Inward photon at ``(r0, theta)`` with ``E = 1``, ``L_z = xi``, ``Q = eta`` (``Q > 0``: off-equatorial)."""
    return photon_from_constants(st, r0, math.radians(theta_deg), 1.0, xi, eta, sign_r=-1, sign_theta=1)


def trajectory_row(traj: Trajectory, **extra: Any) -> dict[str, Any]:
    """State, steps, runtime, closest approach, winding and drifts of one ray as a plain mapping."""
    dphi = azimuthal_winding(traj)
    return {
        **extra,
        "state": traj.state.name,
        "n_steps": traj.n_steps,
        "n_rejected": traj.n_rejected,
        "runtime_s": traj.runtime_s,
        "closest_approach": closest_approach(traj),
        "affine_length": float(traj.lam[-1]),
        "delta_phi": dphi,
        "turns": dphi / (2.0 * math.pi),
        "max_drift": drift_summary(traj).as_dict(),
    }


def reversibility_roundtrip(
    st: Spacetime,
    y0: NDArray[np.float64],
    integ: IntegratorOptions,
    term: TerminationOptions,
    affine_length: float,
) -> dict[str, Any]:
    """Integrate forward for ``affine_length``, reverse ``p_mu``, integrate back and compare with ``y0``.

    The geodesic equation is invariant under ``(lambda, p) -> (-lambda, -p)``
    (docs/architecture.md section 1), so the exact return trip reproduces
    ``y0``. Escape is disabled (``escape_radius`` set to
    :data:`NO_ESCAPE_RADIUS`) so both legs run the full budget and end
    ``MAX_AFFINE_PARAMETER`` unless the ray is captured. Errors: radial
    (relative to ``r0``), angular (radians), time (relative to the elapsed
    ``t``) and momentum (relative to ``E``).
    """
    integ_l = replace(integ, lambda_max=affine_length)
    term_l = replace(term, escape_radius=NO_ESCAPE_RADIUS)
    fwd = integrate(st, y0, integ_l, term_l, record=False)
    y1 = fwd.y_end.copy()
    y1[4:] *= -1.0
    bwd = integrate(st, y1, integ_l, term_l, record=False)
    y2 = bwd.y_end.copy()
    y2[4:] *= -1.0
    dx, dp = np.abs(y2[:4] - y0[:4]), np.abs(y2[4:] - y0[4:])
    return {
        "forward_state": fwd.state.name,
        "backward_state": bwd.state.name,
        "turning_radius": float(fwd.y_end[1]),
        "n_steps": [fwd.n_steps, bwd.n_steps],
        "runtime_s": fwd.runtime_s + bwd.runtime_s,
        "start": y0.tolist(),
        "returned": y2.tolist(),
        "errors": {
            "radius": float(dx[1] / y0[1]),
            "angles": float(max(dx[2], dx[3])),
            "time": float(dx[0] / max(1.0, abs(float(fwd.y_end[0])))),
            "momentum": float(np.max(dp) / abs(float(y0[4]))),
        },
    }
