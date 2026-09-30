"""Conserved quantities of Kerr null geodesics and their drift (PROJECT.md section 10).

In Boyer-Lindquist coordinates the Kerr metric is stationary and axisymmetric,
so the covariant momenta conjugate to ``t`` and ``phi`` are conserved
(Carter 1968, Phys. Rev. 174, 1559; Bardeen, Press and Teukolsky 1972, ApJ
178, 347, section II; Misner, Thorne and Wheeler 1973, section 33.5)::

    E   = -p_t          energy at infinity
    L_z =  p_phi        axial angular momentum

The separability of the Hamilton-Jacobi equation (Carter 1968) gives the
third constant, the Carter constant, which for null geodesics (docs/
architecture.md section 1; BPT 1972 eq. 2.10 with ``mu = 0``) is ::

    Q = p_theta^2 + cos^2(theta) [L_z^2 / sin^2(theta) - a^2 E^2]

and the null constraint is the super-Hamiltonian ``H = (1/2) g^{mu nu} p_mu
p_nu = 0``. The consistency of ``Q`` with the metric is the identity
``2 Sigma H = (Delta p_r^2 - R(r)/Delta) + (p_theta^2 - Theta(theta))``
(docs/equations_geodesics.md), verified in ``tests/test_conservation.py``;
its numerical conservation along integrated rays is the physical test.

Drift (PROJECT.md section 10): ``eps(lambda) = |v(lambda) - v_0| / max(|v_0|,
floor)``. The floor keeps the drift finite when the reference vanishes
(equatorial rays have ``Q_0 = 0`` exactly): below the floor the reported
number is an *absolute* drift in units of the floor. The diagnostics use
the natural scales of ``E = 1`` photons in geometric units, ``E_0`` for the
energy, ``|E_0| M`` for ``L_z`` and ``E_0^2 M^2`` for ``Q`` (a photon's
``L_z`` and ``Q`` scale with ``E`` and ``E^2``); :func:`relative_drift`
itself defaults to ``floor = 1``. The null error is reported as
``|H| / E_0^2``, dimensionless because ``H`` scales as ``E^2``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from kerrray.geodesics.equations import hamiltonian
from kerrray.geodesics.state import IDX_PPH, IDX_PT, IDX_PTH, IDX_TH
from kerrray.geometry import Spacetime

__all__ = [
    "DEFAULT_DRIFT_FLOOR",
    "ConservationDiagnostics",
    "angular_momentum",
    "carter_constant",
    "conservation_diagnostics",
    "energy",
    "drift_floors",
    "null_constraint",
    "null_constraint_from_rhs",
    "null_error",
    "relative_drift",
]

FloatArray = NDArray[np.floating]

DEFAULT_DRIFT_FLOOR = 1.0
"""Default floor of :func:`relative_drift` (one unit of the quantity, geometric units)."""


def energy(y: ArrayLike) -> FloatArray:
    """Photon energy at infinity ``E = -p_t`` for states ``y`` of shape ``(..., 8)``."""
    return -np.asarray(y)[..., IDX_PT]


def angular_momentum(y: ArrayLike) -> FloatArray:
    """Axial angular momentum ``L_z = p_phi`` for states ``y`` of shape ``(..., 8)``."""
    return np.asarray(y)[..., IDX_PPH]


def carter_constant(st: Spacetime, y: ArrayLike) -> FloatArray:
    """Carter constant ``Q = p_theta^2 + cos^2(theta) [L_z^2 / sin^2(theta) - a^2 E^2]``.

    Null case (BPT 1972 eq. 2.10 with ``mu = 0``; docs/architecture.md
    section 1). Rays with exactly ``L_z = 0`` drop the ``L_z^2 / sin^2(theta)``
    term so that the polar axis gives the finite limit ``p_theta^2 - a^2 E^2
    cos^2(theta)``.
    """
    y_arr = np.asarray(y)
    th = y_arr[..., IDX_TH]
    p_th = y_arr[..., IDX_PTH]
    lz = y_arr[..., IDX_PPH]
    e = -y_arr[..., IDX_PT]
    s2 = np.sin(th) ** 2
    lz2_over_s2 = np.divide(lz * lz, s2, out=np.zeros(np.broadcast(lz, s2).shape, dtype=y_arr.dtype), where=lz != 0)
    return p_th * p_th + np.cos(th) ** 2 * (lz2_over_s2 - st.a**2 * e * e)


def null_constraint(st: Spacetime, y: ArrayLike) -> FloatArray:
    """The null constraint ``H = (1/2) g^{mu nu} p_mu p_nu`` (zero on null geodesics)."""
    return hamiltonian(st, y)


def null_constraint_from_rhs(y: ArrayLike, dydl: ArrayLike) -> FloatArray:
    """``H = (1/2) p_mu dx^mu/dlambda`` from a state and its Hamiltonian right-hand side.

    Identical to :func:`null_constraint` because ``dx^mu/dlambda = g^{mu nu}
    p_nu``; it reuses the derivative the integrator already evaluated instead
    of re-evaluating the metric.
    """
    y_arr = np.asarray(y)
    d_arr = np.asarray(dydl)
    return 0.5 * np.sum(y_arr[..., 4:] * d_arr[..., :4], axis=-1)


def null_error(st: Spacetime, y: ArrayLike, e0: ArrayLike) -> FloatArray:
    """Dimensionless null error ``|H| / E_0^2`` with ``E_0`` the reference energy."""
    e0_arr = np.asarray(e0)
    return np.abs(null_constraint(st, y)) / (e0_arr * e0_arr)


def relative_drift(values: ArrayLike, reference: ArrayLike, *, floor: float = DEFAULT_DRIFT_FLOOR) -> FloatArray:
    """``|values - reference| / max(|reference|, floor)`` elementwise (broadcasting).

    ``floor > 0`` must hold; below it the result is an absolute drift in
    units of the floor (module docstring).
    """
    if not floor > 0.0:
        raise ValueError(f"floor must be > 0, got {floor!r}")
    v = np.asarray(values)
    v0 = np.asarray(reference)
    return np.abs(v - v0) / np.maximum(np.abs(v0), floor)


@dataclass
class ConservationDiagnostics:
    """Conserved quantities along a trajectory and their drift from the first point.

    Attributes:
        energy, lz, carter, null: The four monitored quantities at each
            recorded point (shape ``(n,)``); ``null`` is ``H`` itself.
        energy_drift, lz_drift, carter_drift: Relative drifts against the
            first point (floors: ``E_0``-based scales, module docstring).
        null_error: ``|H| / E_0^2`` at each point.
        max_energy_drift, max_lz_drift, max_carter_drift, max_null_error:
            The maxima over the recorded points (``max_null`` is an alias of
            ``max_null_error``).
    """

    energy: FloatArray
    lz: FloatArray
    carter: FloatArray
    null: FloatArray
    energy_drift: FloatArray
    lz_drift: FloatArray
    carter_drift: FloatArray
    null_error: FloatArray
    max_energy_drift: float
    max_lz_drift: float
    max_carter_drift: float
    max_null_error: float

    @property
    def max_null(self) -> float:
        """Alias of ``max_null_error``."""
        return self.max_null_error


def drift_floors(st: Spacetime, e0: ArrayLike) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Floors ``(|E_0|, |E_0| M, E_0^2 M^2)`` used for the energy, ``L_z`` and ``Q`` drifts.

    A vanishing ``E_0`` (unphysical) falls back to :data:`DEFAULT_DRIFT_FLOOR`.
    """
    e = np.abs(np.asarray(e0, dtype=np.float64))
    e = np.where(e > 0.0, e, DEFAULT_DRIFT_FLOOR)
    m = st.mass
    return e, e * m, e * e * m * m


def conservation_diagnostics(st: Spacetime, y: ArrayLike) -> ConservationDiagnostics:
    """Evaluate the four conserved quantities along states ``y`` of shape ``(n, 8)``.

    The first row is the reference for the drifts.
    """
    y_arr = np.asarray(y, dtype=np.float64)
    if y_arr.ndim != 2:
        raise ValueError(f"y must have shape (n, 8), got {y_arr.shape}")
    e = energy(y_arr)
    lz = angular_momentum(y_arr)
    q = carter_constant(st, y_arr)
    h = null_constraint(st, y_arr)
    fe, fl, fq = drift_floors(st, e[0])
    e_drift = relative_drift(e, e[0], floor=float(fe))
    l_drift = relative_drift(lz, lz[0], floor=float(fl))
    q_drift = relative_drift(q, q[0], floor=float(fq))
    n_err = np.abs(h) / float(fe) ** 2
    return ConservationDiagnostics(
        energy=e,
        lz=lz,
        carter=q,
        null=h,
        energy_drift=e_drift,
        lz_drift=l_drift,
        carter_drift=q_drift,
        null_error=n_err,
        max_energy_drift=float(np.max(e_drift)),
        max_lz_drift=float(np.max(l_drift)),
        max_carter_drift=float(np.max(q_drift)),
        max_null_error=float(np.max(n_err)),
    )
