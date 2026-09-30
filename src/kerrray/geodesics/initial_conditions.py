"""Photon initial states from constants of motion (PROJECT.md sections 8, 10, 12).

Carter's separated equations for null geodesics (Carter 1968; Bardeen, Press
and Teukolsky 1972, eqs. 2.9-2.10 with ``mu = 0``; docs/architecture.md
section 1.2) give the radial and polar potentials ::

    R(r)         = [E (r^2 + a^2) - a L_z]^2 - Delta [Q + (L_z - a E)^2]
    Theta(theta) = Q - cos^2(theta) [L_z^2 / sin^2(theta) - a^2 E^2]

with ``Sigma dr/dlambda = +-sqrt(R)`` and ``Sigma dtheta/dlambda =
+-sqrt(Theta)``. Since ``dr/dlambda = g^{rr} p_r = (Delta / Sigma) p_r`` and
``dtheta/dlambda = p_theta / Sigma``, the covariant momenta are ::

    p_r = +- sqrt(R) / Delta,      p_theta = +- sqrt(Theta),
    p_t = -E,                      p_phi = L_z.

The potentials are consistent with the metric through the identity
``2 Sigma H = (Delta p_r^2 - R/Delta) + (p_theta^2 - Theta)``
(docs/equations_geodesics.md), verified symbolically and numerically in
``tests/test_conservation.py``; every constructor's output is checked to be
null (``H = 0``) in ``tests/test_null_condition.py``.

Sign conventions (docs/architecture.md section 1): ``a > 0`` rotates towards
``+phi`` and a photon is *prograde* when ``a L_z > 0``; for ``a = 0``
prograde means ``L_z > 0``. Photons are future-directed with ``E > 0``.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from kerrray.geodesics.state import pack
from kerrray.geometry import Spacetime, delta, outer_horizon

__all__ = [
    "RADICAND_TOLERANCE",
    "carter_potentials",
    "equatorial_photon",
    "photon_from_constants",
    "tangential_photon",
]

RADICAND_TOLERANCE = 1e-12
"""Radicands ``R`` or ``Theta`` more negative than this times their natural scale raise."""


def carter_potentials(
    st: Spacetime, r: float, theta: float, E: float, Lz: float, Q: float
) -> tuple[float, float, float, float]:
    """Return ``(R(r), Theta(theta), scale_R, scale_Theta)`` for the given constants.

    The scales are the sums of the magnitudes of the terms, used to decide
    whether a slightly negative radicand is round-off (module docstring).
    """
    a = st.a
    dlt = float(delta(st, r))
    term1 = E * (r * r + a * a) - a * Lz
    term2 = dlt * (Q + (Lz - a * E) ** 2)
    big_r = term1 * term1 - term2
    scale_r = term1 * term1 + abs(term2)
    c2, s2 = math.cos(theta) ** 2, math.sin(theta) ** 2
    lz_term = 0.0 if Lz == 0.0 else Lz * Lz / s2
    big_theta = Q - c2 * (lz_term - a * a * E * E)
    scale_theta = abs(Q) + c2 * (lz_term + a * a * E * E)
    return big_r, big_theta, scale_r, scale_theta


def _sqrt_radicand(value: float, scale: float, name: str) -> float:
    if value < -RADICAND_TOLERANCE * max(scale, 1.0):
        raise ValueError(f"{name} = {value!r} < 0: the requested constants of motion are inconsistent here")
    return math.sqrt(max(value, 0.0))


def _check_sign(sign: int, name: str) -> float:
    if sign not in (-1, 1):
        raise ValueError(f"{name} must be +1 or -1, got {sign!r}")
    return float(sign)


def photon_from_constants(
    st: Spacetime,
    r: float,
    theta: float,
    E: float,
    Lz: float,
    Q: float,
    *,
    sign_r: int,
    sign_theta: int,
    t: float = 0.0,
    phi: float = 0.0,
) -> NDArray[np.float64]:
    """Photon state at ``(t, r, theta, phi)`` with constants ``(E, L_z, Q)``.

    ``p_r = sign_r sqrt(R)/Delta`` and ``p_theta = sign_theta sqrt(Theta)``
    (module docstring). Raises ``ValueError`` if ``E <= 0``, if ``r`` is not
    outside the horizon, or if ``R`` or ``Theta`` is negative beyond round-off
    (the constants are inconsistent with a photon at that point).
    """
    if not E > 0.0:
        raise ValueError(f"E must be > 0 for a future-directed photon, got {E!r}")
    if not r > outer_horizon(st):
        raise ValueError(f"r = {r!r} must lie outside the outer horizon r_plus = {outer_horizon(st)!r}")
    sr = _check_sign(sign_r, "sign_r")
    sth = _check_sign(sign_theta, "sign_theta")
    big_r, big_theta, scale_r, scale_theta = carter_potentials(st, r, theta, E, Lz, Q)
    p_r = sr * _sqrt_radicand(big_r, scale_r, "R(r)") / float(delta(st, r))
    p_theta = sth * _sqrt_radicand(big_theta, scale_theta, "Theta(theta)")
    return pack([t, r, theta, phi], [-E, p_r, p_theta, Lz])


def _lz_sign(st: Spacetime, prograde: bool) -> float:
    """``+1`` or ``-1`` so that ``a L_z > 0`` for prograde (``L_z > 0`` when ``a = 0``)."""
    same_as_axis = st.a >= 0.0
    return 1.0 if prograde == same_as_axis else -1.0


def equatorial_photon(
    st: Spacetime,
    r0: float,
    b: float,
    *,
    inward: bool = True,
    prograde: bool = True,
    E: float = 1.0,
) -> NDArray[np.float64]:
    """Equatorial photon at ``r0`` with impact parameter ``b = |L_z| / E >= 0``.

    ``theta = pi/2``, ``p_theta = 0`` (``Q = 0``), ``p_phi = +-b E`` with the
    sign fixed by ``prograde`` (module docstring), ``p_t = -E`` and ``p_r``
    from the null condition with the inward (``p_r < 0``) or outward sign.
    """
    if b < 0.0:
        raise ValueError(f"b must be >= 0, got {b!r}")
    lz = _lz_sign(st, prograde) * b * E
    return photon_from_constants(
        st, r0, 0.5 * math.pi, E, lz, 0.0, sign_r=-1 if inward else 1, sign_theta=1
    )


def tangential_photon(st: Spacetime, r0: float, *, prograde: bool = True, E: float = 1.0) -> NDArray[np.float64]:
    """Equatorial photon launched tangentially (``p_r = 0``, ``p_theta = 0``) at ``r0``.

    ``R(r0) = 0`` with ``Q = 0`` is ``E (r0^2 + a^2) - a L_z = s sqrt(Delta)
    (L_z - a E)`` with ``s = +-1``, whence ::

        L_z = E [(r0^2 + a^2) + s a sqrt(Delta)] / (a + s sqrt(Delta)),

    which for ``a = 0`` is ``L_z = s E r0 / sqrt(1 - 2 M / r0)`` (the familiar
    ``b = r0 / sqrt(1 - 2M/r0)``). ``s = +1`` is the root with ``L_z > 0`` at
    large radius, i.e. prograde for ``a >= 0``. Inside the ergosphere
    (equatorial ``r0 < 2 M``, ``a != 0``) both roots have ``a L_z > 0``; the
    ``s = -1`` root is singular at ``r0 = 2 M`` exactly (``L_z / E -> inf``,
    the tangential null direction is the Killing direction there) and a
    ``ValueError`` is raised in that case. The state is checked to be null by
    the tests, not re-normalised.
    """
    if not E > 0.0:
        raise ValueError(f"E must be > 0, got {E!r}")
    if not r0 > outer_horizon(st):
        raise ValueError(f"r0 = {r0!r} must lie outside the outer horizon r_plus = {outer_horizon(st)!r}")
    a = st.a
    s = _lz_sign(st, prograde)
    sqrt_delta = math.sqrt(float(delta(st, r0)))
    denom = a + s * sqrt_delta
    if denom == 0.0:
        raise ValueError("no finite tangential photon: r0 is on the equatorial ergosphere for this direction")
    lz = E * ((r0 * r0 + a * a) + s * a * sqrt_delta) / denom
    return pack([0.0, r0, 0.5 * math.pi, 0.0], [-E, 0.0, 0.0, lz])
