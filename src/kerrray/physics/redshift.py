"""Relativistic redshift factor ``g = nu_obs / nu_em`` (PROJECT.md section 24;
docs/rendering.md section 2).

For a photon with covariant momentum ``p_mu`` the frequency measured by an
observer with four-velocity ``u^mu`` is ``nu = -p_mu u^mu``; the ratio of the
frequencies measured at reception and at emission is the redshift factor
(Cunningham 1975, ApJ 202, 788; Luminet 1979, A&A 75, 228)::

    g = nu_obs / nu_em = (-p_mu u^mu_obs) / (-p_mu u^mu_em)

For a static observer at infinity ``u_obs = (1, 0, 0, 0)`` so ``nu_obs = E =
-p_t``, the conserved energy. The emitter is a gas element on a circular
equatorial Keplerian orbit with ``u^mu = u^t (1, 0, 0, Omega)``, ``Omega =
dphi/dt`` from Bardeen, Press and Teukolsky 1972 eq. 2.16
(:func:`kerrray.photons.orbits.keplerian_angular_velocity`, re-exported here)
and the normalisation ``u_mu u^mu = -1`` giving ::

    u^t = 1 / sqrt(-(g_tt + 2 Omega g_tphi + Omega^2 g_phph)).

Then ``-p_mu u^mu_em = u^t (E - Omega L_z)`` and the closed form is ``g = 1 /
[u^t (1 - Omega L_z / E)]``, which the tests compare with the direct
contraction implemented here (agreement to ``1e-12``). For Schwarzschild
``u^t = (1 - 3M/r)^{-1/2}`` and ``g = sqrt(1 - 3M/r) / (1 - Omega L_z / E)``,
Luminet's eq. for ``1 + z``.

Backward-traced rays. The ray tracer integrates the momentum *reversed*
(``p_mu -> -p_mu``, ``E < 0`` for the stored state, docs/raytracing.md section
6). Both the numerator ``-p_t`` and the denominator ``-p_mu u^mu`` flip sign
under the reversal, so the ratio ``g`` is *invariant*: :func:`redshift_factor`
accepts the stored (reversed) or the physical momentum alike and never needs
the sign of ``E``. A non-positive result would mean the photon cannot have
been emitted by that emitter (negative energy in its frame) and raises.

A static emitter, ``u = (1 / sqrt(-g_tt), 0, 0, 0)`` (defined outside the
ergosphere), gives the gravitational redshift alone,
:func:`gravitational_redshift_factor`: ``g = sqrt(-g_tt)``, which tends to 1
at large radius.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

from kerrray.geodesics.state import IDX_PT, IDX_R, IDX_TH, STATE_SIZE
from kerrray.geometry import Spacetime, metric_components
from kerrray.photons.orbits import keplerian_angular_velocity
from kerrray.photons.trajectories import EQUATORIAL_TOLERANCE

__all__ = [
    "emitter_four_velocity",
    "gravitational_redshift_factor",
    "keplerian_angular_velocity",
    "redshift_factor",
    "static_four_velocity",
]

FloatArray = NDArray[np.float64]


def emitter_four_velocity(st: Spacetime, r: ArrayLike, *, prograde: bool = True) -> FloatArray:
    """Four-velocity ``u^mu = u^t (1, 0, 0, Omega)`` of a circular equatorial Keplerian orbit.

    ``Omega`` is BPT 1972 eq. 2.16 (prograde means ``a L_z > 0``, docs/
    architecture.md section 1) and ``u^t`` follows from ``g_mu_nu u^mu u^nu =
    -1`` at ``theta = pi/2`` (module docstring). Vectorised over ``r``; the
    result has shape ``r.shape + (4,)``.

    Raises:
        ValueError: If the radicand ``-(g_tt + 2 Omega g_tphi + Omega^2
            g_phph)`` is not positive somewhere: no timelike circular orbit
            exists at or below the circular photon orbit (or inside the
            horizon).
    """
    r_arr = np.asarray(r, dtype=np.float64)
    omega = np.asarray(keplerian_angular_velocity(st, r_arr, prograde), dtype=np.float64)
    g = metric_components(st, r_arr, 0.5 * math.pi)
    radicand = -(g.g_tt + 2.0 * omega * g.g_tphi + omega * omega * g.g_phph)
    if not np.all(np.isfinite(radicand)) or np.any(radicand <= 0.0):
        raise ValueError(
            "no timelike circular orbit at the requested radius (r must exceed the "
            "circular photon orbit radius, kerrray.photons.orbits.equatorial_photon_orbit_radius)"
        )
    u_t = 1.0 / np.sqrt(radicand)
    u = np.zeros(np.shape(u_t) + (4,), dtype=np.float64)
    u[..., 0] = u_t
    u[..., 3] = omega * u_t
    return u


def static_four_velocity(st: Spacetime, r: ArrayLike, theta: ArrayLike) -> FloatArray:
    """Four-velocity ``u = (1 / sqrt(-g_tt), 0, 0, 0)`` of a static emitter.

    Static observers exist only outside the ergosphere (``g_tt < 0``); a
    ``ValueError`` is raised otherwise. Shape ``broadcast(r, theta) + (4,)``.
    """
    g = metric_components(st, r, theta)
    g_tt = np.asarray(g.g_tt, dtype=np.float64)
    if not np.all(np.isfinite(g_tt)) or np.any(g_tt >= 0.0):
        raise ValueError("a static emitter needs g_tt < 0 (outside the ergosphere)")
    u = np.zeros(g_tt.shape + (4,), dtype=np.float64)
    u[..., 0] = 1.0 / np.sqrt(-g_tt)
    return u


def _as_states(y: ArrayLike) -> FloatArray:
    y_arr = np.asarray(y, dtype=np.float64)
    if y_arr.shape[-1:] != (STATE_SIZE,):
        raise ValueError(f"state must end in a dimension of size {STATE_SIZE}, got {y_arr.shape}")
    if not np.all(np.isfinite(y_arr)):
        raise ValueError("photon state must be finite")
    return y_arr


def _frequency_ratio(y: FloatArray, u: FloatArray) -> FloatArray:
    """``g = (-p_t) / (-p_mu u^mu)``; reversal invariant (module docstring)."""
    p = y[..., 4:8]
    p_dot_u = np.sum(p * u, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        g = y[..., IDX_PT] / p_dot_u
    if not np.all(np.isfinite(g)) or np.any(g <= 0.0):
        raise ValueError(
            "non-positive frequency ratio: the photon has non-positive energy in the "
            "emitter frame and cannot have been emitted there"
        )
    return g


def redshift_factor(st: Spacetime, y_hit: ArrayLike, *, prograde: bool = True) -> FloatArray:
    """``g = nu_obs / nu_em`` for a photon state at the equatorial disk, observer at infinity.

    ``y_hit`` holds one or more states ``[t, r, theta, phi, p_t, p_r, p_theta,
    p_phi]`` (shape ``(..., 8)``) at the emission point, with ``theta`` within
    :data:`kerrray.photons.trajectories.EQUATORIAL_TOLERANCE` of ``pi/2``;
    the momentum may be the physical one or the reversed one stored by the
    backward ray tracer (the ratio is invariant, module docstring). The
    emitter is the Keplerian orbit of :func:`emitter_four_velocity` at the
    state's ``r``. Returns an array of shape ``y_hit.shape[:-1]`` (0-d for a
    single state).

    Raises:
        ValueError: If a state is not equatorial, no circular orbit exists
            at its radius, or the frequency ratio is not positive.
    """
    y = _as_states(y_hit)
    if np.any(np.abs(y[..., IDX_TH] - 0.5 * math.pi) > EQUATORIAL_TOLERANCE):
        raise ValueError("redshift_factor needs states in the equatorial plane (theta = pi/2)")
    u = emitter_four_velocity(st, y[..., IDX_R], prograde=prograde)
    return _frequency_ratio(y, u)


def gravitational_redshift_factor(st: Spacetime, y: ArrayLike) -> FloatArray:
    """``g`` for a *static* emitter at the state's ``(r, theta)`` (gravitational redshift only).

    ``g = E / (u^t E) = sqrt(-g_tt(r, theta))``, computed by the same
    contraction as :func:`redshift_factor` with
    :func:`static_four_velocity`; ``g < 1`` everywhere outside the
    ergosphere and ``g -> 1 - M / r`` at large ``r`` (verified in the tests).
    """
    y_arr = _as_states(y)
    u = static_four_velocity(st, y_arr[..., IDX_R], y_arr[..., IDX_TH])
    return _frequency_ratio(y_arr, u)
