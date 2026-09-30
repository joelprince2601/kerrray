"""Kerr photon orbits, shadow curve, ISCO and Keplerian angular velocity.

Every closed form here was *derived* with SymPy from the Carter radial
potential or from the metric (the derivations are re-run in
``tests/test_orbits.py`` and ``tests/test_orbits_timelike.py`` and written
out in ``docs/derivations.md``) and then
matched against the published forms cited in each docstring. Conventions
follow ``docs/architecture.md`` section 1: G = c = 1, Boyer-Lindquist
coordinates, Delta = r^2 - 2 M r + a^2, ``a = spin * M`` with |a| < M, and
*prograde* means a * L_z > 0. For a < 0 the co-rotating (prograde) orbit is
the one with L_z < 0: all radii depend on |a| only and the signs of xi, L
and Omega follow sign(a).

References:

* Carter 1968, Phys. Rev. 174, 1559 (separated null geodesic equations).
* Bardeen, Press & Teukolsky 1972, ApJ 178, 347 (BPT): eq. (2.9) radial
  potential, (2.12)-(2.13) circular-orbit E and L, (2.16) Omega, (2.18)
  equatorial photon orbit, (2.21) marginally stable orbit.
* Bardeen 1973, in *Black Holes* (Les Houches 1972): spherical photon
  orbits and the shadow curve.
* Johannsen & Psaltis 2010, ApJ 718, 446, eqs. (8)-(9) (xi(r), eta(r)).
* Cunha & Herdeiro 2018, Gen. Rel. Grav. 50, 42, eqs. (10)-(11).
* Teo 2003, Gen. Rel. Grav. 35, 1909 (range of spherical photon orbits).

``st`` is any object with ``mass`` and ``a`` attributes
(``kerrray.geometry.metric.Spacetime``). No SymPy is imported here.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.optimize import brentq

Array = Any

# brentq tolerances (absolute, in units of M, and relative): far below the
# 1e-12 agreement demanded of the closed-form comparisons in the tests.
_ROOT_XTOL = 1e-14
_ROOT_RTOL = 1e-14
# The marginally stable orbit never lies beyond 9 M (retrograde, a -> M, BPT
# eq. 2.21), so 12 M bounds the ISCO bracket; the sign change is asserted.
_ISCO_BRACKET_UPPER = 12.0


def _mass_and_spin(st: Any) -> tuple[float, float]:
    """Return (M, a) from a spacetime object and validate M > 0, |a| < M."""
    mass = float(st.mass)
    a = float(st.a)
    if not mass > 0.0:
        raise ValueError(f"mass must be positive, got {mass}")
    if not abs(a) < mass:
        raise ValueError(f"|a| < M required for a black hole, got a={a}, M={mass}")
    return mass, a


def _direction_sign(prograde: bool) -> int:
    """+1 for prograde (a L_z > 0), -1 for retrograde."""
    return 1 if prograde else -1


def _spin_sign(a: float) -> int:
    """sign(a) with sign(0) = +1 (for a = 0 the two senses are degenerate)."""
    return -1 if a < 0.0 else 1


# --------------------------------------------------------------------------
# Carter potentials (null case)
# --------------------------------------------------------------------------


def radial_potential(st: Any, r: Array, xi: Array, eta: Array) -> Array:
    """R(r)/E^2 = [(r^2 + a^2) - a xi]^2 - Delta [eta + (xi - a)^2] (Carter 1968; BPT eq. 2.9, mu = 0)."""
    mass, a = _mass_and_spin(st)
    r = np.asarray(r, dtype=np.float64)
    delta = r * r - 2.0 * mass * r + a * a
    return ((r * r + a * a) - a * xi) ** 2 - delta * (eta + (xi - a) ** 2)


def radial_potential_derivative(st: Any, r: Array, xi: Array, eta: Array) -> Array:
    """dR/dr of :func:`radial_potential` (Delta' = 2 (r - M))."""
    mass, a = _mass_and_spin(st)
    r = np.asarray(r, dtype=np.float64)
    return 4.0 * r * ((r * r + a * a) - a * xi) - 2.0 * (r - mass) * (eta + (xi - a) ** 2)


def theta_potential(st: Any, theta: Array, xi: Array, eta: Array) -> Array:
    """Theta/E^2 = eta + a^2 cos^2(theta) - xi^2 cot^2(theta) (Carter 1968)."""
    _, a = _mass_and_spin(st)
    theta = np.asarray(theta, dtype=np.float64)
    cot2 = (np.cos(theta) / np.sin(theta)) ** 2
    return eta + a * a * np.cos(theta) ** 2 - xi * xi * cot2


# --------------------------------------------------------------------------
# Spherical photon orbits
# --------------------------------------------------------------------------


def spherical_orbit_constants(st: Any, r: Array) -> tuple[Array, Array]:
    """Constants (xi, eta) of the spherical photon orbit at radius r (vectorised).

    Derived with SymPy (``tests/test_orbits.py``) from R = 0, dR/dr = 0::

        xi(r)  = -[r^2 (r - 3M) + a^2 (r + M)] / [a (r - M)]
        eta(r) = r^3 [4 M a^2 - r (r - 3M)^2] / [a^2 (r - M)^2]

    identical to Bardeen 1973, Johannsen & Psaltis 2010 eqs. (8)-(9) and Cunha
    & Herdeiro 2018 eqs. (10)-(11); the other branch (xi = (r^2 + a^2)/a,
    eta = -r^4/a^2 < 0) is discarded. The explicit (r - 3M) avoids
    cancellation at small |a|. Valid for r > M. Raises ``ValueError`` for
    a = 0, where every spherical photon orbit sits at r = 3M and only
    xi^2 + eta = 27 M^2 is fixed (the parametrisation by r degenerates).
    """
    mass, a = _mass_and_spin(st)
    if a == 0.0:
        raise ValueError(
            "spherical_orbit_constants is undefined for a = 0: all spherical photon "
            "orbits of Schwarzschild lie at r = 3M with xi^2 + eta = 27 M^2"
        )
    r = np.asarray(r, dtype=np.float64)
    r_m3 = r - 3.0 * mass
    xi = -(r * r * r_m3 + a * a * (r + mass)) / (a * (r - mass))
    eta = r**3 * (4.0 * mass * a * a - r * r_m3 * r_m3) / (a * a * (r - mass) ** 2)
    return xi, eta


def equatorial_photon_orbit_radius(st: Any, prograde: bool = True) -> float:
    """Equatorial circular photon orbit radius by bracketed root finding.

    eta(r) = 0 is the cubic r (r - 3M)^2 = 4 M a^2, whose *double* root at
    a = 0 bracketing solvers resolve only to about sqrt(epsilon). The
    equivalent regular form r - 3M = -+ 2 |a| sqrt(M / r) (upper sign
    prograde, r < 3M; lower retrograde, r > 3M) is monotone on the brackets
    [M, 3.5M] and [2.5M, 4M], so its root is unique and simple. Agreement
    with BPT eq. (2.18) is asserted in the tests to 1e-12.
    """
    mass, a = _mass_and_spin(st)
    s = _direction_sign(prograde)
    abs_a = abs(a)

    def f(r: float) -> float:
        return r - 3.0 * mass + s * 2.0 * abs_a * math.sqrt(mass / r)

    lo, hi = (mass, 3.5 * mass) if prograde else (2.5 * mass, 4.0 * mass)
    return float(brentq(f, lo, hi, xtol=_ROOT_XTOL * mass, rtol=_ROOT_RTOL))


def equatorial_photon_orbit_radius_reference(st: Any, prograde: bool = True) -> float:
    """BPT 1972 eq. (2.18): r_ph = 2M {1 + cos[(2/3) arccos(-+ |a|/M)]}, upper sign prograde.

    Reference closed form used only to check :func:`equatorial_photon_orbit_radius`.
    """
    mass, a = _mass_and_spin(st)
    s = _direction_sign(prograde)
    return 2.0 * mass * (1.0 + math.cos((2.0 / 3.0) * math.acos(-s * abs(a) / mass)))


def _equatorial_orbit_xi(mass: float, a: float, r: float, prograde: bool) -> float:
    """xi on the equatorial photon orbit in a form regular at a = 0.

    From the SymPy identity eta + (xi - a)^2 = 4 r^2 Delta / (r - M)^2 with
    eta = 0; the sign of xi - a = -r (r^2 - 3 M r + 2 a^2) / [a (r - M)] is
    sign(a) for prograde and -sign(a) for retrograde.
    """
    s = _direction_sign(prograde)
    delta = r * r - 2.0 * mass * r + a * a
    return _spin_sign(a) * (abs(a) + s * 2.0 * r * math.sqrt(delta) / (r - mass))


def critical_impact_parameters(st: Any) -> tuple[float, float]:
    """(b_prograde, b_retrograde) = |xi| = |L_z|/E on the equatorial photon orbits.

    For a = 0 both equal the Schwarzschild critical impact parameter, which
    the tests compare with the reference 3 sqrt(3) M.
    """
    mass, a = _mass_and_spin(st)
    out = []
    for prograde in (True, False):
        r_ph = equatorial_photon_orbit_radius(st, prograde)
        out.append(abs(_equatorial_orbit_xi(mass, a, r_ph, prograde)))
    return out[0], out[1]


def spherical_orbit_radius_range(st: Any) -> tuple[float, float]:
    """(r_min, r_max) of the spherical photon orbits: the prograde and retrograde equatorial radii.

    Spherical photon orbits outside the horizon exist exactly where
    eta(r) >= 0, i.e. between the two equatorial circular photon orbits
    (Teo 2003). For a = 0 both bounds are 3M.
    """
    return (
        equatorial_photon_orbit_radius(st, prograde=True),
        equatorial_photon_orbit_radius(st, prograde=False),
    )


# --------------------------------------------------------------------------
# Shadow curve (Bardeen 1973)
# --------------------------------------------------------------------------


def shadow_curve(st: Any, inclination_deg: float, n: int = 720) -> tuple[np.ndarray, np.ndarray]:
    """Analytic shadow boundary (alpha, beta) for an observer at infinity.

    Bardeen 1973 celestial coordinates at inclination i (the observer's
    Boyer-Lindquist theta; architecture section 1.3)::

        alpha = -xi / sin(i),  beta = +- sqrt(eta + a^2 cos^2(i) - xi^2 cot^2(i))

    along the spherical photon orbits r in [r_a, r_b], the sub-interval of
    [r_min, r_max] where the radicand is non-negative; r_a, r_b are found by
    bracketed root finding on either side of the polar orbit r_0 (xi = 0,
    radicand > 0), which resolves arbitrarily small inclinations. The
    +-beta branches form one closed curve of ``2 * (n // 2)`` points ordered
    by polar angle atan2(beta, alpha) about the origin (inside every Kerr
    shadow: the central ray xi = 0, eta = -a^2 cos^2 i is captured);
    consecutive points, including last -> first, are neighbours. a = 0 is an
    explicit branch: the circle of radius :func:`critical_impact_parameters`
    (computed). i = 90 uses cos(i) = 0 exactly. Requires 0 < i < 180 deg.
    """
    mass, a = _mass_and_spin(st)
    if not 0.0 < inclination_deg < 180.0:
        raise ValueError("inclination_deg must lie strictly between 0 and 180")
    n_points = 2 * (int(n) // 2)
    if n_points < 4:
        raise ValueError("n must be at least 4")
    inc = math.radians(inclination_deg)
    sin_i = math.sin(inc)
    cos_i = 0.0 if inclination_deg == 90.0 else math.cos(inc)
    if a == 0.0:
        b_c = critical_impact_parameters(st)[0]
        angle = np.linspace(-math.pi, math.pi, n_points, endpoint=False)
        return b_c * np.cos(angle), b_c * np.sin(angle)

    cot2 = (cos_i / sin_i) ** 2
    r_lo, r_hi = spherical_orbit_radius_range(st)
    tol = {"xtol": _ROOT_XTOL * mass, "rtol": _ROOT_RTOL}

    def xi_of(r: float) -> float:
        return float(spherical_orbit_constants(st, r)[0])

    def radicand(r: float) -> float:
        xi, eta = spherical_orbit_constants(st, r)
        return float(eta + a * a * cos_i * cos_i - xi * xi * cot2)

    if xi_of(r_lo) * xi_of(r_hi) >= 0.0:
        raise RuntimeError("xi(r) does not change sign on the spherical-orbit range")
    r_0 = brentq(xi_of, r_lo, r_hi, **tol)
    if radicand(r_0) <= 0.0:
        raise RuntimeError("radicand is not positive at the polar orbit")
    r_a = r_lo if radicand(r_lo) >= 0.0 else brentq(radicand, r_lo, r_0, **tol)
    r_b = r_hi if radicand(r_hi) >= 0.0 else brentq(radicand, r_0, r_hi, **tol)

    m = n_points // 2 + 1
    u = np.linspace(0.0, 1.0, m)
    r = r_a + (r_b - r_a) * 0.5 * (1.0 - np.cos(math.pi * u))  # cluster at the turning points
    xi, eta = spherical_orbit_constants(st, r)
    rad = eta + a * a * cos_i * cos_i - xi * xi * cot2
    beta_up = np.sqrt(np.clip(rad, 0.0, None))
    # r_a and r_b are roots of the radicand by construction; the root finder
    # leaves a residual of order xtol * |d(radicand)/dr| ~ 1e-14 whose square
    # root (~1e-7) is pure noise, so the turning points are placed on beta = 0.
    beta_up[0] = 0.0
    beta_up[-1] = 0.0
    alpha_up = -xi / sin_i
    alpha = np.concatenate([alpha_up, alpha_up[-2:0:-1]])
    beta = np.concatenate([beta_up, -beta_up[-2:0:-1]])
    order = np.argsort(np.arctan2(beta, alpha), kind="stable")
    return alpha[order], beta[order]


# --------------------------------------------------------------------------
# Timelike equatorial circular orbits: Omega, E, L, ISCO
# --------------------------------------------------------------------------


def keplerian_angular_velocity(st: Any, r: Array, prograde: bool = True) -> Array:
    """Omega = dphi/dt of equatorial circular geodesics, BPT 1972 eq. (2.16).

    Omega = +- sqrt(M) / (r^{3/2} +- a sqrt(M)), upper sign prograde, written
    with |a| times sign(a) so that ``prograde`` keeps meaning a L_z > 0.
    Verified in ``tests/test_orbits_timelike.py`` by solving
    d/dr (g_tt + 2 Omega g_tphi + Omega^2 g_phiphi) = 0 at theta = pi/2.
    """
    mass, a = _mass_and_spin(st)
    s = _direction_sign(prograde)
    r = np.asarray(r, dtype=np.float64)
    return s * _spin_sign(a) * math.sqrt(mass) / (r**1.5 + s * abs(a) * math.sqrt(mass))


def circular_orbit_energy(st: Any, r: Array, prograde: bool = True) -> Array:
    """E per unit rest mass of an equatorial circular orbit, BPT 1972 eq. (2.12).

    E = (r^{3/2} - 2 M r^{1/2} +- a M^{1/2}) / [r^{3/4} (r^{3/2} - 3 M r^{1/2}
    +- 2 a M^{1/2})^{1/2}] with |a|, upper sign prograde; NaN below the
    photon orbit.
    """
    mass, a = _mass_and_spin(st)
    s = _direction_sign(prograde)
    r = np.asarray(r, dtype=np.float64)
    sq = np.sqrt(r)
    sm = math.sqrt(mass)
    num = r * sq - 2.0 * mass * sq + s * abs(a) * sm
    return num / (r**0.75 * np.sqrt(r * sq - 3.0 * mass * sq + 2.0 * s * abs(a) * sm))


def circular_orbit_angular_momentum(st: Any, r: Array, prograde: bool = True) -> Array:
    """L_z per unit rest mass of an equatorial circular orbit, BPT 1972 eq. (2.13).

    L = +- M^{1/2} (r^2 -+ 2 a M^{1/2} r^{1/2} + a^2) / [r^{3/4} (r^{3/2}
    - 3 M r^{1/2} +- 2 a M^{1/2})^{1/2}] with |a|, upper sign prograde, times
    sign(a) so that a L > 0 is prograde for either sign of a.
    """
    mass, a = _mass_and_spin(st)
    s = _direction_sign(prograde)
    r = np.asarray(r, dtype=np.float64)
    sq = np.sqrt(r)
    sm = math.sqrt(mass)
    num = sm * (r * r - 2.0 * s * abs(a) * sm * sq + a * a)
    den = r**0.75 * np.sqrt(r * sq - 3.0 * mass * sq + 2.0 * s * abs(a) * sm)
    return s * _spin_sign(a) * num / den


def timelike_radial_potential(st: Any, r: Array, energy: Array, lz: Array) -> Array:
    """R(r) = [E (r^2 + a^2) - a L]^2 - Delta [r^2 + (L - a E)^2], BPT eq. (2.9) with mu = 1, Q = 0."""
    mass, a = _mass_and_spin(st)
    r = np.asarray(r, dtype=np.float64)
    delta = r * r - 2.0 * mass * r + a * a
    return (energy * (r * r + a * a) - a * lz) ** 2 - delta * (r * r + (lz - a * energy) ** 2)


def isco_radius(st: Any, prograde: bool = True) -> float:
    """Marginally stable circular orbit radius, BPT 1972 eq. (2.21).

    Z1 = 1 + (1 - a^2/M^2)^{1/3} [(1 + a/M)^{1/3} + (1 - a/M)^{1/3}],
    Z2 = (3 a^2/M^2 + Z1^2)^{1/2},
    r_ms = M {3 + Z2 -+ [(3 - Z1)(3 + Z1 + 2 Z2)]^{1/2}}, upper sign prograde,
    with a -> |a|. Verified against :func:`isco_radius_numeric` to 1e-10 in
    the tests; a = 0 gives Z1 = Z2 = 3 and r_ms = 6M.
    """
    mass, a = _mass_and_spin(st)
    s = _direction_sign(prograde)
    chi = abs(a) / mass
    z1 = 1.0 + (1.0 - chi * chi) ** (1.0 / 3.0) * ((1.0 + chi) ** (1.0 / 3.0) + (1.0 - chi) ** (1.0 / 3.0))
    z2 = math.sqrt(3.0 * chi * chi + z1 * z1)
    return mass * (3.0 + z2 - s * math.sqrt((3.0 - z1) * (3.0 + z1 + 2.0 * z2)))


def marginal_stability_function(st: Any, r: Array, prograde: bool = True) -> Array:
    """R''(r) / E^2 along the circular-orbit family; zero at the ISCO.

    SymPy (``tests/test_orbits_timelike.py``) gives for :func:`timelike_radial_potential`
    R'' = 2 E^2 a^2 + 12 E^2 r^2 - 2 L^2 + 12 M r - 2 a^2 - 12 r^2, hence
    R''/E^2 = 12 r^2 + 2 a^2 - 2 (L/E)^2 + (12 M r - 12 r^2 - 2 a^2)/E^2
    with E(r), L(r) of BPT eqs. (2.12)-(2.13). Stable orbits have R'' < 0;
    dividing by E^2 keeps the function finite at the photon orbit
    (E diverges, L/E stays finite, 1/E^2 -> 0).
    """
    mass, a = _mass_and_spin(st)
    s = _direction_sign(prograde)
    r = np.asarray(r, dtype=np.float64)
    sq = np.sqrt(r)
    sm = math.sqrt(mass)
    abs_a = abs(a)
    e_num = r * sq - 2.0 * mass * sq + s * abs_a * sm
    inv_e2 = r * sq * (r * sq - 3.0 * mass * sq + 2.0 * s * abs_a * sm) / (e_num * e_num)
    xi_t = sm * (r * r - 2.0 * s * abs_a * sm * sq + a * a) / e_num  # |L / E|
    return 12.0 * r * r + 2.0 * a * a - 2.0 * xi_t * xi_t + (12.0 * mass * r - 12.0 * r * r - 2.0 * a * a) * inv_e2


def isco_radius_numeric(st: Any, prograde: bool = True) -> float:
    """ISCO from the zero of :func:`marginal_stability_function`, independent of eq. (2.21).

    Brackets [r_ph, 12 M]: positive at the photon orbit (every circular
    orbit between it and the ISCO is unstable), negative at 12 M (the ISCO
    never exceeds 9 M). Raises ``RuntimeError`` without that sign change.
    """
    mass, _ = _mass_and_spin(st)
    r_ph = equatorial_photon_orbit_radius(st, prograde)
    r_hi = _ISCO_BRACKET_UPPER * mass

    def f(r: float) -> float:
        return float(marginal_stability_function(st, r, prograde))

    f_lo, f_hi = f(r_ph), f(r_hi)
    if not (f_lo > 0.0 and f_hi < 0.0):
        raise RuntimeError(f"no ISCO sign change on [{r_ph}, {r_hi}]: f = ({f_lo}, {f_hi})")
    return float(brentq(f, r_ph, r_hi, xtol=_ROOT_XTOL * mass, rtol=_ROOT_RTOL))


__all__ = [
    "circular_orbit_angular_momentum",
    "circular_orbit_energy",
    "critical_impact_parameters",
    "equatorial_photon_orbit_radius",
    "equatorial_photon_orbit_radius_reference",
    "isco_radius",
    "isco_radius_numeric",
    "keplerian_angular_velocity",
    "marginal_stability_function",
    "radial_potential",
    "radial_potential_derivative",
    "shadow_curve",
    "spherical_orbit_constants",
    "spherical_orbit_radius_range",
    "theta_potential",
    "timelike_radial_potential",
]
