"""Frame dragging of photon geodesics (PROJECT.md section 14; EXP-005).

Frame dragging in Kerr spacetime is the coupling ``g_tphi = -2 M a r
sin^2(theta) / Sigma`` between the time and azimuth directions
(docs/equations.md section 2). Two families of rays isolate it:

**Equatorial family.** The same photon (``theta = pi/2``, ``E``, ``L_z = +b E``,
``Q = 0``, same launch point) is integrated for ``a < 0``, ``a = 0`` and
``a > 0``. Under ``a -> -a`` the metric components ``g_tt``, ``g_rr``,
``g_thth`` and ``g_phph`` are unchanged (they depend on ``a^2`` only) and
``g_tphi`` flips sign, so the difference ``Delta phi(+a) - Delta phi(-a)``
is odd in ``a`` and comes from ``g_tphi`` alone, while
``[Delta phi(+a) + Delta phi(-a)]/2 - Delta phi(0)`` collects the even
(``a^2``) quadrupole-like effects. With ``L_z > 0`` the ray is prograde for
``a > 0`` and retrograde for ``a < 0``.

Leading order (derivation in docs/experiments_kerr.md section 2, from the
separated equations of Carter 1968 and Bardeen, Press and Teukolsky 1972,
eqs. 2.9-2.10 with ``mu = 0``): for ``b = L_z/E > 0`` ::

    Delta phi(a) = pi + 4 M / b - 4 a M / b^2 + O(M^2/b^2, a^2, a M^2/b^3)
    Delta phi(+a) - Delta phi(-a) = -8 a M / b^2

in agreement with the weak-deflection Kerr expansion
``alpha = 4M/b -+ 4aM/b^2 + ...`` (upper sign prograde; Sereno and De Luca
2006, Phys. Rev. D 74, 123009; Edery and Godin 2006, Gen. Rel. Grav. 38,
1715). A prograde photon (``a > 0``, ``L_z > 0``) is deflected *less* than
the retrograde one at the same ``|b|``, and its capture threshold is lower
(docs/derivations.md section 6). When one ray of a ``+-a`` pair is captured
(for example ``b = 6 M`` at ``|a| = 0.9``, where the retrograde threshold is
larger than ``6 M``) the azimuths are not comparable and the pair reports the
two outcomes with ``nan`` asymmetries.

**Polar family.** A photon with exactly ``L_z = 0`` launched parallel to the
spin axis at distance ``b`` (``Q = b^2 E^2``). Then ``dphi/dt = g^{tphi} /
g^{tt} = 2 M a r / A = omega``, the angular velocity of the zero-angular-
momentum observers (Bardeen, Press and Teukolsky 1972, section II): its
azimuth is *pure* frame dragging, zero for ``a = 0`` and exactly odd in
``a``; leading order ``Delta phi_polar = 4 a M / b^2``.

Both leading-order formulas are approximations (PROJECT.md section 43) --
WHAT: first-order weak-field azimuths; WHY: they fix the sign and the
``1/b^2`` scaling; LIMITATION: ``b >> M`` and ``r_0 -> infinity`` (the ratio
of measured to estimated value is reported by the experiment). The exact
reference is :func:`azimuth_quadrature` (no expansion). Boyer-Lindquist
limitation of the polar family: a meridional ray bent by ``~4M/b`` crosses
the axis ``~b^2/4M`` beyond the hole, where the integrator stops it as
``OUT_OF_DOMAIN``; the default polar ``b`` is therefore 1.25 times
:func:`polar_impact_parameter_bound` ``= 2 sqrt(M r_escape)``, and the run
reports the actual outcome. Derivations: docs/experiments_kerr.md section 2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import quad
from scipy.optimize import brentq

from kerrray.geodesics import IntegratorOptions, TerminationOptions, integrate, photon_from_constants
from kerrray.geometry import Spacetime, kerr, outer_horizon
from kerrray.photons import (
    TerminationState,
    Trajectory,
    azimuthal_winding,
    closest_approach,
    deflection_angle,
)
from kerrray.photons.orbits import radial_potential

__all__ = [
    "QUADRATURE_TOLERANCE",
    "FrameDraggingResult",
    "RaySummary",
    "SpinPair",
    "azimuth_quadrature",
    "equatorial_ray",
    "frame_dragging_experiment",
    "leading_order_asymmetry",
    "leading_order_polar_drag",
    "polar_impact_parameter_bound",
    "polar_ray",
    "summarise_ray",
    "turning_point_radius",
]

QUADRATURE_TOLERANCE = 1e-13
"""Absolute and relative tolerance passed to ``scipy.integrate.quad``."""

_ROOT_IMAG_TOLERANCE = 1e-9
"""A quartic root counts as real when ``|imag| <= tol * max(1, |real|)``."""


# --------------------------------------------------------------------------
# Initial states
# --------------------------------------------------------------------------


def equatorial_ray(st: Spacetime, r0: float, b: float, *, E: float = 1.0) -> NDArray[np.float64]:
    """Inward equatorial photon at ``r0`` with ``L_z = +b E`` (sign fixed, not by ``prograde``).

    Identical constants of motion for every spin; the ray is prograde when
    ``a > 0`` and retrograde when ``a < 0`` (module docstring).
    """
    if not b > 0.0:
        raise ValueError(f"b must be > 0, got {b!r}")
    return photon_from_constants(st, r0, 0.5 * math.pi, E, b * E, 0.0, sign_r=-1, sign_theta=1)


def polar_ray(st: Spacetime, r0: float, b: float, *, E: float = 1.0) -> NDArray[np.float64]:
    """Photon with ``L_z = 0`` launched parallel to the spin axis at distance ``b``.

    Constants ``(E, L_z = 0, Q = b^2 E^2)`` at ``theta_0 = pi - arcsin(b / r0)``
    (below the equatorial plane, ``x = b``, ``z = -sqrt(r0^2 - b^2)`` in the
    flat-space picture), moving inward (``p_r < 0``) and towards the north
    pole (``p_theta < 0``). Requires ``0 < b < r0``.
    """
    if not 0.0 < b < r0:
        raise ValueError(f"need 0 < b < r0, got b={b!r}, r0={r0!r}")
    theta0 = math.pi - math.asin(b / r0)
    return photon_from_constants(st, r0, theta0, E, 0.0, (b * E) ** 2, sign_r=-1, sign_theta=-1)


# --------------------------------------------------------------------------
# Exact reference: quadrature of the separated radial and azimuthal equations
# --------------------------------------------------------------------------


def turning_point_radius(st: Spacetime, xi: float, eta: float) -> float | None:
    """Largest root of ``R(r)`` outside the outer horizon, or ``None`` if there is none.

    ``R(r) = [(r^2 + a^2) - a xi]^2 - Delta [eta + (xi - a)^2]`` is the quartic
    ``r^4 + (2c - K) r^2 + 2 M K r + (c^2 - a^2 K)`` with ``c = a^2 - a xi`` and
    ``K = eta + (xi - a)^2``; its roots come from ``numpy.roots`` and the
    selected one is polished with ``brentq``. ``None`` means the photon has
    no radial turning point outside the horizon (it is captured).
    """
    mass, a = st.mass, st.a
    k = eta + (xi - a) ** 2
    c = a * a - a * xi
    roots = np.roots([1.0, 0.0, 2.0 * c - k, 2.0 * mass * k, c * c - a * a * k])
    r_plus = outer_horizon(st)
    real = [
        float(z.real)
        for z in roots
        if abs(z.imag) <= _ROOT_IMAG_TOLERANCE * max(1.0, abs(z.real)) and z.real > r_plus
    ]
    if not real:
        return None
    r_t = max(real)

    def big_r(r: float) -> float:
        return float(radial_potential(st, r, xi, eta))

    lo, hi = r_t * (1.0 - 1e-6), r_t * (1.0 + 1e-6)
    if big_r(lo) * big_r(hi) < 0.0:
        r_t = float(brentq(big_r, lo, hi, xtol=1e-15, rtol=1e-15))
    return r_t


def azimuth_quadrature(st: Spacetime, xi: float, eta: float, r_start: float, r_end: float) -> float:
    """Azimuth swept by a photon from ``r_start`` inward to its turning point and out to ``r_end``.

    Integrates ``dphi/dr = [xi - a + a P/Delta] / sqrt(R)`` (module docstring;
    ``E = 1``, so ``xi = L_z`` and ``eta = Q``) on both legs with the
    substitution ``r = r_t + s^2`` that removes the inverse-square-root
    singularity at the turning point ``r_t``. Raises ``ValueError`` when the
    photon has no turning point (captured) or a radius lies below it.
    """
    r_t = turning_point_radius(st, xi, eta)
    if r_t is None:
        raise ValueError("no radial turning point outside the horizon: the photon is captured")
    if r_start < r_t or r_end < r_t:
        raise ValueError(f"r_start and r_end must be >= the turning point {r_t!r}")
    mass, a = st.mass, st.a

    def dphi_dr(r: float) -> float:
        dl = r * r - 2.0 * mass * r + a * a
        p = (r * r + a * a) - a * xi
        return ((xi - a) + a * p / dl) / math.sqrt(max(float(radial_potential(st, r, xi, eta)), 0.0))

    def leg(r_far: float) -> float:
        s_max = math.sqrt(r_far - r_t)
        if s_max == 0.0:
            return 0.0
        value, _ = quad(
            lambda s: dphi_dr(r_t + s * s) * 2.0 * s,
            0.0,
            s_max,
            limit=500,
            epsabs=QUADRATURE_TOLERANCE,
            epsrel=QUADRATURE_TOLERANCE,
        )
        return float(value)

    return leg(r_start) + leg(r_end)


# --------------------------------------------------------------------------
# Leading-order estimates (approximations; module docstring)
# --------------------------------------------------------------------------


def leading_order_asymmetry(st: Spacetime, b: float) -> float:
    """``-8 a M / b^2``: first-order estimate of ``Delta phi(+a) - Delta phi(-a)`` (``L_z > 0``)."""
    return -8.0 * abs(st.a) * st.mass / (b * b)


def leading_order_polar_drag(st: Spacetime, b: float) -> float:
    """``4 a M / b^2``: first-order estimate of the azimuth of the ``L_z = 0`` polar ray."""
    return 4.0 * st.a * st.mass / (b * b)


def polar_impact_parameter_bound(st: Spacetime, r_escape: float) -> float:
    """``2 sqrt(M r_escape)``: smallest polar ``b`` whose axis crossing lies beyond ``r_escape``.

    Weak-field estimate (deflection ``4M/b``, crossing at ``b^2 / 4M``); the
    driver uses it only to warn, the run itself reports the real outcome.
    """
    return 2.0 * math.sqrt(st.mass * r_escape)


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RaySummary:
    """Measured quantities of one integrated ray (``nan`` where not applicable)."""

    spin: float
    family: str
    impact_parameter: float
    lz: float
    carter: float
    state: str
    delta_phi: float
    turns: float
    closest_approach: float
    deflection: float
    delta_phi_quadrature: float
    quadrature_difference: float
    n_steps: int
    n_rejected: int
    runtime_s: float
    max_null_error: float
    max_energy_drift: float
    max_lz_drift: float
    max_carter_drift: float
    trajectory: Trajectory = field(repr=False, compare=False)

    def as_row(self) -> dict[str, Any]:
        """Plain mapping of the scalar fields (JSON-serialisable)."""
        return {
            key: value for key, value in self.__dict__.items() if key != "trajectory"
        }


@dataclass(frozen=True)
class SpinPair:
    """Azimuth asymmetry between ``+spin`` and ``-spin`` for both families.

    ``odd_part`` and ``even_part`` are ``nan`` unless the equatorial rays of the
    pair (and the ``a = 0`` ray, for ``even_part``) all escaped;
    ``state_plus``/``state_minus`` give the outcomes.
    """

    spin: float
    state_plus: str
    state_minus: str
    delta_phi_plus: float
    delta_phi_minus: float
    delta_phi_zero: float
    odd_part: float
    even_part: float
    estimate: float
    polar_plus: float
    polar_minus: float
    polar_zero: float
    polar_antisymmetry_residual: float
    polar_estimate: float

    def as_row(self) -> dict[str, Any]:
        """Plain mapping of the fields."""
        return dict(self.__dict__)


@dataclass(frozen=True)
class FrameDraggingResult:
    """Outcome of :func:`frame_dragging_experiment`."""

    spins: tuple[float, ...]
    impact_parameter: float
    polar_impact_parameter: float
    launch_radius: float
    energy: float
    equatorial: tuple[RaySummary, ...]
    polar: tuple[RaySummary, ...]
    pairs: tuple[SpinPair, ...]

    @property
    def all_escaped(self) -> bool:
        """True when every ray of both families ended ``ESCAPED``."""
        return all(s.state == TerminationState.ESCAPED.name for s in (*self.equatorial, *self.polar))


def summarise_ray(traj: Trajectory, spin: float, family: str, b: float) -> RaySummary:
    """Measure a trajectory: azimuth, closest approach, deflection, quadrature reference, drifts."""
    y0 = traj.y0
    lz = float(y0[7])
    e0 = float(-y0[4])
    q0 = float(traj.diagnostics.carter[0])
    escaped = traj.state == TerminationState.ESCAPED
    dphi = azimuthal_winding(traj)
    deflection = math.nan
    if escaped and family == "equatorial":
        deflection = deflection_angle(traj)
    dphi_quad = math.nan
    if escaped:
        r_t = turning_point_radius(traj.spacetime, lz / e0, q0 / (e0 * e0))
        if r_t is not None:
            dphi_quad = azimuth_quadrature(
                traj.spacetime, lz / e0, q0 / (e0 * e0), float(y0[1]), float(traj.y_end[1])
            )
    diag = traj.diagnostics
    return RaySummary(
        spin=spin,
        family=family,
        impact_parameter=b,
        lz=lz,
        carter=q0,
        state=traj.state.name,
        delta_phi=dphi,
        turns=dphi / (2.0 * math.pi),
        closest_approach=closest_approach(traj),
        deflection=deflection,
        delta_phi_quadrature=dphi_quad,
        quadrature_difference=dphi - dphi_quad if math.isfinite(dphi_quad) else math.nan,
        n_steps=traj.n_steps,
        n_rejected=traj.n_rejected,
        runtime_s=traj.runtime_s,
        max_null_error=diag.max_null_error,
        max_energy_drift=diag.max_energy_drift,
        max_lz_drift=diag.max_lz_drift,
        max_carter_drift=diag.max_carter_drift,
        trajectory=traj,
    )


def _lookup(rays: tuple[RaySummary, ...], spin: float) -> RaySummary | None:
    for ray in rays:
        if ray.spin == spin:
            return ray
    return None


def _pairs(
    st_mass: float,
    spins: tuple[float, ...],
    equatorial: tuple[RaySummary, ...],
    polar: tuple[RaySummary, ...],
    b: float,
    b_polar: float,
) -> tuple[SpinPair, ...]:
    pairs = []
    for spin in sorted({abs(s) for s in spins if s != 0.0}):
        plus, minus = _lookup(equatorial, spin), _lookup(equatorial, -spin)
        if plus is None or minus is None:
            continue
        zero = _lookup(equatorial, 0.0)
        p_plus, p_minus, p_zero = _lookup(polar, spin), _lookup(polar, -spin), _lookup(polar, 0.0)
        esc = TerminationState.ESCAPED.name
        dphi0 = zero.delta_phi if zero is not None and zero.state == esc else math.nan
        both = plus.state == esc and minus.state == esc
        st = kerr(st_mass, spin)
        pairs.append(
            SpinPair(
                spin=spin,
                state_plus=plus.state,
                state_minus=minus.state,
                delta_phi_plus=plus.delta_phi,
                delta_phi_minus=minus.delta_phi,
                delta_phi_zero=zero.delta_phi if zero is not None else math.nan,
                odd_part=plus.delta_phi - minus.delta_phi if both else math.nan,
                even_part=0.5 * (plus.delta_phi + minus.delta_phi) - dphi0 if both else math.nan,
                estimate=leading_order_asymmetry(st, b),
                polar_plus=p_plus.delta_phi if p_plus is not None else math.nan,
                polar_minus=p_minus.delta_phi if p_minus is not None else math.nan,
                polar_zero=p_zero.delta_phi if p_zero is not None else math.nan,
                polar_antisymmetry_residual=(
                    p_plus.delta_phi + p_minus.delta_phi
                    if p_plus is not None and p_minus is not None
                    else math.nan
                ),
                polar_estimate=leading_order_polar_drag(st, b_polar),
            )
        )
    return tuple(pairs)


def frame_dragging_experiment(
    spins: Any,
    b: float,
    *,
    integ: IntegratorOptions,
    term: TerminationOptions,
    r0: float,
    polar_b: float | None = None,
    E: float = 1.0,
    mass: float = 1.0,
) -> FrameDraggingResult:
    """Integrate the equatorial and polar families for every spin and pair ``+-a``.

    Args:
        spins: Dimensionless spins, e.g. ``[-0.9, 0.0, 0.9]``.
        b: Impact parameter ``|L_z| / E`` of the equatorial family (units of M).
        integ, term: Integrator and termination options shared by every ray.
        r0: Launch radius (units of M).
        polar_b: Distance from the axis of the polar family; defaults to
            ``1.25 * polar_impact_parameter_bound(st, term.escape_radius)``.
        E: Photon energy at infinity.
        mass: Black-hole mass.

    Returns:
        A :class:`FrameDraggingResult` with every trajectory recorded.
    """
    spin_list = tuple(float(s) for s in spins)
    if polar_b is None:
        polar_b = 1.25 * polar_impact_parameter_bound(kerr(mass, 0.0), term.escape_radius)
    equatorial = []
    polar = []
    for spin in spin_list:
        st = kerr(mass, spin)
        traj = integrate(st, equatorial_ray(st, r0, b, E=E), integ, term, record=True)
        equatorial.append(summarise_ray(traj, spin, "equatorial", b))
        traj_p = integrate(st, polar_ray(st, r0, polar_b, E=E), integ, term, record=True)
        polar.append(summarise_ray(traj_p, spin, "polar", polar_b))
    eq_t, po_t = tuple(equatorial), tuple(polar)
    return FrameDraggingResult(
        spins=spin_list,
        impact_parameter=b,
        polar_impact_parameter=polar_b,
        launch_radius=r0,
        energy=E,
        equatorial=eq_t,
        polar=po_t,
        pairs=_pairs(mass, spin_list, eq_t, po_t, b, polar_b),
    )
