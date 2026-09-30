"""Tests for the null-geodesic part of kerrray.photons.orbits.

The symbolic tests re-run the SymPy derivation of the spherical photon orbit
constants from the Carter radial potential and compare it with the published
forms; the numerical tests check the computed photon-orbit radii, critical
impact parameters and shadow curve against the BPT 1972 references and
against each other. No expected physical number is produced by the code
under test: the references (3M, 3 sqrt(3) M, BPT closed forms) only *check*
computed values. Timelike circular orbits (Omega, E, L, ISCO) are tested in
``tests/test_orbits_timelike.py``.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
import pytest

from kerrray.photons import orbits as ob

try:  # geometry role's Spacetime; the fallback keeps this file runnable before integration
    from kerrray.geometry.metric import Spacetime
except ImportError:  # pragma: no cover

    @dataclass(frozen=True)
    class Spacetime:  # type: ignore[no-redef]
        mass: float = 1.0
        spin: float = 0.0

        @property
        def a(self) -> float:
            return self.spin * self.mass


MASS = 1.0
SPINS = [0.0, 0.3, 0.7, 0.95, -0.6, 0.999]
NONZERO_SPINS = [s for s in SPINS if s != 0.0]
CLOSED_FORM_TOL = 1e-12


def st(spin: float, mass: float = MASS) -> Spacetime:
    return Spacetime(mass=mass, spin=spin)


# --------------------------------------------------------------------------
# SymPy derivation of xi(r), eta(r)
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def spherical_solution():
    """Solve R = 0, dR/dr = 0 for (xi, eta) from the Carter radial potential."""
    import sympy as sp

    r, M, a, xi, eta = sp.symbols("r M a xi eta", real=True)
    delta = r**2 - 2 * M * r + a**2
    big_r = ((r**2 + a**2) - a * xi) ** 2 - delta * (eta + (xi - a) ** 2)
    sols = sp.solve([big_r, sp.diff(big_r, r)], [xi, eta], dict=True)
    return sp, (r, M, a, xi, eta), big_r, sols


def test_spherical_orbit_constants_are_the_physical_branch(spherical_solution) -> None:
    sp, (r, M, a, xi, eta), _, sols = spherical_solution
    assert len(sols) == 2
    xi_impl = -(r**2 * (r - 3 * M) + a**2 * (r + M)) / (a * (r - M))
    eta_impl = r**3 * (4 * M * a**2 - r * (r - 3 * M) ** 2) / (a**2 * (r - M) ** 2)
    physical = [s for s in sols if sp.simplify(s[xi] - xi_impl) == 0 and sp.simplify(s[eta] - eta_impl) == 0]
    assert len(physical) == 1
    (other,) = [s for s in sols if s is not physical[0]]
    assert sp.simplify(other[xi] - (r**2 + a**2) / a) == 0
    assert sp.simplify(other[eta] + r**4 / a**2) == 0  # eta < 0: unphysical branch


def test_spherical_orbit_constants_match_published_forms(spherical_solution) -> None:
    """The derived branch equals Bardeen 1973, Johannsen & Psaltis 2010 and Cunha & Herdeiro 2018."""
    sp, (r, M, a, xi, eta), _, sols = spherical_solution
    delta = r**2 - 2 * M * r + a**2
    published = {
        "Bardeen 1973 / Chandrasekhar 1983 ch. 7": (
            (M * (r**2 - a**2) - r * delta) / (a * (r - M)),
            r**3 * (4 * M * delta - r * (r - M) ** 2) / (a**2 * (r - M) ** 2),
        ),
        "Johannsen & Psaltis 2010 eqs. 8-9": (
            -(r**3 - 3 * M * r**2 + a**2 * r + a**2 * M) / (a * (r - M)),
            -(r**3) * (r**3 - 6 * M * r**2 + 9 * M**2 * r - 4 * a**2 * M) / (a**2 * (r - M) ** 2),
        ),
        "Cunha & Herdeiro 2018 eqs. 10-11": (
            (r**2 * (3 * M - r) - a**2 * (r + M)) / (a * (r - M)),
            r**3 * (4 * M * a**2 - r * (r - 3 * M) ** 2) / (a**2 * (r - M) ** 2),
        ),
    }
    for name, (xi_ref, eta_ref) in published.items():
        matches = [s for s in sols if sp.simplify(s[xi] - xi_ref) == 0 and sp.simplify(s[eta] - eta_ref) == 0]
        assert len(matches) == 1, name


def test_spherical_orbit_identities(spherical_solution) -> None:
    """eta + (xi - a)^2 = 4 r^2 Delta/(r - M)^2 and the sign form of xi - a used for b_c."""
    sp, (r, M, a, xi, eta), _, sols = spherical_solution
    delta = r**2 - 2 * M * r + a**2
    phys = [s for s in sols if sp.simplify(s[eta] + r**4 / a**2) != 0][0]
    assert sp.simplify(phys[eta] + (phys[xi] - a) ** 2 - 4 * r**2 * delta / (r - M) ** 2) == 0
    assert sp.simplify(phys[xi] - a + r * (r**2 - 3 * M * r + 2 * a**2) / (a * (r - M))) == 0
    # eta = 0  <=>  r (r - 3M)^2 = 4 M a^2 (the cubic behind equatorial_photon_orbit_radius)
    assert sp.simplify(sp.numer(sp.together(phys[eta])) / r**3 + (r * (r - 3 * M) ** 2 - 4 * M * a**2)) == 0


# --------------------------------------------------------------------------
# Spherical orbits, photon-orbit radii, critical impact parameters
# --------------------------------------------------------------------------


@pytest.mark.parametrize("spin", NONZERO_SPINS)
def test_spherical_orbit_constants_satisfy_r_and_dr(spin: float) -> None:
    s = st(spin)
    r_lo, r_hi = ob.spherical_orbit_radius_range(s)
    r = np.linspace(r_lo, r_hi, 25)
    xi, eta = ob.spherical_orbit_constants(s, r)
    scale = r**4
    assert np.all(np.abs(ob.radial_potential(s, r, xi, eta)) < 1e-12 * scale)
    assert np.all(np.abs(ob.radial_potential_derivative(s, r, xi, eta)) < 1e-12 * scale)
    assert np.all(eta >= -1e-12 * scale)  # physical branch on its range
    assert xi.shape == (25,) and eta.shape == (25,)


def test_spherical_orbit_constants_reject_zero_spin() -> None:
    with pytest.raises(ValueError):
        ob.spherical_orbit_constants(st(0.0), 3.0)


@pytest.mark.parametrize("spin", SPINS)
@pytest.mark.parametrize("prograde", [True, False])
def test_equatorial_photon_orbit_radius_matches_bpt_2_18(spin: float, prograde: bool) -> None:
    s = st(spin)
    computed = ob.equatorial_photon_orbit_radius(s, prograde)
    reference = ob.equatorial_photon_orbit_radius_reference(s, prograde)
    assert abs(computed - reference) < CLOSED_FORM_TOL * MASS
    if spin != 0.0:
        _, eta = ob.spherical_orbit_constants(s, computed)
        assert abs(float(eta)) < 1e-10 * computed**4


def test_schwarzschild_photon_sphere_and_critical_impact_parameter() -> None:
    s = st(0.0)
    r_pro, r_retro = ob.spherical_orbit_radius_range(s)
    assert abs(r_pro - 3.0 * MASS) < CLOSED_FORM_TOL and abs(r_retro - 3.0 * MASS) < CLOSED_FORM_TOL
    b_pro, b_retro = ob.critical_impact_parameters(s)
    reference = 3.0 * math.sqrt(3.0) * MASS
    assert abs(b_pro - reference) < CLOSED_FORM_TOL
    assert abs(b_retro - reference) < CLOSED_FORM_TOL


def test_photon_orbit_radius_scales_with_mass() -> None:
    for prograde in (True, False):
        assert ob.equatorial_photon_orbit_radius(st(0.4, mass=2.5), prograde) == pytest.approx(
            2.5 * ob.equatorial_photon_orbit_radius(st(0.4), prograde), rel=1e-13
        )
    b_scaled = ob.critical_impact_parameters(st(0.4, mass=2.5))
    b_unit = ob.critical_impact_parameters(st(0.4))
    assert b_scaled[0] == pytest.approx(2.5 * b_unit[0], rel=1e-13)
    assert b_scaled[1] == pytest.approx(2.5 * b_unit[1], rel=1e-13)


@pytest.mark.parametrize("spin", NONZERO_SPINS)
def test_critical_impact_parameters_match_spherical_orbit_xi(spin: float) -> None:
    s = st(spin)
    b_pro, b_retro = ob.critical_impact_parameters(s)
    for prograde, b in ((True, b_pro), (False, b_retro)):
        r_ph = ob.equatorial_photon_orbit_radius(s, prograde)
        xi, _ = ob.spherical_orbit_constants(s, r_ph)
        assert abs(float(xi)) == pytest.approx(b, rel=1e-10)
        assert (spin * float(xi) > 0) == prograde  # prograde <=> a L_z > 0
    assert b_pro < 3.0 * math.sqrt(3.0) * MASS < b_retro


def test_critical_impact_parameters_monotone_in_spin() -> None:
    values = [ob.critical_impact_parameters(st(spin)) for spin in (0.0, 0.3, 0.7, 0.95)]
    assert all(v[0] > w[0] for v, w in zip(values, values[1:]))
    assert all(v[1] < w[1] for v, w in zip(values, values[1:]))


def test_negative_spin_mirrors_positive_spin() -> None:
    plus, minus = st(0.6), st(-0.6)
    for prograde in (True, False):
        assert ob.equatorial_photon_orbit_radius(plus, prograde) == ob.equatorial_photon_orbit_radius(minus, prograde)
    assert ob.critical_impact_parameters(plus) == ob.critical_impact_parameters(minus)
    r = np.linspace(2.5, 3.5, 7)
    xi_plus, eta_plus = ob.spherical_orbit_constants(plus, r)
    xi_minus, eta_minus = ob.spherical_orbit_constants(minus, r)
    np.testing.assert_allclose(xi_minus, -xi_plus, rtol=0, atol=0)
    np.testing.assert_allclose(eta_minus, eta_plus, rtol=0, atol=0)


# --------------------------------------------------------------------------
# Shadow curve
# --------------------------------------------------------------------------


def test_shadow_curve_schwarzschild_is_circle_of_critical_radius() -> None:
    with warnings.catch_warnings(), np.errstate(all="raise"):
        warnings.simplefilter("error")
        alpha, beta = ob.shadow_curve(st(0.0), 60.0, 100)
    assert alpha.shape == beta.shape == (100,)
    radius = np.hypot(alpha, beta)
    reference = 3.0 * math.sqrt(3.0) * MASS
    assert np.all(np.abs(radius - reference) < CLOSED_FORM_TOL)
    assert np.all(np.diff(np.arctan2(beta, alpha)) > 0)


@pytest.mark.parametrize("spin, inclination", [(0.9, 60.0), (0.9, 90.0), (0.3, 17.0), (-0.9, 45.0), (0.999, 89.0)])
def test_shadow_curve_is_closed_ordered_and_on_the_orbit_family(spin: float, inclination: float) -> None:
    s = st(spin)
    n = 360
    alpha, beta = ob.shadow_curve(s, inclination, n)
    assert alpha.shape == beta.shape == (n,)
    assert np.all(np.isfinite(alpha)) and np.all(np.isfinite(beta))
    angles = np.arctan2(beta, alpha)
    assert np.all(np.diff(angles) >= 0)
    # symmetric about beta = 0
    upper = {(round(float(x), 9), round(float(y), 9)) for x, y in zip(alpha, beta) if y > 0}
    lower = {(round(float(x), 9), round(float(-y), 9)) for x, y in zip(alpha, beta) if y < 0}
    assert upper == lower
    # every point corresponds to a spherical photon orbit: recover (xi, eta) and check it lies on the family
    inc = math.radians(inclination)
    a = spin * MASS
    xi = -alpha * math.sin(inc)
    cos2 = 0.0 if inclination == 90.0 else math.cos(inc) ** 2
    cot2 = cos2 / math.sin(inc) ** 2
    eta = beta**2 - a * a * cos2 + xi * xi * cot2
    r_lo, r_hi = ob.spherical_orbit_radius_range(s)
    from scipy.optimize import brentq

    for k in range(0, n, 23):

        def f(r: float, target: float = float(xi[k])) -> float:
            return float(ob.spherical_orbit_constants(s, r)[0]) - target

        f_lo, f_hi = f(r_lo), f(r_hi)
        if f_lo * f_hi > 0.0:  # xi[k] sits at an endpoint up to rounding
            r_star = r_lo if abs(f_lo) < abs(f_hi) else r_hi
            assert min(abs(f_lo), abs(f_hi)) < 1e-9
        else:
            r_star = brentq(f, r_lo, r_hi, xtol=1e-14)
        _, eta_r = ob.spherical_orbit_constants(s, r_star)
        assert float(eta_r) == pytest.approx(eta[k], abs=1e-8 * max(1.0, abs(eta[k])))
    # alpha^2 + beta^2 = xi^2 + eta + a^2 cos^2 i identically
    np.testing.assert_allclose(alpha**2 + beta**2, xi**2 + eta + a * a * cos2, rtol=1e-10)


def test_shadow_curve_equatorial_extent_is_the_critical_impact_parameters() -> None:
    s = st(0.9)
    alpha, beta = ob.shadow_curve(s, 90.0, 2000)
    b_pro, b_retro = ob.critical_impact_parameters(s)
    assert alpha.min() == pytest.approx(-b_pro, abs=1e-9)
    assert alpha.max() == pytest.approx(b_retro, abs=1e-9)
    alpha_m, _ = ob.shadow_curve(st(-0.9), 90.0, 2000)
    np.testing.assert_allclose(np.sort(alpha_m), np.sort(-alpha), atol=1e-12)


def test_shadow_curve_supplementary_inclinations_coincide() -> None:
    a1, b1 = ob.shadow_curve(st(0.7), 40.0, 200)
    a2, b2 = ob.shadow_curve(st(0.7), 140.0, 200)
    np.testing.assert_allclose(a1, a2, atol=1e-12)
    np.testing.assert_allclose(b1, b2, atol=1e-12)


def test_shadow_curve_small_inclination_tends_to_axis_circle() -> None:
    """Near the axis the shadow tends to the circle of radius sqrt(eta(r_0) + a^2), xi(r_0) = 0.

    The deviation is first order in i: on the curve |xi| <= sqrt(eta + a^2) tan(i),
    and the radius sqrt(xi^2 + eta(r) + a^2) varies through eta'(r_0) dr ~ xi.
    """
    s = st(0.9)
    r_lo, r_hi = ob.spherical_orbit_radius_range(s)
    from scipy.optimize import brentq

    r_0 = brentq(lambda r: float(ob.spherical_orbit_constants(s, r)[0]), r_lo, r_hi, xtol=1e-14)
    _, eta_0 = ob.spherical_orbit_constants(s, r_0)
    expected = math.sqrt(float(eta_0) + (0.9 * MASS) ** 2)
    deviations = []
    for inc_deg in (1e-2, 1e-3):
        alpha, beta = ob.shadow_curve(s, inc_deg, 400)
        deviation = float(np.max(np.abs(np.hypot(alpha, beta) - expected))) / expected
        assert deviation < 10.0 * math.radians(inc_deg)
        deviations.append(deviation)
    assert deviations[1] < deviations[0]


class _BadSpacetime:
    mass = 1.0
    a = 1.0  # extremal: |a| < M violated


def test_shadow_curve_rejects_invalid_arguments() -> None:
    with pytest.raises(ValueError):
        ob.shadow_curve(st(0.5), 0.0)
    with pytest.raises(ValueError):
        ob.shadow_curve(st(0.5), 180.0)
    with pytest.raises(ValueError):
        ob.shadow_curve(st(0.5), 60.0, n=2)
    with pytest.raises(ValueError):
        ob.critical_impact_parameters(_BadSpacetime())
