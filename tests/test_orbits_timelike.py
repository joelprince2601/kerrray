"""Tests for the timelike circular-orbit part of kerrray.photons.orbits.

Keplerian angular velocity, circular-orbit energy and angular momentum and
the ISCO. The SymPy tests derive Omega, E and L from the symbolic Kerr
metric (circular-orbit condition) and compare them with the BPT 1972 closed
forms used by the implementation; the ISCO closed form (BPT eq. 2.21) is
checked against an independent numerical location of the marginally stable
orbit. The reference 6M for Schwarzschild only *checks* computed values.
"""

from __future__ import annotations

import math
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
ISCO_TOL = 1e-10


def st(spin: float, mass: float = MASS) -> Spacetime:
    return Spacetime(mass=mass, spin=spin)


# --------------------------------------------------------------------------
# SymPy derivations from the metric
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def equatorial_metric():
    """g_tt, g_tphi, g_phiphi at theta = pi/2 from the symbolic Kerr metric, with positive M, r."""
    import sympy as sp

    from kerrray.geometry import symbolic as sym

    g, s = sym.symbolic_metric()
    M, r = sp.symbols("M r", positive=True)
    a = sp.symbols("a", real=True)
    sub = {s.mass: M, s.r: r, s.a: a, s.theta: sp.pi / 2}
    return sp, (M, r, a), g[0, 0].subs(sub), g[0, 3].subs(sub), g[3, 3].subs(sub)


def test_keplerian_angular_velocity_from_circular_orbit_condition(equatorial_metric) -> None:
    """Solve d/dr (g_tt + 2 Omega g_tphi + Omega^2 g_phiphi) = 0 and compare with BPT eq. 2.16."""
    sp, (M, r, a), gtt, gtp, gpp = equatorial_metric
    omega = sp.symbols("Omega", real=True)
    roots = sp.solve(sp.diff(gtt + 2 * omega * gtp + omega**2 * gpp, r), omega)
    assert len(roots) == 2
    bpt = {s: s * sp.sqrt(M) / (r ** sp.Rational(3, 2) + s * a * sp.sqrt(M)) for s in (1, -1)}
    for s, expr in bpt.items():
        assert any(sp.simplify(root - expr) == 0 for root in roots), s
    f_roots = sp.lambdify((M, r, a), roots, modules="math")
    for spin in SPINS:
        for prograde in (True, False):
            for rv in (2.5, 4.0, 9.0):
                got = float(ob.keplerian_angular_velocity(st(spin), rv, prograde))
                candidates = f_roots(MASS, rv, spin * MASS)
                same_sense = [c for c in candidates if math.copysign(1.0, c) == math.copysign(1.0, got)]
                assert len(same_sense) == 1
                assert got == pytest.approx(same_sense[0], rel=1e-13)
                assert (got > 0) == (prograde if spin >= 0 else not prograde)


def test_circular_orbit_energy_and_momentum_from_metric(equatorial_metric) -> None:
    """E = -(g_tt + Omega g_tphi) u^t and L = (g_tphi + Omega g_phiphi) u^t match BPT eqs. 2.12-2.13."""
    sp, (M, r, a), gtt, gtp, gpp = equatorial_metric
    for s in (1, -1):
        omega = s * sp.sqrt(M) / (r ** sp.Rational(3, 2) + s * a * sp.sqrt(M))
        u_t = 1 / sp.sqrt(-(gtt + 2 * omega * gtp + omega**2 * gpp))
        energy = -(gtt + omega * gtp) * u_t
        lz = (gtp + omega * gpp) * u_t
        root = sp.sqrt(r ** sp.Rational(3, 2) - 3 * M * sp.sqrt(r) + 2 * s * a * sp.sqrt(M))
        e_bpt = (r ** sp.Rational(3, 2) - 2 * M * sp.sqrt(r) + s * a * sp.sqrt(M)) / (r ** sp.Rational(3, 4) * root)
        l_bpt = s * sp.sqrt(M) * (r**2 - 2 * s * a * sp.sqrt(M) * sp.sqrt(r) + a**2) / (r ** sp.Rational(3, 4) * root)
        assert sp.simplify(energy**2 - e_bpt**2) == 0
        assert sp.simplify(lz**2 - l_bpt**2) == 0
        f = sp.lambdify((M, r, a), (energy, lz), modules="math")
        for spin in (0.0, 0.3, 0.95):
            for rv in (4.0, 6.0, 15.0):
                e_num, l_num = f(MASS, rv, spin * MASS)
                assert float(ob.circular_orbit_energy(st(spin), rv, s == 1)) == pytest.approx(e_num, rel=1e-13)
                assert float(ob.circular_orbit_angular_momentum(st(spin), rv, s == 1)) == pytest.approx(l_num, rel=1e-13)
        # the same E(r), L(r) satisfy R = 0 and R' = 0 of the timelike radial potential
        E, L = sp.symbols("E L", real=True)
        delta = r**2 - 2 * M * r + a**2
        big_r = (E * (r**2 + a**2) - a * L) ** 2 - delta * (r**2 + (L - a * E) ** 2)
        sub = {E: e_bpt, L: l_bpt}
        assert sp.simplify(big_r.subs(sub)) == 0
        assert sp.simplify(sp.diff(big_r, r).subs(sub)) == 0
        # second derivative used by marginal_stability_function
        d2 = sp.expand(sp.diff(big_r, r, 2))
        assert sp.simplify(d2 - (2 * E**2 * a**2 + 12 * E**2 * r**2 - 2 * L**2 + 12 * M * r - 2 * a**2 - 12 * r**2)) == 0


def test_marginal_stability_function_equals_second_derivative_over_e2() -> None:
    for spin in (0.0, 0.7, -0.6):
        for prograde in (True, False):
            for rv in (4.5, 6.0, 9.0):
                e = float(ob.circular_orbit_energy(st(spin), rv, prograde))
                lz = float(ob.circular_orbit_angular_momentum(st(spin), rv, prograde))
                a = spin * MASS
                d2 = 2 * e * e * a * a + 12 * e * e * rv * rv - 2 * lz * lz + 12 * MASS * rv - 2 * a * a - 12 * rv * rv
                assert float(ob.marginal_stability_function(st(spin), rv, prograde)) == pytest.approx(d2 / e**2, rel=1e-11)
                assert abs(float(ob.timelike_radial_potential(st(spin), rv, e, lz))) < 1e-11


# --------------------------------------------------------------------------
# ISCO and Keplerian angular velocity, numerically
# --------------------------------------------------------------------------


@pytest.mark.parametrize("spin", SPINS)
@pytest.mark.parametrize("prograde", [True, False])
def test_isco_closed_form_agrees_with_numerical_marginal_stability(spin: float, prograde: bool) -> None:
    s = st(spin)
    closed = ob.isco_radius(s, prograde)
    numeric = ob.isco_radius_numeric(s, prograde)
    assert abs(closed - numeric) < ISCO_TOL * MASS
    assert closed > ob.equatorial_photon_orbit_radius(s, prograde)


def test_isco_schwarzschild_is_6m_both_ways() -> None:
    s = st(0.0)
    for prograde in (True, False):
        assert abs(ob.isco_radius(s, prograde) - 6.0 * MASS) < ISCO_TOL
        assert abs(ob.isco_radius_numeric(s, prograde) - 6.0 * MASS) < ISCO_TOL


@pytest.mark.parametrize("spin", [0.0, 0.5, 0.95])
@pytest.mark.parametrize("prograde", [True, False])
def test_isco_is_the_minimum_of_the_circular_orbit_energy(spin: float, prograde: bool) -> None:
    s = st(spin)
    r_isco = ob.isco_radius(s, prograde)
    e_isco = float(ob.circular_orbit_energy(s, r_isco, prograde))
    h = 1e-3 * MASS
    assert float(ob.circular_orbit_energy(s, r_isco - h, prograde)) > e_isco
    assert float(ob.circular_orbit_energy(s, r_isco + h, prograde)) > e_isco
    assert 0.0 < e_isco < 1.0  # bound orbit


def test_isco_ordering_and_monotonicity() -> None:
    pro = [ob.isco_radius(st(spin), True) for spin in (0.0, 0.3, 0.7, 0.95, 0.999)]
    retro = [ob.isco_radius(st(spin), False) for spin in (0.0, 0.3, 0.7, 0.95, 0.999)]
    assert all(x > y for x, y in zip(pro, pro[1:]))
    assert all(x < y for x, y in zip(retro, retro[1:]))
    assert all(x < 6.0 * MASS < y for x, y in zip(pro[1:], retro[1:]))
    assert retro[-1] < 9.0 * MASS  # bounded by the extremal value


def test_isco_scales_with_mass() -> None:
    for prograde in (True, False):
        assert ob.isco_radius(st(0.4, mass=2.5), prograde) == pytest.approx(
            2.5 * ob.isco_radius(st(0.4), prograde), rel=1e-13
        )
        assert ob.isco_radius_numeric(st(0.4, mass=2.5), prograde) == pytest.approx(
            2.5 * ob.isco_radius_numeric(st(0.4), prograde), rel=1e-11
        )


def test_negative_spin_mirrors_positive_spin() -> None:
    plus, minus = st(0.6), st(-0.6)
    for prograde in (True, False):
        assert ob.isco_radius(plus, prograde) == ob.isco_radius(minus, prograde)
        assert float(ob.keplerian_angular_velocity(plus, 5.0, prograde)) == -float(
            ob.keplerian_angular_velocity(minus, 5.0, prograde)
        )
        assert float(ob.circular_orbit_angular_momentum(plus, 7.0, prograde)) == -float(
            ob.circular_orbit_angular_momentum(minus, 7.0, prograde)
        )
        assert float(ob.circular_orbit_energy(plus, 7.0, prograde)) == float(
            ob.circular_orbit_energy(minus, 7.0, prograde)
        )


def test_keplerian_angular_velocity_schwarzschild_and_vectorised() -> None:
    r = np.array([3.0, 6.0, 20.0])
    omega = ob.keplerian_angular_velocity(st(0.0), r, True)
    np.testing.assert_allclose(omega, np.sqrt(MASS / r**3), rtol=1e-14)
    np.testing.assert_allclose(ob.keplerian_angular_velocity(st(0.0), r, False), -omega, rtol=1e-14)
    assert ob.keplerian_angular_velocity(st(0.8), r, True).shape == (3,)
