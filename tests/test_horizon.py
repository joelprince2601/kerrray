"""Horizon and ergosphere tests (PROJECT.md sections 6, 7 and 41)."""

from __future__ import annotations

import numpy as np
import pytest

from kerrray.geometry import (
    delta,
    ergosphere_radius,
    horizon_radii,
    inside_horizon,
    kerr,
    metric_components,
    outer_horizon,
    schwarzschild,
)

SPINS = (0.0, 0.3, -0.5, 0.9, 0.998)


@pytest.mark.parametrize("spin", SPINS)
@pytest.mark.parametrize("mass", [1.0, 3.0])
def test_delta_vanishes_at_both_horizons(mass: float, spin: float) -> None:
    st = kerr(mass, spin)
    r_plus, r_minus = horizon_radii(st)
    np.testing.assert_allclose(delta(st, r_plus), 0.0, atol=1e-12 * mass**2)
    np.testing.assert_allclose(delta(st, r_minus), 0.0, atol=1e-12 * mass**2)
    # Delta > 0 strictly outside the outer horizon.
    assert np.all(delta(st, np.linspace(r_plus + 1e-9, 50 * mass, 100)) > 0.0)


@pytest.mark.parametrize("spin", SPINS)
def test_horizon_pair_relations(spin: float) -> None:
    st = kerr(1.0, spin)
    r_plus, r_minus = horizon_radii(st)
    m, a = st.mass, st.a
    assert r_minus <= r_plus
    assert m < r_plus <= 2 * m or spin == 0.0
    np.testing.assert_allclose(r_plus + r_minus, 2 * m, rtol=1e-15)
    np.testing.assert_allclose(r_plus * r_minus, a * a, atol=1e-15)
    assert outer_horizon(st) == r_plus
    assert isinstance(r_plus, float) and isinstance(r_minus, float)


def test_schwarzschild_limit() -> None:
    st = schwarzschild(1.0)
    assert horizon_radii(st) == (2.0, 0.0)
    theta = np.linspace(0.0, np.pi, 7)
    np.testing.assert_array_equal(ergosphere_radius(st, theta), 2.0)
    tiny = kerr(1.0, 1e-8)
    np.testing.assert_allclose(outer_horizon(tiny), 2.0, atol=1e-12)
    np.testing.assert_allclose(ergosphere_radius(tiny, theta), 2.0, atol=1e-12)


def test_extremal_limit_outer_horizon_tends_to_mass() -> None:
    spins = [0.9, 0.99, 0.999, 0.9999, 1.0 - 1e-8, 1.0 - 1e-12]
    gaps = [outer_horizon(kerr(1.0, s)) - 1.0 for s in spins]
    assert all(g > 0.0 for g in gaps)
    assert all(later < earlier for earlier, later in zip(gaps, gaps[1:]))
    assert gaps[-1] < 1e-5
    r_plus, r_minus = horizon_radii(kerr(1.0, spins[-1]))
    assert r_plus - r_minus < 1e-5


@pytest.mark.parametrize("spin", SPINS)
def test_ergosphere_is_where_g_tt_vanishes(spin: float) -> None:
    st = kerr(1.0, spin)
    rng = np.random.default_rng(21)
    theta = 0.05 + (np.pi - 0.1) * rng.random(25)
    r_e = ergosphere_radius(st, theta)
    # For a = 0 the ergosphere coincides with the horizon, where g_rr = Sigma / Delta
    # diverges; only g_tt is inspected here, so silence that expected division.
    with np.errstate(divide="ignore"):
        g_tt_on_surface = metric_components(st, r_e, theta).g_tt
    np.testing.assert_allclose(g_tt_on_surface, 0.0, atol=1e-12)
    # Just outside r_E the Killing vector d/dt is timelike (g_tt < 0), just inside it is spacelike.
    assert np.all(metric_components(st, r_e + 1e-3, theta).g_tt < 0.0)
    if spin != 0.0:
        assert np.all(metric_components(st, r_e - 1e-3, theta).g_tt > 0.0)


@pytest.mark.parametrize("spin", SPINS)
def test_ergosphere_limits_and_ordering(spin: float) -> None:
    st = kerr(1.0, spin)
    r_plus = outer_horizon(st)
    np.testing.assert_allclose(ergosphere_radius(st, 0.0), r_plus, rtol=1e-15)
    np.testing.assert_allclose(ergosphere_radius(st, np.pi), r_plus, rtol=1e-15)
    np.testing.assert_allclose(ergosphere_radius(st, np.pi / 2), 2.0 * st.mass, rtol=1e-15)
    theta = np.linspace(0.0, np.pi / 2, 50)
    r_e = ergosphere_radius(st, theta)
    assert np.all(r_e >= r_plus - 1e-15) and np.all(r_e <= 2.0 * st.mass + 1e-15)
    assert np.all(np.diff(r_e) >= 0.0)  # grows monotonically from the pole to the equator
    assert r_e.shape == theta.shape and r_e.dtype == np.float64


@pytest.mark.parametrize("epsilon", [0.0, 1e-6])
def test_inside_horizon_logic(epsilon: float) -> None:
    st = kerr(1.0, 0.9)
    r_plus = outer_horizon(st)
    r = np.array([r_plus - 0.1, r_plus, r_plus + 0.5e-6, r_plus + 1.0])
    expected = np.array([True, True, epsilon > 0.5e-6, False])
    result = inside_horizon(st, r, epsilon)
    assert result.dtype == np.bool_
    np.testing.assert_array_equal(result, expected)
    assert bool(inside_horizon(st, r_plus + 10.0, epsilon)) is False
    assert bool(inside_horizon(st, 0.5 * r_plus, epsilon)) is True
    assert inside_horizon(st, r[:, None], epsilon).shape == (4, 1)


def test_inside_horizon_rejects_negative_epsilon() -> None:
    with pytest.raises(ValueError):
        inside_horizon(kerr(1.0, 0.5), 3.0, epsilon=-1e-6)


def test_horizons_scale_linearly_with_mass() -> None:
    one, two = kerr(1.0, 0.5), kerr(2.0, 0.5)
    np.testing.assert_allclose(horizon_radii(two), 2.0 * np.array(horizon_radii(one)), rtol=1e-15)
    theta = np.linspace(0.0, np.pi, 9)
    np.testing.assert_allclose(ergosphere_radius(two, theta), 2.0 * ergosphere_radius(one, theta), rtol=1e-15)


def test_horizons_are_even_in_spin() -> None:
    theta = np.linspace(0.0, np.pi, 9)
    for spin in (0.3, 0.9, 0.998):
        pos, neg = kerr(1.0, spin), kerr(1.0, -spin)
        assert horizon_radii(pos) == horizon_radii(neg)
        np.testing.assert_array_equal(ergosphere_radius(pos, theta), ergosphere_radius(neg, theta))
