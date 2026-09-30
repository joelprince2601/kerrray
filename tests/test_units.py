"""Geometric-unit conversion tests (PROJECT.md section 33)."""

from __future__ import annotations

import numpy as np
import pytest

from kerrray.utils.units import C_SI, GM_SUN_SI, G_SI, GeometricUnits


def test_constants_are_the_cited_values() -> None:
    assert G_SI == 6.67430e-11  # CODATA 2018
    assert C_SI == 299_792_458.0  # exact SI definition
    assert GM_SUN_SI == 1.3271244e20  # IAU 2015 Resolution B3 nominal value


def test_solar_gravitational_radius_matches_check_value() -> None:
    """GM_sun / c^2 computed from the constants against the quoted 1476.6 m.

    The check value (the Sun's gravitational radius, 1.4766 km) has five
    significant figures, so the comparison uses half a unit in its last place.
    It only checks the constants; it produces nothing.
    """
    units = GeometricUnits(1.0)
    assert units.length_unit_m == GM_SUN_SI / C_SI**2
    assert units.length_unit_m == pytest.approx(1476.6, abs=0.05)


def test_time_unit_is_length_unit_over_c() -> None:
    units = GeometricUnits(4.0e6)
    assert units.time_unit_s == pytest.approx(units.length_unit_m / C_SI, rel=1e-15)


def test_mass_kg_is_consistent_with_g_and_gm() -> None:
    units = GeometricUnits(1.0)
    assert units.mass_kg() * G_SI == pytest.approx(GM_SUN_SI, rel=1e-15)
    assert units.mass_kg(2.5) == pytest.approx(2.5 * units.mass_kg(), rel=1e-15)
    assert units.mass_geometric(units.mass_kg(3.0)) == pytest.approx(3.0, rel=1e-15)


def test_units_scale_linearly_with_mass() -> None:
    one, ten = GeometricUnits(1.0), GeometricUnits(10.0)
    assert ten.length_unit_m == pytest.approx(10.0 * one.length_unit_m, rel=1e-15)
    assert ten.time_unit_s == pytest.approx(10.0 * one.time_unit_s, rel=1e-15)
    assert ten.mass_kg() == pytest.approx(10.0 * one.mass_kg(), rel=1e-15)


@pytest.mark.parametrize("mass_solar", [1.0, 4.3e6, 6.5e9])
def test_scalar_round_trips(mass_solar: float) -> None:
    units = GeometricUnits(mass_solar)
    for value in (0.0, 1.0, 3.0, 1000.0):
        assert units.length_geometric(units.length_m(value)) == pytest.approx(value, rel=1e-14)
        assert units.time_geometric(units.time_s(value)) == pytest.approx(value, rel=1e-14)
    for nu in (1.0, 0.1, 1e-3):
        assert units.frequency_geometric(units.frequency_hz(nu)) == pytest.approx(nu, rel=1e-14)
    assert isinstance(units.length_m(2.0), float)
    assert units.length_m(1.0) == units.length_unit_m
    assert units.time_s(1.0) == units.time_unit_s
    assert units.frequency_hz(1.0) == pytest.approx(1.0 / units.time_unit_s, rel=1e-15)


def test_array_conversions_are_vectorised() -> None:
    units = GeometricUnits(2.0)
    radii = np.array([[1.0, 2.0, 3.0], [10.0, 100.0, 1000.0]])
    metres = units.length_m(radii)
    assert isinstance(metres, np.ndarray) and metres.shape == radii.shape
    np.testing.assert_allclose(metres, radii * units.length_unit_m, rtol=1e-15)
    np.testing.assert_allclose(units.length_geometric(metres), radii, rtol=1e-14)
    lam = np.linspace(0.0, 50.0, 7)
    np.testing.assert_allclose(units.time_geometric(units.time_s(lam)), lam, rtol=1e-14)
    nu = np.array([1.0, 2.0])
    np.testing.assert_allclose(units.frequency_geometric(units.frequency_hz(nu)), nu, rtol=1e-14)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_mass_rejected(bad: float) -> None:
    with pytest.raises(ValueError, match="mass_solar"):
        GeometricUnits(bad)


def test_units_are_immutable() -> None:
    units = GeometricUnits(1.0)
    with pytest.raises(AttributeError):
        units.mass_solar = 2.0  # type: ignore[misc]
