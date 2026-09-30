"""Coordinate helper tests: Cartesian embedding for plots and angle units."""

from __future__ import annotations

import numpy as np
import pytest

from kerrray.geometry import bl_to_cartesian, deg_to_rad, kerr, outer_horizon, rad_to_deg, schwarzschild

SPINS = (0.0, 0.3, -0.5, 0.9, 0.998)


def test_schwarzschild_reduces_to_spherical_polar() -> None:
    st = schwarzschild()
    r = np.array([2.0, 5.0, 40.0])
    theta = np.array([0.3, 1.5, 2.6])
    phi = np.array([0.0, 2.0, 5.5])
    x, y, z = bl_to_cartesian(st, r, theta, phi)
    np.testing.assert_allclose(x, r * np.sin(theta) * np.cos(phi), rtol=1e-15)
    np.testing.assert_allclose(y, r * np.sin(theta) * np.sin(phi), rtol=1e-15)
    np.testing.assert_allclose(z, r * np.cos(theta), rtol=1e-15)
    np.testing.assert_allclose(np.sqrt(x**2 + y**2 + z**2), r, rtol=1e-15)


@pytest.mark.parametrize("spin", SPINS)
def test_confocal_ellipsoid_identity(spin: float) -> None:
    """(x^2 + y^2) / (r^2 + a^2) + z^2 / r^2 = 1 for every (theta, phi)."""
    st = kerr(1.0, spin)
    rng = np.random.default_rng(41)
    r = 1.5 + 20.0 * rng.random(30)
    theta = np.pi * rng.random(30)
    phi = 2 * np.pi * rng.random(30)
    x, y, z = bl_to_cartesian(st, r, theta, phi)
    np.testing.assert_allclose((x**2 + y**2) / (r**2 + st.a**2) + z**2 / r**2, 1.0, rtol=1e-13)
    np.testing.assert_allclose(np.arctan2(y, x) % (2 * np.pi), phi % (2 * np.pi), atol=1e-12)


def test_axis_and_equatorial_plane() -> None:
    st = kerr(1.0, 0.8)
    x, y, z = bl_to_cartesian(st, 3.0, 0.0, 1.0)
    assert x == 0.0 and y == 0.0 and z == 3.0
    x, y, z = bl_to_cartesian(st, 3.0, np.pi / 2, 0.0)
    np.testing.assert_allclose(x, np.sqrt(9.0 + st.a**2), rtol=1e-15)
    assert y == 0.0
    np.testing.assert_allclose(z, 0.0, atol=1e-15)


def test_horizon_embeds_as_an_oblate_spheroid() -> None:
    """Delta(r_plus) = 0 gives r_plus^2 + a^2 = 2 M r_plus, so the equatorial
    embedded radius sqrt(2 M r_plus) exceeds the polar radius r_plus when a != 0."""
    for spin in (0.3, 0.9, 0.998):
        st = kerr(1.0, spin)
        r_plus = outer_horizon(st)
        x_eq, _, _ = bl_to_cartesian(st, r_plus, np.pi / 2, 0.0)
        _, _, z_pole = bl_to_cartesian(st, r_plus, 0.0, 0.0)
        np.testing.assert_allclose(x_eq, np.sqrt(2.0 * st.mass * r_plus), rtol=1e-14)
        assert x_eq > z_pole == r_plus


def test_broadcasting_and_dtype() -> None:
    st = kerr(1.0, 0.5)
    r = np.linspace(2, 4, 3)[:, None, None]
    theta = np.linspace(0.1, 3.0, 2)[None, :, None]
    phi = np.linspace(0, 6, 5)[None, None, :]
    for arr in bl_to_cartesian(st, r, theta, phi):
        assert arr.shape == (3, 2, 5) and arr.dtype == np.float64
    for arr in bl_to_cartesian(st, [2, 3], 1, 0):
        assert arr.shape == (2,) and arr.dtype == np.float64


def test_deg_rad_conversions() -> None:
    np.testing.assert_allclose(deg_to_rad(180.0), np.pi, rtol=1e-15)
    np.testing.assert_allclose(deg_to_rad([0, 90, 360]), [0.0, np.pi / 2, 2 * np.pi], rtol=1e-15)
    np.testing.assert_allclose(rad_to_deg(np.pi / 4), 45.0, rtol=1e-15)
    angles = np.linspace(-720.0, 720.0, 13)
    np.testing.assert_allclose(rad_to_deg(deg_to_rad(angles)), angles, rtol=1e-14, atol=1e-13)
    assert deg_to_rad([1, 2]).dtype == np.float64 and rad_to_deg([1, 2]).dtype == np.float64
    assert np.asarray(deg_to_rad(60)).dtype == np.float64
