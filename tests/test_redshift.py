"""Tests for kerrray.physics.redshift (PROJECT.md section 24; docs/rendering.md section 2)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from kerrray.geometry import kerr, metric, metric_components, schwarzschild
from kerrray.photons.orbits import keplerian_angular_velocity as orbits_omega
from kerrray.physics.redshift import (
    emitter_four_velocity,
    gravitational_redshift_factor,
    keplerian_angular_velocity,
    redshift_factor,
    static_four_velocity,
)

SPACETIMES = [schwarzschild(), kerr(1.0, 0.9), kerr(1.0, -0.5)]
REVERSE = np.array([1, 1, 1, 1, -1, -1, -1, -1], dtype=float)


def _states(rng: np.random.Generator, n: int, r_lo: float = 4.5, r_hi: float = 30.0) -> np.ndarray:
    """Equatorial states with E = 1, |L_z| <= 5 and arbitrary p_r, p_theta (they do not enter)."""
    y = np.zeros((n, 8))
    y[:, 1] = rng.uniform(r_lo, r_hi, n)
    y[:, 2] = 0.5 * math.pi
    y[:, 4] = -1.0
    y[:, 5] = rng.normal(size=n)
    y[:, 6] = rng.normal(size=n)
    y[:, 7] = rng.uniform(-5.0, 5.0, n)
    return y


def _closed_form(st, y: np.ndarray, prograde: bool) -> np.ndarray:
    """g = 1 / [u^t (1 - Omega L_z / E)] (docs/rendering.md section 2)."""
    r = y[:, 1]
    omega = keplerian_angular_velocity(st, r, prograde)
    g = metric_components(st, r, 0.5 * math.pi)
    u_t = 1.0 / np.sqrt(-(g.g_tt + 2.0 * omega * g.g_tphi + omega**2 * g.g_phph))
    xi = y[:, 7] / (-y[:, 4])
    return 1.0 / (u_t * (1.0 - omega * xi))


@pytest.mark.parametrize("st", SPACETIMES, ids=lambda s: f"a={s.spin}")
@pytest.mark.parametrize("prograde", [True, False])
def test_direct_contraction_matches_closed_form(st, prograde: bool) -> None:
    y = _states(np.random.default_rng(1), 64)
    direct = redshift_factor(st, y, prograde=prograde)
    closed = _closed_form(st, y, prograde)
    assert np.all(direct > 0.0)
    assert np.max(np.abs(direct / closed - 1.0)) < 1e-12


@pytest.mark.parametrize("st", SPACETIMES, ids=lambda s: f"a={s.spin}")
def test_reversed_momentum_gives_the_same_factor(st) -> None:
    y = _states(np.random.default_rng(2), 16)
    assert np.array_equal(redshift_factor(st, y), redshift_factor(st, y * REVERSE))
    single = redshift_factor(st, y[0])
    assert single.shape == () and single == redshift_factor(st, y)[0]


@pytest.mark.parametrize("st", SPACETIMES, ids=lambda s: f"a={s.spin}")
@pytest.mark.parametrize("prograde", [True, False])
def test_emitter_four_velocity_is_unit_timelike_keplerian(st, prograde: bool) -> None:
    r = np.array([4.5, 6.0, 10.0, 50.0])
    u = emitter_four_velocity(st, r, prograde=prograde)
    g = metric(st, r, 0.5 * math.pi)
    norm = np.einsum("ni,nij,nj->n", u, g, u)
    assert np.max(np.abs(norm + 1.0)) < 1e-12
    assert np.all(u[:, 1] == 0.0) and np.all(u[:, 2] == 0.0)
    assert np.allclose(u[:, 3] / u[:, 0], orbits_omega(st, r, prograde), rtol=1e-14)
    assert keplerian_angular_velocity is orbits_omega


def test_schwarzschild_closed_forms() -> None:
    """u^t = (1 - 3M/r)^(-1/2); a radial photon (L_z = 0) has g = sqrt(1 - 3M/r)."""
    st = schwarzschild()
    r = np.array([4.0, 6.0, 20.0])
    u = emitter_four_velocity(st, r)
    assert np.allclose(u[:, 0], 1.0 / np.sqrt(1.0 - 3.0 / r), rtol=1e-14)
    y = np.zeros((3, 8))
    y[:, 1], y[:, 2], y[:, 4] = r, 0.5 * math.pi, -1.0
    assert np.allclose(redshift_factor(st, y), np.sqrt(1.0 - 3.0 / r), rtol=1e-14)


def test_doppler_sign_prograde_lz_blue_shifts() -> None:
    """A photon with a L_z > 0 is emitted along the gas motion: g larger than with -L_z."""
    for st in SPACETIMES:
        y = np.zeros((2, 8))
        y[:, 1], y[:, 2], y[:, 4] = 8.0, 0.5 * math.pi, -1.0
        sign = 1.0 if st.a >= 0.0 else -1.0
        y[0, 7], y[1, 7] = 3.0 * sign, -3.0 * sign
        g = redshift_factor(st, y, prograde=True)
        assert g[0] > g[1]


def test_static_emitter_gravitational_redshift_tends_to_one() -> None:
    st = kerr(1.0, 0.7)
    radii = np.array([3.0, 10.0, 1.0e3, 1.0e6])
    y = np.zeros((4, 8))
    y[:, 1], y[:, 2], y[:, 4], y[:, 7] = radii, 1.2, -1.0, 0.4
    g = gravitational_redshift_factor(st, y)
    assert np.all(g < 1.0) and np.all(np.diff(g) > 0.0)
    assert abs(g[-1] - 1.0) < 2.0e-6  # g = 1 - M/r + ... at r = 1e6 M
    assert np.allclose(g, np.sqrt(-metric_components(st, radii, 1.2).g_tt), rtol=1e-14)
    assert np.array_equal(g, gravitational_redshift_factor(st, y * REVERSE))
    u = static_four_velocity(st, radii, 1.2)
    assert np.allclose(u[:, 0], 1.0 / g, rtol=1e-14) and np.all(u[:, 1:] == 0.0)


def test_error_paths() -> None:
    st = schwarzschild()
    y = np.zeros(8)
    y[1], y[2], y[4] = 8.0, 0.5 * math.pi + 1e-3, -1.0
    with pytest.raises(ValueError):
        redshift_factor(st, y)  # not equatorial
    y[2] = 0.5 * math.pi
    y[1] = 3.0
    with pytest.raises(ValueError):
        redshift_factor(st, y)  # at the photon orbit: no timelike circular orbit
    y[1] = 8.0
    y[7] = 100.0  # 1 - Omega L_z / E < 0: cannot have been emitted by the gas
    with pytest.raises(ValueError):
        redshift_factor(st, y)
    with pytest.raises(ValueError):
        static_four_velocity(kerr(1.0, 0.9), 1.5, 0.5 * math.pi)  # inside the ergosphere
    with pytest.raises(ValueError):
        redshift_factor(st, np.zeros(7))
