"""Null condition of the initial-state constructors and along integrated rays."""

from __future__ import annotations

import numpy as np
import pytest

from kerrray.geodesics import (
    IDX_PPH,
    IDX_PR,
    IDX_PT,
    IDX_PTH,
    IDX_R,
    IDX_TH,
    IntegratorOptions,
    TerminationOptions,
    carter_potentials,
    equatorial_photon,
    geodesic_rhs,
    hamiltonian,
    integrate,
    photon_from_constants,
    tangential_photon,
)
from kerrray.geometry import inverse_metric_components, kerr, outer_horizon, schwarzschild
from kerrray.photons import TerminationState, carter_constant, energy, null_constraint

NULL_TOL = 1e-12


@pytest.mark.parametrize("spin", [0.0, 0.9, -0.5])
def test_photon_from_constants_is_null_and_reproduces_constants(spin: float) -> None:
    st = kerr(1.0, spin)
    rng = np.random.default_rng(11)
    for _ in range(30):
        r = rng.uniform(outer_horizon(st) + 0.5, 40.0)
        th = rng.uniform(0.2, np.pi - 0.2)
        e = rng.uniform(0.5, 2.0)
        lz = rng.uniform(-6.0, 6.0)
        p_th = rng.uniform(-5.0, 5.0)
        q = float(carter_constant(st, np.array([0, r, th, 0, -e, 0, p_th, lz])))
        if carter_potentials(st, r, th, e, lz, q)[0] < 0.0:
            continue  # no photon with these constants passes through r (turning point further out)
        sign_r = int(rng.choice([-1, 1]))
        sign_th = 1 if p_th >= 0 else -1
        y = photon_from_constants(st, r, th, e, lz, q, sign_r=sign_r, sign_theta=sign_th, t=1.0, phi=0.3)
        assert abs(hamiltonian(st, y)) < NULL_TOL * max(1.0, e * e)
        assert y[IDX_PT] == -e and y[IDX_PPH] == lz and y[0] == 1.0 and y[3] == 0.3
        assert np.sign(y[IDX_PR]) == sign_r or y[IDX_PR] == 0.0
        assert abs(y[IDX_PTH] - p_th) < 1e-9 * max(1.0, abs(p_th))
        assert abs(float(carter_constant(st, y)) - q) < 1e-9 * max(1.0, abs(q))


def test_photon_from_constants_rejects_inconsistent_constants() -> None:
    st = kerr(1.0, 0.9)
    with pytest.raises(ValueError, match="Theta"):
        photon_from_constants(st, 30.0, 1.0, 1.0, 9.0, 5.0, sign_r=-1, sign_theta=1)
    with pytest.raises(ValueError, match="R"):
        photon_from_constants(st, 3.0, 0.5 * np.pi, 1.0, 40.0, 0.0, sign_r=-1, sign_theta=1)
    with pytest.raises(ValueError):
        photon_from_constants(st, 1.0, 1.0, 1.0, 0.0, 0.0, sign_r=-1, sign_theta=1)  # inside horizon
    with pytest.raises(ValueError):
        photon_from_constants(st, 10.0, 1.0, -1.0, 0.0, 0.0, sign_r=-1, sign_theta=1)  # E <= 0
    with pytest.raises(ValueError):
        photon_from_constants(st, 10.0, 1.0, 1.0, 0.0, 0.0, sign_r=2, sign_theta=1)


@pytest.mark.parametrize("spin", [0.0, 0.9, -0.9])
@pytest.mark.parametrize("prograde", [True, False])
@pytest.mark.parametrize("inward", [True, False])
def test_equatorial_photon_null_and_sign_conventions(spin: float, prograde: bool, inward: bool) -> None:
    st = kerr(1.0, spin)
    b, e = 6.0, 1.5
    y = equatorial_photon(st, 1000.0, b, inward=inward, prograde=prograde, E=e)
    assert abs(hamiltonian(st, y)) < NULL_TOL * e * e
    assert y[IDX_TH] == 0.5 * np.pi and y[IDX_PTH] == 0.0 and y[IDX_PT] == -e
    assert abs(abs(y[IDX_PPH]) - b * e) < 1e-14
    if st.a != 0.0:
        assert (st.a * y[IDX_PPH] > 0) == prograde
    else:
        assert (y[IDX_PPH] > 0) == prograde
    dr = geodesic_rhs(st, y)[IDX_R]
    assert (dr < 0) == inward
    assert float(energy(y)) == e


@pytest.mark.parametrize("spin", [0.0, 0.9, -0.9])
@pytest.mark.parametrize("prograde", [True, False])
def test_tangential_photon_has_zero_radial_velocity_and_is_null(spin: float, prograde: bool) -> None:
    st = kerr(1.0, spin)
    for r0 in (3.0, 4.5, 10.0):
        y = tangential_photon(st, r0, prograde=prograde, E=1.0)
        f = geodesic_rhs(st, y)
        assert y[IDX_PR] == 0.0 and f[IDX_R] == 0.0
        assert abs(hamiltonian(st, y)) < NULL_TOL * max(1.0, y[IDX_PPH] ** 2)
        sign_expected = 1.0 if prograde == (st.a >= 0.0) else -1.0
        assert np.sign(y[IDX_PPH]) == sign_expected
        # p_r = 0 and the null condition give L_z^2 / E^2 = -g^{tt} / g^{phph} when a = 0.
        if st.a == 0.0:
            g = inverse_metric_components(st, r0, 0.5 * np.pi)
            assert abs(y[IDX_PPH] ** 2 - (-g.gtt / g.gphph)) < 1e-10 * y[IDX_PPH] ** 2


def test_tangential_photon_spin_symmetry_and_errors() -> None:
    st = kerr(1.0, 0.9)
    pro = tangential_photon(st, 5.0, prograde=True)
    retro = tangential_photon(st, 5.0, prograde=False)
    assert abs(pro[IDX_PPH]) < abs(retro[IDX_PPH])  # prograde photons need less angular momentum
    mirror = tangential_photon(kerr(1.0, -0.9), 5.0, prograde=True)
    assert abs(mirror[IDX_PPH] + pro[IDX_PPH]) < 1e-12
    with pytest.raises(ValueError):
        tangential_photon(st, 2.0, prograde=False)  # equatorial ergosphere, retrograde root singular
    with pytest.raises(ValueError):
        tangential_photon(st, 1.0)


def test_null_constraint_long_schwarzschild_ray() -> None:
    """b = 6 from r0 = 1000 at rtol 1e-9: |H| stays below 1e-8 along the whole ray."""
    st = schwarzschild()
    y0 = equatorial_photon(st, 1000.0, 6.0)
    traj = integrate(st, y0, IntegratorOptions(method="rk45", rtol=1e-9, atol=1e-11), TerminationOptions())
    assert traj.state == TerminationState.ESCAPED
    h_along = null_constraint(st, traj.y)
    assert h_along.shape == (traj.y.shape[0],)
    assert np.max(np.abs(h_along)) < 1e-8
    assert traj.diagnostics.max_null_error < 1e-8
    assert traj.y[-1, IDX_R] >= 1000.0 and traj.n_steps > 10


def test_hamiltonian_scales_with_energy_squared() -> None:
    st = kerr(1.0, 0.6)
    y = np.array([0.0, 8.0, 1.1, 0.0, -1.0, 0.3, 0.7, 2.0])
    h1 = float(hamiltonian(st, y))
    y2 = y.copy()
    y2[4:] *= 3.0
    assert abs(float(hamiltonian(st, y2)) - 9.0 * h1) < 1e-12 * max(1.0, abs(h1))
