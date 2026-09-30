"""Conserved quantities: definitions, separated potentials, drift along rays, diagnostics."""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from kerrray.geodesics import (
    IDX_PTH,
    IntegratorOptions,
    TerminationOptions,
    carter_potentials,
    equatorial_photon,
    geodesic_rhs,
    integrate,
    integrate_batch,
    photon_from_constants,
)
from kerrray.geometry import kerr, schwarzschild
from kerrray.photons import (
    ConservationDiagnostics,
    TerminationState,
    angular_momentum,
    carter_constant,
    conservation_diagnostics,
    energy,
    null_constraint,
    null_constraint_from_rhs,
    relative_drift,
)


def test_separated_potentials_identity_with_the_metric() -> None:
    """2 Sigma H = (Delta p_r^2 - R/Delta) + (p_theta^2 - Theta), evaluated at 40 digits.

    Both sides are built independently in SymPy: H from the closed-form
    inverse metric of docs/architecture.md section 1.1, R and Theta from
    BPT 1972 eqs. 2.9-2.10 (null case) as implemented in initial_conditions.
    """
    r, th, mass, a = sp.symbols("r theta M a", positive=True)
    e, lz, q, pr, pth = sp.symbols("E L_z Q p_r p_theta", real=True)
    sigma = r**2 + a**2 * sp.cos(th) ** 2
    delta = r**2 - 2 * mass * r + a**2
    big_a = (r**2 + a**2) ** 2 - a**2 * delta * sp.sin(th) ** 2
    gtt = -big_a / (sigma * delta)
    gtph = -2 * mass * a * r / (sigma * delta)
    gphph = (delta - a**2 * sp.sin(th) ** 2) / (sigma * delta * sp.sin(th) ** 2)
    two_sigma_h = sigma * (gtt * e**2 - 2 * gtph * e * lz + (delta / sigma) * pr**2 + pth**2 / sigma + gphph * lz**2)
    big_r = (e * (r**2 + a**2) - a * lz) ** 2 - delta * (q + (lz - a * e) ** 2)
    big_theta = q - sp.cos(th) ** 2 * (lz**2 / sp.sin(th) ** 2 - a**2 * e**2)
    separated = (delta * pr**2 - big_r / delta) + (pth**2 - big_theta)
    diff = two_sigma_h - separated
    rng = np.random.default_rng(3)
    for _ in range(8):
        subs = {
            r: sp.Rational(int(rng.integers(30, 400)), 10),
            th: sp.Rational(int(rng.integers(2, 30)), 10),
            mass: 1,
            a: sp.Rational(int(rng.integers(-9, 10)), 10),
            e: sp.Rational(int(rng.integers(5, 20)), 10),
            lz: sp.Rational(int(rng.integers(-60, 60)), 10),
            q: sp.Rational(int(rng.integers(0, 300)), 10),
            pr: sp.Rational(int(rng.integers(-50, 50)), 10),
            pth: sp.Rational(int(rng.integers(-50, 50)), 10),
        }
        value = sp.N(diff.subs(subs), 40)
        scale = sp.N(sp.Abs(two_sigma_h.subs(subs)) + sp.Abs(big_r.subs(subs) / delta.subs(subs)) + 1, 40)
        assert abs(float(value / scale)) < 1e-30


def test_carter_constant_definition_against_potentials() -> None:
    st = kerr(1.0, 0.8)
    rng = np.random.default_rng(5)
    for _ in range(20):
        r, th = rng.uniform(3.0, 40.0), rng.uniform(0.2, np.pi - 0.2)
        e, lz, q = rng.uniform(0.5, 2.0), rng.uniform(-3.0, 3.0), rng.uniform(1.0, 20.0)
        big_r, big_theta, _, _ = carter_potentials(st, r, th, e, lz, q)
        if big_r < 0 or big_theta < 0:
            continue
        y = photon_from_constants(st, r, th, e, lz, q, sign_r=1, sign_theta=-1)
        assert abs(float(carter_constant(st, y)) - q) < 1e-10 * max(1.0, q)
        assert abs(y[IDX_PTH] ** 2 - big_theta) < 1e-10 * max(1.0, big_theta)
        assert float(energy(y)) == e and float(angular_momentum(y)) == lz
    # Schwarzschild: Q + L_z^2 is the total squared angular momentum p_theta^2 + L_z^2 / sin^2(theta).
    st0 = schwarzschild()
    y = np.array([0.0, 10.0, 0.7, 0.0, -1.0, 0.1, 2.0, 1.5])
    total = y[IDX_PTH] ** 2 + 1.5**2 / np.sin(0.7) ** 2
    assert abs(float(carter_constant(st0, y)) + 1.5**2 - total) < 1e-12
    # On the axis with L_z = 0 the guarded form is finite.
    y_axis = np.array([0.0, 10.0, 0.0, 0.0, -1.0, 0.1, 2.0, 0.0])
    assert float(carter_constant(st, y_axis)) == pytest.approx(4.0 - st.a**2)


def test_relative_drift_floor() -> None:
    v0 = np.array([2.0, 0.0, -0.5])
    v = v0 + np.array([2e-6, 3e-6, -4e-6])
    d = relative_drift(v, v0)
    assert d[0] == pytest.approx(1e-6) and d[1] == pytest.approx(3e-6) and d[2] == pytest.approx(4e-6)
    assert relative_drift(v, v0, floor=0.1)[2] == pytest.approx(8e-6)
    with pytest.raises(ValueError):
        relative_drift(v, v0, floor=0.0)


def test_energy_and_lz_are_exactly_constant_along_a_kerr_ray() -> None:
    st = kerr(1.0, 0.9)
    y0 = photon_from_constants(st, 30.0, 1.1, 1.0, 3.0, 6.0, sign_r=-1, sign_theta=1)
    traj = integrate(st, y0, IntegratorOptions(method="rk45", rtol=1e-9, atol=1e-11), TerminationOptions(), record=True)
    assert traj.state in (TerminationState.ESCAPED, TerminationState.CAPTURED)
    assert traj.diagnostics.max_energy_drift <= 1e-13 and traj.diagnostics.max_lz_drift <= 1e-13
    assert np.all(energy(traj.y) == 1.0) and np.all(angular_momentum(traj.y) == 3.0)


def test_carter_and_null_conserved_for_off_equatorial_kerr_ray() -> None:
    st = kerr(1.0, 0.9)
    y0 = photon_from_constants(st, 30.0, 1.1, 1.0, 3.0, 6.0, sign_r=-1, sign_theta=1)
    traj = integrate(st, y0, IntegratorOptions(method="rk45", rtol=1e-9, atol=1e-11), TerminationOptions(), record=True)
    assert traj.state == TerminationState.ESCAPED
    d = traj.diagnostics
    assert d.max_carter_drift < 1e-7 and d.max_null_error < 1e-8
    assert np.max(np.abs(null_constraint(st, traj.y))) < 1e-8
    assert np.max(relative_drift(carter_constant(st, traj.y), carter_constant(st, y0))) < 1e-7
    # Both null-constraint evaluations agree.
    f = geodesic_rhs(st, traj.y)
    assert np.max(np.abs(null_constraint_from_rhs(traj.y, f) - null_constraint(st, traj.y))) < 1e-12


def test_conservation_diagnostics_container() -> None:
    st = schwarzschild()
    y0 = equatorial_photon(st, 1000.0, 6.0)
    traj = integrate(st, y0, IntegratorOptions(method="rk45", rtol=1e-9, atol=1e-11), TerminationOptions())
    d = traj.diagnostics
    assert isinstance(d, ConservationDiagnostics)
    n = traj.y.shape[0]
    for arr in (d.energy, d.lz, d.carter, d.null, d.energy_drift, d.lz_drift, d.carter_drift, d.null_error):
        assert arr.shape == (n,)
    assert d.energy_drift[0] == 0.0 and d.lz_drift[0] == 0.0 and d.carter_drift[0] == 0.0
    assert d.max_null == d.max_null_error
    # The running maximum uses (1/2) p_mu dx^mu/dlambda, the recorded array the inverse metric.
    assert d.max_null_error == pytest.approx(np.max(d.null_error), rel=1e-6)
    assert d.max_null_error >= np.max(d.null_error) * (1 - 1e-6)
    assert d.max_carter_drift < 1e-12  # Q_0 = 0 exactly for an equatorial ray; absolute drift within the floor
    again = conservation_diagnostics(st, traj.y)
    assert again.max_null_error == pytest.approx(d.max_null_error)


def test_batch_diagnostics_agree_with_scalar() -> None:
    st = kerr(1.0, 0.9)
    y0 = photon_from_constants(st, 30.0, 1.1, 1.0, 3.0, 6.0, sign_r=-1, sign_theta=1)
    integ = IntegratorOptions(method="rk45", rtol=1e-9, atol=1e-11)
    traj = integrate(st, y0, integ, TerminationOptions(), record=False)
    batch = integrate_batch(st, y0[None, :], integ, TerminationOptions())
    assert batch.max_carter_drift[0] == pytest.approx(traj.diagnostics.max_carter_drift, rel=1e-6, abs=1e-14)
    assert batch.max_null_error[0] == pytest.approx(traj.diagnostics.max_null_error, rel=1e-6, abs=1e-14)
    assert batch.max_energy_drift[0] == traj.diagnostics.max_energy_drift == 0.0
    assert batch.max_lz_drift[0] == traj.diagnostics.max_lz_drift == 0.0
