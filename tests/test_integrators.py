"""Integrators: tableau, error control, convergence, batch/scalar agreement, precision, reference solver."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import RK45

from kerrray.geodesics import (
    IDX_PH,
    IDX_R,
    IntegratorOptions,
    TerminationOptions,
    equatorial_photon,
    error_norm,
    integrate,
    integrate_batch,
    make_step_rk4,
    make_step_rk45,
    pack,
    photon_from_constants,
    step_factor,
)
from kerrray.geodesics.tableaus import DP_A, DP_B, DP_C, DP_E, FACTOR_MAX, FACTOR_MIN, RK4_B, RK4_C
from kerrray.geometry import kerr, schwarzschild
from kerrray.photons import TerminationState, azimuthal_winding, closest_approach, deflection_angle, to_cartesian

ST = schwarzschild()
FAR = TerminationOptions(escape_radius=1e6)
TERM = TerminationOptions()


def test_dormand_prince_tableau_matches_scipy_rk45() -> None:
    a = np.zeros((7, 7))
    for i, row in enumerate(DP_A):
        a[i, : len(row)] = row
    assert np.array_equal(np.asarray(DP_C[:6]), RK45.C)
    assert np.allclose(a[:6, :5], RK45.A, rtol=0.0, atol=1e-15)
    assert np.allclose(np.asarray(DP_B[:6]), RK45.B, rtol=0.0, atol=1e-15)
    assert np.allclose(np.asarray(DP_E), RK45.E, rtol=0.0, atol=1e-15)
    # Consistency: row sums equal the nodes, weights sum to one, FSAL row equals b.
    assert np.allclose([sum(row) for row in DP_A], DP_C, atol=1e-15)
    assert abs(sum(DP_B) - 1.0) < 1e-15 and abs(sum(DP_E)) < 1e-15
    assert np.allclose(DP_A[6], DP_B[:6], atol=0.0)
    assert abs(sum(RK4_B) - 1.0) < 1e-15 and RK4_C == (0.0, 0.5, 0.5, 1.0)


def test_step_helpers_on_a_linear_test_equation() -> None:
    """dy/dl = -y: one RK4 step reproduces exp(-h) to O(h^5); the DP error estimate is O(h^5)."""
    rhs = lambda y: -y  # noqa: E731
    y = np.array([[1.0, 2.0]])
    for h in (0.1, 0.05):
        y4, err4, f4 = make_step_rk4(rhs)(y, np.array([h]), rhs(y))
        assert err4 is None and np.allclose(f4, -y4)
        assert np.max(np.abs(y4 - y * np.exp(-h))) < 2 * h**5 / 120 * 2.0
        y5, err5, f5 = make_step_rk45(rhs)(y, np.array([h]), rhs(y))
        assert np.max(np.abs(y5 - y * np.exp(-h))) < 1e-3 * h**5
        assert np.allclose(f5, -y5) and np.all(np.abs(err5) < 1e-2 * h**5)
    # Scalar h with a (8,) state also works (the scalar integrator's usage).
    y1, _, _ = make_step_rk45(rhs)(np.ones(8), np.float64(0.1), None)
    assert y1.shape == (8,)


def test_error_norm_and_step_factor_follow_hairer() -> None:
    y = np.array([[1.0, -2.0]])
    y_new = np.array([[1.5, -1.0]])
    err = np.array([[1e-6, 2e-6]])
    sc = 1e-8 + 1e-6 * np.array([1.5, 2.0])
    expected = np.sqrt(np.mean((err[0] / sc) ** 2))
    assert abs(float(error_norm(err, y, y_new, 1e-8, 1e-6)[0]) - expected) < 1e-15
    assert float(step_factor(np.array(1.0))) == pytest.approx(0.9)
    assert float(step_factor(np.array(0.0))) == FACTOR_MAX
    assert float(step_factor(np.array(1e12))) == FACTOR_MIN
    assert float(step_factor(np.array(1e-12))) == FACTOR_MAX
    assert float(step_factor(np.array(1e-12), rejected_before=True)) == 1.0
    assert float(step_factor(np.array(2.0 ** 5))) == pytest.approx(0.9 / 2.0)


def test_rk4_fourth_order_convergence() -> None:
    y0 = equatorial_photon(ST, 20.0, 6.0)
    lam_end = 30.0
    ref = integrate(ST, y0, IntegratorOptions(method="dop853", rtol=1e-13, atol=1e-15, lambda_max=lam_end), FAR, record=False)
    assert ref.state == TerminationState.MAX_AFFINE_PARAMETER
    hs = np.array([0.5, 0.25, 0.125, 0.0625, 0.03125])
    errs = []
    for h in hs:
        traj = integrate(ST, y0, IntegratorOptions(method="rk4", step_size=float(h), lambda_max=lam_end), FAR, record=False)
        assert traj.state == TerminationState.MAX_AFFINE_PARAMETER and traj.n_steps == round(lam_end / h)
        errs.append(np.max(np.abs(traj.y[-1] - ref.y[-1])))
    slope = np.polyfit(np.log(hs), np.log(errs), 1)[0]
    assert 3.5 <= slope <= 4.5, (slope, errs)


def test_rk45_error_decreases_with_rtol() -> None:
    y0 = equatorial_photon(ST, 20.0, 6.0)
    lam_end = 30.0
    ref = integrate(ST, y0, IntegratorOptions(method="dop853", rtol=1e-13, atol=1e-15, lambda_max=lam_end), FAR, record=False)
    errs = []
    steps = []
    for rtol in (1e-5, 1e-7, 1e-9, 1e-11):
        traj = integrate(ST, y0, IntegratorOptions(method="rk45", rtol=rtol, atol=rtol * 1e-2, lambda_max=lam_end), FAR, record=False)
        errs.append(np.max(np.abs(traj.y[-1] - ref.y[-1])))
        steps.append(traj.n_steps)
    assert all(errs[k] > errs[k + 1] for k in range(len(errs) - 1)), errs
    assert all(steps[k] < steps[k + 1] for k in range(len(steps) - 1)), steps
    assert errs[-1] < 1e-8


def test_batch_matches_scalar_rk45_float64() -> None:
    bs = (4.0, 6.0, 8.0, 12.0)
    Y0 = np.stack([equatorial_photon(ST, 1000.0, b) for b in bs])
    integ = IntegratorOptions(method="rk45", rtol=1e-9, atol=1e-11)
    batch = integrate_batch(ST, Y0, integ, TERM)
    assert batch.Y.dtype == np.float64 and batch.n_rejected is not None
    for k, b in enumerate(bs):
        traj = integrate(ST, Y0[k], integ, TERM, record=False)
        assert traj.state == batch.state[k]
        assert traj.n_steps == batch.n_steps[k] and traj.n_rejected == batch.n_rejected[k]
        rel = np.abs(traj.y[-1] - batch.Y[k]) / np.maximum(1.0, np.abs(traj.y[-1]))
        assert np.max(rel) < 1e-9, (b, rel)
        assert abs(traj.lam[-1] - batch.lam[k]) < 1e-9 * max(1.0, traj.lam[-1])
        assert abs(traj.diagnostics.max_null_error - batch.max_null_error[k]) < 1e-9
    assert batch.state.tolist() == [int(TerminationState.CAPTURED)] + [int(TerminationState.ESCAPED)] * 3


def test_batch_rk4_matches_scalar_rk4() -> None:
    st = kerr(1.0, 0.9)
    y0 = photon_from_constants(st, 15.0, 1.0, 1.0, 2.0, 4.0, sign_r=1, sign_theta=-1)  # outgoing: no capture
    integ = IntegratorOptions(method="rk4", step_size=0.05, lambda_max=20.0)
    traj = integrate(st, y0, integ, FAR, record=False)
    batch = integrate_batch(st, np.stack([y0, y0]), integ, FAR)
    assert traj.n_steps == 400 and np.all(batch.n_steps == 400)
    assert np.max(np.abs(batch.Y[0] - traj.y[-1]) / np.maximum(1.0, np.abs(traj.y[-1]))) < 1e-12
    assert np.array_equal(batch.Y[0], batch.Y[1])


def test_float32_batch_runs_and_is_less_accurate_than_float64() -> None:
    st = kerr(1.0, 0.9)
    Y0 = np.stack([photon_from_constants(st, 200.0, 1.2, 1.0, lz, 15.0, sign_r=-1, sign_theta=1) for lz in (-6.0, 3.0, 6.0, 9.0)])
    lam_end = 300.0
    ref = integrate_batch(st, Y0, IntegratorOptions(method="rk45", rtol=1e-12, atol=1e-14, lambda_max=lam_end), FAR)
    assert np.all(ref.state == TerminationState.MAX_AFFINE_PARAMETER)
    kw = dict(method="rk45", rtol=1e-7, atol=1e-9, lambda_max=lam_end)
    r64 = integrate_batch(st, Y0, IntegratorOptions(**kw), FAR)
    r32 = integrate_batch(st, Y0, IntegratorOptions(dtype="float32", **kw), FAR)
    assert r32.Y.dtype == np.float32 and r32.lam.dtype == np.float64
    assert np.all(np.isfinite(r32.Y)) and np.all(r32.state == TerminationState.MAX_AFFINE_PARAMETER)
    assert np.all(np.isfinite(r32.max_null_error)) and np.all(np.isfinite(r32.max_carter_drift))
    scale = np.maximum(1.0, np.abs(ref.Y))
    e64 = np.max(np.abs(r64.Y - ref.Y) / scale, axis=1)
    e32 = np.max(np.abs(r32.Y.astype(np.float64) - ref.Y) / scale, axis=1)
    assert np.all(e32 > 0) and np.mean(e32) > np.mean(e64) and np.max(e32) > np.max(e64)
    assert np.mean(r32.max_null_error) > np.mean(r64.max_null_error)
    assert np.all(r32.max_energy_drift == 0.0) and np.all(r32.max_lz_drift == 0.0)


def test_dop853_reference_terminates_at_events() -> None:
    integ = IntegratorOptions(method="dop853", rtol=1e-10, atol=1e-12)
    captured = integrate(ST, equatorial_photon(ST, 1000.0, 4.0), integ, TERM)
    escaped = integrate(ST, equatorial_photon(ST, 1000.0, 6.0), integ, TERM)
    assert captured.state == TerminationState.CAPTURED
    assert abs(captured.y[-1, IDX_R] - (2.0 + TERM.horizon_epsilon)) < 1e-9
    assert escaped.state == TerminationState.ESCAPED
    assert abs(escaped.y[-1, IDX_R] - TERM.escape_radius) < 1e-6
    assert escaped.n_rejected == 0 and escaped.n_steps == len(escaped.lam) - 1
    assert escaped.diagnostics.max_null_error < 1e-8
    rk = integrate(ST, equatorial_photon(ST, 1000.0, 6.0), IntegratorOptions(method="rk45", rtol=1e-10, atol=1e-12), TERM)
    # Both solvers must agree on the deflection of the same ray (corrected for the different exit radii).
    assert abs(deflection_angle(rk) - deflection_angle(escaped)) < 1e-7
    with pytest.raises(ValueError):
        integrate_batch(ST, equatorial_photon(ST, 1000.0, 6.0)[None, :], integ, TERM)


def test_time_reversal_returns_to_start() -> None:
    st = kerr(1.0, 0.9)
    y0 = photon_from_constants(st, 30.0, 1.1, 1.0, 3.0, 6.0, sign_r=-1, sign_theta=1)
    integ = IntegratorOptions(method="rk45", rtol=1e-10, atol=1e-12, lambda_max=80.0)
    fwd = integrate(st, y0, integ, FAR, record=False)
    assert fwd.state == TerminationState.MAX_AFFINE_PARAMETER
    y_rev = fwd.y[-1].copy()
    y_rev[4:] *= -1.0
    back = integrate(st, y_rev, integ, FAR, record=False)
    assert back.state == TerminationState.MAX_AFFINE_PARAMETER
    assert np.max(np.abs(back.y[-1, :4] - y0[:4])) < 1e-7
    assert np.max(np.abs(back.y[-1, 4:] + y0[4:])) < 1e-7


def test_record_false_keeps_end_points_and_full_diagnostics() -> None:
    y0 = equatorial_photon(ST, 1000.0, 6.0)
    integ = IntegratorOptions(method="rk45", rtol=1e-9, atol=1e-11)
    full = integrate(ST, y0, integ, TERM, record=True)
    ends = integrate(ST, y0, integ, TERM, record=False)
    assert full.y.shape[0] == full.n_steps + 1 and ends.y.shape[0] == 2
    assert np.array_equal(full.y[-1], ends.y[-1]) and full.lam[-1] == ends.lam[-1]
    assert ends.diagnostics.max_null_error == full.diagnostics.max_null_error
    assert full.runtime_s > 0.0 and ends.n_rejected == full.n_rejected


def test_trajectory_geometry_helpers() -> None:
    integ = IntegratorOptions(method="rk45", rtol=1e-9, atol=1e-11)
    traj = integrate(ST, equatorial_photon(ST, 1000.0, 6.0), integ, TERM)
    assert 2.0 < closest_approach(traj) < 6.0
    dphi = azimuthal_winding(traj)
    assert dphi == traj.y[-1, IDX_PH] - traj.y[0, IDX_PH] and dphi > np.pi
    delta = deflection_angle(traj)
    assert 0.0 < delta < np.pi
    x, y, z = to_cartesian(traj)
    assert x.shape == y.shape == z.shape == traj.lam.shape and np.max(np.abs(z)) < 1e-9
    # Weak-field limit: the deflection approaches the labelled approximation 4 M / b from above,
    # with a residual that scales as (M / b)^2 (Phase 3 does the exact comparison).
    weak = integrate(ST, equatorial_photon(ST, 1000.0, 50.0), integ, TERM)
    weaker = integrate(ST, equatorial_photon(ST, 1000.0, 100.0), integ, TERM)
    res50 = deflection_angle(weak) - 4.0 / 50.0
    res100 = deflection_angle(weaker) - 4.0 / 100.0
    assert 0.0 < res100 < res50 < 0.1 * (4.0 / 50.0)
    assert 3.0 < res50 / res100 < 5.0
    retro = integrate(ST, equatorial_photon(ST, 1000.0, 50.0, prograde=False), integ, TERM)
    assert azimuthal_winding(retro) < 0 and abs(deflection_angle(retro) - deflection_angle(weak)) < 1e-9
    captured = integrate(ST, equatorial_photon(ST, 1000.0, 4.0), integ, TERM, record=False)
    with pytest.raises(ValueError):
        deflection_angle(captured)


def test_integrate_input_validation() -> None:
    with pytest.raises(ValueError):
        integrate(ST, np.zeros(7), IntegratorOptions(), TERM)
    with pytest.raises(ValueError):
        integrate_batch(ST, np.zeros((3, 7)), IntegratorOptions(), TERM)
    bad = pack([0.0, 10.0, 1.0, 0.0], [np.nan, 0.0, 0.0, 1.0])
    with pytest.raises(ValueError):
        integrate(ST, bad, IntegratorOptions(), TERM)
