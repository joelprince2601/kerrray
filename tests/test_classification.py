"""Termination rules (PROJECT.md section 11; docs/architecture.md section 1; D-002)."""

from __future__ import annotations

import numpy as np
import pytest

from kerrray.geodesics import (
    IDX_PPH,
    IDX_R,
    IDX_TH,
    EventOptions,
    IntegratorOptions,
    TerminationOptions,
    equatorial_photon,
    geodesic_rhs,
    integrate,
    integrate_batch,
    photon_from_constants,
)
from kerrray.geometry import kerr, outer_horizon, schwarzschild
from kerrray.photons import (
    AXIS_SIN_TOLERANCE,
    THETA_DOMAIN_TOLERANCE,
    TerminationState,
    classify_batch,
    classify_state,
)

ST = schwarzschild()
INTEG = IntegratorOptions()
TERM = TerminationOptions(horizon_epsilon=1e-6, escape_radius=1000.0)


def _state(r: float, theta: float = 1.0, lz: float = 1.0) -> np.ndarray:
    return np.array([0.0, r, theta, 0.0, -1.0, 0.0, 0.0, lz])


def _deriv(dr: float) -> np.ndarray:
    d = np.zeros(8)
    d[IDX_R] = dr
    return d


def test_termination_state_values() -> None:
    assert [s.value for s in TerminationState] == [0, 1, 2, 3, 4, 5, 6]
    assert TerminationState.RUNNING == 0 and TerminationState.DISK_HIT == 6


def test_captured_rule_uses_horizon_epsilon() -> None:
    r_plus = outer_horizon(ST)
    y_in = _state(r_plus + 0.5e-6)
    y_out = _state(r_plus + 2e-6)
    assert classify_state(ST, y_in, _deriv(-1.0), TERM, lam=0.0, integ=INTEG) == TerminationState.CAPTURED
    assert classify_state(ST, y_out, _deriv(-1.0), TERM, lam=0.0, integ=INTEG) == TerminationState.RUNNING


def test_escaped_requires_outward_motion_decision_d002() -> None:
    y = _state(TERM.escape_radius)
    assert classify_state(ST, y, _deriv(-1.0), TERM, lam=0.0, integ=INTEG) == TerminationState.RUNNING
    assert classify_state(ST, y, _deriv(+1.0), TERM, lam=0.0, integ=INTEG) == TerminationState.ESCAPED
    assert classify_state(ST, _state(2000.0), _deriv(0.0), TERM, lam=0.0, integ=INTEG) == TerminationState.RUNNING
    # The real launch states: inward from r0 = escape_radius is not escaped at step 0.
    y_in = equatorial_photon(ST, TERM.escape_radius, 6.0, inward=True)
    y_out = equatorial_photon(ST, TERM.escape_radius, 6.0, inward=False)
    assert classify_state(ST, y_in, geodesic_rhs(ST, y_in), TERM, lam=0.0, integ=INTEG) == TerminationState.RUNNING
    assert classify_state(ST, y_out, geodesic_rhs(ST, y_out), TERM, lam=0.0, integ=INTEG) == TerminationState.ESCAPED


def test_numerical_failure_on_nonfinite_state_or_small_step() -> None:
    y = _state(10.0)
    y_nan = y.copy()
    y_nan[IDX_TH] = np.nan
    assert classify_state(ST, y_nan, _deriv(0.0), TERM, lam=0.0, integ=INTEG) == TerminationState.NUMERICAL_FAILURE
    d_inf = _deriv(np.inf)
    assert classify_state(ST, y, d_inf, TERM, lam=0.0, integ=INTEG) == TerminationState.NUMERICAL_FAILURE
    assert classify_state(ST, y, _deriv(0.0), TERM, lam=0.0, integ=INTEG, h=0.5 * INTEG.h_min) == TerminationState.NUMERICAL_FAILURE
    assert classify_state(ST, y, _deriv(0.0), TERM, lam=0.0, integ=INTEG, h=2 * INTEG.h_min) == TerminationState.RUNNING


def test_out_of_domain_rules() -> None:
    assert classify_state(ST, _state(-0.5), _deriv(0.0), TERM, lam=0.0, integ=INTEG) == TerminationState.OUT_OF_DOMAIN
    below = _state(10.0, theta=-2 * THETA_DOMAIN_TOLERANCE)
    above = _state(10.0, theta=np.pi + 2 * THETA_DOMAIN_TOLERANCE)
    inside = _state(10.0, theta=np.pi + 0.5 * THETA_DOMAIN_TOLERANCE)
    assert classify_state(ST, below, _deriv(0.0), TERM, lam=0.0, integ=INTEG) == TerminationState.OUT_OF_DOMAIN
    assert classify_state(ST, above, _deriv(0.0), TERM, lam=0.0, integ=INTEG) == TerminationState.OUT_OF_DOMAIN
    assert classify_state(ST, inside, _deriv(0.0), TERM, lam=0.0, integ=INTEG) == TerminationState.RUNNING


def test_polar_axis_rule_depends_on_lz() -> None:
    on_axis_lz = _state(10.0, theta=0.0, lz=0.3)
    on_axis_no_lz = _state(10.0, theta=0.0, lz=0.0)
    near_axis_lz = _state(10.0, theta=0.5 * AXIS_SIN_TOLERANCE, lz=0.3)
    off_axis_lz = _state(10.0, theta=1e-6, lz=0.3)
    assert classify_state(ST, on_axis_lz, _deriv(0.0), TERM, lam=0.0, integ=INTEG) == TerminationState.OUT_OF_DOMAIN
    assert classify_state(ST, near_axis_lz, _deriv(0.0), TERM, lam=0.0, integ=INTEG) == TerminationState.OUT_OF_DOMAIN
    assert classify_state(ST, on_axis_no_lz, _deriv(0.0), TERM, lam=0.0, integ=INTEG) == TerminationState.RUNNING
    assert classify_state(ST, off_axis_lz, _deriv(0.0), TERM, lam=0.0, integ=INTEG) == TerminationState.RUNNING


def test_max_affine_parameter_by_lambda_and_by_steps() -> None:
    y = _state(10.0)
    assert classify_state(ST, y, _deriv(0.0), TERM, lam=INTEG.lambda_max, integ=INTEG) == TerminationState.MAX_AFFINE_PARAMETER
    assert classify_state(ST, y, _deriv(0.0), TERM, lam=0.0, integ=INTEG, n_steps=INTEG.max_steps) == TerminationState.MAX_AFFINE_PARAMETER
    assert classify_state(ST, y, _deriv(0.0), TERM, lam=0.0, integ=INTEG, n_steps=INTEG.max_steps - 1) == TerminationState.RUNNING


def test_priority_order_when_several_rules_hold() -> None:
    r_plus = outer_horizon(ST)
    y = _state(r_plus + 0.5e-6)  # captured ...
    y[IDX_TH] = -1.0  # ... and out of domain
    assert classify_state(ST, y, _deriv(-1.0), TERM, lam=INTEG.lambda_max, integ=INTEG) == TerminationState.OUT_OF_DOMAIN
    y[IDX_TH] = np.nan  # non-finite beats everything
    assert classify_state(ST, y, _deriv(-1.0), TERM, lam=INTEG.lambda_max, integ=INTEG) == TerminationState.NUMERICAL_FAILURE
    y_esc = _state(2 * TERM.escape_radius)
    assert classify_state(ST, y_esc, _deriv(1.0), TERM, lam=INTEG.lambda_max, integ=INTEG) == TerminationState.ESCAPED


def test_classify_batch_vectorised_matches_scalar() -> None:
    r_plus = outer_horizon(ST)
    Y = np.stack([_state(r_plus + 1e-7), _state(1500.0), _state(1500.0), _state(10.0), _state(-1.0), _state(10.0)])
    Y[5, IDX_TH] = np.inf
    D = np.stack([_deriv(-1.0), _deriv(1.0), _deriv(-1.0), _deriv(0.0), _deriv(0.0), _deriv(0.0)])
    lam = np.array([0.0, 0.0, 0.0, INTEG.lambda_max, 0.0, 0.0])
    codes = classify_batch(ST, Y, D, TERM, lam=lam, integ=INTEG)
    expected = [
        TerminationState.CAPTURED,
        TerminationState.ESCAPED,
        TerminationState.RUNNING,
        TerminationState.MAX_AFFINE_PARAMETER,
        TerminationState.OUT_OF_DOMAIN,
        TerminationState.NUMERICAL_FAILURE,
    ]
    assert codes.tolist() == [int(s) for s in expected]
    for k in range(len(expected)):
        assert classify_state(ST, Y[k], D[k], TERM, lam=float(lam[k]), integ=INTEG) == expected[k]
    with pytest.raises(ValueError):
        classify_batch(ST, Y, D[:2], TERM, lam=lam, integ=INTEG)


def test_schwarzschild_b4_captured_b6_escaped() -> None:
    """Inputs b = 4 and b = 6 from r0 = 1000 (no critical value is asserted here)."""
    integ = IntegratorOptions(method="rk45", rtol=1e-9, atol=1e-11)
    captured = integrate(ST, equatorial_photon(ST, 1000.0, 4.0), integ, TERM, record=False)
    escaped = integrate(ST, equatorial_photon(ST, 1000.0, 6.0), integ, TERM, record=False)
    assert captured.state == TerminationState.CAPTURED
    assert captured.y[-1, IDX_R] <= outer_horizon(ST) + TERM.horizon_epsilon
    assert escaped.state == TerminationState.ESCAPED
    assert escaped.y[-1, IDX_R] >= TERM.escape_radius
    assert geodesic_rhs(ST, escaped.y[-1])[IDX_R] > 0.0


def test_max_steps_exhaustion_is_max_affine_parameter() -> None:
    integ = IntegratorOptions(method="rk45", max_steps=25)
    traj = integrate(ST, equatorial_photon(ST, 1000.0, 6.0), integ, TERM, record=False)
    assert traj.state == TerminationState.MAX_AFFINE_PARAMETER and traj.n_steps == 25
    batch = integrate_batch(ST, equatorial_photon(ST, 1000.0, 6.0)[None, :], integ, TERM)
    assert batch.state[0] == TerminationState.MAX_AFFINE_PARAMETER and batch.n_steps[0] == 25


def test_lambda_max_termination_reaches_budget_exactly() -> None:
    integ = IntegratorOptions(method="rk45", lambda_max=37.5)
    traj = integrate(ST, equatorial_photon(ST, 1000.0, 6.0), integ, TERM, record=True)
    assert traj.state == TerminationState.MAX_AFFINE_PARAMETER
    assert traj.lam[-1] == 37.5 and np.all(np.diff(traj.lam) > 0)


def test_disk_plane_event_in_batch() -> None:
    st = kerr(1.0, 0.9)
    # Off-equatorial rays launched towards the equator; they cross theta = pi/2.
    Y0 = np.stack(
        [photon_from_constants(st, 60.0, 1.2, 1.0, lz, 15.0, sign_r=-1, sign_theta=1) for lz in (-4.0, 2.0, 5.0)]
    )
    integ = IntegratorOptions(method="rk45", rtol=1e-9, atol=1e-11)
    term = TerminationOptions(escape_radius=200.0)
    wide = integrate_batch(st, Y0, integ, term, events=EventOptions(disk_plane=True, r_in=1.0, r_out=1e3))
    assert wide.event_hit is not None and wide.event_Y is not None
    assert np.all(wide.event_hit) and np.all(wide.state == TerminationState.DISK_HIT)
    assert np.all(wide.event_Y[:, IDX_TH] == 0.5 * np.pi)
    assert np.array_equal(wide.Y, wide.event_Y)
    assert np.all((wide.event_Y[:, IDX_R] >= 1.0) & (wide.event_Y[:, IDX_R] <= 1e3))
    assert np.all(wide.lam > 0)
    # A ring that the crossings miss: the rays continue to their natural end.
    narrow = integrate_batch(st, Y0, integ, term, events=EventOptions(disk_plane=True, r_in=1e5, r_out=2e5))
    assert not np.any(narrow.event_hit)
    assert np.all(np.isin(narrow.state, [TerminationState.CAPTURED, TerminationState.ESCAPED]))
    plain = integrate_batch(st, Y0, integ, term)
    assert plain.event_hit is None and plain.event_Y is None
    assert np.array_equal(plain.state, narrow.state)


def test_option_validation() -> None:
    with pytest.raises(ValueError):
        IntegratorOptions(method="euler")
    with pytest.raises(ValueError):
        IntegratorOptions(rtol=0.0)
    with pytest.raises(ValueError):
        IntegratorOptions(max_steps=0)
    with pytest.raises(ValueError):
        IntegratorOptions(dtype="float16")
    with pytest.raises(ValueError):
        TerminationOptions(horizon_epsilon=-1e-6)
    with pytest.raises(ValueError):
        TerminationOptions(escape_radius=0.0)
    with pytest.raises(ValueError):
        EventOptions(disk_plane=True, r_in=10.0, r_out=5.0)
    assert IntegratorOptions(dtype="float32").np_dtype == np.float32
