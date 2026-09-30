"""State layout, Hamiltonian right-hand side, Christoffel cross-check, null momentum."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import sympy as sp
from scipy.integrate import solve_ivp

from kerrray.geodesics import (
    IDX_PH,
    IDX_PPH,
    IDX_PR,
    IDX_PT,
    IDX_PTH,
    IDX_R,
    IDX_T,
    IDX_TH,
    PhotonState,
    christoffel_to_hamiltonian_state,
    geodesic_rhs,
    geodesic_rhs_christoffel,
    hamiltonian,
    hamiltonian_to_christoffel_state,
    null_momentum_pt,
    pack,
    photon_from_constants,
    unpack,
)
from kerrray.geometry import Spacetime, kerr, outer_horizon, schwarzschild


def _sympy_rhs():
    """Hamilton's equations from a SymPy differentiation of H = (1/2) g^{mu nu} p_mu p_nu.

    The inverse metric is typed independently from docs/architecture.md
    section 1.1 (closed form), not taken from the package.
    """
    r, th = sp.symbols("r theta", real=True)
    mass, a = sp.symbols("M a", real=True)
    pt, pr, pth, pph = sp.symbols("p_t p_r p_theta p_phi", real=True)
    sigma = r**2 + a**2 * sp.cos(th) ** 2
    delta = r**2 - 2 * mass * r + a**2
    big_a = (r**2 + a**2) ** 2 - a**2 * delta * sp.sin(th) ** 2
    gtt = -big_a / (sigma * delta)
    gtph = -2 * mass * a * r / (sigma * delta)
    grr = delta / sigma
    gthth = 1 / sigma
    gphph = (delta - a**2 * sp.sin(th) ** 2) / (sigma * delta * sp.sin(th) ** 2)
    ham = sp.Rational(1, 2) * (gtt * pt**2 + 2 * gtph * pt * pph + grr * pr**2 + gthth * pth**2 + gphph * pph**2)
    rhs = [sp.diff(ham, pt), sp.diff(ham, pr), sp.diff(ham, pth), sp.diff(ham, pph),
           sp.Integer(0), -sp.diff(ham, r), -sp.diff(ham, th), sp.Integer(0)]
    return sp.lambdify((mass, a, r, th, pt, pr, pth, pph), rhs, "numpy"), sp.lambdify(
        (mass, a, r, th, pt, pr, pth, pph), ham, "numpy"
    )


def _random_states(st: Spacetime, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y = np.zeros((n, 8))
    y[:, IDX_R] = rng.uniform(outer_horizon(st) + 0.5, 30.0, n)
    y[:, IDX_TH] = rng.uniform(0.1, np.pi - 0.1, n)
    y[:, IDX_PH] = rng.uniform(0, 2 * np.pi, n)
    y[:, 4:] = rng.normal(size=(n, 4))
    return y


def test_index_constants_and_pack_unpack() -> None:
    assert (IDX_T, IDX_R, IDX_TH, IDX_PH, IDX_PT, IDX_PR, IDX_PTH, IDX_PPH) == tuple(range(8))
    x = np.arange(4.0)
    p = np.arange(10.0, 14.0)
    y = pack(x, p)
    assert y.shape == (8,)
    xx, pp = unpack(y)
    assert np.array_equal(xx, x) and np.array_equal(pp, p)
    y_batch = pack(np.tile(x, (3, 1)), p)
    assert y_batch.shape == (3, 8) and np.array_equal(y_batch[2, 4:], p)
    with pytest.raises(ValueError):
        unpack(np.zeros(7))


def test_photon_state_roundtrip() -> None:
    y = np.arange(8.0)
    s = PhotonState.from_array(y)
    assert np.array_equal(s.to_array(), y)
    assert s.energy == -y[IDX_PT] and s.angular_momentum == y[IDX_PPH]
    with pytest.raises(ValueError):
        PhotonState(np.zeros(3), np.zeros(4))


@pytest.mark.parametrize("spin", [0.0, 0.9, -0.6])
def test_hamiltonian_rhs_matches_sympy_gradient(spin: float) -> None:
    st = kerr(1.0, spin)
    rhs_fn, ham_fn = _sympy_rhs()
    ys = _random_states(st, 25, seed=int(10 * abs(spin)) + 3)
    num = geodesic_rhs(st, ys)
    ham = hamiltonian(st, ys)
    for y, f, h in zip(ys, num, ham, strict=True):
        ref = np.array(rhs_fn(st.mass, st.a, *y[1:3], *y[4:]), dtype=float)
        scale = np.maximum(1.0, np.abs(ref))
        assert np.max(np.abs(f - ref) / scale) < 1e-10
        h_ref = float(ham_fn(st.mass, st.a, *y[1:3], *y[4:]))
        assert abs(h - h_ref) <= 1e-10 * max(1.0, abs(h_ref))


def test_rhs_conserves_p_t_and_p_phi_identically() -> None:
    st = kerr(1.0, 0.7)
    f = geodesic_rhs(st, _random_states(st, 10, seed=1))
    assert np.all(f[:, IDX_PT] == 0.0) and np.all(f[:, IDX_PPH] == 0.0)


def test_rhs_shapes_and_dtypes() -> None:
    st = kerr(1.0, 0.5)
    ys = _random_states(st, 6, seed=2)
    assert geodesic_rhs(st, ys).shape == (6, 8)
    assert geodesic_rhs(st, ys[0]).shape == (8,)
    f32 = geodesic_rhs(st, ys.astype(np.float32))
    assert f32.dtype == np.float32
    assert np.allclose(f32, geodesic_rhs(st, ys), rtol=1e-4, atol=1e-4)


def test_schwarzschild_limit_of_rhs() -> None:
    ys = _random_states(schwarzschild(), 8, seed=4)
    f0 = geodesic_rhs(schwarzschild(), ys)
    f_small = geodesic_rhs(kerr(1.0, 1e-7), ys)
    assert np.max(np.abs(f_small - f0) / np.maximum(1.0, np.abs(f0))) < 1e-5


def test_state_conversions_roundtrip() -> None:
    st = kerr(1.0, 0.9)
    ys = _random_states(st, 5, seed=5)
    z = hamiltonian_to_christoffel_state(st, ys)
    assert np.array_equal(z[:, :4], ys[:, :4])
    back = christoffel_to_hamiltonian_state(st, z)
    assert np.max(np.abs(back - ys)) < 1e-12


def test_christoffel_form_agrees_with_hamiltonian_form() -> None:
    """The same ray integrated in both forms with SciPy DOP853 at tight tolerance."""
    st = kerr(1.0, 0.9)
    y0 = photon_from_constants(st, 30.0, 1.1, 1.0, 3.0, 6.0, sign_r=-1, sign_theta=1)
    z0 = hamiltonian_to_christoffel_state(st, y0)
    lam_end = 60.0
    sol_h = solve_ivp(lambda _l, y: geodesic_rhs(st, y), (0.0, lam_end), y0, method="DOP853", rtol=1e-12, atol=1e-14)
    sol_c = solve_ivp(
        lambda _l, z: geodesic_rhs_christoffel(st, z), (0.0, lam_end), z0, method="DOP853", rtol=1e-12, atol=1e-14
    )
    assert sol_h.success and sol_c.success
    y_h = sol_h.y[:, -1]
    y_c = christoffel_to_hamiltonian_state(st, sol_c.y[:, -1])
    assert y_h[IDX_R] > outer_horizon(st) + 1.0  # still outside, meaningful comparison
    assert np.max(np.abs(y_h - y_c)) < 1e-7
    # The Christoffel form also preserves the null condition.
    assert abs(hamiltonian(st, y_c)) < 1e-9


def test_null_momentum_pt_gives_null_future_directed_states() -> None:
    st = kerr(1.0, 0.8)
    rng = np.random.default_rng(7)
    n = 40
    x = np.zeros((n, 4))
    x[:, 1] = rng.uniform(outer_horizon(st) + 0.3, 50.0, n)
    x[:, 2] = rng.uniform(0.05, np.pi - 0.05, n)
    p_r, p_th, p_ph = rng.normal(size=(3, n))
    p_t = null_momentum_pt(st, x, p_r, p_th, p_ph)
    y = pack(x, np.stack([p_t, p_r, p_th, p_ph], axis=1))
    scale = np.sum(y[:, 4:] ** 2, axis=1)
    assert np.max(np.abs(hamiltonian(st, y)) / scale) < 1e-12
    assert np.all(geodesic_rhs(st, y)[:, IDX_T] > 0.0)  # dt/dlambda > 0
    p_t_past = null_momentum_pt(st, x, p_r, p_th, p_ph, future_directed=False)
    y_past = pack(x, np.stack([p_t_past, p_r, p_th, p_ph], axis=1))
    assert np.all(geodesic_rhs(st, y_past)[:, IDX_T] < 0.0)
    # Outside the ergosphere the future-directed root has E > 0.
    far = x[:, 1] > 2.5
    assert np.all(-p_t[far] > 0.0)


def test_null_momentum_pt_raises_inside_horizon() -> None:
    st = schwarzschild()
    with pytest.raises(ValueError):
        null_momentum_pt(st, [0.0, 1.5, 1.0, 0.0], 1.0, 0.0, 1.0)


def test_polar_axis_guard_for_zero_lz() -> None:
    """Exactly on the axis with L_z = 0 the equations are finite and warning-free."""
    st = kerr(1.0, 0.9)
    y = photon_from_constants(st, 20.0, 0.0, 1.0, 0.0, 4.0, sign_r=-1, sign_theta=1)
    assert y[IDX_TH] == 0.0 and y[IDX_PPH] == 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        f = geodesic_rhs(st, y)
        h = hamiltonian(st, y)
    assert np.all(np.isfinite(f))
    assert abs(h) < 1e-12
    # The finite limit: with L_z = 0, dphi/dlambda = g^{tphi} p_t and the polar
    # force vanishes on the axis (all d_theta terms carry sin(2 theta)).
    assert f[IDX_PTH] == 0.0
