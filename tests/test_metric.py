"""Kerr metric tests (PROJECT.md sections 5, 32 and 41; architecture.md 1.1).

Symbolic checks build the metric independently in ``tests/symbolic_kerr.py``
from the published line elements; numerical checks compare the closed-form
NumPy implementation against the lambdified symbolic expressions at random
points and test the Schwarzschild limit, symmetry, signature and the
determinant identity.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import sympy as sp

import symbolic_kerr as sk
from kerrray.geometry import (
    InverseMetricComponents,
    MetricComponents,
    Spacetime,
    delta,
    ergosphere_radius,
    inverse_metric,
    inverse_metric_components,
    inverse_metric_derivatives,
    kerr,
    metric,
    metric_components,
    outer_horizon,
    schwarzschild,
    sigma,
)

SPINS = (0.0, 0.3, -0.5, 0.9, 0.998)


def _stack(c: MetricComponents | InverseMetricComponents) -> np.ndarray:
    return np.stack([np.asarray(x, dtype=np.float64) for x in c])


def _diag_of(m: np.ndarray) -> np.ndarray:
    """The five independent entries ``(tt, tphi, rr, thth, phph)`` of a 4x4."""
    return np.array([m[0, 0], m[0, 3], m[1, 1], m[2, 2], m[3, 3]])


# --- Spacetime dataclass ------------------------------------------------------


def test_spacetime_defaults_and_properties() -> None:
    st = Spacetime()
    assert st.mass == 1.0 and st.spin == 0.0
    assert st.a == 0.0 and st.is_schwarzschild
    st2 = Spacetime(mass=2.0, spin=-0.25)
    assert st2.a == -0.5 and not st2.is_schwarzschild
    assert isinstance(Spacetime(mass=np.float64(1.5), spin=np.float32(0.5)).mass, float)


@pytest.mark.parametrize(
    "mass, spin",
    [(0.0, 0.0), (-1.0, 0.0), (np.nan, 0.0), (np.inf, 0.0), (1.0, 1.0), (1.0, -1.0), (1.0, 1.5), (1.0, np.nan)],
)
def test_spacetime_rejects_invalid_parameters(mass: float, spin: float) -> None:
    with pytest.raises(ValueError):
        Spacetime(mass=mass, spin=spin)


def test_spacetime_is_frozen() -> None:
    st = kerr(1.0, 0.5)
    with pytest.raises(dataclasses.FrozenInstanceError):
        st.spin = 0.1  # type: ignore[misc]


def test_constructors() -> None:
    assert schwarzschild().is_schwarzschild and schwarzschild(3.0).mass == 3.0
    st = kerr(2.0, 0.5)
    assert st.mass == 2.0 and st.spin == 0.5 and st.a == 1.0


def test_sigma_and_delta_match_their_definitions() -> None:
    st = kerr(1.5, 0.8)
    r = np.array([2.0, 3.0, 10.0])
    theta = np.array([0.3, 1.2, 2.9])
    np.testing.assert_allclose(sigma(st, r, theta), r**2 + st.a**2 * np.cos(theta) ** 2, rtol=1e-15)
    np.testing.assert_allclose(delta(st, r), r**2 - 2 * st.mass * r + st.a**2, rtol=1e-15)
    assert sigma(st, r[:, None], theta[None, :]).shape == (3, 3)


# --- Symbolic verification ----------------------------------------------------


def test_symbolic_mtw_and_bpt_forms_agree() -> None:
    s = sk.symbols()
    difference = (sk.mtw_metric(s) - sk.bpt_metric(s)).applyfunc(sp.simplify)
    assert difference == sp.zeros(4, 4)


def test_symbolic_metric_times_inverse_is_identity() -> None:
    s = sk.symbols()
    product = (sk.mtw_metric(s) * sk.closed_form_inverse(s)).applyfunc(sp.simplify)
    assert product == sp.eye(4)


def test_symbolic_determinant_identity() -> None:
    s = sk.symbols()
    sig, _ = sk.sigma_delta(s)
    residual = sp.simplify(sk.mtw_metric(s).det() + sig**2 * sp.sin(s.theta) ** 2)
    assert residual == 0


# --- Numerical agreement with the symbolic forms -----------------------------


@pytest.mark.parametrize("spin", SPINS)
def test_components_match_symbolic_metric(spin: float) -> None:
    st = kerr(1.0, spin)
    rng = np.random.default_rng(11)
    r, theta = sk.sample_points(st, rng, 6)
    g_fn, gi_fn = sk.lambdified_metric(), sk.lambdified_inverse()
    for ri, ti in zip(r, theta):
        sk.assert_relative_close(metric(st, ri, ti), g_fn(st.mass, st.a, ri, ti), 1e-12)
        sk.assert_relative_close(inverse_metric(st, ri, ti), gi_fn(st.mass, st.a, ri, ti), 1e-12)


@pytest.mark.parametrize("spin", SPINS)
def test_metric_times_inverse_is_identity_numerically(spin: float) -> None:
    st = kerr(1.0, spin)
    rng = np.random.default_rng(12)
    r, theta = sk.sample_points(st, rng, 50)
    g = metric(st, r, theta)
    gi = inverse_metric(st, r, theta)
    identity = np.broadcast_to(np.eye(4), g.shape)
    np.testing.assert_allclose(np.einsum("...ij,...jk->...ik", g, gi), identity, atol=1e-11)
    np.testing.assert_allclose(np.einsum("...ij,...jk->...ik", gi, g), identity, atol=1e-11)
    np.testing.assert_allclose(gi, np.linalg.inv(g), rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("spin", SPINS)
def test_inverse_metric_derivatives_match_sympy(spin: float) -> None:
    st = kerr(1.0, spin)
    rng = np.random.default_rng(13)
    r, theta = sk.sample_points(st, rng, 6)
    d_r_fn, d_th_fn = sk.lambdified_inverse_derivatives()
    for ri, ti in zip(r, theta):
        d_r, d_th = inverse_metric_derivatives(st, ri, ti)
        sk.assert_relative_close(_stack(d_r), _diag_of(d_r_fn(st.mass, st.a, ri, ti)), 1e-10)
        sk.assert_relative_close(_stack(d_th), _diag_of(d_th_fn(st.mass, st.a, ri, ti)), 1e-10)


# --- Schwarzschild limit ------------------------------------------------------


@pytest.mark.parametrize("mass", [1.0, 2.5])
def test_schwarzschild_limit_is_diagonal(mass: float) -> None:
    st = schwarzschild(mass)
    r = np.array([2.5, 4.0, 30.0]) * mass
    theta = np.array([0.4, 1.3, 2.7])
    f = 1.0 - 2.0 * mass / r
    c = metric_components(st, r, theta)
    np.testing.assert_allclose(c.g_tt, -f, rtol=1e-15)
    np.testing.assert_array_equal(c.g_tphi, 0.0)
    np.testing.assert_allclose(c.g_rr, 1.0 / f, rtol=1e-15)
    np.testing.assert_allclose(c.g_thth, r**2, rtol=1e-15)
    np.testing.assert_allclose(c.g_phph, r**2 * np.sin(theta) ** 2, rtol=1e-15)
    ci = inverse_metric_components(st, r, theta)
    np.testing.assert_allclose(ci.gtt, -1.0 / f, rtol=1e-14)
    np.testing.assert_array_equal(ci.gtphi, 0.0)
    np.testing.assert_allclose(ci.grr, f, rtol=1e-14)
    np.testing.assert_allclose(ci.gthth, 1.0 / r**2, rtol=1e-15)
    np.testing.assert_allclose(ci.gphph, 1.0 / (r**2 * np.sin(theta) ** 2), rtol=1e-14)


def test_small_spin_difference_from_schwarzschild_is_order_a() -> None:
    r, theta = 4.0, 1.1
    g0, gi0 = metric(schwarzschild(), r, theta), inverse_metric(schwarzschild(), r, theta)
    a1, a2 = 1e-3, 1e-4
    for reference, fn in ((g0, metric), (gi0, inverse_metric)):
        d1 = np.max(np.abs(fn(kerr(1.0, a1), r, theta) - reference))
        d2 = np.max(np.abs(fn(kerr(1.0, a2), r, theta) - reference))
        assert d1 > 0.0 and d2 > 0.0
        q1, q2 = d1 / a1, d2 / a2
        # d(a) = C a + O(a^2): the ratio d/a is constant up to O(a) corrections.
        assert abs(q1 - q2) / q1 < 1e-2
        assert d2 < d1


# --- Structure: symmetry, signature, determinant ------------------------------


@pytest.mark.parametrize("spin", SPINS)
def test_metric_is_symmetric_with_only_tphi_off_diagonal(spin: float) -> None:
    st = kerr(1.0, spin)
    rng = np.random.default_rng(14)
    r, theta = sk.sample_points(st, rng, 20)
    for m in (metric(st, r, theta), inverse_metric(st, r, theta)):
        np.testing.assert_array_equal(m, np.swapaxes(m, -1, -2))
        mask = ~np.eye(4, dtype=bool)
        mask[0, 3] = mask[3, 0] = False
        np.testing.assert_array_equal(m[..., mask], 0.0)
        if spin != 0.0:
            assert np.all(m[..., 0, 3] != 0.0)


@pytest.mark.parametrize("spin", SPINS)
def test_signature_is_lorentzian_outside_horizon(spin: float) -> None:
    st = kerr(1.0, spin)
    rng = np.random.default_rng(15)
    r, theta = sk.sample_points(st, rng, 20)
    r_plus = outer_horizon(st)
    # Add an equatorial point strictly between r_plus and r_E (inside the
    # ergosphere, outside the horizon); for a = 0 the two coincide, so use r_plus + 0.05.
    r_e = float(ergosphere_radius(st, np.pi / 2))
    r = np.append(r, 0.5 * (r_plus + r_e) if spin != 0.0 else r_plus + 0.05)
    theta = np.append(theta, np.pi / 2)
    for m in (metric(st, r, theta), inverse_metric(st, r, theta)):
        eig = np.linalg.eigvalsh(m)
        assert np.all(np.sum(eig < 0, axis=-1) == 1)
        assert np.all(np.sum(eig > 0, axis=-1) == 3)
    if spin != 0.0:
        assert metric_components(st, r[-1], theta[-1]).g_tt > 0.0  # inside the ergosphere


@pytest.mark.parametrize("spin", SPINS)
def test_determinant_identity_numerically(spin: float) -> None:
    st = kerr(1.0, spin)
    rng = np.random.default_rng(16)
    r, theta = sk.sample_points(st, rng, 30)
    expected = -sigma(st, r, theta) ** 2 * np.sin(theta) ** 2
    np.testing.assert_allclose(np.linalg.det(metric(st, r, theta)), expected, rtol=1e-10)
    np.testing.assert_allclose(np.linalg.det(inverse_metric(st, r, theta)), 1.0 / expected, rtol=1e-10)


def test_mass_scaling_of_components() -> None:
    """Dimensional consistency: g(M, r, theta) depends on r only through r / M."""
    lam = 2.5
    st1, st2 = kerr(1.0, 0.7), kerr(lam, 0.7)
    r, theta = np.array([3.0, 8.0]), np.array([0.7, 2.0])
    c1, c2 = metric_components(st1, r, theta), metric_components(st2, lam * r, theta)
    np.testing.assert_allclose(c2.g_tt, c1.g_tt, rtol=1e-14)
    np.testing.assert_allclose(c2.g_rr, c1.g_rr, rtol=1e-14)
    np.testing.assert_allclose(c2.g_tphi, lam * c1.g_tphi, rtol=1e-14)
    np.testing.assert_allclose(c2.g_thth, lam**2 * c1.g_thth, rtol=1e-14)
    np.testing.assert_allclose(c2.g_phph, lam**2 * c1.g_phph, rtol=1e-14)


# --- Vectorisation ------------------------------------------------------------


def test_broadcasting_shapes_and_dtype() -> None:
    st = kerr(1.0, 0.6)
    assert metric(st, 5.0, 1.0).shape == (4, 4)
    assert inverse_metric(st, 5.0, 1.0).shape == (4, 4)
    r = np.linspace(3.0, 6.0, 3)[:, None]
    theta = np.linspace(0.2, 3.0, 4)[None, :]
    assert metric(st, r, theta).shape == (3, 4, 4, 4)
    assert inverse_metric(st, r, theta).shape == (3, 4, 4, 4)
    for c in (metric_components(st, r, theta), inverse_metric_components(st, r, theta)):
        for x in c:
            assert x.shape == (3, 4) and x.dtype == np.float64
    for comps in inverse_metric_derivatives(st, r, theta):
        for x in comps:
            assert x.shape == (3, 4) and x.dtype == np.float64
    assert metric(st, [3, 4], 1).shape == (2, 4, 4)
    assert metric(st, [3, 4], 1).dtype == np.float64
