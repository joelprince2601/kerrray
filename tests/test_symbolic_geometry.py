"""Symbolic verification of the Kerr geometry and cross-checks of the numerical layer.

Part 1 tests ``kerrray.geometry.symbolic`` on its own (architecture section
1.1: g times g^-1 = I, the Schwarzschild limit, the vacuum condition
R_{mu nu} = 0 at random points). Part 2 cross-checks the NumPy geometry
(``kerrray.geometry.metric``, ``kerrray.geometry.christoffel``, written by
the geometry role) against the lambdified symbolic objects at random points.
Those modules are imported inside the test functions: until they exist the
Part 2 tests fail, which is expected before integration.
"""

from __future__ import annotations

import numpy as np
import pytest

from kerrray.geometry import symbolic as sym

MASS = 1.0
SPINS = [0.0, 0.3, 0.7, 0.95, -0.6]
N_POINTS = 20
# Round-off in R_{mu nu} is a few epsilon times the Gamma-Gamma scale (measured
# worst |R| / scale = 1.6e-16 over 100 points); 1e-12 leaves four orders of margin.
RICCI_RELATIVE_TOL = 1e-12
# Cross-check tolerance between the closed-form NumPy geometry and SymPy:
# both are exact expressions evaluated in float64 (round-off of order 1e-15).
CROSS_RTOL = 1e-11
CROSS_ATOL = 1e-13


def random_points(rng: np.random.Generator, a: float, n: int = N_POINTS) -> list[tuple[float, float]]:
    """Random (r, theta) outside the horizon and away from the axis."""
    r_plus = MASS + np.sqrt(MASS**2 - a**2)
    r = rng.uniform(r_plus + 0.25 * MASS, 12.0 * MASS, size=n)
    theta = rng.uniform(0.15, np.pi - 0.15, size=n)
    return list(zip(r.tolist(), theta.tolist()))


# --------------------------------------------------------------------------
# Part 1: symbolic self-consistency
# --------------------------------------------------------------------------


def test_metric_is_symmetric_with_only_t_phi_off_diagonal() -> None:
    g, _ = sym.symbolic_metric()
    assert g == g.T
    for i in range(4):
        for j in range(4):
            if i != j and {i, j} != {0, 3}:
                assert g[i, j] == 0


def test_metric_times_inverse_is_identity() -> None:
    import sympy as sp

    g, _ = sym.symbolic_metric()
    ginv = sym.symbolic_inverse_metric()
    assert (g * ginv).applyfunc(sp.simplify) == sp.eye(4)
    assert (ginv * g).applyfunc(sp.simplify) == sp.eye(4)


def test_schwarzschild_limit() -> None:
    """a -> 0 must give diag(-(1 - 2M/r), 1/(1 - 2M/r), r^2, r^2 sin^2 theta) (MTW eq. 23.1)."""
    import sympy as sp

    g, s = sym.symbolic_metric()
    f = 1 - 2 * s.mass / s.r
    expected = sp.diag(-f, 1 / f, s.r**2, s.r**2 * sp.sin(s.theta) ** 2)
    assert (g.subs(s.a, 0) - expected).applyfunc(sp.simplify) == sp.zeros(4, 4)
    ginv = sym.symbolic_inverse_metric()
    assert (ginv.subs(s.a, 0) - expected.inv()).applyfunc(sp.simplify) == sp.zeros(4, 4)


def test_metric_determinant() -> None:
    """det g = -Sigma^2 sin^2(theta) (Chandrasekhar 1983 ch. 6; Visser 2007)."""
    import sympy as sp

    g, s = sym.symbolic_metric()
    sigma = s.r**2 + s.a**2 * sp.cos(s.theta) ** 2
    assert sp.simplify(g.det() + sigma**2 * sp.sin(s.theta) ** 2) == 0


@pytest.mark.parametrize("spin", [0.3, 0.95])
def test_frame_dragging_angular_velocity_sign(spin: float) -> None:
    """omega = -g_tphi / g_phiphi > 0 for a > 0: the hole drags towards +phi (architecture section 1)."""
    rng = np.random.default_rng(1)
    g = sym.lambdified_metric()
    for r, theta in random_points(rng, spin * MASS, n=5):
        comps = g(MASS, spin * MASS, r, theta)
        assert -comps[0, 3] / comps[3, 3] > 0.0
        comps_neg = g(MASS, -spin * MASS, r, theta)
        assert -comps_neg[0, 3] / comps_neg[3, 3] < 0.0


def test_christoffel_symmetric_in_lower_indices() -> None:
    gamma = sym.symbolic_christoffel()
    assert gamma.shape == (4, 4, 4)
    f = sym.lambdified_christoffel()
    rng = np.random.default_rng(2)
    for r, theta in random_points(rng, 0.7 * MASS, n=5):
        val = f(MASS, 0.7 * MASS, r, theta)
        assert val.shape == (4, 4, 4)
        np.testing.assert_allclose(val, np.swapaxes(val, 1, 2), rtol=1e-13, atol=1e-15)


def test_christoffel_metric_compatibility() -> None:
    """d_alpha g_{mu nu} = Gamma^lambda_{alpha mu} g_{lambda nu} + Gamma^lambda_{alpha nu} g_{mu lambda}.

    Holds identically for the Levi-Civita connection and pins down the index
    order [mu, alpha, beta] and the sign convention of the implementation.
    """
    import sympy as sp

    g, s = sym.symbolic_metric()
    x = sym.coordinates()
    dg_sym = [g.applyfunc(lambda e, xk=xk: sp.diff(e, xk)) for xk in x]
    dg = sp.lambdify((s.mass, s.a, s.r, s.theta), dg_sym, modules="numpy", cse=True)
    gf = sym.lambdified_metric()
    gam = sym.lambdified_christoffel()
    rng = np.random.default_rng(3)
    for spin in (0.0, 0.7):
        a = spin * MASS
        for r, theta in random_points(rng, a, n=5):
            gv = gf(MASS, a, r, theta)
            gm = gam(MASS, a, r, theta)
            dgv = np.asarray(dg(MASS, a, r, theta), dtype=float)  # [alpha, mu, nu]
            recon = np.einsum("lam,ln->amn", gm, gv) + np.einsum("lan,ml->amn", gm, gv)
            np.testing.assert_allclose(dgv, recon, rtol=1e-12, atol=1e-13)


def test_schwarzschild_christoffels_match_closed_form() -> None:
    """a = 0 Christoffels versus the Schwarzschild list of Carroll 2004 eq. (5.53)."""
    gam = sym.lambdified_christoffel()
    rng = np.random.default_rng(4)
    for r, th in random_points(rng, 0.0, n=5):
        g = gam(MASS, 0.0, r, th)
        expected = {
            (0, 0, 1): MASS / (r * (r - 2 * MASS)),
            (1, 0, 0): MASS * (r - 2 * MASS) / r**3,
            (1, 1, 1): -MASS / (r * (r - 2 * MASS)),
            (1, 2, 2): -(r - 2 * MASS),
            (1, 3, 3): -(r - 2 * MASS) * np.sin(th) ** 2,
            (2, 1, 2): 1.0 / r,
            (2, 3, 3): -np.sin(th) * np.cos(th),
            (3, 1, 3): 1.0 / r,
            (3, 2, 3): np.cos(th) / np.sin(th),
        }
        for (mu, al, be), value in expected.items():
            assert g[mu, al, be] == pytest.approx(value, rel=1e-12, abs=1e-14)
            assert g[mu, be, al] == pytest.approx(value, rel=1e-12, abs=1e-14)
        # every component not in the list (or its mirror) vanishes
        listed = set(expected) | {(mu, be, al) for (mu, al, be) in expected}
        for idx in np.ndindex(4, 4, 4):
            if idx not in listed:
                assert g[idx] == pytest.approx(0.0, abs=1e-14)


@pytest.mark.parametrize("spin", SPINS)
def test_ricci_tensor_vanishes(spin: float) -> None:
    """Kerr is a vacuum solution: R_{mu nu} = 0 relative to the Gamma-Gamma scale."""
    a = spin * MASS
    rng = np.random.default_rng(int(abs(spin) * 1000) + (10 if spin < 0 else 0))
    worst = 0.0
    for r, theta in random_points(rng, a):
        ricci = sym.ricci_tensor_numeric(MASS, a, r, theta)
        assert ricci.shape == (4, 4)
        assert np.all(np.isfinite(ricci))
        scale = sym.christoffel_scale(MASS, a, r, theta)
        assert scale > 0.0
        residual = np.abs(ricci).max() / max(scale, 1.0)
        worst = max(worst, residual)
    assert worst < RICCI_RELATIVE_TOL


def test_ricci_symbolic_is_symmetric_numerically() -> None:
    a = 0.7 * MASS
    ricci = sym.ricci_tensor_numeric(MASS, a, 4.0, 1.0)
    scale = sym.christoffel_scale(MASS, a, 4.0, 1.0)
    np.testing.assert_allclose(ricci, ricci.T, atol=RICCI_RELATIVE_TOL * scale, rtol=0)


def test_lambdified_inverse_metric_derivatives_are_symmetric_and_finite() -> None:
    d_r, d_th = sym.lambdified_inverse_metric_derivatives()(MASS, 0.5, 3.7, 0.9)
    for d in (d_r, d_th):
        assert d.shape == (4, 4)
        assert np.all(np.isfinite(d))
        np.testing.assert_allclose(d, d.T, rtol=0, atol=0)
    # d_theta g^{rr} = -Delta d_theta(Sigma) / Sigma^2 = 2 a^2 Delta sin cos / Sigma^2
    r, th, a = 3.7, 0.9, 0.5
    sigma = r * r + a * a * np.cos(th) ** 2
    delta = r * r - 2 * MASS * r + a * a
    assert d_th[1, 1] == pytest.approx(2 * a * a * delta * np.sin(th) * np.cos(th) / sigma**2, rel=1e-12)


def test_lambdified_callables_are_cached() -> None:
    assert sym.lambdified_christoffel() is sym.lambdified_christoffel()
    assert sym.lambdified_ricci() is sym.lambdified_ricci()
    assert sym.symbolic_metric() is sym.symbolic_metric()


# --------------------------------------------------------------------------
# Part 2: cross-checks of the numerical geometry (geometry role) against SymPy
# --------------------------------------------------------------------------


def _spacetime(spin: float):
    from kerrray.geometry.metric import Spacetime

    return Spacetime(mass=MASS, spin=spin)


@pytest.mark.parametrize("spin", SPINS)
def test_numeric_metric_matches_symbolic(spin: float) -> None:
    from kerrray.geometry.metric import metric, metric_components

    st = _spacetime(spin)
    a = st.a
    ref = sym.lambdified_metric()
    rng = np.random.default_rng(11)
    for r, theta in random_points(rng, a, n=8):
        expected = ref(MASS, a, r, theta)
        np.testing.assert_allclose(np.asarray(metric(st, r, theta)), expected, rtol=CROSS_RTOL, atol=CROSS_ATOL)
        comps = metric_components(st, r, theta)
        got = [comps.g_tt, comps.g_tphi, comps.g_rr, comps.g_thth, comps.g_phph]
        want = [expected[0, 0], expected[0, 3], expected[1, 1], expected[2, 2], expected[3, 3]]
        np.testing.assert_allclose(np.asarray(got, dtype=float), want, rtol=CROSS_RTOL, atol=CROSS_ATOL)


@pytest.mark.parametrize("spin", SPINS)
def test_numeric_inverse_metric_matches_symbolic(spin: float) -> None:
    from kerrray.geometry.metric import inverse_metric, inverse_metric_components

    st = _spacetime(spin)
    a = st.a
    ref = sym.lambdified_inverse_metric()
    rng = np.random.default_rng(12)
    for r, theta in random_points(rng, a, n=8):
        expected = ref(MASS, a, r, theta)
        np.testing.assert_allclose(np.asarray(inverse_metric(st, r, theta)), expected, rtol=CROSS_RTOL, atol=CROSS_ATOL)
        comps = inverse_metric_components(st, r, theta)
        got = [comps.gtt, comps.gtphi, comps.grr, comps.gthth, comps.gphph]
        want = [expected[0, 0], expected[0, 3], expected[1, 1], expected[2, 2], expected[3, 3]]
        np.testing.assert_allclose(np.asarray(got, dtype=float), want, rtol=CROSS_RTOL, atol=CROSS_ATOL)


@pytest.mark.parametrize("spin", SPINS)
def test_numeric_inverse_metric_derivatives_match_symbolic(spin: float) -> None:
    from kerrray.geometry.metric import inverse_metric_derivatives

    st = _spacetime(spin)
    a = st.a
    ref = sym.lambdified_inverse_metric_derivatives()
    rng = np.random.default_rng(13)
    for r, theta in random_points(rng, a, n=8):
        d_r, d_th = ref(MASS, a, r, theta)
        got_r, got_th = inverse_metric_derivatives(st, r, theta)
        for got, want in ((got_r, d_r), (got_th, d_th)):
            values = [got.gtt, got.gtphi, got.grr, got.gthth, got.gphph]
            expected = [want[0, 0], want[0, 3], want[1, 1], want[2, 2], want[3, 3]]
            np.testing.assert_allclose(np.asarray(values, dtype=float), expected, rtol=CROSS_RTOL, atol=CROSS_ATOL)


@pytest.mark.parametrize("spin", SPINS)
def test_numeric_christoffel_matches_symbolic(spin: float) -> None:
    from kerrray.geometry.christoffel import christoffel

    st = _spacetime(spin)
    a = st.a
    ref = sym.lambdified_christoffel()
    rng = np.random.default_rng(14)
    for r, theta in random_points(rng, a, n=8):
        expected = ref(MASS, a, r, theta)
        got = np.asarray(christoffel(st, r, theta))
        assert got.shape == (4, 4, 4)
        np.testing.assert_allclose(got, expected, rtol=CROSS_RTOL, atol=CROSS_ATOL)


def test_numeric_geometry_broadcasts_over_arrays() -> None:
    from kerrray.geometry.christoffel import christoffel
    from kerrray.geometry.metric import inverse_metric, metric

    st = _spacetime(0.6)
    rng = np.random.default_rng(15)
    pts = random_points(rng, st.a, n=6)
    r = np.array([p[0] for p in pts]).reshape(2, 3)
    theta = np.array([p[1] for p in pts]).reshape(2, 3)
    g = np.asarray(metric(st, r, theta))
    ginv = np.asarray(inverse_metric(st, r, theta))
    gam = np.asarray(christoffel(st, r, theta))
    assert g.shape == (2, 3, 4, 4) and ginv.shape == (2, 3, 4, 4) and gam.shape == (2, 3, 4, 4, 4)
    identity = np.einsum("...ij,...jk->...ik", g, ginv)
    np.testing.assert_allclose(identity, np.broadcast_to(np.eye(4), identity.shape), atol=1e-12)
    gam_ref = sym.lambdified_christoffel()
    for i in range(2):
        for j in range(3):
            np.testing.assert_allclose(
                gam[i, j], gam_ref(MASS, st.a, r[i, j], theta[i, j]), rtol=CROSS_RTOL, atol=CROSS_ATOL
            )
