"""Christoffel-symbol and vacuum tests (PROJECT.md sections 8, 32 and 41).

The closed-form implementation is compared with the defining formula
evaluated symbolically (SymPy, lambdified without simplification), checked
for symmetry, for metric compatibility against the independent closed-form
inverse-metric derivatives, for the Schwarzschild limit, and the symbolic
Ricci tensor built from the same metric is checked to vanish numerically:
the implemented metric is the vacuum Kerr solution.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

import symbolic_kerr as sk
from kerrray.geometry import (
    christoffel,
    inverse_metric,
    inverse_metric_derivatives,
    kerr,
    metric_derivatives,
    schwarzschild,
)

SPINS = (0.0, 0.3, -0.5, 0.9, 0.998)

# Index triples (mu, alpha, beta), alpha <= beta, that may be non-zero for a
# stationary axisymmetric metric whose only off-diagonal term is g_tphi.
NONZERO_TRIPLES = {
    (0, 0, 1), (0, 0, 2), (0, 1, 3), (0, 2, 3),
    (1, 0, 0), (1, 0, 3), (1, 1, 1), (1, 1, 2), (1, 2, 2), (1, 3, 3),
    (2, 0, 0), (2, 0, 3), (2, 1, 1), (2, 1, 2), (2, 2, 2), (2, 3, 3),
    (3, 0, 1), (3, 0, 2), (3, 1, 3), (3, 2, 3),
}

# For a = 0 the g_tphi terms and every theta derivative of g_tt, g_rr, g_thth
# vanish as well, leaving the nine Schwarzschild components.
SCHWARZSCHILD_NONZERO_TRIPLES = {
    (0, 0, 1), (1, 0, 0), (1, 1, 1), (1, 2, 2), (1, 3, 3),
    (2, 1, 2), (2, 3, 3), (3, 1, 3), (3, 2, 3),
}


def _five(m: np.ndarray) -> np.ndarray:
    return np.array([m[0, 0], m[0, 3], m[1, 1], m[2, 2], m[3, 3]])


@pytest.mark.parametrize("spin", SPINS)
def test_metric_derivatives_match_sympy(spin: float) -> None:
    st = kerr(1.0, spin)
    rng = np.random.default_rng(31)
    r, theta = sk.sample_points(st, rng, 6)
    d_r_fn, d_th_fn = sk.lambdified_metric_derivatives()
    for ri, ti in zip(r, theta):
        d_r, d_th = metric_derivatives(st, ri, ti)
        sk.assert_relative_close(np.stack(d_r), _five(d_r_fn(st.mass, st.a, ri, ti)), 1e-10)
        sk.assert_relative_close(np.stack(d_th), _five(d_th_fn(st.mass, st.a, ri, ti)), 1e-10)


@pytest.mark.parametrize("spin", SPINS)
def test_christoffel_matches_sympy_definition(spin: float) -> None:
    st = kerr(1.0, spin)
    rng = np.random.default_rng(32)
    r, theta = sk.sample_points(st, rng, 5)
    gamma_fn = sk.lambdified_christoffel()
    for ri, ti in zip(r, theta):
        sk.assert_relative_close(christoffel(st, ri, ti), gamma_fn(st.mass, st.a, ri, ti), 1e-10)


@pytest.mark.parametrize("spin", SPINS)
def test_symmetric_in_lower_indices(spin: float) -> None:
    st = kerr(1.0, spin)
    rng = np.random.default_rng(33)
    r, theta = sk.sample_points(st, rng, 20)
    gamma = christoffel(st, r, theta)
    np.testing.assert_array_equal(gamma, np.swapaxes(gamma, -1, -2))


@pytest.mark.parametrize("spin", SPINS)
def test_zero_pattern_follows_from_the_symmetries(spin: float) -> None:
    """Exactly the triples with an even number of t/phi indices (upper index
    included) are non-zero; for a = 0 only the nine Schwarzschild ones remain."""
    st = kerr(1.0, spin)
    gamma = christoffel(st, 4.0, 1.0)
    expected_nonzero = NONZERO_TRIPLES if spin != 0.0 else SCHWARZSCHILD_NONZERO_TRIPLES
    for mu in range(4):
        for al in range(4):
            for be in range(al, 4):
                if (mu, al, be) in expected_nonzero:
                    assert gamma[mu, al, be] != 0.0, (mu, al, be)
                else:
                    assert gamma[mu, al, be] == 0.0, (mu, al, be)


def test_schwarzschild_limit_matches_textbook_forms() -> None:
    """Schwarzschild Christoffel symbols in the standard form (e.g. Carroll
    2004, *Spacetime and Geometry*, section 5.4), with G = c = 1."""
    for mass in (1.0, 2.0):
        st = schwarzschild(mass)
        r, theta = 5.0 * mass, 1.1
        gamma = christoffel(st, r, theta)
        m = mass
        expected = {
            (0, 0, 1): m / (r * (r - 2 * m)),
            (1, 0, 0): m * (r - 2 * m) / r**3,
            (1, 1, 1): -m / (r * (r - 2 * m)),
            (1, 2, 2): -(r - 2 * m),
            (1, 3, 3): -(r - 2 * m) * np.sin(theta) ** 2,
            (2, 1, 2): 1.0 / r,
            (2, 3, 3): -np.sin(theta) * np.cos(theta),
            (3, 1, 3): 1.0 / r,
            (3, 2, 3): 1.0 / np.tan(theta),
        }
        for (mu, al, be), value in expected.items():
            np.testing.assert_allclose(gamma[mu, al, be], value, rtol=1e-13, err_msg=str((mu, al, be)))
            np.testing.assert_allclose(gamma[mu, be, al], value, rtol=1e-13)
        # Every other independent component vanishes for a = 0.
        nonzero = {key for key in NONZERO_TRIPLES} - set(expected)
        for mu, al, be in nonzero:
            assert gamma[mu, al, be] == 0.0, (mu, al, be)


@pytest.mark.parametrize("spin", SPINS)
def test_metric_compatibility_with_inverse_metric_derivatives(spin: float) -> None:
    """d_c g^{ab} = -(Gamma^a_{cd} g^{db} + Gamma^b_{cd} g^{ad}); two independent closed forms."""
    st = kerr(1.0, spin)
    rng = np.random.default_rng(34)
    r, theta = sk.sample_points(st, rng, 8)
    gamma = christoffel(st, r, theta)
    ginv = inverse_metric(st, r, theta)
    d_r, d_th = inverse_metric_derivatives(st, r, theta)
    for c, comps in ((1, d_r), (2, d_th)):
        gamma_c = gamma[..., :, c, :]  # gamma_c[..., a, d] = Gamma^a_{c d}
        derived = -(
            np.einsum("...ad,...db->...ab", gamma_c, ginv)
            + np.einsum("...bd,...ad->...ab", gamma_c, ginv)
        )
        expected = np.zeros_like(derived)
        expected[..., 0, 0] = comps.gtt
        expected[..., 0, 3] = expected[..., 3, 0] = comps.gtphi
        expected[..., 1, 1] = comps.grr
        expected[..., 2, 2] = comps.gthth
        expected[..., 3, 3] = comps.gphph
        sk.assert_relative_close(derived, expected, 1e-10)


@pytest.mark.parametrize("spin", (0.0, 0.6, 0.95))
def test_ricci_tensor_vanishes(spin: float) -> None:
    """Vacuum check: the symbolic Ricci tensor of the implemented metric is zero.

    ``R_ab`` is evaluated from the lambdified symbolic expression (no
    simplification, no finite differences) and compared with the size of the
    Gamma*Gamma terms whose cancellation produces zero.
    """
    st = kerr(1.0, spin)
    rng = np.random.default_rng(35)
    r, theta = sk.sample_points(st, rng, 10)
    ricci_fn = sk.lambdified_ricci()
    for ri, ti in zip(r, theta):
        ricci = ricci_fn(st.mass, st.a, ri, ti)
        gamma = christoffel(st, ri, ti)
        quadratic = np.abs(np.einsum("ccd,dab->ab", gamma, gamma)) + np.abs(
            np.einsum("cbd,dac->ab", gamma, gamma)
        )
        scale = float(np.max(quadratic))
        assert scale > 0.0
        assert np.max(np.abs(ricci)) < 1e-8 * scale
        assert ricci.shape == (4, 4)


def test_shapes_and_dtype() -> None:
    st = kerr(1.0, 0.7)
    assert christoffel(st, 4.0, 1.0).shape == (4, 4, 4)
    out = christoffel(st, np.linspace(3, 5, 3)[:, None], np.linspace(0.3, 2.8, 2)[None, :])
    assert out.shape == (3, 2, 4, 4, 4) and out.dtype == np.float64
    assert christoffel(st, [3.0, 4.0], 1.0).shape == (2, 4, 4, 4)
    assert np.all(np.isfinite(out))


def test_geometry_import_does_not_load_sympy() -> None:
    code = "import sys, kerrray.geometry; sys.exit(1 if 'sympy' in sys.modules else 0)"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
