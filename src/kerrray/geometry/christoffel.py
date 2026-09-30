"""Christoffel symbols of the Kerr metric (PROJECT.md sections 8 and 32).

Definition (Misner, Thorne and Wheeler 1973, eq. 8.24b; Chandrasekhar 1983,
chapter 1)::

    Gamma^mu_{alpha beta} = (1/2) g^{mu nu} (d_alpha g_{nu beta}
                                             + d_beta g_{nu alpha}
                                             - d_nu g_{alpha beta})

The Kerr metric in Boyer-Lindquist coordinates depends on ``r`` and ``theta``
only, so just ``d_r`` and ``d_theta`` of the five non-zero covariant
components enter. Those derivatives are closed-form NumPy expressions
(:func:`kerrray.geometry.metric.metric_derivatives`) and the contraction
above is carried out numerically with ``einsum``; nothing is approximated
(no finite differences) and SymPy is not used at runtime. The result is
verified in ``tests/test_christoffel.py`` against the same definition
evaluated symbolically with SymPy (lambdified, unsimplified) to 1e-10
relative, and the symbolic Ricci tensor built from the metric is checked to
vanish numerically (vacuum check).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from kerrray.geometry.metric import (
    Array,
    Spacetime,
    inverse_metric,
    metric_derivatives,
)

__all__ = ["christoffel"]


def christoffel(st: Spacetime, r: ArrayLike, theta: ArrayLike) -> Array:
    """Christoffel symbols ``Gamma^mu_{alpha beta}`` of the Kerr metric.

    Args:
        st: The spacetime.
        r: Boyer-Lindquist radius (scalar or array), in units of ``M``.
        theta: Boyer-Lindquist polar angle in radians (broadcastable with ``r``).

    Returns:
        Array of shape ``(..., 4, 4, 4)`` indexed ``[..., mu, alpha, beta]``
        for ``Gamma^mu_{alpha beta}`` with coordinate order ``(t, r, theta,
        phi)``; symmetric in the last two indices.
    """
    d_r, d_th = metric_derivatives(st, r, theta)
    ginv = inverse_metric(st, r, theta)
    base = ginv.shape[:-2]
    # dg[..., c, m, n] = d_c g_{mn}; only c = 1 (r) and c = 2 (theta) are non-zero.
    dg = np.zeros(base + (4, 4, 4), dtype=np.float64)
    for c, comps in ((1, d_r), (2, d_th)):
        dg[..., c, 0, 0] = comps.g_tt
        dg[..., c, 0, 3] = comps.g_tphi
        dg[..., c, 3, 0] = comps.g_tphi
        dg[..., c, 1, 1] = comps.g_rr
        dg[..., c, 2, 2] = comps.g_thth
        dg[..., c, 3, 3] = comps.g_phph
    # term[..., nu, a, b] = d_a g_{nu b} + d_b g_{nu a} - d_nu g_{ab}
    term = np.swapaxes(dg, -3, -2) + np.moveaxis(dg, -3, -1) - dg
    return 0.5 * np.einsum("...mn,...nab->...mab", ginv, term)
