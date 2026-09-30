"""Independent SymPy Kerr metric for the mathematical audit.

Written from scratch from the Boyer-Lindquist line element (MTW eq. 33.2);
does NOT import kerrray.geometry.symbolic.
"""
import sympy as sp

t, r, th, ph = sp.symbols("t r theta phi", real=True)
M, a = sp.symbols("M a", real=True)
X = (t, r, th, ph)

Sigma = r**2 + a**2 * sp.cos(th) ** 2
Delta = r**2 - 2 * M * r + a**2

g = sp.zeros(4, 4)
g[0, 0] = -(1 - 2 * M * r / Sigma)
g[0, 3] = g[3, 0] = -2 * M * a * r * sp.sin(th) ** 2 / Sigma
g[1, 1] = Sigma / Delta
g[2, 2] = Sigma
g[3, 3] = (r**2 + a**2 + 2 * M * a**2 * r * sp.sin(th) ** 2 / Sigma) * sp.sin(th) ** 2


def inverse_tphi_block():
    """Invert the (t, phi) block by hand: det2 = g_tt g_pp - g_tp^2."""
    det2 = sp.simplify(g[0, 0] * g[3, 3] - g[0, 3] ** 2)
    ginv = sp.zeros(4, 4)
    ginv[0, 0] = sp.simplify(g[3, 3] / det2)
    ginv[3, 3] = sp.simplify(g[0, 0] / det2)
    ginv[0, 3] = ginv[3, 0] = sp.simplify(-g[0, 3] / det2)
    ginv[1, 1] = sp.simplify(1 / g[1, 1])
    ginv[2, 2] = sp.simplify(1 / g[2, 2])
    return det2, ginv
