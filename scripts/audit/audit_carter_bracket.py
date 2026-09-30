"""Item 4: {Q_null, H} is proportional to H (vanishes only on the null shell).

Q_null = p_th^2 + cos^2 (L^2/sin^2 - a^2 E^2). The full Carter constant for any
geodesic is Q_full = Q_null + a^2 cos^2(theta) mu^2 with mu^2 = -2H, and
{Q_full, H} = 0 identically. Hence {Q_null, H} = 2 H {a^2 cos^2 theta, H}.
"""
import sys

import sympy as sp

sys.path.insert(0, __file__.rsplit("audit_carter_bracket.py", 1)[0].rstrip("/\\"))
from _kerr_sym import Sigma, a, inverse_tphi_block, r, th  # noqa: E402

_, ginv = inverse_tphi_block()
pt, pr, pth, pph = sp.symbols("p_t p_r p_theta p_phi", real=True)
P = sp.Matrix([pt, pr, pth, pph])
H = sp.Rational(1, 2) * (P.T * ginv * P)[0, 0]
coords = (sp.Symbol("t"), r, th, sp.Symbol("phi"))


def pb(F, G):
    return sum(sp.diff(F, x) * sp.diff(G, p) - sp.diff(F, p) * sp.diff(G, x) for x, p in zip(coords, P))


E, L = -pt, pph
Qn = pth**2 + sp.cos(th) ** 2 * (L**2 / sp.sin(th) ** 2 - a**2 * E**2)
Qfull = Qn + a**2 * sp.cos(th) ** 2 * (-2 * H)
print("{Q_full, H} ->", sp.simplify(sp.together(pb(Qfull, H))))
expected = 2 * H * pb(a**2 * sp.cos(th) ** 2, H)
print("{Q_null, H} - 2H{a^2 cos^2, H} ->", sp.simplify(sp.together(pb(Qn, H) - expected)))
print("{a^2 cos^2 theta, H} =", sp.simplify(pb(a**2 * sp.cos(th) ** 2, H)), " (= -a^2 sin(2 theta) p_theta / Sigma)")
print("K = Q_null + (L - aE)^2 also:", sp.simplify(sp.together(pb(Qn + (L - a * E) ** 2, H) - expected)))
