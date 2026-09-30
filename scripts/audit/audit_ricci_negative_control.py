"""Negative control for the Ricci test: perturbed (non-vacuum) metrics must give R_mu nu != 0.

Note: flipping the sign of g_tphi alone is NOT a valid control (it is Kerr with
phi -> -phi, still vacuum; the script shows this too).
"""
import sys
import mpmath as mp
import sympy as sp

sys.path.insert(0, __file__.rsplit("audit_ricci_negative_control.py", 1)[0].rstrip("/\\"))
import _kerr_sym as K  # noqa: E402


def max_ricci(g):
    ginv = g.inv()
    X = K.X
    Gam = [[[sum(ginv[l, k] * (sp.diff(g[k, m], X[n]) + sp.diff(g[k, n], X[m]) - sp.diff(g[m, n], X[k]))
                 for k in range(4)) / 2 for n in range(4)] for m in range(4)] for l in range(4)]
    Ric = sp.zeros(4, 4)
    for m in range(4):
        for n in range(4):
            Ric[m, n] = sum(sp.diff(Gam[l][m][n], X[l]) - sp.diff(Gam[l][m][l], X[n]) for l in range(4)) + sum(
                Gam[l][l][k] * Gam[k][m][n] - Gam[l][n][k] * Gam[k][m][l] for l in range(4) for k in range(4))
    f = sp.lambdify((K.r, K.th, K.M, K.a), Ric, "mpmath")
    mp.mp.dps = 30
    v = f(mp.mpf(4), mp.mpf(1), mp.mpf(1), mp.mpf("0.9"))
    return max(abs(v[i, j]) for i in range(4) for j in range(4))


g1 = K.g.copy(); g1[0, 3] = g1[3, 0] = -g1[0, 3]
print("g_tphi sign flipped (= Kerr with a -> -a in that term only, i.e. phi -> -phi):", mp.nstr(max_ricci(g1), 3))
g2 = K.g.copy(); g2[1, 1] = g2[1, 1] * (1 + K.M / (10 * K.r))
print("g_rr perturbed by (1 + M/10r):", mp.nstr(max_ricci(g2), 3))
g3 = K.g.copy(); g3[0, 3] = g3[3, 0] = 2 * g3[0, 3]
print("g_tphi doubled:", mp.nstr(max_ricci(g3), 3))
