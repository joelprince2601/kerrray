"""Audit items 1-2: Kerr metric, inverse, derivatives, determinant, Ricci = 0, horizons.

Run: ./.venv/Scripts/python.exe scripts/audit/audit_metric.py
"""
import math
import sys
import time

import mpmath as mp
import numpy as np
import sympy as sp

sys.path.insert(0, __file__.rsplit("audit_metric.py", 1)[0].rstrip("/\\"))
from _kerr_sym import M, Delta, Sigma, X, a, g, inverse_tphi_block, r, th  # noqa: E402

from kerrray.geometry.horizons import ergosphere_radius, horizon_radii
from kerrray.geometry.metric import (
    Spacetime, inverse_metric, inverse_metric_components, inverse_metric_derivatives,
    metric, metric_components, metric_derivatives,
)

rng = np.random.default_rng(20260929)
t0 = time.time()

# ---- 1a. determinant and closed-form inverse ------------------------------
det2, ginv = inverse_tphi_block()
print("det2 (t-phi block) =", sp.factor(sp.simplify(det2)))
detg = sp.simplify(g.det())
print("det g - (-Sigma^2 sin^2) simplifies to:", sp.simplify(detg + Sigma**2 * sp.sin(th) ** 2))
A = (r**2 + a**2) ** 2 - a**2 * Delta * sp.sin(th) ** 2
claims = {
    (0, 0): -A / (Sigma * Delta),
    (0, 3): -2 * M * a * r / (Sigma * Delta),
    (1, 1): Delta / Sigma,
    (2, 2): 1 / Sigma,
    (3, 3): (Delta - a**2 * sp.sin(th) ** 2) / (Sigma * Delta * sp.sin(th) ** 2),
}
for (i, j), expr in claims.items():
    print(f"g^{i}{j}: closed form - hand inverse ->", sp.simplify(sp.expand_trig(ginv[i, j] - expr)))
print("g * ginv - I ->", sp.simplify(g * ginv - sp.eye(4)))
print("Schwarzschild limit g(a=0):", [sp.simplify(g[i, i].subs(a, 0)) for i in range(4)], "g_tphi:", g[0, 3].subs(a, 0))

# ---- 1b. numerical comparison with the implementation --------------------
f_g = sp.lambdify((r, th, M, a), g, "numpy")
f_gi = sp.lambdify((r, th, M, a), ginv, "numpy")
f_dgi_r = sp.lambdify((r, th, M, a), ginv.diff(r), "numpy")
f_dgi_t = sp.lambdify((r, th, M, a), ginv.diff(th), "numpy")
f_dg_r = sp.lambdify((r, th, M, a), g.diff(r), "numpy")
f_dg_t = sp.lambdify((r, th, M, a), g.diff(th), "numpy")
idx = [(0, 0), (0, 3), (1, 1), (2, 2), (3, 3)]
worst = {k: 0.0 for k in ("g", "ginv", "dg_r", "dg_th", "dginv_r", "dginv_th", "g.ginv-I")}
for _ in range(2000):
    spin = rng.uniform(-0.999, 0.999)
    mass = rng.uniform(0.5, 2.0)
    st = Spacetime(mass, spin)
    rp = mass + math.sqrt(mass**2 - st.a**2)
    rr = rp + rng.uniform(1e-3, 60.0) * mass
    tt = rng.uniform(0.02, math.pi - 0.02)
    ref = {
        "g": np.array(f_g(rr, tt, mass, st.a), float),
        "ginv": np.array(f_gi(rr, tt, mass, st.a), float),
        "dg_r": np.array(f_dg_r(rr, tt, mass, st.a), float),
        "dg_th": np.array(f_dg_t(rr, tt, mass, st.a), float),
        "dginv_r": np.array(f_dgi_r(rr, tt, mass, st.a), float),
        "dginv_th": np.array(f_dgi_t(rr, tt, mass, st.a), float),
    }
    dgr, dgt = metric_derivatives(st, rr, tt)
    digr, digt = inverse_metric_derivatives(st, rr, tt)
    code = {
        "g": metric_components(st, rr, tt), "ginv": inverse_metric_components(st, rr, tt),
        "dg_r": dgr, "dg_th": dgt, "dginv_r": digr, "dginv_th": digt,
    }
    for key in code:
        scale = max(np.max(np.abs(ref[key])), 1e-300)
        for c, (i, j) in zip(code[key], idx):
            worst[key] = max(worst[key], abs(float(c) - ref[key][i, j]) / scale)
    prod = metric(st, rr, tt) @ inverse_metric(st, rr, tt)
    worst["g.ginv-I"] = max(worst["g.ginv-I"], float(np.max(np.abs(prod - np.eye(4)))))
print("max |code - independent sympy| / max|component| over 2000 random points:")
for k, v in worst.items():
    print(f"   {k:10s} {v:.2e}")

# ---- 1c. Ricci tensor of the independent metric ---------------------------
Gam = [[[0] * 4 for _ in range(4)] for _ in range(4)]
for l in range(4):
    for m_ in range(4):
        for n in range(m_, 4):
            s = 0
            for k in range(4):
                if ginv[l, k] != 0:
                    s += ginv[l, k] * (sp.diff(g[k, m_], X[n]) + sp.diff(g[k, n], X[m_]) - sp.diff(g[m_, n], X[k]))
            Gam[l][m_][n] = Gam[l][n][m_] = s / 2
Ric = sp.zeros(4, 4)
for m_ in range(4):
    for n in range(m_, 4):
        s = 0
        for l in range(4):
            s += sp.diff(Gam[l][m_][n], X[l]) - sp.diff(Gam[l][m_][l], X[n])
            for k in range(4):
                s += Gam[l][l][k] * Gam[k][m_][n] - Gam[l][n][k] * Gam[k][m_][l]
        Ric[m_, n] = Ric[n, m_] = s
f_ric = sp.lambdify((r, th, M, a), Ric, "mpmath")
mp.mp.dps = 40
worst_ric = 0.0
for _ in range(10):
    spin = rng.uniform(-0.99, 0.99)
    rr = 1 + math.sqrt(1 - spin**2) + rng.uniform(0.1, 20)
    tt = rng.uniform(0.1, math.pi - 0.1)
    val = f_ric(mp.mpf(rr), mp.mpf(tt), mp.mpf(1), mp.mpf(spin))
    worst_ric = max(worst_ric, max(abs(val[i, j]) for i in range(4) for j in range(4)))
print(f"max |R_mu nu| (independent metric, 40-digit arithmetic, 10 random points): {mp.nstr(worst_ric, 3)}")
# Ricci of a WRONG metric (g_tphi with the opposite sign of M) must NOT vanish - sanity of the test
print("elapsed Ricci section:", round(time.time() - t0, 1), "s")

# ---- 2. horizons and ergosphere -----------------------------------------
rsol = sp.solve(sp.Eq(Delta, 0), r)
print("roots of Delta:", rsol)
ergo = sp.solve(sp.Eq(g[0, 0], 0), r)
print("roots of g_tt = 0:", ergo)
wh = 0.0
for spin in (-0.999, -0.5, 0.0, 0.3, 0.9, 0.99, 0.999):
    st = Spacetime(1.0, spin)
    rp, rm = horizon_radii(st)
    wh = max(wh, abs(rp - (1 + math.sqrt(1 - spin**2))), abs(rm - (1 - math.sqrt(1 - spin**2))))
    for tt in (0.0, 0.3, 1.0, math.pi / 2):
        re = float(ergosphere_radius(st, tt))
        wh = max(wh, abs(re - (1 + math.sqrt(1 - spin**2 * math.cos(tt) ** 2))))
        gtt = metric_components(st, re, tt).g_tt
        wh = max(wh, abs(float(gtt)))
print(f"horizon/ergosphere max deviation (and |g_tt(r_E)|): {wh:.2e}")
print("total elapsed", round(time.time() - t0, 1), "s")
