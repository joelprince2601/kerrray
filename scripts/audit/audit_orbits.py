"""Audit item 7: spherical photon orbits, shadow curve, r_ph, b_c, ISCO (independent derivations).

Run: ./.venv/Scripts/python.exe scripts/audit/audit_orbits.py
"""
import math

import numpy as np
import sympy as sp
from scipy.optimize import brentq, minimize_scalar

from kerrray.geometry.metric import Spacetime
from kerrray.photons import orbits as O

r, M, a, xi, eta = sp.symbols("r M a xi eta", real=True)
Delta = r**2 - 2 * M * r + a**2
R = ((r**2 + a**2) - a * xi) ** 2 - Delta * (eta + (xi - a) ** 2)
sols = sp.solve([R, sp.diff(R, r)], [xi, eta], dict=True)
print("solutions of R = R' = 0 for (xi, eta):")
for s in sols:
    print("   xi  =", sp.factor(s[xi]), "\n   eta =", sp.factor(s[eta]))
xi_ref = -(r**2 * (r - 3 * M) + a**2 * (r + M)) / (a * (r - M))
eta_ref = r**3 * (4 * M * a**2 - r * (r - 3 * M) ** 2) / (a**2 * (r - M) ** 2)
match = [s for s in sols if sp.simplify(s[xi] - xi_ref) == 0]
print("physical branch matches code closed forms:", len(match) == 1 and sp.simplify(match[0][eta] - eta_ref) == 0)
# eta = 0 <=> r (r - 3M)^2 = 4 M a^2
print("numerator of eta:", sp.factor(sp.numer(sp.together(eta_ref))))

rows = []
w_rph = w_bc = w_isco = w_curve = w_shape = 0.0
for spin in (0.0, 0.1, 0.3, 0.5, 0.6, 0.9, 0.99, 0.999, -0.9):
    st = Spacetime(1.0, spin)
    A = abs(st.a)
    for pro in (True, False):
        s = 1 if pro else -1
        # independent: root of r (r - 3)^2 - 4 a^2 on the right bracket
        f = lambda x: x * (x - 3) ** 2 - 4 * A * A
        rph = 3.0 if A == 0 else (brentq(f, 1.0, 3.0, xtol=1e-15) if pro else brentq(f, 3.0, 4.0, xtol=1e-15))
        bpt = 2 * (1 + math.cos(2 / 3 * math.acos(-s * A)))
        w_rph = max(w_rph, abs(rph - O.equatorial_photon_orbit_radius(st, pro)), abs(bpt - rph))
        # b_c: xi at r_ph with eta = 0, via xi(r) (a != 0) or 3 sqrt 3
        if A == 0:
            bc = 3 * math.sqrt(3)
        else:
            bc = abs(float(xi_ref.subs({r: rph, M: 1, a: A})))
        # independent check: minimise b(r) of tangential equatorial photons (b_c = min over r of b(r))
        def b_of_r(x):
            d = x * x - 2 * x + A * A
            L = ((x * x + A * A) + s * A * math.sqrt(d)) / (A + s * math.sqrt(d))
            return abs(L)
        mm = minimize_scalar(b_of_r, bounds=(1 + math.sqrt(1 - A * A) + 1e-6, 6.0), method="bounded", options={"xatol": 1e-12})
        code_bc = O.critical_impact_parameters(st)[0 if pro else 1]
        w_bc = max(w_bc, abs(bc - code_bc), abs(mm.fun - bc) / bc * 1e-3)  # min over r: quadratic -> 1e-6 precision
        # ISCO independent: minimum of E(r) of circular timelike orbits, E from Veff numerics
        def E_circ(x):
            v = x**1.5
            return (v - 2 * math.sqrt(x) + s * A) / (x**0.75 * math.sqrt(v - 3 * math.sqrt(x) + 2 * s * A))
        lo = rph + 1e-6
        mi = minimize_scalar(E_circ, bounds=(lo + 1e-3, 12.0), method="bounded", options={"xatol": 1e-10})
        w_isco = max(w_isco, abs(mi.x - O.isco_radius(st, pro)))
        rows.append((spin, "pro" if pro else "retro", rph, bc, mm.fun, O.isco_radius(st, pro), mi.x))
    if st.a != 0:
        # shadow curve: independent parametrisation vs code at i = 60 deg
        th_o = math.radians(60.0)
        al, be = O.shadow_curve(st, 60.0, 4000)
        # exact on-curve residual: invert xi = -alpha sin(i) for r on [r_min, r_max] (xi(r) is monotone there),
        # then compare beta^2 with eta(r) + a^2 cos^2 i - xi^2 cot^2 i
        lo_, hi_ = O.spherical_orbit_radius_range(st)
        xi_f = lambda x: -(x**2 * (x - 3) + st.a**2 * (x + 1)) / (st.a * (x - 1))
        eta_f = lambda x: x**3 * (4 * st.a**2 - x * (x - 3) ** 2) / (st.a**2 * (x - 1) ** 2)
        res = 0.0
        for al_k, be_k in zip(al, be):
            target = -al_k * math.sin(th_o)
            rk = brentq(lambda x: xi_f(x) - target, lo_, hi_, xtol=1e-15)
            b2 = eta_f(rk) + st.a**2 * math.cos(th_o) ** 2 - target**2 / math.tan(th_o) ** 2
            res = max(res, abs(abs(be_k) - math.sqrt(max(b2, 0.0))))
        w_curve = max(w_curve, res)
        # Bardeen check: code alpha, beta invert to (xi, eta) satisfying R = R' = 0 at some r
print("\nspin dir   r_ph(indep)   b_c(xi(r_ph))  b_c(min b(r))  ISCO(BPT code)  ISCO(min E)")
for row in rows:
    print("  {:+.3f} {:5s} {:.10f} {:.10f} {:.10f} {:.10f} {:.10f}".format(*row))
print(f"max |r_ph code - indep| & |BPT - indep|: {w_rph:.1e}; max |b_c code - xi(r_ph)|: {w_bc:.1e}; "
      f"max |ISCO BPT - min E|: {w_isco:.1e}; max distance code shadow curve -> independent curve: {w_curve:.1e} M")
