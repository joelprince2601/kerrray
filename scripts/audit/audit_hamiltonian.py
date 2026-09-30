"""Audit items 3-5: Hamiltonian RHS, conserved quantities, Carter constant, initial conditions.

Run: ./.venv/Scripts/python.exe scripts/audit/audit_hamiltonian.py
"""
import math
import sys

import numpy as np
import sympy as sp

sys.path.insert(0, __file__.rsplit("audit_hamiltonian.py", 1)[0].rstrip("/\\"))
from _kerr_sym import M, Delta, Sigma, a, inverse_tphi_block, r, th  # noqa: E402

from kerrray.geodesics.equations import geodesic_rhs, hamiltonian, null_momentum_pt
from kerrray.geodesics.initial_conditions import equatorial_photon, photon_from_constants, tangential_photon
from kerrray.geometry.metric import Spacetime
from kerrray.photons.constants import carter_constant

rng = np.random.default_rng(7)
_, ginv = inverse_tphi_block()
pt, pr, pth, pph = sp.symbols("p_t p_r p_theta p_phi", real=True)
P = sp.Matrix([pt, pr, pth, pph])
H = sp.Rational(1, 2) * (P.T * ginv * P)[0, 0]
coords = (sp.Symbol("t"), r, th, sp.Symbol("phi"))
rhs_sym = [sp.diff(H, p) for p in P] + [-sp.diff(H, x) for x in coords]
f_rhs = sp.lambdify((r, th, M, a, pt, pr, pth, pph), rhs_sym, "numpy")
f_H = sp.lambdify((r, th, M, a, pt, pr, pth, pph), H, "numpy")

# ---- 3. RHS vs independent gradient of H at random (off-shell and null) states
worst, worst_h = 0.0, 0.0
for k in range(3000):
    spin = rng.uniform(-0.999, 0.999)
    st = Spacetime(1.0, spin)
    rp = 1 + math.sqrt(1 - spin**2)
    rr = rp + 10 ** rng.uniform(-3, 2)
    tt = rng.uniform(0.01, math.pi - 0.01)
    p = rng.normal(size=4) * np.array([1, 3, 5, 5])
    y = np.array([0.0, rr, tt, 0.3, *p])
    ref = np.array(f_rhs(rr, tt, 1.0, st.a, *p), float)
    code = geodesic_rhs(st, y)
    worst = max(worst, float(np.max(np.abs(code - ref) / (np.abs(ref) + 1e-12 * np.max(np.abs(ref)) + 1e-300))))
    worst_h = max(worst_h, abs(float(hamiltonian(st, y)) - f_H(rr, tt, 1.0, st.a, *p)) / (abs(f_H(rr, tt, 1.0, st.a, *p)) + 1e-12))
print(f"geodesic_rhs vs independent grad(H), max componentwise relative error (3000 states): {worst:.2e}")
print(f"hamiltonian vs independent H, max relative error: {worst_h:.2e}")
# explicit check of the cross term: dphi/dlambda = g^{t phi} p_t + g^{phi phi} p_phi, dp_r contains 2 d_r g^{t phi} p_t p_phi
print("dH/dp_phi =", sp.simplify(sp.diff(H, pph) - (ginv[0, 3] * pt + ginv[3, 3] * pph)), "(0 means cross term handled with factor 2 in H)")

# ---- 4. Carter constant: {Q, H} = 0 symbolically, and separation identity
E, L = -pt, pph
Q = pth**2 + sp.cos(th) ** 2 * (L**2 / sp.sin(th) ** 2 - a**2 * E**2)
pb = sum(sp.diff(Q, x) * sp.diff(H, p) - sp.diff(Q, p) * sp.diff(H, x) for x, p in zip(coords, P))
print("Poisson bracket {Q, H} simplifies to:", sp.simplify(sp.together(pb)))
Rr = (E * (r**2 + a**2) - a * L) ** 2 - Delta * (Q + (L - a * E) ** 2)
Th = Q - sp.cos(th) ** 2 * (L**2 / sp.sin(th) ** 2 - a**2 * E**2)
ident = 2 * Sigma * H - ((Delta * pr**2 - Rr / Delta) + (pth**2 - Th))
print("2 Sigma H - [(Delta p_r^2 - R/Delta) + (p_th^2 - Theta)] ->", sp.simplify(ident))
# Also BPT form Q = p_th^2 + cos^2 [ a^2 (mu^2 - E^2) + L^2/sin^2 ] with mu = 0: identical expression.
# compare implementation
f_Q = sp.lambdify((th, a, pt, pth, pph), Q, "numpy")
wq = 0.0
for _ in range(1000):
    spin = rng.uniform(-0.99, 0.99); st = Spacetime(1.0, spin)
    y = np.array([0, 5.0, rng.uniform(0.05, 3.09), 0, *rng.normal(size=4) * 4])
    ref = f_Q(y[2], st.a, y[4], y[6], y[7])
    wq = max(wq, abs(float(carter_constant(st, y)) - ref) / max(abs(ref), 1))
print(f"carter_constant vs independent Q, max relative error: {wq:.2e}")

# ---- 5. initial conditions
wn, wsign, wq2 = 0.0, 0, 0.0
for _ in range(2000):
    spin = rng.uniform(-0.99, 0.99); st = Spacetime(1.0, spin)
    rr = 1 + math.sqrt(1 - spin**2) + 10 ** rng.uniform(-1, 2)
    tt = rng.uniform(0.2, math.pi - 0.2)
    Ee = rng.uniform(0.5, 2.0)
    xi = rng.uniform(-6, 6); eta = rng.uniform(0, 30)
    Lz, Qc = xi * Ee, eta * Ee**2
    Rv = float(Rr.subs({r: rr, a: st.a, M: 1, pt: -Ee, pph: Lz, sp.Symbol("p_theta", real=True): 0}).subs(pth, sp.sqrt(Qc) if False else pth)) if False else None
    R_num = (Ee * (rr**2 + st.a**2) - st.a * Lz) ** 2 - (rr**2 - 2 * rr + st.a**2) * (Qc + (Lz - st.a * Ee) ** 2)
    T_num = Qc - math.cos(tt) ** 2 * (Lz**2 / math.sin(tt) ** 2 - st.a**2 * Ee**2)
    if R_num <= 0 or T_num <= 0:
        continue
    sr, sth = rng.choice([-1, 1]), rng.choice([-1, 1])
    y = photon_from_constants(st, rr, tt, Ee, Lz, Qc, sign_r=int(sr), sign_theta=int(sth))
    d = geodesic_rhs(st, y)
    wn = max(wn, abs(float(hamiltonian(st, y))) / Ee**2)
    wq2 = max(wq2, abs(float(carter_constant(st, y)) - Qc) / max(Qc, 1))
    sig = rr**2 + st.a**2 * math.cos(tt) ** 2
    # Sigma dr/dlambda = sign_r sqrt(R), Sigma dtheta/dlambda = sign_theta sqrt(Theta)
    ok = (np.sign(d[1]) == sr and np.sign(d[2]) == sth and abs(sig * d[1] - sr * math.sqrt(R_num)) < 1e-9 * math.sqrt(R_num)
          and abs(sig * d[2] - sth * math.sqrt(T_num)) < 1e-9 * math.sqrt(T_num) and d[0] > 0)
    wsign += 0 if ok else 1
print(f"photon_from_constants: max |H|/E^2 = {wn:.2e}, max Q error = {wq2:.2e}, sign/magnitude violations = {wsign}")

for spin in (0.0, 0.7, -0.7):
    st = Spacetime(1.0, spin)
    for pro in (True, False):
        y = equatorial_photon(st, 30.0, 4.0, prograde=pro)
        d = geodesic_rhs(st, y)
        # prograde: a L_z > 0 ; dphi/dlambda has the sign of L_z at large r
        print(f"equatorial a={spin:+.1f} prograde={pro}: L_z={y[7]:+.3f} a*L_z={st.a*y[7]:+.3f} dphi/dl={d[3]:+.4f} "
              f"dr/dl={d[1]:+.4f} H={float(hamiltonian(st, y)):.1e}")
        for r0 in (1.3 if abs(spin) > 0.5 else 3.5, 3.0, 5.0, 12.0):
            if r0 <= 1 + math.sqrt(1 - spin**2):
                continue
            try:
                y = tangential_photon(st, r0, prograde=pro)
            except ValueError as e:
                print("   tangential", r0, "raised", e); continue
            xi = y[7] / -y[4]
            R_num = ((r0**2 + st.a**2) - st.a * xi) ** 2 - (r0**2 - 2 * r0 + st.a**2) * (xi - st.a) ** 2
            d = geodesic_rhs(st, y)
            print(f"   tangential r0={r0:5.2f}: xi={xi:+9.4f} R(r0)={R_num:+.1e} H={float(hamiltonian(st, y)):+.1e} dt/dl={d[0]:+.3f} a*L={st.a*y[7]:+.3f}")

# null_momentum_pt: future-directed root has dt/dlambda > 0
st = Spacetime(1.0, 0.9)
x = np.array([0, 1.8, 1.4, 0])  # inside the ergosphere
pt_f = null_momentum_pt(st, x, 0.3, 1.0, 2.0)
y = np.array([*x, pt_f, 0.3, 1.0, 2.0])
print("null_momentum_pt inside ergosphere: p_t =", float(pt_f), "dt/dl =", float(geodesic_rhs(st, y)[0]), "H =", float(hamiltonian(st, y)))
