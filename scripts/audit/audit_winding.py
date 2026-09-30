"""Audit item 9: near-critical winding law, independent of the ODE integrator.

Delta phi(delta) is computed by adaptive quadrature (mpmath, 30 digits) of
dphi/dr = Phi(r)/sqrt(R(r)) from the launch radius 50 M in to r_min and out to
60 M (the geometry of scripts/study_winding.py), with the endpoint singularity
removed by r = r_min + s^2.
Run: ./.venv/Scripts/python.exe scripts/audit/audit_winding.py
"""
import json
import math
from pathlib import Path

import mpmath as mp
import numpy as np

mp.mp.dps = 40
M = 1


def r_ph_of(A, pro):
    s = 1 if pro else -1
    return 2 * M * (1 + mp.cos(mp.mpf(2) / 3 * mp.acos(-s * A / M)))


def xi_c_of(a, pro):
    """Signed xi on the equatorial photon orbit (E = 1)."""
    a = mp.mpf(a)
    A = abs(a)
    if A == 0:
        return (1 if pro else -1) * 3 * mp.sqrt(3)
    rph = r_ph_of(A, pro)
    xi_abs = abs(-(rph**2 * (rph - 3) + A**2 * (rph + 1)) / (A * (rph - 1)))
    sgn = (1 if a >= 0 else -1) * (1 if pro else -1)
    return sgn * xi_abs


def R(r, a, xi):
    D = r * r - 2 * M * r + a * a
    return ((r * r + a * a) - a * xi) ** 2 - D * (xi - a) ** 2


def Rpp(r, a, xi):  # exact second derivative
    return 12 * r * r + 4 * a * a - 4 * a * xi - 2 * (xi - a) ** 2


def Phi(r, a, xi):
    D = r * r - 2 * M * r + a * a
    return (xi - a) + a / D * ((r * r + a * a) - a * xi)


def gamma_phi(a, pro):
    a = mp.mpf(a)
    rph = r_ph_of(abs(a), pro)
    xi = xi_c_of(a, pro)
    return mp.sqrt(Rpp(rph, a, xi) / 2) / abs(Phi(rph, a, xi)), rph, xi


def delta_phi(a, pro, delta, r_in=50, r_out=60):
    a = mp.mpf(a)  # all-mp arithmetic (a double-rounded a*a shifts b_c by ~1e-17)
    xi = xi_c_of(a, pro) * (1 + delta)
    rph = r_ph_of(abs(a), pro)
    # r_min: largest root of R, just above r_ph
    assert R(rph, a, xi) < 0, "photon must have a turning point above r_ph"
    hi = rph + mp.sqrt(delta)
    while R(hi, a, xi) <= 0:
        hi = rph + 2 * (hi - rph)
    rmin = mp.findroot(lambda x: R(x, a, xi), (rph, hi), solver="anderson")
    if not (rph < rmin <= hi):
        raise RuntimeError("wrong root")
    D0 = rmin * rmin - 2 * M * rmin + a * a
    Rp = 4 * rmin * ((rmin**2 + a * a) - a * xi) - (2 * rmin - 2 * M) * (xi - a) ** 2  # R'(r_min)
    lim = 2 * Phi(rmin, a, xi) / mp.sqrt(Rp)

    def f(s):
        if s * s < mp.mpf("1e-22") * rmin:
            return lim
        return 2 * s * Phi(rmin + s * s, a, xi) / mp.sqrt(R(rmin + s * s, a, xi))
    brk = [0] + [mp.sqrt(x) for x in (mp.sqrt(delta) * 1e-2, mp.sqrt(delta), mp.sqrt(delta) * 1e2, mp.mpf("0.1"), 1)]
    brk = sorted(set([b for b in brk if b < mp.sqrt(r_in - rmin)]))
    I_in = mp.quad(f, brk + [mp.sqrt(r_in - rmin)])
    I_out = mp.quad(f, brk + [mp.sqrt(r_out - rmin)])
    return I_in + I_out, rmin - rph


data = json.loads(Path("paper/data/winding.json").read_text())
deltas = data["deltas"]
print("case             gamma_phi(exact)  gamma(paper)  pred/decade  paper_pred  quad fit 1e-3..1e-8  paper measured  quad asymptotic(1e-12..1e-16)")
out_rows = []
for row in data["rows"]:
    a = row["spin"]; pro = row["direction"] == "prograde"
    g, rph, xi = gamma_phi(a, pro)
    pred = mp.log(10) / (2 * mp.pi * g)
    turns, offs = [], []
    for d in deltas:
        dphi, off = delta_phi(a, pro, mp.mpf(d))
        turns.append(float(abs(dphi) / (2 * mp.pi))); offs.append(float(off))
    slope = -np.polyfit(np.log10(deltas), turns, 1)[0]
    far = [mp.mpf(10) ** -k for k in (12, 14, 16)]
    tf = [float(abs(delta_phi(a, pro, d)[0]) / (2 * mp.pi)) for d in far]
    slope_far = -np.polyfit([-12, -14, -16], tf, 1)[0]
    off_slope = np.polyfit(np.log10(deltas), np.log10(offs), 1)[0]
    turn_err = max(abs(t1 - t2) for t1, t2 in zip(turns, row["turns"]))
    print(f"a={a:<4} {'pro ' if pro else 'retro'}  {float(g):.6f}   {row['gamma_phi']:.6f}   {float(pred):.5f}   "
          f"{row['predicted_turns_per_decade']:.5f}   {slope:.5f}   {row['rate_per_decade']:.5f}   {slope_far:.6f}   "
          f"| offset ~ delta^{off_slope:.3f}; max |turns_quad - turns_paper| = {turn_err:.1e}")
    out_rows.append(dict(spin=a, pro=pro, gamma=float(g), pred=float(pred), quad_fit=slope, paper=row["rate_per_decade"], asym=slope_far))

# Lyapunov exponent and orbital frequency from first principles (coordinate time t)
print("\nLyapunov exponent lambda and orbital frequency Omega_c (per unit t) of the equatorial photon orbit:")
for a, pro in ((0.0, True), (0.9, True), (0.9, False), (0.99, True)):
    g, rph, xi = gamma_phi(mp.mpf(a), pro)
    D = rph**2 - 2 * rph + a * a
    A_ = (rph**2 + a * a) ** 2 - a * a * D
    Sig = rph**2
    gtt = -A_ / (Sig * D); gtp = -2 * a * rph / (Sig * D); gpp = (D - a * a) / (Sig * D)
    tdot = -gtt + gtp * xi          # p_t = -1, p_phi = xi
    phidot = -gtp + gpp * xi
    Omega = phidot / tdot
    # rdot^2 = R/Sigma^2 = V(r); linearised: d^2 x/dlambda^2 = (V''/2) x, V'' = R''/r^4 at the orbit
    lam_affine = mp.sqrt(Rpp(rph, a, xi) / (2 * rph**4))
    lyap = lam_affine / tdot
    s = 1 if pro else -1
    Omega_bpt = s * (1 if a >= 0 else -1) / (rph**1.5 + s * abs(a))
    print(f"  a={a:<4} {'pro ' if pro else 'retro'}: lambda={float(lyap):.8f} Omega={float(Omega):.8f} "
          f"(BPT 1/(r^1.5 +- a) = {float(Omega_bpt):.8f}) lambda/|Omega|={float(lyap/abs(Omega)):.8f} gamma_phi={float(g):.8f}")
print("Schwarzschild reference 1/(3 sqrt 3) =", 1 / (3 * math.sqrt(3)))
