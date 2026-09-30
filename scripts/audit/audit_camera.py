"""Audit items 6 and 8: ZAMO tetrad, pixel -> momentum map, orientation, finite-distance map.

Exact finite-distance map derived in the audit (E = 1, xi = L_z/E, p_theta at the observer):
    alpha_cam = -r_o xi N / (sqrt(g_phph) (1 - omega xi))
    beta_cam  =  r_o p_theta N / (sqrt(Sigma) (1 - omega xi)),   p_theta^2 = Theta(theta_o)
with N^2 = Sigma Delta / A the ZAMO lapse and omega = 2 M a r / A, all at (r_o, theta_o).
Run: ./.venv/Scripts/python.exe scripts/audit/audit_camera.py
"""
import math
import warnings

import numpy as np
import sympy as sp

from kerrray.geodesics.equations import geodesic_rhs, hamiltonian
from kerrray.geodesics.integrators import IntegratorOptions, TerminationOptions
from kerrray.geometry.metric import Spacetime, metric
from kerrray.photons.classification import TerminationState
from kerrray.photons.constants import carter_constant
from kerrray.photons.orbits import shadow_curve
from kerrray.raytracing.boundary import finite_distance_scale
from kerrray.raytracing.camera import Camera, initial_states, zamo_tetrad
from kerrray.raytracing.shadow import compute_shadow

warnings.simplefilter("ignore")
rng = np.random.default_rng(3)


def zamo_quantities(M, a, r, th):
    sig = r * r + a * a * math.cos(th) ** 2
    dlt = r * r - 2 * M * r + a * a
    A = (r * r + a * a) ** 2 - a * a * dlt * math.sin(th) ** 2
    N = math.sqrt(sig * dlt / A)
    omega = 2 * M * a * r / A
    gpp = A * math.sin(th) ** 2 / sig
    return sig, N, omega, gpp


def exact_map(M, a, r_o, th_o, xi, p_th):
    """(alpha_cam, beta_cam) of a photon with E = 1, L_z = xi, p_theta at the observer."""
    sig, N, omega, gpp = zamo_quantities(M, a, r_o, th_o)
    den = 1.0 - omega * xi
    return -r_o * xi * N / (math.sqrt(gpp) * den), r_o * p_th * N / (math.sqrt(sig) * den)


# ---- 6a. tetrad orthonormality ------------------------------------------
eta = np.diag([-1.0, 1, 1, 1])
w = 0.0
for _ in range(500):
    spin = rng.uniform(-0.999, 0.999); st = Spacetime(1.0, spin)
    r = 1 + math.sqrt(1 - spin**2) + 10 ** rng.uniform(-3, 3)
    th = rng.uniform(1e-3, math.pi - 1e-3)
    e = zamo_tetrad(st, r, th)
    w = max(w, float(np.max(np.abs(e @ metric(st, r, th) @ e.T - eta))))
    # ZAMO: e_(t) has zero angular momentum: u_phi = g_phi mu e_(t)^mu = 0
    u_low = metric(st, r, th) @ e[0]
    w = max(w, abs(u_low[3]) / max(1, abs(u_low[0])))
print(f"tetrad: max |e g e^T - eta| and |u_phi| of e_(t) over 500 points: {w:.2e}")

# ---- 6b. pixel -> momentum; exact finite-distance map reproduces the pixel coordinates
for spin, incl, r_o in ((0.0, 60.0, 1000.0), (0.9, 60.0, 1000.0), (0.99, 17.0, 50.0), (-0.7, 120.0, 20.0)):
    st = Spacetime(1.0, spin)
    cam = Camera(radius=r_o, inclination_deg=incl, fov=8.0, resolution=16)
    al, be = cam.pixel_coordinates()
    Yf = initial_states(cam, st, reverse=False)
    Yb = initial_states(cam, st, reverse=True)
    E = -Yf[:, 4]
    xi = Yf[:, 7] / E
    pth = Yf[:, 6] / E
    th_o = cam.inclination_rad
    pred = np.array([exact_map(1.0, st.a, r_o, th_o, x, p) for x, p in zip(xi, pth)])
    err = max(np.max(np.abs(pred[:, 0] - al.ravel())), np.max(np.abs(pred[:, 1] - be.ravel())))
    # backward states: same xi and eta, E < 0, moving inward
    Eb = -Yb[:, 4]
    xib = Yb[:, 7] / Eb
    etab = carter_constant(st, Yb) / Eb**2
    etaf = carter_constant(st, Yf) / E**2
    drb = geodesic_rhs(st, Yb)[:, 1]
    print(f"a={spin:+.2f} i={incl:5.1f} r_o={r_o:6.0f}: |exact map - pixel| = {err:.1e}; "
          f"E_f>0: {bool(np.all(E > 0))}, E_b<0: {bool(np.all(Eb < 0))}, dr/dl_b<0: {bool(np.all(drb < 0))}, "
          f"max|xi_b-xi_f| = {np.max(np.abs(xib - xi)):.1e}, max|eta_b-eta_f| = {np.max(np.abs(etab - etaf)):.1e}, "
          f"max|H| = {np.max(np.abs(hamiltonian(st, Yb))):.1e}")

# ---- 8a. asymptotic expansion of the correction factors (symbolic) ------
u, M_, a_, th_, xi_ = sp.symbols("u M a theta xi", positive=True)
r_ = 1 / u
sig = r_**2 + a_**2 * sp.cos(th_) ** 2
dlt = r_**2 - 2 * M_ * r_ + a_**2
A = (r_**2 + a_**2) ** 2 - a_**2 * dlt * sp.sin(th_) ** 2
N = sp.sqrt(sig * dlt / A)
omega = 2 * M_ * a_ * r_ / A
gpp = A * sp.sin(th_) ** 2 / sig
k = 1 / sp.sqrt(1 - 2 * M_ * u)
fa = sp.series(k * r_ * sp.sin(th_) * N / (sp.sqrt(gpp) * (1 - omega * xi_)), u, 0, 4).removeO()
fb = sp.series(k * r_ * N / (sp.sqrt(sig) * (1 - omega * xi_)), u, 0, 4).removeO()
print("k * alpha_cam / alpha_Bardeen = ", sp.simplify(sp.expand(fa)))
print("k * beta_cam  / beta_Bardeen  = ", sp.simplify(sp.expand(fb)))
print("   lapse^2 series:", sp.series(sig * dlt / A, u, 0, 4))

# ---- 8b. residual on the analytic curve at r_o = 1000 M, i = 60 deg -----
print("\nresidual after the 1/sqrt(1-2M/r_o) correction, i = 60 deg (analytic curve mapped exactly):")
for r_o in (1000.0, 100.0):
    for spin in (0.0, 0.5, 0.9, 0.99):
        st = Spacetime(1.0, spin)
        th_o = math.radians(60.0)
        kf = finite_distance_scale(st, r_o)
        if spin == 0.0:
            ang = np.linspace(0, 2 * math.pi, 721)[:-1]
            alB, beB = 3 * math.sqrt(3) * np.cos(ang), 3 * math.sqrt(3) * np.sin(ang)
        else:
            alB, beB = shadow_curve(st, 60.0, 2000)
        xi = -alB * math.sin(th_o)
        mapped = np.array([exact_map(1.0, st.a, r_o, th_o, x, b) for x, b in zip(xi, beB)]) * kf
        cx, cy = alB.mean(), beB.mean()
        rad_B = np.hypot(alB - cx, beB - cy)
        dvec = np.stack([mapped[:, 0] - alB, mapped[:, 1] - beB], 1)
        radial = (dvec[:, 0] * (alB - cx) + dvec[:, 1] * (beB - cy)) / rad_B
        raw = np.array([exact_map(1.0, st.a, r_o, th_o, x, b) for x, b in zip(xi, beB)])
        raw_rad = ((raw[:, 0] - alB) * (alB - cx) + (raw[:, 1] - beB) * (beB - cy)) / rad_B
        print(f"  r_o={r_o:6.0f} a={spin:4.2f}: uncorrected radial offset mean {raw_rad.mean():+.3e} M; "
              f"corrected: max|disp| {np.max(np.hypot(*dvec.T)):.2e} M, radial mean {radial.mean():+.2e}, "
              f"max|radial| {np.max(np.abs(radial)):.2e} M, alpha-shift of centroid {mapped[:,0].mean()-alB.mean():+.2e} M")

# ---- 6c. orientation from actual images ----------------------------------
print("\norientation check (numba backend, 64^2, fov 8, r_o 1000, rtol 1e-8):")
integ = IntegratorOptions(method="rk45", rtol=1e-8, atol=1e-10)
for spin, incl in ((0.9, 60.0), (0.99, 90.0), (-0.9, 60.0)):
    st = Spacetime(1.0, spin)
    cam = Camera(radius=1000.0, inclination_deg=incl, fov=8.0, resolution=64)
    img = compute_shadow(st, cam, integ, TerminationOptions(1e-6, 1000.0), backend="numba")
    sh = img.state != int(TerminationState.ESCAPED)
    kf = finite_distance_scale(st, 1000.0)
    al = img.alpha[sh] * kf
    alB, beB = shadow_curve(st, incl, 2000)
    print(f"  a={spin:+.2f} i={incl:.0f}: image alpha range [{al.min():+.3f}, {al.max():+.3f}] centroid {al.mean():+.3f}; "
          f"analytic alpha range [{alB.min():+.3f}, {alB.max():+.3f}]; pixel {img.pixel_size*kf:.3f}")
