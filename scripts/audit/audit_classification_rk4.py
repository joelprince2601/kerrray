"""Audit item 11a: what happens to captured photons under fixed-step RK4.

Records, for every captured photon, the last state before termination and
the termination code; reclassifies with an alternative rule (CAPTURED if
r <= r_+ + eps, evaluated BEFORE the r < 0 rule) to separate integration
failure from the rule priority.
Run: ./.venv/Scripts/python.exe scripts/audit/audit_classification_rk4.py
"""
import math
import warnings

import numpy as np

from kerrray.geodesics.initial_conditions import equatorial_photon, photon_from_constants
from kerrray.geodesics.integrators import IntegratorOptions, TerminationOptions
from kerrray.geodesics.integrators_batch import integrate_batch
from kerrray.geometry.horizons import outer_horizon
from kerrray.geometry.metric import Spacetime
from kerrray.photons.classification import TerminationState as TS
from kerrray.photons.orbits import critical_impact_parameters

warnings.simplefilter("ignore")
rng = np.random.default_rng(11)

# single representative Schwarzschild case, h = 0.05 (paper section 5.3)
st = Spacetime(1.0, 0.0)
y0 = equatorial_photon(st, 50.0, 4.0)
trail = []
res = integrate_batch(st, y0[None], IntegratorOptions(method="rk4", step_size=0.05, max_steps=10**6), TerminationOptions(1e-6, 1000.0),
                      on_step=lambda i, y, c: trail.append((float(y[0, 1]), float(y[0, 5]), int(c[0]))))
print("Schwarzschild b=4, h=0.05, last 5 accepted (r, p_r, code):", [(round(r, 4), f"{p:.3g}", TS(c).name) for r, p, c in trail[-5:]])

# many captured photons, a = 0 and 0.9, equatorial and off-equatorial
for spin in (0.0, 0.9):
    st = Spacetime(1.0, spin)
    rp = outer_horizon(st)
    Y = []
    bpro, bret = critical_impact_parameters(st)
    for k in range(200):
        pro = bool(k % 2)
        f = rng.uniform(0.2, 0.9)
        if k % 4 < 2:
            Y.append(equatorial_photon(st, 50.0, f * (bpro if pro else bret), prograde=pro))
        else:
            while True:
                xi = rng.uniform(-2, 2); eta = rng.uniform(0, 15); th0 = rng.uniform(0.5, 2.6)
                try:
                    Y.append(photon_from_constants(st, 50.0, th0, 1.0, xi, eta, sign_r=-1, sign_theta=int(rng.choice([-1, 1]))))
                    break
                except ValueError:
                    continue
    Y = np.array(Y)
    ref = integrate_batch(st, Y, IntegratorOptions(method="rk45", rtol=1e-10, atol=1e-12, lambda_max=1e5, max_steps=10**6), TerminationOptions(1e-6, 1000.0))
    Y = Y[ref.state == TS.CAPTURED]
    print(f"a={spin}: {len(Y)} of 200 photons are captured under RK45 rtol 1e-10 (reference fate); RK4 runs use these")
    for h in (0.1, 0.05, 0.01):
        last = {}
        prev = {}

        def rec(idx, ys, codes):
            for i, y, c in zip(idx, ys, codes):
                prev[i] = last.get(i, (50.0, 0))
                last[i] = (float(y[1]), int(c))

        r = integrate_batch(st, Y, IntegratorOptions(method="rk4", step_size=h, max_steps=10**6, lambda_max=1e5),
                            TerminationOptions(1e-6, 1000.0), on_step=rec)
        codes = np.bincount(r.state, minlength=7)
        r_end = r.Y[:, 1]
        r_prev = np.array([prev[i][0] for i in range(len(Y))])
        n = len(Y)
        alt_captured = np.sum((r_end <= rp + 1e-6) | ~np.isfinite(r_end))
        print(f"a={spin} h={h}: codes {dict((TS(i).name, int(n)) for i, n in enumerate(codes) if n)}; "
              f"final r: min {np.nanmin(r_end):.3g} max {np.nanmax(r_end):.3g}; last r before termination in "
              f"[{r_prev.min():.3f}, {r_prev.max():.3f}] (r_+ = {rp:.3f}); "
              f"with capture evaluated before r<0: {alt_captured}/{n} captured, escaped {int(codes[1])}")
