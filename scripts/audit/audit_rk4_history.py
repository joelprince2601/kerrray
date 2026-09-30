"""Audit item 11a (continued): step-by-step history of RK4 near the horizon, and RK4 shadow masks.

Run: ./.venv/Scripts/python.exe scripts/audit/audit_rk4_history.py
"""
import warnings
from collections import defaultdict

import numpy as np

from kerrray.geodesics.initial_conditions import equatorial_photon, photon_from_constants
from kerrray.geodesics.integrators import IntegratorOptions, TerminationOptions
from kerrray.geodesics.integrators_batch import integrate_batch
from kerrray.geometry.horizons import outer_horizon
from kerrray.geometry.metric import Spacetime
from kerrray.photons.classification import TerminationState as TS
from kerrray.raytracing.camera import Camera
from kerrray.raytracing.shadow import compute_shadow

warnings.simplefilter("ignore")
rng = np.random.default_rng(11)
st = Spacetime(1.0, 0.9)
rp = outer_horizon(st)
Y = []
while len(Y) < 200:
    xi = rng.uniform(-2, 2); eta = rng.uniform(0, 15); th0 = rng.uniform(0.5, 2.6)
    try:
        Y.append(photon_from_constants(st, 50.0, th0, 1.0, xi, eta, sign_r=-1, sign_theta=int(rng.choice([-1, 1]))))
    except ValueError:
        pass
Y = np.array(Y)
ref0 = integrate_batch(st, Y, IntegratorOptions(method="rk45", rtol=1e-10, atol=1e-12, lambda_max=1e5, max_steps=10**6), TerminationOptions(1e-6, 1000.0))
Y = Y[ref0.state == TS.CAPTURED]
print(f"{len(Y)} of 200 photons captured under RK45 rtol 1e-10; RK4 h = 0.1 history of these:")
hist = defaultdict(list)
integrate_batch(st, Y, IntegratorOptions(method="rk4", step_size=0.1, max_steps=10**6, lambda_max=1e5), TerminationOptions(1e-6, 1000.0),
                on_step=lambda idx, ys, c: [hist[i].append((y[1], y[2], int(cc))) for i, y, cc in zip(idx, ys, c)])
kinds = defaultdict(int)
examples = {}
for i, h in hist.items():
    r = np.array([x[0] for x in h])
    # first step leaving the physical neighbourhood of the horizon: r_new <= r_+ or a jump of > 1 M from r < 3 M
    bad = next((k for k in range(1, len(r)) if r[k] <= rp or (r[k - 1] < 3 and r[k] - r[k - 1] > 1.0) or not np.isfinite(r[k])), None)
    if bad is None:
        th = np.array([x[1] for x in h])
        kind = f"other (last theta {th[-1]:.3g}, min r {r.min():.3f})"
        bad = len(r) - 1
    else:
        kind = "crossed r_+ (r_new <= r_+)" if r[bad] <= rp else "jumped outward"
    fate = TS(h[-1][2]).name
    kinds[(kind, fate)] += 1
    examples.setdefault((kind, fate), [round(float(x), 4) for x in r[max(0, bad - 3): bad + 3]])
for k, v in kinds.items():
    print(f"{k}: {v} photons; example r sequence around the bad step: {examples[k]}")

print("\nshadow masks, a = 0.9, i = 60, r_o = 1000, fov 8 (numba): RK4 vs RK45 rtol 1e-8")
cam = Camera(radius=1000.0, inclination_deg=60.0, fov=8.0, resolution=64)
term = TerminationOptions(1e-6, 1000.0)
ref = compute_shadow(st, cam, IntegratorOptions(method="rk45", rtol=1e-8, atol=1e-10), term, backend="numba")
ref_sh = ref.state != TS.ESCAPED
for h in (0.1, 0.05, 0.01):
    img = compute_shadow(st, cam, IntegratorOptions(method="rk4", step_size=h, max_steps=10**6, lambda_max=1e5), term, backend="numba")
    sh = img.state != TS.ESCAPED
    cnt = {TS(i).name: int(n) for i, n in enumerate(np.bincount(img.state.ravel(), minlength=7)) if n}
    print(f"  h={h}: states {cnt}; reference shadow {int(ref_sh.sum())} px; non-escaped-mask differs from reference at "
          f"{int((sh != ref_sh).sum())} px (false escapes {int((ref_sh & ~sh).sum())}, false shadow {int((~ref_sh & sh).sum())})")

print("\nlocation of RK4 false escapes (distance in pixels to the nearest reference-escaped pixel):")
from scipy.ndimage import distance_transform_edt
dist = distance_transform_edt(ref_sh.astype(bool))
for h in (0.1, 0.01):
    img = compute_shadow(st, cam, IntegratorOptions(method="rk4", step_size=h, max_steps=10**6, lambda_max=1e5), term, backend="numba")
    fe = ref_sh & (img.state != TS.ESCAPED) == False  # noqa: E712
    fe = ref_sh & (img.state == TS.ESCAPED)
    print(f"  h={h}: distances {sorted(np.round(dist[fe], 1).tolist())}")
