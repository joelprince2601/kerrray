"""How far past the 60 M escape radius does the last step of each winding photon land?

research.md section 4.3 attributes the small difference between integrated and
quadrature turn counts to the final accepted step overshooting the escape
radius. This script repeats the integrations of scripts/study_winding.py
(same photons, tolerances and termination) and prints r_final - 60 M.
Run: ./.venv/Scripts/python.exe scripts/audit/audit_winding_overshoot.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import study_winding as sw  # noqa: E402

from kerrray.geodesics.integrators import IntegratorOptions, TerminationOptions, integrate

DELTAS = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
integ = IntegratorOptions(method="rk45", rtol=1e-12, atol=1e-12 * 1e-2, max_steps=400_000)
term = TerminationOptions(horizon_epsilon=1e-6, escape_radius=60.0)
allv = []
for spin, prograde in [(0.0, True), (0.3, True), (0.3, False), (0.6, True), (0.6, False), (0.9, True), (0.9, False), (0.99, True), (0.99, False)]:
    st = sw.Spacetime(1.0, spin)
    b_pro, b_retro = sw.critical_impact_parameters(st)
    b_c = b_pro if prograde else b_retro
    over = []
    for d in DELTAS:
        tr = integrate(st, sw.equatorial_photon(st, 50.0, b_c * (1 + d), prograde=prograde), integ, term, record=False)
        over.append(float(tr.y_end[1]) - 60.0)
    allv += over
    print(f"a={spin:<4} {'prograde' if prograde else 'retrograde':10s} r_final - 60 M: " + " ".join(f"{o:.3f}" for o in over))
print(f"all: {min(allv):.3f} .. {max(allv):.3f} M")
