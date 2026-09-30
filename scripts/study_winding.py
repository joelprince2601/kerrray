"""Near-critical winding of equatorial photons versus the orbit's instability.

A photon with impact parameter b slightly above the critical value b_c of an
unstable circular photon orbit winds around the hole many times before it
escapes. Linearising the equatorial Carter equations about the orbit radius
r_ph gives the growth rate of a radial perturbation per radian of azimuth,

    gamma_phi = sqrt(R''(r_ph) / 2) / |Phi(r_ph)|,

with (E = 1, theta = pi/2, Sigma = r^2)

    R(r)   = [(r^2 + a^2) - a xi]^2 - Delta (xi - a)^2        (Sigma dr/dlambda)^2
    Phi(r) = (xi - a) + (a / Delta) [(r^2 + a^2) - a xi]      Sigma dphi/dlambda

(Carter 1968; Bardeen, Press & Teukolsky 1972). The closest approach sits at
r_ph + O(sqrt(b/b_c - 1)), so the azimuth accumulated near the orbit is
Delta phi ~ -(1/gamma_phi) ln(b/b_c - 1) and the number of turns grows by

    ln(10) / (2 pi gamma_phi)   per decade of (b/b_c - 1).

For Schwarzschild gamma_phi = 1 and the rate is ln(10)/(2 pi) = 0.3665, the
strong-deflection coefficient of Bozza (2002). This script evaluates the
prediction for several spins (R'' by central differences of the exact
polynomial) and measures the rate by integrating photons with the engine at
b = b_c (1 + delta), delta = 1e-3 ... 1e-8, then fitting turns against
log10(delta).

Outputs: paper/data/winding.json and paper/figures/winding.png.
"""

from __future__ import annotations

import json
import math
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from kerrray.geodesics.initial_conditions import equatorial_photon
from kerrray.geodesics.integrators import IntegratorOptions, TerminationOptions, integrate
from kerrray.geometry.metric import Spacetime
from kerrray.photons.classification import TerminationState
from kerrray.photons.orbits import critical_impact_parameters, equatorial_photon_orbit_radius
from kerrray.photons.trajectories import azimuthal_winding
from kerrray.utils.manifest import collect_environment

ROOT = Path(__file__).resolve().parents[1]
DATA, FIGS = ROOT / "paper" / "data", ROOT / "paper" / "figures"


def predicted_gamma(spin: float, prograde: bool) -> tuple[float, float, float]:
    """(gamma_phi, r_ph, xi) for the equatorial circular photon orbit."""
    st = Spacetime(1.0, spin)
    a = st.a
    r_ph = equatorial_photon_orbit_radius(st, prograde=prograde)
    b_pro, b_retro = critical_impact_parameters(st)
    # signed xi = L_z / E: positive for prograde when a >= 0
    xi = b_pro if prograde else -b_retro

    def R(r: float) -> float:
        delta = r * r - 2 * r + a * a
        return (r * r + a * a - a * xi) ** 2 - delta * (xi - a) ** 2

    h = 1e-4 * r_ph
    r2 = (R(r_ph + h) - 2 * R(r_ph) + R(r_ph - h)) / (h * h)
    delta = r_ph**2 - 2 * r_ph + a * a
    phi_rate = (xi - a) + a / delta * (r_ph**2 + a * a - a * xi)
    return math.sqrt(r2 / 2.0) / abs(phi_rate), r_ph, xi


def measured_rate(spin: float, prograde: bool, deltas: list[float], rtol: float = 1e-12) -> dict:
    st = Spacetime(1.0, spin)
    b_pro, b_retro = critical_impact_parameters(st)
    b_c = b_pro if prograde else b_retro
    integ = IntegratorOptions(method="rk45", rtol=rtol, atol=rtol * 1e-2, max_steps=400_000)
    term = TerminationOptions(horizon_epsilon=1e-6, escape_radius=60.0)
    turns, states = [], []
    for d in deltas:
        tr = integrate(st, equatorial_photon(st, 50.0, b_c * (1 + d), prograde=prograde), integ, term, record=True)
        turns.append(abs(azimuthal_winding(tr)) / (2 * math.pi))
        states.append(tr.state.name)
    ok = [i for i, s in enumerate(states) if s == TerminationState.ESCAPED.name]
    slope = float(np.polyfit(np.log10([deltas[i] for i in ok]), [turns[i] for i in ok], 1)[0]) if len(ok) >= 3 else float("nan")
    return {"b_c": b_c, "turns": turns, "states": states, "rate_per_decade": -slope}


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    spins = [0.0, 0.3, 0.6, 0.9, 0.99]
    deltas = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
    rows = []
    t0 = time.perf_counter()
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        for spin in spins:
            for prograde in (True, False):
                if spin == 0.0 and not prograde:
                    continue
                gamma, r_ph, xi = predicted_gamma(spin, prograde)
                pred = math.log(10) / (2 * math.pi * gamma)
                m = measured_rate(spin, prograde, deltas)
                rows.append({"spin": spin, "direction": "prograde" if prograde else "retrograde", "r_ph": r_ph, "xi": xi,
                             "gamma_phi": gamma, "predicted_turns_per_decade": pred, **m,
                             "relative_difference": (m["rate_per_decade"] - pred) / pred})
                print(f"a={spin:<5} {'pro  ' if prograde else 'retro'} r_ph={r_ph:.4f} gamma_phi={gamma:.5f} "
                      f"predicted={pred:.5f} measured={m['rate_per_decade']:.5f} ({100 * (m['rate_per_decade'] - pred) / pred:+.2f}%) "
                      f"states={set(m['states'])}", flush=True)
    env = collect_environment()
    out = {"deltas": deltas, "rows": rows, "rtol": 1e-12, "launch_radius": 50.0, "git_commit": env.git_commit,
           "git_dirty": env.git_dirty, "hardware": env.hardware(), "runtime_s": time.perf_counter() - t0}
    (DATA / "winding.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for r in rows:
        c = "#D95319" if r["direction"] == "prograde" else "#0072BD"
        axes[0].semilogx(deltas, r["turns"], marker="o", ms=3, color=c, alpha=0.4 + 0.6 * r["spin"], label=f"a*={r['spin']} {r['direction'][:5]}")
    axes[0].invert_xaxis()
    axes[0].set(xlabel="b / b_c - 1", ylabel="turns around the hole", title="Winding of near-critical photons")
    axes[0].legend(fontsize=7, ncol=2)
    pro = [r for r in rows if r["direction"] == "prograde"]
    ret = [r for r in rows if r["direction"] == "retrograde"]
    fine = np.linspace(0, 0.99, 100)
    axes[1].plot(fine, [math.log(10) / (2 * math.pi * predicted_gamma(s, True)[0]) for s in fine], color="#D95319", lw=1, label="prediction, prograde")
    axes[1].plot(fine, [math.log(10) / (2 * math.pi * predicted_gamma(s, False)[0]) for s in fine], color="#0072BD", lw=1, label="prediction, retrograde")
    axes[1].plot([r["spin"] for r in pro], [r["rate_per_decade"] for r in pro], "o", color="#D95319", label="measured, prograde")
    axes[1].plot([r["spin"] for r in ret], [r["rate_per_decade"] for r in ret], "s", color="#0072BD", label="measured, retrograde")
    axes[1].set(xlabel="spin a*", ylabel="turns per decade of (b/b_c - 1)", title="Measured vs predicted winding rate")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGS / "winding.png", dpi=150)
    print(f"done in {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
