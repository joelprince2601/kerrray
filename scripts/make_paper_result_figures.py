"""Example and validation figures of research.md, drawn from recorded runs.

Writes to paper/figures/ (PDF for LaTeX, PNG for research.md). Every figure
either re-plots stored data or repeats the integration of a recorded run with
that run's configuration; in the second case the script checks that the
recomputed values equal the recorded ones and stops otherwise. No new
experiment is made.

- fig_trajectories:   equatorial photon orbits for a* = 0 and 0.9. Repeats the
                      trajectories of scripts/poc_demo.py (RK45, rtol 1e-9,
                      atol 1e-11, launch at 40 M) and checks the end states and
                      closest approaches against results/poc/summary.json.
- fig_shadow_example: the 64 x 64 shadow of run 20260929T054034Z-de5a37
                      (a* = 0.9, i = 60 deg, field of view +-12 M), re-plotted
                      from runs/<id>/shadow.npz with the Bardeen curve.
- fig_convergence_bc: convergence of the Schwarzschild critical impact parameter
                      with rtol (RK45) and step size (RK4), re-plotted from
                      reports/20260929T085733Z-e881ee/summary.json.
- fig_kerr_validation: the near-extremal prograde photon (a* = 0.999) and the
                      conservation drifts along an off-equatorial photon
                      (a* = 0.999). Repeats two integrations of run
                      20260929T085922Z-0e266a and checks them against its summary.
- fig_frame_dragging: equatorial photons at b = 6 M for a* = -0.9, 0, +0.9 and
                      zero-angular-momentum photons. Repeats the main family of
                      run 20260929T114056Z-8c5289 and checks it against its summary.

Usage: ./.venv/Scripts/python.exe scripts/make_paper_result_figures.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402

from kerrray.geodesics.initial_conditions import equatorial_photon  # noqa: E402
from kerrray.geodesics.integrators import IntegratorOptions, TerminationOptions, integrate  # noqa: E402
from kerrray.geometry.horizons import horizon_radii  # noqa: E402
from kerrray.geometry.metric import kerr, schwarzschild  # noqa: E402
from kerrray.photons.orbits import critical_impact_parameters, equatorial_photon_orbit_radius, shadow_curve  # noqa: E402
from kerrray.photons.trajectories import closest_approach, to_cartesian  # noqa: E402
from kerrray.physics.frame_dragging import frame_dragging_experiment  # noqa: E402
from kerrray.utils.config import experiment_block, load_config  # noqa: E402
from kerrray.validation import integrator_options_from_config, termination_options_from_config  # noqa: E402
from kerrray.validation.conservation import drift_summary, off_equatorial_ray  # noqa: E402
from kerrray.validation.convergence import loglog_order  # noqa: E402
from kerrray.validation.kerr import KerrValidationParams  # noqa: E402
from kerrray.geodesics import photon_from_constants  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures"
RUN_SHADOW = "20260929T054034Z-de5a37"
RUN_SCHW = "20260929T085733Z-e881ee"
RUN_KERR = "20260929T085922Z-0e266a"
RUN_DRAG = "20260929T114056Z-8c5289"

plt.rcParams.update({
    "font.size": 8,
    "mathtext.fontset": "cm",
    "axes.linewidth": 0.7,
    "lines.linewidth": 1.1,
    "legend.fontsize": 6.8,
    "legend.frameon": False,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
})
# palette of the existing paper figures
C_BLUE, C_RED, C_GREEN, C_PURPLE = "#0072BD", "#D95319", "#77AC30", "#7E2F8E"
INK = "#1a1a1a"


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.02,
                    metadata={"CreationDate": None} if ext == "pdf" else None)
    plt.close(fig)
    print("wrote", OUT / f"{name}.pdf", "and .png")


def close(a, b, rel=1e-9, what=""):
    ok = abs(a - b) <= rel * max(abs(a), abs(b), 1e-300)
    if not ok:
        raise SystemExit(f"recomputed {what} = {a!r} differs from the recorded {b!r}")


def summary(run_id):
    return json.loads((ROOT / "reports" / run_id / "summary.json").read_text(encoding="utf-8"))["results"]


def horizon_disk(ax, st, **kw):
    """Outer horizon in the plotting embedding rho = sqrt(r^2 + a^2) sin(theta)."""
    rh = math.hypot(horizon_radii(st)[0], st.a)
    ax.add_patch(Circle((0.0, 0.0), rh, color=INK, zorder=3, **kw))
    return rh


def spin_text(spin):
    return "0" if spin == 0.0 else f"{spin:+g}"


def ring(ax, st, r, **kw):
    ax.add_patch(Circle((0.0, 0.0), math.hypot(r, st.a), fill=False, **kw))


# --------------------------------------------------------------------------- trajectories
def trajectories():
    poc = json.loads((ROOT / "results" / "poc" / "summary.json").read_text(encoding="utf-8"))["trajectories"]
    integ = IntegratorOptions(method="rk45", rtol=1e-9, atol=1e-11)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.45))
    k = 0
    for ax, spin, tag in zip(axes, (0.0, 0.9), ("(a)", "(b)")):
        st = kerr(1.0, spin) if spin else schwarzschild(1.0)
        bp, br = critical_impact_parameters(st)
        rays = [(bp * 0.97, True, C_PURPLE, r"$b = 0.97\,b_c$, captured"),
                (bp * 1.003, True, C_RED, r"$b = 1.003\,b_c$, escapes"),
                (bp * 1.3, True, C_GREEN, r"$b = 1.3\,b_c$, escapes"),
                (br * 1.003, False, C_BLUE, r"retrograde, $b = 1.003\,b_c$, escapes")]
        for b, pro, col, label in rays:
            tr = integrate(st, equatorial_photon(st, 40.0, b, prograde=pro), integ,
                           TerminationOptions(horizon_epsilon=1e-6, escape_radius=60.0))
            rec = poc[k]
            k += 1
            if tr.state.name != rec["state"] or rec["prograde"] != pro:
                raise SystemExit(f"trajectory {k}: state {tr.state.name} differs from the recorded {rec['state']}")
            close(b, rec["b"], 1e-14, "impact parameter")
            close(closest_approach(tr), rec["closest_approach"], 1e-12, "closest approach")
            x, y, _ = to_cartesian(tr)
            if spin == 0.0 and not pro:
                ax.plot(x, y, color=col, lw=1.0, ls=(0, (4, 2)), label=label + " (mirror image)", zorder=2)
            else:
                ax.plot(x, y, color=col, lw=1.0, label=label, zorder=2)
        horizon_disk(ax, st)
        ring(ax, st, equatorial_photon_orbit_radius(st, True), color="0.45", lw=0.6, ls=":", zorder=1)
        if spin:
            ring(ax, st, equatorial_photon_orbit_radius(st, False), color="0.45", lw=0.6, ls=":", zorder=1)
        ax.set_xlim(-14, 14)
        ax.set_ylim(-14, 14)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$x$ [$M$]")
        ax.set_ylabel(r"$y$ [$M$]")
        ax.set_title(f"{tag} $a_* = {spin:g}$", fontsize=8.5)
        ax.legend(loc="lower left", handlelength=1.8, borderaxespad=0.3)
        ax.tick_params(labelsize=7)
    if k != len(poc):
        raise SystemExit("unexpected number of recorded trajectories")
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig_trajectories")


# --------------------------------------------------------------------------- shadow example
def shadow_example():
    z = np.load(ROOT / "runs" / RUN_SHADOW / "shadow.npz")
    rec = summary(RUN_SHADOW)
    spin, incl = float(z["spin"]), float(z["camera_inclination_deg"])
    st = kerr(float(z["mass"]), spin)
    k = float(rec["finite_distance_scale"])
    close(k, 1.0 / math.sqrt(1.0 - 2.0 / float(z["camera_radius"])), 1e-14, "finite-distance factor")
    alpha, beta, cap = z["alpha"] * k, z["beta"] * k, z["captured"].astype(bool)
    if int(cap.sum()) != rec["counts"]["CAPTURED"] or int((~cap).sum()) != rec["counts"]["ESCAPED"]:
        raise SystemExit("mask counts differ from the recorded summary")
    # pixel grid: alpha along one axis, beta along the other
    if np.ptp(alpha[0, :]) > 0:
        a1, b1, m = alpha[0, :], beta[:, 0], cap
    else:
        a1, b1, m = alpha[:, 0], beta[0, :], cap.T
    ia, ib = np.argsort(a1), np.argsort(b1)
    a1, b1, m = a1[ia], b1[ib], m[np.ix_(ib, ia)]
    h = float(np.diff(a1).mean())
    ae = np.concatenate([a1 - h / 2, [a1[-1] + h / 2]])
    be = np.concatenate([b1 - h / 2, [b1[-1] + h / 2]])
    ca, cb = shadow_curve(st, incl)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.3), gridspec_kw={"width_ratios": [1.0, 1.0]})
    ax = axes[0]
    ax.pcolormesh(ae, be, m.astype(float), cmap=matplotlib.colors.ListedColormap(["white", "#2b2b2b"]),
                  vmin=0, vmax=1, shading="flat", rasterized=True)
    ax.plot(ca, cb, color=C_RED, lw=1.0, label="Bardeen curve")
    zx, zy, zw, zh = 4.4, -1.6, 2.8, 3.2
    ax.add_patch(Rectangle((zx, zy), zw, zh, fill=False, ec=C_BLUE, lw=0.9))
    ax.set_xlim(-12, 12)
    ax.set_ylim(-12, 12)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$\alpha$ [$M$]")
    ax.set_ylabel(r"$\beta$ [$M$]")
    ax.set_title(r"(a) $64 \times 64$ pixels, $a_* = 0.9$, $i = 60^\circ$", fontsize=8.5)
    ax.tick_params(labelsize=7)

    ax = axes[1]
    ax.pcolormesh(ae, be, m.astype(float), cmap=matplotlib.colors.ListedColormap(["white", "#bdbdbd"]),
                  vmin=0, vmax=1, shading="flat", edgecolors="0.8", linewidth=0.3, rasterized=True)
    A, B = np.meshgrid(a1, b1)
    sel = (A > zx) & (A < zx + zw) & (B > zy) & (B < zy + zh)
    ax.plot(A[sel & m], B[sel & m], "o", color=INK, ms=2.6)
    ax.plot(A[sel & ~m], B[sel & ~m], "o", mfc="white", mec=INK, mew=0.6, ms=2.6)
    ax.plot(ca, cb, color=C_RED, lw=1.1)
    ax.set_xlim(zx, zx + zw)
    ax.set_ylim(zy, zy + zh)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$\alpha$ [$M$]")
    ax.set_ylabel(r"$\beta$ [$M$]")
    ax.set_title("(b) detail of the edge", fontsize=8.5)
    for s in ax.spines.values():
        s.set_edgecolor(C_BLUE)
    ax.tick_params(labelsize=7)
    handles = [plt.Line2D([], [], color=C_RED, lw=1.1, label="Bardeen curve"),
               plt.Line2D([], [], ls="none", marker="s", color="#2b2b2b", ms=5, label="pixel not escaped (shadow)"),
               plt.Line2D([], [], ls="none", marker="o", color=INK, ms=3, label="pixel centre, not escaped"),
               plt.Line2D([], [], ls="none", marker="o", mfc="white", mec=INK, mew=0.6, ms=3,
                          label="pixel centre, escaped")]
    fig.tight_layout(w_pad=1.5, rect=(0, 0.07, 1, 1))
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0))
    save(fig, "fig_shadow_example")


# --------------------------------------------------------------------------- b_c convergence
def convergence_bc():
    chk = summary(RUN_SCHW)["checks"]
    fig, axes = plt.subplots(2, 1, figsize=(3.4, 4.9))
    for ax, name, xlabel, tag, col in ((axes[0], "rtol_convergence", "relative tolerance rtol", "(a) RK45", C_RED),
                                       (axes[1], "rk4_convergence", r"RK4 step size $h$ [$M$]", "(b) RK4", C_BLUE)):
        c = chk[name]["computed"]
        x, e = np.array(c["levels"], float), np.array(c["errors"], float)
        used = e > c["noise_floor"]
        order = loglog_order(x[used], e[used])
        close(order, c["fitted_order"], 1e-10, f"{name} fitted order")
        coef = np.polyfit(np.log10(x[used]), np.log10(e[used]), 1)
        xf = np.logspace(np.log10(x[used].min()), np.log10(x[used].max()), 20)
        ax.loglog(x[used], e[used], "o", color=col, ms=4.5, zorder=3, label="measured")
        if (~used).any():
            ax.loglog(x[~used], e[~used], "o", mfc="white", mec=col, ms=4.5, zorder=3,
                      label="below the bisection tolerance (not fitted)")
        ax.loglog(xf, 10 ** np.polyval(coef, np.log10(xf)), "-", color=INK, lw=0.8,
                  label=f"fit, order {order:.2f}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"$|b_c - 3\sqrt{3}\,M|$ [$M$]")
        ax.set_title(tag, fontsize=8.5)
        ax.grid(True, which="major", alpha=0.25)
        ax.legend(loc="upper left", borderaxespad=0.3)
        ax.tick_params(labelsize=7)
    axes[0].set_ylim(3e-13, 3e-5)
    axes[1].set_ylim(2e-9, 2e-5)
    axes[1].set_xticks([0.1, 0.2, 0.4])
    axes[1].set_xticklabels(["0.1", "0.2", "0.4"])
    axes[1].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    fig.tight_layout(h_pad=1.0)
    save(fig, "fig_convergence_bc")


# --------------------------------------------------------------------------- Kerr validation
def kerr_validation():
    cfg = load_config(ROOT / "runs" / RUN_KERR / "config.yaml")
    params = experiment_block(cfg, "validation", KerrValidationParams)
    integ, term = integrator_options_from_config(cfg), termination_options_from_config(cfg)
    chk = summary(RUN_KERR)["checks"]
    mass = cfg.black_hole.mass

    # (a) near-extremal prograde photon, a* = 0.999, horizon margin 1e-6 M
    row = next(r for r in chk["near_extremal"]["computed"]["rows"] if r["spin"] == 0.999)
    run = next(r for r in row["runs"] if r["horizon_epsilon"] == min(params.horizon_epsilons))
    st = kerr(mass, 0.999)
    b = critical_impact_parameters(st)[0] * (1.0 + params.near_extremal_offset)
    close(b, row["b"], 1e-13, "near-extremal b")
    y0 = photon_from_constants(st, params.launch_radius * mass, 0.5 * math.pi, 1.0, b, 0.0, sign_r=-1, sign_theta=1)
    from dataclasses import replace
    tr = integrate(st, y0, integ, replace(term, horizon_epsilon=min(params.horizon_epsilons)), record=True)
    if tr.state.name != run["state"] or tr.n_steps != run["n_steps"]:
        raise SystemExit("near-extremal photon differs from the recorded run")
    close(closest_approach(tr), run["closest_approach"], 1e-12, "near-extremal closest approach")
    turns = (tr.y[:, 3] - tr.y[0, 3]) / (2.0 * math.pi)
    close(float(turns[-1]), run["turns"], 1e-10, "near-extremal turns")
    r_plus = row["r_plus"]

    # (b) conservation along the off-equatorial photon at a* = 0.999
    crow = next(r for r in chk["conservation"]["computed"]["rows"] if r["spin"] == 0.999)
    y1 = off_equatorial_ray(st, params.launch_radius * mass, params.conservation_theta_deg,
                            params.conservation_xi * mass, params.conservation_eta * mass**2)
    tc = integrate(st, y1, integ, term, record=True)
    d = drift_summary(tc)
    if tc.state.name != crow["state"] or tc.n_steps != crow["n_steps"]:
        raise SystemExit("conservation photon differs from the recorded run")
    close(d.carter, crow["max_drift"]["carter"], 1e-9, "Carter drift")
    close(d.null, crow["max_drift"]["null"], 1e-9, "null error")
    if d.energy != 0.0 or d.lz != 0.0:
        raise SystemExit("energy or L_z drift is not zero")
    diag = tc.diagnostics

    fig, axes = plt.subplots(2, 1, figsize=(3.4, 5.2))
    ax = axes[0]
    ax.semilogy(turns, tr.y[:, 1] - r_plus, color=C_BLUE, lw=1.0, label="traced photon")
    ax.axhline(row["r_turn_analytic"] - r_plus, color=C_RED, lw=0.8, ls="--",
               label="analytic turning point")
    ax.axhline(row["r_ph_prograde"] - r_plus, color="0.45", lw=0.7, ls=":",
               label="prograde circular photon orbit")
    ax.set_xlabel(r"turns around the hole, $(\phi - \phi_0)/2\pi$")
    ax.set_ylabel(r"$r - r_+$ [$M$]")
    ax.set_title(r"(a) $a_* = 0.999$, $b = b_c\,(1 + 10^{-3})$", fontsize=8.5)
    ax.set_ylim(3e-3, 3e3)
    ax.grid(True, which="major", alpha=0.25)
    ax.legend(loc="upper center", borderaxespad=0.3)
    ax.tick_params(labelsize=7)

    ax = axes[1]
    floor = 1e-18
    ax.semilogy(tc.lam, np.maximum(np.abs(diag.carter_drift), floor), color=C_GREEN, lw=1.0,
                label=r"Carter constant $Q$, relative")
    ax.semilogy(tc.lam, np.maximum(np.abs(diag.null_error), floor), color=C_PURPLE, lw=1.0,
                label=r"null constraint $|H|/E^2$")
    ax.set_xlabel(r"affine parameter $\lambda$ [$M$]")
    ax.set_ylabel("relative drift, constraint error")
    ax.set_title(r"(b) off-equatorial photon, $a_* = 0.999$", fontsize=8.5)
    ax.set_ylim(1e-17, 1e-7)
    ax.grid(True, which="major", alpha=0.25)
    ax.legend(loc="lower right", borderaxespad=0.3)
    ax.tick_params(labelsize=7)
    fig.tight_layout(h_pad=1.0)
    save(fig, "fig_kerr_validation")


# --------------------------------------------------------------------------- frame dragging
def frame_dragging():
    cfg = load_config(ROOT / "runs" / RUN_DRAG / "config.yaml")
    rec = summary(RUN_DRAG)
    integ, term = integrator_options_from_config(cfg), termination_options_from_config(cfg)
    par = cfg.experiment.parameters["frame_dragging"]
    mass = cfg.black_hole.mass
    res = frame_dragging_experiment(par["spins"], par["impact_parameter"] * mass, integ=integ, term=term,
                                    r0=par["launch_radius"] * mass, mass=mass, polar_b=None)
    close(res.polar_impact_parameter, rec["polar_impact_parameter"], 1e-14, "polar impact parameter")
    for ray in (*res.equatorial, *res.polar):
        r = next(x for x in rec["rays"] if x["family"] == ray.family and x["spin"] == ray.spin)
        if ray.state != r["state"] or ray.trajectory.n_steps != r["n_steps"]:
            raise SystemExit(f"{ray.family} ray a = {ray.spin} differs from the recorded run")
        close(ray.delta_phi, r["delta_phi"], 1e-10, f"{ray.family} delta phi, a = {ray.spin}")

    cols = {-0.9: C_BLUE, 0.0: C_RED, 0.9: C_GREEN}
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1), gridspec_kw={"width_ratios": [1.0, 1.25]})
    ax = axes[0]
    for ray in res.equatorial:
        x, y, _ = to_cartesian(ray.trajectory)
        fate = "captured" if ray.state == "CAPTURED" else "escapes"
        ax.plot(x, y, color=cols[ray.spin], lw=1.0, label=f"$a_* = {spin_text(ray.spin)}$, {fate}")
    for spin in (0.0, 0.9):
        st = kerr(mass, spin)
        ax.add_patch(Circle((0, 0), math.hypot(horizon_radii(st)[0], st.a), fill=False, color=INK, lw=0.6))
    ax.set_xlim(-16, 16)
    ax.set_ylim(-16, 16)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x$ [$M$]")
    ax.set_ylabel(r"$y$ [$M$]")
    ax.set_title(r"(a) equatorial photons, $L_z = +6\,ME$", fontsize=8.5)
    ax.legend(loc="upper left", borderaxespad=0.3)
    ax.tick_params(labelsize=7)

    ax = axes[1]
    for ray in res.polar:
        _, y, zz = to_cartesian(ray.trajectory)
        ax.plot(zz, y, color=cols[ray.spin], lw=1.0, label=f"$a_* = {spin_text(ray.spin)}$")
    ax.axhline(0.0, color="0.6", lw=0.4)
    ax.set_xlabel(r"$z$ [$M$] (along the spin axis)")
    ax.set_ylabel(r"$y$ [$M$]")
    ax.set_title(rf"(b) $L_z = 0$ photons, {res.polar_impact_parameter:.1f}$\,M$ from the axis", fontsize=8.5)
    ax.legend(loc="upper left", borderaxespad=0.3)
    ax.grid(True, which="major", alpha=0.25)
    ax.tick_params(labelsize=7)
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig_frame_dragging_paper")


if __name__ == "__main__":
    trajectories()
    shadow_example()
    convergence_bc()
    kerr_validation()
    frame_dragging()
