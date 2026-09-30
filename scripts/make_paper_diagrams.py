"""Methodology diagrams and the failure-mechanism figure for research.md.

Writes to paper/figures/ (PDF for LaTeX, PNG for research.md):

- diagram_pipeline:   the computational pipeline (schematic; no data).
- diagram_geometry:   observer, image plane and backward-traced photons
                      (schematic, not to scale; no data).
- diagram_edge:       the three ways of measuring the edge error
                      (schematic; no data).
- failure_mechanisms: the two failure mechanisms at rtol = 1e-4, recomputed
                      with the configuration of scripts/audit/audit_axis_mechanism.py
                      (a* = 0.5, i = 60 deg, 256^2 image, alpha ~ 0 column).
                      The script checks that the last four radii of the
                      step-through photon reproduce the audited values
                      (805, 605, 406, -412 M) and stops otherwise.

Usage: ./.venv/Scripts/python.exe scripts/make_paper_diagrams.py
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures"

plt.rcParams.update({
    "font.size": 8,
    "mathtext.fontset": "cm",
    "axes.linewidth": 0.7,
    "lines.linewidth": 1.0,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
})
INK = "#1a1a1a"
ACCENT = "#c0392b"   # red-orange, as in the existing figures
BLUE = "#1f5fa8"
GREY = "#d9d9d9"


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.02,
                    metadata={"CreationDate": None} if ext == "pdf" else None)
    plt.close(fig)
    print("wrote", OUT / f"{name}.pdf", "and .png")


def box(ax, x, y, w, h, title, body, fc="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.012",
                                fc=fc, ec=INK, lw=0.7))
    ax.text(x + w / 2, y + h - 0.035, title, ha="center", va="top", fontsize=7.0, fontweight="bold")
    ax.text(x + w / 2, y + h - 0.115, body, ha="center", va="top", fontsize=6.1, linespacing=1.3)


def arrow(ax, p, q):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=8, lw=0.8, color=INK))


# --------------------------------------------------------------------------- pipeline
def pipeline():
    """Snake layout: 1 -> 2 -> 3 -> 4 on the top row, 5 (below 4) -> 6 -> 7 -> 8 leftwards."""
    fig, ax = plt.subplots(figsize=(7.0, 2.35))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    w, h = 0.215, 0.38
    xs = [0.01, 0.265, 0.52, 0.775]
    yt, yb = 0.55, 0.04
    top = [
        ("1  Observer", "zero-angular-momentum\nobserver at $r_o$, $\\theta_o$;\none photon per pixel\n$(\\alpha, \\beta)$"),
        ("2  Initial data", "tetrad maps the pixel\ndirection to $p_\\mu$;\nnull condition checked;\ntraced backwards"),
        ("3  Integration", "Hamiltonian equations;\nDormand-Prince 5(4);\natol $= 10^{-2}$ rtol;\nNumPy or Numba"),
        ("4  Tests, each step", "numerical failure;\nout of domain ($r<0$, axis);\ncaptured, $r \\leq r_+ + 10^{-6}M$;\nescaped, $r \\geq r_{\\rm esc}$"),
    ]
    bottom = [  # drawn right to left: 5 under 4
        ("5  Termination state", "numerical failure,\nout of domain, captured,\nescaped, or affine\nbudget exhausted"),
        ("6  Shadow mask", "every pixel whose\nphoton did not escape"),
        ("7  Edge measurement", "mask edge (720 rays);\noracle mask;\nbisection (16 directions)"),
        ("8  Comparison", "Bardeen curve with the\nfinite-distance factor $k$:\nedge error and\ndisagreeing pixels"),
    ]
    for x, (t, b) in zip(xs, top):
        box(ax, x, yt, w, h, t, b)
    for x, (t, b) in zip(reversed(xs), bottom):
        box(ax, x, yb, w, h, t, b, fc="#f2f2f2")
    for i in range(3):
        arrow(ax, (xs[i] + w, yt + h / 2), (xs[i + 1], yt + h / 2))
        arrow(ax, (xs[3 - i], yb + h / 2), (xs[2 - i] + w, yb + h / 2))
    arrow(ax, (xs[3] + w / 2, yt), (xs[3] + w / 2, yb + h))
    ax.add_patch(FancyArrowPatch((xs[3] + 0.25 * w, yt + h), (xs[2] + 0.75 * w, yt + h),
                                 connectionstyle="arc3,rad=0.55", arrowstyle="-|>", mutation_scale=7,
                                 lw=0.7, color="#555555"))
    ax.text(xs[3] - 0.018, yt + h + 0.075, "running: next step", ha="center", fontsize=6.0,
            style="italic", color="#555555")
    ax.text(xs[3] + w / 2 + 0.01, 0.485, "terminated", ha="left", va="center", fontsize=6.0,
            style="italic", color="#555555")
    save(fig, "diagram_pipeline")


# --------------------------------------------------------------------------- geometry
def bezier(p0, p1, p2, p3, n=120):
    t = np.linspace(0, 1, n)[:, None]
    return ((1 - t) ** 3) * p0 + 3 * ((1 - t) ** 2) * t * p1 + 3 * (1 - t) * (t ** 2) * p2 + (t ** 3) * p3


def geometry():
    fig, ax = plt.subplots(figsize=(3.4, 2.75))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-3.0, 8.4)
    ax.set_ylim(-2.6, 5.1)
    ax.annotate("", xy=(0, 4.8), xytext=(0, -2.5), arrowprops=dict(arrowstyle="-|>", lw=0.8, color=INK))
    ax.text(0.15, 4.75, "spin axis", fontsize=6.8, va="top")
    ax.add_patch(Circle((0, 0), 0.55, color="black"))
    ax.annotate("horizon", xy=(-0.4, -0.38), xytext=(-2.7, -1.6), fontsize=6.5,
                arrowprops=dict(arrowstyle="-", lw=0.5, color=INK))
    th = math.radians(60.0)
    d = np.array([math.sin(th), math.cos(th)])
    obs = 6.1 * d
    ax.plot([0, obs[0]], [0, obs[1]], ls="--", lw=0.6, color="#777777")
    arc = np.linspace(0, th, 40)
    ax.plot(1.5 * np.sin(arc), 1.5 * np.cos(arc), lw=0.7, color=INK)
    ax.text(0.45, 1.75, "$\\theta_o$", fontsize=8)
    ax.text(2.7, 1.25, "$r_o$ (not to scale)", fontsize=6.3, rotation=math.degrees(math.atan2(d[1], d[0])),
            rotation_mode="anchor", color="#555555")
    perp = np.array([-d[1], d[0]])     # in the meridional plane, perpendicular to the line of sight
    p0, p1 = obs - 1.25 * perp, obs + 1.25 * perp
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], lw=2.4, color=BLUE, solid_capstyle="butt")
    ax.annotate("", xy=obs + 1.65 * perp, xytext=obs + 1.25 * perp,
                arrowprops=dict(arrowstyle="-|>", lw=0.8, color=BLUE))
    ax.text(*(obs + 1.8 * perp), "$\\beta$", fontsize=8, color=BLUE, ha="center", va="bottom")
    ax.text(obs[0] + 0.85, obs[1] + 0.2, "image plane\n$\\beta$: projected\nspin axis\n$\\alpha$: perpendicular\nto the page",
            fontsize=6.0, color=BLUE, va="top")
    # captured photon: from a pixel to the horizon
    a = obs + 0.55 * perp
    cap = bezier(a, a - 2.2 * d + np.array([0.0, 0.2]), np.array([1.2, 1.6]), np.array([0.3, 0.45]))
    ax.plot(cap[:, 0], cap[:, 1], color=INK, lw=0.9)
    ax.text(2.35, 3.5, "captured", fontsize=6.5)
    # escaped photon: bends around the hole and leaves
    b = obs - 0.65 * perp
    esc = bezier(b, np.array([1.2, -1.6]), np.array([-1.9, -1.7]), np.array([-2.7, 1.3]))
    ax.plot(esc[:, 0], esc[:, 1], color=ACCENT, lw=0.9)
    ax.annotate("", xy=esc[-1], xytext=esc[-4], arrowprops=dict(arrowstyle="-|>", lw=0.8, color=ACCENT))
    ax.text(-2.85, 1.55, "escaped", fontsize=6.5, color=ACCENT)
    save(fig, "diagram_geometry")


# --------------------------------------------------------------------------- edge measures
def edge_measures():
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.45))
    n = 10
    cx, cy, rad = 4.6, 4.9, 3.35   # schematic analytic curve (a circle)
    ang = np.linspace(0, 2 * np.pi, 400)
    centres = [(i + 0.5, j + 0.5) for i in range(n) for j in range(n)]
    inside = lambda x, y: (x - cx) ** 2 + (y - cy) ** 2 < rad ** 2  # noqa: E731
    titles = ["(a) Mask edge", "(b) Oracle mask", "(c) Bisection edge"]
    for ax, title in zip(axes, titles):
        ax.set_aspect("equal")
        ax.set_xlim(0, n)
        ax.set_ylim(0, n)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=8, loc="left")
    # (a) traced mask: filled pixels = non-escaped; analytic curve; ray from centroid; 0.5 crossing
    ax = axes[0]
    for (x, y) in centres:
        ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, fc="#3a3a3a" if inside(x, y) else "white", ec="#bbbbbb", lw=0.4))
    ax.plot(cx + rad * np.cos(ang), cy + rad * np.sin(ang), color=ACCENT, lw=1.0)
    phi = math.radians(28)
    ax.plot([cx, cx + 4.6 * math.cos(phi)], [cy, cy + 4.6 * math.sin(phi)], color=BLUE, lw=0.9)
    ax.plot(cx, cy, marker="+", color=BLUE, ms=6)
    ax.plot(cx + 3.05 * math.cos(phi), cy + 3.05 * math.sin(phi), marker="o", mfc="white", mec=BLUE, ms=4)
    ax.text(0.2, -1.05, "0.5 crossing of the bilinear mask\nalong 720 rays from the centroid", fontsize=6.3, va="top")
    # (b) oracle: pixels whose centres lie inside the (corrected) analytic curve
    ax = axes[1]
    for (x, y) in centres:
        ins = inside(x, y)
        ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, fc="#9a9a9a" if ins else "white", ec="#bbbbbb", lw=0.4))
        ax.plot(x, y, marker=".", ms=2.2, color="black" if ins else "#999999")
    ax.plot(cx + rad * np.cos(ang), cy + rad * np.sin(ang), color=ACCENT, lw=1.0)
    ax.text(0.2, -1.05, "pixel in the oracle shadow when its\ncentre lies inside the analytic curve", fontsize=6.3, va="top")
    # (c) bisection along one direction: bracket between escaping and non-escaping photons
    ax = axes[2]
    ax.plot(cx + rad * np.cos(ang), cy + rad * np.sin(ang), color=ACCENT, lw=1.0)
    phi = math.radians(28)
    u = np.array([math.cos(phi), math.sin(phi)])
    c = np.array([cx, cy])
    ax.plot(*np.column_stack([c, c + 4.6 * u]), color=BLUE, lw=0.9)
    ax.plot(cx, cy, marker="+", color=BLUE, ms=6)
    lo, hi = 0.25 * rad, 1.3 * rad
    normal = np.array([-u[1], u[0]])
    for k in range(3):
        mid = 0.5 * (lo + hi)
        esc = mid > rad
        p = c + mid * u
        ax.plot(*p, marker="o", ms=3.6, mfc="white" if esc else "black", mec="black", mew=0.6, zorder=3)
        side = 1.0 if k < 2 else -1.0
        ax.text(*(p + side * 0.55 * normal), str(k + 1), fontsize=6.0, ha="center", va="center")
        lo, hi = (lo, mid) if esc else (mid, hi)
    ax.plot(*np.column_stack([c + lo * u, c + hi * u]), color=INK, lw=2.4, solid_capstyle="butt", zorder=2)
    ax.text(*(c + hi * u + np.array([0.15, -0.55])), "bracket", fontsize=6.0)
    ax.text(0.2, -1.05, "bisect the radius between photons that\nescape (open) and do not (filled): 34 steps", fontsize=6.3, va="top")
    fig.subplots_adjust(wspace=0.12, bottom=0.2)
    save(fig, "diagram_edge")


# --------------------------------------------------------------------------- failure mechanisms
def failure_mechanisms():
    from kerrray.geodesics.integrators import IntegratorOptions, TerminationOptions, integrate
    from kerrray.geodesics.integrators_batch import integrate_batch
    from kerrray.geometry.horizons import outer_horizon
    from kerrray.geometry.metric import Spacetime
    from kerrray.photons.classification import TerminationState as TS
    from kerrray.raytracing.camera import Camera, initial_states

    warnings.simplefilter("ignore")
    np.seterr(all="ignore")
    st = Spacetime(1.0, 0.5)
    cam = Camera(radius=1000.0, inclination_deg=60.0, fov=8.0, resolution=256)
    Y0 = initial_states(cam, st)
    al = cam.pixel_coordinates()[0].ravel()
    cols = np.flatnonzero(np.abs(al) < 0.05)
    term = TerminationOptions(1e-6, 1000.0)
    loose = IntegratorOptions(method="rk45", rtol=1e-4, atol=1e-6)
    res = integrate_batch(st, Y0[cols], loose, term)
    ood = cols[res.state == TS.OUT_OF_DOMAIN]
    Yend = res.Y[res.state == TS.OUT_OF_DOMAIN]
    through = ood[Yend[:, 1] < 0]
    axis_n = ood[(Yend[:, 1] >= 0) & (Yend[:, 2] < 0)]          # crossed the north pole (theta < 0)
    axis_s = ood[(Yend[:, 1] >= 0) & (Yend[:, 2] > math.pi)]    # crossed the south pole (theta > pi)
    print(f"alpha ~ 0 column: {len(ood)} out of domain; {len(through)} with r < 0, "
          f"{len(axis_n)} across the north pole, {len(axis_s)} across the south pole")
    axis = axis_n if len(axis_n) else axis_s
    i_hole = ood[len(ood) // 2]          # the photon of audit_axis_mechanism.py
    assert i_hole in through, "the audited photon no longer steps through the hole"
    i_axis = axis[len(axis) // 2]
    rp = outer_horizon(st)

    def path(i, rtol):
        tr = integrate(st, Y0[i], IntegratorOptions(method="rk45", rtol=rtol, atol=rtol * 1e-2, max_steps=400_000,
                                                     lambda_max=4000.0), term, record=True)
        return np.asarray(tr.lam), np.asarray(tr.y), tr.state

    lam1, y1, s1 = path(i_hole, 1e-4)
    last = np.round(y1[-4:, 1]).astype(int).tolist()
    assert last == [805, 605, 406, -412], f"step history differs from the audit: {last}"
    lam1b, y1b, s1b = path(i_hole, 1e-6)
    lam2, y2, s2 = path(i_axis, 1e-4)
    lam2b, y2b, s2b = path(i_axis, 1e-10)
    print(f"step-through photon: {len(lam1) - 1} steps, {s1.name}; rtol 1e-6: {len(lam1b) - 1} steps, {s1b.name}")
    print(f"axis-crossing photon {i_axis}: rtol 1e-4 {s2.name} (final theta {y2[-1, 2]:+.3e}); rtol 1e-10 {s2b.name}")

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(3.4, 4.9))
    a1.plot(lam1b, y1b[:, 1], color="#888888", lw=0.9, label=f"rtol $10^{{-6}}$: {s1b.name.lower()}")
    a1.plot(lam1, y1[:, 1], "o-", color=ACCENT, ms=3, lw=0.9, label=f"rtol $10^{{-4}}$: {s1.name.lower().replace('_', ' ')}")
    a1.axhline(rp, color=INK, lw=0.6, ls="--")
    a1.axhline(0.0, color=INK, lw=0.4)
    a1.text(40, rp + 25, f"$r_+ = {rp:.2f}\\,M$", fontsize=6.5)
    a1.annotate("one step of 818 M", xy=(0.5 * (lam1[-2] + lam1[-1]), 0.5 * (y1[-2, 1] + y1[-1, 1])),
                xytext=(250, -300), fontsize=6.5, arrowprops=dict(arrowstyle="-", lw=0.5))
    a1.set_xlabel("affine parameter $\\lambda$ [M]")
    a1.set_ylabel("$r$ [M]")
    a1.set_title("(a) Step through the hole", fontsize=8, loc="left")
    a1.legend(fontsize=6.3, frameon=False, loc="upper right")
    a2.plot(lam2b, np.degrees(y2b[:, 2]), color="#888888", lw=0.9, label=f"rtol $10^{{-10}}$: {s2b.name.lower()}")
    a2.plot(lam2, np.degrees(y2[:, 2]), "o-", color=ACCENT, ms=3, lw=0.9,
            label=f"rtol $10^{{-4}}$: {s2.name.lower().replace('_', ' ')}")
    north = y2[-1, 2] < 0
    pole = 0.0 if north else 180.0
    a2.axhline(pole, color=INK, lw=0.6, ls="--")
    a2.text(0.02, 0.06 if north else 0.9, f"spin axis, $\\theta = {pole:.0f}^\\circ$",
            transform=a2.transAxes, fontsize=6.5)
    lo = max(0.0, lam2[-1] - 250.0)
    a2.set_xlim(lo, lam2[-1] + 60.0)
    th_win = np.concatenate([np.degrees(y2[lam2 >= lo, 2]), np.degrees(y2b[lam2b >= lo, 2])])
    a2.set_ylim(th_win.min() - 3, th_win.max() + 3)
    a2.set_xlabel("affine parameter $\\lambda$ [M]")
    a2.set_ylabel("$\\theta$ [deg]")
    a2.set_title("(b) Step across the spin axis", fontsize=8, loc="left")
    a2.legend(fontsize=6.3, frameon=False, loc="upper left")
    fig.tight_layout(h_pad=1.0)
    save(fig, "failure_mechanisms")
    return {"i_hole": int(i_hole), "i_axis": int(i_axis), "hole": (len(lam1) - 1, s1.name, len(lam1b) - 1, s1b.name),
            "axis": (len(lam2) - 1, s2.name, float(y2[-1, 2]), len(lam2b) - 1, s2b.name)}


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    pipeline()
    geometry()
    edge_measures()
    info = failure_mechanisms()
    print(info)
