"""Regenerate the solver-comparison and precision figures of research.md from raw data.

Reads the stored summaries of the solver benchmark and the precision study
(paths below; both are written by `kerrray benchmark solver` and
`kerrray benchmark precision`) and writes clean, labelled figures to
paper/figures/. No computation is repeated here.

Usage:
    ./.venv/Scripts/python.exe scripts/make_paper_figures.py [--solver RUN_ID] [--precision RUN_ID]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "paper" / "figures"
DEFAULT_SOLVER = "20260929T051346Z-f444b5"
DEFAULT_PRECISION = "20260929T060841Z-f365e3"
COLORS = {"rk4": "#0072BD", "rk45": "#D95319", "dop853": "#77AC30"}
NAMES = {"rk4": "RK4 (fixed step h)", "rk45": "RK45 (adaptive, rtol)", "dop853": "DOP853 (adaptive, rtol)"}


def load(run_id: str) -> dict:
    return json.loads((ROOT / "reports" / run_id / "summary.json").read_text(encoding="utf-8"))["results"]


def solver_figure(run_id: str) -> None:
    r = load(run_id)
    fig, ax = plt.subplots(figsize=(7.0, 5.6))
    for solver in ("rk4", "rk45", "dop853"):
        rows = sorted((c for c in r["configurations"] if c["solver"] == solver), key=lambda c: c["runtime_s"])
        t = [c["runtime_s"] for c in rows]
        ax.loglog(t, [c["median_trajectory_error"] for c in rows], "o-", color=COLORS[solver], label=f"{NAMES[solver]}, median")
        ax.loglog(t, [c["max_trajectory_error"] for c in rows], "s:", color=COLORS[solver], alpha=0.45, ms=4, label=f"{NAMES[solver]}, max")
        for c in rows:
            tag = f"h={c['setting']:g}" if solver == "rk4" else f"{c['setting']:.0e}"
            ax.annotate(tag, (c["runtime_s"], c["median_trajectory_error"]), textcoords="offset points", xytext=(5, -10), fontsize=7, color=COLORS[solver])
        if solver == "rk4":
            fr = rows[0]["failure_rate"]
            ax.annotate(f"RK4: {fr:.0%} of photons end out of domain\n(every captured photon)", (rows[-1]["runtime_s"], rows[-1]["median_trajectory_error"]),
                        textcoords="offset points", xytext=(-150, 18), fontsize=7.5, color=COLORS["rk4"])
    ax.set(xlabel="run time for the 40-photon set [s] (best of 3)", ylabel="trajectory error vs DOP853 at rtol 1e-13 [M]",
           title="Solver comparison (a* = 0 and 0.9, launch at 50 M)")
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=6))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGS / "solver_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def precision_figure(run_id: str) -> None:
    r = load(run_id)
    runs = [x for x in r["runs"] if x["backend"] == "numba"] or r["runs"]
    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    for dtype, role, color, label in (("float32", "main", "#D95319", "float32, margin 1e-3 M"),
                                      ("float64", "control", "#0072BD", "float64, margin 1e-3 M"),
                                      ("float64", "main", "#77AC30", "float64, margin 1e-6 M")):
        sub = sorted((x for x in runs if x["dtype"] == dtype and x["role"] == role), key=lambda x: x["rtol"])
        ls = "--" if (dtype, role) == ("float64", "main") else "-"
        ax.loglog([x["rtol"] for x in sub], [x["max_null_error_escaped"] for x in sub], "o", linestyle=ls, color=color, label=label)
    caps = {(x["dtype"], x["role"], x["rtol"]): (x["captured"], x["escaped"]) for x in runs}
    same = len(set(caps.values())) == 1
    ax.set(xlabel="relative tolerance rtol", ylabel="max null-constraint error, escaping photons",
           title="Single vs double precision (64², a* = 0.9, i = 60°)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, title=("identical capture/escape counts in every run" if same else "capture/escape counts differ"), title_fontsize=7.5)
    fig.tight_layout()
    fig.savefig(FIGS / "precision.png", dpi=150)
    plt.close(fig)


def error_budget_figure() -> None:
    """Two regimes of shadow-edge error: mask-based (pixel-limited) and bisection (tolerance-limited)."""
    bis = json.loads((ROOT / "paper" / "data" / "edge_bisection.json").read_text(encoding="utf-8"))["rows"]
    conv = json.loads((ROOT / "paper" / "data" / "shadow_convergence.json").read_text(encoding="utf-8"))["rows"]
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    colors = {0.0: "#0072BD", 0.9: "#D95319", 0.99: "#7E2F8E"}
    for spin, c in colors.items():
        sub = sorted((r for r in bis if r["spin"] == spin), key=lambda r: r["rtol"])
        ax.loglog([r["rtol"] for r in sub], [r["rms_error_M"] for r in sub], "o-", color=c, label=f"bisection edge, a* = {spin}")
    x = [1e-12, 1e-4]
    ax.loglog(x, [2 * v for v in x], color="0.55", lw=1, ls="--", label="2 x rtol (guide)")
    for n, ls in ((64, ":"), (128, ":"), (256, ":"), (512, ":")):
        vals = [r["rms_M"] for r in conv if r["resolution"] == n and r["rtol"] <= 1e-5]
        v = sum(vals) / len(vals)
        ax.axhline(v, color="0.35", lw=0.8, ls=ls)
        ax.text(1.3e-12, v * 1.12, f"mask edge, {n}x{n} pixels", fontsize=7, color="0.3")
    ax.axvspan(3e-5, 2e-4, color="#A2142F", alpha=0.08)
    ax.text(3.2e-5, 3e-8, "photons start" + chr(10) + "to fail", fontsize=7, color="#A2142F")
    ax.axhline(1.6e-8, color="0.75", lw=0.8)
    ax.text(1.3e-12, 1.9e-8, "reference-curve resolution", fontsize=7, color="0.55")
    ax.set(xlabel="relative tolerance rtol", ylabel="rms edge error vs analytic curve [M]",
           title="Shadow-edge error budget (i = 60 deg, 16 directions)")
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(fontsize=7.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGS / "error_budget.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solver", default=DEFAULT_SOLVER)
    ap.add_argument("--precision", default=DEFAULT_PRECISION)
    args = ap.parse_args()
    FIGS.mkdir(parents=True, exist_ok=True)
    solver_figure(args.solver)
    precision_figure(args.precision)
    error_budget_figure()
    print("wrote", FIGS / "solver_comparison.png", "and", FIGS / "precision.png")


if __name__ == "__main__":
    main()
