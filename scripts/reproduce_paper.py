"""Reproduce every result in research.md, in order, with logs.

Runs the studies, validations and benchmarks behind each table and figure of
the paper and writes one log per step to paper/repro/. Steps can be selected
with --steps. Timings depend on the machine and its load; numerical results
are deterministic for a given software environment (paper/requirements-lock.txt).

Usage:
    python scripts/reproduce_paper.py                  # everything (about 1 h 45 min measured on 8 logical cores)
    python scripts/reproduce_paper.py --steps winding,oracle
    python scripts/reproduce_paper.py --list

Environment used for the paper: Python 3.13.13 and the versions pinned in
paper/requirements-lock.txt (install with `pip install -r paper/requirements-lock.txt`
then `pip install -e .`).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "paper" / "repro"
PY = sys.executable
KERRRAY = [PY, "-m", "kerrray"]

STEPS: dict[str, tuple[str, list[list[str]]]] = {
    "validate_schwarzschild": ("Table 1, Figure 6: Schwarzschild photon sphere, critical impact parameter and its convergence", [KERRRAY + ["validate", "schwarzschild"]]),
    "validate_kerr": ("Section 4.2, Figure 7: Kerr validation", [KERRRAY + ["validate", "kerr"]]),
    "frame_dragging": ("Section 4.2, Figure 8: frame dragging of equatorial and zero-angular-momentum photons", [KERRRAY + ["experiment", "--config", "configs/kerr.yaml", "--name", "frame_dragging"]]),
    "winding": ("Table 2, Figure 9: near-critical winding", [[PY, "scripts/study_winding.py"]]),
    "convergence": ("Table 3, Figures 10, 11: mask edge vs resolution and tolerance", [[PY, "scripts/study_shadow_convergence.py"]]),
    "oracle": ("Sections 5.2, 5.5, 5.7, Tables A1, A2, Figures 13, 14: oracle comparison, inclination and precision checks", [[PY, "scripts/study_oracle.py"]]),
    "bisection": ("Table 4, Figure 12: bisection edge error vs tolerance", [[PY, "scripts/study_edge_bisection.py"]]),
    "finite_distance": ("Table 5: finite observer distance", [[PY, "scripts/study_finite_distance.py"]]),
    "failures": ("Table 6: failure mechanisms at rtol 1e-4 (true fates by re-tracing)",
                 [[PY, "scripts/audit/audit_axis_failures.py", "512", str(a)] for a in (0.0, 0.5, 0.9, 0.99)]),
    "bisection_failures": ("Section 5.3: mechanism of the bisection failures at rtol 1e-4", [[PY, "scripts/audit/audit_bisection_failures.py"]]),
    "max_step": ("Tables 7, A3 and Section 5.3: failures with a maximum step size", [[PY, "scripts/audit/audit_max_step.py"]]),
    "rk4": ("Section 5.6: RK4 capture, false escapes and 64x64 masks", [[PY, "scripts/audit/audit_classification_rk4.py"], [PY, "scripts/audit/audit_rk4_history.py"]]),
    "winding_quadrature": ("Section 4.3: winding by exact quadrature", [[PY, "scripts/audit/audit_winding.py"]]),
    "mechanism": ("Section 5.5: step history of one step-through-the-hole photon and |L_z| of the failed photons",
                  [[PY, "scripts/audit/audit_axis_mechanism.py"], [PY, "scripts/audit/audit_failure_lz.py"]]),
    "winding_overshoot": ("Section 4.3: overshoot of the last step beyond the 60 M escape radius", [[PY, "scripts/audit/audit_winding_overshoot.py"]]),
    "solver": ("Table 8, Figure 16: solver comparison", [KERRRAY + ["benchmark", "solver"]]),
    "precision": ("Section 5.7, Figure 17: float32 vs float64", [KERRRAY + ["benchmark", "precision"]]),
    "cpu": ("Table 9: CPU performance", [KERRRAY + ["benchmark", "cpu"]]),
    "shadow_example": ("Figure 4: the 64x64 example shadow (a new run; the figure itself re-plots run 20260929T054034Z-de5a37)",
                       [KERRRAY + ["shadow"]]),
    "figures": ("Figures 12, 16, 17 from the stored data", [[PY, "scripts/make_paper_figures.py"]]),
    "diagrams": ("Figures 1, 3, 5 (schematics) and 15 (failure mechanisms, recomputed and checked against the audit)",
                 [[PY, "scripts/make_paper_diagrams.py"]]),
    "result_figures": ("Figures 2, 4, 6, 7, 8 from recorded runs (repeated integrations checked against the records)",
                       [[PY, "scripts/make_paper_result_figures.py"]]),
    "trace": ("Evidence for every number in the paper (docs/publication/FINAL_CLAIM_AUDIT.md)", [[PY, "scripts/audit/trace_paper_numbers.py"]]),
    "build": ("Paper HTML and PDF preview (paper/build/)", [[PY, "scripts/build_paper.py"]]),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default=",".join(STEPS))
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.list:
        for k, (desc, _) in STEPS.items():
            print(f"{k:24s} {desc}")
        return
    LOGS.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    failures = []
    for name in args.steps.split(","):
        desc, cmds = STEPS[name]
        t0 = time.perf_counter()
        print(f"== {name}: {desc}", flush=True)
        with (LOGS / f"{name}.log").open("w", encoding="utf-8") as log:
            for cmd in cmds:
                code = subprocess.call(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)
                if code != 0:
                    failures.append((name, code))
        print(f"   done in {time.perf_counter() - t0:.0f} s -> paper/repro/{name}.log", flush=True)
    if failures:
        print("FAILED:", failures)
        sys.exit(1)


if __name__ == "__main__":
    main()
