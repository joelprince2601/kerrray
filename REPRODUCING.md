# Reproducing the results of research.md

Every table, figure and number of the manuscript is produced by a script in
this repository. This page gives the commands, what each produces, and the
run times measured on the machine used for the paper.

> **This public repository contains the software, configurations, tests and
> scripts only.** The manuscript (`research.md`, `paper/latex/`) and the
> stored results of the published runs (`paper/data/`, `paper/repro/`,
> `paper/figures/`, `runs/`, `reports/`) are not included. Running the steps
> below regenerates the results; the steps that re-plot or compare with the
> stored records (`figures`, `result_figures`, `trace`, `build`, and
> `scripts/audit/compare_reproduction.py`) need those records and do not run
> without them. References below to these paths describe the author's full
> project tree.

## 1. Environment

The published results were produced with Python 3.13.13 on Windows 11
(Intel64 Family 6 Model 142, 8 logical CPUs). Other platforms were not
tested.

```bash
python -m venv .venv
.venv/Scripts/activate                 # macOS/Linux: source .venv/bin/activate
pip install numpy==2.5.3 scipy==1.18.1 sympy==1.14.0 numba==0.67.0 llvmlite==0.49.0 matplotlib==3.11.2 typer==0.27.2 rich==15.0.0 PyYAML==6.0.3 pytest==9.1.1
pip install -e .[fast]
python -m pytest                       # expected: 919 passed, 1 skipped (PROJECT.md is not published)
```

These are the versions of the direct dependencies used for the published
results (the author's full lock file also pins their dependencies). The `fast` extra installs Numba, which the studies use;
without it the NumPy backend gives the same classifications but is 10 to 22
times slower. The test suite took 11 minutes on the machine above.

## 2. Everything at once

```bash
python scripts/reproduce_paper.py --list
python scripts/reproduce_paper.py
python scripts/audit/compare_reproduction.py
python scripts/build_paper.py
```

- `--list` prints every step and the paper item it produces.
- The full run writes one log per step to `paper/repro/<step>.log`,
  rewrites `paper/data/*.json` and `paper/figures/*.png`, and adds run
  folders to `runs/` and `reports/`. It stops with a non-zero exit status
  and prints `FAILED` if any step fails.
- `compare_reproduction.py` compares the regenerated `paper/data/*.json`
  with the committed versions (default revision `7788ef7`), ignoring run
  times and provenance fields. Expected output: `IDENTICAL` for every file.
- `build_paper.py` writes `paper/build/research.html` and, if Chrome or Edge
  is installed, `paper/build/research.pdf`.

Individual steps run with `--steps`, for example
`python scripts/reproduce_paper.py --steps winding,oracle`.

## 3. Steps and measured run times

Times are wall-clock seconds from the reproduction logs of 2026-09-29, on a
laptop that was at times running other jobs. They are indicative only; the
first step that uses Numba also compiles it.

| Step | Paper item | Measured time |
|---|---|---|
| `validate_schwarzschild` | Table 1, §4.1, Figure 6 data | 109 s |
| `validate_kerr` | §4.2, Figure 7 data | 87 s |
| `frame_dragging` | §4.2 frame dragging, Figure 8 data | 33 s |
| `winding` | Table 2, Figure 9 | 94 s |
| `convergence` | Table 3, Figures 10, 11 | 891 s |
| `oracle` | §5.2, §5.7, Tables A1, A2, Figures 13, 14 | 1,081 s |
| `bisection` | Table 4, Figure 12 | 10 s |
| `bisection_failures` | §5.3 | 4 s |
| `finite_distance` | Table 5 | 239 s |
| `failures` | Table 6 | 69 s |
| `max_step` | Tables 7, A3 | 285 s |
| `mechanism` | §5.5 | 57 s |
| `rk4` | §5.6 | 646 s |
| `winding_quadrature` | §4.3 | 38 s |
| `winding_overshoot` | §4.3 | 129 s |
| `solver` | Table 8, Figure 16 | 522 s |
| `precision` | §5.7, Figure 17 | 101 s |
| `cpu` | Table 9 | 1,877 s |
| `shadow_example` | Figure 4 data (a new run of the example image) | 58 s |
| `figures` | Figures 12, 16, 17 | 3 s |
| `diagrams` | Figures 1, 3, 5, 15 | 13 s |
| `result_figures` | Figures 2, 4, 6, 7, 8 | 48 s |
| `trace` | numbers in the text | under 1 s |
| `build` | HTML and PDF | 3 s |
| **Total** | | **about 6,280 s (1 h 45 min)** |

The steps `frame_dragging`, `bisection_failures`, `max_step`, `mechanism`,
`winding_overshoot` and `trace` were added after the full run of 15 steps
(5,770 s) and were timed separately. The steps `shadow_example`, `diagrams`
and `result_figures` were added on 2026-09-30 for the revised manuscript and
timed separately (the `shadow_example` time was measured while the test suite
was running); together they add about 2 minutes. The re-run of the example image
(run 20260929T194520Z-4f2393) was bit-for-bit identical to the run the
figure is drawn from (20260929T054034Z-de5a37).

## 4. What to expect

- **Numerical values** are deterministic: on the machine above, a full re-run
  from commit `7788ef7` reproduced every stored value bit for bit, and all
  six main-text figures byte for byte (`docs/publication/RESULT_REPRODUCTION.md`).
- **Run times** vary from run to run. The paper states the spread where it
  quotes a timing.
- **Loose-tolerance counts.** At rtol 10⁻⁴ the failure counts depend on the
  last bit of atol (one script gives 219 failed photons for a* = 0.99, another
  220). On another platform, last-bit differences in the maths library could
  move such counts by a few photons. At rtol ≤ 10⁻⁶ no photon failed in any
  test, but a pixel whose centre lies almost on the analytic curve can flip
  (one such pixel, 0.002 pixel from the curve, is reported in §5.2).
- **Figures 16 and 17** are drawn from the solver and precision run IDs fixed
  in `scripts/make_paper_figures.py`, and **Figures 2, 4, 6, 7 and 8** from
  the run records fixed in `scripts/make_paper_result_figures.py`, so they do
  not change when the benchmarks or validations are re-run. That script and
  `scripts/make_paper_diagrams.py` (Figure 15) repeat the recorded
  integrations and stop unless the repeated values equal the recorded ones;
  their PDF output is byte-identical from run to run.
- **Evidence for each number** in the text: `scripts/audit/trace_paper_numbers.py`
  prints it; `docs/publication/FINAL_CLAIM_AUDIT.md` maps it.

## 5. Where the outputs go

| Output | Location |
|---|---|
| Study data | `paper/data/*.json` (with git commit, hardware, run time) |
| Figures | `paper/figures/*.png` (and `*.pdf` for the LaTeX version) |
| Step logs | `paper/repro/*.log` |
| Validation and benchmark runs | `runs/<run id>/manifest.json`, `reports/<run id>/` |
| Audit diagnostics | `scripts/audit/*.out` |
| Paper preview | `paper/build/` (not tracked) |
| LaTeX manuscript | `paper/latex/` (see its README) |
