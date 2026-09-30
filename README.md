# KerrRay

KerrRay is an open-source Python ray tracer that integrates photon geodesics
in Kerr spacetime on an ordinary CPU and measures how accurately it
reconstructs the black-hole shadow.

Author: Joel Prince (independent researcher). Licence: MIT. The accompanying
manuscript has not been peer reviewed and is not part of this repository.

## Scientific purpose

Ray-traced black-hole images are routinely checked against the analytic Kerr
shadow (the Bardeen curve), but the numerical error of the reconstructed
shadow edge is seldom reported as a function of the choices that control it.
This project measures how image resolution and integrator tolerance set that
error, separates integration error from sampling error, and diagnoses the
integration failures that appear at loose tolerance in Boyer-Lindquist
coordinates. The physics (Kerr metric, photon orbits, Bardeen curve) is
established and is used only for validation.

## Main findings

For spins 0 to 0.99, inclinations 17° to 85°, resolutions 64² to 512² and
relative tolerances 10⁻⁴ to 10⁻¹²:

- **Edges read from a pixel mask are limited by sampling.** At the tolerances
  tested at or below 10⁻⁶ the traced mask equals the mask of an ideal
  integrator pixel for pixel, apart from one pixel whose centre lies 0.002
  pixel from the curve; the rms edge error is 0.21 to 0.25 pixel and falls
  as 1/N.
- **Edges located by bisection are limited by the tolerance,** at close to
  2 × rtol in units of M from 10⁻⁵ to 10⁻⁸.
- **At rtol 10⁻⁴ photons fail.** Most take a single oversized step through
  the hole, because the adaptive step is unbounded by default. The rest
  cross the spin axis, and those that should have escaped become spurious
  shadow pixels far from the edge. A step cap removes the first mechanism
  but not the second.
- **Fixed-step RK4 breaks down at the horizon,** and a few captured photons
  are falsely recorded as escaping.
- **The finite observer distance** of 1000 M is corrected to a residual below
  10⁻⁴ M.
- **Single precision** classified three tested images identically to double
  precision.

The engine is validated against exact results: the Schwarzschild photon
sphere and critical impact parameter to relative errors of 5 × 10⁻⁹ and
1.1 × 10⁻⁸, and the near-critical winding rate to 0.04% for nine spin and
direction cases.

### Limitations

- KerrRay was compared with analytic results only, not with another ray
  tracer; Kerr-Schild coordinates were not tested.
- Face-on and exactly edge-on views were not tested; the tested
  inclinations are 17°, 60° and 85°.
- atol was not varied independently of rtol (atol = 10⁻² rtol throughout);
  the mask-edge constant of about 0.23 pixel belongs to the extraction method
  used.
- At rtol 10⁻⁴ photons fail; at 10⁻⁵ no image pixel failed and two
  bisection searches (a* = 0.9 and 0.99) each lost one photon; at 10⁻⁶ and
  tighter nothing failed.
- Run times come from one laptop that was sometimes shared with other jobs;
  only Windows 11 was used.
- KerrRay makes no observational claims. GPU acceleration is not implemented.

## Reproducibility

- Every table and figure of the study is produced by a script;
  `python scripts/reproduce_paper.py --list` maps each one to its command.
- A full re-run of every study from a committed tree reproduced every stored
  numerical value bit for bit. The figure scripts
  `scripts/make_paper_diagrams.py` and `scripts/make_paper_result_figures.py`
  repeat recorded integrations and stop unless the repeated values equal the
  recorded ones.
- The full reproduction took about 1 h 45 min on an 8-core laptop.
- Environment of the published results: Python 3.13.13 with NumPy 2.5.3,
  SciPy 1.18.1, SymPy 1.14.0, Numba 0.67.0 and Matplotlib 3.11.2.
- The test suite has 920 tests.
- Every run records its git commit, configuration, package versions and
  hardware in `runs/<run id>/manifest.json`.
- Only Windows 11 was used; timings are indicative.

This repository contains the software, configurations, tests and scripts.
The manuscript and the stored results of the study (study data, run records,
figures and reproduction logs) are not included; running the scripts
regenerates them under `paper/`, `runs/` and `reports/`.

## Installation

Python 3.11 or newer.

```bash
git clone https://github.com/joelprince2601/kerrray.git
cd kerrray
python -m venv .venv
.venv/Scripts/activate              # macOS/Linux: source .venv/bin/activate
pip install -e ".[dev,fast]"        # "fast" adds the Numba backend
python -m pytest                    # 920 tests: 919 pass, 1 skips (it checks PROJECT.md, not published)
```

Without Numba the NumPy backend gives the same classifications but was 10 to
22 times slower in the measurements of the study.

## Reproducing the paper

```bash
python scripts/reproduce_paper.py --list    # every result and the command behind it
python scripts/reproduce_paper.py           # everything; about 1 h 45 min measured on 8 logical cores
python scripts/reproduce_paper.py --steps winding,oracle   # selected steps
```

[REPRODUCING.md](REPRODUCING.md) gives the steps, outputs and measured run
times. The steps `figures`, `result_figures`, `trace` and `build`, and
`scripts/audit/compare_reproduction.py`, read the stored records of the
published runs or the manuscript, which are not part of this repository;
they do not run without them. Every other step regenerates its results from
scratch.

## Repository structure

| Path | Contents |
|---|---|
| `src/kerrray/` | The engine: geometry, geodesics, photons, ray tracing, validation, benchmarks, experiments, and a local web app |
| `tests/` | The test suite (920 tests) |
| `configs/` | Parameters of every run |
| `scripts/` | Studies behind the paper, `reproduce_paper.py`, figure scripts, `audit/` diagnostics and their recorded outputs |
| `docs/` | Equations, derivations, numerical methods, validation and experiment notes |

## Paper

The accompanying manuscript, *How accurate is a numerically reconstructed
Kerr black-hole shadow? Resolution, tolerance and integration failures in an
open CPU ray tracer* (Joel Prince), is not part of this repository. It has not
been peer reviewed or published; a link will be added here when it is public.

## Data availability

The code, configurations, tests and reproduction scripts are in this
repository: https://github.com/joelprince2601/kerrray. An archived release
with a DOI does not exist yet.

## Citation

Citation metadata are in [CITATION.cff](CITATION.cff). Until a DOI exists,
cite the software as:

> Prince, J. KerrRay: a CPU ray tracer for photon geodesics in Kerr
> spacetime, version 0.1.0 (software), https://github.com/joelprince2601/kerrray.

## License

MIT; see [LICENSE](LICENSE).

## Use of AI tools

The software, experiments and documentation were produced with extensive
assistance from an AI coding assistant (Claude, Anthropic) under the author's
direction. The AI is not an author.
