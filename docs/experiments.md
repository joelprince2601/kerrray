# Experiments

Index of the required experiments EXP-001 to EXP-011 (PROJECT.md section 37).
Every experiment reads its parameters from a configuration file, writes a
reproducible run under `runs/<run_id>/` (manifest and configuration, section
35) and a report under `reports/<run_id>/` (section 36). The detailed method
and the measured results live in the linked documents; numbers appear there
only after they were measured.

| ID | Experiment | How to run | Configuration | Results and method |
|---|---|---|---|---|
| EXP-001 | Schwarzschild photon sphere | `kerrray validate schwarzschild` | `configs/schwarzschild.yaml` | [validation.md](validation.md) |
| EXP-002 | Schwarzschild critical impact parameter | `kerrray validate schwarzschild` | `configs/schwarzschild.yaml` | [validation.md](validation.md) |
| (Kerr) | Horizon, a to 0 limit, prograde and retrograde b_c, conservation, near-extremal | `kerrray validate kerr` | `configs/kerr.yaml` | [validation.md](validation.md) |
| EXP-003 | Spin sweep | `kerrray experiment --config configs/kerr.yaml --name spin_sweep`; shadows with `--name shadow` on `configs/shadow.yaml` | `configs/kerr.yaml`, `configs/shadow.yaml` | [experiments_kerr.md](experiments_kerr.md), [shadow.md](shadow.md) |
| EXP-004 | Inclination sweep | `kerrray experiment --config configs/shadow.yaml --name shadow --set experiment.parameters.mode=inclination_sweep` | `configs/shadow.yaml` | [shadow.md](shadow.md) |
| EXP-005 | Frame dragging | `kerrray experiment --config configs/kerr.yaml --name frame_dragging` | `configs/kerr.yaml` | [experiments_kerr.md](experiments_kerr.md) |
| EXP-006 | Solver convergence and step size | `kerrray benchmark solver`; `kerrray experiment --config configs/convergence.yaml --name step_size` | `configs/convergence.yaml`, `configs/benchmark.yaml` | [numerical_analysis.md](numerical_analysis.md) |
| EXP-007 | Near-critical instability | `kerrray experiment --config configs/near_critical.yaml --name near_critical` | `configs/near_critical.yaml` | [numerical_analysis.md](numerical_analysis.md) |
| EXP-008 | Shadow resolution convergence | `kerrray experiment --config configs/shadow.yaml --name shadow --set experiment.parameters.mode=convergence` | `configs/shadow.yaml` | [shadow.md](shadow.md) |
| EXP-009 | CPU performance | `kerrray benchmark cpu` | `configs/benchmark.yaml` | [numerical_analysis.md](numerical_analysis.md), [performance.md](performance.md) |
| EXP-010 | GPU performance | `kerrray benchmark gpu` (reports that GPU is optional future work; no device is used) | `configs/gpu.yaml` | [decisions.md](decisions.md) D-007 |
| EXP-011 | FP32 vs FP64 | `kerrray benchmark precision` | `configs/benchmark.yaml` | [numerical_analysis.md](numerical_analysis.md) |
| (extra) | Strong-field lensing | `kerrray lens --spin 0 --impact-range 4,20` | `configs/lensing.yaml` | [lensing.md](lensing.md) |
| (extra) | Simplified accretion disk with redshift | `kerrray render --spin 0.9 --inclination 75 --fov 25` | `configs/shadow.yaml` | [rendering.md](rendering.md) |

Related background: [scientific_background.md](scientific_background.md),
[equations.md](equations.md), [derivations.md](derivations.md),
[numerical_methods.md](numerical_methods.md), [raytracing.md](raytracing.md).
