# Validation

## Purpose

Records how KerrRay's physics is validated before any result is trusted: the
Schwarzschild checks (photon sphere and critical impact parameter, EXP-001 and
EXP-002), the Kerr checks (horizon, Schwarzschild limit, conserved
quantities, prograde and retrograde behaviour, near-extremal spin,
forward/backward consistency) and the convergence studies demanded by
PROJECT.md sections 12, 13 and 41. Reference values are used only to check
computed results; every reported error is computed at run time and never
hard-coded (CLAUDE.md scientific rules).

Commands: `kerrray validate schwarzschild` (`configs/schwarzschild.yaml`) and
`kerrray validate kerr` (`configs/kerr.yaml`). Both print the computed values,
references and errors, record the run through
`kerrray.experiments.base.run_experiment` (`runs/<run_id>/manifest.json`,
`config.yaml`; `reports/<run_id>/summary.json`, `report.md`, figures) and exit
with code 1 when any check fails or a search cannot classify its rays.

## 1. Validation policy

* The references are **comparison-only constants**, defined once:
  `PHOTON_SPHERE_REFERENCE = 3` and `CRITICAL_IMPACT_REFERENCE = 3 sqrt(3)`
  (units of M) in `kerrray.validation.schwarzschild` (Misner, Thorne and
  Wheeler 1973, section 25.6; Chandrasekhar 1983, chapter 3). They enter only
  the error columns. All Kerr references (`r_+-`, equatorial photon orbits,
  critical impact parameters) are computed at run time by
  `kerrray.geometry` and `kerrray.photons.orbits`, which were derived with
  SymPy and matched to Bardeen, Press and Teukolsky 1972 (docs/derivations.md).
* The numerical values are **measured from integrated rays**: the capture
  threshold is located by classifying rays (`CAPTURED` or `ESCAPED`), never
  by evaluating a formula.
* Every tolerance is an input in the configuration file or a documented
  default of the parameter dataclass (section 7). A ray that ends in any
  other state, or a non-monotone outcome, stops the validation with an error
  instead of producing a number.

## 2. Schwarzschild (EXP-001, EXP-002; `kerrray.validation.schwarzschild`)

* **Photon sphere (EXP-001).** An equatorial photon launched tangentially
  (`p_r = 0`, `kerrray.geodesics.tangential_photon`) at `r0` sits at a radial
  turning point; below the unstable circular orbit it falls in, above it
  escapes. `photon_sphere_experiment` keeps a bracket `[r_lo, r_hi]` (captured,
  escaped) and shrinks it with `n_probe` rays per iteration integrated in one
  batch (`n_probe = 1` is classical bisection; the default 15 shrinks the
  bracket 16-fold per iteration) until the width is `<= photon_sphere_tol`.
  The discriminant `dp_r/dlambda` at launch is linear in `r0 - r_ph`, so the
  bracket resolves the radius itself.
* **Critical impact parameter (EXP-002).** `critical_impact_experiment`
  applies the same search to the impact parameter `b = |L_z|/E` of photons
  launched inward in the equatorial plane from `launch_radius`
  (`kerrray.validation.kerr_bisection.bisect_critical_impacts`).
* **Pass criteria.** `|r_ph - 3M| <= photon_sphere_tol` and
  `|b_c - 3 sqrt(3) M| <= critical_b_tol`: the error includes the bisection
  half-width and the integration error.
* **Convergence of b_c.** The search is repeated, centred on the measured
  `b_c` with half-width `convergence_halfwidth` and a bracket tolerance
  `convergence_tol` far below the errors, (a) with RK45 at each
  `rtol_levels` value (`atol` keeps the configured `atol/rtol` ratio) and (b)
  with fixed-step RK4 at each `step_sizes` value. Rays start and escape at
  `convergence_launch_radius` (an outbound photon beyond the single maximum of
  the effective potential never turns again, so the outcome equals that of a
  distant launch). The error against `3 sqrt(3) M` is fitted with a
  least-squares log-log slope and, when three levels are in a constant ratio,
  a reference-free Richardson order (`kerrray.validation.convergence`;
  Richardson 1911; Roache 1998). Errors at or below `convergence_tol` are
  excluded from the fits. Pass: the resolved rtol errors do not grow as
  `rtol` shrinks and the finest is within `critical_b_tol`; the fitted RK4
  order lies within `order_tolerance` of the formal order 4 (Hairer, Norsett
  and Wanner 1993, section II.1).
* **Fixed-step horizon crossings.** In Boyer-Lindquist coordinates
  `p_r = -sqrt(R)/Delta` diverges at the horizon. The adaptive scheme shrinks
  its step and lands in the capture margin; fixed-step RK4 evaluates stages
  across `Delta = 0`, and the step throws `r` to large negative values
  (measured: every Schwarzschild ray with `b = b_c - 0.01` and `h = 0.2 ...
  0.01 M` ends `OUT_OF_DOMAIN` after an inbound point at `r ~ 2.0-2.2 M`). The
  RK4 study therefore counts an `OUT_OF_DOMAIN` ray whose final radius is
  finite and `<= r_+ + horizon_epsilon` as captured and reports how many rays
  were relabelled (WHAT/WHY/LIMITATION in `kerr_bisection`). No other state is
  relabelled.

## 3. Kerr (`kerrray.validation.kerr`)

1. **horizon**: for the configured spin and every spin used by the other
   checks: `Delta(r_+) = Delta(r_-) = 0`, `r_+` against the larger root of
   `r^2 - 2Mr + a^2` from `numpy.roots` (independent of the closed form),
   `g_tt = 0` on `r_E(theta)`, `r_E(0) = r_+`, `r_E(pi/2) = 2M`. Pass: every
   error `<= horizon_tol`.
2. **schwarzschild_limit**: bisected prograde and retrograde `b_c(a)` for
   `small_spins`: `|b(a) - 3 sqrt(3) M|` must scale linearly (fitted exponent
   within `scaling_exponent_tol` of 1), the odd part `(b_pro - b_ret)/2` is
   linear with the slope of the derived closed form, the even part
   `(b_pro + b_ret)/2 - 3 sqrt(3) M` (`O(a^2)`) must be `<=
   schwarzschild_limit_tol` at the smallest spin, the metric difference
   `max |g(a) - g(0)|` must decrease linearly, and the bisected values must
   match the derived ones within `critical_b_tol` (relative).
3. **critical_impact**: prograde and retrograde `b_c` at `critical_spins`,
   bracket width `0.1 critical_b_tol M` (so the midpoint error stays well
   inside the tolerance), against `photons.orbits.critical_impact_parameters`
   (relative error `<= critical_b_tol`).
4. **conservation**: off-equatorial (`Q > 0`: `xi = 2`, `eta = 32`,
   `theta_0 = 60 deg`) rays from `launch_radius` at `conservation_spins`; all
   must escape and the maximum drift of `E`, `L_z`, `Q` and `|H|/E^2` must be
   `<= conservation_tol`. (`E` and `L_z` are exactly conserved by the
   Hamiltonian form; `Q` and the null constraint are the informative ones.)
   Escaping rays are used because captured rays accumulate a `1/Delta` null
   error at the horizon margin in Boyer-Lindquist coordinates.
5. **near_extremal**: a prograde equatorial ray with
   `b = b_c (1 + near_extremal_offset)` at `near_extremal_spins`, for every
   `horizon_epsilons` value: outcome, steps, turns, closest approach and
   drifts; also the analytic turning radius and the margin
   `r_turn - r_+` above which a `horizon_epsilon` would misclassify the ray.
   Pass: at the smallest margin every ray escapes, none fails or leaves the
   domain, and the `Q` and null drifts are `<= conservation_tol`.
6. **reversibility**: the off-equatorial ray at `reversibility_spin` is
   integrated for `lambda = reversibility_affine_fraction * launch_radius`
   with escape disabled, `p_mu` is reversed, the ray is integrated back for
   the same `lambda` (the integrator clips its last step to `lambda_max`
   exactly) and compared with the start. The geodesic equation is invariant
   under `(lambda, p) -> (-lambda, -p)` (docs/architecture.md section 1).
   Pass: both legs end `MAX_AFFINE_PARAMETER` and the radial (relative),
   angular, time (relative) and momentum (relative to `E`) errors are `<=
   reversibility_tol`.

## 4. Conservation diagnostics

`kerrray.validation.conservation` condenses the per-step drifts of
`kerrray.photons.constants` (relative to the first point with `E_0`-based
floors) into `DriftSummary` records, provides the off-equatorial launch
state, the per-ray summary row and the forward/backward round trip used by
the Kerr checks.

## 5. Test-suite map (PROJECT.md section 41)

| Requirement | Test module |
|---|---|
| Photon sphere and `b_c` recovered by bisection; invalid brackets rejected | `tests/test_schwarzschild.py` |
| RK4 fourth order and rtol convergence of `b_c`; horizon-crossing relabelling | `tests/test_schwarzschild.py` |
| Log-log and Richardson order estimates; drift summaries | `tests/test_validation_convergence.py` |
| Kerr horizon (Vieta relations), `a -> 0` limit linear in `a`, prograde/retrograde `b_c`, conservation, near-extremal, reversibility, report files | `tests/test_kerr.py` |
| Frame dragging (EXP-005) and spin sweep (EXP-003) | `tests/test_frame_dragging.py` |
| `kerrray validate schwarzschild/kerr` (output lines, exit codes, manifests) | `tests/test_commands_validate.py` |
| `kerrray geodesic`, `kerrray blackhole create` | `tests/test_commands_geodesic.py` |

The tests run shrunken versions (rays launched and escaping at 50 M, `rtol =
1e-8`, bracket tolerance `1e-6 M`, one spin per family, looser conservation
tolerance `1e-7`); together they take about 100 s. The default-configuration
results are in section 8.

## 6. Units and conventions

Geometric units `G = c = 1`, `M = 1`, lengths in M, Boyer-Lindquist
coordinates, spin `a/M`, prograde means `a L_z > 0` (docs/architecture.md
section 1).

## 7. Tolerances and parameters

`experiment.parameters.validation` of `configs/schwarzschild.yaml`
(`SchwarzschildValidationParams`); keys marked * are documented extensions
with defaults (not in the shipped file):

| Key | Default | Meaning |
|---|---|---|
| `photon_sphere_tol` | 1e-6 | bracket tolerance and pass tolerance on `r_ph` [M] |
| `critical_b_tol` | 1e-6 | bracket tolerance and pass tolerance on `b_c` [M] |
| `bisection_iterations` | 60 | iteration cap of every search |
| `photon_sphere_bracket`* | [2.5, 4.0] | initial tangential-launch bracket [M] (endpoints verified) |
| `impact_bracket`* | [4.0, 7.0] | initial `b` bracket [M] (endpoints verified) |
| `launch_radius`* | 1000 | launch radius of the EXP-002 rays [M] |
| `probes_per_iteration`* | 15 | rays per search iteration (1 = bisection) |
| `run_convergence`* | true | run the rtol and RK4 studies |
| `rtol_levels`* | [1e-6, 1e-8, 1e-10, 1e-12] | RK45 tolerances of the rtol study |
| `step_sizes`* | [0.4, 0.2, 0.1] | RK4 steps of the step study [M] |
| `convergence_launch_radius`* | 20 | launch and escape radius of the studies [M] |
| `convergence_halfwidth`* | 1e-2 | half-width of the study brackets around the measured `b_c` [M] |
| `convergence_tol`* | 1e-11 | bracket tolerance of the studies [M] |
| `order_tolerance`* | 0.5 | allowed deviation of the fitted RK4 order from 4 |

`experiment.parameters.validation` of `configs/kerr.yaml`
(`KerrValidationParams`; provisional D-009 values, confirmed by the run in
section 8):

| Key | Default | Meaning |
|---|---|---|
| `horizon_tol` | 1e-12 | horizon, ergosphere and quadratic-root errors |
| `small_spins` | [1e-2, 1e-3, 1e-4] | spins of the `a -> 0` check |
| `schwarzschild_limit_tol` | 1e-6 | even part at the smallest spin [M] |
| `conservation_tol` | 1e-8 | max drift of `E`, `L_z`, `Q`, `|H|/E^2` |
| `critical_b_tol` | 1e-6 | relative error of the bisected `b_c`; bracket `0.1 critical_b_tol M` |
| `bisection_iterations` | 60 | iteration cap |
| `near_extremal_spins` | [0.99, 0.999] | spins of the near-extremal check |
| `critical_spins`* | [0.5, 0.9, 0.99] | spins of the prograde/retrograde check |
| `conservation_spins`* | [0.9, 0.99, 0.999] | spins of the conservation check |
| `launch_radius`* | 1000 | launch radius [M] |
| `impact_bracket`* | [1.0, 10.0] | initial `b` bracket [M] |
| `probes_per_iteration`* | 8 | rays per direction per iteration |
| `scaling_exponent_tol`* | 0.1 | allowed deviation of the fitted exponents from 1 |
| `conservation_xi`*, `conservation_eta`*, `conservation_theta_deg`* | 2, 32, 60 | off-equatorial test ray |
| `near_extremal_offset`* | 1e-3 | `b = b_c (1 + offset)` |
| `horizon_epsilons`* | [1e-3, 1e-6] | capture margins compared |
| `reversibility_spin`*, `reversibility_affine_fraction`*, `reversibility_tol`* | 0.9, 1.9, 1e-6 | round trip |

## 8. Validation results

The two commands were run at their default configurations (no overrides)
from the repository root on 2026-09-29, each on a fresh `typer.Typer()` app
with `kerrray.commands.validate.register` and `typer.testing.CliRunner`
(the section panels below are the verbatim output; the banner belongs to
`cli.py` and is not part of a sub-app). `git rev-parse HEAD` before the runs
gave `a55ea1cb01dc69ea20244b180a156ddff007c167`; the lead committed during
the session, so each manifest records its own commit, both with a dirty
working tree (the Phase 3/4 files of this work were not yet committed):

| Command | Run id | Manifest `git_commit` | Exit | Runtime |
|---|---|---|---:|---:|
| `kerrray validate schwarzschild` | `20260929T053059Z-8efb29` | `a55ea1cb01dc69ea20244b180a156ddff007c167` (dirty) | 0 | 135.8 s |
| `kerrray validate kerr` | `20260929T053316Z-453d1f` | `a9c864d5fd7bdedc570a6442efc4d17374ffbfe5` (dirty) | 0 | 77.0 s |

Runtimes are wall-clock on the 8-core development machine while other jobs
were running; an earlier identical Kerr run took 135.8 s under heavier load.
Manifests, configuration copies, `summary.json`, `report.md` and figures are
in `runs/<run id>/` and `reports/<run id>/`.

### 8.1 `kerrray validate schwarzschild`

```text
Photon sphere
──────────────────────────────────────────────
Analytical photon sphere  3.000000 M                    
Numerical photon sphere   3.000000014901 M              
Absolute error            1.490e-08 M                   
Relative error            4.967e-09                     
Bracket width             8.941e-08 M after 6 iterations
Rays / max steps          92 / 340                      
Runtime                   8.71 s                        

Critical impact b_c
──────────────────────────────────────────────
Analytical critical impact b_c  5.196152 M                    
Numerical critical impact b_c   5.196152478456 M              
Absolute error                  5.575e-08 M                   
Relative error                  1.073e-08                     
Bracket width                   1.788e-07 M after 6 iterations
Rays / max steps                92 / 497                      
Runtime                         13.48 s                       

Convergence of b_c with rtol (rk45)
──────────────────────────────────────────────
rtol 1e-06        b_c 5.196152791472 M  error 3.688e-07  (6.5 s) 
rtol 1e-08        b_c 5.196152425704 M  error 2.997e-09  (10.0 s)
rtol 1e-10        b_c 5.196152422742 M  error 3.582e-11  (24.4 s)
rtol 1e-12        b_c 5.196152422705 M  error 1.434e-12  (50.1 s)
Fitted order      1.0032 (errors above 1.0e-11 M)                
Richardson order  0.9502                                         

Convergence of b_c with the RK4 step
──────────────────────────────────────────────
step_size 0.4      b_c 5.196154168036 M  error 1.745e-06  (3.6 s)              
step_size 0.2      b_c 5.196152529891 M  error 1.072e-07  (6.6 s)              
step_size 0.1      b_c 5.196152429346 M  error 6.639e-09  (10.5 s)             
Fitted order       4.0192 (errors above 1.0e-11 M)                             
Richardson order   4.0261                                                      
Horizon crossings  164 rays relabelled CAPTURED (fixed step, see               
                   kerr_bisection)                                             

Verdict
──────────────────────────────────────────────
photon_sphere     PASS  (absolute 1e-06)                     
critical_impact   PASS  (absolute 1e-06)                     
rtol_convergence  PASS  (finest_absolute 1e-06)              
rk4_convergence   PASS  (formal_order 4, order_tolerance 0.5)
All checks        PASS                                       
Run               20260929T053059Z-8efb29                    
Report dir        reports\20260929T053059Z-8efb29            
Runtime           135.8 s
```

Reading: the photon sphere and the critical impact parameter are recovered
to 1.5e-8 M and 5.6e-8 M (relative 5.0e-9 and 1.1e-8), inside the 1e-6 M
tolerances; both errors are below the half-widths of the final brackets
(4.5e-8 and 8.9e-8 M), so at `rtol = 1e-9` the result is limited by the
bracket, not by the integration. The rtol study shows the error of `b_c` falling in
proportion to `rtol` (fitted order 1.003 over the three resolved levels; the
`1e-12` level is below the `1e-11` bracket tolerance and excluded from the
fit, which is also why the Richardson estimate from the three finest values,
0.950, is less reliable). The fixed-step RK4 study gives the formal fourth
order (fitted 4.019, Richardson 4.026); its 164 horizon-crossing
relabellings are the captured rays of the fixed-step runs (section 2).

### 8.2 `kerrray validate kerr`

```text
Horizon and ergosphere
──────────────────────────────────────────────
Spins checked      0.0001, 0.001, 0.01, 0.5, 0.9, 0.99, 0.999
Worst error        3.043e-16                                 
r+ / r- (a = 0.9)  1.435889894354 / 0.564110105646 M         

Schwarzschild limit (a -> 0)
──────────────────────────────────────────────
a = 0.01             b_pro 5.176123415  b_ret 5.216123697 M  (b - 3 sqrt3 M)/a 
                     -2.00290 / +1.99713                                       
a = 0.001            b_pro 5.194152128  b_ret 5.198152131 M  (b - 3 sqrt3 M)/a 
                     -2.00029 / +1.99971                                       
a = 0.0001           b_pro 5.195952428  b_ret 5.196352412 M  (b - 3 sqrt3 M)/a 
                     -1.99995 / +1.99989                                       
Reference 3 sqrt3 M  5.196152423 M                                             
Exponent pro / ret   1.00032 / 0.99970                                         
Even part (min a)    2.642e-09 M                                               
Max rel. error       1.687e-09                                                 

Critical impact parameters
──────────────────────────────────────────────
a = 0.5           pro 4.096266670 (err 2.840e-09)  retro 6.138155715 (err      
                  1.652e-09)                                                   
a = 0.9           pro 2.844421402 (err 5.053e-10)  retro 6.832319226 (err      
                  6.663e-10)                                                   
a = 0.99          pro 2.251724342 (err 2.736e-09)  retro 6.983323434 (err      
                  3.611e-10)                                                   

Conservation (off-equatorial rays)
──────────────────────────────────────────────
a = 0.9           ESCAPED  E 0.000e+00  Lz 0.000e+00  Q 1.441e-09  null        
                  1.868e-09                                                    
a = 0.99          ESCAPED  E 0.000e+00  Lz 0.000e+00  Q 1.401e-09  null        
                  1.867e-09                                                    
a = 0.999         ESCAPED  E 0.000e+00  Lz 0.000e+00  Q 1.397e-09  null        
                  1.865e-09                                                    

Near-extremal prograde rays
──────────────────────────────────────────────
a = 0.99, eps 0.001   ESCAPED  r_min 1.180762 (r+ 1.141067)  turns 6.948  steps
                      290  Q 2.048e-30                                         
a = 0.99, eps 1e-06   ESCAPED  r_min 1.180762 (r+ 1.141067)  turns 6.948  steps
                      290  Q 2.048e-30                                         
a = 0.999, eps 0.001  ESCAPED  r_min 1.059570 (r+ 1.044710)  turns 16.332      
                      steps 322  Q 1.990e-30                                   
a = 0.999, eps 1e-06  ESCAPED  r_min 1.059570 (r+ 1.044710)  turns 16.332      
                      steps 322  Q 1.990e-30                                   

Forward/backward consistency
──────────────────────────────────────────────
radius            3.801e-09                                  
angles            5.624e-09                                  
time              4.966e-11                                  
momentum          1.086e-08                                  
Legs              MAX_AFFINE_PARAMETER / MAX_AFFINE_PARAMETER

Verdict
──────────────────────────────────────────────
horizon              PASS  (horizon_tol 1e-12)                                 
schwarzschild_limit  PASS  (critical_b_tol 1e-06, schwarzschild_limit_tol      
                     1e-06, scaling_exponent_tol 0.1)                          
critical_impact      PASS  (critical_b_tol 1e-06)                              
conservation         PASS  (conservation_tol 1e-08)                            
near_extremal        PASS  (conservation_tol 1e-08)                            
reversibility        PASS  (reversibility_tol 1e-06)                           
All checks           PASS                                                      
Run                  20260929T053316Z-453d1f                                   
Report dir           reports\20260929T053316Z-453d1f                           
Runtime              77.0 s
```

Reading: the bisected critical impact parameters agree with the closed forms
of `photons.orbits` to relative errors between 3.6e-10 and 2.8e-9 at spins
0.5, 0.9 and 0.99; as `a -> 0` the deviation from `3 sqrt(3) M` is linear
(`(b - 3 sqrt(3) M)/a = -2.000 / +2.000` for prograde / retrograde at
`a = 1e-4`, fitted exponents 1.0003 and 0.9997) and the `O(a^2)` even part
is 2.6e-9 M at `a = 1e-4`. `E` and `L_z` are exactly conserved by the
Hamiltonian form; `Q` and the null constraint drift by at most 1.4e-9 and
1.9e-9 along the off-equatorial escaping rays up to `a = 0.999`. The
near-extremal prograde rays at `b = b_c (1 + 1e-3)` wind 6.9 (`a = 0.99`)
and 16.3 (`a = 0.999`) turns with closest approach 0.040 M and 0.015 M
outside `r_+` (the analytic turning radii give the same margins, 0.0397 and
0.0149 M, as the largest `horizon_epsilon` that would still classify them
correctly), and their outcome does not change between the capture margins
`1e-3` and `1e-6`; their null-constraint error peaks at 3.9e-9. The forward/backward round trip returns to the start
within 3.8e-9 (radius, relative) and 1.1e-8 (momentum, relative to `E`).

These runs confirm the provisional D-009 tolerances of `configs/kerr.yaml`
(docs/decisions.md): every check passes with at least an order of magnitude
of margin except the null-constraint drifts, which pass `conservation_tol =
1e-8` by factors of about 5 (off-equatorial rays, 1.9e-9) and 2.6
(near-extremal rays, 3.9e-9) at `rtol = 1e-9`; a tighter `conservation_tol`
would need a tighter `rtol`.
