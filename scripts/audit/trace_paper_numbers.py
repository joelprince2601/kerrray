"""Print the evidence behind the numerical statements of research.md.

Reads the stored study data (paper/data/*.json), the committed original of
shadow_convergence.json (for run times measured before the re-run), and the
benchmark summaries in reports/, and prints each value next to the section
of the paper that quotes it. Used by docs/publication/FINAL_CLAIM_AUDIT.md.
Run: ./.venv/Scripts/python.exe scripts/audit/trace_paper_numbers.py
"""
import json
import math
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "paper" / "data"


def load(name):
    return json.loads((D / name).read_text(encoding="utf-8"))


def git_json(rev, rel):
    out = subprocess.run(["git", "show", f"{rev}:{rel}"], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", check=True)
    return json.loads(out.stdout)


def rng(vals, fmt="{:.3g}"):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return f"{fmt.format(min(vals))} .. {fmt.format(max(vals))}"


print("== Table 3 / 5.1: convergence (paper/data/shadow_convergence.json)")
sc = load("shadow_convergence.json")
rows = sc["rows"]
for a in (0.0, 0.5, 0.9, 0.99):
    r5 = {r["resolution"]: r for r in rows if r["spin"] == a and r["rtol"] == 1e-5}
    tight = [r for r in rows if r["spin"] == a and r["rtol"] <= 1e-5]
    same = all(r["rms_M"] == r5[r["resolution"]]["rms_M"] for r in tight)
    fit = [f for f in sc["fits"] if f.get("spin") == a] if isinstance(sc["fits"], list) else sc["fits"]
    print(f" a={a}: rms 64..512 = " + ", ".join(f"{r5[n]['rms_M']:.3g}" for n in (64, 128, 256, 512)) +
          f"; rms px@512 {r5[512]['rms_px']:.3f}; max@512 {r5[512]['max_M']:.3g} M ({r5[512]['max_px']:.2f} px); area@512 {r5[512]['area_rel_err']:.2g};"
          f" identical for rtol<=1e-5: {same}")
print(" fits:", json.dumps(sc["fits"])[:600])
print(" rms_px over all rtol<=1e-5 rows:", rng([r["rms_px"] for r in rows if r["rtol"] <= 1e-5]))
print(" max_px at 512, rtol<=1e-5:", rng([r["max_px"] for r in rows if r["rtol"] <= 1e-5 and r["resolution"] == 512]))
print(" failed rays by (N, rtol):")
for n in (64, 128, 256, 512):
    for t in (1e-4, 1e-5):
        print(f"   N={n} rtol={t:g}: " + ", ".join(str(r["failed_rays"]) for r in rows if r["resolution"] == n and r["rtol"] == t))
print(" failed max |alpha| (M) at rtol 1e-4:", rng([r["failed_max_abs_alpha_M"] for r in rows if r["rtol"] == 1e-4 and r["failed_rays"]]))
for label, data in (("re-run (current file)", sc), ("original @7788ef7", git_json("7788ef7", "paper/data/shadow_convergence.json"))):
    rr = data["rows"]
    t5 = [r["runtime_s"] for r in rr if r["resolution"] == 512 and r["rtol"] == 1e-5]
    t10 = [r["runtime_s"] for r in rr if r["resolution"] == 512 and r["rtol"] == 1e-10]
    ratios = [b / a for a, b in zip(t5, t10)]
    print(f" runtime N=512 {label}: rtol1e-5 {rng(t5)} s, rtol1e-10 {rng(t10)} s, ratio {rng(ratios)}")
print(" max error at rtol 1e-4, N=512 (M):", ", ".join(f"{r['max_M']:.3g}" for r in rows if r["rtol"] == 1e-4 and r["resolution"] == 512))
m4 = {r["spin"]: r["max_M"] for r in rows if r["rtol"] == 1e-4 and r["resolution"] == 512}
m5 = {r["spin"]: r["max_M"] for r in rows if r["rtol"] == 1e-5 and r["resolution"] == 512}
print(" max-error ratio 1e-4/1e-5 at 512:", ", ".join(f"{m4[a]/m5[a]:.2f}" for a in m4))

print("\n== 5.2, 5.5, 5.7: oracle (paper/data/oracle.json)")
orc = load("oracle.json")["rows"]
for g in sorted({r["group"] for r in orc}):
    sub = [r for r in orc if r["group"] == g]
    print(f" group {g}: {len(sub)} rows")
    for r in sub:
        te, oe = r["traced_edge"], r["oracle_edge"]
        print(f"   a={r['spin']} i={r['inclination_deg']} N={r['resolution']} rtol={r['rtol']:g} r_o={r['observer_radius']:g} {r['dtype']}: disagree {r['disagree']} "
              f"(max {r['disagree_max_dist_px']:.1f} px), failed {r['failed']} (outside {r['failed_outside_oracle']}), "
              f"rms px traced {te.get('rms_px', float('nan')):.3f} oracle {oe.get('rms_px', float('nan')):.3f}")

print("\n== Table 4: bisection (paper/data/edge_bisection.json)")
eb = load("edge_bisection.json")
for r in eb["rows"]:
    print(f" a={r['spin']} rtol={r['rtol']:g}: rms {r['rms_error_M']:.2g} M, rms/rtol {r['rms_error_M']/r['rtol']:.2f}, failed {r['failed_rays']}, bracket {r['bracket_M']:.2g}, {r['runtime_s']:.2f} s")

print("\n== Table 5: finite distance (paper/data/finite_distance.json)")
for r in load("finite_distance.json")["rows"]:
    print(f" a={r['spin']} r_o={r['observer_radius']:g}: uncorrected {r['disagree_uncorrected']}, corrected {r['disagree_corrected']}, "
          f"expected flips {r['expected_flips_for_uncorrected_shift']:.1f}, bound {r['residual_shift_bound_95_M']:.2g} M, k-1 {r['k_minus_1']:.3g}, L {r['curve_length_M']:.3g} M")

print("\n== Table 2: winding (paper/data/winding.json)")
w = load("winding.json")
print(" rtol", w["rtol"], "launch", w["launch_radius"], "deltas", w["deltas"])
for r in w["rows"]:
    print(f" a={r['spin']} {r['direction']}: r_ph {r['r_ph']:.4f} gamma {r['gamma_phi']:.4f} pred {r['predicted_turns_per_decade']:.5f} meas {r['rate_per_decade']:.5f} diff {100*r['relative_difference']:+.3f}%")

print("\n== Tables 8, 9 and 5.7: benchmark summaries")
for rid, what in (("20260929T051346Z-f444b5", "solver original"), ("20260929T095157Z-057992", "solver re-run")):
    s = json.loads((ROOT / "reports" / rid / "summary.json").read_text())["results"]
    print(f" {what}: repeats {s['repeats']}, reference rtol {s['reference']['rtol']}")
    for c in s["configurations"]:
        print(f"   {c['label']}: {c['runtime_s']:.1f} s, steps {c['mean_steps']:.0f}, median err {c['median_trajectory_error']:.2g}, max {c['max_trajectory_error']:.2g}, "
              f"compared {c['n_compared']}, OOD {c['outcomes']['OUT_OF_DOMAIN']}")
sol = json.loads((ROOT / "reports" / "20260929T051346Z-f444b5" / "summary.json").read_text())["results"]["configurations"]
rk = [c for c in sol if c["solver"] == "rk45"]
lt = np.log10([c["setting"] for c in rk])
print(" RK45 error slope vs rtol: max %.3f, median %.3f" % (np.polyfit(lt, np.log10([c["max_trajectory_error"] for c in rk]), 1)[0],
                                                          np.polyfit(lt, np.log10([c["median_trajectory_error"] for c in rk]), 1)[0]))
for rid in ("20260929T052912Z-f23a49", "20260929T100220Z-46bc35"):
    s = json.loads((ROOT / "reports" / rid / "summary.json").read_text())["results"]
    p = s["parameters"]
    print(f" cpu {rid}: parameters {json.dumps(p)[:300]}")
    print(f"   camera {json.dumps(s['camera'])[:200]}; integration {json.dumps(s['integration'])}")
for rid in ("20260929T060841Z-f365e3",):
    s = json.loads((ROOT / "reports" / rid / "summary.json").read_text())["results"]
    print(f" precision {rid}: keys {list(s.keys())}")
    for c in s.get("comparisons", [])[:12]:
        print("   ", {k: v for k, v in c.items() if not isinstance(v, (list, dict))})
