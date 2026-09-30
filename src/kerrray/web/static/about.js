/* KerrRay "About the project" page: what the project shows, every formula the
   engine uses (native MathML), and the numbers computed in this session. */
"use strict";

const KerrAbout = (() => {
  // --- tiny MathML helpers -------------------------------------------------
  const mi = x => `<mi>${x}</mi>`, mn = x => `<mn>${x}</mn>`, mo = x => `<mo>${x}</mo>`;
  const row = (...xs) => `<mrow>${xs.join("")}</mrow>`;
  const frac = (a, b) => `<mfrac>${row(a)}${row(b)}</mfrac>`;
  const sup = (a, b) => `<msup>${row(a)}${row(b)}</msup>`;
  const sub = (a, b) => `<msub>${row(a)}${row(b)}</msub>`;
  const subsup = (a, b, c) => `<msubsup>${row(a)}${row(b)}${row(c)}</msubsup>`;
  const sqrt = a => `<msqrt>${row(a)}</msqrt>`;
  const root = (a, n) => `<mroot>${row(a)}${row(n)}</mroot>`;
  // Brackets stretch only around fractions; simple arguments keep normal-size brackets.
  const fence = (o, c) => a => a.includes("<mfrac") ? `${mo(o)}${a}${mo(c)}` : `<mo stretchy="false">${o}</mo>${a}<mo stretchy="false">${c}</mo>`;
  const par = fence("(", ")");
  const brk = fence("[", "]");
  const block = (...xs) => `<math display="block">${row(...xs)}</math>`;
  const inl = (...xs) => `<math>${row(...xs)}</math>`;
  const eq = mo("="), plus = mo("+"), minus = mo("&#x2212;"), times = mo("&#x2062;"), pm = mo("&#xB1;"), mp = mo("&#x2213;");
  const S2 = sup(mi("sin"), mn(2)), C2 = sup(mi("cos"), mn(2));
  const th = mi("&#x3B8;"), ph = mi("&#x3C6;"), Sig = mi("&#x3A3;"), Del = mi("&#x394;"), lam = mi("&#x3BB;");
  const d = x => row(mi("d"), x);

  const F = {
    sigma: block(Sig, eq, sup(mi("r"), mn(2)), plus, sup(mi("a"), mn(2)), C2, th, mo(",&#x2003;"), Del, eq, sup(mi("r"), mn(2)), minus, mn(2), mi("M"), mi("r"), plus, sup(mi("a"), mn(2)), mo(",&#x2003;"), mi("a"), eq, sub(mi("a"), mo("*")), mi("M")),
    metric: block(
      mi("d"), sup(mi("s"), mn(2)), eq, minus, par(row(mn(1), minus, frac(row(mn(2), mi("M"), mi("r")), Sig))), d(sup(mi("t"), mn(2))),
      minus, frac(row(mn(4), mi("M"), mi("a"), mi("r"), S2, th), Sig), d(mi("t")), d(ph),
      plus, frac(Sig, Del), d(sup(mi("r"), mn(2))), plus, Sig, d(sup(th, mn(2))),
      plus, par(row(sup(mi("r"), mn(2)), plus, sup(mi("a"), mn(2)), plus, frac(row(mn(2), mi("M"), sup(mi("a"), mn(2)), mi("r"), S2, th), Sig))), S2, th, d(sup(ph, mn(2)))),
    horizons: block(sub(mi("r"), mo("&#xB1;")), eq, mi("M"), pm, sqrt(row(sup(mi("M"), mn(2)), minus, sup(mi("a"), mn(2)))), mo(",&#x2003;"),
      sub(mi("r"), mi("E")), par(th), eq, mi("M"), plus, sqrt(row(sup(mi("M"), mn(2)), minus, sup(mi("a"), mn(2)), C2, th))),
    hamiltonian: block(mi("H"), eq, frac(mn(1), mn(2)), sup(mi("g"), row(mi("&#x3BC;"), mi("&#x3BD;"))), sub(mi("p"), mi("&#x3BC;")), sub(mi("p"), mi("&#x3BD;")), eq, mn(0)),
    eom: block(frac(row(mi("d"), sup(mi("x"), mi("&#x3BC;"))), row(mi("d"), lam)), eq, sup(mi("g"), row(mi("&#x3BC;"), mi("&#x3BD;"))), sub(mi("p"), mi("&#x3BD;")), mo(",&#x2003;"),
      frac(row(mi("d"), sub(mi("p"), mi("&#x3BC;"))), row(mi("d"), lam)), eq, minus, frac(mn(1), mn(2)), frac(row(mo("&#x2202;"), sup(mi("g"), row(mi("&#x3B1;"), mi("&#x3B2;")))), row(mo("&#x2202;"), sup(mi("x"), mi("&#x3BC;")))), sub(mi("p"), mi("&#x3B1;")), sub(mi("p"), mi("&#x3B2;"))),
    conserved: block(mi("E"), eq, minus, sub(mi("p"), mi("t")), mo(",&#x2003;"), sub(mi("L"), mi("z")), eq, sub(mi("p"), ph), mo(",&#x2003;"),
      mi("Q"), eq, subsup(mi("p"), th, mn(2)), plus, C2, th, par(row(frac(subsup(mi("L"), mi("z"), mn(2)), row(S2, th)), minus, sup(mi("a"), mn(2)), sup(mi("E"), mn(2))))),
    potentials: block(frac(row(mi("R"), par(mi("r"))), sup(mi("E"), mn(2))), eq, sup(brk(row(sup(mi("r"), mn(2)), plus, sup(mi("a"), mn(2)), minus, mi("a"), mi("&#x3BE;"))), mn(2)), minus, Del, brk(row(mi("&#x3B7;"), plus, sup(par(row(mi("&#x3BE;"), minus, mi("a"))), mn(2)))), mo(",&#x2003;"),
      frac(row(mi("&#x398;"), par(th)), sup(mi("E"), mn(2))), eq, mi("&#x3B7;"), plus, sup(mi("a"), mn(2)), C2, th, minus, sup(mi("&#x3BE;"), mn(2)), sup(mi("cot"), mn(2)), th),
    spherical: block(mi("&#x3BE;"), par(mi("r")), eq, frac(row(sup(mi("r"), mn(2)), par(row(mn(3), mi("M"), minus, mi("r"))), minus, sup(mi("a"), mn(2)), par(row(mi("r"), plus, mi("M")))), row(mi("a"), par(row(mi("r"), minus, mi("M"))))), mo(",&#x2003;"),
      mi("&#x3B7;"), par(mi("r")), eq, frac(row(sup(mi("r"), mn(3)), brk(row(mn(4), mi("M"), sup(mi("a"), mn(2)), minus, mi("r"), sup(par(row(mi("r"), minus, mn(3), mi("M"))), mn(2))))), row(sup(mi("a"), mn(2)), sup(par(row(mi("r"), minus, mi("M"))), mn(2))))),
    photonOrbit: block(sub(mi("r"), mi("ph")), eq, mn(2), mi("M"), brk(row(mn(1), plus, mi("cos"), par(row(frac(mn(2), mn(3)), mi("arccos"), par(row(mp, mi("a"), mo("/"), mi("M"))))))), mo(",&#x2003;"),
      sub(mi("b"), mi("c")), eq, mo("|"), mi("&#x3BE;"), par(sub(mi("r"), mi("ph"))), mo("|"), mo(",&#x2003;"), subsup(mi("b"), mi("c"), row(mi("a"), eq, mn(0))), eq, mn(3), sqrt(mn(3)), mi("M")),
    isco: block(sub(mi("Z"), mn(1)), eq, mn(1), plus, root(row(mn(1), minus, sup(sub(mi("a"), mo("*")), mn(2))), mn(3)), brk(row(root(row(mn(1), plus, sub(mi("a"), mo("*"))), mn(3)), plus, root(row(mn(1), minus, sub(mi("a"), mo("*"))), mn(3)))), mo(",&#x2003;"),
      sub(mi("Z"), mn(2)), eq, sqrt(row(mn(3), sup(sub(mi("a"), mo("*")), mn(2)), plus, subsup(mi("Z"), mn(1), mn(2)))), mo(",&#x2003;"),
      sub(mi("r"), mi("isco")), eq, mi("M"), brk(row(mn(3), plus, sub(mi("Z"), mn(2)), mp, sqrt(row(par(row(mn(3), minus, sub(mi("Z"), mn(1)))), par(row(mn(3), plus, sub(mi("Z"), mn(1)), plus, mn(2), sub(mi("Z"), mn(2))))))))),
    kepler: block(mi("&#x3A9;"), eq, frac(row(pm, sup(mi("M"), row(mn(1), mo("/"), mn(2)))), row(sup(mi("r"), row(mn(3), mo("/"), mn(2))), pm, mi("a"), sup(mi("M"), row(mn(1), mo("/"), mn(2)))))),
    celestial: block(mi("&#x3B1;"), eq, minus, sub(mi("r"), mi("o")), frac(sup(mi("p"), par(ph)), sup(mi("p"), par(mi("t")))), mo(",&#x2003;"), mi("&#x3B2;"), eq, sub(mi("r"), mi("o")), frac(sup(mi("p"), par(th)), sup(mi("p"), par(mi("t")))),
      mo("&#x2003;&#x2192;&#x2003;"), mi("&#x3B1;"), eq, minus, frac(mi("&#x3BE;"), row(mi("sin"), sub(th, mi("o")))), mo(",&#x2003;"), mi("&#x3B2;"), eq, pm, sqrt(row(mi("&#x3B7;"), plus, sup(mi("a"), mn(2)), C2, sub(th, mi("o")), minus, sup(mi("&#x3BE;"), mn(2)), sup(mi("cot"), mn(2)), sub(th, mi("o"))))),
    zamo: block(mi("&#x3C9;"), eq, minus, frac(sub(mi("g"), row(mi("t"), ph)), sub(mi("g"), row(ph, ph))), mo(",&#x2003;"), sup(mi("N"), mn(2)), eq, minus, par(row(sub(mi("g"), row(mi("t"), mi("t"))), minus, frac(subsup(mi("g"), row(mi("t"), ph), mn(2)), sub(mi("g"), row(ph, ph))))), mo(",&#x2003;"),
      sub(mi("e"), par(mi("t"))), eq, frac(mn(1), mi("N")), par(row(sub(mo("&#x2202;"), mi("t")), plus, mi("&#x3C9;"), sub(mo("&#x2202;"), ph)))),
    rk45: block(mi("err"), eq, sqrt(row(frac(mn(1), mi("n")), mo("&#x2211;"), sup(par(frac(sub(mi("e"), mi("i")), row(mi("atol"), plus, mi("rtol"), mo("&#xB7;"), mi("max"), par(row(mo("|"), sub(mi("y"), mi("i")), mo("|"), mo(","), mo("|"), subsup(mi("y"), mi("i"), mi("new")), mo("|")))))), mn(2)))), mo(",&#x2003;"),
      sub(mi("h"), mi("new")), eq, mi("h"), mo("&#xB7;"), mi("clip"), par(row(mn("0.9"), sup(mi("err"), row(minus, mn(1), mo("/"), mn(5))), mo(","), mn("0.2"), mo(","), mn(5)))),
    redshift: block(mi("g"), eq, frac(sub(mi("&#x3BD;"), mi("obs")), sub(mi("&#x3BD;"), mi("em"))), eq, frac(mi("E"), row(minus, sub(mi("p"), mi("&#x3BC;")), sup(mi("u"), mi("&#x3BC;")))), eq, frac(mn(1), row(sup(mi("u"), mi("t")), par(row(mn(1), minus, mi("&#x3A9;"), mi("&#x3BE;"))))), mo(",&#x2003;"),
      sub(mi("I"), mi("obs")), eq, sup(mi("g"), mn(3)), sub(mi("I"), mi("em"))),
    deflection: block(mi("&#x3B4;"), par(mi("b")), eq, mn(2), `<munderover><mo>&#x222B;</mo>${sub(mi("r"), mn(0))}<mi>&#x221E;</mi></munderover>`, frac(row(mi("d"), mi("r")), row(sup(mi("r"), mn(2)), sqrt(row(frac(mn(1), sup(mi("b"), mn(2))), minus, frac(row(mn(1), minus, mn(2), mi("M"), mo("/"), mi("r")), sup(mi("r"), mn(2))))))), minus, mi("&#x3C0;"), mo("&#x2003;&#x2248;&#x2003;"), frac(row(mn(4), mi("M")), mi("b")), mo("&#x2003;(weak field)")),
    area: block(mi("A"), eq, frac(mn(1), mn(2)), mo("|"), mo("&#x2211;"), par(row(sub(mi("&#x3B1;"), mi("i")), sub(mi("&#x3B2;"), row(mi("i"), plus, mn(1))), minus, sub(mi("&#x3B1;"), row(mi("i"), plus, mn(1))), sub(mi("&#x3B2;"), mi("i")))), mo("|"), mo(",&#x2003;"),
      sub(mi("A"), mi("num")), eq, sub(mi("N"), mi("captured")), mo("&#xB7;"), sup(par(row(mn(2), mi("F"), mo("/"), mi("n"))), mn(2))),
  };

  const fmt = (v, dg = 6) => (v === undefined || v === null || Number.isNaN(v)) ? "—" : (Math.abs(v) !== 0 && (Math.abs(v) < 1e-3 || Math.abs(v) >= 1e5) ? v.toExponential(2) : (+v).toFixed(dg));
  const eqn = (title, math, text, ref) => `<div class="eqn"><h4>${title}</h4>${math}<p>${text}</p>${ref ? `<div class="ref">${ref}</div>` : ""}</div>`;

  function liveTables(D) {
    const rows = [];
    if (D.validation) D.validation.checks.forEach(c => rows.push([c.name, `${c.reference.toFixed(8)} M`, `${c.computed.toFixed(8)} M`, c.rel_error.toExponential(2), c.passed ? `<span class="badge pass">PASS</span>` : `<span class="badge fail">FAIL</span>`]));
    const val = rows.length ? `<table><tr><th>Check</th><th>Exact</th><th>Computed</th><th>Relative error</th><th></th></tr>${rows.map(r => `<tr>${r.map(x => `<td>${x}</td>`).join("")}</tr>`).join("")}</table>`
      : `<p class="muted">Not run in this session yet. Open the Validation view and press Run to fill this table with live numbers.</p>`;
    const b = D.bh;
    const bh = b ? `<table><tr><th>Quantity (a* = ${b.spin.toFixed(3)})</th><th>Value</th><th>Formula</th></tr>
      ${[["Outer horizon r+", b.r_plus, "horizons"], ["Inner horizon r−", b.r_minus, "horizons"], ["Ergosphere at the equator", b.ergosphere_equator, "stationary limit"],
        ["Prograde photon orbit", b.photon_orbit_prograde, "r_ph, upper sign"], ["Retrograde photon orbit", b.photon_orbit_retrograde, "r_ph, lower sign"],
        ["Prograde ISCO", b.isco_prograde, "BPT Z1, Z2"], ["Retrograde ISCO", b.isco_retrograde, "BPT Z1, Z2"],
        ["Prograde critical b", b.b_crit_prograde, "|ξ(r_ph)|"], ["Retrograde critical b", b.b_crit_retrograde, "|ξ(r_ph)|"]].map(([k, v, f]) => `<tr><td>${k}</td><td>${fmt(v)} M</td><td class="muted">${f}</td></tr>`).join("")}</table>` : "";
    const s = D.shadow;
    const sh = s ? `<table><tr><th>Latest shadow trace</th><th>Value</th></tr>
      ${[["Spin, inclination", `${s.spin}, ${s.inclination}°`], ["Photons traced", s.n_rays.toLocaleString()], ["Captured / escaped", `${s.counts.CAPTURED || 0} / ${s.counts.ESCAPED || 0}`],
        ["Shadow area, numerical", `${fmt(s.shadow_area_numerical, 3)} M²`], ["Shadow area, analytic curve", `${fmt(s.shadow_area_analytic, 3)} M²`], ["Relative difference", `${(100 * s.area_rel_diff).toFixed(2)} %`],
        ["Max null-constraint error (escaped)", fmt(s.max_null_error_escaped)], ["Runtime", `${fmt(s.runtime_s, 2)} s`]].map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("")}</table>` : "";
    const bm = D.beam;
    const beam = bm ? `<table><tr><th>Latest photon beam</th><th>Value</th></tr>
      ${[["Photons", bm.n_photons], ["Captured / escaped", `${bm.captured} / ${bm.escaped}`], ["Accepted integration steps", bm.accepted_steps.toLocaleString()], ["Engine time", `${fmt(bm.runtime_s, 2)} s`], ["Max null error (escaped)", fmt(bm.max_null_error_escaped)]].map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("")}</table>` : "";
    return { val, bh, sh, beam };
  }

  function render(host, D) {
    const T = liveTables(D);
    host.innerHTML = `<article class="about">
      <header class="about-hero">
        <div class="eyebrow">About the project</div>
        <h1>KerrRay</h1>
        <p class="lede">A numerical relativistic ray-tracing engine that follows individual photons through the curved spacetime of a spinning black hole, and reconstructs what an observer would see from nothing but Einstein's equations.</p>
        <div class="byline"><div class="avatar">JP</div><div><b>Joel Prince</b><span>An independent research project, built out of curiosity for aerospace and the physics of spacetime.</span></div></div>
      </header>

      <section><h2>What the project shows</h2>
        <div class="cards3">
          <div class="c"><b>Light bends and is captured</b><p>Every photon is integrated along its exact null geodesic until it falls through the event horizon or escapes to a distant sphere.</p></div>
          <div class="c"><b>The shadow is computed, not drawn</b><p>A camera fires one photon per pixel. Pixels whose photons fall in form the shadow. Its edge matches the exact analytic curve.</p></div>
          <div class="c"><b>Spin reshapes everything</b><p>Photons moving with the rotation can pass much closer than photons moving against it, so the shadow shifts and flattens on one side.</p></div>
          <div class="c"><b>Frame dragging</b><p>The off-diagonal metric term drags orbits around the hole, which the geodesic and animation views make visible.</p></div>
          <div class="c"><b>Numerical honesty</b><p>Every result reports its own error: the null constraint, energy and angular momentum drift, and the Carter constant.</p></div>
          <div class="c"><b>Validated first</b><p>Before any spinning results, the engine recovers the Schwarzschild photon sphere 3M and critical impact parameter 3√3M from traced photons.</p></div>
        </div>
      </section>

      <section><h2>How it works</h2>
        <div class="pipeline">${["Kerr metric", "Hamiltonian geodesics", "Adaptive RK45", "Capture / escape", "Image and analysis"].map((x, i) => `<div class="step"><span>${i + 1}</span>${x}</div>`).join('<div class="arrow">→</div>')}</div>
        <p>Units are geometric, G = c = 1, with lengths in units of the black-hole mass M. Coordinates are Boyer-Lindquist (t, r, θ, φ) with metric signature (−, +, +, +). Spin is the dimensionless a* = a / M with |a*| &lt; 1.</p>
      </section>

      <section><h2>Formulas used by the engine</h2>
        <p class="muted">Each formula below is implemented in the code and checked there. The symbolic checks use SymPy. The numerical checks run in the automated test suite.</p>
        <div class="eqns">
          ${eqn("1. Kerr metric in Boyer-Lindquist coordinates", F.sigma + F.metric, "The spacetime of an uncharged rotating black hole. Setting a = 0 recovers Schwarzschild. The code implements the covariant and contravariant components and their derivatives in closed form.", "Bardeen, Press &amp; Teukolsky 1972, ApJ 178, 347; Misner, Thorne &amp; Wheeler 1973, §33")}
          ${eqn("2. Horizons and ergosphere", F.horizons, "Photons are classified as captured when they reach r ≤ r₊ + ε with ε = 10⁻⁶ M. The ergosphere is where no observer can stay at rest.", "")}
          ${eqn("3. Hamiltonian form of the null geodesic equation", F.hamiltonian + F.eom, "The engine integrates eight first-order equations for position and covariant momentum. Only r and θ derivatives of the inverse metric are non-zero, so p_t and p_φ are exactly constant. H = 0 is the null constraint and is monitored along every ray.", "Chosen over the second-order Christoffel form for exact conservation of E and L_z and a cleaner constraint check; both forms are implemented and cross-checked.")}
          ${eqn("4. Conserved quantities", F.conserved, "Energy, axial angular momentum and the Carter constant. Their drift along each ray is the main measure of numerical quality.", "Carter 1968, Phys. Rev. 174, 1559")}
          ${eqn("5. Carter potentials", F.potentials, "With ξ = L_z / E and η = Q / E². Radial and polar motion separate, which is what makes the photon orbits below solvable.", "")}
          ${eqn("6. Spherical photon orbits", F.spherical, "Solving R = R′ = 0. The engine derives these with SymPy rather than typing them in, and the result matches these published forms to rounding error.", "Bardeen 1973; Teo 2003, Gen. Rel. Grav. 35, 1909")}
          ${eqn("7. Equatorial photon orbit and critical impact parameter", F.photonOrbit, "Upper sign prograde, lower sign retrograde. Photons with impact parameter below b_c are captured.", "Bardeen, Press &amp; Teukolsky 1972, eq. 2.18")}
          ${eqn("8. Innermost stable circular orbit", F.isco, "Also located independently by finding the minimum of circular-orbit energy; both methods agree.", "Bardeen, Press &amp; Teukolsky 1972, eq. 2.21")}
          ${eqn("9. Keplerian angular velocity", F.kepler, "Used for the emitting disk in the redshift model.", "Bardeen, Press &amp; Teukolsky 1972, eq. 2.16")}
          ${eqn("10. Camera and celestial coordinates", F.zamo + F.celestial, "The camera is a zero-angular-momentum observer at r_o = 1000 M. Each pixel (α, β) sets a local photon direction, which the tetrad converts to Boyer-Lindquist momentum, and the photon is traced backwards. For a distant observer the shadow edge is the Bardeen curve on the right.", "Bardeen 1973; Cunha &amp; Herdeiro 2018, Gen. Rel. Grav. 50, 42")}
          ${eqn("11. Adaptive integration", F.rk45, "Dormand-Prince 5(4) with the standard tableau, verified against SciPy. Fixed-step RK4 and SciPy DOP853 are available for comparison.", "Dormand &amp; Prince 1980; Hairer, Nørsett &amp; Wanner 1993, Table 5.2")}
          ${eqn("12. Gravitational and Doppler redshift", F.redshift, "For an emitter on a circular orbit and an observer at infinity. The simplified disk model uses I_obs = g³ I_em at fixed observed frequency.", "Cunningham 1975; Luminet 1979")}
          ${eqn("13. Light deflection in Schwarzschild", F.deflection, "The exact integral is evaluated numerically and compared with deflections measured from traced photons. 4M/b is only the weak-field limit.", "Darwin 1959; Bozza 2002, Phys. Rev. D 66, 103001")}
          ${eqn("14. Shadow area comparison", F.area, "The analytic area comes from the shoelace formula over the Bardeen curve. The numerical area counts captured pixels, each of area (2F / n)², where F is the field of view and n the resolution.", "")}
        </div>
      </section>

      <section><h2>Calculations in this session</h2>
        <p class="muted">These tables are filled live from the engine as you use the app. Nothing here is typed in.</p>
        <h3>Schwarzschild validation</h3>${T.val}
        <div class="two">${T.bh ? `<div><h3>Black hole</h3>${T.bh}</div>` : ""}<div>${T.sh ? `<h3>Shadow</h3>${T.sh}` : ""}${T.beam ? `<h3>Photon beam</h3>${T.beam}` : ""}</div></div>
      </section>

      <section><h2>How the engine is verified</h2>
        <ul>
          <li>The metric times its inverse equals the identity symbolically, and two independently published forms of the metric are proven identical with SymPy.</li>
          <li>The Ricci tensor of the implemented metric vanishes numerically at random points, which confirms it is the vacuum Kerr solution.</li>
          <li>The Christoffel symbols and all metric derivatives match symbolic differentiation to about 10⁻¹⁴.</li>
          <li>The integrator's tableau matches SciPy's, RK4 shows fourth-order convergence, and batched and single-ray integration agree.</li>
          <li>Energy and angular momentum are conserved exactly by construction; the Carter constant and null constraint stay near 10⁻⁷ or better on escaping rays.</li>
          <li>The photon sphere and critical impact parameter are recovered to about 10⁻⁷ relative error, and numerical shadows match the analytic edge to within pixel size.</li>
        </ul>
      </section>

      <section><h2>Limitations and approximations</h2>
        <ul>
          <li><b>Coordinates near the horizon.</b> Boyer-Lindquist coordinates are singular at the horizon, so a ray's accuracy drops in its last steps before capture. This does not change its classification.</li>
          <li><b>Finite observer distance.</b> The camera sits at 1000 M, not infinity, so shadows differ from the analytic curve by order M / r_o.</li>
          <li><b>Polar axis.</b> The camera tetrad is singular on the spin axis, so inclination 0° is moved to 0.001°.</li>
          <li><b>Disk model.</b> The accretion disk is a simplified, optically thin, geometrically thin emitter with a power-law emissivity. It is not a magnetohydrodynamic simulation.</li>
          <li><b>Fixed-step integration at the horizon.</b> The covariant radial momentum diverges as a photon approaches the horizon in these coordinates. Fixed-step RK4 cannot shrink its step there, so its stages overshoot and captured photons end as out of domain. Adaptive RK45 resolves the approach and is used for every shadow.</li>
          <li><b>Hardware.</b> Everything runs on a laptop CPU. The compiled Numba engine traces a 64 by 64 shadow about 14 times faster than the NumPy reference with identical results. GPU acceleration is optional future work.</li>
          <li><b>No claims beyond the evidence.</b> This is a computational study of known physics. It makes no observational or novelty claims.</li>
        </ul>
      </section>

      <section class="credit"><div class="avatar big">JP</div><div><h2>Joel Prince</h2><p>KerrRay is a personal research project by Joel Prince, started out of curiosity about aerospace, gravity and how light behaves in the most extreme environments in the universe. It is built physics-first: validate the equations, then make it fast, then make it beautiful.</p></div></section>
    </article>`;
  }
  return { render };
})();
