/* KerrRay front end. Four views (shadow, geodesic, spin sweep, validation),
   a controls rail, a results panel and an optional console. Every number shown
   is computed by the engine through POST /api/<endpoint>. */
"use strict";

(() => {
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const STATE_NAMES = { 1: "ESCAPED", 2: "CAPTURED", 3: "MAX_AFFINE_PARAMETER", 4: "NUMERICAL_FAILURE", 5: "OUT_OF_DOMAIN", 6: "DISK_HIT" };
  const RUN_LABEL = { shadow: "Trace shadow", geodesic: "Integrate photon", sweep: "Compute sweep", animate: "Launch photons", validation: "Run validation", about: "Run validation" };

  /* ---------------------------------------------------------------- state */
  const S = {
    view: "shadow", spin: 0.9, inclination: 60, resolution: 64, fov: 10, observer_radius: 1000,
    method: "rk45", rtol: 1e-8, b: 3.0, prograde: true, r0: 50, tol: 1e-6,
    display: "sky", overlay: true, hold: false,
    beam_n: 48, beam_b: 10, speed: 1, trail: "long", backend: "numba", geo_method: "rk45",
  };
  const data = { shadow: null, geodesic: [], sweep: null, validation: null, bh: null, beam: null };
  let stale = false;
  const running = new Map();  // label -> start time, for the status pill

  /* ---------------------------------------------------------------- formatting */
  function num(v, digits = 4) {
    if (v === null || v === undefined || Number.isNaN(v)) return "—";
    if (typeof v !== "number") return String(v);
    if (!Number.isFinite(v)) return v > 0 ? "∞" : "−∞";
    if (Number.isInteger(v) && Math.abs(v) < 1e7) return v.toLocaleString("en-US");
    const a = Math.abs(v);
    if (a !== 0 && (a >= 1e5 || a < 1e-3)) { const [m, e] = v.toExponential(2).split("e"); return `${m}e${e.replace("+", "")}`; }
    return v.toFixed(digits);
  }
  const sci = v => { const [m, e] = v.toExponential(0).split("e"); return `${m}e${e.replace("+", "")}`; };

  /* ---------------------------------------------------------------- api */
  async function api(name, params = {}) {
    const r = await fetch(`/api/${name}`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(params) });
    const d = await r.json().catch(() => ({ error: `HTTP ${r.status}` }));
    if (!r.ok || d.error) throw new Error(d.error || `HTTP ${r.status}`);
    return d;
  }
  let statusTimer = null, statusDetail = {};
  function paintStatus() {
    const st = $("#status");
    if (!running.size) { clearInterval(statusTimer); statusTimer = null; $("#canvas-host").classList.remove("busy"); return; }
    const [label, t0] = [...running.entries()].pop();
    st.className = "busy";
    const extra = statusDetail[label] ? ` · ${statusDetail[label]}` : "";
    $("#status-text").textContent = `${label}${extra} · ${((performance.now() - t0) / 1000).toFixed(1)} s`;
  }
  async function task(label, fn) {
    running.set(label, performance.now()); $("#canvas-host").classList.add("busy");
    if (!statusTimer) statusTimer = setInterval(paintStatus, 100);
    paintStatus();
    const t0 = performance.now();
    try {
      const r = await fn();
      if (r !== null) { $("#status").className = ""; $("#status-text").textContent = `Done in ${((performance.now() - t0) / 1000).toFixed(1)} s`; }
      return r;
    } catch (e) {
      $("#status").className = "error"; $("#status-text").textContent = "Error";
      log(`<span class="err">${esc(label)}: ${esc(e.message)}</span>`); openConsole(); return null;
    } finally { running.delete(label); delete statusDetail[label]; paintStatus(); }
  }
  // Start a background job and poll it, reporting progress until it ends.
  const jobIds = {};
  async function job(kind, params, onProgress) {
    if (jobIds[kind]) api("job_cancel", { id: jobIds[kind] }).catch(() => {});
    const { id } = await api("job_start", { kind, params });
    jobIds[kind] = id;
    for (;;) {
      await new Promise(r => setTimeout(r, 180));
      const p = await api("job_poll", { id });
      if (jobIds[kind] !== id) { api("job_cancel", { id }).catch(() => {}); return null; }
      if (onProgress && p.progress) onProgress(p.progress, p.elapsed_s);
      if (p.status === "done") { delete jobIds[kind]; return p.result; }
      if (p.status === "cancelled") return null;
      if (p.status === "error") { delete jobIds[kind]; throw new Error(p.error); }
    }
  }

  /* ---------------------------------------------------------------- controls */
  const CTRL = {
    spin: { label: "Spin a*", min: -0.999, max: 0.999, step: 0.001, presets: [0, 0.5, 0.9, 0.99], fmt: v => v.toFixed(3) },
    inclination: { label: "Inclination", min: 0, max: 180, step: 1, presets: [5, 30, 60, 90], unit: "°", fmt: v => v.toFixed(0) },
    fov: { label: "Field of view", min: 3, max: 30, step: 0.5, unit: " M", fmt: v => v.toFixed(1), hint: "Half-width of the image plane." },
    b: { label: "Impact parameter b", min: 0, max: 12, step: 0.001, unit: " M", fmt: v => v.toFixed(3) },
    beam_b: { label: "Beam half-width", min: 3, max: 20, step: 0.5, unit: " M", fmt: v => v.toFixed(1), hint: "Photons start on a line 30 M to the right, spread over plus and minus this height." },
  };
  function slider(key) {
    const c = CTRL[key];
    const presets = c.presets ? `<div class="presets" data-presets="${key}">${c.presets.map(p => `<button data-v="${p}">${p}${c.unit === "°" ? "°" : ""}</button>`).join("")}</div>` : "";
    return `<div class="ctl"><div class="ctl-head"><label>${c.label}</label><span class="val"><input data-num="${key}" inputmode="decimal">${c.unit || ""}</span></div>
      <input type="range" data-range="${key}" min="${c.min}" max="${c.max}" step="${c.step}">${presets}${c.hint ? `<div class="hint">${c.hint}</div>` : ""}</div>`;
  }
  function seg(key, label, opts) {
    return `<div class="ctl"><div class="ctl-head"><label>${label}</label></div><div class="seg" data-seg="${key}">${opts.map(([v, l]) => `<button data-v="${v}">${l}</button>`).join("")}</div></div>`;
  }
  function presetRow(key, label, values, fmt = v => v) {
    return `<div class="ctl"><div class="ctl-head"><label>${label}</label><span class="val" data-show="${key}"></span></div><div class="presets" data-presets="${key}">${values.map(v => `<button data-v="${v}">${fmt(v)}</button>`).join("")}</div></div>`;
  }
  const PANELS = {
    shadow: () => `
      <div class="group"><h4>Black hole</h4>${slider("spin")}</div>
      <div class="group"><h4>Camera</h4>${slider("inclination")}${slider("fov")}${presetRow("resolution", "Resolution", [32, 48, 64, 96, 128, 160])}</div>
      <div class="group"><h4>Solver</h4>${seg("backend", "Engine", [["numba", "Numba compiled"], ["numpy", "NumPy (live)"]])}
        ${seg("rtol", "Relative tolerance", [[1e-6, "1e-6"], [1e-8, "1e-8"], [1e-10, "1e-10"]])}
        <div class="hint">Adaptive Dormand-Prince RK45. Both engines give identical results; Numba is about 14 times faster, NumPy shows every photon moving in and out while it traces.</div></div>`,
    geodesic: () => `
      <div class="group"><h4>Black hole</h4>${slider("spin")}</div>
      <div class="group"><h4>Photon</h4>${slider("b")}
        <div class="presets" style="margin:-4px 0 12px"><button data-act="bcrit" data-f="0.99">0.99 b<sub>c</sub></button><button data-act="bcrit" data-f="1.001">1.001 b<sub>c</sub></button><button data-act="bcrit" data-f="1.3">1.3 b<sub>c</sub></button></div>
        ${seg("prograde", "Direction", [["true", "Prograde"], ["false", "Retrograde"]])}
        ${presetRow("r0", "Launch radius", [20, 50, 200], v => `${v} M`)}</div>
      <div class="group"><h4>Solver</h4>${seg("geo_method", "Integrator", [["rk45", "RK45 adaptive"], ["rk4", "RK4 fixed step"]])}
        <div class="hint">Fixed-step RK4 cannot shrink its step where the radial momentum diverges at the horizon, so captured photons end as out of domain. Adaptive RK45 resolves it.</div></div>
      <div class="group"><h4>Plot</h4><label class="toggle">Overlay previous photons<input type="checkbox" data-check="hold"><span class="sw"></span></label>
        <button class="ghost" data-act="clear-rays">Clear photons</button></div>
      <div class="hint">The photon is re-integrated whenever a control changes.</div>`,
    sweep: () => `
      <div class="group"><h4>Black hole</h4>${slider("spin")}<div class="hint">The dashed marker shows the selected spin on each curve.</div></div>`,
    animate: () => `
      <div class="group"><h4>Black hole</h4>${slider("spin")}</div>
      <div class="group"><h4>Photon beam</h4>${presetRow("beam_n", "Evenly spaced photons", [24, 48, 96, 144], v => v)}${slider("beam_b")}</div>
      <div class="hint">Every photon is integrated by the engine. The animation replays the paths in the coordinate time of a distant observer, so photons falling in slow down near the horizon. Eight extra photons sit just either side of the two critical impact parameters. Scroll on the view to zoom. Changing a control recomputes the beam in about a second.</div>`,
    about: () => "",
    validation: () => `
      <div class="group"><h4>Solver</h4>${seg("rtol", "Relative tolerance", [[1e-8, "1e-8"], [1e-9, "1e-9"], [1e-10, "1e-10"]])}
        ${presetRow("tol", "Bisection tolerance", [1e-5, 1e-6, 1e-7], sci)}</div>
      <div class="hint">Recovers the Schwarzschild photon sphere and critical impact parameter by bisecting on traced photons. Takes about 20 to 60 seconds.</div>`,
  };
  function renderControls() { $("#controls").innerHTML = PANELS[S.view](); sync(); }
  function sync() {
    $$("[data-range]").forEach(el => { const k = el.dataset.range; el.value = S[k]; paintRange(el); });
    $$("[data-num]").forEach(el => { if (document.activeElement !== el) el.value = CTRL[el.dataset.num].fmt(S[el.dataset.num]); });
    $$("[data-seg]").forEach(sg => $$("button", sg).forEach(b => b.classList.toggle("on", String(S[sg.dataset.seg]) === b.dataset.v || Number(S[sg.dataset.seg]) === Number(b.dataset.v))));
    $$("[data-presets]").forEach(p => $$("button", p).forEach(b => b.classList.toggle("on", Number(b.dataset.v) === Number(S[p.dataset.presets]))));
    $$("[data-check]").forEach(el => el.checked = !!S[el.dataset.check]);
    $$("[data-show]").forEach(el => { const k = el.dataset.show; el.textContent = k === "resolution" ? `${S[k]}×${S[k]}` : k === "tol" ? sci(S[k]) : k === "beam_n" ? `${S[k]}` : `${S[k]} M`; });
    $("#run-label").textContent = RUN_LABEL[S.view];
  }
  function paintRange(el) { const p = (el.value - el.min) / (el.max - el.min) * 100; el.style.setProperty("--p", `${p}%`); }

  let geoTimer = null, bhTimer = null, shadowTimer = null;
  function set(key, value) {
    S[key] = value;
    if (["spin", "inclination", "fov", "resolution", "method", "rtol"].includes(key)) stale = true;
    if (key === "spin") { clearTimeout(bhTimer); bhTimer = setTimeout(refreshBH, 120); if (data.sweep) drawSweep(); }
    if (["display", "overlay"].includes(key) && data.shadow) { drawShadow(); renderChips(); }
    sync();
    if (S.view === "geodesic" && ["spin", "b", "prograde", "r0", "rtol", "geo_method"].includes(key)) { clearTimeout(geoTimer); geoTimer = setTimeout(() => runGeodesic(), 250); }
    if (S.view === "shadow" && ["spin", "inclination", "fov", "resolution", "backend", "rtol"].includes(key)) { clearTimeout(shadowTimer); shadowTimer = setTimeout(() => runShadow(), 450); }
    if (S.view === "animate" && ["spin", "beam_n", "beam_b"].includes(key)) { clearTimeout(geoTimer); geoTimer = setTimeout(() => runBeam(), 250); }
    if (key === "speed" && anim) anim.setSpeed(value);
    if (key === "trail" && anim) anim.setTrail(value);
  }
  document.addEventListener("input", e => {
    const t = e.target;
    if (t.dataset.range) { paintRange(t); const k = t.dataset.range; S[k] = parseFloat(t.value); const n = $(`[data-num="${k}"]`); if (n) n.value = CTRL[k].fmt(S[k]); }
  });
  document.addEventListener("change", e => {
    const t = e.target;
    if (t.dataset.range) set(t.dataset.range, parseFloat(t.value));
    if (t.dataset.num) { const k = t.dataset.num, c = CTRL[k], v = parseFloat(t.value); if (Number.isFinite(v)) set(k, Math.min(c.max, Math.max(c.min, v))); else sync(); }
    if (t.dataset.check) { set(t.dataset.check, t.checked); renderChips(); }
  });
  document.addEventListener("keydown", e => { if (e.target.dataset && e.target.dataset.num && e.key === "Enter") e.target.blur(); });
  document.addEventListener("click", e => {
    const sb = e.target.closest("[data-seg] button");
    if (sb) { const k = sb.closest("[data-seg]").dataset.seg; set(k, k === "prograde" ? sb.dataset.v === "true" : ["method", "backend", "geo_method"].includes(k) ? sb.dataset.v : Number(sb.dataset.v)); return; }
    const pb = e.target.closest("[data-presets] button");
    if (pb) { set(pb.closest("[data-presets]").dataset.presets, Number(pb.dataset.v)); return; }
    const act = e.target.closest("[data-act]");
    if (act) ACTIONS[act.dataset.act]?.(act);
  });

  /* ---------------------------------------------------------------- figures */
  const figs = {};
  let figMode = "tip";
  function figure(key) {
    if (!figs[key]) {
      const host = document.createElement("div"); host.className = "fig"; $("#canvas-host").appendChild(host);
      figs[key] = { host, fig: null };
    }
    return figs[key];
  }
  function plotFigure(key) {
    const f = figure(key);
    if (!f.fig) { f.host.innerHTML = ""; f.fig = new Plot.Figure(f.host, { onHover: hover }); f.fig.setMode(figMode); }
    return f.fig;
  }
  function showStage() {
    const f = figure(S.view);
    if (!f.fig && !["validation", "about"].includes(S.view) && !f.host.innerHTML) f.host.innerHTML = `<div class="placeholder"><b>${RUN_LABEL[S.view]}</b>Press Run to compute this view.</div>`;
    for (const [k, g] of Object.entries(figs)) g.host.classList.toggle("active", k === S.view);
    if (f.fig) f.fig.draw();
  }
  function hover(hit, xy) {
    $("#readout").textContent = hit ? hit.text.replace(/\n/g, "   ·   ") : xy ? `x ${Plot.fmtData(xy[0])}   y ${Plot.fmtData(xy[1])}` : "";
  }
  $$("#stagebar [data-mode]").forEach(b => b.addEventListener("click", () => {
    figMode = b.dataset.mode; $$("#stagebar [data-mode]").forEach(x => x.classList.toggle("on", x === b));
    Object.values(figs).forEach(f => f.fig && f.fig.setMode(figMode));
  }));

  /* ---------------------------------------------------------------- shadow */
  const SKY = [["#143a52", "#23658c"], ["#4d3314", "#8a5c24"], ["#163f2c", "#2a7250"], ["#2a1f4a", "#4f3a8f"]]
    .map(p => p.map(h => [1, 3, 5].map(j => parseInt(h.slice(j, j + 2), 16))));
  function shadowPixels(d, mode) {
    const n = d.resolution, px = new Uint8ClampedArray(n * n * 4);
    let smin = Infinity, smax = -Infinity, emin = Infinity, emax = -Infinity;
    for (let i = 0; i < n * n; i++) { smin = Math.min(smin, d.steps[i]); smax = Math.max(smax, d.steps[i]); emin = Math.min(emin, d.log_null_error[i]); emax = Math.max(emax, d.log_null_error[i]); }
    const thObs = Math.PI - d.inclination * Math.PI / 180, step = Math.PI / 12;
    for (let i = 0; i < n * n; i++) {
      const st = d.state[i]; let c;
      if (mode === "steps") c = Plot.inferno((d.steps[i] - smin) / Math.max(1, smax - smin));
      else if (mode === "error") c = Plot.inferno((d.log_null_error[i] - emin) / Math.max(1e-9, emax - emin));
      else if (st === 2) c = [0, 0, 0];
      else if (st !== 1) c = [255, 80, 80];
      else if (mode === "capture") c = [30, 37, 48];
      else {
        // sky grid centred on the point directly behind the hole (theta = pi - i, phi = pi)
        const th = d.theta_end[i], dth = th - thObs;
        let dph = d.phi_end[i] - Math.PI; dph -= 2 * Math.PI * Math.round(dph / (2 * Math.PI));
        const u = dph * Math.sin(th), q = (u >= 0 ? 0 : 1) + (dth >= 0 ? 0 : 2);
        c = SKY[q][((Math.floor(u / step) + Math.floor(dth / step)) % 2 + 2) % 2];
      }
      px.set([c[0], c[1], c[2], 255], i * 4);
    }
    return { px, smin, smax, emin, emax };
  }
  let live = null;
  async function runShadow() {
    const p = { spin: S.spin, inclination: S.inclination, resolution: S.resolution, fov: S.fov, observer_radius: S.observer_radius, method: "rk45", rtol: S.rtol, backend: S.backend };
    const label = "Tracing photons";
    log(`<span class="dim">Tracing ${(S.resolution ** 2).toLocaleString()} photons: a* = ${S.spin}, i = ${S.inclination}°, RK45 on ${S.backend}, rtol ${sci(S.rtol)}</span>`);
    live = { ...p, done: 0, total: p.resolution ** 2, state: null, elapsed: 0 };
    stale = false; sync(); drawLive(); if (S.view === "shadow") renderResults();
    const d = await task(label, () => job("shadow", p, (pg, el) => {
      if (!live || live.resolution !== p.resolution || live.spin !== p.spin || live.inclination !== p.inclination) return;
      Object.assign(live, { done: pg.done || 0, total: pg.total || live.total, state: pg.state || live.state, r: pg.r || live.r, out: pg.out || live.out, fraction: pg.fraction ?? live.fraction, elapsed: el });
      statusDetail[label] = `${Math.floor(100 * (live.fraction || 0))}%`;
      drawLive(); if (S.view === "shadow") renderResults();
    }));
    if (!d) return;
    live = null; data.shadow = d;
    drawShadow(); renderChips(); if (S.view === "shadow") renderResults();
    log(`shadow: ${d.n_rays} rays, captured ${d.counts.CAPTURED || 0}, area ${num(d.shadow_area_numerical)} vs ${num(d.shadow_area_analytic)} M², ${num(d.runtime_s, 2)} s`);
  }
  // Partial image while the job runs: each pixel appears when its photon is classified.
  function drawLive() {
    if (!live) return;
    // Running photons are shaded by where they are now: warm and brightening on the way in,
    // cool on the way out. Finished photons show their outcome.
    const n = live.resolution, F = live.fov, px = new Uint8ClampedArray(n * n * 4), st = live.state, R = live.r, O = live.out, ro = live.observer_radius;
    for (let i = 0; i < n * n; i++) {
      let c;
      if (st && st[i]) c = st[i] === 2 ? [0, 0, 0] : st[i] === 1 ? [36, 64, 84] : [255, 80, 80];
      else if (R) {
        const u = Math.max(0, Math.min(1, R[i] / ro));
        c = O && O[i] ? [Math.round(40 + 60 * (1 - u)), Math.round(110 + 90 * (1 - u)), Math.round(150 + 90 * (1 - u))] : [Math.round(40 + 215 * (1 - u) ** 3), Math.round(30 + 110 * (1 - u) ** 3), Math.round(30 + 20 * (1 - u) ** 3)];
      } else c = [22, 27, 35];
      px.set([c[0], c[1], c[2], 255], i * 4);
    }
    const fig = plotFigure("shadow"); fig.clear();
    const pct = Math.floor(100 * (live.fraction || 0));
    const ax = fig.addAxes([0, 0, 1, 1], { title: `Tracing ${live.total.toLocaleString()} photons ... ${pct}%   (orange: travelling in, blue: travelling out, black: captured)`, xlabel: "α  [M]", ylabel: "β  [M]", equal: "image", grid: false });
    ax.image(px, n, n, [-F, F, -F, F]);
    ax.xlim = [-F, F]; ax.ylim = [-F, F]; ax.home = [[-F, F], [-F, F]];
    fig.draw(); if (S.view === "shadow") showStage();
  }
  function drawShadow() {
    const d = data.shadow; if (!d) return;
    const fig = plotFigure("shadow"), old = fig.axes[0], keep = old && old.fovKey === d.fov ? [old.xlim, old.ylim] : null;
    fig.clear();
    const mode = S.display, { px, smin, smax, emin, emax } = shadowPixels(d, mode), F = d.fov, n = d.resolution;
    const ax = fig.addAxes([0, 0, 1, 1], {
      title: `a* = ${d.spin}   ·   i = ${d.inclination}°   ·   ${n} × ${n} photons`,
      xlabel: "α  [M]", ylabel: "β  [M]", equal: "image", grid: false, legend: "southwest",
      colorbar: mode === "steps" ? { cmap: Plot.inferno, clim: [smin, smax], label: "integration steps" } : mode === "error" ? { cmap: Plot.inferno, clim: [emin, emax], label: "log10 null-constraint error" } : null,
    });
    ax.fovKey = d.fov;
    ax.image(px, n, n, [-F, F, -F, F], {
      lookup: (x, y) => {
        const col = Math.floor((x + F) / (2 * F) * n), row = Math.floor((F - y) / (2 * F) * n);
        if (col < 0 || row < 0 || col >= n || row >= n) return null;
        const i = row * n + col, pc = 2 * F / n, cx = -F + (col + 0.5) * pc, cy = F - (row + 0.5) * pc;
        const sky = d.state[i] === 1 ? `\nlands on sky at θ ${num(d.theta_end[i] * 180 / Math.PI, 1)}°, φ ${num(d.phi_end[i] * 180 / Math.PI, 1)}°` : "";
        return { x: cx, y: cy, text: `α ${num(cx, 3)}  β ${num(cy, 3)}\n${STATE_NAMES[d.state[i]]} after ${d.steps[i]} steps\nnull error 1e${num(d.log_null_error[i], 1)}${sky}` };
      },
    });
    if (S.overlay) ax.plot(d.analytic_alpha, d.analytic_beta, { color: "#ffffff", halo: "rgba(0,0,0,0.8)", width: 1.4, dash: [5, 4], name: "Analytic shadow edge (Bardeen 1973)" });
    ax.xlim = keep ? keep[0] : [-F, F]; ax.ylim = keep ? keep[1] : [-F, F]; ax.home = [[-F, F], [-F, F]];
    fig.draw();
  }

  /* ---------------------------------------------------------------- geodesic */
  async function runGeodesic() {
    const p = { spin: S.spin, b: S.b, prograde: S.prograde, r0: S.r0, method: S.geo_method, rtol: Math.min(S.rtol, 1e-9) };
    const d = await task("Integrating photon", () => api("geodesic", p));
    if (!d) return;
    if (!S.hold || (data.geodesic.length && data.geodesic[0].spin !== d.spin)) data.geodesic = [];
    data.geodesic.push(d);
    drawGeodesic(); if (S.view === "geodesic") renderResults();
    log(`geodesic: b = ${d.b}, ${d.prograde ? "prograde" : "retrograde"} → ${d.state}, ${num(d.turns, 2)} turns, null error ${num(d.max_null_error)}`);
  }
  function drawGeodesic() {
    const rays = data.geodesic; if (!rays.length) return;
    const fig = plotFigure("geodesic"), d0 = rays[0], a = d0.spin, rho = r => Math.sqrt(r * r + a * a);
    const prev = fig.axes[0] ? [fig.axes[0].xlim, fig.axes[0].ylim] : null;
    fig.clear();
    const ax = fig.addAxes([0, 0, 0.6, 1], { title: `Equatorial photon orbits   ·   a* = ${a}`, xlabel: "x  [M]", ylabel: "y  [M]", equal: true, legend: "southwest" });
    ax.circle(0, 0, rho(2.0), { fill: true, color: "rgba(179,136,255,0.08)" });
    ax.circle(0, 0, rho(2.0), { color: "#b388ff", dash: [5, 4], width: 1.1, name: "Ergosphere" });
    ax.circle(0, 0, rho(d0.photon_orbit), { color: "#7bd88f", dash: [2, 3], width: 1.1, name: "Photon orbit" });
    ax.circle(0, 0, rho(d0.r_plus), { fill: true, color: "#000000", edge: "#ff8a3d", name: "Event horizon" });
    const colors = ["#4cc9f0", "#ff8a3d", "#f7c948", "#ff6b8b", "#7bd88f", "#e8ecf2"];
    rays.forEach((d, k) => ax.plot(d.x, d.y, { color: colors[k % colors.length], width: 1.7, name: `b = ${d.b.toFixed(3)} ${d.prograde ? "pro" : "retro"} → ${d.state.toLowerCase()}` }));
    const R = 7; ax.xlim = prev && S.hold ? prev[0] : [-R, R]; ax.ylim = prev && S.hold ? prev[1] : [-R, R]; ax.home = [[-R, R], [-R, R]];
    const ax2 = fig.addAxes([0.6, 0, 0.4, 1], { title: "Null-constraint error along the path", xlabel: "affine parameter λ  [M]", ylabel: "|H| / E0²", ylog: true });
    rays.forEach((d, k) => ax2.plot(d.lam, d.null_error, { color: colors[k % colors.length], width: 1.3 }));
    fig.draw();
  }

  /* ---------------------------------------------------------------- sweep */
  async function runSweep() {
    const d = await task("Computing spin sweep", () => api("spin_sweep", { n: 160 }));
    if (!d) return;
    data.sweep = d; drawSweep(); if (S.view === "sweep") renderResults();
  }
  function drawSweep() {
    const d = data.sweep; if (!d) return;
    const fig = plotFigure("sweep"); fig.clear();
    const ax = fig.addAxes([0, 0, 0.52, 1], { title: "Characteristic radii", xlabel: "spin a*", ylabel: "r  [M]", legend: "northwest", xlim: [0, 1], ylim: [0, 10] });
    ax.plot(d.spin, d.isco_retro, { name: "ISCO retrograde", color: "#ff8a3d" });
    ax.plot(d.spin, d.isco_pro, { name: "ISCO prograde", color: "#4cc9f0" });
    ax.plot(d.spin, d.ph_retro, { name: "Photon orbit retro", color: "#ff8a3d", dash: [6, 4] });
    ax.plot(d.spin, d.ph_pro, { name: "Photon orbit pro", color: "#4cc9f0", dash: [6, 4] });
    ax.plot(d.spin, d.r_plus, { name: "Event horizon", color: "#e8ecf2" });
    const ax2 = fig.addAxes([0.52, 0, 0.48, 1], { title: "Critical impact parameter", xlabel: "spin a*", ylabel: "b_c  [M]", legend: "southwest", xlim: [0, 1], ylim: [2, 7] });
    ax2.plot(d.spin, d.b_retro, { name: "Retrograde", color: "#ff8a3d" });
    ax2.plot(d.spin, d.b_pro, { name: "Prograde", color: "#4cc9f0" });
    const s = Math.abs(S.spin);
    ax.plot([s, s], [0, 10], { color: "rgba(232,236,242,0.4)", width: 1, dash: [3, 3] });
    ax2.plot([s, s], [2, 7], { color: "rgba(232,236,242,0.4)", width: 1, dash: [3, 3] });
    fig.draw();
  }

  /* ---------------------------------------------------------------- animation */
  let anim = null;
  function ensureAnim() {
    const f = figure("animate");
    if (!anim) { f.host.innerHTML = ""; f.host.classList.add("anim-host"); anim = KerrAnim.create(f.host, { onState: () => { if (S.view === "animate") renderChips(); } }); anim.setSpeed(S.speed); anim.setTrail(S.trail); }
    return anim;
  }
  async function runBeam() {
    const d = await task("Integrating photon beam", () => api("photon_beam", { spin: S.spin, n: S.beam_n, b_max: S.beam_b }));
    if (!d) return;
    data.beam = d; const a = ensureAnim(); a.load(d);
    if (S.view === "animate") { a.play(); renderResults(); renderChips(); }
    log(`beam: ${d.n_photons} photons, captured ${d.captured}, escaped ${d.escaped}, ${d.accepted_steps} steps in ${num(d.runtime_s, 2)} s`);
  }

  /* ---------------------------------------------------------------- validation */
  let vlive = null;
  async function runValidation() {
    const label = "Validating";
    vlive = { photon_sphere: { status: "running" }, critical_b: { status: "pending" }, tol: S.tol, rtol: S.rtol };
    data.validation = null; drawValidation();
    const checks = [];
    for (const which of ["photon_sphere", "critical_b"]) {
      vlive[which].status = "running"; drawValidation();
      const r = await task(label, () => job("validate", { which, tol: S.tol, rtol: S.rtol }, (pg, el) => {
        if (!vlive || pg.check !== which) return;
        Object.assign(vlive[which], pg, { elapsed: el });
        statusDetail[label] = `${which === "photon_sphere" ? "photon sphere" : "critical b"} step ${pg.iteration}`;
        drawValidation();
      }));
      if (!r) { vlive = null; drawValidation(); return; }
      vlive[which] = { status: "done", result: r.checks[0] };
      checks.push(r.checks[0]); drawValidation();
      data.validationMeta = r;
    }
    const m = data.validationMeta;
    data.validation = { checks, method: m.method, rtol: m.rtol, tol: m.tol, pass_rel: m.pass_rel, runtime_s: checks.reduce((a, c) => a + c.runtime_s, 0) };
    vlive = null; drawValidation(); if (S.view === "validation") renderResults();
    checks.forEach(c => log(`${c.name}: computed ${c.computed.toFixed(8)} vs exact ${c.reference.toFixed(8)}, rel. error ${c.rel_error.toExponential(2)} <span class="${c.passed ? "ok" : "err"}">${c.passed ? "PASS" : "FAIL"}</span>`));
  }
  function liveCard(key, title, exact, lo0, hi0) {
    const v = vlive[key];
    if (v.status === "done") return cardHtml(v.result);
    if (v.status === "pending") return `<div class="vcard"><div class="top">${title}<span class="badge wait">QUEUED</span></div><div class="big dim">—</div><dl><dt>Exact value</dt><dd>${exact.toFixed(8)} M</dd></dl></div>`;
    const lo = v.lo ?? lo0, hi = v.hi ?? hi0, width = hi - lo, frac = Math.max(0, Math.min(1, Math.log10((hi0 - lo0) / Math.max(width, 1e-12)) / Math.log10((hi0 - lo0) / vlive.tol)));
    return `<div class="vcard live"><div class="top">${title}<span class="badge run">BISECTING</span></div>
      <div class="big">${(0.5 * (lo + hi)).toFixed(8)}<small>M</small></div>
      <div class="bar"><i style="width:${100 * frac}%;background:#ff8a3d"></i></div>
      <dl><dt>Bracket</dt><dd>[${lo.toFixed(7)}, ${hi.toFixed(7)}]</dd><dt>Bracket width</dt><dd>${width.toExponential(2)} M</dd>
      <dt>Iteration</dt><dd>${v.iteration ?? 0}</dd><dt>Exact value (for comparison)</dt><dd>${exact.toFixed(8)} M</dd><dt>Elapsed</dt><dd>${num(v.elapsed ?? 0, 1)} s</dd></dl></div>`;
  }
  const cardHtml = c => `<div class="vcard"><div class="top">${esc(c.name)}<span class="badge ${c.passed ? "pass" : "fail"}">${c.passed ? "PASS" : "FAIL"}</span></div>
      <div class="big">${c.computed.toFixed(8)}<small>M</small></div>
      <dl><dt>Exact value</dt><dd>${c.reference.toFixed(8)} M</dd><dt>Relative error</dt><dd>${c.rel_error.toExponential(2)}</dd>
      <dt>Bisection bracket</dt><dd>${c.bracket.toExponential(2)} M</dd><dt>Iterations</dt><dd>${c.iterations}</dd><dt>Time</dt><dd>${c.runtime_s.toFixed(1)} s</dd></dl></div>`;
  function drawValidation() {
    const f = figure("validation"), d = data.validation;
    if (vlive) {
      f.host.innerHTML = `<div class="report"><h1>Schwarzschild validation</h1>
        <p>Running now. Each step integrates a photon and asks one question: did it fall in or escape? The bracket around the answer halves every step.</p>
        <div class="vcards">${liveCard("photon_sphere", "Photon sphere radius", 3, 2.5, 3.5)}${liveCard("critical_b", "Critical impact parameter", 3 * Math.sqrt(3), 4, 6)}</div></div>`;
      return;
    }
    if (!d) { f.host.innerHTML = `<div class="placeholder"><b>Schwarzschild validation</b>Press Run to recover 3 M and 3√3 M from traced photons.</div>`; return; }
    const card = cardHtml;
    f.host.innerHTML = `<div class="report"><h1>Schwarzschild validation</h1>
      <p>Each value is found by integrating photon paths and bisecting between photons that fall in and photons that escape. The engine is never given the answer. The exact values only measure the error.</p>
      <div class="vcards">${d.checks.map(card).join("")}</div>
      <div class="method">${esc(d.method.toUpperCase())}, relative tolerance ${sci(d.rtol)}, bisection to ${sci(d.tol)} M. A check passes when its relative error is below ${sci(d.pass_rel)}.</div></div>`;
  }

  /* ---------------------------------------------------------------- results panel */
  const kv = rows => `<div class="kv">${rows.map(([k, v, u]) => `<div class="k">${k}</div><div class="v">${v}${u ? `<small>${u}</small>` : ""}</div>`).join("")}</div>`;
  const metric = (l, v, u = "") => `<div class="m"><div class="lbl">${l}</div><div class="num">${v}${u ? `<small>${u}</small>` : ""}</div></div>`;
  function renderResults() {
    const el = $("#metrics");
    $("#results-title").textContent = { shadow: "Shadow", geodesic: "Photon", sweep: "Spin sweep", validation: "Validation" }[S.view];
    if (S.view === "shadow" && live) {
      let cap = 0, escd = 0; if (live.state) for (const v of live.state) { if (v === 2) cap++; else if (v === 1) escd++; }
      const pct = 100 * (live.fraction || 0), rate = live.elapsed > 0 ? live.done / live.elapsed : 0;
      let inbound = 0; if (live.out && live.state) for (let i = 0; i < live.total; i++) if (!live.state[i] && !live.out[i]) inbound++;
      el.innerHTML = `<div class="hero"><div class="lbl">Tracing photons</div><div class="num">${pct.toFixed(0)}<small>%</small></div>
          <div class="cmp">${live.done.toLocaleString()} of ${live.total.toLocaleString()} photons classified · ${num(live.elapsed, 1)} s</div>
          <div class="bar"><i style="width:${100 * cap / live.total}%;background:#ff8a3d"></i><i style="width:${100 * escd / live.total}%;background:#4cc9f0"></i></div>
          <div class="legend-row"><span style="--c:#ff8a3d">captured ${cap.toLocaleString()}</span><span style="--c:#4cc9f0">escaped ${escd.toLocaleString()}</span></div></div>
        <div class="mgrid">${metric("Travelling in", inbound.toLocaleString())}${metric("Travelling out", (live.total - live.done - inbound).toLocaleString())}</div>
        <div class="hint" style="margin-top:10px">Each photon is traced backwards from the camera at ${live.observer_radius} M. Captured photons finish when they reach the horizon. Escaping photons must travel back out to ${live.observer_radius} M, so they finish last. Progress is estimated from where every photon is right now.</div>`;
      return;
    }
    if (S.view === "shadow") {
      const d = data.shadow; if (!d) { el.innerHTML = `<div class="empty-metrics">No shadow traced yet.</div>`; return; }
      const cap = d.counts.CAPTURED || 0, escd = d.counts.ESCAPED || 0, other = d.n_rays - cap - escd;
      el.innerHTML = `<div class="hero"><div class="lbl">Shadow area</div><div class="num">${num(d.shadow_area_numerical, 2)}<small>M²</small></div>
          <div class="cmp">exact ${num(d.shadow_area_analytic, 2)} M² · <b>${(100 * d.area_rel_diff).toFixed(2)}%</b> difference</div>
          <div class="bar"><i style="width:${100 * cap / d.n_rays}%;background:#ff8a3d"></i><i style="width:${100 * escd / d.n_rays}%;background:#4cc9f0"></i><i style="width:${100 * other / d.n_rays}%;background:#ff6b6b"></i></div>
          <div class="legend-row"><span style="--c:#ff8a3d">captured ${cap.toLocaleString()}</span><span style="--c:#4cc9f0">escaped ${escd.toLocaleString()}</span>${other ? `<span style="--c:#ff6b6b">other ${other}</span>` : ""}</div></div>
        <div class="mgrid">${metric("Photons", d.n_rays.toLocaleString())}${metric("Runtime", num(d.runtime_s, 2), " s")}
          ${metric("Throughput", Math.round(d.rays_per_s).toLocaleString(), " /s")}${metric("Engine", `${d.backend === "numba" ? "Numba" : "NumPy"} · ${sci(d.rtol)}`)}
          ${metric("Null error (esc.)", num(d.max_null_error_escaped))}${metric("Carter drift (esc.)", num(d.max_carter_drift_escaped))}</div>`;
    } else if (S.view === "geodesic") {
      const d = data.geodesic[data.geodesic.length - 1]; if (!d) { el.innerHTML = `<div class="empty-metrics">No photon integrated yet.</div>`; return; }
      el.innerHTML = `<div class="hero"><div class="lbl">Outcome</div><div class="num" style="color:${d.state === "CAPTURED" ? "#ff8a3d" : "#4cc9f0"}">${d.state.toLowerCase()}</div>
          <div class="cmp">b = ${d.b.toFixed(4)} M · critical ${d.b_crit.toFixed(4)} M (${d.prograde ? "prograde" : "retrograde"})</div></div>
        <div class="mgrid">${metric("Closest approach", num(d.closest_approach), " M")}${metric("Orbits", num(d.turns, 3), " turns")}
          ${metric("Steps", d.steps.toLocaleString())}${metric("Runtime", num(d.runtime_s * 1000, 1), " ms")}
          ${metric("Max null error", num(d.max_null_error))}${metric("Energy drift", num(d.max_energy_drift))}</div>
        ${d.state === "OUT_OF_DOMAIN" ? `<div class="hint" style="margin-top:10px;color:#f7c948">Fixed-step ${d.method.toUpperCase()} stepped across the horizon, where the radial momentum diverges in Boyer-Lindquist coordinates, and left the valid domain. Switch to RK45 to resolve the capture.</div>` : ""}`;
    } else if (S.view === "animate") {
      const d = data.beam; if (!d) { el.innerHTML = `<div class="empty-metrics">No beam computed yet.</div>`; return; }
      el.innerHTML = `<div class="hero"><div class="lbl">Photon beam</div><div class="num">${d.captured}<small>captured</small>&nbsp; ${d.escaped}<small>escaped</small></div>
          <div class="bar"><i style="width:${100 * d.captured / d.n_photons}%;background:#ff8a3d"></i><i style="width:${100 * d.escaped / d.n_photons}%;background:#4cc9f0"></i></div>
          <div class="cmp">critical b: ${num(d.b_crit_prograde, 3)} M prograde, ${num(d.b_crit_retrograde, 3)} M retrograde</div></div>
        <div class="mgrid">${metric("Photons", d.n_photons)}${metric("Engine time", num(d.runtime_s, 2), " s")}
          ${metric("Accepted steps", d.accepted_steps.toLocaleString())}${metric("Null error (esc.)", num(d.max_null_error_escaped))}
          ${metric("Frames", d.t.length)}${metric("Time span", num(d.t[d.t.length - 1], 1), " M")}</div>
        <div class="hint" style="margin-top:10px">Photons in the ${d.spin >= 0 ? "upper" : "lower"} half of the beam move with the spin, so they can pass closer before being captured.</div>`;
    } else if (S.view === "sweep") {
      const b = data.bh;
      el.innerHTML = b ? kv([["Spin", num(b.spin, 3)], ["Prograde b_c", num(b.b_crit_prograde), "M"], ["Retrograde b_c", num(b.b_crit_retrograde), "M"], ["Gap", num(b.b_crit_retrograde - b.b_crit_prograde), "M"]]) +
        `<div class="hint" style="margin-top:8px">Photons moving against the spin must pass farther out to escape, so the gap grows with spin.</div>` : "";
    } else {
      if (S.view === "about") { el.innerHTML = ""; return; }
      const d = data.validation; if (!d) { el.innerHTML = `<div class="empty-metrics">Not run yet.</div>`; return; }
      el.innerHTML = kv(d.checks.map(c => [c.name, c.rel_error.toExponential(1), c.passed ? "pass" : "FAIL"]).concat([["Total time", num(d.runtime_s, 1), "s"]]));
    }
  }
  async function refreshBH() {
    try {
      const b = await api("blackhole", { spin: S.spin }); data.bh = b;
      $("#bh-spin").textContent = `a* = ${b.spin.toFixed(3)}`;
      $("#derived").innerHTML = kv([["Event horizon r+", num(b.r_plus), "M"], ["Inner horizon r−", num(b.r_minus), "M"], ["Ergosphere (equator)", num(b.ergosphere_equator), "M"],
        ["Photon orbit, pro", num(b.photon_orbit_prograde), "M"], ["Photon orbit, retro", num(b.photon_orbit_retrograde), "M"],
        ["ISCO, pro", num(b.isco_prograde), "M"], ["ISCO, retro", num(b.isco_retrograde), "M"], ["Critical b, pro", num(b.b_crit_prograde), "M"], ["Critical b, retro", num(b.b_crit_retrograde), "M"]]);
      if (S.view === "sweep") renderResults();
    } catch (e) { $("#derived").innerHTML = `<div class="empty-metrics">${esc(e.message)}</div>`; }
  }

  /* ---------------------------------------------------------------- chips */
  function renderChips() {
    const el = $("#chips");
    if (S.view === "shadow") {
      const modes = [["sky", "Lensed sky"], ["capture", "Capture"], ["steps", "Steps"], ["error", "Null error"]];
      el.innerHTML = modes.map(([m, l]) => `<button class="chip ${S.display === m ? "on" : ""}" data-display="${m}"><canvas></canvas>${l}</button>`).join("") +
        `<button class="chip toggle-chip ${S.overlay ? "on" : ""}" data-toggle="overlay"><span class="tick"></span>Analytic edge</button>`;
      const d = data.shadow;
      if (d) $$("[data-display] canvas", el).forEach(c => { const { px } = shadowPixels(d, c.parentElement.dataset.display); c.width = c.height = d.resolution; c.getContext("2d").putImageData(new ImageData(px, d.resolution, d.resolution), 0, 0); });
    } else if (S.view === "animate") {
      const playing = anim && anim.playing, rec = anim && anim.recording;
      el.innerHTML = `<button class="chip icon ${playing ? "on" : ""}" data-anim="toggle"><svg><use href="#${playing ? "i-pause" : "i-play"}"/></svg>${playing ? "Pause" : "Play"}</button>
        <button class="chip icon" data-anim="restart"><svg><use href="#i-restart"/></svg>Restart</button><span class="chip-sep"></span>` +
        [0.5, 1, 2, 4].map(v => `<button class="chip toggle-chip ${S.speed === v ? "on" : ""}" data-speed="${v}">${v}×</button>`).join("") + `<span class="chip-sep"></span>` +
        [["short", "Short trails"], ["long", "Long trails"], ["full", "Full paths"]].map(([v, l]) => `<button class="chip toggle-chip ${S.trail === v ? "on" : ""}" data-trail="${v}">${l}</button>`).join("") +
        `<span class="chip-sep"></span><button class="chip icon rec ${rec ? "on" : ""}" data-anim="record"><svg><use href="#i-rec"/></svg>${rec ? "Recording..." : "Record video"}</button>`;
    } else if (S.view === "geodesic") {
      el.innerHTML = `<button class="chip toggle-chip ${S.hold ? "on" : ""}" data-toggle="hold"><span class="tick"></span>Overlay photons</button>`;
    } else el.innerHTML = "";
  }
  $("#chips").addEventListener("click", e => {
    const d = e.target.closest("[data-display]"); if (d) { set("display", d.dataset.display); return; }
    const t = e.target.closest("[data-toggle]"); if (t) { set(t.dataset.toggle, !S[t.dataset.toggle]); renderChips(); return; }
    const sp = e.target.closest("[data-speed]"); if (sp) { set("speed", Number(sp.dataset.speed)); renderChips(); return; }
    const tr = e.target.closest("[data-trail]"); if (tr) { set("trail", tr.dataset.trail); renderChips(); return; }
    const an = e.target.closest("[data-anim]");
    if (an && anim && anim.loaded) {
      if (an.dataset.anim === "toggle") anim.toggle();
      if (an.dataset.anim === "restart") { anim.restart(); anim.play(); }
      if (an.dataset.anim === "record" && !anim.recording) {
        const ok = anim.record(`kerrray_photon_beam_a${data.beam.spin}.webm`, () => { renderChips(); log("video saved"); });
        if (!ok) log(`<span class="err">This browser cannot record the canvas.</span>`);
      }
      renderChips();
    }
  });

  /* ---------------------------------------------------------------- views */
  function setView(v) {
    S.view = v;
    $("#main").classList.toggle("about-mode", v === "about");
    if (v === "about") { const f = figure("about"); f.host.classList.add("about-host"); KerrAbout.render(f.host, data); }
    $$(".view").forEach(b => b.classList.toggle("active", b.dataset.view === v));
    if (v === "validation" && !data.validation) drawValidation();
    $("#stagebar .tools").style.visibility = ["validation", "animate", "about"].includes(v) ? "hidden" : "";
    if (v !== "animate" && anim && anim.playing && !anim.recording) anim.pause();
    if (v === "animate" && data.beam) {
      ensureAnim();
      if (data.beam.spin !== S.spin || data.beam.n_photons !== S.beam_n + 8) runBeam();
      else requestAnimationFrame(() => { anim.draw(); anim.play(); });
    }
    // Recompute a view on entry when its parameters changed elsewhere.
    if (v === "shadow" && data.shadow && !live && (data.shadow.spin !== S.spin || data.shadow.inclination !== S.inclination || data.shadow.resolution !== S.resolution || data.shadow.fov !== S.fov)) runShadow();
    if (v === "geodesic" && data.geodesic.length && data.geodesic[data.geodesic.length - 1].spin !== S.spin) runGeodesic();
    renderControls(); renderChips(); renderResults(); showStage(); hover(null, null);
  }
  $$(".view").forEach(b => b.addEventListener("click", () => setView(b.dataset.view)));
  const RUN = { shadow: runShadow, geodesic: runGeodesic, sweep: runSweep, animate: runBeam, validation: runValidation, about: () => { setView("validation"); runValidation(); } };
  $("#run").addEventListener("click", () => RUN[S.view]());
  document.addEventListener("keydown", e => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); RUN[S.view](); }
    if (e.key === "`" && e.target.tagName !== "INPUT") { e.preventDefault(); toggleConsole(); }
  });
  const ACTIONS = {
    bcrit: btn => { if (!data.bh) return; const bc = S.prograde ? data.bh.b_crit_prograde : data.bh.b_crit_retrograde; set("b", +(bc * Number(btn.dataset.f)).toFixed(4)); },
    "clear-rays": () => { data.geodesic = []; if (figs.geodesic && figs.geodesic.fig) { figs.geodesic.fig.clear(); figs.geodesic.fig.draw(); } renderResults(); },
    restore: () => { const f = figs[S.view]; if (f && f.fig) f.fig.restore(); },
    save: () => {
      if (S.view === "animate" && anim) { const a = document.createElement("a"); a.href = anim.png(); a.download = "kerrray_animation_frame.png"; a.click(); return; }
      const f = figs[S.view]; if (!f || !f.fig) return;
      const a = document.createElement("a"); a.href = f.fig.png(); a.download = `kerrray_${S.view}.png`; a.click();
    },
  };

  /* ---------------------------------------------------------------- console */
  const out = $("#con-out");
  function log(html) { out.insertAdjacentHTML("beforeend", html + "\n"); out.scrollTop = out.scrollHeight; }
  function redraw() { requestAnimationFrame(() => Object.values(figs).forEach(f => f.fig && f.fig.draw())); }
  function openConsole() { $("#console").classList.remove("collapsed"); $("#console-toggle").classList.add("on"); redraw(); }
  function toggleConsole() {
    const c = $("#console"); c.classList.toggle("collapsed");
    const open = !c.classList.contains("collapsed");
    $("#console-toggle").classList.toggle("on", open); if (open) $("#con-in").focus();
    redraw();
  }
  $("#console-toggle").addEventListener("click", toggleConsole);
  const history = []; let hIdx = 0;
  const HELP = `<b>Commands</b>
  shadow(spin, incl, res)        trace a shadow, e.g. shadow(0.99, 80, 96)
  geodesic(spin, b)              integrate one photon; add retro for retrograde
  sweep                          radii and critical impact parameter across spin
  validate                       Schwarzschild validation
  spin = 0.5                     set spin, inclination, resolution, fov, b, r0 or rtol
  mode sky|capture|steps|error   shadow display
  hold on|off   clc   ver   help`;
  function evalMath(expr) {
    if (!/^[\d\s.+\-*/^()eE]*(sqrt|pi)?[\d\s.+\-*/^()eE]*$/.test(expr)) return null;
    try { const v = Function(`"use strict";return (${expr.replace(/\^/g, "**").replace(/\bpi\b/g, "Math.PI").replace(/\bsqrt\b/g, "Math.sqrt")});`)(); return typeof v === "number" && Number.isFinite(v) ? v : null; } catch { return null; }
  }
  const LIMITS = { spin: [-0.999, 0.999], inclination: [0, 180], resolution: [8, 160], fov: [2, 40], b: [0, 200], r0: [5, 5000], rtol: [1e-12, 1e-4] };
  async function execute(raw) {
    const cmd = raw.trim().replace(/;$/, ""); if (!cmd) return;
    log(`<span class="in">${esc(cmd)}</span>`); history.push(cmd); hIdx = history.length;
    const asg = cmd.match(/^(\w+)\s*=\s*(.+)$/);
    if (asg) {
      const [, k, rhs] = asg, v = evalMath(rhs);
      if (!(k in LIMITS)) { log(`<span class="err">Unknown property '${esc(k)}'.</span>`); return; }
      if (v === null || v < LIMITS[k][0] || v > LIMITS[k][1]) { log(`<span class="err">${k} must be in [${LIMITS[k].join(", ")}].</span>`); return; }
      set(k, k === "resolution" ? Math.round(v) + (Math.round(v) % 2) : v); log(`${k} = ${S[k]}`); return;
    }
    const m = cmd.match(/^(\w+)\s*(?:\((.*)\))?\s*(.*)$/); if (!m) { log(`<span class="err">Invalid expression.</span>`); return; }
    const [, name, argStr = "", rest] = m;
    const args = (argStr ? argStr.split(",") : rest.split(/\s+/)).map(s => s.trim().replace(/^['"]|['"]$/g, "")).filter(Boolean);
    const nums = args.map(a => evalMath(a)).filter(v => v !== null);
    switch (name) {
      case "help": log(HELP); return;
      case "clc": out.innerHTML = ""; return;
      case "hold": set("hold", args[0] !== "off"); renderChips(); log(`hold ${S.hold ? "on" : "off"}`); return;
      case "mode": if (["sky", "capture", "steps", "error"].includes(args[0])) set("display", args[0]); else log(`<span class="err">mode must be sky, capture, steps or error.</span>`); return;
      case "ver": { const d = await api("info"); log(`KerrRay ${d.version} (commit ${d.git_commit}) · Python ${d.python} · ${d.cpu_count} CPUs · numpy ${d.packages.numpy} · scipy ${d.packages.scipy}`); return; }
      case "shadow":
        if (nums[0] !== undefined) S.spin = nums[0]; if (nums[1] !== undefined) S.inclination = nums[1];
        if (nums[2] !== undefined) S.resolution = Math.round(nums[2]) + (Math.round(nums[2]) % 2);
        setView("shadow"); refreshBH(); return runShadow();
      case "geodesic":
        if (args.includes("retro")) S.prograde = false; if (args.includes("pro")) S.prograde = true;
        if (nums[0] !== undefined) S.spin = nums[0]; if (nums[1] !== undefined) S.b = nums[1];
        setView("geodesic"); refreshBH(); return runGeodesic();
      case "sweep": setView("sweep"); return runSweep();
      case "validate": setView("validation"); return runValidation();
      default: { const v = evalMath(cmd); if (v !== null) { log(`ans = ${v}`); return; } log(`<span class="err">Unrecognized command '${esc(name)}'. Type help.</span>`); }
    }
  }
  $("#con-in").addEventListener("keydown", e => {
    if (e.key === "Enter") { const v = e.target.value; e.target.value = ""; execute(v); }
    else if (e.key === "ArrowUp") { if (hIdx > 0) e.target.value = history[--hIdx]; e.preventDefault(); }
    else if (e.key === "ArrowDown") { hIdx = Math.min(history.length, hIdx + 1); e.target.value = history[hIdx] || ""; e.preventDefault(); }
  });

  /* ---------------------------------------------------------------- start */
  async function start() {
    renderControls(); renderChips(); renderResults();
    log(`<b>KerrRay</b> <span class="dim">· photon null geodesics in Kerr spacetime · type help</span>`);
    $("#empty-text").textContent = "Computing black-hole properties...";
    await refreshBH();
    // The shadow job runs on the server while the quick views compute alongside it.
    const shadowDone = runShadow();
    $("#empty").classList.add("hidden");
    await runSweep();
    await runGeodesic();
    await runBeam();
    await shadowDone;
    const want = new URLSearchParams(location.hash.slice(1));
    if (want.get("mode")) set("display", want.get("mode"));
    if (want.get("trail")) set("trail", want.get("trail"));
    if (want.get("seek") && anim) { setView("animate"); anim.pause(); anim.seek(Number(want.get("seek"))); renderChips(); return; }
    if (want.get("console")) openConsole();
    if (want.get("validate")) { setView("validation"); await runValidation(); }
    else setView(want.get("view") || "shadow");
  }
  start();
})();
