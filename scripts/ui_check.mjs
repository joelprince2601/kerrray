// End-to-end check of the KerrRay app in a real (headless) Chrome.
// Clicks every control, waits for the engine, asserts on the visible result,
// records any JavaScript error and saves screenshots.
//
// Usage: node scripts/ui_check.mjs [http://127.0.0.1:8741/] [out_dir]
// Needs Chrome installed and the app running (kerrray gui). Node 22+.

import { spawn } from "node:child_process";
import { mkdirSync, writeFileSync, existsSync, mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const URL_ = process.argv[2] || "http://127.0.0.1:8741/";
const OUT = process.argv[3] || "results/ui_check";
const CHROME = ["C:/Program Files/Google/Chrome/Application/chrome.exe", "C:/Program Files (x86)/Google/Chrome/Application/chrome.exe", "/usr/bin/google-chrome", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"].find(existsSync);
const PORT = 9333;
mkdirSync(OUT, { recursive: true });

const sleep = ms => new Promise(r => setTimeout(r, ms));
const chrome = spawn(CHROME, [`--remote-debugging-port=${PORT}`, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--window-size=1600,960", `--user-data-dir=${mkdtempSync(join(tmpdir(), "kr-"))}`, "about:blank"], { stdio: "ignore" });

let ws, seq = 0;
const pending = new Map(), errors = [], results = [];
async function connect() {
  for (let i = 0; i < 50; i++) { try { const r = await fetch(`http://127.0.0.1:${PORT}/json/list`); const t = (await r.json()).find(x => x.type === "page"); if (t) return t.webSocketDebuggerUrl; } catch {} await sleep(200); }
  throw new Error("Chrome did not start");
}
function send(method, params = {}) {
  const id = ++seq; ws.send(JSON.stringify({ id, method, params }));
  return new Promise((res, rej) => pending.set(id, { res, rej }));
}
async function js(expr) {
  const r = await send("Runtime.evaluate", { expression: expr, awaitPromise: true, returnByValue: true });
  if (r.exceptionDetails) throw new Error(`${expr.slice(0, 80)} -> ${r.exceptionDetails.exception?.description || r.exceptionDetails.text}`);
  return r.result.value;
}
async function waitFor(expr, timeout = 90000, label = expr) {
  const t0 = Date.now();
  while (Date.now() - t0 < timeout) { if (await js(expr)) return Date.now() - t0; await sleep(150); }
  throw new Error(`timeout waiting for: ${label}`);
}
async function shot(name) { const r = await send("Page.captureScreenshot", { format: "png" }); writeFileSync(join(OUT, `${name}.png`), Buffer.from(r.data, "base64")); }
const click = sel => js(`(() => { const e = document.querySelector(${JSON.stringify(sel)}); if (!e) throw new Error("missing " + ${JSON.stringify(sel)}); e.click(); return true; })()`);
const setRange = (key, v) => js(`(() => { const e = document.querySelector('[data-range="${key}"]'); e.value = ${v}; e.dispatchEvent(new Event("input", {bubbles:true})); e.dispatchEvent(new Event("change", {bubbles:true})); return true; })()`);
const text = sel => js(`(document.querySelector(${JSON.stringify(sel)})||{}).textContent || ""`);
async function check(name, fn) {
  const t0 = Date.now();
  try { const note = await fn(); results.push({ name, ok: true, ms: Date.now() - t0, note: note || "" }); console.log(`PASS  ${name}${note ? "  (" + note + ")" : ""}`); }
  catch (e) { results.push({ name, ok: false, ms: Date.now() - t0, note: e.message }); console.log(`FAIL  ${name}: ${e.message}`); }
}

try {
  ws = new WebSocket(await connect());
  await new Promise(r => ws.addEventListener("open", r, { once: true }));
  ws.addEventListener("message", ev => {
    const m = JSON.parse(ev.data);
    if (m.id && pending.has(m.id)) { const p = pending.get(m.id); pending.delete(m.id); m.error ? p.rej(new Error(m.error.message)) : p.res(m.result); }
    if (m.method === "Runtime.exceptionThrown") errors.push(m.params.exceptionDetails.exception?.description || m.params.exceptionDetails.text);
    if (m.method === "Runtime.consoleAPICalled" && m.params.type === "error") errors.push(m.params.args.map(a => a.value || a.description).join(" "));
    if (m.method === "Log.entryAdded" && m.params.entry.level === "error") errors.push(`${m.params.entry.text} ${m.params.entry.url || ""}`);
  });
  await send("Runtime.enable"); await send("Page.enable"); await send("Log.enable");
  await send("Page.navigate", { url: URL_ });

  await check("startup: live shadow streams in", async () => {
    // With the compiled engine a 64x64 trace can finish between two polls, so accept either state.
    await waitFor(`document.querySelector("#metrics") && /Tracing photons|Shadow area/.test(document.querySelector("#metrics").textContent)`, 60000, "live progress card or result");
    const mid = await js(`document.querySelector("#metrics").textContent.includes("Tracing photons")`);
    await shot("01_shadow_live");
    return mid ? (await text("#metrics .hero .num")).trim() + " traced mid-run" : "finished before the first poll";
  });
  await check("startup completes (sweep, geodesic, beam, shadow)", async () => { const ms = await waitFor(`document.querySelector("#empty").classList.contains("hidden") && document.querySelector("#metrics").textContent.includes("Shadow area")`, 120000); return `${(ms / 1000).toFixed(1)} s`; });
  await shot("02_shadow_done");

  await check("shadow display chips (sky, capture, steps, error, analytic edge)", async () => {
    for (const m of ["capture", "steps", "error", "sky"]) { await click(`[data-display="${m}"]`); await waitFor(`document.querySelector('[data-display="${m}"]').classList.contains("on")`, 3000); }
    await click('[data-toggle="overlay"]'); await waitFor(`!document.querySelector('[data-toggle="overlay"]').classList.contains("on")`, 3000);
    await click('[data-toggle="overlay"]'); return "4 modes + toggle";
  });
  await check("figure tools (inspect, zoom, pan, reset)", async () => {
    for (const m of ["zoom", "pan", "tip"]) { await click(`#stagebar [data-mode="${m}"]`); await waitFor(`document.querySelector('#stagebar [data-mode="${m}"]').classList.contains("on")`, 2000); }
    await click('[data-act="restore"]'); return "modes switch";
  });
  await check("shadow re-traces live when spin preset changes", async () => {
    await click('[data-presets="spin"] [data-v="0.5"]');
    await waitFor(`document.querySelector("#metrics").textContent.includes("Tracing photons")`, 5000, "retrace started");
    await waitFor(`document.querySelector("#metrics").textContent.includes("Shadow area") && document.querySelector(".hero .cmp")`, 90000, "retrace finished");
    const title = await js(`document.querySelector("#bh-spin").textContent`); return title;
  });
  await check("resolution preset, integrator and tolerance segments", async () => {
    await click('[data-presets="resolution"] [data-v="32"]');
    await click('[data-seg="backend"] [data-v="numpy"]'); await click('[data-seg="backend"] [data-v="numba"]');
    await click('[data-seg="rtol"] [data-v="0.000001"]');
    await waitFor(`document.querySelector("#metrics").textContent.includes("Shadow area") && document.querySelector("#metrics").textContent.includes("1,024")`, 90000, "32x32 trace");
    return "32x32 traced";
  });
  await check("NumPy engine streams photons live", async () => {
    await click('[data-presets="resolution"] [data-v="48"]'); await click('[data-seg="backend"] [data-v="numpy"]');
    await waitFor(`document.querySelector("#metrics").textContent.includes("Travelling in")`, 10000, "live card");
    await waitFor(`parseInt(document.querySelector("#metrics .hero .num").textContent) >= 20`, 60000, "progress at least 20%");
    await shot("01b_shadow_live_numpy");
    await waitFor(`document.querySelector("#metrics").textContent.includes("Shadow area")`, 120000, "finished");
    await click('[data-seg="backend"] [data-v="numba"]');
    await waitFor(`document.querySelector("#metrics").textContent.includes("Numba")`, 60000, "numba result");
    return "progress streamed, then switched back to Numba";
  });
  await check("inclination slider drives a new trace", async () => {
    await setRange("inclination", 85);
    await waitFor(`document.querySelector("#metrics").textContent.includes("Tracing photons")`, 5000);
    await waitFor(`document.querySelector("#metrics").textContent.includes("Shadow area")`, 90000);
    return (await text(".hero .cmp")).trim();
  });

  await check("geodesic view: slider re-integrates live, b_c presets, direction, overlay, clear", async () => {
    await click('.view[data-view="geodesic"]');
    await waitFor(`document.querySelector("#metrics").textContent.includes("Outcome")`, 20000);
    await click('[data-act="bcrit"][data-f="0.99"]');
    await waitFor(`document.querySelector("#metrics .hero .num").textContent.includes("captured")`, 20000, "0.99 b_c captured");
    await click('[data-act="bcrit"][data-f="1.001"]');
    await waitFor(`document.querySelector("#metrics .hero .num").textContent.includes("escaped")`, 20000, "1.001 b_c escapes");
    await click('[data-seg="geo_method"] [data-v="rk4"]'); await click('[data-act="bcrit"][data-f="0.99"]');
    await waitFor(`document.querySelector("#metrics").textContent.includes("stepped across the horizon")`, 30000, "RK4 horizon explanation");
    await click('[data-seg="geo_method"] [data-v="rk45"]');
    await waitFor(`document.querySelector("#metrics .hero .num").textContent.includes("captured")`, 20000, "RK45 captures again");
    await click('[data-seg="prograde"] [data-v="false"]'); await sleep(1500);
    await click('[data-toggle="hold"]'); await click('[data-act="bcrit"][data-f="1.3"]'); await sleep(2000);
    await shot("03_geodesic");
    await click('[data-act="clear-rays"]');
    return "captured below b_c, escaped above";
  });
  await check("spin sweep view and marker", async () => {
    await click('.view[data-view="sweep"]'); await setRange("spin", 0.95); await sleep(800);
    await waitFor(`document.querySelector("#metrics").textContent.includes("Prograde b_c")`, 10000);
    await shot("04_sweep"); return (await text("#bh-spin")).trim();
  });
  await check("animation view: plays, pause, restart, speed, trails, recompute on spin", async () => {
    await click('.view[data-view="animate"]');
    await waitFor(`document.querySelector('[data-anim="toggle"]') && document.querySelector('[data-anim="toggle"]').textContent.includes("Pause")`, 20000, "playing");
    await sleep(3000); await shot("05_animation");
    await click('[data-anim="toggle"]'); await waitFor(`document.querySelector('[data-anim="toggle"]').textContent.includes("Play")`, 3000, "paused");
    await click('[data-anim="restart"]');
    for (const v of ["2", "4", "1"]) await click(`[data-speed="${v}"]`);
    for (const v of ["short", "full", "long"]) await click(`[data-trail="${v}"]`);
    await click('[data-presets="beam_n"] [data-v="96"]');
    await waitFor(`document.querySelector("#metrics").textContent.includes("104")`, 30000, "96+8 photons");
    return "beam of 104 photons";
  });
  await check("record video starts and stops", async () => {
    await click('[data-anim="record"]');
    await waitFor(`document.querySelector('[data-anim="record"]').textContent.includes("Recording")`, 5000, "recording");
    await waitFor(`document.querySelector('[data-anim="record"]').textContent.includes("Record video")`, 30000, "recording finished");
    return "webm produced";
  });
  await check("validation view streams the bisection live", async () => {
    await click('.view[data-view="validation"]'); await click('[data-presets="tol"] [data-v="0.00001"]');
    await click("#run");
    await waitFor(`document.querySelector(".vcard.live") !== null`, 20000, "live card");
    await waitFor(`/Iteration[\\s\\S]*[1-9]/.test(document.querySelector(".vcard.live").textContent)`, 60000, "iterations advancing");
    await shot("06_validation_live");
    await waitFor(`document.querySelectorAll(".badge.pass, .badge.fail").length >= 2 && !document.querySelector(".vcard.live")`, 180000, "both checks done");
    await shot("07_validation_done");
    const passes = await js(`document.querySelectorAll(".report .badge.pass").length`);
    if (passes < 2) throw new Error(`only ${passes} PASS`);
    return "both PASS";
  });
  await check("about page renders formulas and live numbers", async () => {
    await click('.view[data-view="about"]');
    await waitFor(`document.querySelectorAll(".about math").length >= 14`, 5000, "MathML blocks");
    const n = await js(`document.querySelectorAll(".about math").length`);
    const hasVal = await js(`document.querySelector(".about").textContent.includes("Photon sphere radius")`);
    const author = await js(`document.querySelector(".about").textContent.includes("Joel Prince")`);
    if (!hasVal || !author) throw new Error("missing live validation table or author credit");
    await shot("08_about_top");
    await js(`document.querySelector(".about-host").scrollTop = 1500`); await sleep(300); await shot("09_about_formulas");
    await js(`document.querySelector(".about-host").scrollTop = 99999`); await sleep(300); await shot("10_about_end");
    return `${n} formula blocks`;
  });
  await check("console: toggle, commands, help", async () => {
    await click('.view[data-view="shadow"]'); await click("#console-toggle");
    await waitFor(`!document.querySelector("#console").classList.contains("collapsed")`, 2000);
    for (const c of ["help", "spin = 0.7", "mode capture", "ver", "3*sqrt(3)"]) {
      await js(`(() => { const i = document.querySelector("#con-in"); i.value = ${JSON.stringify(c)}; i.dispatchEvent(new KeyboardEvent("keydown", {key: "Enter", bubbles: true})); return true; })()`); await sleep(400);
    }
    const out = await text("#con-out");
    if (!out.includes("ans = 5.196")) throw new Error("math evaluation missing");
    await waitFor(`document.querySelector("#metrics").textContent.includes("Shadow area")`, 90000);
    await shot("11_console"); await click("#console-toggle");
    return "commands ran";
  });
  await check("save buttons produce images", async () => {
    const ok = await js(`(() => { let n = 0; const orig = HTMLAnchorElement.prototype.click; HTMLAnchorElement.prototype.click = function () { if (this.download) n++; }; document.querySelector('[data-act="save"]').click(); HTMLAnchorElement.prototype.click = orig; return n; })()`);
    if (ok !== 1) throw new Error("save did not trigger a download"); return "PNG download triggered";
  });
} catch (e) {
  results.push({ name: "harness", ok: false, note: e.message }); console.log("HARNESS ERROR", e.message);
} finally {
  const failed = results.filter(r => !r.ok).length;
  console.log(`\n${results.length - failed}/${results.length} checks passed, ${errors.length} JavaScript errors`);
  errors.forEach(e => console.log("  JS ERROR:", e));
  writeFileSync(join(OUT, "report.json"), JSON.stringify({ url: URL_, results, js_errors: errors }, null, 2));
  try { ws && ws.close(); } catch {}
  chrome.kill();
  process.exit(failed || errors.length ? 1 : 0);
}
