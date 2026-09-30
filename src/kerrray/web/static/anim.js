/* KerrRay photon-beam animation. Plays back photon paths that the engine
   computed and resampled onto a common grid of coordinate time t. Nothing
   here integrates or invents motion: frame k shows every photon at t[k]. */
"use strict";

const KerrAnim = (() => {
  const C = { bg: "#07090d", grid: "rgba(150,165,190,0.07)", gridText: "rgba(150,165,190,0.35)", esc: [76, 201, 240], cap: [255, 138, 61], crit: [255, 230, 168], ergo: "#b388ff", orbit: "#7bd88f", text: "#e8ecf2", text2: "#8d97a8" };
  const TRAIL = { short: 24, long: 90, full: 100000 };

  function create(host, opts = {}) {
    const canvas = document.createElement("canvas");
    host.appendChild(canvas);
    const g = canvas.getContext("2d");
    let D = null, frame = 0, playing = false, speed = 1, trail = "long", last = 0, half = 20, W = 0, H = 0, dpr = 1;
    let recorder = null, onEnd = null;
    const ro = new ResizeObserver(() => { resize(); draw(); }); ro.observe(host);

    function resize() {
      const r = host.getBoundingClientRect(); dpr = window.devicePixelRatio || 1;
      W = r.width; H = r.height; canvas.width = Math.max(1, Math.round(W * dpr)); canvas.height = Math.max(1, Math.round(H * dpr));
      canvas.style.width = `${W}px`; canvas.style.height = `${H}px`;
    }
    const scale = () => Math.min(W, H) / (2 * half);
    const sx = x => W / 2 + x * scale();
    const sy = y => H / 2 - y * scale();
    const rho = r => Math.sqrt(r * r + D.spin * D.spin);

    function load(data) { D = data; frame = 0; half = Math.max(12, Math.min(24, data.x0 * 0.62)); resize(); draw(); }

    function pos(i, f) {
      const k = Math.floor(f), u = f - k, xs = D.x[i], ys = D.y[i];
      const x0 = xs[k], y0 = ys[k];
      if (x0 === null || x0 === undefined) return null;
      const x1 = xs[k + 1], y1 = ys[k + 1];
      if (x1 === null || x1 === undefined) return [x0, y0];
      return [x0 + u * (x1 - x0), y0 + u * (y1 - y0)];
    }

    function drawBackground() {
      g.fillStyle = C.bg; g.fillRect(0, 0, W, H);
      const s = scale();
      g.lineWidth = 1; g.font = `10.5px "JetBrains Mono", Consolas, monospace`; g.textAlign = "left"; g.textBaseline = "bottom";
      for (const r of [5, 10, 15, 20, 30]) {
        if (r > half * 1.5) continue;
        g.strokeStyle = C.grid; g.beginPath(); g.arc(W / 2, H / 2, r * s, 0, 2 * Math.PI); g.stroke();
        g.fillStyle = C.gridText; g.fillText(`${r} M`, W / 2 + r * s * 0.707 + 4, H / 2 - r * s * 0.707 - 2);
      }
      g.strokeStyle = C.grid; g.beginPath(); g.moveTo(0, H / 2); g.lineTo(W, H / 2); g.moveTo(W / 2, 0); g.lineTo(W / 2, H); g.stroke();
    }

    function drawHole() {
      const s = scale(), cx = W / 2, cy = H / 2, rh = rho(D.r_plus) * s;
      // photon orbits (prograde and retrograde circular photon orbits)
      g.setLineDash([2, 4]); g.lineWidth = 1; g.strokeStyle = C.orbit; g.globalAlpha = 0.55;
      for (const r of [D.photon_orbit_prograde, D.photon_orbit_retrograde]) { g.beginPath(); g.arc(cx, cy, rho(r) * s, 0, 2 * Math.PI); g.stroke(); }
      // ergosphere, equatorial radius 2M
      g.setLineDash([6, 5]); g.strokeStyle = C.ergo; g.globalAlpha = 0.6; g.beginPath(); g.arc(cx, cy, rho(2) * s, 0, 2 * Math.PI); g.stroke();
      g.setLineDash([]); g.globalAlpha = 1;
      // horizon with a soft rim
      const glow = g.createRadialGradient(cx, cy, rh * 0.9, cx, cy, rh * 1.9);
      glow.addColorStop(0, "rgba(255,138,61,0.35)"); glow.addColorStop(1, "rgba(255,138,61,0)");
      g.fillStyle = glow; g.beginPath(); g.arc(cx, cy, rh * 1.9, 0, 2 * Math.PI); g.fill();
      g.fillStyle = "#000"; g.beginPath(); g.arc(cx, cy, rh, 0, 2 * Math.PI); g.fill();
      g.strokeStyle = "rgba(255,178,122,0.9)"; g.lineWidth = 1.2; g.stroke();
      // rotation sense: counter-clockwise (seen from +z) for a > 0
      if (Math.abs(D.spin) > 1e-6) {
        const ra = rh + 10, dir = D.spin > 0 ? -1 : 1, a0 = -Math.PI * 0.15, a1 = a0 + dir * Math.PI * 0.55;
        g.strokeStyle = "rgba(232,236,242,0.55)"; g.lineWidth = 1.4; g.beginPath(); g.arc(cx, cy, ra, a0, a1, dir < 0); g.stroke();
        const ex = cx + ra * Math.cos(a1), ey = cy + ra * Math.sin(a1), tx = -dir * Math.sin(a1), ty = dir * Math.cos(a1);
        g.fillStyle = "rgba(232,236,242,0.7)"; g.beginPath();
        g.moveTo(ex + tx * 6, ey + ty * 6); g.lineTo(ex - ty * 4 - tx * 2, ey + tx * 4 - ty * 2); g.lineTo(ex + ty * 4 - tx * 2, ey - tx * 4 - ty * 2); g.fill();
      }
    }

    function drawPhotons() {
      const L = TRAIL[trail], s = scale(), n = D.b.length, BUCKETS = 6;
      g.lineCap = "round"; g.lineJoin = "round";
      for (let i = 0; i < n; i++) {
        const crit = D.near_critical[i], captured = D.state[i] === "CAPTURED";
        const col = crit ? C.crit : captured ? C.cap : C.esc;
        const k1 = Math.floor(frame), k0 = Math.max(0, k1 - L);
        // trail in alpha buckets, oldest faintest
        const span = k1 - k0;
        if (span > 0) {
          for (let bkt = 0; bkt < BUCKETS; bkt++) {
            const a = k0 + Math.floor(span * bkt / BUCKETS), b = k0 + Math.floor(span * (bkt + 1) / BUCKETS);
            g.beginPath(); let pen = false;
            for (let k = a; k <= b; k++) {
              const x = D.x[i][k], y = D.y[i][k];
              if (x === null) { pen = false; continue; }
              if (pen) g.lineTo(sx(x), sy(y)); else { g.moveTo(sx(x), sy(y)); pen = true; }
            }
            const p = pos(i, frame);
            if (bkt === BUCKETS - 1 && p && pen) g.lineTo(sx(p[0]), sy(p[1]));
            const alpha = trail === "full" ? 0.55 : Math.pow((bkt + 1) / BUCKETS, 1.6) * (crit ? 0.95 : 0.7);
            g.strokeStyle = `rgba(${col[0]},${col[1]},${col[2]},${alpha})`; g.lineWidth = crit ? 1.8 : 1.25; g.stroke();
          }
        }
        const p = pos(i, frame);
        if (p) {
          const X = sx(p[0]), Y = sy(p[1]);
          if (X < -20 || X > W + 20 || Y < -20 || Y > H + 20) continue;
          g.shadowColor = `rgb(${col[0]},${col[1]},${col[2]})`; g.shadowBlur = crit ? 14 : 9;
          g.fillStyle = crit ? "#fff7e0" : `rgb(${Math.min(255, col[0] + 60)},${Math.min(255, col[1] + 40)},${Math.min(255, col[2] + 20)})`;
          g.beginPath(); g.arc(X, Y, crit ? 2.8 : 2.2, 0, 2 * Math.PI); g.fill(); g.shadowBlur = 0;
        }
      }
    }

    function drawHud() {
      const k = Math.min(D.t.length - 1, Math.floor(frame)), t = D.t[k];
      let cap = 0, esc = 0;
      for (let i = 0; i < D.b.length; i++) { if (D.t_end[i] <= t) { if (D.state[i] === "CAPTURED") cap++; else esc++; } }
      g.textAlign = "left"; g.textBaseline = "top";
      g.fillStyle = C.text; g.font = `600 22px "JetBrains Mono", Consolas, monospace`; g.fillText(`t = ${t.toFixed(1)} M`, 20, 18);
      g.fillStyle = C.text2; g.font = `11.5px "Inter", "Segoe UI", sans-serif`; g.fillText("coordinate time of a distant observer", 20, 46);
      g.font = `12px "Inter", "Segoe UI", sans-serif`;
      const rows = [[C.cap, `captured  ${cap} / ${D.captured}`], [C.esc, `escaped   ${esc} / ${D.escaped}`], [C.crit, "near critical impact parameter"]];
      rows.forEach(([c, txt], j) => { const y = H - 22 - (rows.length - 1 - j) * 19; g.fillStyle = `rgb(${c})`; g.beginPath(); g.arc(26, y + 1, 4, 0, 2 * Math.PI); g.fill(); g.fillStyle = C.text2; g.textBaseline = "middle"; g.fillText(txt, 38, y + 1); });
      g.textAlign = "right"; g.textBaseline = "top"; g.fillStyle = C.text2;
      g.fillText(`a* = ${D.spin}   ·   ${D.b.length} photons   ·   frame ${k + 1} / ${D.t.length}`, W - 20, 20);
      // progress bar
      g.fillStyle = "rgba(255,255,255,0.06)"; g.fillRect(0, H - 3, W, 3); g.fillStyle = "#ff8a3d"; g.fillRect(0, H - 3, W * (frame / (D.t.length - 1)), 3);
    }

    function draw() {
      if (!W || !H) return;
      g.setTransform(dpr, 0, 0, dpr, 0, 0);
      if (!D) { g.fillStyle = C.bg; g.fillRect(0, 0, W, H); return; }
      drawBackground(); drawHole(); drawPhotons(); drawHud();
    }

    // Full pass lasts about 12 s at 1x regardless of the number of frames.
    function tick(ts) {
      if (!playing) return;
      const dt = last ? Math.min(0.1, (ts - last) / 1000) : 0; last = ts;
      frame += dt * speed * (D.t.length / 12);
      if (frame >= D.t.length - 1) {
        frame = D.t.length - 1; draw();
        if (recorder) { stopRecording(); playing = false; opts.onState && opts.onState(false); return; }
        frame = 0;
      }
      draw(); requestAnimationFrame(tick);
    }
    function play() { if (!D || playing) return; playing = true; last = 0; requestAnimationFrame(tick); opts.onState && opts.onState(true); }
    function pause() { playing = false; draw(); opts.onState && opts.onState(false); }
    function restart() { frame = 0; draw(); }
    function seek(f) { if (!D) return; frame = Math.max(0, Math.min(D.t.length - 1, f * (D.t.length - 1))); draw(); }

    function record(filename, done) {
      if (!D || recorder || typeof MediaRecorder === "undefined") return false;
      const mime = ["video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm"].find(m => MediaRecorder.isTypeSupported(m));
      if (!mime) return false;
      const chunks = [], stream = canvas.captureStream(60);
      recorder = new MediaRecorder(stream, { mimeType: mime, videoBitsPerSecond: 8e6 });
      recorder.ondataavailable = e => e.data.size && chunks.push(e.data);
      recorder.onstop = () => {
        const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob(chunks, { type: "video/webm" })); a.download = filename; a.click();
        recorder = null; done && done();
      };
      recorder.start(); frame = 0; playing = false; play();
      return true;
    }
    function stopRecording() { if (recorder && recorder.state !== "inactive") recorder.stop(); }

    host.addEventListener("wheel", e => { e.preventDefault(); half = Math.max(4, Math.min(80, half * (e.deltaY > 0 ? 1.12 : 1 / 1.12))); draw(); }, { passive: false });

    return {
      load, play, pause, restart, seek, record, draw,
      toggle() { playing ? pause() : play(); },
      setSpeed(v) { speed = v; }, setTrail(v) { trail = v; draw(); },
      get playing() { return playing; }, get recording() { return !!recorder; }, get loaded() { return !!D; },
      png() { return canvas.toDataURL("image/png"); },
    };
  }
  return { create };
})();
