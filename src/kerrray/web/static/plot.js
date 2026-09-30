/* KerrRay Desktop plotting: a small canvas axes library in the style of an
   engineering desktop (box axes, inward ticks, colour order, parula colormap,
   legends, colorbars, zoom / pan / data tips). No dependencies. */
"use strict";

const Plot = (() => {
  // Dark instrument theme.
  const T = {
    figure: "#0f131a", axes: "#0a0d12", axis: "#3a4453", tick: "#8d97a8", text: "#c9d1dc", title: "#e8ecf2",
    grid: "rgba(160,175,200,0.08)", legendBg: "rgba(15,19,26,0.88)", legendEdge: "#2c3441", tipBg: "#e8ecf2", tipText: "#0f131a",
  };
  const COLORS = ["#4cc9f0", "#ff8a3d", "#f7c948", "#b388ff", "#7bd88f", "#ff6b8b", "#e8ecf2"];
  const AX = T.axis;
  const FONT = '"Inter", "Segoe UI Variable Text", "Segoe UI", system-ui, sans-serif';

  // Parula colormap: stops sampled from the published 64-entry table.
  const PARULA = [
    [0.2422, 0.1504, 0.6603], [0.2691, 0.2141, 0.8408], [0.2803, 0.2939, 0.9519], [0.2687, 0.3822, 0.9910],
    [0.2006, 0.4702, 0.9819], [0.1592, 0.5534, 0.9280], [0.1386, 0.6219, 0.8664], [0.0795, 0.6809, 0.7960],
    [0.0304, 0.7327, 0.7012], [0.1476, 0.7690, 0.5926], [0.3129, 0.7950, 0.4721], [0.4969, 0.7967, 0.3571],
    [0.6813, 0.7817, 0.2491], [0.8387, 0.7631, 0.1810], [0.9611, 0.7512, 0.2193], [0.9933, 0.8156, 0.2071],
    [0.9722, 0.8940, 0.1576], [0.9769, 0.9839, 0.0805],
  ];
  function parula(t) {
    if (!Number.isFinite(t)) return [255, 255, 255];
    t = Math.min(1, Math.max(0, t)) * (PARULA.length - 1);
    const i = Math.min(PARULA.length - 2, Math.floor(t)), f = t - i, a = PARULA[i], b = PARULA[i + 1];
    return [0, 1, 2].map(k => Math.round(255 * (a[k] + f * (b[k] - a[k]))));
  }
  const INFERNO = [[0, 0, 4], [22, 11, 57], [66, 10, 104], [106, 23, 110], [147, 38, 103], [188, 55, 84], [221, 81, 58], [243, 120, 25], [252, 165, 10], [246, 215, 70], [252, 255, 164]];
  function inferno(t) {
    if (!Number.isFinite(t)) return [0, 0, 0];
    t = Math.min(1, Math.max(0, t)) * (INFERNO.length - 1);
    const i = Math.min(INFERNO.length - 2, Math.floor(t)), f = t - i, a = INFERNO[i], b = INFERNO[i + 1];
    return [0, 1, 2].map(k => Math.round(a[k] + f * (b[k] - a[k])));
  }
  const gray = t => { const v = Math.round(255 * Math.min(1, Math.max(0, t))); return [v, v, v]; };

  function niceStep(range, target) {
    const raw = range / Math.max(1, target), mag = Math.pow(10, Math.floor(Math.log10(raw))), n = raw / mag;
    return (n < 1.5 ? 1 : n < 3 ? 2 : n < 7 ? 5 : 10) * mag;
  }
  function ticks(lo, hi, target) {
    if (!(hi > lo)) return [lo];
    const step = niceStep(hi - lo, target), out = [];
    for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + step * 1e-9; v += step) out.push(Math.abs(v) < step * 1e-9 ? 0 : v);
    return out;
  }
  function fmt(v, step) {
    if (v === 0) return "0";
    const a = Math.abs(v);
    if (a >= 1e5 || a < 1e-3) return v.toExponential(1).replace("e+", "e");
    const d = Math.max(0, Math.min(6, -Math.floor(Math.log10(step || a) + 1e-9)));
    return v.toFixed(d);
  }
  function fmtData(v) {
    if (!Number.isFinite(v)) return String(v);
    const a = Math.abs(v);
    return a !== 0 && (a >= 1e5 || a < 1e-3) ? v.toExponential(4) : (+v.toPrecision(6)).toString();
  }

  class Axes {
    constructor(fig, pos, opts = {}) {
      this.fig = fig; this.pos = pos; this.items = []; this.tips = [];
      Object.assign(this, { title: "", xlabel: "", ylabel: "", grid: true, equal: false, ylog: false, legend: false,
        colorbar: null, xlim: null, ylim: null, home: null }, opts);
      this.colorIndex = 0;
    }
    nextColor() { return COLORS[this.colorIndex++ % COLORS.length]; }
    plot(x, y, o = {}) { const it = { type: "line", x, y, color: o.color || this.nextColor(), width: o.width || 1.5, dash: o.dash || null, name: o.name || null, marker: o.marker || null, halo: o.halo || null }; this.items.push(it); return it; }
    fill(x, y, o = {}) { const it = { type: "fill", x, y, color: o.color || "#000", edge: o.edge || null, name: o.name || null, alpha: o.alpha ?? 1 }; this.items.push(it); return it; }
    circle(cx, cy, r, o = {}) {
      const n = 180, x = [], y = [];
      for (let i = 0; i <= n; i++) { const t = 2 * Math.PI * i / n; x.push(cx + r * Math.cos(t)); y.push(cy + r * Math.sin(t)); }
      return o.fill ? this.fill(x, y, o) : this.plot(x, y, o);
    }
    image(rgba, w, h, extent, o = {}) {
      const c = document.createElement("canvas"); c.width = w; c.height = h;
      c.getContext("2d").putImageData(new ImageData(rgba, w, h), 0, 0);
      const it = { type: "image", canvas: c, w, h, extent, lookup: o.lookup || null, name: null }; this.items.push(it); return it;
    }
    clear() { this.items = []; this.tips = []; this.colorIndex = 0; }

    dataBounds() {
      let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
      for (const it of this.items) {
        if (it.type === "image") { x0 = Math.min(x0, it.extent[0]); x1 = Math.max(x1, it.extent[1]); y0 = Math.min(y0, it.extent[2]); y1 = Math.max(y1, it.extent[3]); continue; }
        for (let i = 0; i < it.x.length; i++) {
          const xv = it.x[i], yv = this.ylog ? it.y[i] : it.y[i];
          if (!Number.isFinite(xv) || !Number.isFinite(yv) || (this.ylog && yv <= 0)) continue;
          if (xv < x0) x0 = xv; if (xv > x1) x1 = xv; if (yv < y0) y0 = yv; if (yv > y1) y1 = yv;
        }
      }
      if (!Number.isFinite(x0)) return [0, 1, 0, 1];
      if (x0 === x1) { x0 -= 1; x1 += 1; }
      if (y0 === y1) { y0 = this.ylog ? y0 / 10 : y0 - 1; y1 = this.ylog ? y1 * 10 : y1 + 1; }
      return [x0, x1, y0, y1];
    }
    autoLimits() {
      let [x0, x1, y0, y1] = this.dataBounds();
      const hasImage = this.items.some(i => i.type === "image");
      if (this.ylog) {
        const l0 = Math.floor(Math.log10(y0)), l1 = Math.ceil(Math.log10(y1));
        y0 = Math.pow(10, l0); y1 = Math.pow(10, l1 === l0 ? l0 + 1 : l1);
      } else if (!hasImage) {
        const sx = niceStep(x1 - x0, 6), sy = niceStep(y1 - y0, 5);
        x0 = Math.floor(x0 / sx + 1e-9) * sx; x1 = Math.ceil(x1 / sx - 1e-9) * sx;
        y0 = Math.floor(y0 / sy + 1e-9) * sy; y1 = Math.ceil(y1 / sy - 1e-9) * sy;
      }
      return [x0, x1, y0, y1];
    }
    ensureLimits() {
      if (!this.xlim || !this.ylim) {
        const [x0, x1, y0, y1] = this.autoLimits();
        this.xlim = this.xlim || [x0, x1]; this.ylim = this.ylim || [y0, y1];
      }
      if (!this.home) this.home = [this.xlim.slice(), this.ylim.slice()];
    }
    restore() { if (this.home) { this.xlim = this.home[0].slice(); this.ylim = this.home[1].slice(); } }

    layout(W, H) {
      const [l, b, w, h] = this.pos;
      const ox = l * W, oy = (1 - b - h) * H, ow = w * W, oh = h * H;
      const ml = 62, mr = this.colorbar ? 84 : 16, mt = this.title ? 34 : 14, mb = this.xlabel ? 46 : 28;
      let px = ox + ml, py = oy + mt, pw = Math.max(20, ow - ml - mr), ph = Math.max(20, oh - mt - mb);
      this.ensureLimits();
      let [x0, x1] = this.xlim, [y0, y1] = this.ylim;
      if (this.equal === "image") {
        const k = Math.min(pw / (x1 - x0), ph / (y1 - y0)), nw = k * (x1 - x0), nh = k * (y1 - y0);
        px += (pw - nw) / 2; py += (ph - nh) / 2; pw = nw; ph = nh;
      } else if (this.equal) {
        const sx = (x1 - x0) / pw, sy = (y1 - y0) / ph;
        if (sx > sy) { const c = (y0 + y1) / 2, r = sx * ph / 2; y0 = c - r; y1 = c + r; }
        else { const c = (x0 + x1) / 2, r = sy * pw / 2; x0 = c - r; x1 = c + r; }
      }
      this.box = { px, py, pw, ph, x0, x1, y0, y1, ox, oy, ow, oh };
      return this.box;
    }
    tx(x) { const b = this.box; return b.px + (x - b.x0) / (b.x1 - b.x0) * b.pw; }
    ty(y) {
      const b = this.box;
      if (this.ylog) { const l0 = Math.log10(b.y0), l1 = Math.log10(b.y1); return b.py + b.ph - (Math.log10(y) - l0) / (l1 - l0) * b.ph; }
      return b.py + b.ph - (y - b.y0) / (b.y1 - b.y0) * b.ph;
    }
    ix(px) { const b = this.box; return b.x0 + (px - b.px) / b.pw * (b.x1 - b.x0); }
    iy(py) {
      const b = this.box, f = (b.py + b.ph - py) / b.ph;
      if (this.ylog) { const l0 = Math.log10(b.y0), l1 = Math.log10(b.y1); return Math.pow(10, l0 + f * (l1 - l0)); }
      return b.y0 + f * (b.y1 - b.y0);
    }
    contains(px, py) { const b = this.box; return b && px >= b.px && px <= b.px + b.pw && py >= b.py && py <= b.py + b.ph; }

    draw(g) {
      const b = this.box;
      g.save();
      g.fillStyle = T.axes; g.fillRect(b.px, b.py, b.pw, b.ph);
      // ticks
      const xt = ticks(b.x0, b.x1, Math.max(3, Math.round(b.pw / 90)));
      let yt;
      if (this.ylog) { yt = []; for (let e = Math.ceil(Math.log10(b.y0) - 1e-9); e <= Math.floor(Math.log10(b.y1) + 1e-9); e++) yt.push(Math.pow(10, e)); const k = Math.ceil(yt.length / 8); yt = yt.filter((_, i) => i % k === 0); }
      else yt = ticks(b.y0, b.y1, Math.max(3, Math.round(b.ph / 60)));
      if (this.grid) {
        g.strokeStyle = T.grid; g.lineWidth = 1; g.beginPath();
        for (const v of xt) { const X = Math.round(this.tx(v)) + .5; g.moveTo(X, b.py); g.lineTo(X, b.py + b.ph); }
        for (const v of yt) { const Y = Math.round(this.ty(v)) + .5; g.moveTo(b.px, Y); g.lineTo(b.px + b.pw, Y); }
        g.stroke();
      }
      // content
      g.save(); g.beginPath(); g.rect(b.px, b.py, b.pw, b.ph); g.clip();
      for (const it of this.items) this.drawItem(g, it);
      for (const tip of this.tips) this.drawTip(g, tip);
      g.restore();
      // box and ticks
      g.strokeStyle = AX; g.lineWidth = 1;
      g.strokeRect(Math.round(b.px) + .5, Math.round(b.py) + .5, Math.round(b.pw), Math.round(b.ph));
      g.beginPath();
      const tl = 5;
      for (const v of xt) { const X = Math.round(this.tx(v)) + .5; g.moveTo(X, b.py + b.ph); g.lineTo(X, b.py + b.ph - tl); g.moveTo(X, b.py); g.lineTo(X, b.py + tl); }
      for (const v of yt) { const Y = Math.round(this.ty(v)) + .5; g.moveTo(b.px, Y); g.lineTo(b.px + tl, Y); g.moveTo(b.px + b.pw, Y); g.lineTo(b.px + b.pw - tl, Y); }
      g.stroke();
      g.fillStyle = T.tick; g.font = `11px ${FONT}`;
      g.textAlign = "center"; g.textBaseline = "top";
      const xs = xt.length > 1 ? xt[1] - xt[0] : 1;
      for (const v of xt) g.fillText(fmt(v, xs), this.tx(v), b.py + b.ph + 5);
      g.textAlign = "right"; g.textBaseline = "middle";
      const ys = yt.length > 1 ? yt[1] - yt[0] : 1;
      for (const v of yt) {
        if (this.ylog) {
          const e = String(Math.round(Math.log10(v))); g.font = `8.5px ${FONT}`; const ew = g.measureText(e).width;
          g.fillText(e, b.px - 5, this.ty(v) - 5); g.font = `11px ${FONT}`; g.fillText("10", b.px - 6 - ew, this.ty(v));
        }
        else g.fillText(fmt(v, ys), b.px - 5, this.ty(v));
      }
      g.font = `12px ${FONT}`; g.textAlign = "center"; g.fillStyle = T.text;
      if (this.xlabel) { g.textBaseline = "top"; g.fillText(this.xlabel, b.px + b.pw / 2, b.py + b.ph + 23); }
      if (this.ylabel) { g.save(); g.translate(b.ox + 14, b.py + b.ph / 2); g.rotate(-Math.PI / 2); g.textBaseline = "middle"; g.fillText(this.ylabel, 0, 0); g.restore(); }
      if (this.title) { g.font = `600 13px ${FONT}`; g.textBaseline = "bottom"; g.fillStyle = T.title; g.fillText(this.title, b.px + b.pw / 2, b.py - 9); }
      if (this.legend) this.drawLegend(g);
      if (this.colorbar) this.drawColorbar(g);
      g.restore();
    }
    drawItem(g, it) {
      if (it.type === "image") {
        const [x0, x1, y0, y1] = it.extent;
        g.imageSmoothingEnabled = false;
        const X0 = this.tx(x0), X1 = this.tx(x1), Y0 = this.ty(y1), Y1 = this.ty(y0);
        g.drawImage(it.canvas, X0, Y0, X1 - X0, Y1 - Y0);
        return;
      }
      g.beginPath();
      let pen = false;
      for (let i = 0; i < it.x.length; i++) {
        const xv = it.x[i], yv = it.y[i];
        if (!Number.isFinite(xv) || !Number.isFinite(yv) || (this.ylog && yv <= 0)) { pen = false; continue; }
        const X = this.tx(xv), Y = this.ty(yv);
        if (pen) g.lineTo(X, Y); else { g.moveTo(X, Y); pen = true; }
      }
      if (it.type === "fill") {
        g.closePath(); g.globalAlpha = it.alpha; g.fillStyle = it.color; g.fill(); g.globalAlpha = 1;
        if (it.edge) { g.strokeStyle = it.edge; g.lineWidth = 1; g.stroke(); }
      } else {
        g.lineJoin = "round";
        if (it.halo) { g.strokeStyle = it.halo; g.lineWidth = it.width + 2.2; g.stroke(); }
        g.strokeStyle = it.color; g.lineWidth = it.width;
        g.setLineDash(it.dash || []); g.stroke(); g.setLineDash([]);
        if (it.marker) {
          g.fillStyle = it.color;
          for (let i = 0; i < it.x.length; i++) if (Number.isFinite(it.x[i]) && Number.isFinite(it.y[i])) { g.beginPath(); g.arc(this.tx(it.x[i]), this.ty(it.y[i]), 2.6, 0, 2 * Math.PI); g.fill(); }
        }
      }
    }
    drawLegend(g) {
      const named = this.items.filter(i => i.name);
      if (!named.length) return;
      const b = this.box;
      g.font = `11px ${FONT}`;
      const w = Math.max(...named.map(i => g.measureText(i.name).width)) + 46, h = named.length * 17 + 8;
      const loc = this.legend === "southwest" ? [b.px + 8, b.py + b.ph - h - 8] : this.legend === "northwest" ? [b.px + 8, b.py + 8] : this.legend === "southeast" ? [b.px + b.pw - w - 8, b.py + b.ph - h - 8] : [b.px + b.pw - w - 8, b.py + 8];
      const [X, Y] = loc;
      g.fillStyle = T.legendBg; g.fillRect(X, Y, w, h);
      g.strokeStyle = T.legendEdge; g.lineWidth = 1; g.strokeRect(Math.round(X) + .5, Math.round(Y) + .5, Math.round(w), Math.round(h));
      named.forEach((it, k) => {
        const yy = Y + 12 + k * 17;
        if (it.type === "fill") { g.fillStyle = it.color; g.fillRect(X + 8, yy - 5, 26, 10); }
        else {
          g.beginPath(); g.moveTo(X + 7, yy); g.lineTo(X + 35, yy);
          if (it.halo) { g.strokeStyle = it.halo; g.lineWidth = it.width + 2.2; g.stroke(); }
          g.strokeStyle = it.color; g.lineWidth = it.width; g.setLineDash(it.dash || []); g.stroke(); g.setLineDash([]);
        }
        g.fillStyle = T.text; g.textAlign = "left"; g.textBaseline = "middle"; g.fillText(it.name, X + 40, yy);
      });
    }
    drawColorbar(g) {
      const b = this.box, cb = this.colorbar, X = b.px + b.pw + 14, W = 14;
      for (let i = 0; i < b.ph; i++) { const c = cb.cmap(1 - i / b.ph); g.fillStyle = `rgb(${c[0]},${c[1]},${c[2]})`; g.fillRect(X, b.py + i, W, 1.5); }
      g.strokeStyle = AX; g.strokeRect(X + .5, Math.round(b.py) + .5, W, Math.round(b.ph));
      const t = ticks(cb.clim[0], cb.clim[1], 6), s = t.length > 1 ? t[1] - t[0] : 1;
      g.fillStyle = T.tick; g.font = `11px ${FONT}`; g.textAlign = "left"; g.textBaseline = "middle";
      for (const v of t) { const Y = b.py + b.ph - (v - cb.clim[0]) / (cb.clim[1] - cb.clim[0]) * b.ph; g.fillText(fmt(v, s), X + W + 5, Y); g.beginPath(); g.moveTo(X + W - 3, Y); g.lineTo(X + W, Y); g.stroke(); }
      if (cb.label) { g.save(); g.translate(X + W + 52, b.py + b.ph / 2); g.rotate(Math.PI / 2); g.textAlign = "center"; g.font = `11.5px ${FONT}`; g.fillStyle = T.text; g.fillText(cb.label, 0, 0); g.restore(); }
    }
    drawTip(g, tip) {
      const X = this.tx(tip.x), Y = this.ty(tip.y);
      g.fillStyle = T.tipBg; g.beginPath(); g.arc(X, Y, 3.2, 0, 2 * Math.PI); g.fill();
      g.strokeStyle = T.figure; g.lineWidth = 1; g.stroke();
      g.font = `11px ${FONT}`;
      const lines = tip.text.split("\n"), w = Math.max(...lines.map(l => g.measureText(l).width)) + 12, h = lines.length * 14 + 6;
      let bx = X + 10, by = Y - h - 8;
      if (bx + w > this.box.px + this.box.pw) bx = X - w - 10;
      if (by < this.box.py) by = Y + 8;
      g.fillStyle = T.tipBg; g.fillRect(bx, by, w, h);
      g.fillStyle = T.tipText; g.textAlign = "left"; g.textBaseline = "top";
      lines.forEach((l, i) => g.fillText(l, bx + 6, by + 4 + i * 14));
    }
    // nearest line vertex or image pixel under (px, py)
    pick(px, py) {
      let best = null, bd = 14 * 14;
      for (const it of this.items) {
        if (it.type !== "line") continue;
        const n = it.x.length, stride = Math.max(1, Math.floor(n / 6000));
        for (let i = 0; i < n; i += stride) {
          if (!Number.isFinite(it.x[i]) || !Number.isFinite(it.y[i])) continue;
          const dx = this.tx(it.x[i]) - px, dy = this.ty(it.y[i]) - py, d = dx * dx + dy * dy;
          if (d < bd) { bd = d; best = { x: it.x[i], y: it.y[i], text: `X ${fmtData(it.x[i])}\nY ${fmtData(it.y[i])}` + (it.name ? `\n${it.name}` : "") }; }
        }
      }
      if (best) return best;
      const x = this.ix(px), y = this.iy(py);
      for (const it of this.items) if (it.type === "image" && it.lookup) { const r = it.lookup(x, y); if (r) return { x: r.x, y: r.y, text: r.text }; }
      return null;
    }
  }

  class Figure {
    constructor(host, opts = {}) {
      this.host = host; this.axes = []; this.mode = "tip"; this.onHover = opts.onHover || null;
      this.canvas = document.createElement("canvas"); host.appendChild(this.canvas);
      this.g = this.canvas.getContext("2d");
      this.ro = new ResizeObserver(() => this.draw()); this.ro.observe(host);
      this.bind();
    }
    addAxes(pos, opts) { const a = new Axes(this, pos, opts); this.axes.push(a); return a; }
    clear() { this.axes = []; }
    setMode(m) { this.mode = m; this.host.classList.remove("cursor-zoom", "cursor-pan", "cursor-tip"); this.host.classList.add(`cursor-${m}`); }
    draw() {
      const r = this.host.getBoundingClientRect(); if (r.width < 2 || r.height < 2) return;
      const dpr = window.devicePixelRatio || 1;
      this.canvas.width = Math.round(r.width * dpr); this.canvas.height = Math.round(r.height * dpr);
      const g = this.g; g.setTransform(dpr, 0, 0, dpr, 0, 0);
      g.fillStyle = T.figure; g.fillRect(0, 0, r.width, r.height);
      for (const a of this.axes) { a.layout(r.width, r.height); a.draw(g); }
      if (this.rubber) { const q = this.rubber; g.strokeStyle = "#ff8a3d"; g.setLineDash([4, 3]); g.strokeRect(q.x0, q.y0, q.x1 - q.x0, q.y1 - q.y0); g.setLineDash([]); g.fillStyle = "rgba(255,138,61,0.08)"; g.fillRect(q.x0, q.y0, q.x1 - q.x0, q.y1 - q.y0); }
    }
    axesAt(px, py) { return this.axes.find(a => a.contains(px, py)) || null; }
    restore() { for (const a of this.axes) { a.restore(); a.tips = []; } this.draw(); }
    png() { return this.canvas.toDataURL("image/png"); }
    bind() {
      const c = this.canvas, pos = e => { const r = c.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
      let drag = null;
      c.addEventListener("wheel", e => {
        const [px, py] = pos(e), a = this.axesAt(px, py); if (!a) return;
        e.preventDefault();
        const f = e.deltaY > 0 ? 1.18 : 1 / 1.18, x = a.ix(px), b = a.box;
        a.xlim = [x + (b.x0 - x) * f, x + (b.x1 - x) * f];
        if (a.ylog) { const y = Math.log10(a.iy(py)), l0 = Math.log10(b.y0), l1 = Math.log10(b.y1); a.ylim = [Math.pow(10, y + (l0 - y) * f), Math.pow(10, y + (l1 - y) * f)]; }
        else { const y = a.iy(py); a.ylim = [y + (b.y0 - y) * f, y + (b.y1 - y) * f]; }
        this.draw();
      }, { passive: false });
      c.addEventListener("mousedown", e => {
        const [px, py] = pos(e), a = this.axesAt(px, py); if (!a || e.button !== 0) return;
        drag = { a, px, py, box: { ...a.box } };
        if (this.mode === "tip") { const hit = a.pick(px, py); a.tips = hit ? [hit] : []; this.draw(); drag = null; }
      });
      window.addEventListener("mousemove", e => {
        const [px, py] = pos(e);
        if (drag && this.mode === "pan") {
          const a = drag.a, b = drag.box, dx = (px - drag.px) / b.pw * (b.x1 - b.x0);
          a.xlim = [b.x0 - dx, b.x1 - dx];
          if (a.ylog) { const l0 = Math.log10(b.y0), l1 = Math.log10(b.y1), d = (py - drag.py) / b.ph * (l1 - l0); a.ylim = [Math.pow(10, l0 + d), Math.pow(10, l1 + d)]; }
          else { const dy = (py - drag.py) / b.ph * (b.y1 - b.y0); a.ylim = [b.y0 + dy, b.y1 + dy]; }
          this.draw(); return;
        }
        if (drag && this.mode === "zoom") { this.rubber = { x0: Math.min(drag.px, px), y0: Math.min(drag.py, py), x1: Math.max(drag.px, px), y1: Math.max(drag.py, py) }; this.draw(); return; }
        if (e.target !== c) return;
        const a = this.axesAt(px, py);
        if (this.onHover) this.onHover(a ? a.pick(px, py) : null, a ? [a.ix(px), a.iy(py)] : null);
      });
      window.addEventListener("mouseup", () => {
        if (drag && this.mode === "zoom" && this.rubber) {
          const q = this.rubber, a = drag.a;
          if (q.x1 - q.x0 > 6 && q.y1 - q.y0 > 6) { a.xlim = [a.ix(q.x0), a.ix(q.x1)]; a.ylim = [a.iy(q.y1), a.iy(q.y0)]; }
          else { const x = a.ix(q.x0), y = a.iy(q.y0), b = a.box; a.xlim = [x - (b.x1 - b.x0) / 4, x + (b.x1 - b.x0) / 4]; if (!a.ylog) a.ylim = [y - (b.y1 - b.y0) / 4, y + (b.y1 - b.y0) / 4]; }
          this.rubber = null; this.draw();
        }
        drag = null;
      });
      c.addEventListener("mouseleave", () => { if (this.onHover) this.onHover(null, null); });
    }
    destroy() { this.ro.disconnect(); this.canvas.remove(); }
  }

  return { Figure, Axes, COLORS, THEME: T, parula, inferno, gray, fmtData };
})();
