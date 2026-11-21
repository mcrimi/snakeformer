// Ink-only SVG charts for the findings. Colours come from CSS variables, so
// the same drawing works on Warm, Grey and Night paper.

export const DASHES = ["", "7 3", "1.5 3.5", "9 3 1.5 3", "4 3"];

const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const r = (n) => Math.round(n * 10) / 10;

export function linearScale([d0, d1], [r0, r1]) {
  const k = d1 === d0 ? 0 : (r1 - r0) / (d1 - d0);
  return (v) => r0 + (v - d0) * k;
}

/** Logarithmic scale; values at or below zero are drawn at the low end. */
export function logScale([d0, d1], [r0, r1]) {
  const a = Math.log(d0);
  const k = (r1 - r0) / (Math.log(d1) - a);
  return (v) => r0 + (Math.log(Math.max(v, d0)) - a) * k;
}

/** Polyline through the points, skipping gaps (null or NaN). */
export function linePath(xs, ys, sx, sy) {
  let d = "";
  let pen = false;
  for (let i = 0; i < xs.length; i += 1) {
    const y = ys[i];
    if (y === null || y === undefined || Number.isNaN(y)) { pen = false; continue; }
    d += `${pen ? "L" : "M"}${r(sx(xs[i]))},${r(sy(y))}`;
    pen = true;
  }
  return d;
}

export function areaPath(xs, lo, hi, sx, sy) {
  const idx = xs.map((_, i) => i).filter((i) => Number.isFinite(lo[i]) && Number.isFinite(hi[i]));
  if (!idx.length) return "";
  const top = idx.map((i) => `${r(sx(xs[i]))},${r(sy(hi[i]))}`);
  const bottom = idx.reverse().map((i) => `${r(sx(xs[i]))},${r(sy(lo[i]))}`);
  return `M${top.join("L")}L${bottom.join("L")}Z`;
}

/** Spread label positions so they do not overlap; keeps their order. */
export function spreadLabels(ys, gap, min, max) {
  const order = ys.map((y, i) => [y, i]).sort((a, b) => a[0] - b[0]);
  const out = new Array(ys.length);
  let prev = -Infinity;
  for (const [y, i] of order) {
    out[i] = Math.max(y, prev + gap, min);
    prev = out[i];
  }
  let ceiling = max;
  for (let k = order.length - 1; k >= 0; k -= 1) {
    const i = order[k][1];
    out[i] = Math.min(out[i], ceiling);
    ceiling = out[i] - gap;
  }
  return out;
}

export function ticks(min, max, count = 5) {
  const span = max - min;
  if (!(span > 0)) return [min];
  const raw = span / count;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= count + 0.5) || raw;
  const out = [];
  for (let v = Math.ceil(min / step - 1e-9) * step; v <= max + step * 1e-9; v += step) out.push(Math.round(v * 1e9) / 1e9 + 0);
  return out;
}

/** Powers of ten (and optionally 2x, 5x) inside [min, max]. */
export function logTicks(min, max, dense = false) {
  const out = [];
  for (let e = Math.floor(Math.log10(min)); e <= Math.ceil(Math.log10(max)); e += 1) {
    for (const m of dense ? [1, 2, 5] : [1]) {
      const v = Number((m * 10 ** e).toPrecision(1));
      if (v >= min * (1 - 1e-9) && v <= max * (1 + 1e-9)) out.push(v);
    }
  }
  return out;
}

/** "182%", "24%", "1.6%", "0.12%": as many digits as the size deserves. */
export function percent(p) {
  if (p === null || p === undefined || !Number.isFinite(p)) return "–";
  const v = 100 * p;
  const a = Math.abs(v);
  return `${a >= 9.95 ? v.toFixed(0) : a >= 0.995 ? v.toFixed(1) : v.toFixed(2)}%`;
}

const hatch = `<defs><pattern id="hatch" width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><line class="hatch-line" x1="0" y1="0" x2="0" y2="5"/></pattern></defs>`;

function frame(svg, ratio, minHeight = 230) {
  const box = svg.getBoundingClientRect().width || svg.parentNode.clientWidth || 640;
  const width = Math.max(280, Math.round(box));
  const narrow = width < 480;
  const height = Math.max(minHeight, Math.round(width * (narrow ? Math.max(ratio, 0.7) : ratio)));
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  return { width, height, narrow };
}

/**
 * A line chart with labels at the right-hand end of each series.
 * series: [{ label, x, y, dash, lo, hi }]   lo/hi draw a hatched band
 * points: [{ x, y, open }]                  square markers
 * spans:  [{ x0, x1, y, label, place }]     a heavy horizontal bar with end ticks
 * xLog / yLog: logarithmic axes; xTicks / yTicks override the tick positions
 */
export function drawLines(svg, o) {
  const { x, y, series = [], points = [], spans = [], xLabel = "", yFormat = (v) => v, xFormat = (v) => v, labelWidth = 120, ratio = 0.5 } = o;
  const { width, height, narrow } = frame(svg, ratio);
  const m = { top: 12, right: narrow ? Math.min(labelWidth, 96) : labelWidth, bottom: 38, left: 46 };
  const sx = (o.xLog ? logScale : linearScale)(x, [m.left, width - m.right]);
  const sy = (o.yLog ? logScale : linearScale)(y, [height - m.bottom, m.top]);
  let out = hatch;

  for (const t of o.yTicks || (o.yLog ? logTicks(y[0], y[1]) : ticks(y[0], y[1], 4))) {
    if (t > y[0]) out += `<line class="grid" x1="${m.left}" x2="${r(sx(x[1]))}" y1="${r(sy(t))}" y2="${r(sy(t))}"/>`;
    out += `<text x="${m.left - 7}" y="${r(sy(t)) + 4}" text-anchor="end">${esc(yFormat(t))}</text>`;
  }
  for (const t of o.xTicks || (o.xLog ? logTicks(x[0], x[1], !narrow) : ticks(x[0], x[1], narrow ? 4 : 8))) {
    out += `<line class="axis" x1="${r(sx(t))}" x2="${r(sx(t))}" y1="${height - m.bottom}" y2="${height - m.bottom + 4}"/>`;
    out += `<text x="${r(sx(t))}" y="${height - m.bottom + 16}" text-anchor="middle">${esc(xFormat(t))}</text>`;
  }
  out += `<text x="${m.left}" y="${height - 4}">${esc(xLabel)}</text>`;
  out += `<path class="axis" d="M${m.left},${m.top}V${height - m.bottom}H${r(sx(x[1]))}"/>`;

  for (const s of series) if (s.lo && s.hi) out += `<path class="band" d="${areaPath(s.x, s.lo, s.hi, sx, sy)}"/>`;

  const clamp = (v) => Math.min(Math.max(v, y[0]), y[1]);
  const lastY = (s) => { for (let i = s.y.length - 1; i >= 0; i -= 1) if (Number.isFinite(s.y[i])) return s.y[i]; return y[0]; };
  const labelled = series.filter((s) => s.label);
  const ends = labelled.map((s) => sy(clamp(lastY(s))));
  const ys = spreadLabels(ends, 13, m.top + 4, height - m.bottom - 2);
  for (const s of series) {
    out += `<path class="line" d="${linePath(s.x, s.y.map((v) => (Number.isFinite(v) ? clamp(v) : v)), sx, sy)}"${s.dash ? ` stroke-dasharray="${s.dash}"` : ""}/>`;
  }
  labelled.forEach((s, i) => {
    if (Math.abs(ys[i] - ends[i]) > 2) out += `<path class="grid" d="M${r(sx(x[1]) + 1)},${r(ends[i])}L${r(sx(x[1]) + 6)},${r(ys[i])}"/>`;
    out += `<text class="lbl" x="${r(sx(x[1]) + 9)}" y="${r(ys[i] + 4)}">${esc(s.label)}</text>`;
  });
  for (const p of points) {
    out += `<rect class="pt${p.open ? " open" : ""}" x="${r(sx(p.x) - 3)}" y="${r(sy(clamp(p.y)) - 3)}" width="6" height="6"/>`;
  }
  for (const s of spans) {
    const yy = r(sy(clamp(s.y)));
    const x0 = r(sx(Math.min(s.x0, s.x1)));
    const x1 = r(sx(Math.max(s.x0, s.x1)));
    out += `<path class="span" d="M${x0},${yy - 5}V${yy + 5}M${x0},${yy}H${x1}M${x1},${yy - 5}V${yy + 5}"/>`;
    if (!s.label) continue;
    // place: where the label sits relative to the bar, so neighbours do not collide.
    const at = {
      above: [x1, yy - 8, "end"], below: [x1, yy + 17, "end"],
      "below-start": [x0, yy + 17, "start"], left: [x0 - 7, yy + 4, "end"],
    }[s.place || "above"];
    out += `<text class="lbl" x="${at[0]}" y="${at[1]}" text-anchor="${at[2]}">${esc(s.label)}</text>`;
  }
  svg.innerHTML = out;
}

/**
 * Horizontal bars, one per row, on a linear or logarithmic axis.
 * rows: [{ label, value, lo, hi, note, hatched }]
 */
export function drawBars(svg, { rows, x, xLog = false, xFormat = (v) => v, xLabel = "", xTicks, labelWidth = 190 }) {
  const box = svg.getBoundingClientRect().width || svg.parentNode.clientWidth || 640;
  const width = Math.max(280, Math.round(box));
  const narrow = width < 520;
  const rowH = narrow ? 44 : 30;
  const m = { top: 6, right: 56, bottom: 38, left: narrow ? 8 : labelWidth };
  const height = m.top + rows.length * rowH + m.bottom;
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  const sx = (xLog ? logScale : linearScale)(x, [m.left, width - m.right]);
  let out = hatch;
  const base = height - m.bottom;
  for (const t of xTicks || (xLog ? logTicks(x[0], x[1]) : ticks(x[0], x[1], narrow ? 3 : 6))) {
    out += `<line class="grid" x1="${r(sx(t))}" x2="${r(sx(t))}" y1="${m.top}" y2="${base}"/>`;
    out += `<text x="${r(sx(t))}" y="${base + 16}" text-anchor="middle">${esc(xFormat(t))}</text>`;
  }
  out += `<text x="${m.left}" y="${height - 4}">${esc(xLabel)}</text>`;
  rows.forEach((row, i) => {
    const top = m.top + i * rowH;
    const barTop = narrow ? top + 20 : top + 7;
    const barH = narrow ? 14 : 16;
    const end = sx(row.value);
    out += `<rect class="${row.hatched ? "band" : "bar"}" x="${m.left}" y="${barTop}" width="${r(Math.max(end - m.left, 1))}" height="${barH}"/>`;
    if (row.hatched) out += `<rect class="bar-outline" x="${m.left}" y="${barTop}" width="${r(Math.max(end - m.left, 1))}" height="${barH}"/>`;
    if (Number.isFinite(row.lo) && Number.isFinite(row.hi)) {
      const mid = barTop + barH / 2;
      out += `<path class="whisker" d="M${r(sx(row.lo))},${mid}H${r(sx(row.hi))}"/>`;
    }
    out += narrow
      ? `<text class="lbl" x="${m.left}" y="${top + 14}">${esc(row.label)}</text>`
      : `<text class="lbl" x="${m.left - 10}" y="${barTop + barH / 2 + 4}" text-anchor="end">${esc(row.label)}</text>`;
    out += `<text class="lbl" x="${r(Math.max(end, Number.isFinite(row.hi) ? sx(row.hi) : end) + 6)}" y="${barTop + barH / 2 + 4}">${esc(row.note ?? xFormat(row.value))}</text>`;
  });
  out += `<path class="axis" d="M${m.left},${m.top}V${base}"/>`;
  svg.innerHTML = out;
}

/**
 * The selection funnel on its side: population kept per generation, drawn
 * symmetrically about a centre line.
 */
export function drawFunnel(svg, { shapes, ratio = 0.3 }) {
  const { width, height } = frame(svg, ratio, 150);
  const m = { top: 8, right: 8, bottom: 22, left: 8 };
  const rowH = (height - m.top - m.bottom) / shapes.length;
  const peak = Math.max(...shapes.flatMap((s) => s.sizes));
  let out = hatch;
  shapes.forEach((shape, k) => {
    const mid = m.top + rowH * (k + 0.5);
    const sx = linearScale([0, shape.sizes.length - 1], [m.left + 150, width - m.right]);
    const half = (v) => (v / peak) * (rowH / 2 - 5);
    const xs = shape.sizes.map((_, i) => i);
    out += `<path class="band" d="${areaPath(xs, shape.sizes.map((v) => -half(v)), shape.sizes.map(half), sx, (v) => mid + v)}"/>`;
    out += `<path class="line" d="${linePath(xs, shape.sizes.map(half), sx, (v) => mid + v)}"/>`;
    out += `<path class="line" d="${linePath(xs, shape.sizes.map((v) => -half(v)), sx, (v) => mid + v)}"/>`;
    out += `<text class="lbl" x="${m.left}" y="${r(mid - 2)}">${esc(shape.label)}</text>`;
    out += `<text x="${m.left}" y="${r(mid + 12)}">${esc(shape.note)}</text>`;
  });
  out += `<text x="${m.left + 150}" y="${height - 5}">First generation</text><text x="${width - m.right}" y="${height - 5}" text-anchor="end">Last</text>`;
  svg.innerHTML = out;
}
