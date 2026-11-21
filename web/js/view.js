// Canvas drawings: the board, and where the model looked. Ink and paper are
// read from CSS at draw time, so the drawings follow the reader's paper.

import { SIZE } from "./game.js";

function palette() {
  const css = getComputedStyle(document.documentElement);
  const get = (name) => css.getPropertyValue(name).trim();
  return { ink: get("--ink"), ink2: get("--ink-2"), line: get("--line"), paper: get("--paper"), sans: get("--sans") };
}

function prepare(canvas) {
  const width = Math.max(160, Math.round(canvas.getBoundingClientRect().width || 320));
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  if (canvas.width !== width * dpr || canvas.height !== width * dpr) {
    canvas.width = width * dpr;
    canvas.height = width * dpr;
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const c = palette();
  ctx.fillStyle = c.paper;
  ctx.fillRect(0, 0, width, width);
  return { ctx, width, c };
}

const BODY = "O^v<>";
const TAILS = "#udlr";
const VECTOR = { "^": [-1, 0], v: [1, 0], "<": [0, -1], ">": [0, 1], u: [-1, 0], d: [1, 0], l: [0, -1], r: [0, 1] };

/** Which cell of the board a canvas point is over, or null. */
export function cellAt(canvas, event) {
  const box = canvas.getBoundingClientRect();
  const pad = box.width * 0.03;
  const size = (box.width - 2 * pad) / SIZE;
  const c = Math.floor((event.clientX - box.left - pad) / size);
  const r = Math.floor((event.clientY - box.top - pad) / size);
  return r >= 0 && c >= 0 && r < SIZE && c < SIZE ? [r, c] : null;
}

/**
 * Draw a board from its text.
 *   grid:   16 rows of characters; rows or cells not written yet may be missing
 *   wrong:  Set of "r,c" where the model and the rules disagree
 *   weight: Float32Array(256) in 0..1, drawn as a square in each cell (attention)
 *   mark:   [r, c] to ring (the cell being asked about)
 *   dead:   draw a cross over the board
 *   faint:  the previous grid, shown in grey where `grid` has nothing yet
 *   span:   draw only the top-left span x span cells
 */
export function drawBoard(canvas, grid, { wrong = null, weight = null, mark = null, dead = false, faint = null, span = SIZE } = {}) {
  const { ctx, width, c } = prepare(canvas);
  const pad = width * 0.03;
  const size = (width - 2 * pad) / span;
  const gap = Math.max(1, size * 0.1);
  const at = (i) => pad + i * size;
  ctx.fillStyle = c.line;
  for (let r = 0; r < span; r++) for (let k = 0; k < span; k++) ctx.fillRect(Math.round(at(k) + size / 2), Math.round(at(r) + size / 2), 1, 1);

  for (let r = 0; r < span; r++) {
    for (let k = 0; k < span; k++) {
      // rows the model has not written yet are drawn, faintly, from the frame before
      const written = grid && grid[r] && grid[r][k] !== undefined;
      const old = !written && faint && faint[r] ? faint[r][k] : undefined;
      const ch = written ? grid[r][k] : old;
      const x = at(k);
      const y = at(r);
      if (weight) {
        const w = weight[r * SIZE + k];
        if (w > 0.03) {
          const s = Math.max(2, (size - 2) * Math.sqrt(w));
          ctx.fillStyle = c.ink;
          ctx.fillRect(x + (size - s) / 2, y + (size - s) / 2, s, s);
        }
        if (ch && ch !== ".") {
          ctx.strokeStyle = c.ink2;
          ctx.lineWidth = 1;
          ctx.strokeRect(Math.round(x + gap) + 0.5, Math.round(y + gap) + 0.5, Math.round(size - 2 * gap) - 1, Math.round(size - 2 * gap) - 1);
        }
        continue;
      }
      if (!ch || ch === ".") continue;
      ctx.fillStyle = written ? c.ink : c.ink2;
      ctx.strokeStyle = written ? c.ink : c.ink2;
      ctx.globalAlpha = written ? 1 : 0.6;
      if (ch === "F") {
        ctx.lineWidth = Math.max(1.5, size * 0.14);
        ctx.strokeRect(x + size * 0.26, y + size * 0.26, size * 0.48, size * 0.48);
      } else if (ch === "H") {
        ctx.fillRect(x + gap * 0.4, y + gap * 0.4, size - gap * 0.8, size - gap * 0.8);
        ctx.fillStyle = c.paper;
        ctx.fillRect(x + size * 0.4, y + size * 0.4, size * 0.2, size * 0.2);
      } else if (BODY.includes(ch) || TAILS.includes(ch)) {
        const inset = TAILS.includes(ch) ? size * 0.26 : gap;
        ctx.fillRect(x + inset, y + inset, size - 2 * inset, size - 2 * inset);
        const v = VECTOR[ch];
        if (v) { // join the segment to the one it points at
          ctx.fillRect(x + size / 2 - size * 0.14 + v[1] * size * 0.4, y + size / 2 - size * 0.14 + v[0] * size * 0.4, size * 0.28, size * 0.28);
        }
      } else { // a character that does not belong on a board
        ctx.font = `600 ${Math.round(size * 0.7)}px ${c.sans}`;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(ch, x + size / 2, y + size / 2);
      }
    }
  }
  ctx.globalAlpha = 1;
  if (wrong) {
    ctx.strokeStyle = c.ink;
    ctx.lineWidth = 2;
    for (const key of wrong) {
      const [r, k] = key.split(",").map(Number);
      ctx.strokeRect(at(k) - 2, at(r) - 2, size + 4, size + 4);
      ctx.beginPath();
      ctx.moveTo(at(k) - 2, at(r) + size + 2);
      ctx.lineTo(at(k) + size + 2, at(r) - 2);
      ctx.stroke();
    }
  }
  if (mark) {
    ctx.strokeStyle = c.ink;
    ctx.lineWidth = 2;
    ctx.strokeRect(at(mark[1]) - 1, at(mark[0]) - 1, size + 2, size + 2);
  }
  if (dead) {
    ctx.strokeStyle = c.ink;
    ctx.lineWidth = Math.max(2, width * 0.012);
    ctx.beginPath();
    ctx.moveTo(pad + size * 4, pad + size * 4); ctx.lineTo(width - pad - size * 4, width - pad - size * 4);
    ctx.moveTo(width - pad - size * 4, pad + size * 4); ctx.lineTo(pad + size * 4, width - pad - size * 4);
    ctx.stroke();
  }
}
