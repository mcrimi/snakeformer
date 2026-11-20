// The figures below the game: the one-picture-two-snakes drawing, and the
// charts, which are drawn from data/experiments.json.

import { board, cells, step } from "./game.js";
import { drawBoard } from "./view.js";
import { drawBars, drawLines } from "./charts.js";

const $ = (id) => document.getElementById(id);
let data = null;

// Two snakes that draw the same H/O/# picture. In A the segment before the
// tail is above it; in B it is to its right.
const A = { snake: [[3, 3], [3, 2], [2, 2], [2, 3], [1, 3], [1, 2], [1, 1], [2, 1], [3, 1]], food: null };
const B = { snake: [[3, 3], [2, 3], [1, 3], [1, 2], [1, 1], [2, 1], [2, 2], [3, 2], [3, 1]], food: null };

function twins() {
  if (!$("twin-picture")) return;
  const span = 6;
  drawBoard($("twin-picture"), cells(board(A, "frame")), { span });
  drawBoard($("twin-a"), cells(board(A, "arrows")), { span });
  drawBoard($("twin-b"), cells(board(B, "arrows")), { span });
  drawBoard($("twin-a-next"), cells(board(step(A, "R").state, "frame")), { span });
  drawBoard($("twin-b-next"), cells(board(step(B, "R").state, "frame")), { span });
}

function fill(values) {
  for (const el of document.querySelectorAll("[data-fill]")) {
    const v = values[el.dataset.fill];
    if (v !== undefined) el.textContent = v;
  }
}

const pct = (x, digits = 0) => `${(100 * x).toFixed(digits)}%`;

function charts() {
  if (!data) return;
  for (const draw of FIGURES) {
    try { draw(data); } catch (e) { console.error(e); }
  }
}

const FIGURES = [];
export const figure = (fn) => FIGURES.push(fn);
export { drawBars, drawLines, fill, pct, $ };

export function renderFindings() {
  twins();
  charts();
}

export async function loadFindings() {
  try {
    data = await (await fetch("data/experiments.json")).json();
  } catch (e) {
    return;
  }
  charts();
}
