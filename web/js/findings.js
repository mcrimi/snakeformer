// The figures below the game: the one-picture-two-snakes drawing, and the
// charts, which are drawn from data/experiments.json.

import { board, cells, step } from "./game.js";
import { drawBoard } from "./view.js";
import { DASHES as DASH, drawBars, drawLines, percent } from "./charts.js";

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
const figure = (fn) => FIGURES.push(fn);
const NAMES = {
  original: "Published, 2026",
  "arrows-aligned": "Retrained, arrows",
  "frame-aligned": "Retrained, H O #",
  "frame-stream": "2026 recipe, H O #",
  "arrows-stream": "2026 recipe, arrows",
  "arrows-aligned-aggregated": "Retrained, then mistakes",
  "original-aggregated": "Published, then mistakes",
};
const KINDS = [
  ["move", "Ordinary move"], ["eat", "Eating"], ["wall", "Hitting a wall"],
  ["self", "Hitting itself"], ["reversal", "Turning back on itself"], ["all", "All"],
];
const svg = (id) => { const el = $(id); return el && el.clientWidth !== undefined ? el : null; };

// How often the H/O/# picture has two readings, by snake length.
figure((d) => {
  const el = svg("amb-chart");
  if (!el) return;
  const top = Math.max(...d.ambiguity.map((a) => a.hi));
  drawBars(el, {
    rows: d.ambiguity.map((a) => ({ label: `Length ${a.length}`, value: a.share, lo: a.lo, hi: a.hi, note: pct(a.share, 1) })),
    x: [0, Math.ceil(top * 20) / 20], xFormat: (v) => pct(v), xLabel: "Share of ordinary moves whose picture has two readings", labelWidth: 110,
  });
});

// Games still identical to the real one, turn by turn.
figure((d) => {
  const el = svg("alive-chart");
  if (!el) return;
  const ids = ["arrows-aligned-aggregated", "original", "arrows-aligned", "frame-aligned"].filter((k) => d.models[k]);
  drawLines(el, {
    x: [0, 400], y: [0, 1], xLabel: "Turn", yFormat: (v) => pct(v), labelWidth: 160,
    series: ids.map((k, i) => {
      const alive = d.models[k].play.alive;
      return { label: NAMES[k], x: alive.map((_, t) => t), y: alive, dash: DASH[i] };
    }),
  });
});

// Exact frames by kind of position, one column per model.
figure((d) => {
  const table = $("kinds-table");
  if (!table) return;
  const ids = ["original", "frame-aligned", "arrows-aligned", "arrows-aligned-aggregated"].filter((k) => d.models[k]);
  const cell = (m, kind) => {
    const k = kind === "all"
      ? Object.values(m.stress.kinds).reduce((a, v) => ({ right: a.right + v.right, n: a.n + v.n }), { right: 0, n: 0 })
      : m.stress.kinds[kind];
    return k ? `<td class="n">${pct(k.right / k.n, 1)}</td>` : `<td class="n">–</td>`;
  };
  const count = (kind) => {
    const kinds = d.models[ids[0]].stress.kinds;
    return kind === "all" ? d.test_n : kinds[kind] ? kinds[kind].n : 0;
  };
  table.innerHTML = `<thead><tr><th>Position</th><th class="n">Positions</th>${ids.map((k) => `<th class="n">${NAMES[k]}</th>`).join("")}</tr></thead><tbody>`
    + KINDS.map(([kind, label]) => `<tr${kind === "all" ? ' class="picked"' : ""}><th>${label}</th><td class="n">${count(kind).toLocaleString("en")}</td>${ids.map((k) => cell(d.models[k], kind)).join("")}</tr>`).join("")
    + `<tr><th>In ordinary play</th><td class="n">${d.games.frames.toLocaleString("en")}</td>${ids.map((k) => `<td class="n">${pct(d.models[k].play.frames.share, 1)}</td>`).join("")}</tr></tbody>`;
  const twins = $("twins-table");
  const both = ["original", "frame-aligned", "arrows-aligned"].filter((k) => d.models[k] && d.models[k].two_readings);
  if (twins && both.length) {
    twins.innerHTML = `<thead><tr><th>Model</th><th>Boards drawn with</th><th class="n">Two-reading positions right</th><th class="n">95% interval</th></tr></thead><tbody>`
      + both.map((k) => { const t = d.models[k].two_readings; return `<tr><th>${NAMES[k]}</th><td>${d.models[k].encoding === "arrows" ? "Arrows" : "H O #"}</td><td class="n">${pct(t.share, 1)}</td><td class="n">${pct(t.lo, 0)} to ${pct(t.hi, 0)}</td></tr>`; }).join("") + "</tbody>";
  }
});

// Learning curves of the four ways to train the same network.
figure((d) => {
  const el = svg("curve-chart");
  if (!el) return;
  const ids = ["arrows-aligned", "frame-aligned", "arrows-stream", "frame-stream"].filter((k) => d.runs[k]);
  drawLines(el, {
    x: [0, 20000], y: [0, 1], xLabel: "Training steps", yFormat: (v) => pct(v), xFormat: (v) => (v ? `${v / 1000}k` : "0"), labelWidth: 150,
    series: ids.map((k, i) => ({ label: NAMES[k], x: [0, ...d.runs[k].curve.map((r) => r.step)], y: [0, ...d.runs[k].curve.map((r) => r.all)], dash: DASH[i] })),
  });
});

// Learning from mistakes: the one-at-a-time loop, game by game.
figure((d) => {
  const el = svg("polish-play");
  if (!el || !d.polish) return;
  const runs = [["original-polished-mortal", "Published"], ["arrows-aligned-polished-mortal", "Retrained"], ["original-polished", "Published, no deaths"]]
    .filter(([k]) => d.polish[k]);
  if (!runs.length) return;
  const low = Math.min(...runs.flatMap(([k]) => d.polish[k].log.map((r) => r.play)));
  drawLines(el, {
    x: [0, 300], y: [Math.floor(low * 50) / 50, 1], xLabel: "Games played with corrections switched on", yFormat: (v) => pct(v), labelWidth: 150,
    series: runs.map(([k, label], i) => ({ label, x: d.polish[k].log.map((r) => r.games), y: d.polish[k].log.map((r) => r.play), dash: DASH[i] })),
  });
});

// Both ways of learning from mistakes, before and after.
figure((d) => {
  const table = $("learn-table");
  if (!table) return;
  const rows = [
    ["One at a time", "Published", d.polish && d.polish["original-polished-mortal"]],
    ["One at a time", "Retrained", d.polish && d.polish["arrows-aligned-polished-mortal"]],
    ["Collect, then retrain", "Published", d.aggregate && d.aggregate["original-aggregated"]],
    ["Collect, then retrain", "Retrained", d.aggregate && d.aggregate["arrows-aligned-aggregated"]],
  ].filter((r) => r[2]);
  const move = (a, b, digits = 1) => `${pct(a, digits)} → ${pct(b, digits)}`;
  table.innerHTML = `<thead><tr><th>Method</th><th>Model</th><th class="n">Games</th><th class="n">In play</th><th class="n">Hard positions</th><th class="n">Median first wrong frame</th></tr></thead><tbody>`
    + rows.map(([method, model, run]) => {
      const a = run.log[0];
      const b = run.log[run.log.length - 1];
      return `<tr><th>${method}</th><td>${model}</td><td class="n">${b.games.toLocaleString("en")}</td><td class="n">${move(a.play, b.play)}</td><td class="n">${move(a.all, b.all)}</td><td class="n">${a.median_first_error} → ${b.median_first_error}</td></tr>`;
    }).join("") + "</tbody>";
});

// Random food: the chance the model gives to food at each free cell, against 1/k.
figure((d) => {
  const run = d.runs["arrows-random-food"];
  const el = svg("hazard-chart");
  if (!el || !run || !run.hazard) return;
  const pts = run.hazard.points;
  const lo = Math.min(...pts.map((p) => Math.min(p.ideal, p.model))) * 0.7;
  drawLines(el, {
    x: [lo, 1], y: [lo, 1], xLog: true, yLog: true, xLabel: "Right answer: 1 in the number of free cells left", ratio: 0.62,
    xFormat: (v) => percent(v), yFormat: (v) => percent(v), labelWidth: 96,
    series: [{ label: "Perfect", x: [lo, 1], y: [lo, 1], dash: DASH[1] }],
    points: pts.map((p) => ({ x: p.ideal, y: p.model })),
  });
  const bars = svg("tenths-chart");
  if (!bars) return;
  const t = run.samples.tenths;
  const total = t.reduce((a, b) => a + b, 0) || 1;
  drawBars(bars, {
    rows: t.map((v, i) => ({ label: `${i * 10}–${i * 10 + 10}%`, value: v / total, note: String(v) })),
    x: [0, Math.max(0.2, Math.ceil(Math.max(...t) / total * 10) / 10)], xFormat: (v) => pct(v), labelWidth: 80,
    xLabel: "Where the sampled food landed, as a rank among the free cells",
  });
});

// Numbers quoted in the text.
function numbers(d) {
  const out = {};
  const put = (k, v) => { if (v !== undefined && v !== null && !Number.isNaN(v)) out[k] = v; };
  for (const [id, m] of Object.entries(d.models)) {
    const p = m.play;
    put(`${id}.play`, pct(p.frames.share, 1));
    put(`${id}.median`, p.median_first_error > 1e6 ? "past 400" : p.median_first_error);
    put(`${id}.clean`, p.clean_games);
    put(`${id}.params`, `${(m.params / 1e6).toFixed(2)}M`);
    const all = Object.values(m.stress.kinds).reduce((a, v) => ({ right: a.right + v.right, n: a.n + v.n }), { right: 0, n: 0 });
    put(`${id}.stress`, pct(all.right / all.n, 1));
    for (const [k, v] of Object.entries(m.stress.kinds)) put(`${id}.${k}`, pct(v.share, 0));
    for (const [k, v] of Object.entries(p.events || {})) put(`${id}.play.${k}`, pct(v.share, 1));
    if (m.two_readings) put(`${id}.twins`, pct(m.two_readings.share, 0));
  }
  put("games", d.games.n);
  put("frames", d.games.frames.toLocaleString("en"));
  put("test", d.test_n.toLocaleString("en"));
  const amb = d.ambiguity.reduce((a, v) => ({ right: a.right + v.right, n: a.n + v.n }), { right: 0, n: 0 });
  put("ambiguous", pct(amb.right / amb.n, 1));
  for (const [id, run] of Object.entries(d.aggregate || {})) {
    const last = run.log[run.log.length - 1];
    put(`${id}.mistakes`, last.mistakes.toLocaleString("en"));
    put(`${id}.frames`, run.log.reduce((a, r) => a + r.frames_seen, 0).toLocaleString("en"));
    put(`${id}.last`, last.new_mistakes.toLocaleString("en"));
    put(`${id}.first`, run.log[1].new_mistakes.toLocaleString("en"));
  }
  for (const [id, run] of Object.entries(d.polish || {})) {
    if (run.gradient) {
      put(`${id}.grad.steps`, (run.gradient.weighted_steps + run.gradient.plain_steps).toLocaleString("en"));
      put(`${id}.grad.plain`, Math.round(run.gradient.plain_median));
      put(`${id}.grad.weighted`, Math.round(run.gradient.weighted_median).toLocaleString("en"));
    }
    const first = run.log[0];
    const last = run.log[run.log.length - 1];
    for (const k of ["play", "wall", "self", "eat", "move", "all"]) {
      put(`${id}.${k}.before`, pct(first[k], 1));
      put(`${id}.${k}.after`, pct(last[k], 1));
    }
    put(`${id}.clean.before`, first.clean_games);
    put(`${id}.clean.after`, last.clean_games);
    put(`${id}.corrections`, last.corrections.toLocaleString("en"));
    put(`${id}.steps`, last.steps.toLocaleString("en"));
  }
  const food = d.runs["arrows-random-food"];
  if (food && food.samples) {
    put("food.legal", pct(food.samples.share, 0));
    put("food.frames", food.samples.frames);
  }
  return out;
}

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
  fill(numbers(data));
  charts();
}
