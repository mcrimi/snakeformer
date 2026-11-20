// The page: a game of Snake in which every frame is written by the model.

import { board, cells, start } from "./game.js";
import { Model, parseWeights } from "./model.js";
import { PROMPT_LENGTH, Turn, autopilot, heading, isReversal } from "./play.js";
import { cellAt, drawBoard } from "./view.js";
import { renderFindings } from "./findings.js";

const $ = (id) => document.getElementById(id);
const KEYS = { ArrowUp: "U", ArrowDown: "D", ArrowLeft: "L", ArrowRight: "R", w: "U", s: "D", a: "L", d: "R", W: "U", S: "D", A: "L", D: "R" };

const game = {
  models: {},        // id -> Model, once loaded
  list: [],          // the manifest
  model: null,
  state: start(),
  turn: null,        // the turn being written
  last: null,        // the last finished turn, for the attention view
  running: false,
  over: false,
  follow: "model",   // on a mistake, carry on from the model's frame or from the rules'
  speed: 6,          // tokens per animation frame; 0 = as many as fit
  queue: [],         // moves waiting to be played
  steered: false,    // has the reader taken over from the autopilot?
  turns: 0,
  right: 0,
  first: null,
  hold: 0,           // frames to wait so a mistake can be seen
  wrong: new Set(),
  ask: null,         // the cell the attention view is asked about
  layer: -1,
};

/* ---------- Paper settings (shared with the main site) ---------- */

function initPaper(onChange) {
  const KEY = "mcrimi-site";
  const read = () => { try { return JSON.parse(localStorage.getItem(KEY) || "{}") || {}; } catch (e) { return {}; } };
  const root = document.documentElement;
  const buttons = [...document.querySelectorAll("[data-paper]")];
  const show = (paper) => buttons.forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.paper === paper)));
  show(read().paper || "warm");
  buttons.forEach((b) => b.addEventListener("click", () => {
    const paper = b.dataset.paper;
    root.dataset.theme = paper === "night" ? "dark" : "light";
    if (paper === "grey") root.dataset.paper = "grey"; else delete root.dataset.paper;
    try { localStorage.setItem(KEY, JSON.stringify({ ...read(), paper })); } catch (e) { /* private mode: keep it for this visit only */ }
    show(paper);
    onChange();
  }));
  const aa = $("aa");
  const panel = $("panel");
  const toggle = (open) => { panel.hidden = !open; aa.setAttribute("aria-expanded", String(open)); };
  aa.addEventListener("click", () => toggle(panel.hidden));
  document.addEventListener("click", (e) => { if (!panel.hidden && !panel.contains(e.target) && e.target !== aa) toggle(false); });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") toggle(false); });
}

/* ---------- Drawing ---------- */

/** The rows of a board the model has written so far (the last may be partial). */
function partial(text) {
  return text.replace(/\$$/, "").split("\n").slice(0, 16).map((row) => [...row]);
}

function draw() {
  const enc = game.model ? game.model.encoding : "frame";
  const before = cells(board(game.state, enc));
  const turn = game.turn;
  if (turn && !turn.reading) {
    const text = turn.text;
    const dead = text.startsWith("X");
    drawBoard($("board"), dead ? before : partial(text), { faint: before, wrong: game.wrong, dead });
  } else if (game.over && game.last) {
    const text = game.last.text;
    const dead = text.startsWith("X");
    drawBoard($("board"), dead ? cells(board(game.last.state, enc)) : partial(text), { wrong: game.wrong, dead, faint: cells(board(game.last.state, enc)) });
  } else {
    drawBoard($("board"), before, { wrong: game.wrong });
  }
  drawTape();
  drawAttention();
}

function drawTape() {
  const tape = $("tape");
  // while the model is still reading, keep the frame it wrote last on show
  const turn = game.turn && (!game.turn.reading || !game.last) ? game.turn : game.last;
  tape.textContent = "";
  if (!turn) { tape.textContent = "T:\n"; return; }
  const add = (text, cls) => {
    const span = document.createElement(cls ? "span" : "b");
    if (cls) span.className = cls;
    span.textContent = text;
    tape.append(span);
  };
  tape.append(`A:${turn.action}\nT:\n`);
  const text = turn.text;
  const truth = turn.truth;
  const same = text.length === truth.length || !turn.done;
  for (let i = 0; i < text.length; i++) {
    const bad = turn.done && same && text[i] !== truth[i];
    const now = !turn.done && i === text.length - 1;
    add(text[i], bad ? "bad" : now ? "now" : "");
  }
  $("tape-note").textContent = game.turn && game.turn.reading
    ? `Reading the ${PROMPT_LENGTH} characters of the board and the next move.`
    : `${text.length} written.`;
}

function drawAttention() {
  const turn = game.last;
  const enc = game.model ? game.model.encoding : "frame";
  if (!turn || turn.text.startsWith("X")) {
    const blank = cells(board(game.state, enc));
    drawBoard($("ask"), blank);
    drawBoard($("att-in"), blank);
    drawBoard($("att-out"), []);
    $("att-action").textContent = "";
    return;
  }
  const out = partial(turn.text);
  const input = cells(board(turn.state, enc));
  let ask = game.ask;
  if (!ask) { // start with the new head
    for (let r = 0; r < out.length; r++) for (let c = 0; c < (out[r] || []).length; c++) if (out[r][c] === "H") ask = [r, c];
    ask = ask || [0, 0];
  }
  drawBoard($("ask"), out, { mark: ask });
  const looked = turn.attention(ask[0], ask[1], game.layer);
  if (!looked) return;
  drawBoard($("att-in"), input, { weight: looked.input });
  drawBoard($("att-out"), out, { weight: looked.output, mark: ask });
  $("att-action").textContent = looked.action > 0.15 ? `It also read your move (${Math.round(looked.action * 100)}% of the strongest).` : "";
}

/* ---------- The game ---------- */

function say(text) { $("verdict").textContent = text; }

function readout() {
  $("r-turn").textContent = String(game.turns);
  $("r-length").textContent = String(game.state.snake.length);
  $("r-right").textContent = game.turns ? `${game.right} of ${game.turns}` : "–";
  $("r-first").textContent = game.first === null ? "–" : `turn ${game.first}`;
}

function setRunning(on) {
  game.running = on && !game.over && !!game.model;
  $("go").textContent = game.over ? "Game over" : game.running ? "Pause" : game.turns ? "Resume" : "Start";
  $("go").disabled = !game.model || game.over;
}

function newGame() {
  game.state = start();
  game.turn = null;
  game.last = null;
  game.over = false;
  game.queue = [];
  game.steered = false;
  game.turns = 0;
  game.right = 0;
  game.first = null;
  game.hold = 0;
  game.wrong = new Set();
  game.ask = null;
  $("pilot").textContent = "Arrow keys or W A S D. Until you steer, a simple autopilot heads for the food.";
  say(game.model ? "Press Start. Every frame you see will be written by the model." : "Loading the model.");
  setRunning(false);
  readout();
  draw();
}

function nextAction() {
  while (game.queue.length) {
    const action = game.queue.shift();
    if (!isReversal(game.state, action)) return action; // as in the terminal game, no turning back on yourself
  }
  if (game.steered) return heading(game.state); // keep going the way the snake is heading
  return autopilot(game.state);
}

function finish(turn) {
  const v = turn.verdict();
  game.last = turn;
  game.turn = null;
  game.ask = null;
  game.turns += 1;
  game.wrong = new Set(v.wrong);
  if (v.right) game.right += 1;
  else if (game.first === null) game.first = game.turns;

  const truth = turn.truthState;
  let end = null;
  if (v.right) {
    if (!truth) end = turn.event === "wall" ? "It hit the wall, and the model said so." : "It ran into itself, and the model said so.";
    else if (turn.event === "eat") say("It ate, grew by one, and the model put the next food where the rules do.");
    else if (game.first === null) say("Every frame so far is exactly what the rules would have drawn.");
  } else if (v.says === "dead") {
    end = "The model declared the snake dead. By the rules it was still alive.";
  } else if (v.says === "nonsense") {
    end = "The model wrote something that is not a legal board, so the game cannot go on from it.";
  } else if (!truth) {
    say("By the rules the snake just died. The model let it live.");
    if (game.follow === "rules") end = "By the rules the snake died here. The model drew it carrying on.";
  } else {
    say(`Turn ${game.turns}: ${v.wrong.length} ${v.wrong.length === 1 ? "cell differs" : "cells differ"} from the rules. ${game.follow === "model" ? "The game carries on from the model’s frame." : "Corrected, and on we go."}`);
  }

  if (end) {
    game.over = true;
    say(`${end} ${game.right} of ${game.turns} frames were exact.`);
    setRunning(false);
  } else {
    game.state = !v.right && game.follow === "model" && v.next ? v.next : truth || v.next;
    game.hold = v.right ? 0 : 45;
  }
  readout();
}

function frame() {
  requestAnimationFrame(frame);
  if (!game.running) return;
  if (game.hold > 0) { game.hold -= 1; return; }
  if (!game.turn) {
    game.wrong = new Set();
    game.turn = new Turn(game.model, game.state, nextAction());
  }
  const turn = game.turn;
  const until = performance.now() + 12;
  if (turn.reading || game.speed === 0) {
    while (!turn.done && performance.now() < until && (turn.reading || game.speed === 0)) turn.advance(8);
  } else {
    turn.advance(game.speed);
  }
  if (turn.done) finish(turn);
  draw();
}

/* ---------- Loading and controls ---------- */

async function loadModel(id) {
  if (!game.models[id]) {
    const entry = game.list.find((m) => m.id === id);
    const response = await fetch(`data/${entry.file}`);
    if (!response.ok) throw new Error(`could not load ${entry.file}`);
    game.models[id] = new Model(parseWeights(await response.arrayBuffer()));
  }
  return game.models[id];
}

async function choose(id) {
  const entry = game.list.find((m) => m.id === id);
  for (const b of $("models").children) b.setAttribute("aria-pressed", String(b.dataset.model === id));
  $("model-note").textContent = entry.note;
  game.model = null;
  setRunning(false);
  $("go").textContent = "Loading the model…";
  try {
    game.model = await loadModel(id);
  } catch (e) {
    say("The model could not be loaded. Reload the page to try again.");
    return;
  }
  const layers = $("layers");
  layers.textContent = "";
  game.layer = -1;
  ["All", ...game.model.layers.map((_, i) => String(i + 1))].forEach((label, i) => {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = label;
    b.setAttribute("aria-pressed", String(i === 0));
    b.addEventListener("click", () => {
      game.layer = i - 1;
      for (const other of layers.children) other.setAttribute("aria-pressed", String(other === b));
      drawAttention();
    });
    layers.append(b);
  });
  newGame();
}

function steer(action) {
  if (!game.model || game.over) return;
  if (!game.steered) $("pilot").textContent = "You are steering. The snake keeps its heading until you turn it.";
  game.steered = true;
  if (game.queue.length < 3 && game.queue[game.queue.length - 1] !== action) game.queue.push(action);
  if (!game.running) setRunning(true);
}

function segmented(attribute, apply) {
  const buttons = [...document.querySelectorAll(`[data-${attribute}]`)];
  buttons.forEach((b) => b.addEventListener("click", () => {
    buttons.forEach((other) => other.setAttribute("aria-pressed", String(other === b)));
    apply(b.dataset[attribute]);
  }));
}

async function init() {
  initPaper(() => { draw(); renderFindings(); });
  segmented("follow", (v) => { game.follow = v; });
  segmented("speed", (v) => { game.speed = Number(v); });
  $("go").addEventListener("click", () => setRunning(!game.running));
  $("again").addEventListener("click", newGame);
  $("pad").addEventListener("click", (e) => { const b = e.target.closest("[data-key]"); if (b) steer(b.dataset.key); });
  document.addEventListener("keydown", (e) => {
    const action = KEYS[e.key];
    if (!action || e.metaKey || e.ctrlKey || e.altKey || /^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName)) return;
    const box = $("board").getBoundingClientRect();
    if (box.bottom < 0 || box.top > window.innerHeight) return; // the game is off screen: let the keys scroll
    e.preventDefault();
    steer(action);
  });
  const point = (e) => { const cell = cellAt($("ask"), e); if (cell) { game.ask = cell; drawAttention(); } };
  $("ask").addEventListener("pointermove", point);
  $("ask").addEventListener("pointerdown", point);
  window.addEventListener("resize", () => { draw(); renderFindings(); });
  document.fonts?.ready.then(draw);

  draw();
  requestAnimationFrame(frame);
  renderFindings();
  try {
    game.list = await (await fetch("data/models.json")).json();
  } catch (e) {
    say("The model list could not be loaded. Reload the page to try again.");
    return;
  }
  const holder = $("models");
  for (const entry of game.list) {
    const b = document.createElement("button");
    b.type = "button";
    b.dataset.model = entry.id;
    b.textContent = entry.label;
    b.addEventListener("click", () => choose(entry.id));
    holder.append(b);
  }
  const preferred = game.list.find((m) => m.default) || game.list[0];
  await choose(preferred.id);
}

init();
